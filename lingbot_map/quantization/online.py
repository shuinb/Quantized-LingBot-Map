"""Incremental inference without retaining a video's dense predictions in RAM."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import closing
from itertools import chain
import math
from threading import Event
import time

import torch

from .lazy_images import LazyImageSequence


def _image_chunks(images, anchors, prefetch, input_fps=None):
    """Read in order, retaining at most one future input beside the current one."""
    starts = iter(chain((0,), range(anchors, images.shape[1])))
    source_start = time.perf_counter()
    stopped = Event()

    def read(start):
        count = anchors if start == 0 else 1
        available = source_start + (start + count - 1) / input_fps if input_fps else None
        if available is not None:
            delay = available - time.perf_counter()
            if delay > 0 and stopped.wait(delay):
                raise CancelledError('Input replay closed')
        begin = time.perf_counter()
        frame = images[:, start:start + count]
        return start, frame, begin, time.perf_counter(), available if available is not None else begin

    if not prefetch:
        for start in starts:
            yield read(start)
        return

    # Only CPU image loading runs in the worker; all model/CUDA work stays
    # on the consumer thread, including device copies and graph replay.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='lingbot-input') as pool:
        future = pool.submit(read, next(starts))
        try:
            while future is not None:
                chunk = future.result()
                next_start = next(starts, None)
                future = pool.submit(read, next_start) if next_start is not None else None
                yield chunk
                del chunk
        finally:
            stopped.set()
            if future is not None:
                future.cancel()


@torch.no_grad()
def iter_predictions(model, images, scale_frames=2, keyframe_interval=16, output_device='cpu',
                     prefetch_images=False, input_timing=None, input_fps=None):
    """Yield ``(first_frame_index, predictions)`` in original streaming order.

    ``images`` can be a tensor or LazyImageSequence. The first yield contains
    the joint anchor predictions; subsequent yields contain one frame each.
    Consume outputs before requesting the next frame. Close the generator on
    early exit (e.g. with contextlib.closing) to release the live KV history.
    This does not truncate the model's retained special-token history.

    ``prefetch_images`` overlaps one LazyImageSequence slice with inference.
    If supplied, ``input_timing`` is updated before each yield with that
    chunk's frame index and perf_counter timestamps for CPU decode. This
    allows consumers to include prefetch waiting in input-to-output latency.
    ``input_fps`` optionally simulates uniformly arriving input frames: no
    frame is read before its scheduled arrival. Source-to-output latency is
    measured from ``input_available`` (the last arrival for joint anchors).
    """
    if scale_frames < 1 or keyframe_interval < 1:
        raise ValueError('scale_frames and keyframe_interval must be positive')
    if input_fps is not None and (not math.isfinite(input_fps) or input_fps <= 0):
        raise ValueError('input_fps must be finite and positive')
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    if len(images.shape) != 5 or images.shape[1] < 1:
        raise ValueError('Expected nonempty images [B,S,3,H,W] or [S,3,H,W]')
    if prefetch_images and not isinstance(images, LazyImageSequence):
        raise ValueError('Image prefetch requires a CPU LazyImageSequence')
    device = next(model.parameters()).device
    length = images.shape[1]
    anchors = min(scale_frames, length)
    fields = ('pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf')
    model.clean_kv_cache()
    try:
        with closing(_image_chunks(images, anchors, prefetch_images, input_fps)) as chunks:
            for start, frame, decode_started, decode_finished, available in chunks:
                count = anchors if start == 0 else 1
                keyframe = start == 0 or (start - anchors) % keyframe_interval == 0
                frame = frame.to(device, non_blocking=True)
                model._set_skip_append(not keyframe)
                torch.compiler.cudagraph_mark_step_begin()
                try:
                    prediction = model(frame, num_frame_for_scale=anchors,
                                       num_frame_per_block=count, causal_inference=True)
                finally:
                    model._set_skip_append(False)
                output = {key: prediction[key].to(output_device) for key in fields if key in prediction}
                output['frame_type'] = torch.full((images.shape[0], count),
                                                  0 if start == 0 else 1 if keyframe else 2,
                                                  dtype=torch.uint8, device=output_device)
                output['is_keyframe'] = output['frame_type'] != 2
                if input_timing is not None:
                    input_timing.update(first_frame=start, decode_started=decode_started,
                                        decode_finished=decode_finished, input_available=available)
                del prediction, frame
                yield start, output
                del output
    finally:
        model.clean_kv_cache()
