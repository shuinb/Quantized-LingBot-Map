"""Calibrate direct W4A4 or DSFQ using independent image sequences."""
import argparse
from contextlib import closing, contextmanager, nullcontext
import json
from pathlib import Path
import time

import torch

from lingbot_map.quantization import QuantizationConfig, Quantizer
from lingbot_map.quantization.deployment import export_deployment
from lingbot_map.quantization.lazy_images import LazyImageSequence
from lingbot_map.quantization.online import iter_predictions
from lingbot_map.quantization.quality import (STREAM_KEYS, image_files, load_quality_float,
                                             pack_quality_model, window_ranges)
from lingbot_map.quantization.reconstruction import BlockReconstruction
from lingbot_map.quantization.selection import balanced_capture_steps, stream_steps
from qmap.common import CPUCachePlacement, depth_chunk_one, write_json


@contextmanager
def calibration_cache_placement(model, device, offload=False):
    """Optional original-dtype CPU KV between attention calls, observation only."""
    if not offload:
        yield
        return
    if device.type != 'cuda':
        raise ValueError('Calibration cache offload requires a CUDA model')
    placement = CPUCachePlacement([block.attn for block in model.aggregator.global_blocks], device)
    try:
        yield
    finally:
        # Block reconstruction replays one block's saved state on the GPU. It
        # must not inherit observation hooks or keep live cache references.
        for handle in placement.handles:
            handle.remove()
        placement.active.clear()


def calibrate(quantizer, clips, quality, settings, device):
    """Observe full continuous clips, optionally reconstruct actual saved states."""
    epochs = settings['reconstruction_epochs']
    states = [stream_steps(len(clip['files']), quality['num_scale_frames'],
                           quality['kv_cache_sliding_window'], quality['kv_cache_scale_frames'],
                           quality['keyframe_interval']) for clip in clips]
    schedules = ([balanced_capture_steps(records, settings['samples_per_clip']) for records in states]
                 if epochs else [[] for _ in clips])
    reconstruction = BlockReconstruction(quantizer, sum(map(len, schedules))) if epochs else None
    totals = {name: torch.zeros_like(layer.observer.ema) for name, layer in quantizer.layers.items()}
    coverage = []
    for clip, schedule, records in zip(clips, schedules, states):
        quantizer.reset_observers()
        before = {name: len(samples) for name, samples in reconstruction.samples.items()} if reconstruction else {}
        images = LazyImageSequence(clip['files'], settings['image_size'], max_height=settings['max_height'])
        autocast = torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False) if device.type == 'cuda' else nullcontext()
        with calibration_cache_placement(quantizer.model, device, settings.get('offload_cache', False)), \
                torch.no_grad(), autocast, quantizer.calibration(), \
                (reconstruction.capture(steps=schedule) if reconstruction else nullcontext()), \
                closing(iter_predictions(quantizer.model, images, quality['num_scale_frames'],
                                         quality['keyframe_interval'])) as iterator:
            for index, prediction in iterator:
                if not all(torch.isfinite(value).all() for value in prediction.values() if value.is_floating_point()):
                    raise ValueError(f'Nonfinite calibration output at {clip["id"]}:{index}')
                print(f'Observe {clip["id"]}: {index + prediction["pose_enc"].shape[1]}/{len(clip["files"])}', flush=True)
        counts = {name: len(samples) - before[name] for name, samples in reconstruction.samples.items()} if reconstruction else {}
        if counts and set(counts.values()) != {len(schedule)}:
            raise ValueError('Incomplete calibration state capture')
        for name, layer in quantizer.layers.items():
            if not layer.observer.calls.item():
                raise ValueError(f'No observations for {name}')
            totals[name].add_(layer.observer.ema)
        coverage.append({'id': clip['id'], 'frames': len(clip['files']), 'image_hw': list(images.shape[-2:]),
                         'observed_phases': sorted({row['phase'] for row in records}),
                         'capture_steps': schedule if reconstruction else [], 'samples_per_block': counts})
    for name, layer in quantizer.layers.items():
        layer.observer.ema.copy_(totals[name] / len(clips))
        layer.initialize_smoothing()
    optimized = reconstruction.optimize(epochs=epochs, lr=settings['reconstruction_lr']) if reconstruction else {}
    return {'coverage': coverage, 'reconstruction': optimized,
            'observer_initialization': 'equal mean of per-clip final EMA, canonical clip ID order',
            'calibration_precision': 'BF16 trunk / FP32 heads' if device.type == 'cuda' else 'FP32 CPU; not GPU-equivalent calibration',
            'cache_placement': 'CPU between attention calls, original dtype' if settings.get('offload_cache') else str(device),
            'reconstruction_epochs': epochs}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qmap calibrate', description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--quality-config', type=Path, default=Path(__file__).with_name('configs') / 'stream.json')
    parser.add_argument('--config', dest='calibration_config', type=Path, default=Path('configs/calibration.json'))
    parser.add_argument('--method', choices=('w4a4', 'dsfq'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, help='Positive reconstruction epoch count; default comes from the config')
    parser.add_argument('--frames-per-clip', type=int, help='Explicit shorter pilot; recorded in artifact metadata')
    parser.add_argument('--clips', type=int, help='Use the first N calibration clips, for a recorded pilot')
    parser.add_argument('--offload-cache', dest='offload_calibration_cache', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep original-dtype GCT KV on CPU between observation calls; reconstruction still runs one block at a time on CUDA')
    args = parser.parse_args(argv)
    args.mode = 'naive' if args.method == 'w4a4' else 'dsfq'
    args.learned_clipping = True
    quality = json.loads(args.quality_config.read_text())
    window_ranges(1, quality)
    settings = json.loads(args.calibration_config.read_text())
    if args.epochs is not None:
        settings['reconstruction_epochs'] = args.epochs
    if settings['reconstruction_epochs'] < 0 or (args.frames_per_clip is not None and args.frames_per_clip < 1):
        parser.error('Epochs must be nonnegative and clip length positive')
    if args.clips is not None and not 1 <= args.clips <= len(settings['clips']):
        parser.error('Invalid number of calibration clips')
    if args.learned_clipping and settings['reconstruction_epochs'] == 0:
        parser.error('Learned clipping requires reconstruction with at least one epoch')
    needs_calibration = args.mode == 'dsfq' or args.learned_clipping
    device = torch.device(args.device)
    if args.offload_calibration_cache and (device.type != 'cuda' or not needs_calibration):
        parser.error('Calibration cache offload requires calibration and a CUDA device')
    settings['offload_cache'] = args.offload_calibration_cache
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not accessible; CPU packing is available with --device cpu. GPU timing is not.')
    clips = []
    if needs_calibration:
        for clip in sorted(settings['clips'][:args.clips], key=lambda row: row['id']):
            directory = (args.calibration_config.parent / clip['images']).resolve()
            files = image_files(directory, clip['start'], args.frames_per_clip or clip['frames'])
            clips.append({**clip, 'configured_frames': clip['frames'], 'frames': len(files),
                          'files': [str(path) for path in files]})
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'loading', 'method': args.method, 'mode': args.mode, 'quality_config': quality,
              'calibration_settings': settings if clips else None, 'calibration_clips': clips,
              'checkpoint': str(args.checkpoint.resolve()), 'device': str(device),
              'torch_version': torch.__version__, 'seed': 0, 'runtime_quality_and_speed_verified': False}
    destination = args.output / 'build.json'
    write_json(destination, report)
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_quality_float(args.checkpoint, quality,
                                   trunk_bf16=not needs_calibration or device.type == 'cuda').to(device)
        model.depth_head.register_forward_pre_hook(depth_chunk_one, with_kwargs=True)
        quantizer = Quantizer(model, QuantizationConfig(mode=args.mode, lwc=args.learned_clipping,
                                                       lac=args.learned_clipping))
        for layer in quantizer.layers.values():
            layer.cache_weight = False
        report.update(status='calibrating' if clips else 'packing', quantized_layers=len(quantizer.layers),
                      quantization_config=quantizer.config.to_dict())
        write_json(destination, report)
        if clips:
            report['calibration'] = calibrate(quantizer, clips, quality, settings, device)
        quantizer.save_parameters(args.output / 'parameters.pt')
        model.clean_kv_cache()
        # Packing itself is architecture independent and does not need CUDA.
        model.cpu()
        # All exports quantize the same BF16-stored trunk as the quality reference,
        # including artifacts whose observer was run in FP32 on a CPU.
        for layer in quantizer.layers.values():
            layer.linear.to(dtype=torch.bfloat16)
        pack_quality_model(quantizer)
        model.aggregator.to(dtype=torch.bfloat16)
        report['artifact'] = export_deployment(model, {name: quality[name] for name in STREAM_KEYS},
                                                args.output / 'model.pt')
        report.update(status='complete', seconds=time.perf_counter() - started,
                      unquantized_modules='DINO BF16, camera/DPT FP32, Norm/RoPE/attention and original KV')
        print(json.dumps(report['artifact'], indent=2), flush=True)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(destination, report)


if __name__ == '__main__':
    main()
