"""BF16 / calibrated W4A4 / DSFQ windowed inference and timing.

All frames are inferred at 294x518; overlapping frames count once in FPS.
The original quality configuration and FP32 heads are preserved by default.
"""
import argparse
from contextlib import closing, nullcontext
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from lingbot_map.quantization.compact_cache import compact_streaming_cache
from lingbot_map.quantization.compressed_cache import compressed_streaming_cache
from lingbot_map.quantization.camera_graphs import camera_cuda_graph
from lingbot_map.quantization.cuda_graphs import stateless_cuda_graphs
from lingbot_map.quantization.deployment import enable_camera_compute_dtype, load_deployment
from lingbot_map.quantization.lazy_images import LazyImageSequence
from lingbot_map.quantization.kv_codec import FORMATS
from lingbot_map.quantization.native_int4 import PackedInt4Linear, enable_fused_int4_epilogue
from lingbot_map.quantization.online import iter_predictions
from lingbot_map.quantization.prefill_memory import chunked_prefill_mlp
from lingbot_map.quantization.quality import (STREAM_KEYS, construct_quality_model, image_files,
                                             load_quality_float, window_ranges)
from qmap.common import prediction_is_finite
from qmap.common import depth_chunk_one, write_json
from qmap.presets import inference_defaults, validate_deployment


def quality_cache(model, quality, format_name='original', *, store_compute_dtype=False, defer_full_reservation=False):
    """Keep the same streaming history while optionally compressing its storage."""
    if format_name == 'original':
        return compact_streaming_cache(model, reserve_frames=quality['window_size'],
                                       store_compute_dtype=store_compute_dtype)
    if store_compute_dtype:
        raise ValueError('Compute-dtype storage is only applicable to original KV')
    return compressed_streaming_cache(model, format_name, reserve_frames=quality['window_size'],
                                      deduplicate_special_scales=True, fuse_sdpa_decode=True,
                                      defer_full_reservation=defer_full_reservation)


def run_windows(model, files, quality, *, prefetch=True, repeat=0, cache=None):
    """Include decoding, transfer, all predictions, overlap, and CPU alignment."""
    ranges, overlap = window_ranges(len(files), quality)
    windows, rows, cache_reports = [], [], []
    start = time.perf_counter()
    for window, (left, right) in enumerate(ranges):
        images = LazyImageSequence(files[left:right], 518, max_height=294)
        if list(images.shape[-2:]) != [294, 518]:
            raise ValueError(f'Expected 294x518 input, got {images.shape[-2:]}')
        fields, input_timing = {}, {}
        with closing(iter_predictions(model, images, quality['num_scale_frames'],
                                      quality['keyframe_interval'], prefetch_images=prefetch,
                                      input_timing=input_timing)) as iterator:
            while True:
                begin = time.perf_counter()
                try:
                    index, prediction = next(iterator)
                except StopIteration:
                    break
                if not prediction_is_finite(prediction):
                    raise ValueError(f'Nonfinite prediction at window {window}, frame {left + index}')
                finished = time.perf_counter()
                count = prediction['pose_enc'].shape[1]
                rows.append({'repeat': repeat, 'window': window, 'first_frame': left + index,
                             'frames_in_call': count, 'call_ms': (finished - begin) * 1000,
                             'decode_to_prediction_ms': (finished - input_timing['decode_started']) * 1000})
                for key, value in prediction.items():
                    fields.setdefault(key, []).append(value)
                if index + count == right - left and cache is not None and hasattr(cache, 'memory_report'):
                    cache_reports.append(cache.memory_report())
        windows.append({key: torch.cat(values, dim=1) for key, values in fields.items()})
        print(f'repeat {repeat + 1}: window {window + 1}/{len(ranges)}, unique frames {right}/{len(files)}', flush=True)
    model._last_overlap_size = overlap
    model._last_window_size = overlap
    alignment_start = time.perf_counter()
    merged = model._align_and_stitch_windows(windows, scale_mode='median')
    alignment_seconds = time.perf_counter() - alignment_start
    if merged['pose_enc'].shape[1] != len(files):
        raise ValueError('Stitching did not cover every unique input frame')
    if not prediction_is_finite({key: value for key, value in merged.items() if torch.is_tensor(value)}):
        raise ValueError('Nonfinite stitched outputs')
    seconds = time.perf_counter() - start
    measurement = {'seconds': seconds, 'unique_fps': len(files) / seconds,
                   'alignment_seconds': alignment_seconds,
                   'processed_frames_including_overlap': sum(b - a for a, b in ranges)}
    if cache_reports:
        measurement['kv_cache_largest_window'] = max(cache_reports, key=lambda row: row['storage_bytes'])
    return merged, rows, measurement


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qmap infer', description=__doc__)
    parser.add_argument('--method', choices=('bf16', 'w4a4', 'dsfq'), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Original checkpoint for BF16 quality reference')
    source.add_argument('--deployment', type=Path, help='Packed model.pt produced by qmap calibrate')
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--quality-config', type=Path, default=Path(__file__).with_name('configs') / 'stream.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--frames', type=int, default=5000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--backend', dest='linear_backend', choices=('auto', 'native', 'int8-fused'), default='auto',
                        help='auto selects native S4 on SM8x and W4A4 codes + INT8 GEMM otherwise')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmup-frames', type=int, default=248)
    parser.add_argument('--graphs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--extended-graphs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile-stateless', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--camera-graph', action=argparse.BooleanOptionalAction, default=None,
                        help='Replay the existing camera graph with FP32 heads and all retained camera history; validate geometry because padded SDPA can change rounding')
    parser.add_argument('--camera-precision', choices=('fp32', 'fp16', 'bf16'), default='fp32',
                        help='Optional camera Linear precision tradeoff; norms and DPT remain FP32')
    parser.add_argument('--prefetch-images', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fused-int4-epilogue', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--store-kv-compute-dtype', action=argparse.BooleanOptionalAction, default=None,
                        help='Store the BF16 operands already consumed by SDPA autocast; reduces FP32 K cache storage')
    parser.add_argument('--kv-format', choices=('original', *FORMATS), default=None,
                        help='Optional persistent compressed KV; same anchors, local history and special tokens')
    parser.add_argument('--defer-kv-reserve', action=argparse.BooleanOptionalAction, default=None,
                        help='Reserve compressed KV for the initial joint frames first, then expand once without requantizing')
    parser.add_argument('--memory-budget-gb', type=float,
                        help='Optional PyTorch allocator budget in decimal GB; CUDA context is additional')
    parser.add_argument('--prefill-mlp-chunk-tokens', type=int, default=None,
                        help='Optional GCT MLP row chunk size for joint prefill; attention still sees all anchors')
    parser.add_argument('--prefill-frame-batch-size', type=int, default=None,
                        help='Optional batch limit for independent GCT frame blocks; global attention stays joint')
    args = parser.parse_args(argv)
    if (args.method == 'bf16') != bool(args.checkpoint):
        parser.error('BF16 requires --checkpoint; w4a4 and dsfq require --deployment')
    if args.linear_backend == 'auto':
        native = torch.cuda.is_available() and torch.cuda.get_device_capability(args.device)[0] == 8
        args.linear_backend = 'native' if native else 'int8-fused'
    for name, value in inference_defaults(args.method, args.linear_backend).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.frames, args.repeats, args.warmup_frames) < 1 or args.start < 0:
        parser.error('Positive frame/repetition counts and nonnegative start required')
    if args.compile_stateless and not args.graphs:
        parser.error('--compile-stateless requires graphs')
    if args.fused_int4_epilogue and (not args.deployment or args.linear_backend != 'native'):
        parser.error('Fused INT4 epilogue requires a native S4 deployment')
    if args.kv_format != 'original' and args.store_kv_compute_dtype:
        parser.error('--store-kv-compute-dtype only applies to --kv-format original')
    if args.defer_kv_reserve and args.kv_format == 'original':
        parser.error('--defer-kv-reserve requires a compressed KV format')
    if args.memory_budget_gb is not None and (not np.isfinite(args.memory_budget_gb) or args.memory_budget_gb <= 0):
        parser.error('--memory-budget-gb must be finite and positive')
    if min(args.prefill_mlp_chunk_tokens, args.prefill_frame_batch_size) < 0:
        parser.error('Prefill chunk sizes must be nonnegative')
    quality = json.loads(args.quality_config.read_text())
    ranges, overlap = window_ranges(args.frames, quality)
    files = image_files(args.images, args.start, args.frames)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / 'report.json'
    report = {'status': 'initializing', 'method': args.method, 'quality_config': quality, 'frames': len(files),
              'files': [str(path) for path in files], 'input_start': args.start,
              'image_hw': [294, 518], 'torch_version': torch.__version__,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'allocator_config': os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
              'source': str((args.checkpoint or args.deployment).resolve()),
              'trunk_precision': 'bf16', 'camera_precision': args.camera_precision, 'depth_precision': 'fp32',
              'kv_format': args.kv_format, 'attention_backend': 'sdpa',
              'cache_placement': 'GPU resident compact' if args.kv_format == 'original' else 'GPU resident compressed',
              'compressed_kv_fused_sdpa_decode': args.kv_format != 'original',
              'compressed_kv_deduplicate_special_scales': args.kv_format != 'original',
              'requested_allocator_budget_gb': args.memory_budget_gb,
              'prefill_mlp_chunk_tokens': args.prefill_mlp_chunk_tokens,
              'prefill_frame_batch_size': args.prefill_frame_batch_size,
              'defer_kv_reserve': args.defer_kv_reserve,
              'graphs': args.graphs, 'extended_graphs': args.extended_graphs and args.graphs,
              'compile_stateless': args.compile_stateless, 'prefetch_images': args.prefetch_images,
              'camera_graph': args.camera_graph,
              'fused_int4_epilogue': args.fused_int4_epilogue,
              'store_kv_compute_dtype': args.store_kv_compute_dtype,
              'window_count': len(ranges), 'overlap_input_frames': overlap,
              'timing_scope': 'Unique frames / complete sequence wall time including lazy image decode, H2D, model, D2H, finite checks, repeated overlap and CPU Sim3 stitching. Model load, warmup/compile/capture and disk writing excluded and separately reported.',
              'memory_scope': 'PyTorch allocated and reserved for this process; model load, warmup/capture and measured phases reported separately. CUDA context and other processes excluded.',
              'latency_scope': 'Raw prediction call latency, not final window-aligned output latency; windowed output is buffered.',
              'ground_truth_used_for_inference': False, 'measurements': []}
    write_json(destination, report)
    try:
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('GPU benchmarking requires an accessible CUDA device. No CPU FPS substitutes are reported.')
        torch.cuda.set_device(device)
        if args.memory_budget_gb is not None:
            budget = min(int(args.memory_budget_gb * 1e9), torch.cuda.get_device_properties(device).total_memory)
            torch.cuda.set_per_process_memory_fraction(budget / torch.cuda.get_device_properties(device).total_memory, device)
            report['pytorch_allocator_budget_bytes'] = budget
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
        report.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
                      device_index=torch.cuda.current_device(),
                      gpu_uuid=str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable')),
                      memory_free_before_load_bytes=torch.cuda.mem_get_info(device)[0])
        begin = time.perf_counter()
        if args.checkpoint:
            model = load_quality_float(args.checkpoint, quality).to(device)
            report.update(variant='bf16', linear_backend='bf16')
        else:
            model, _ = load_deployment(args.deployment, device, model_factory=construct_quality_model,
                                        stream_overrides={key: quality[key] for key in STREAM_KEYS},
                                        linear_backend=args.linear_backend)
            named_layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, PackedInt4Linear)]
            layers = [layer for _, layer in named_layers]
            validate_deployment(layers, args.method)
            report.update(variant='dsfq' if any(layer.smoothing is not None for layer in layers) else
                          'rotation' if any(layer.signs is not None for layer in layers) else 'naive',
                          linear_backend='native_s4' if args.linear_backend == 'native' else 'w4a4_codes_int8_gemm',
                          quantized_layers=len(layers), quantized_layer_names=[name for name, _ in named_layers])
            build = args.deployment.parent / 'build.json'
            if build.exists():
                report['build_metadata'] = json.loads(build.read_text())
        model.eval().requires_grad_(False)
        if args.camera_precision != 'fp32':
            enable_camera_compute_dtype(model, torch.float16 if args.camera_precision == 'fp16' else torch.bfloat16)
        model.depth_head.register_forward_pre_hook(depth_chunk_one, with_kwargs=True)
        if args.fused_int4_epilogue:
            enable_fused_int4_epilogue(model)
        torch.cuda.synchronize(device)
        report.update(model_load_seconds=time.perf_counter() - begin,
                      load_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      load_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        graphs = stateless_cuda_graphs(model, share_pool=True, extended=args.extended_graphs,
                                       compile_stateless=args.compile_stateless) if args.graphs else nullcontext()
        stored_frames = quality['window_size']
        camera_graph = camera_cuda_graph(model, stored_frames) if args.camera_graph else nullcontext()
        prefill = (chunked_prefill_mlp(model, args.prefill_mlp_chunk_tokens, frame_batch_size=args.prefill_frame_batch_size)
                   if args.prefill_mlp_chunk_tokens or args.prefill_frame_batch_size else nullcontext())
        rows = []
        with prefill, quality_cache(model, quality, args.kv_format,
                           store_compute_dtype=args.store_kv_compute_dtype,
                           defer_full_reservation=args.defer_kv_reserve) as cache, graphs as controller, \
                camera_graph as camera_controller, \
                torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            warm_count = min(max(args.warmup_frames, quality['num_scale_frames'] + 1), len(files))
            warm_images = LazyImageSequence(files[:warm_count], 518, max_height=294)
            begin = time.perf_counter()
            with closing(iter_predictions(model, warm_images, quality['num_scale_frames'], quality['keyframe_interval'],
                                          prefetch_images=args.prefetch_images)) as iterator:
                for _, prediction in iterator:
                    if not prediction_is_finite(prediction):
                        raise ValueError('Nonfinite warmup prediction')
            del prediction
            torch.cuda.synchronize(device)
            report.update(warmup_seconds=time.perf_counter() - begin, warmup_frames=warm_count,
                          startup_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          startup_peak_reserved_bytes=torch.cuda.max_memory_reserved(device), status='measuring')
            write_json(destination, report)
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats(device)
                prediction, batch_rows, measurement = run_windows(model, files, quality,
                    prefetch=args.prefetch_images, repeat=repeat, cache=cache)
                torch.cuda.synchronize(device)
                measurement.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                rows.extend(batch_rows)
                report['measurements'].append(measurement)
                write_json(destination, report)
                if repeat == 0:
                    begin = time.perf_counter()
                    torch.save(prediction, args.output / 'predictions.pt')
                    report['prediction_save_seconds'] = time.perf_counter() - begin
                del prediction
            if controller is not None:
                report['graph_captures'] = controller.captures
                report['stateless_compilations'] = controller.compilations
            if camera_controller is not None:
                report['camera_graph_captures'] = camera_controller.captures
                report['camera_graph_cache_bytes'] = camera_controller.cache_bytes
        fps = [row['unique_fps'] for row in report['measurements']]
        calls = [row['call_ms'] for row in rows if row['frames_in_call'] == 1]
        report.update(status='complete', unique_fps_median=float(np.median(fps)), unique_fps_min=min(fps),
                      single_frame_call_ms_p50_p95_p99=np.percentile(calls, [50, 95, 99]).tolist() if calls else None,
                      peak_allocated_bytes=max(report['startup_peak_allocated_bytes'],
                                               *(row['peak_allocated_bytes'] for row in report['measurements'])),
                      peak_reserved_bytes=max(report['startup_peak_reserved_bytes'],
                                              *(row['peak_reserved_bytes'] for row in report['measurements'])))
        with (args.output / 'frame_latency.csv').open('w') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        if device.type == 'cuda' and torch.cuda.is_initialized():
            report.update(failure_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          failure_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        raise
    finally:
        write_json(destination, report)


if __name__ == '__main__':
    main()
