"""Compare measured memory variants against a BF16 quality reference.

Runtime differences are explicitly listed. This report does not attribute speed
differences across hardware/cache settings to weight quantization alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qmap.evaluation.geometry import official_geometry_metrics
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from qmap.evaluation.quality import (
    depth_consistency, depth_ground_truth_metrics, iter_tum_depth, load_pose_ground_truth,
)


RUNTIME_FIELDS = ('gpu', 'gpu_uuid', 'linear_backend', 'trunk_precision', 'camera_precision',
                  'depth_precision', 'kv_format', 'attention_backend', 'cache_placement',
                  'graphs', 'extended_graphs', 'compile_stateless', 'camera_graph',
                  'store_kv_compute_dtype', 'prefetch_images', 'fused_int4_epilogue',
                  'compressed_kv_fused_sdpa_decode', 'compressed_kv_deduplicate_special_scales',
                  'allocator_config', 'requested_allocator_budget_gb', 'prefill_mlp_chunk_tokens',
                  'prefill_frame_batch_size', 'defer_kv_reserve', 'warmup_frames')


def read_run(path):
    report = json.loads((path / 'report.json').read_text())
    if report.get('status') != 'complete' or not report.get('measurements'):
        raise ValueError(f'Inference has not completed: {path}')
    prediction = torch.load(path / 'predictions.pt', map_location='cpu', mmap=True, weights_only=True)
    if prediction['pose_enc'].shape[1] != report['frames'] or prediction['depth'].shape[1] != report['frames']:
        raise ValueError(f'Predictions must cover every reported input frame: {path}')
    return report, prediction


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qmap evaluate', description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True, help='Label=run_directory')
    parser.add_argument('--memory-reports', nargs='*', default=[], help='Label=measure_gpu_memory_JSON')
    parser.add_argument('--ground-truth', type=Path)
    parser.add_argument('--ground-truth-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.ground_truth) != bool(args.ground_truth_manifest):
        parser.error('GT pose arrays and image manifest must be supplied together')
    runs = [entry.split('=', 1) for entry in args.runs]
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        parser.error('Run labels must be unique')
    memory_paths = dict(entry.split('=', 1) for entry in args.memory_reports)
    if set(memory_paths) - set(labels):
        parser.error('Memory report labels must match a run label')
    reference_report, reference = read_run(args.reference)
    if reference_report.get('variant') != 'bf16':
        raise ValueError('The reference must be a BF16 quality run')
    ref_pose = pose_encoding_to_extri_intri(reference['pose_enc'].float(), reference_report['image_hw'])[0]
    truth = load_pose_ground_truth(args.ground_truth, args.ground_truth_manifest,
                                   reference_report['files'], reference_report['input_start']) if args.ground_truth else None
    rows = []
    for label, folder in runs:
        path = Path(folder)
        report, prediction = read_run(path)
        for field in ('files', 'frames', 'input_start', 'quality_config', 'image_hw'):
            if report[field] != reference_report[field]:
                raise ValueError(f'Quality comparison requires the same {field}: {label}')
        pose = pose_encoding_to_extri_intri(prediction['pose_enc'].float(), report['image_hw'])[0]
        fps = [value['unique_fps'] for value in report['measurements']]
        row = dict(label=label, run=str(path.resolve()), variant=report['variant'],
                   frames=report['frames'], runtime={field: report.get(field) for field in RUNTIME_FIELDS},
                   runtime_differences_from_bf16={field: {'reference': reference_report.get(field), 'run': report.get(field)}
                       for field in RUNTIME_FIELDS if report.get(field) != reference_report.get(field)},
                   quantization_config=report.get('build_metadata', {}).get('quantization_config'),
                   fps_repeats=fps, fps_median=float(np.median(fps)), fps_min=min(fps),
                   peak_allocated_bytes=report['peak_allocated_bytes'], peak_reserved_bytes=report['peak_reserved_bytes'])
        if path.resolve() != args.reference.resolve():
            row['pose_consistency_vs_bf16'] = official_geometry_metrics({'c2w': ref_pose}, {'c2w': pose})
            row['pose_consistency_vs_bf16']['reference_kind'] = 'BF16 prediction, not ground truth'
            row['depth_consistency_vs_bf16'] = depth_consistency(reference['depth'], prediction['depth'])
        if truth is not None:
            row['pose_ground_truth'] = official_geometry_metrics({'c2w': truth}, {'c2w': pose})
            row['depth_ground_truth'] = depth_ground_truth_metrics(prediction['depth'], iter_tum_depth(
                args.ground_truth_manifest, report['files'], report['input_start'], report['image_hw']))
        if label in memory_paths:
            memory_path = Path(memory_paths[label])
            memory = json.loads(memory_path.read_text())
            # Ordinary command/output matching prevents attaching a different run's sampled peak.
            command = memory['command']
            output_index = command.index('--output') + 1
            if Path(command[output_index]).resolve() != path.resolve() or memory['status'] != 'complete':
                raise ValueError(f'Process memory measurement does not match the completed run: {label}')
            row['process_memory'] = {key: memory[key] for key in (
                'sampled_peak_process_bytes', 'sample_count', 'sample_interval_seconds', 'sampler', 'scope')}
            row['process_memory']['source'] = str(memory_path.resolve())
        process_peak = row.get('process_memory', {}).get('sampled_peak_process_bytes')
        row['measured_memory_below_6GB'] = (max(row['peak_allocated_bytes'], row['peak_reserved_bytes'], process_peak) < 6e9
                                          if process_peak is not None else None)
        rows.append(row)
        print(f'Analyzed {label}: {report["frames"]} frames', flush=True)
    result = dict(reference=str(args.reference.resolve()), rows=rows,
                  ground_truth=str(args.ground_truth.resolve()) if args.ground_truth else None,
                  caveat='BF16 consistency is not GT accuracy. Runtime differences are explicit; no causal speedup is inferred across hardware/cache changes. Memory uses decimal GB and includes startup. Process memory is sampled and can miss brief transients; allocated/reserved peaks are recorded separately.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# Memory and quality comparison', '', '| Method | GPU / KV | FPS median / min | allocated GB | reserved GB | Sampled process peak GB |',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        peak = row.get('process_memory', {}).get('sampled_peak_process_bytes')
        process = '—' if peak is None else f'{peak / 1e9:.4f}'
        lines.append(f'| {row["label"]} | {row["runtime"]["gpu"]} / {row["runtime"]["kv_format"]} | '
                     f'{row["fps_median"]:.3f} / {row["fps_min"]:.3f} | {row["peak_allocated_bytes"] / 1e9:.4f} | '
                     f'{row["peak_reserved_bytes"] / 1e9:.4f} | {process} |')
    lines += ['', '| Method | Trajectory ATE vs BF16 | Depth AbsRel vs BF16 | GT ATE (m) | GT depth AbsRel |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        metrics = [row.get('pose_consistency_vs_bf16', {}).get('camera', {}).get('ate'),
                   row.get('depth_consistency_vs_bf16', {}).get('abs_rel'),
                   row.get('pose_ground_truth', {}).get('camera', {}).get('ate'),
                   row.get('depth_ground_truth', {}).get('metrics', {}).get('abs_rel')]
        lines.append('| ' + row['label'] + ' | ' + ' | '.join('—' if v is None else f'{v:.6f}' for v in metrics) + ' |')
    lines += ['', result['caveat'], '']
    (args.output / 'comparison.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
