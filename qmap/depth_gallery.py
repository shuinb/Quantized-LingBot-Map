"""Show the same RGB frames and depth predictions across completed runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qmap.gallery import rgb_at
from qmap.visualize import depth_at, load_run


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qmap depth-gallery', description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Label=run_directory; BF16 first')
    parser.add_argument('--frames', nargs='+', type=int, default=[100, 365, 630], help='Indices relative to each run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {label: load_run(folder) for label, folder in (item.split('=', 1) for item in args.runs)}
    first = next(iter(runs.values()))
    if first[0].get('method', first[0].get('variant')) != 'bf16':
        parser.error('Put a BF16 reference run first')
    if any(frame < 0 or frame >= first[0]['frames'] for frame in args.frames):
        parser.error('Frame indices must be within the completed sequence')
    for report, *_ in runs.values():
        for field in ('files', 'frames', 'image_hw', 'quality_config'):
            if report[field] != first[0][field]:
                raise ValueError(f'Cannot compare different {field}')
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    columns, rows = len(runs) + 1, len(args.frames)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.15 * rows + .65), squeeze=False)
    fig.subplots_adjust(left=.04, right=.995, top=.94, bottom=.07, wspace=.035, hspace=.1)
    records = []
    for row, frame in enumerate(args.frames):
        axes[row, 0].imshow(rgb_at(first[0], frame))
        values, medians = {}, {}
        for label, (_, prediction, *_rest) in runs.items():
            depth = depth_at(prediction, frame)
            if not np.isfinite(depth).all() or not (depth > 0).all():
                raise ValueError(f'Nonfinite or nonpositive depth: {label}, frame {frame}')
            medians[label] = float(np.median(depth))
            values[label] = depth / medians[label]
        low, high = np.percentile(values[next(iter(runs))], [2, 98])
        for column, (label, depth) in enumerate(values.items(), 1):
            axes[row, column].imshow(depth, vmin=low, vmax=high, cmap='magma_r', interpolation='nearest')
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        source_frame = first[0]['input_start'] + frame
        axes[row, 0].set_ylabel(f'Frame {source_frame}', fontsize=13, labelpad=12)
        records.append({'source_frame': source_frame, 'run_frame': frame,
                        'filename': Path(first[0]['files'][frame]).name,
                        'depth_medians': medians, 'shared_color_limits': [float(low), float(high)]})
    for ax, label in zip(axes[0], ['RGB input', *runs]):
        ax.text(.5, 1.035, label, ha='center', va='bottom', transform=ax.transAxes, fontsize=15, weight='bold')
    cax = fig.add_axes([.72, .027, .22, .012])
    cax.imshow(np.linspace(0, 1, 256)[None], cmap='magma_r', aspect='auto')
    cax.set_axis_off()
    fig.text(.705, .033, 'Near', ha='right', va='center', fontsize=11)
    fig.text(.955, .033, 'Far', ha='left', va='center', fontsize=11)
    fig.text(.04, .033, 'Relative depth · per-frame median normalization · shared colors within each row · no confidence mask',
             fontsize=11, va='center', color='#526176')
    fig.savefig(args.output / 'depth_comparison.png', dpi=150)
    fig.savefig(args.output / 'depth_comparison.pdf')
    plt.close(fig)
    metadata = {
        'purpose': 'Qualitative relative-depth comparison; BF16 is a prediction reference, not ground truth',
        'runs': {label: {'run_name': Path(folder).name, 'method': run[0].get('method', run[0].get('variant')),
                         'kv_format': run[0]['kv_format'], 'backend': run[0].get('linear_backend')}
                 for (label, run), folder in zip(runs.items(), [item.split('=', 1)[1] for item in args.runs])},
        'sequence_frames': first[0]['frames'], 'image_hw': first[0]['image_hw'],
        'normalization': 'Divide each method/frame by its own median depth to display monocular shape, not scale',
        'color_limits': 'BF16 2nd and 98th percentiles after normalization; same limits for every method in each row',
        'confidence_filter': False, 'rows': records, 'evaluation_metrics_changed': False,
    }
    (args.output / 'depth_comparison.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(f'Saved depth comparison to {args.output}')


if __name__ == '__main__':
    main()
