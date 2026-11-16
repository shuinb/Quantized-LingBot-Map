"""Render trajectories and RGB point maps from saved, actually inferred depths."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from evo.core.geometry import umeyama_alignment

from lingbot_map.quantization.lazy_images import crop_max_height
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from qmap.common import crop_geometry


def write_ply(path, points, colors):
    vertices = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for index, key in enumerate(('x', 'y', 'z')):
        vertices[key] = points[:, index]
    for index, key in enumerate(('red', 'green', 'blue')):
        vertices[key] = colors[:, index]
    with path.open('wb') as handle:
        handle.write(('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(len(points)) +
                      '\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\n'
                      'property uchar green\nproperty uchar blue\nend_header\n').encode())
        vertices.tofile(handle)


def unproject(depth, camera, intrinsic, u, v):
    z = depth[v, u]
    local = np.stack(((u - intrinsic[0, 2]) * z / intrinsic[0, 0],
                      (v - intrinsic[1, 2]) * z / intrinsic[1, 1], z), -1)
    return local @ camera[:3, :3].T + camera[:3, 3]


def load_run(source):
    folder = Path(source)
    report = json.loads((folder / 'report.json').read_text())
    report['stream_config'] = report['quality_config']
    if report['status'] != 'complete':
        raise ValueError(f'Inference is incomplete: {folder}')
    predictions = torch.load(folder / 'predictions.pt', map_location='cpu', mmap=True, weights_only=True)
    if predictions['pose_enc'].shape[1] != report['frames']:
        raise ValueError('Incomplete saved prediction sequence')
    camera, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'].float(), report['image_hw'])
    return report, predictions, camera[0].numpy(), intrinsic[0].numpy()


def depth_at(prediction, frame):
    return prediction['depth'][0, frame, ..., 0].float().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qmap visualize', description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Label=run_directory; BF16 first when no GT is supplied')
    parser.add_argument('--data', type=Path, help='TUM data with manifest.json; otherwise first run is prediction reference, NOT GT')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map-frames', type=int, default=64)
    parser.add_argument('--pixel-stride', type=int, default=8)
    args = parser.parse_args(argv)
    if args.map_frames < 1 or args.pixel_stride < 1:
        parser.error('Sampling intervals must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {label: load_run(folder) for label, folder in (entry.split('=', 1) for entry in args.runs)}
    first = next(iter(runs.values()))
    config = first[0]
    for report, *_ in runs.values():
        for name in ('files', 'frames', 'image_hw', 'stream_config'):
            actual, expected = report[name], config[name]
            if actual != expected:
                raise ValueError(f'Cannot overlay different {name}')
    h, w = config['image_hw']
    gt_pairs = None
    if args.data:
        manifest = json.loads((args.data / 'manifest.json').read_text())
        pairs = {Path(pair['rgb']).name: pair for pair in manifest['pairs']}
        gt_pairs = [pairs[Path(path).name] for path in config['files']]
        reference = np.tile(np.eye(4), (config['frames'], 1, 1))
        for index, pair in enumerate(gt_pairs):
            reference[index, :3, :3] = Rotation.from_quat(pair['pose_row'][4:8]).as_matrix()
            reference[index, :3, 3] = pair['pose_row'][1:4]
        reference_label, units = 'GT', 'm'
    else:
        reference = first[2]
        reference_label, units = next(iter(runs)) + ' (prediction reference)', 'reference units'
    aligned, transforms, errors = {}, {}, {}
    for label, (_, _, camera, _) in runs.items():
        rotation, translation, scale = umeyama_alignment(camera[:, :3, 3].T, reference[:, :3, 3].T, with_scale=True)
        transforms[label] = (rotation, translation, scale)
        aligned[label] = scale * camera[:, :3, 3] @ rotation.T + translation
        errors[label] = np.linalg.norm(aligned[label] - reference[:, :3, 3], axis=-1)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    compare_error = gt_pairs is not None or len(runs) > 1
    fig, axes = plt.subplots(1, 3 if compare_error else 2,
                             figsize=(16 if compare_error else 11, 4.8), constrained_layout=True)
    for ax, (a, b) in zip(axes[:2], ((0, 2), (0, 1))):
        ax.plot(reference[:, a, 3], reference[:, b, 3], color='black', lw=1.4, label=reference_label)
        for label, positions in aligned.items():
            if gt_pairs is None and label == next(iter(runs)):
                continue
            ax.plot(positions[:, a], positions[:, b], lw=1.1, alpha=.85, label=label)
        ax.set(xlabel=f'{"XYZ"[a]} ({units})', ylabel=f'{"XYZ"[b]} ({units})')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    if compare_error:
        for label, error in errors.items():
            if gt_pairs is None and label == next(iter(runs)):
                continue
            axes[2].plot(error, label=label, lw=.9)
        axes[2].set(xlabel='Frame', ylabel=f'Position error ({units})', title='One global Sim(3); no segment realignment')
        axes[2].grid(alpha=.2)
    fig.savefig(args.output / 'trajectory_comparison.png', dpi=180, bbox_inches='tight')
    fig.savefig(args.output / 'trajectory_comparison.svg', bbox_inches='tight')
    plt.close(fig)

    available = set(range(config['frames']))
    for _, prediction, _, _ in runs.values():
        if 'sample_by_frame' in prediction:
            available.intersection_update(prediction['sample_by_frame'])
    if not available:
        raise ValueError('No common dense map samples')
    available = np.array(sorted(available))
    indices = available[np.unique(np.linspace(0, len(available) - 1, min(args.map_frames, len(available)), dtype=int))]
    rgb = crop_max_height(load_and_preprocess_images([config['files'][index] for index in indices], image_size=w, patch_size=14), h)
    pixels_v, pixels_u = np.meshgrid(np.arange(0, h, args.pixel_stride), np.arange(0, w, args.pixel_stride), indexing='ij')
    u, v = pixels_u.ravel(), pixels_v.ravel()
    clouds = {label: [] for label in runs}
    if gt_pairs is not None:
        clouds = {'GT': [], **clouds}
    colors = []
    depth_agreement = {label: [] for label in runs}
    for offset, index in enumerate(indices):
        if gt_pairs is not None:
            pair = gt_pairs[index]
            source = cv2.imread(config['files'][index])
            raw = cv2.imread(str(args.data / 'depth' / Path(pair['depth']).name), cv2.IMREAD_UNCHANGED)
            if source is None or raw is None:
                raise ValueError('Missing original RGB/depth')
            ih, iw = source.shape[:2]
            resized_h, final_h, top = crop_geometry((ih, iw), image_size=w, max_height=h)
            depth = cv2.resize(raw, (w, resized_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)[top:top + final_h] / 5000
            intrinsic = np.array([[517.306408 * w / iw, 0, 318.643040 * w / iw],
                                  [0, 516.469215 * resized_h / ih, 255.313989 * resized_h / ih - top], [0, 0, 1]])
            valid = np.isfinite(depth[v, u]) & (depth[v, u] > 0) & (depth[v, u] < 80)
            clouds['GT'].append(unproject(depth, reference[index], intrinsic, u[valid], v[valid]))
        else:
            depth = depth_at(first[1], index)
            valid = np.isfinite(depth[v, u]) & (depth[v, u] > 0)
        colors.append((rgb[offset].permute(1, 2, 0).numpy()[v[valid], u[valid]] * 255).round().astype(np.uint8))
        for label, (_, prediction, camera, intrinsic) in runs.items():
            depth = depth_at(prediction, index)
            if not np.isfinite(depth).all():
                raise ValueError(f'Nonfinite depth: {label} / {index}')
            if gt_pairs is None:
                expected_depth = depth_at(first[1], index)
                valid_depth = np.isfinite(expected_depth) & (expected_depth > 1e-6)
                if not valid_depth.any():
                    raise ValueError(f'No valid reference depth at frame {index}')
                depth_agreement[label].append(float(np.mean(
                    np.abs(depth[valid_depth] - expected_depth[valid_depth]) / expected_depth[valid_depth])))
            world = unproject(depth, camera[index], intrinsic[index], u[valid], v[valid])
            rotation, translation, scale = transforms[label]
            clouds[label].append(scale * world @ rotation.T + translation)
    colors = np.concatenate(colors)
    clouds = {label: np.concatenate(parts) for label, parts in clouds.items()}
    all_points = np.concatenate(list(clouds.values()))
    lower, upper = all_points.min(0), all_points.max(0)
    center = (lower + upper) / 2
    radius = max((upper - lower).max() / 2, 1e-6)
    columns = 2 if len(clouds) == 4 else min(3, len(clouds))
    rows = (len(clouds) + columns - 1) // columns
    fig = plt.figure(figsize=(5 * columns, 5.5 * rows), constrained_layout=True)
    for number, (label, points) in enumerate(clouds.items(), 1):
        filename = ''.join(c if c.isalnum() or c in '-_' else '_' for c in label)
        write_ply(args.output / (filename + '.ply'), points, colors)
        ax = fig.add_subplot(rows, columns, number, projection='3d')
        ax.scatter(*points.T, c=colors / 255, s=.35, depthshade=False, rasterized=True)
        ax.view_init(elev=22, azim=-65)
        ax.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius),
               zlim=(center[2] - radius, center[2] + radius), xlabel='X', ylabel='Y', zlabel='Z', title=label)
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle(f'Depth-derived RGB maps ({units}); same frames, pixels, axes and global alignment')
    fig.savefig(args.output / 'mapping_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    summary = {'reference': reference_label, 'units': units, 'frames': config['frames'], 'image_hw': [h, w],
               'stream_configs': {label: report['stream_config'] for label, (report, *_) in runs.items()},
               'camera_precision': {label: report.get('camera_precision', 'fp32') for label, (report, *_) in runs.items()},
               'additional_quantization': {label: report.get('additional_quantization', {}) for label, (report, *_) in runs.items()},
               'compile_stateless': {label: report.get('compile_stateless', False) for label, (report, *_) in runs.items()},
               'depth_position_cache': {label: report.get('depth_position_cache', False) for label, (report, *_) in runs.items()},
               'camera_graph': {label: report.get('camera_graph', False) for label, (report, *_) in runs.items()},
               'fused_int4_epilogue': {label: report.get('fused_int4_epilogue', False) for label, (report, *_) in runs.items()},
               'fuse_sage_value_decode': {label: report.get('fuse_sage_value_decode', False) for label, (report, *_) in runs.items()},
               'alignment': 'One full-trajectory Sim3 per prediction; same transform applied to its map',
               'point_source': 'Predicted depth + decoded camera; no independent point head, ICP or mesh ground truth',
               'map_frame_indices': indices.tolist(), 'pixel_stride': args.pixel_stride,
               'map_points_each': len(colors), 'axis_bounds_include_all_points': True,
               'trajectory_position_rmse': {label: float(np.sqrt(np.mean(error ** 2))) for label, error in errors.items()},
               'trajectory_bins': {label: [
                   {'start': start, 'end_exclusive': min(start + 1000, len(error)),
                    'position_rmse': float(np.sqrt(np.mean(error[start:start + 1000] ** 2)))}
                   for start in range(0, len(error), 1000)] for label, error in errors.items()},
               'sources': args.runs}
    if gt_pairs is None:
        reference_cloud = clouds[next(iter(runs))]
        summary['sampled_agreement_not_gt'] = {
            'scope': 'Depth uses all valid pixels of the common uniform map frames; point EPE uses the map pixel stride. Trajectory above uses every frame.',
            'dense_frames': len(indices),
            'depth_absrel_raw': {label: float(np.mean(values)) for label, values in depth_agreement.items()},
            'point_epe_reference_units': {label: float(np.linalg.norm(points - reference_cloud, axis=-1).mean())
                                          for label, points in clouds.items()}}
    (args.output / 'visualization.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
