"""Render a saved inference run as RGB, depth, and an orbiting RGB point map.

This is a qualitative display: confidence filtering affects only the point map.
It never changes predictions, evaluation metrics, or quantization parameters.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps, font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lingbot_map.quantization.lazy_images import crop_max_height
from lingbot_map.utils.load_fn import load_and_preprocess_images
from qmap.visualize import depth_at, load_run, unproject, write_ply


BACKGROUND = (18, 25, 37)


def rgb_at(report, frame):
    h, w = report['image_hw']
    rgb = crop_max_height(load_and_preprocess_images(
        [report['files'][frame]], image_size=w, patch_size=14), h)[0]
    return (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def point_map(run, indices, stride, confidence_percentile):
    report, prediction, cameras, intrinsics = run
    h, w = report['image_hw']
    v, u = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    clouds, colors, counts = [], [], []
    for index in indices:
        depth = depth_at(prediction, index)
        confidence = prediction['depth_conf'][0, index].float().numpy()
        finite = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
        if not finite.any():
            raise ValueError(f'No finite positive depths in frame {index}')
        threshold = np.percentile(confidence[finite], confidence_percentile)
        keep = finite[v, u] & (confidence[v, u] >= threshold)
        clouds.append(unproject(depth, cameras[index], intrinsics[index], u[keep], v[keep]))
        colors.append(rgb_at(report, index)[v[keep], u[keep]])
        counts.append(int(keep.sum()))
    return np.concatenate(clouds), np.concatenate(colors), counts


def view_coordinates(points, camera, yaw, elevation):
    """Rotate the virtual view in the selected input camera's coordinate frame."""
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    yaw, elevation = np.deg2rad([yaw, elevation])
    horizontal = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                           [-np.sin(yaw), 0, np.cos(yaw)]])
    vertical = np.array([[1, 0, 0], [0, np.cos(elevation), -np.sin(elevation)],
                         [0, np.sin(elevation), np.cos(elevation)]])
    return local @ (horizontal @ vertical)


def framing(views, width, height):
    # Use all retained points. No quantile crop or hidden removal of outliers.
    low = np.min([view[:, :2].min(0) for view in views], axis=0)
    high = np.max([view[:, :2].max(0) for view in views], axis=0)
    scale = .92 * min(width / max(high[0] - low[0], 1e-6),
                      height / max(high[1] - low[1], 1e-6))
    return (low + high) / 2, scale


def render_points(view, colors, width, height, center, scale, radius=1):
    """Orthographic point splats with a nearest-point z-buffer (CPU only)."""
    xy = np.rint((view[:, :2] - center) * scale + [width / 2, height / 2]).astype(np.int32)
    keep = (xy[:, 0] >= radius) & (xy[:, 0] < width - radius)
    keep &= (xy[:, 1] >= radius) & (xy[:, 1] < height - radius)
    xy, colors, z = xy[keep], colors[keep], view[keep, 2]
    # Stable tie-breaking for points at the same depth, independent of splat order.
    order = np.argsort(z, kind='stable')
    xy, colors = xy[order], colors[order]
    depth_rank = np.full(width * height, len(xy), dtype=np.int32)
    ranks = np.arange(len(xy), dtype=np.int32)
    pixels = xy[:, 1] * width + xy[:, 0]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                np.minimum.at(depth_rank, pixels + dy * width + dx, ranks)
    palette = np.concatenate((colors, np.array([BACKGROUND], dtype=np.uint8)))
    return Image.fromarray(palette[depth_rank].reshape(height, width, 3))


def font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(
        family='DejaVu Sans', weight='bold' if bold else 'normal'))
    return ImageFont.truetype(path, size)


def display_board(rgb, depth, rendered, title, frame, map_frames, kv_format, image_hw):
    board = Image.new('RGB', (2160, 1120), 'white')
    draw = ImageDraw.Draw(board)
    draw.text((48, 24), title, font=font(40, True), fill='#162033')
    draw.text((48, 79), f'DSFQ  /  W4A4 + {kv_format.upper()} KV', font=font(23), fill='#526176')
    draw.text((48, 139), f'RGB input  /  frame {frame}', font=font(24, True), fill='#162033')
    board.paste(Image.fromarray(rgb).resize((624, 354), Image.Resampling.LANCZOS), (48, 181))
    draw.text((48, 573), 'Predicted relative depth', font=font(24, True), fill='#162033')
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.all():
        raise ValueError('Gallery depth panel expects finite positive predictions')
    low, high = np.percentile(depth, [2, 98])
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    depth_rgb = (colormaps['magma_r'](normalized)[..., :3] * 255).round().astype(np.uint8)
    board.paste(Image.fromarray(depth_rgb).resize((624, 354), Image.Resampling.NEAREST), (48, 615))
    gradient = np.tile(np.linspace(0, 1, 624), (14, 1))
    bar = (colormaps['magma_r'](gradient)[..., :3] * 255).astype(np.uint8)
    board.paste(Image.fromarray(bar), (48, 988))
    draw.text((48, 1011), 'Near', font=font(19), fill='#526176')
    draw.text((633, 1011), 'Far', font=font(19), fill='#526176')
    draw.text((718, 139), f'RGB point map  /  {map_frames} sampled views',
              font=font(24, True), fill='#162033')
    board.paste(rendered.resize((1394, 856), Image.Resampling.LANCZOS), (718, 181))
    draw.text((48, 1074), f'{image_hw[0]} × {image_hw[1]} model input  ·  Predicted depth + camera poses  ·  Confidence-filtered map',
              font=font(21), fill='#526176')
    return board, [float(low), float(high)]


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qmap gallery', description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='DSFQ reconstruction')
    parser.add_argument('--frame', type=int, default=100, help='Preview frame, relative to this run')
    parser.add_argument('--map-start', type=int, default=0)
    parser.add_argument('--map-end', type=int, default=248, help='Exclusive; relative to this run')
    parser.add_argument('--map-frames', type=int, default=64)
    parser.add_argument('--pixel-stride', type=int, default=3)
    parser.add_argument('--confidence-percentile', type=float, default=20)
    parser.add_argument('--yaw', type=float, default=18)
    parser.add_argument('--elevation', type=float, default=8)
    parser.add_argument('--animation-frames', type=int, default=0, help='0 disables the virtual-view orbit GIF')
    args = parser.parse_args(argv)
    run = load_run(args.run)
    report, prediction, cameras, _ = run
    if report.get('method') != 'dsfq':
        parser.error('This DSFQ showcase command requires a DSFQ run; use visualize for method comparisons')
    if not (0 <= args.frame < report['frames'] and 0 <= args.map_start < args.map_end <= report['frames']):
        parser.error('Preview/map frame ranges must be within the completed run')
    if not (min(args.map_frames, args.pixel_stride) > 0 and 0 <= args.confidence_percentile < 100):
        parser.error('Positive sampling counts and confidence percentile in [0, 100) required')
    if args.animation_frames < 0 or not np.isfinite([args.yaw, args.elevation]).all():
        parser.error('Animation frame count must be nonnegative and view angles finite')
    args.output.mkdir(parents=True, exist_ok=True)
    indices = np.unique(np.linspace(args.map_start, args.map_end - 1,
                        min(args.map_frames, args.map_end - args.map_start), dtype=int))
    points, colors, counts = point_map(run, indices, args.pixel_stride, args.confidence_percentile)
    camera = cameras[args.frame]
    view = view_coordinates(points, camera, args.yaw, args.elevation)
    center, scale = framing([view], 1394, 856)
    rendered = render_points(view, colors, 1394, 856, center, scale, radius=1)
    rgb, depth = rgb_at(report, args.frame), depth_at(prediction, args.frame)
    source_frame = report['input_start'] + args.frame
    board, depth_range = display_board(rgb, depth, rendered, args.title, source_frame, len(indices),
                                       report['kv_format'], report['image_hw'])
    board.save(args.output / 'gallery.png', optimize=True)
    rendered.save(args.output / 'point_map.png', optimize=True)
    write_ply(args.output / 'point_map.ply', points, colors)
    if args.animation_frames:
        angles = args.yaw + 12 * np.sin(np.linspace(0, 2 * np.pi, args.animation_frames, endpoint=False))
        # One framing across the complete orbit, so the apparent scale does not pulse.
        bounds = [view_coordinates(points, camera, a, args.elevation) for a in (angles.min(), args.yaw, angles.max())]
        center, scale = framing(bounds, 960, 490)
        del bounds
        frames = []
        for number, angle in enumerate(angles):
            view = view_coordinates(points, camera, angle, args.elevation)
            frame = Image.new('RGB', (960, 590), BACKGROUND)
            frame.paste(render_points(view, colors, 960, 490, center, scale, radius=1), (0, 50))
            draw = ImageDraw.Draw(frame)
            draw.text((22, 20), args.title, font=font(22, True), fill='white')
            draw.text((22, 552), 'DSFQ · fixed point map / virtual camera orbit', font=font(17), fill='#ced8e5')
            frames.append(frame)
            print(f'Orbit frame {number + 1}/{len(angles)}', flush=True)
        # Shared palette avoids color flicker; no scene content is generated or inpainted.
        palette = frames[0].quantize(colors=256)
        frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        frames[0].save(args.output / 'orbit.gif', save_all=True, append_images=frames[1:], duration=80,
                       loop=0, optimize=False, disposal=2)
    metadata = {
        'purpose': 'Qualitative display; not an accuracy or throughput benchmark',
        'sequence': Path(report['files'][0]).parent.parent.name,
        'run_name': args.run.name, 'method': report['method'], 'gpu': report['gpu'],
        'cuda_visible_devices': report['cuda_visible_devices'], 'backend': report['linear_backend'],
        'kv_format': report['kv_format'], 'inference_frames': report['frames'],
        'input_start': report['input_start'], 'image_hw': report['image_hw'],
        'quality_config': report['quality_config'], 'preview_source_frame': source_frame,
        'preview_filename': Path(report['files'][args.frame]).name,
        'map_source_range': [report['input_start'] + args.map_start, report['input_start'] + args.map_end],
        'map_source_frames': (indices + report['input_start']).tolist(),
        'pixel_stride': args.pixel_stride, 'confidence_percentile_per_frame': args.confidence_percentile,
        'retained_points_per_frame': counts, 'map_points': len(points),
        'geometry': 'Predicted depth unprojected with predicted intrinsics and camera-to-world poses; no GT, ICP, mesh fitting or inpainting',
        'rendering': 'Orthographic z-buffer point splats, radius 1 pixel, all retained points included in static framing',
        'map_view_angles_degrees': {'yaw': args.yaw, 'elevation': args.elevation},
        'depth_display': {'colormap': 'magma_r', 'percentile_limits': [2, 98], 'raw_limits': depth_range,
                          'mask': 'none', 'units': 'prediction units, not metric GT'},
        'animation': {'frames': args.animation_frames, 'duration_ms': 80, 'yaw_amplitude_degrees': 12,
                      'interpretation': 'Fixed predicted point map; only the virtual camera moves'},
        'calibration_performed': False, 'evaluation_metrics_changed': False,
    }
    (args.output / 'gallery.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(f'Saved {args.output} ({len(points):,} RGB points)', flush=True)


if __name__ == '__main__':
    main()
