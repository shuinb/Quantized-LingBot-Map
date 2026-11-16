"""Depth and pose adapters shared by evaluation and tests."""
import json
from pathlib import Path
import numpy as np
import torch

def depth_consistency(reference, output):
    if reference.shape != output.shape or reference.shape[0] != 1:
        raise ValueError('Depth comparison requires matching [1,S,H,W,1]')
    abs_rel, squared, count = 0., 0., 0
    for expected, predicted in zip(reference[0], output[0]):
        expected, predicted = expected.float(), predicted.float()
        valid = torch.isfinite(expected) & (expected > 1e-8)
        if not torch.isfinite(predicted[valid]).all():
            raise ValueError('Nonfinite output on BF16-valid pixels')
        residual = predicted[valid] - expected[valid]
        abs_rel += (residual.abs() / expected[valid]).double().sum().item()
        squared += residual.double().square().sum().item()
        count += valid.sum().item()
    if not count:
        raise ValueError('No BF16-valid depth pixels')
    return {'abs_rel': abs_rel / count, 'rmse': (squared / count) ** .5, 'pixels': count,
            'mask': 'same positive finite BF16 pixels for both methods; no confidence/residual filtering',
            'alignment': 'none', 'reference_kind': 'BF16 prediction, not ground truth'}


def selected_tum_pairs(manifest_path, files, start):
    manifest = json.loads(manifest_path.read_text())
    selected = manifest['pairs'][start:start + len(files)]
    expected = [(manifest_path.parent / 'images' / Path(row['rgb']).name).resolve() for row in selected]
    if [str(path) for path in expected] != files:
        raise ValueError('Ground-truth manifest does not match inference frame order')
    return manifest, selected


def load_pose_ground_truth(path, manifest_path, files, start):
    manifest, _ = selected_tum_pairs(manifest_path, files, start)
    # prepare_tum_subset excludes calibration_frames from the saved GT arrays,
    # but keeps those images and manifest pairs. Never offset the arrays twice.
    offset = start - manifest.get('gt_start_frame', manifest.get('calibration_frames', 0))
    if offset < 0:
        raise ValueError('Selected frames precede the available evaluation GT')
    with np.load(path) as arrays:
        truth = torch.from_numpy(arrays['c2w'])[:, offset:offset + len(files)]
    if truth.shape[1] != len(files):
        raise ValueError('Ground-truth array does not cover selected frames')
    return truth


def iter_tum_depth(manifest_path, files, start, image_hw):
    """Read original registered TUM depth with the RGB resize/crop geometry.

    Never enlarge the previously downsampled evaluation_gt.npz depth. RGB uses
    bicubic filtering; depth follows the official nearest-neighbor convention.
    """
    import cv2
    from PIL import Image
    manifest, selected = selected_tum_pairs(manifest_path, files, start)
    height, width = image_hw
    units = manifest['depth_units_per_meter']
    if min(height, width, units) <= 0 or height % 14 or width % 14:
        raise ValueError('Positive depth units and patch-aligned input dimensions required')
    for file, row in zip(files, selected, strict=True):
        with Image.open(file) as rgb:
            original_width, original_height = rgb.size
            if rgb.getexif().get(274, 1) != 1:
                raise ValueError('TUM registered-depth adapter does not support EXIF-rotated RGB')
        source = manifest_path.parent / 'depth' / Path(row['depth']).name
        raw = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(source)
        if raw.shape != (original_height, original_width):
            raise ValueError('Registered TUM RGB/depth source dimensions differ')
        resized_height = round(original_height * width / original_width / 14) * 14
        depth = cv2.resize(raw, (width, resized_height), interpolation=cv2.INTER_NEAREST).astype(np.float32) / units
        # load_and_preprocess_images first crops to image_size (= width), then
        # LazyImageSequence.crop_max_height applies the deployment height limit.
        for limit in (width, height):
            top = max((depth.shape[0] - limit) // 2, 0)
            depth = depth[top:top + min(limit, depth.shape[0])]
        if depth.shape != (height, width):
            raise ValueError('GT preprocessing does not match the actual prediction grid')
        yield depth


def depth_ground_truth_metrics(prediction, truth_frames):
    """Official per-frame scale alignment, pixel-weighted aggregate metrics."""
    from .depth import DepthEvaluator
    if prediction.ndim != 5 or prediction.shape[0] != 1 or prediction.shape[-1] != 1:
        raise ValueError('Expected predicted depth [1,S,H,W,1]')
    evaluator = DepthEvaluator(align='scale_only')
    sums, count, scales = {}, 0, []
    square_root_metrics = {'rmse', 'log_rmse'}
    for tensor, truth in zip(prediction[0], truth_frames, strict=True):
        predicted = tensor[..., 0].float().numpy()
        if predicted.shape != truth.shape:
            raise ValueError('GT and prediction pixel grids differ')
        valid = np.isfinite(truth) & (truth > 0) & (truth < 80)
        if valid.sum() < 100:
            raise ValueError('GT frame has fewer than 100 valid depth pixels')
        if not np.isfinite(predicted[valid]).all() or not (predicted[valid] > 0).all():
            raise ValueError('Invalid predicted depth on the common GT-valid mask')
        # Enforcing positivity above makes the official evaluator use the same
        # GT-defined pixels for every method, rather than hiding bad outputs.
        frame, aligned = evaluator.evaluate_single_frame(predicted, truth)
        metrics = evaluator._compute_metrics(aligned[frame['valid_mask']], truth[frame['valid_mask']])
        pixels = int(valid.sum())
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.) + pixels * (value ** 2 if name in square_root_metrics else value)
        count += pixels
        scales.append(float(np.median(truth[valid]) / np.median(predicted[valid])))
    if not count:
        raise ValueError('No GT depth frames')
    return {'metrics': {name: (value / count) ** .5 if name in square_root_metrics else value / count
                        for name, value in sums.items()},
            'frames': len(scales), 'valid_pixels': count, 'alignment_scales': scales,
            'alignment': 'official per-frame median(GT) / median(prediction), not absolute metric scale',
            'mask': 'common finite GT pixels in (0,80)m; invalid predictions cause an error, no confidence filtering',
            'reference_kind': 'ground_truth'}
