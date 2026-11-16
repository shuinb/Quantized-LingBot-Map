import torch

from lingbot_map.quantization.metrics import ErrorMetrics


def rotation_error_degrees(reference, output):
    difference = reference.transpose(-1, -2) @ output
    cosine = ((difference.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def trajectory_errors(reference, output):
    reference, output = reference.double(), output.double()
    if reference.shape != output.shape or reference.shape[-2:] != (3, 4):
        raise ValueError('Trajectory must be matching [..., S, 3, 4] C2W matrices')
    result = {'translation_rmse_unaligned': (reference[..., :3, 3] - output[..., :3, 3]).square().sum(-1).mean().sqrt().item(),
              'rotation_deg_mean': rotation_error_degrees(reference[..., :3], output[..., :3]).mean().item(),
              'alignment': 'none; same coordinates and scale required'}
    if reference.shape[-3] > 1:
        ref_rotation, out_rotation = reference[..., :3], output[..., :3]
        ref_relative_rotation = ref_rotation[..., :-1, :, :].transpose(-1, -2) @ ref_rotation[..., 1:, :, :]
        out_relative_rotation = out_rotation[..., :-1, :, :].transpose(-1, -2) @ out_rotation[..., 1:, :, :]
        ref_relative_translation = (ref_rotation[..., :-1, :, :].transpose(-1, -2) @ (reference[..., 1:, :3, 3] - reference[..., :-1, :3, 3]).unsqueeze(-1)).squeeze(-1)
        out_relative_translation = (out_rotation[..., :-1, :, :].transpose(-1, -2) @ (output[..., 1:, :3, 3] - output[..., :-1, :3, 3]).unsqueeze(-1)).squeeze(-1)
        result['relative_translation_rmse_delta1'] = (ref_relative_translation - out_relative_translation).square().sum(-1).mean().sqrt().item()
        result['relative_rotation_deg_mean_delta1'] = rotation_error_degrees(ref_relative_rotation, out_relative_rotation).mean().item()
    return result


def depth_errors(reference, output):
    if reference.shape != output.shape:
        raise ValueError('Depth shapes differ; GT must match preprocessed pixels')
    reference, output = reference.float(), output.float()
    valid = torch.isfinite(reference) & (reference > 1e-8)
    if not valid.any():
        raise ValueError('No valid positive reference depths')
    if not torch.isfinite(output[valid]).all():
        raise ValueError('Non-finite predicted depth on valid pixels')
    difference = output[valid] - reference[valid]
    return {'valid_pixels': valid.sum().item(), 'rmse': difference.square().mean().sqrt().item(),
            'abs_rel': (difference.abs() / reference[valid]).mean().item(), 'alignment': 'none'}


def decoded_geometry(predictions, image_hw):
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'].float(), image_hw)
    result = {'c2w': extrinsics.cpu()}
    if 'depth' in predictions:
        result['depth'] = predictions['depth'].float().cpu()
    if 'world_points' in predictions:
        result['points'] = predictions['world_points'].float().cpu()
        result['point_source'] = 'point_head'
    elif 'depth' in predictions:
        points = []
        for depth, camera, intrinsic in zip(result['depth'], extrinsics.cpu(), intrinsics.cpu()):
            world_to_camera = closed_form_inverse_se3(camera.numpy())
            points.append(torch.from_numpy(unproject_depth_map_to_point_map(depth.numpy(), world_to_camera[:, :3], intrinsic.numpy())))
        result['points'] = torch.stack(points).float()
        result['point_source'] = 'depth_unprojection_using_official_geometry_and_inverted_C2W'
    return result


def compare_geometry(reference, output, reference_kind='floating_prediction_not_ground_truth'):
    result = {'reference_kind': reference_kind}
    if 'c2w' in reference and 'c2w' in output:
        result['camera'] = trajectory_errors(reference['c2w'], output['c2w'])
    if 'depth' in reference and 'depth' in output:
        result['depth'] = depth_errors(reference['depth'], output['depth'])
    if 'points' in reference and 'points' in output:
        expected, predicted = reference['points'].float(), output['points'].float()
        if expected.shape != predicted.shape:
            raise ValueError('Point map shapes differ')
        valid = torch.isfinite(expected).all(-1)
        if not valid.any() or not torch.isfinite(predicted[valid]).all():
            raise ValueError('Invalid predicted/reference points')
        metrics = ErrorMetrics()
        metrics.update(expected[valid], predicted[valid])
        result['points'] = {**metrics.summary(), 'mean_euclidean': (expected[valid] - predicted[valid]).norm(dim=-1).mean().item(),
                            'source': output.get('point_source', 'provided')}
    return result


def official_geometry_metrics(reference, output):
    import numpy as np
    from .trajectory import TrajectoryEvaluator
    from .depth import DepthEvaluator

    for key in ('c2w', 'depth', 'points'):
        if key in reference and key in output and reference[key].shape != output[key].shape:
            raise ValueError(f'GT {key} shape {tuple(reference[key].shape)} differs from prediction {tuple(output[key].shape)}; align frame lists and preprocessed pixels explicitly')

    class TrajectoryArray:
        def __init__(self, poses):
            if poses.shape[0] != 1:
                raise ValueError('Official trajectory adapter expects batch size 1')
            self.poses = np.broadcast_to(np.eye(4), (poses.shape[1], 4, 4)).copy()
            self.poses[:, :3] = poses[0].double().numpy()

        def load_trajectory(self):
            return self.poses

        def get_frame_indices(self):
            return np.arange(len(self.poses))

    result = {'reference_kind': 'ground_truth', 'trajectory_alignment': 'official TrajectoryEvaluator Sim3',
              'depth_alignment': 'official DepthEvaluator per-frame scale_only, GT clip (0,80)',
              'point_protocol': 'corresponding-pixel EPE after trajectory Sim3, not ICP/Chamfer reconstruction benchmark'}
    transform = None
    if 'c2w' in reference:
        trajectory = TrajectoryEvaluator(align=True, correct_scale=True).evaluate(TrajectoryArray(reference['c2w']), TrajectoryArray(output['c2w']))
        transform = trajectory.pop('traj_transform')
        result['camera'] = trajectory
        result['trajectory_transform'] = transform.tolist()
    if 'depth' in reference and 'depth' in output:
        evaluator = DepthEvaluator(align='scale_only')
        reference_pixels, predicted_pixels = [], []
        for expected, predicted in zip(reference['depth'][0], output['depth'][0]):
            frame, aligned = evaluator.evaluate_single_frame(predicted[..., 0].numpy(), expected[..., 0].numpy())
            if frame is None:
                raise ValueError('GT frame has fewer than 100 valid pixels under official depth protocol')
            reference_pixels.append(expected[..., 0].numpy()[frame['valid_mask']])
            predicted_pixels.append(aligned[frame['valid_mask']])
        result['depth'] = {key: float(value) for key, value in evaluator._compute_metrics(np.concatenate(predicted_pixels), np.concatenate(reference_pixels)).items()}
    if transform is not None and 'points' in reference and 'points' in output:
        expected = reference['points'].double()
        predicted = output['points'].double() @ torch.from_numpy(transform[:3, :3]).T + torch.from_numpy(transform[:3, 3])
        valid = torch.isfinite(expected).all(-1)
        if not valid.any() or not torch.isfinite(predicted[valid]).all():
            raise ValueError('No finite corresponding GT points or invalid predicted points')
        result['points'] = {'mean_euclidean_m': (predicted[valid] - expected[valid]).norm(dim=-1).mean().item(),
                            'rmse_euclidean_m': (predicted[valid] - expected[valid]).square().sum(-1).mean().sqrt().item(),
                            'valid_pixels': valid.sum().item()}
    return result
