"""Prepare an official TUM RGB-D archive without changing source RGB/depth pixels."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation


def prepare(archive_path, output, *, frames=0, stride=1, start=0):
    if frames < 0 or stride < 1 or start < 0:
        raise ValueError('frames/start must be nonnegative and stride must be positive')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        rgb_names = sorted(name for name in members if '/rgb/' in name and name.endswith('.png'))
        depth_names = sorted(name for name in members if '/depth/' in name and name.endswith('.png'))
        if not rgb_names or not depth_names:
            raise ValueError('Archive must contain rgb/ and depth/ images')
        depth_times = np.array([float(Path(name).stem) for name in depth_names])
        gt_name = next(name for name in members if name.endswith('/groundtruth.txt'))
        with archive.extractfile(members[gt_name]) as handle:
            text = handle.read().decode()
        rows = np.array([[float(value) for value in line.split()] for line in text.splitlines()
                         if line.strip() and not line.lstrip().startswith('#')])
        pairs = []
        for name in rgb_names[start::stride]:
            timestamp = float(Path(name).stem)
            pose_index = np.abs(rows[:, 0] - timestamp).argmin()
            depth_index = np.abs(depth_times - timestamp).argmin()
            if abs(rows[pose_index, 0] - timestamp) > .02 or abs(depth_times[depth_index] - timestamp) > .02:
                continue
            pairs.append({'rgb': name, 'depth': depth_names[depth_index], 'pose_row': rows[pose_index].tolist(),
                          'rgb_timestamp': timestamp, 'depth_timestamp': float(depth_times[depth_index])})
            if frames and len(pairs) == frames:
                break
    if not pairs or (frames and len(pairs) != frames):
        raise ValueError('Not enough associated RGB/depth/pose frames')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('images', 'depth'):
        (output / folder).mkdir()
    selected = {pair['rgb']: output / 'images' / Path(pair['rgb']).name for pair in pairs}
    selected.update({pair['depth']: output / 'depth' / Path(pair['depth']).name for pair in pairs})
    # Copy selected files to explicit destinations, without extracting archive paths.
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            if member.name in selected and member.isfile():
                with archive.extractfile(member) as source, selected[member.name].open('wb') as target:
                    shutil.copyfileobj(source, target)
    poses = []
    for pair in pairs:
        row = pair['pose_row']
        pose = np.zeros((3, 4), dtype=np.float64)
        pose[:, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        pose[:, 3] = row[1:4]
        poses.append(pose)
    np.savez_compressed(output / 'evaluation_gt.npz', c2w=np.array(poses)[None])
    manifest = {'source': Path(archive_path).stem, 'frames': len(pairs), 'gt_start_frame': 0,
                'stride': stride, 'start': start, 'pairs': pairs,
                'association_tolerance_seconds': .02, 'depth_units_per_meter': 5000,
                'protocol': 'Original images; nearest RGB/depth/pose association. Calibration uses separate sequences.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=0, help='0 keeps all associated frames')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args(argv)
    manifest = prepare(args.archive, args.output, frames=args.frames, stride=args.stride, start=args.start)
    print(f'Prepared {manifest["frames"]} frames in {args.output}')


if __name__ == '__main__':
    main()
