"""Exercise archive preparation and frame selection through the GT reader."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from qmap.prepare_tum import prepare
from qmap.evaluation.quality import load_pose_ground_truth, iter_tum_depth


class PreparedDataTests(unittest.TestCase):
    def test_original_pixels_and_selected_pose_order_survive_archive_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive_path, destination = folder/'tum.tgz', folder/'prepared'
            rgb_bytes = []
            with tarfile.open(archive_path, 'w:gz') as archive:
                def add(name, content):
                    info = tarfile.TarInfo('sequence/' + name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
                for frame in range(4):
                    rgb = io.BytesIO()
                    Image.fromarray(np.full((28, 28, 3), frame*20, dtype=np.uint8)).save(rgb, format='PNG')
                    depth = io.BytesIO()
                    Image.fromarray(np.full((28, 28), 5000*(frame+1), dtype=np.uint16)).save(depth, format='PNG')
                    rgb_bytes.append(rgb.getvalue())
                    add(f'rgb/{frame+1:.6f}.png', rgb.getvalue())
                    add(f'depth/{frame+1:.6f}.png', depth.getvalue())
                add('groundtruth.txt', ''.join(f'{i+1} {i} 0 0 0 0 0 1\n' for i in range(4)).encode())
            manifest = prepare(archive_path, destination)
            self.assertEqual(manifest['frames'], 4)
            self.assertEqual(manifest['gt_start_frame'], 0)
            files = sorted((destination/'images').glob('*.png'))
            self.assertEqual([p.read_bytes() for p in files], rgb_bytes)
            selected = [str(p.resolve()) for p in files[1:3]]
            truth = load_pose_ground_truth(destination/'evaluation_gt.npz', destination/'manifest.json', selected, 1)
            torch.testing.assert_close(truth[0, :, 0, 3], torch.tensor([1., 2.], dtype=torch.float64))
            depths = list(iter_tum_depth(destination/'manifest.json', selected, 1, (28, 28)))
            np.testing.assert_array_equal(depths[0], np.full((28, 28), 2.))
            np.testing.assert_array_equal(depths[1], np.full((28, 28), 3.))
            with self.assertRaises(FileExistsError):
                prepare(archive_path, destination)


if __name__ == '__main__':
    unittest.main()
