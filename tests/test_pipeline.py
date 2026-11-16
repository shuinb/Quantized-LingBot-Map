"""CPU integration coverage for the quality runner, calibration, and GT adapter.

Small random models/fixtures check execution semantics, not deployment accuracy.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch

from lingbot_map.aggregator.stream import AggregatorStream
from lingbot_map.models.gct_stream_window import GCTStream
from lingbot_map.quantization import QuantizationConfig, Quantizer
from lingbot_map.quantization.deployment import export_deployment, load_deployment
from lingbot_map.quantization.lazy_images import LazyImageSequence
from lingbot_map.quantization.online import iter_predictions
from lingbot_map.quantization.prefill_memory import chunked_prefill_mlp
from lingbot_map.quantization.native_int4 import PackedInt4Linear
from lingbot_map.quantization.quality import pack_quality_model
from qmap.calibrate import calibrate, calibration_cache_placement
from qmap.evaluation.quality import iter_tum_depth, depth_ground_truth_metrics
from qmap.infer import quality_cache, run_windows


class TinyQualityModel(GCTStream):
    def _build_aggregator(self):
        return AggregatorStream(img_size=28, patch_size=14, embed_dim=32, depth=24,
                                num_heads=4, patch_embed='conv', use_sdpa=True,
                                kv_cache_sliding_window=2, kv_cache_scale_frames=1,
                                use_gradient_checkpoint=False)


def tiny_model(_=None):
    return TinyQualityModel(img_size=28, embed_dim=32, patch_embed='conv', use_sdpa=True,
                            kv_cache_sliding_window=2, kv_cache_scale_frames=1,
                            num_frame_for_scale=1, camera_num_iterations=1,
                            enable_depth=False, use_gradient_checkpoint=False).eval().requires_grad_(False)


SMALL_CONFIG = dict(num_scale_frames=1, keyframe_interval=2, kv_cache_sliding_window=2,
                    kv_cache_scale_frames=1, camera_num_iterations=1, max_frame_num=64,
                    window_size=4, overlap_keyframes=1)


class QualityPipelineTests(unittest.TestCase):
    def test_prefill_wrapper_reuses_compiled_graph_across_all_blocks(self):
        model = tiny_model()
        compilations = []
        def backend(graph, inputs):
            compilations.append(graph)
            return graph.forward
        torch._dynamo.reset()
        inputs = torch.rand(1, 10, 32)
        with torch.no_grad(), chunked_prefill_mlp(model, 10):
            for block in (*model.aggregator.frame_blocks, *model.aggregator.global_blocks):
                expected = type(block.mlp).forward(block.mlp, inputs)
                actual = torch.compile(block.mlp.forward, backend=backend, fullgraph=True)(inputs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(len(compilations), 1)
        torch._dynamo.reset()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_chunked_native_mlp_preserves_joint_prefill_and_stream_outputs(self):
        if torch.cuda.get_device_capability()[0] != 8:
            self.skipTest('Native S4 kernel requires SM8x')
        torch.manual_seed(29)
        model = tiny_model()
        quantizer = Quantizer(model, QuantizationConfig(mode='rotation', lwc=True, lac=True))
        pack_quality_model(quantizer)
        model.aggregator.to(dtype=torch.bfloat16)
        model.cuda()
        for layer in model.modules():
            if isinstance(layer, PackedInt4Linear):
                layer.use_cpp_activation = True
                layer.use_fast_rotation = True
                layer.use_fused_epilogue = True
        images = torch.rand(1, 8, 3, 28, 28)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            expected = list(iter_predictions(model, images, 3, 2))
            with chunked_prefill_mlp(model, 10, frame_batch_size=1):
                actual = list(iter_predictions(model, images, 3, 2))
        self.assertEqual([row[0] for row in expected], [0, 3, 4, 5, 6, 7])
        for (left, prediction), (right, reference) in zip(actual, expected):
            self.assertEqual(left, right)
            for key in prediction:
                torch.testing.assert_close(prediction[key], reference[key], atol=0, rtol=0)
        self.assertTrue(all('forward' not in block.mlp.__dict__ for block in
                            (*model.aggregator.frame_blocks, *model.aggregator.global_blocks)))
        self.assertTrue(all('forward' not in block.__dict__ for block in model.aggregator.frame_blocks))

    def test_calibrated_direct_w4a4_learns_clipping_without_rotation_or_smoothing(self):
        torch.manual_seed(19)
        model = tiny_model()
        quantizer = Quantizer(model, QuantizationConfig(mode='naive', lwc=True, lac=True))
        weights = {name: layer.weight.detach().clone() for name, layer in quantizer.layers.items()}
        before = [p.detach().clone() for layer in quantizer.layers.values() for p in layer.clipping_parameters()]
        settings = dict(reconstruction_epochs=1, samples_per_clip=3, reconstruction_lr=.001,
                        image_size=28, max_height=28)
        clips = [{'id': f'clip_{i}', 'files': [f'{i}/{j}.png' for j in range(8)]} for i in range(2)]
        images = [torch.rand(1, 8, 3, 28, 28) for _ in clips]
        with patch('qmap.calibrate.LazyImageSequence', side_effect=images):
            result = calibrate(quantizer, clips, SMALL_CONFIG, settings, torch.device('cpu'))
        self.assertEqual(len(result['reconstruction']), 48)
        self.assertTrue(all(row['samples'] == 6 and np.isfinite(row['epoch_mse']).all()
                            for row in result['reconstruction'].values()))
        after = [p.detach() for layer in quantizer.layers.values() for p in layer.clipping_parameters()]
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
        for name, layer in quantizer.layers.items():
            torch.testing.assert_close(layer.weight, weights[name], rtol=0, atol=0)
            self.assertIsNone(layer.rotation)
            self.assertFalse(layer.config.smoothing)
        pack_quality_model(quantizer)
        for name in quantizer.layers:
            layer = model.get_submodule(name)
            self.assertIsNone(layer.signs)
            self.assertIsNone(layer.smoothing)
            self.assertEqual(layer.activation_clip.shape, (2,))
            self.assertTrue(torch.isfinite(layer.activation_clip).all())

    def test_observation_offload_hooks_are_removed_before_reconstruction_or_error(self):
        model = SimpleNamespace(aggregator=SimpleNamespace(global_blocks=[SimpleNamespace(attn=object())]))
        placement = SimpleNamespace(handles=[Mock(), Mock()], active={'live_cache': object()})
        with patch('qmap.calibrate.CPUCachePlacement', return_value=placement):
            with self.assertRaisesRegex(RuntimeError, 'observation failed'):
                with calibration_cache_placement(model, torch.device('cuda:0'), offload=True):
                    raise RuntimeError('observation failed')
        for handle in placement.handles:
            handle.remove.assert_called_once_with()
        self.assertEqual(placement.active, {})
        with self.assertRaisesRegex(ValueError, 'CUDA model'):
            with calibration_cache_placement(model, torch.device('cpu'), offload=True):
                pass

    def test_full_state_reconstruction_export_and_streaming_execution(self):
        torch.manual_seed(11)
        model = tiny_model()
        quantizer = Quantizer(model, QuantizationConfig(mode='dsfq'))
        original_weights = {name: layer.weight.detach().clone() for name, layer in quantizer.layers.items()}
        settings = dict(reconstruction_epochs=1, samples_per_clip=3, reconstruction_lr=.001,
                        image_size=28, max_height=28)
        clips = [{'id': f'clip_{clip}', 'files': [f'{clip}/{i}.png' for i in range(8)]} for clip in range(2)]
        images = [torch.rand(1, 8, 3, 28, 28) for _ in clips]
        with patch('qmap.calibrate.LazyImageSequence', side_effect=images):
            result = calibrate(quantizer, clips, SMALL_CONFIG, settings, torch.device('cpu'))
        self.assertEqual(len(result['reconstruction']), 48)
        for row in result['coverage']:
            self.assertEqual(row['observed_phases'], ['anchor', 'local', 'memory'])
            self.assertEqual(set(row['samples_per_block'].values()), {3})
        for row in result['reconstruction'].values():
            self.assertEqual(row['samples'], 6)
            self.assertTrue(np.isfinite(row['epoch_mse']).all())
        for name, layer in quantizer.layers.items():
            torch.testing.assert_close(layer.weight, original_weights[name], rtol=0, atol=0)
            self.assertTrue(layer.smoothing_initialized)
        self.assertEqual(model.aggregator.total_frames_processed, 0)
        self.assertIsNone(model.camera_head.kv_cache)
        pack_quality_model(quantizer)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'model.pt'
            export_deployment(model, {}, artifact)
            loaded, _ = load_deployment(artifact, model_factory=tiny_model, linear_backend='int8')
            with torch.no_grad():
                output = list(iter_predictions(loaded, images[0][:, :4], 1, 2))
            self.assertEqual([index for index, _ in output], [0, 1, 2, 3])
            self.assertTrue(all(torch.isfinite(row['pose_enc']).all() for _, row in output))
            self.assertEqual([row['is_keyframe'].item() for _, row in output], [True, True, False, True])

    def test_real_image_runner_includes_overlap_once_in_output_and_fps(self):
        torch.manual_seed(12)
        model = tiny_model()
        quality = {**SMALL_CONFIG, 'keyframe_interval': 1}
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for index in range(7):
                path = Path(directory) / f'{index:03}.png'
                image = np.zeros((294, 518, 3), dtype=np.uint8)
                image[..., 0] = index * 25
                Image.fromarray(image).save(path)
                files.append(path)
            with torch.no_grad():
                prediction, calls, timing = run_windows(model, files, quality, prefetch=True)
        self.assertEqual(prediction['pose_enc'].shape[1], 7)
        self.assertEqual(prediction['is_keyframe'].shape[1], 7)
        self.assertTrue(torch.isfinite(prediction['pose_enc']).all())
        self.assertEqual(timing['processed_frames_including_overlap'], 8)
        self.assertEqual(sum(row['frames_in_call'] for row in calls), 8)
        self.assertAlmostEqual(timing['unique_fps'] * timing['seconds'], 7)
        self.assertEqual(prediction['chunk_scales'].shape, (1, 2))
        self.assertEqual(model.aggregator.total_frames_processed, 0)

    def test_compressed_runner_preserves_all_frames_and_releases_window_history(self):
        torch.manual_seed(20)
        model = tiny_model()
        images = torch.rand(1, 7, 3, 294, 518)
        quality = {**SMALL_CONFIG, 'keyframe_interval': 1}
        with patch('qmap.infer.LazyImageSequence',
                   side_effect=lambda files, *args, **kwargs: images[:, files[0]:files[-1] + 1]):
            with torch.no_grad(), quality_cache(model, quality, 'fp8') as cache:
                prediction, calls, timing = run_windows(model, list(range(7)), quality,
                                                       prefetch=False, cache=cache)
                self.assertEqual(cache.states, {})
                self.assertIsNone(cache.scratch)
        self.assertEqual(prediction['pose_enc'].shape[1], 7)
        self.assertTrue(torch.isfinite(prediction['pose_enc']).all())
        self.assertEqual(sum(row['frames_in_call'] for row in calls), 8)
        self.assertEqual(timing['processed_frames_including_overlap'], 8)
        self.assertAlmostEqual(timing['unique_fps'] * timing['seconds'], 7)
        memory = timing['kv_cache_largest_window']
        self.assertEqual(memory['layers'], 24)
        self.assertEqual(memory['full_frames'], [3])
        self.assertEqual(memory['special_tokens'], [model.aggregator.patch_start_idx])
        self.assertGreater(memory['storage_bytes'], 0)
        self.assertTrue(memory['fuse_sdpa_decode'])
        self.assertTrue(memory['deduplicate_special_scales'])
        self.assertEqual(model.aggregator.total_frames_processed, 0)
        self.assertTrue(all(not hasattr(block.attn, '_compact_cache') for block in model.aggregator.global_blocks))

    def test_raw_depth_resize_crop_units_and_same_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'depth').mkdir()
            rgb = root / 'images' / '000.png'
            Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(rgb)
            raw = (10000 + np.indices((48, 64))[0] * 50).astype(np.uint16)
            raw[10:14, 30:36] = 0
            cv2.imwrite(str(root / 'depth' / '001.png'), raw)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'pairs': [{'rgb': 'scene/rgb/000.png', 'depth': 'scene/depth/001.png'}],
                                            'depth_units_per_meter': 5000}))
            gt = list(iter_tum_depth(manifest, [str(rgb.resolve())], 0, (14, 28)))
            expected = cv2.resize(raw, (28, 28), interpolation=cv2.INTER_NEAREST).astype(np.float32)[7:21] / 5000
            np.testing.assert_array_equal(gt[0], expected)
            images = LazyImageSequence([rgb], image_size=28, max_height=14)
            self.assertEqual(tuple(images.shape[-2:]), gt[0].shape)
            prediction = torch.from_numpy(gt[0] * 2)[None, None, ..., None]
            result = depth_ground_truth_metrics(prediction, iter(gt))
            self.assertEqual(result['valid_pixels'], int((expected > 0).sum()))
            self.assertEqual(result['metrics']['abs_rel'], 0)
            self.assertEqual(result['alignment_scales'], [.5])
            prediction[0, 0, 0, 0, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'common GT-valid'):
                depth_ground_truth_metrics(prediction, iter(gt))




if __name__ == '__main__':
    unittest.main()
