import copy
import unittest

import torch

from lingbot_map.layers.attention import SDPAAttention
from lingbot_map.quantization.compact_cache import CompactCache
from lingbot_map.quantization.compressed_cache import CompressedCache, compressed_streaming_cache
from lingbot_map.quantization.kv_codec import FORMATS, reference_decode, reference_encode
from lingbot_map.quantization.streaming import run_stream


class CompressedCacheTests(unittest.TestCase):
    def test_deferred_reservation_copies_history_exactly_including_nonkeyframe_growth(self):
        torch.manual_seed(130)
        attention = SDPAAttention(64, num_heads=2, kv_cache_sliding_window=2, kv_cache_scale_frames=2).eval()
        for format_name in FORMATS:
            expected = CompressedCache(format_name, reserve_frames=20, deduplicate_special_scales=True)
            actual = CompressedCache(format_name, reserve_frames=20, deduplicate_special_scales=True,
                                     defer_full_reservation=True)
            expected_cache, actual_cache = {'k_0': None, 'v_0': None}, {'k_0': None, 'v_0': None}
            with torch.no_grad():
                for step in range(10):
                    frames = 2 if step == 0 else 1
                    key = torch.randn(1, 2, frames, 137, 32)
                    value = torch.randn_like(key).bfloat16()
                    expected_cache['_skip_append'] = actual_cache['_skip_append'] = step in (1, 4, 7)
                    target = expected.prepare(attention, expected_cache, 0, key, value, 6)
                    result = actual.prepare(attention, actual_cache, 0, key, value, 6)
                    for left, right in zip(result, target):
                        torch.testing.assert_close(left, right, atol=0, rtol=0)
                    self.assertEqual(actual.states[0].order, expected.states[0].order)
                    self.assertEqual(actual.states[0].key.shape[2], 2 if step == 0 else 4)
                    self.assertEqual(actual_cache['k_0'].untyped_storage().data_ptr(),
                                     actual.states[0].key.codes.untyped_storage().data_ptr())
                    if step == 0:
                        self.assertLess(actual.memory_report()['storage_bytes'], expected.memory_report()['storage_bytes'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fused_sdpa_decode_exact_with_eviction_and_non_keyframes(self):
        torch.manual_seed(128)
        attention = SDPAAttention(64, num_heads=2, kv_cache_sliding_window=2, kv_cache_scale_frames=1).eval()
        for format_name in FORMATS:
            for dtype in (torch.bfloat16, torch.float16):
                old = CompressedCache(format_name, deduplicate_special_scales=True)
                new = CompressedCache(format_name, deduplicate_special_scales=True, fuse_sdpa_decode=True)
                old_cache, new_cache = {'k_0': None, 'v_0': None}, {'k_0': None, 'v_0': None}
                with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
                    for step in range(8):
                        old_cache['_skip_append'] = new_cache['_skip_append'] = step in (2, 5)
                        key = torch.randn(1, 2, 1, 137, 32, device='cuda')
                        value = torch.randn_like(key).to(dtype)
                        query = torch.randn(1, 2, 137, 32, device='cuda')
                        expected = old.attention(attention, old_cache, 0, query, key, value, 6)
                        actual = new.attention(attention, new_cache, 0, query, key, value, 6)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(new.scratch[0].dtype, dtype)
                    self.assertLess(new.memory_report()['shared_scratch_bytes'], old.memory_report()['shared_scratch_bytes'])

    def compare_cache(self, device):
        torch.manual_seed(108)
        for name in FORMATS:
            for camera_only, cross_special, keep_anchors in (
                    (False, True, True), (True, True, True), (False, False, True), (False, True, False)):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    with self.subTest(device=device, format=name, dtype=dtype,
                                      camera=camera_only, special=cross_special, anchors=keep_anchors):
                        attention = SDPAAttention(32, num_heads=4, kv_cache_sliding_window=3,
                                                 kv_cache_scale_frames=2, kv_cache_camera_only=camera_only,
                                                 kv_cache_cross_frame_special=cross_special,
                                                 kv_cache_include_scale_frames=keep_anchors).eval()
                        compressed, reference = CompressedCache(name, reserve_frames=20), CompactCache()
                        actual_cache, expected_cache = {'k_0': None, 'v_0': None}, {'k_0': None, 'v_0': None}
                        for step in range(15):
                            frames = 2 if step == 0 else 1
                            key = torch.randn(2, 2, frames, 9, 8, device=device)
                            value = torch.randn_like(key).to(dtype)
                            skip = step in (3, 8, 9)
                            actual_cache['_skip_append'] = expected_cache['_skip_append'] = skip
                            quantized_key = key if skip else reference_decode(reference_encode(key.cpu(), name, 128)).to(device)
                            quantized_value = value if skip else reference_decode(reference_encode(value.cpu(), name, 0)).to(device)
                            with torch.no_grad():
                                expected = reference.prepare(attention, expected_cache, 0, quantized_key, quantized_value, 6)
                                actual = compressed.prepare(attention, actual_cache, 0, key, value, 6)
                            for output, target in zip(actual, expected):
                                torch.testing.assert_close(output, target, atol=0, rtol=0)
                            self.assertEqual(actual_cache['k_0'].shape[2:4], expected_cache['k_0'].shape[2:4])
                            self.assertEqual(actual_cache['k_0'].dtype, torch.uint8)
                        report = compressed.memory_report()
                        self.assertTrue(report['compressed_storage'])
                        self.assertGreater(report['code_bytes'], 0)
                        self.assertGreater(report['scale_bytes'], 0)
                        compressed.reset()
                        self.assertEqual(compressed.memory_report()['storage_bytes'], 0)
                        self.assertIsNone(compressed.scratch)

    def test_cpu_exact_quantized_history_eviction_special_skip(self):
        self.compare_cache('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gpu_exact_quantized_history_eviction_special_skip(self):
        self.compare_cache('cuda')

    def test_persistent_history_never_requantized_and_skip_not_stored(self):
        attention = SDPAAttention(32, num_heads=4, kv_cache_sliding_window=2, kv_cache_scale_frames=1).eval()
        for name in FORMATS:
            manager = CompressedCache(name)
            cache = {'k_0': None, 'v_0': None}
            with torch.no_grad():
                for step in range(6):
                    values = torch.randn(1, 1, 1, 7, 8) * 10**step
                    previous = copy.deepcopy(manager.states.get(0))
                    cache['_skip_append'] = step == 4
                    manager.prepare(attention, cache, 0, values, values.bfloat16(), 6)
                    state = manager.states[0]
                    if previous is not None:
                        protected = previous.order if step == 4 else [previous.order[0]]
                        for slot in protected:
                            torch.testing.assert_close(state.key.codes[:, :, slot], previous.key.codes[:, :, slot], atol=0, rtol=0)
                            torch.testing.assert_close(state.key.scales[:, :, slot], previous.key.scales[:, :, slot], atol=0, rtol=0)
                        if step == 4:
                            self.assertEqual(state.order, previous.order)
                            self.assertEqual(state.special_count, previous.special_count)

    def test_growth_shared_workspace_and_skip_before_prefill(self):
        attention = SDPAAttention(32, num_heads=4, kv_cache_sliding_window=2, kv_cache_scale_frames=1).eval()
        for name in FORMATS:
            manager, reference = CompressedCache(name), CompactCache()
            cache = {'k_0': None, 'v_0': None, '_skip_append': True}
            reference_cache = {'k_0': None, 'v_0': None, '_skip_append': True}
            with torch.no_grad():
                for step in range(100):
                    values = torch.randn(1, 1, 1, 7, 8) * (1 + step)
                    cache['_skip_append'] = reference_cache['_skip_append'] = step == 0
                    quantized = values if step == 0 else reference_decode(reference_encode(values, name, 128))
                    quantized_value = values if step == 0 else reference_decode(reference_encode(values, name, 0))
                    expected = reference.prepare(attention, reference_cache, 0, quantized, quantized_value, 6)
                    actual = manager.prepare(attention, cache, 0, values, values, 6)
                    for result, target in zip(actual, expected):
                        torch.testing.assert_close(result, target, atol=0, rtol=0)
                self.assertGreater(manager.states[0].special_count, 512)
                self.assertGreater(manager.states[0].special_key.shape[3], 512)
                scratch_pointer = manager.scratch[0].data_ptr()
                previous_codes = manager.states[0].key.codes.clone()
                manager.prepare(attention, {'k_1': None, 'v_1': None}, 1, values, values, 6)
                self.assertEqual(manager.scratch[0].data_ptr(), scratch_pointer)
                torch.testing.assert_close(manager.states[0].key.codes, previous_codes, atol=0, rtol=0)
                self.assertEqual(manager.memory_report()['layers'], 2)

    def test_attention_output_matches_independent_qdq_history(self):
        torch.manual_seed(122)
        for name in FORMATS:
            actual = SDPAAttention(32, num_heads=4, qk_norm=True, kv_cache_sliding_window=2, kv_cache_scale_frames=1).eval()
            expected = copy.deepcopy(actual)
            actual._compact_cache = CompressedCache(name)
            expected._compact_cache = CompactCache()

            def quantize(key, value, cache, index):
                return reference_decode(reference_encode(key, name, 128)), reference_decode(reference_encode(value, name, 0))

            expected._kv_write_quantizer = quantize
            actual_cache, expected_cache = {'k_0': None, 'v_0': None}, {'k_0': None, 'v_0': None}
            with torch.no_grad():
                for step in range(8):
                    inputs = torch.randn(1, 9, 32)
                    actual_cache['_skip_append'] = expected_cache['_skip_append'] = step == 5
                    target = expected(inputs, kv_cache=expected_cache)
                    output = actual(inputs, kv_cache=actual_cache)
                    torch.testing.assert_close(output, target, atol=0, rtol=0)

    def test_inference_boundary_and_invalid_options(self):
        for arguments in (('bad',), ('fp4', 0), ('int4', None, 3)):
            with self.assertRaises(ValueError):
                CompressedCache(*arguments)
        attention = SDPAAttention(32, num_heads=4).eval()
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            CompressedCache('fp8').prepare(attention, {'k_0': None, 'v_0': None}, 0,
                                           torch.zeros(1, 1, 1, 7, 8), torch.zeros(1, 1, 1, 7, 8), 6)

    def test_full_stream_lifecycle_replay_and_restore(self):
        from tests.helpers import TinyGCT
        torch.manual_seed(121)
        model = TinyGCT(img_size=28, embed_dim=32, patch_embed='conv', use_sdpa=True, camera_num_iterations=1).eval()
        images = torch.rand(9, 3, 28, 28)
        original = run_stream(model, images, keyframe_interval=2)
        for name in FORMATS:
            with compressed_streaming_cache(model, name, reserve_frames=9) as manager:
                with torch.no_grad():
                    values = torch.randn(1, 4, 1, 7, 8)
                    manager.prepare(model.aggregator.global_blocks[0].attn, model.aggregator.kv_cache,
                                    0, values, values.bfloat16(), 6)
                info = model.get_kv_cache_info()
                self.assertEqual(info['num_cached_blocks'], 1)
                self.assertEqual(info['cache_storage_bytes'], manager.memory_report()['storage_bytes'])
                self.assertGreater(info['shared_scratch_bytes'], 0)
                model.clean_kv_cache()
                actual = run_stream(model, images, keyframe_interval=2)
                repeated = run_stream(model, images, keyframe_interval=2)
                self.assertFalse(manager.states)
                self.assertIsNone(manager.scratch)
                self.assertTrue(all(value is None or value is False for value in model.aggregator.kv_cache.values()))
                self.assertGreater(manager.updates, 0)
                for key in actual:
                    torch.testing.assert_close(actual[key], repeated[key], atol=0, rtol=0)
                    self.assertTrue(torch.isfinite(actual[key]).all())
            self.assertFalse(any(hasattr(block.attn, '_compact_cache') for block in model.aggregator.global_blocks))
            self.assertNotIn('clean_kv_cache', model.aggregator.__dict__)
        restored = run_stream(model, images, keyframe_interval=2)
        for key in original:
            torch.testing.assert_close(original[key], restored[key], atol=0, rtol=0)

    def test_workspace_shape_dtype_and_borrowed_lifetime(self):
        attention = SDPAAttention(32, num_heads=4, kv_cache_sliding_window=2, kv_cache_scale_frames=1).eval()
        for name in FORMATS:
            manager = CompressedCache(name)
            with torch.no_grad():
                first = torch.ones(1, 1, 1, 7, 8)
                borrowed = manager.prepare(attention, {'k_0': None, 'v_0': None}, 0, first, first.bfloat16(), 6)[0]
                copy_before_next = borrowed.clone()
                second = first * 2
                manager.prepare(attention, {'k_1': None, 'v_1': None}, 1, second, second.bfloat16(), 6)
                self.assertFalse(torch.equal(borrowed, copy_before_next))
                wider = torch.ones(1, 1, 1, 7, 16)
                output = manager.prepare(attention, {'k_2': None, 'v_2': None}, 2, wider, wider.half(), 6)
                self.assertEqual(output[0].shape[-1], 16)
                self.assertEqual(output[1].dtype, torch.float16)


if __name__ == '__main__':
    unittest.main()
