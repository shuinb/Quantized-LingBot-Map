import copy
from pathlib import Path
import tempfile
import unittest

import torch

from lingbot_map.layers.block import Block
from lingbot_map.quantization import QuantizationConfig, Quantizer, fake_quantize
from lingbot_map.quantization.native_int4 import PackedInt4Linear
from lingbot_map.quantization.reconstruction import BlockReconstruction


class ClippingTests(unittest.TestCase):
    def test_official_activation_and_weight_clipping_parity(self):
        from tests import reference_quantvggt as reference
        model = torch.nn.Sequential(torch.nn.Linear(32, 16))
        Quantizer(model, QuantizationConfig(include=('0',), lwc=True, lac=True))
        layer = model[0]
        with torch.no_grad():
            layer.clip_factor_a_min.fill_(-1)
            layer.clip_factor_w_max.fill_(-0.5)
        for shape in ((4, 32), (2, 3, 32)):
            inputs = torch.randn(shape)
            activation = reference.ActivationQuantizer(4, sym=True, lac=True)
            activation.clip_factor_a_min.data.copy_(layer.clip_factor_a_min)
            activation.clip_factor_a_max.data.copy_(layer.clip_factor_a_max)
            torch.testing.assert_close(layer.quantized_activation(inputs), activation(inputs), rtol=1e-6, atol=1e-6)
            inputs.requires_grad_(True)
            layer.clip_factor_a_max.requires_grad_(True)
            layer.clip_factor_a_min.requires_grad_(True)
            expected_gradients = torch.autograd.grad(activation(inputs).sum(),
                                                     (inputs, activation.clip_factor_a_max, activation.clip_factor_a_min))
            actual_gradients = torch.autograd.grad(layer.quantized_activation(inputs, ste=True).sum(),
                                                   (inputs, layer.clip_factor_a_max, layer.clip_factor_a_min))
            for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)
        weight = layer.weight
        clipped = weight.clamp(min=weight.amin(1, keepdim=True) * layer.clip_factor_w_min.sigmoid(),
                               max=weight.amax(1, keepdim=True) * layer.clip_factor_w_max.sigmoid())
        quantizer = reference.WeightQuantizer()
        quantizer.configure(4, perchannel=True, sym=True, mse=False)
        quantizer.find_params(clipped)
        torch.testing.assert_close(layer.quantized_weight(), quantizer(clipped), rtol=1e-6, atol=1e-6)

    def test_save_reload_invalidation_bypass_and_native_packing(self):
        original = torch.nn.Sequential(torch.nn.Linear(32, 16))
        model = copy.deepcopy(original)
        quantizer = Quantizer(model, QuantizationConfig(include=('0',), lwc=True, lac=True))
        inputs = torch.randn(2, 3, 32)
        expected = model(inputs)
        model[0].cache_weight = False
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        model[0].cache_weight = True
        before = model[0].prepared_weight.clone()
        with torch.no_grad():
            model[0].clip_factor_w_max.fill_(-2)
        model(inputs)
        self.assertFalse(torch.equal(before, model[0].prepared_weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'quant.pt'
            quantizer.save_parameters(path)
            restored = copy.deepcopy(original)
            Quantizer.from_parameters(restored, path)
            torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
        quantizer.set_enabled(False)
        torch.testing.assert_close(model(inputs), original(inputs), rtol=0, atol=0)
        packed = PackedInt4Linear.from_fake(model[0])
        self.assertEqual(packed.activation_clip.shape, (2,))
        self.assertEqual(packed.packed_weight.dtype, torch.uint8)
        self.assertTrue(torch.isfinite(expected).all())

    def test_clipping_only_reconstruction_updates_parameters_and_restores(self):
        torch.manual_seed(7)
        block = Block(dim=32, num_heads=4).eval()
        model = torch.nn.Sequential(block)
        quantizer = Quantizer(model, QuantizationConfig(include=('0.attn.qkv', '0.mlp.fc1'), lwc=True, lac=True))
        reconstruction = BlockReconstruction(quantizer, max_samples=2)
        inputs = torch.randn(1, 5, 32)
        original_weights = {name: layer.weight.clone() for name, layer in quantizer.layers.items()}
        with torch.no_grad(), quantizer.calibration(), reconstruction.capture():
            model(inputs)
            model(inputs * 3)
        result = reconstruction.optimize(epochs=3)
        self.assertEqual(len(result['0']['epoch_mse']), 3)
        self.assertEqual(result['0']['clipping_lr'], .01)
        for name, layer in quantizer.layers.items():
            self.assertTrue(any(not torch.equal(parameter, torch.full_like(parameter, 4.)) for parameter in layer.clipping_parameters()))
            torch.testing.assert_close(layer.weight, original_weights[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in layer.clipping_parameters()))
            layer.to(torch.bfloat16)
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in layer.clipping_parameters()))


if __name__ == '__main__':
    unittest.main()
