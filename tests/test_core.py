import unittest
import torch
from torch import nn

from lingbot_map.quantization import (
    ErrorMetrics, QuantizationConfig, Quantizer, fake_quantize, int4_quantize, minmax_scale,
)
from lingbot_map.quantization.core import ChannelObserver


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_uncached_fake_weight_preserves_exact_output(self):
        for mode in ('naive', 'rotation', 'smoothing', 'dsfq'):
            model = nn.Sequential(nn.Linear(32, 64))
            quantizer = Quantizer(model, QuantizationConfig(mode=mode, include=('0',)))
            inputs = torch.randn(2, 3, 32)
            with torch.no_grad():
                expected = model(inputs)
                layer = quantizer.layers['0']
                layer.cache_weight = False
                layer.invalidate()
                actual = model(inputs)
            self.assertIsNone(layer.prepared_weight)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_int4_boundaries(self):
        values = torch.tensor([-99., -8., -7.5, 0., 6.5, 7., 99.])
        scale = torch.tensor(1.)
        self.assertEqual(int4_quantize(values, scale).tolist(), [-8, -8, -8, 0, 6, 7, 7])
        self.assertEqual(fake_quantize(values, scale).tolist(), [-8., -8., -8., 0., 6., 7., 7.])
        for invalid in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                int4_quantize(values, torch.tensor(invalid))

    def test_weight_and_token_scale(self):
        weight = torch.tensor([[0., 7.], [-14., 0.]])
        self.assertEqual(minmax_scale(weight).shape, (2, 1))
        torch.testing.assert_close(minmax_scale(weight), torch.tensor([[1.], [2.]]))
        activation = torch.randn(2, 3, 8)
        self.assertEqual(minmax_scale(activation).shape, (2, 3, 1))
        output = fake_quantize(activation)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output - activation).abs() <= minmax_scale(activation) / 2 + 1e-6).all())

    def test_zero_and_small_values(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = torch.zeros(2, 8, dtype=dtype)
            self.assertTrue((minmax_scale(values) > 0).all())
            torch.testing.assert_close(fake_quantize(values), values)
            self.assertEqual(fake_quantize(values).dtype, dtype)

    def test_observer_and_calibration(self):
        observer = ChannelObserver(4)
        tensor = torch.arange(24.).reshape(2, 3, 4)
        observer(tensor)
        torch.testing.assert_close(observer.ema, tensor.abs().amax(-2).mean(0))
        observer(tensor * 2)
        torch.testing.assert_close(observer.ema, tensor.abs().amax(-2).mean(0) * 1.05)
        model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        original = model(tensor)
        quantizer = Quantizer(model, QuantizationConfig(include=("0", "2")))
        with quantizer.calibration():
            torch.testing.assert_close(model(tensor), original, rtol=0, atol=0)
        self.assertEqual(quantizer.layers['0'].observer.calls.item(), 1)

    def test_switch_remove_and_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)).to(dtype)
            tensor = torch.randn(1, 3, 8).to(dtype)
            original = model(tensor)
            quantizer = Quantizer(model, QuantizationConfig(include=("0", "2")))
            quantizer.set_enabled(False)
            torch.testing.assert_close(model(tensor), original, rtol=0, atol=0)
            quantizer.set_enabled(True)
            quantizer.analyze()
            output = model(tensor)
            self.assertEqual(output.shape, original.shape)
            self.assertEqual(output.dtype, dtype)
            self.assertGreater(quantizer.report()['0']['local_error']['mse'], 0)
            quantizer.remove()
            torch.testing.assert_close(model(tensor), original, rtol=0, atol=0)

    def test_weight_change_invalidates(self):
        model = nn.Sequential(nn.Linear(4, 4))
        quantizer = Quantizer(model, QuantizationConfig(include=("0",)))
        model(torch.ones(1, 4))
        before = model[0].prepared_weight.clone()
        with torch.no_grad():
            model[0].weight.mul_(2)
        model(torch.ones(1, 4))
        torch.testing.assert_close(model[0].prepared_weight, before * 2)

    def test_metrics(self):
        reference = torch.tensor([1., 2., 3.])
        metrics = ErrorMetrics()
        metrics.update(reference, reference)
        self.assertEqual(metrics.summary()['mse'], 0.)
        self.assertAlmostEqual(metrics.summary()['cosine'], 1.)
        self.assertIsNone(metrics.summary()['sqnr_db'])
        metrics = ErrorMetrics()
        metrics.update(reference, reference + 1)
        self.assertAlmostEqual(metrics.summary()['mse'], 1.)

    def test_selection_errors(self):
        model = nn.Sequential(nn.Linear(8, 8))
        with self.assertRaises(ValueError):
            Quantizer(model)
        Quantizer(model, QuantizationConfig(include=("*",)))
        with self.assertRaises(ValueError):
            Quantizer(model, QuantizationConfig(include=("*",)))


if __name__ == '__main__':
    unittest.main()
