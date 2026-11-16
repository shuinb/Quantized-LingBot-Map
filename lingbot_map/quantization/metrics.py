import math
import torch


class ErrorMetrics:
    def __init__(self):
        self.count = 0
        self.reference_energy = 0.0
        self.output_energy = 0.0
        self.error_energy = 0.0
        self.dot = 0.0

    def update(self, reference, output):
        if reference.shape != output.shape:
            raise ValueError("Reference and output shapes differ")
        reference = reference.detach().double()
        output = output.detach().double()
        if not torch.isfinite(reference).all() or not torch.isfinite(output).all():
            raise ValueError("Non-finite model output")
        self.count += reference.numel()
        self.reference_energy += reference.square().sum().item()
        self.output_energy += output.square().sum().item()
        self.error_energy += (reference - output).square().sum().item()
        self.dot += (reference * output).sum().item()

    def summary(self):
        if not self.count:
            return {"elements": 0, "mse": None, "cosine": None, "sqnr_db": None}
        denominator = math.sqrt(self.reference_energy * self.output_energy)
        cosine = self.dot / denominator if denominator else float(self.error_energy == 0)
        sqnr = 10 * math.log10(self.reference_energy / self.error_energy) if self.error_energy and self.reference_energy else None
        return {
            "elements": self.count, "mse": self.error_energy / self.count,
            "cosine": max(-1.0, min(1.0, cosine)), "sqnr_db": sqnr,
            "exact_match": self.error_energy == 0,
        }
