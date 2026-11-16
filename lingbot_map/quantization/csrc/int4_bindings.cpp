#include <torch/extension.h>

at::Tensor int4_gemm_cuda(const at::Tensor& activation, const at::Tensor& weight);
at::Tensor int4_linear_cuda(const at::Tensor& inputs, const at::Tensor& weight,
                           const at::Tensor& weight_scale, const c10::optional<at::Tensor>& bias,
                           const c10::optional<at::Tensor>& smoothing,
                           const c10::optional<at::Tensor>& signs, int64_t output_type,
                           const c10::optional<at::Tensor>& activation_clip, bool fast_rotation, bool fused_epilogue);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemm", &int4_gemm_cuda);
  module.def("linear", &int4_linear_cuda, pybind11::arg("inputs"), pybind11::arg("weight"),
             pybind11::arg("weight_scale"), pybind11::arg("bias"), pybind11::arg("smoothing"),
             pybind11::arg("signs"), pybind11::arg("output_type"), pybind11::arg("activation_clip") = pybind11::none(),
             pybind11::arg("fast_rotation") = false, pybind11::arg("fused_epilogue") = false);
}
