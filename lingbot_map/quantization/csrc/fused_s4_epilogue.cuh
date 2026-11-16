#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

namespace lingbot_s4_fused {
namespace evt = cutlass::epilogue::threadblock;
using TB = cutlass::gemm::GemmShape<64, 128, 256>;
using Warp = cutlass::gemm::GemmShape<32, 64, 256>;
using Map = evt::OutputTileThreadLayout<TB, Warp, float, 4, 1>;
using OutMap = Map;
using Col = evt::VisitorColBroadcast<Map, float>;
using Row = evt::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, cute::_0>>;
using Mul = evt::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using Add = evt::VisitorCompute<cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using ScaledRows = evt::Sm80EVT<Mul, evt::VisitorAccFetch, Col>;
using ScaledCols = evt::Sm80EVT<Mul, ScaledRows, Row>;
using Biased = evt::Sm80EVT<Add, ScaledCols, Row>;
using Store = evt::VisitorAuxStore<OutMap, cutlass::bfloat16_t,
    cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
using Output = evt::Sm80EVT<Store, Biased>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 32,
    cutlass::int4b_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 32,
    int32_t, cutlass::layout::RowMajor, 4, int32_t, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TB, Warp, cutlass::gemm::GemmShape<16, 8, 64>, Output,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
    cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;

__global__ void fused_kernel(Kernel::Params params) {
  extern __shared__ char storage[];
  Kernel::invoke(params, *reinterpret_cast<Kernel::SharedStorage*>(storage));
}

at::Tensor fused_scaled_s4(const at::Tensor& a, const at::Tensor& b,
                          const at::Tensor& sa, const at::Tensor& sb, const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && b.dim() == 2 && a.scalar_type() == at::kByte &&
              b.scalar_type() == at::kByte && a.is_contiguous() && b.is_contiguous(), "packed CUDA matrices required");
  const c10::cuda::CUDAGuard guard(a.device());
  const int m = a.size(0), n = b.size(0), k = a.size(1) * 2;
  TORCH_CHECK(m > 0 && n > 0 && b.size(1) == a.size(1) && k % 32 == 0 && n % 4 == 0, "invalid dimensions");
  TORCH_CHECK(b.device() == a.device(), "matrix devices differ");
  for (const auto& value : {sa, sb}) {
    TORCH_CHECK(value.device() == a.device() && value.scalar_type() == at::kFloat && value.is_contiguous(), "FP32 contiguous scales/bias required");
  }
  TORCH_CHECK(sa.numel() == m && sb.numel() == n, "scale sizes differ");
  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == a.device() && bias->scalar_type() == at::kFloat &&
                bias->is_contiguous() && bias->numel() == n, "invalid bias");
  }
  const auto properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 8, "SM8x only");
  auto out = at::empty({m,n}, a.options().dtype(at::kBFloat16));
  // CUTLASS 3.5 RowBroadcast's null_default field does not guard null loads.
  // Materialize zero bias for the no-bias case rather than passing nullptr.
  const auto bias_values = bias.has_value() ? *bias : at::zeros({n}, sa.options());
  Col::Arguments ac{sa.data_ptr<float>(), 0.f, {}};
  Row::Arguments bc{sb.data_ptr<float>(), 0.f, {}};
  Row::Arguments biasc{bias_values.data_ptr<float>(), 0.f, {}};
  Store::Arguments oc{reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()), {n, cute::_1{}, int64_t(m)*n}};
  Output::Arguments callbacks{{{{{}, ac, {}}, bc, {}}, biasc, {}}, oc};
  Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {m,n,k}, 1, callbacks,
      a.data_ptr(), b.data_ptr(), nullptr, out.data_ptr(), 0, 0, 0, 0,
      int64_t(k), int64_t(k), int64_t(n), int64_t(n)};
  Kernel::Params params(args, properties->multiProcessorCount, 1);
  constexpr int bytes = sizeof(Kernel::SharedStorage);
  if (bytes >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  }
  auto grid = Kernel::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
  fused_kernel<<<grid, 128, bytes, at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace lingbot_s4_fused
