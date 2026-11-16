#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/gemm/device/gemm.h>
#include <cmath>
#include "fused_s4_epilogue.cuh"

template <typename Input, int Channels>
__device__ void warp_hadamard(const Input* inputs, const float* signs, const float* smoothing,
                              float* transformed, int row, float normalization) {
  constexpr int Fragments = Channels / 256;
  float values[Fragments];
  #pragma unroll
  for (int fragment = 0; fragment < Fragments; ++fragment) {
    const int channel = fragment * 256 + threadIdx.x;
    float value = static_cast<float>(inputs[row * Channels + channel]) * signs[channel];
    #pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
      const float partner = __shfl_xor_sync(0xffffffff, value, stride);
      value = (threadIdx.x & stride) == 0 ? value + partner : partner - value;
    }
    transformed[channel] = value;
  }
  __syncthreads();
  const int transposed = (threadIdx.x % 8) * 32 + threadIdx.x / 8;
  #pragma unroll
  for (int fragment = 0; fragment < Fragments; ++fragment) {
    float value = transformed[fragment * 256 + transposed];
    #pragma unroll
    for (int stride = 1; stride < 8; stride *= 2) {
      const float partner = __shfl_xor_sync(0xffffffff, value, stride);
      value = (threadIdx.x & stride) == 0 ? value + partner : partner - value;
    }
    values[fragment] = value;
  }
  __syncthreads();
  #pragma unroll
  for (int fragment = 0; fragment < Fragments; ++fragment)
    transformed[fragment * 256 + transposed] = values[fragment];
  __syncthreads();
  #pragma unroll
  for (int fragment = 0; fragment < Fragments; ++fragment)
    values[fragment] = transformed[fragment * 256 + threadIdx.x];
  #pragma unroll
  for (int stride = 1; stride < Fragments; stride *= 2) {
    #pragma unroll
    for (int first = 0; first < Fragments; first += 2 * stride) {
      #pragma unroll
      for (int offset = 0; offset < stride; ++offset) {
        const float left = values[first + offset];
        const float right = values[first + offset + stride];
        values[first + offset] = left + right;
        values[first + offset + stride] = left - right;
      }
    }
  }
  #pragma unroll
  for (int fragment = 0; fragment < Fragments; ++fragment) {
    const int channel = fragment * 256 + threadIdx.x;
    float value = values[fragment] * normalization;
    if (smoothing != nullptr) value = __fdiv_rn(value, smoothing[channel]);
    transformed[channel] = value;
  }
  __syncthreads();
}

template <typename Input, bool Rotation, int FastChannels = 0>
__global__ void quantize_rows(const Input* inputs, const float* smoothing,
                              const float* signs, const float* activation_clip, uint8_t* packed, float* scales,
                              int channels, float normalization) {
  __shared__ float transformed[Rotation ? 4096 : 1];
  __shared__ float warp_maxima[8];
  __shared__ int warp_invalid[8];
  __shared__ float row_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  if constexpr (Rotation && FastChannels > 0) {
    warp_hadamard<Input, FastChannels>(inputs, signs, smoothing, transformed, row, normalization);
  } else if constexpr (Rotation) {
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x)
      transformed[channel] = static_cast<float>(inputs[row * channels + channel]) * signs[channel];
    __syncthreads();
    for (int stride = 1; stride < channels; stride *= 2) {
      for (int pair = threadIdx.x; pair < channels / 2; pair += blockDim.x) {
        const int first = (pair / stride) * (2 * stride) + pair % stride;
        const float left = transformed[first];
        const float right = transformed[first + stride];
        transformed[first] = left + right;
        transformed[first + stride] = left - right;
      }
      __syncthreads();
    }
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
      float value = transformed[channel] * normalization;
      if (smoothing != nullptr) value = __fdiv_rn(value, smoothing[channel]);
      transformed[channel] = value;
    }
    __syncthreads();
  }
  float maximum = 0.f;
  bool invalid = false;
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    float value;
    if constexpr (Rotation) {
      value = transformed[channel];
    } else {
      value = static_cast<float>(inputs[row * channels + channel]);
      if (smoothing != nullptr) value = __fdiv_rn(value, smoothing[channel]);
    }
    invalid = invalid || !isfinite(value);
    const float magnitude = activation_clip == nullptr ? fabsf(value) :
        (value < 0.f ? -value * activation_clip[0] : value * activation_clip[1]);
    maximum = fmaxf(maximum, magnitude);
  }
  for (int distance = 16; distance > 0; distance /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, distance));
  const unsigned invalid_threads = __ballot_sync(0xffffffff, invalid);
  if (lane == 0) {
    warp_maxima[warp] = maximum;
    warp_invalid[warp] = invalid_threads != 0;
  }
  __syncthreads();
  if (warp == 0) {
    const bool invalid_row = __any_sync(0xffffffff, lane < 8 && warp_invalid[lane] != 0);
    maximum = lane < 8 ? warp_maxima[lane] : 0.f;
    for (int distance = 16; distance > 0; distance /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, distance));
    if (lane == 0) {
      row_scale = maximum == 0.f ? 1.f : fmaxf(maximum * (1.f / 7.f), 1.1754943508222875e-38f);
      if (invalid_row) row_scale = __int_as_float(0x7fffffff);
      scales[row] = row_scale;
    }
  }
  __syncthreads();
  for (int channel_pair = threadIdx.x; channel_pair < channels / 2; channel_pair += blockDim.x) {
    const int channel = channel_pair * 2;
    float low, high;
    if constexpr (Rotation) {
      low = transformed[channel];
      high = transformed[channel + 1];
    } else {
      low = static_cast<float>(inputs[row * channels + channel]);
      high = static_cast<float>(inputs[row * channels + channel + 1]);
      if (smoothing != nullptr) {
        low = __fdiv_rn(low, smoothing[channel]);
        high = __fdiv_rn(high, smoothing[channel + 1]);
      }
    }
    const int low_code = max(-8, min(7, __float2int_rn(__fdiv_rn(low, row_scale))));
    const int high_code = max(-8, min(7, __float2int_rn(__fdiv_rn(high, row_scale))));
    packed[row * (channels / 2) + channel_pair] = (low_code & 15) | ((high_code & 15) << 4);
  }
}

template <typename Output>
__global__ void scale_output(const int32_t* accumulator, const float* activation_scale,
                             const float* weight_scale, const float* bias, Output* output,
                             int size, int columns) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  const int column = index % columns;
  float value = static_cast<float>(accumulator[index]) * activation_scale[index / columns];
  value *= weight_scale[column];
  if (bias != nullptr) value += bias[column];
  output[index] = static_cast<Output>(value);
}

at::Tensor int4_gemm_cuda(const at::Tensor& activation, const at::Tensor& weight) {
  TORCH_CHECK(activation.is_cuda() && weight.is_cuda(), "INT4 GEMM requires CUDA");
  TORCH_CHECK(activation.device() == weight.device(), "INT4 operands must share a device");
  TORCH_CHECK(activation.scalar_type() == at::kByte && weight.scalar_type() == at::kByte,
              "INT4 operands must be packed uint8");
  TORCH_CHECK(activation.dim() == 2 && weight.dim() == 2 && activation.is_contiguous() && weight.is_contiguous(),
              "INT4 operands must be contiguous matrices");
  TORCH_CHECK(activation.size(1) == weight.size(1) && activation.size(1) % 16 == 0 && weight.size(0) % 4 == 0,
              "INT4 requires matching K, K divisible by 32, N divisible by 4");
  TORCH_CHECK(activation.size(0) > 0 && activation.size(1) > 0 && weight.size(0) > 0,
              "INT4 operands cannot be empty");
  const c10::cuda::CUDAGuard guard(activation.device());
  const auto properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 8, "This CUTLASS INT4 backend supports SM80/86/89 only");
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t, cutlass::layout::RowMajor,
      cutlass::int4b_t, cutlass::layout::ColumnMajor,
      int32_t, cutlass::layout::RowMajor, int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 256>,
      cutlass::gemm::GemmShape<32, 64, 256>,
      cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::epilogue::thread::LinearCombinationClamp<int32_t, 4, int32_t, int32_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
  const int rows = activation.size(0);
  const int columns = weight.size(0);
  const int channels = activation.size(1) * 2;
  auto output = at::empty({rows, columns}, activation.options().dtype(at::kInt));
  typename Gemm::Arguments arguments(
      {rows, columns, channels},
      {reinterpret_cast<const cutlass::int4b_t*>(activation.data_ptr<uint8_t>()), channels},
      {reinterpret_cast<const cutlass::int4b_t*>(weight.data_ptr<uint8_t>()), channels},
      {output.data_ptr<int32_t>(), columns}, {output.data_ptr<int32_t>(), columns}, {1, 0});
  Gemm operation;
  auto status = operation.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  const auto stream = at::cuda::getCurrentCUDAStream();
  status = operation(arguments, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  return output;
}

at::Tensor int4_linear_cuda(const at::Tensor& inputs, const at::Tensor& weight,
                           const at::Tensor& weight_scale, const c10::optional<at::Tensor>& bias,
                           const c10::optional<at::Tensor>& smoothing,
                           const c10::optional<at::Tensor>& signs, int64_t output_type,
                           const c10::optional<at::Tensor>& activation_clip, bool fast_rotation, bool fused_epilogue) {
  TORCH_CHECK(inputs.is_cuda() && inputs.dim() == 2 && inputs.is_contiguous(), "Input must be a contiguous CUDA matrix");
  TORCH_CHECK(inputs.scalar_type() == at::kFloat || inputs.scalar_type() == at::kHalf || inputs.scalar_type() == at::kBFloat16,
              "Input must be FP32/FP16/BF16");
  TORCH_CHECK(weight.dim() == 2 && inputs.size(1) == weight.size(1) * 2 && inputs.device() == weight.device(), "Input/weight shape or device mismatch");
  const int rows = inputs.size(0);
  const int channels = inputs.size(1);
  const int columns = weight.size(0);
  TORCH_CHECK(rows > 0 && channels >= 32 && channels <= 4096 && (channels & (channels - 1)) == 0,
              "CUDA activation requires nonempty rows and power-of-two K in [32,4096]");
  TORCH_CHECK(weight_scale.is_cuda() && weight_scale.device() == inputs.device() && weight_scale.scalar_type() == at::kFloat &&
              weight_scale.is_contiguous() && weight_scale.numel() == columns, "Invalid weight scale");
  for (const auto& vector : {bias, smoothing, signs, activation_clip}) {
    if (vector.has_value()) {
      TORCH_CHECK(vector->device() == inputs.device() && vector->scalar_type() == at::kFloat && vector->is_contiguous(), "Invalid optional vector");
    }
  }
  TORCH_CHECK(!bias.has_value() || bias->numel() == columns, "Invalid bias size");
  TORCH_CHECK(!smoothing.has_value() || smoothing->numel() == channels, "Invalid smoothing size");
  TORCH_CHECK(!signs.has_value() || signs->numel() == channels, "Invalid signs size");
  TORCH_CHECK(!activation_clip.has_value() || activation_clip->numel() == 2, "Invalid activation clipping size");
  TORCH_CHECK(output_type >= 0 && output_type <= 2, "Invalid output dtype");
  const c10::cuda::CUDAGuard guard(inputs.device());
  auto packed = at::empty({rows, channels / 2}, inputs.options().dtype(at::kByte));
  auto scales = at::empty({rows}, inputs.options().dtype(at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const float normalization = 1.0 / std::sqrt(static_cast<double>(channels));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, inputs.scalar_type(), "quantize_int4_rows", [&] {
    if (signs.has_value() && fast_rotation && channels == 1024) {
      quantize_rows<scalar_t, true, 1024><<<rows, 256, 0, stream>>>(inputs.data_ptr<scalar_t>(), smoothing.has_value() ? smoothing->data_ptr<float>() : nullptr,
          signs->data_ptr<float>(), activation_clip.has_value() ? activation_clip->data_ptr<float>() : nullptr,
          packed.data_ptr<uint8_t>(), scales.data_ptr<float>(), channels, normalization);
    } else if (signs.has_value() && fast_rotation && channels == 4096) {
      quantize_rows<scalar_t, true, 4096><<<rows, 256, 0, stream>>>(inputs.data_ptr<scalar_t>(), smoothing.has_value() ? smoothing->data_ptr<float>() : nullptr,
          signs->data_ptr<float>(), activation_clip.has_value() ? activation_clip->data_ptr<float>() : nullptr,
          packed.data_ptr<uint8_t>(), scales.data_ptr<float>(), channels, normalization);
    } else if (signs.has_value()) {
      quantize_rows<scalar_t, true><<<rows, 256, 0, stream>>>(inputs.data_ptr<scalar_t>(), smoothing.has_value() ? smoothing->data_ptr<float>() : nullptr,
          signs->data_ptr<float>(), activation_clip.has_value() ? activation_clip->data_ptr<float>() : nullptr,
          packed.data_ptr<uint8_t>(), scales.data_ptr<float>(), channels, normalization);
    } else {
      quantize_rows<scalar_t, false><<<rows, 256, 0, stream>>>(inputs.data_ptr<scalar_t>(), smoothing.has_value() ? smoothing->data_ptr<float>() : nullptr,
          nullptr, activation_clip.has_value() ? activation_clip->data_ptr<float>() : nullptr,
          packed.data_ptr<uint8_t>(), scales.data_ptr<float>(), channels, normalization);
    }
  });
  if (fused_epilogue && output_type == 2) {
    return lingbot_s4_fused::fused_scaled_s4(packed, weight, scales, weight_scale, bias);
  }
  auto accumulator = int4_gemm_cuda(packed, weight);
  const auto dtype = output_type == 0 ? at::kFloat : (output_type == 1 ? at::kHalf : at::kBFloat16);
  auto output = at::empty({rows, columns}, inputs.options().dtype(dtype));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, output.scalar_type(), "scale_int4_output", [&] {
    scale_output<scalar_t><<<(rows * columns + 255) / 256, 256, 0, stream>>>(accumulator.data_ptr<int32_t>(), scales.data_ptr<float>(),
        weight_scale.data_ptr<float>(), bias.has_value() ? bias->data_ptr<float>() : nullptr, output.data_ptr<scalar_t>(), rows * columns, columns);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
