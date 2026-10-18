#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/common/rowsplit_a8_quantize.h"
#include "ops/linear/q3/q3_dispatch.h"
#include "ops/linear/q3/q3_launch.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

int verify_fused_a8_out() {
    using namespace ninfer;
    if (test::cuda_unavailable()) { return 0; }
    constexpr std::int32_t kGateUpRows   = 34816;
    constexpr std::int32_t kInputRows    = 5120;
    constexpr std::int32_t kOutputRows   = 17408;
    constexpr std::array<std::int32_t, 4> kTokens{129, 256, 1024, 1025};
    constexpr std::int32_t kMaxTokens    = 1025;

    test::quantized_weight::PatternedWeightOptions weight_options{};
    const auto host_weight = test::quantized_weight::make_patterned_weight(
        QType::Q3G128_F16S, kGateUpRows, kInputRows, 1601U, weight_options);
    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    std::vector<std::uint16_t> host_x(static_cast<std::size_t>(kInputRows) * kMaxTokens);
    for (std::size_t i = 0; i < host_x.size(); ++i) {
        const int v = static_cast<int>((i * 1315423911ULL >> 16) % 63ULL) - 31;
        host_x[i]   = test::f32_to_bf16(static_cast<float>(v) * 1.25e-3F);
    }
    test::GuardedDeviceBuffer device_x(host_x.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(host_x.data(), host_x.size() * sizeof(std::uint16_t));

    WorkspaceArena ws(256ULL * 1024ULL * 1024ULL);
    int failures = 0;
    for (const std::int32_t tokens : kTokens) {
        ws.reset();
        Tensor x(device_x.data(), DType::BF16, {kInputRows, tokens});
        Tensor bf16_out = ws.alloc(DType::BF16, {kOutputRows, tokens});
        ops::linear_swiglu(x, weight, bf16_out, ops::LinearPolicy::AllowA8, ws, nullptr);
        ops::detail::A8G64Activation ref_act =
            ops::detail::allocate_a8_g64_activation(ws, kOutputRows, tokens);
        ops::detail::a8_g64_quantize(bf16_out, ref_act, nullptr);

        ops::detail::A8G64Activation act_in =
            ops::detail::allocate_a8_g64_activation(ws, kInputRows, tokens);
        ops::detail::a8_g64_quantize(x, act_in, nullptr);
        ops::detail::A8G64Activation fused_act =
            ops::detail::allocate_a8_g64_activation(ws, kOutputRows, tokens);
        if (tokens >= ops::detail::kQ3A8WideMinTokens) {
            ops::detail::launch_q3_mma_tall_a8_swiglu_a8_out_r64_c256(act_in, weight, fused_act,
                                                                      nullptr);
        } else {
            ops::detail::launch_q3_mma_tall_a8_swiglu_a8_out_r64_c128(act_in, weight, fused_act,
                                                                      nullptr);
        }
        test::cuda_check(cudaDeviceSynchronize(), "synchronize fused swiglu a8_out");

        std::vector<std::uint8_t> ref_q(ref_act.q.bytes());
        std::vector<std::uint8_t> fused_q(fused_act.q.bytes());
        std::vector<std::uint8_t> ref_s(ref_act.scale.bytes());
        std::vector<std::uint8_t> fused_s(fused_act.scale.bytes());
        test::cuda_check(cudaMemcpy(ref_q.data(), ref_act.q.data, ref_q.size(),
                                    cudaMemcpyDeviceToHost),
                         "copy ref_q");
        test::cuda_check(cudaMemcpy(fused_q.data(), fused_act.q.data, fused_q.size(),
                                    cudaMemcpyDeviceToHost),
                         "copy fused_q");
        test::cuda_check(cudaMemcpy(ref_s.data(), ref_act.scale.data, ref_s.size(),
                                    cudaMemcpyDeviceToHost),
                         "copy ref_s");
        test::cuda_check(cudaMemcpy(fused_s.data(), fused_act.scale.data, fused_s.size(),
                                    cudaMemcpyDeviceToHost),
                         "copy fused_s");
        if (std::memcmp(ref_q.data(), fused_q.data(), ref_q.size()) != 0 ||
            std::memcmp(ref_s.data(), fused_s.data(), ref_s.size()) != 0) {
            std::cerr << "LinearSwiGLU Q3_A8 fused a8_out mismatch at T=" << tokens << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // Token cases straddle the A16 projection-schedule boundaries and the documented A8
        // threshold at 129 columns, and cover one 64-token tile, several tiles and a partial last
        // tile under AllowA8.
        constexpr std::array<std::int32_t, 20> kTokenCases{
            1,  2,  3,   4,   5,   6,   7,   8,   9,   12,
            16, 17, 64,  65,  128, 129, 192, 256, 385, 513,
        };
        int failures = run_profile(
            "LinearSwiGLU Q3_A8",
            {QType::Q3G128_F16S, 34816, 5120, 17408, 1601U, ActivationCompute::A8},
            kTokenCases, std::array<std::int32_t, 3>{129, 192, 385});
        failures += verify_fused_a8_out();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q3_A8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q3_A8 test failed: " << error.what() << '\n';
        return 1;
    }
}
