#include "ops/linear/linear_test_common.h"

#include "core/device.h"
#include "core/decode_graph.h"
#include "ops/a8_g64_reference.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

int q4_a16_conformance() {
    int failures = 0;

    constexpr std::array kN1024K5120{
        convenience(1), a16(2),  a16(3),  a16(4),  a16(8),
        a16(15),        a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {1024, 5120, 101U, Comparison::Full, true, kN1024K5120});

    constexpr std::array kN4096K5120{
        a16(1), a16(2), a16(3), a16(4), a16(5), a16(6), a16(8), a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {4096, 5120, 103U, Comparison::Sampled, false, kN4096K5120});

    constexpr std::array kN6144K5120{
        a16(1), a16(2),  a16(3),  a16(4),  a16(7),  a16(8),
        a16(9), a16(12), a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {6144, 5120, 107U, Comparison::Sampled, false, kN6144K5120});

    constexpr std::array kN7168K5120{
        a16(1),  a16(2),  a16(3),  a16(4),  a16(7),  a16(8),  a16(9),
        a16(10), a16(12), a16(15), a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {7168, 5120, 109U, Comparison::Sampled, false, kN7168K5120});

    constexpr std::array kN34816K5120{
        a16(1), a16(2), a16(3), a16(4), a16(5), a16(6), a16(8), a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {34816, 5120, 113U, Comparison::Sampled, false, kN34816K5120});

    constexpr std::array kN131072K5120{
        a16(1), a16(2), a16(3), a16(4), a16(5), a16(6), a16(7), a16(8), a16(9), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {131072, 5120, 127U, Comparison::Sampled, false, kN131072K5120});

    constexpr std::array kN248320K5120{
        a16(1), a16(2), a16(3), a16(4), a16(8), a16(15), a16(16), a16(17), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {248320, 5120, 149U, Comparison::Sampled, false, kN248320K5120});

    // DFlash2 companion geometries: attention output [5120,4096], MLP down [5120,17408], and
    // feature projection [5120,25600]. T=1 and the small column region drive decode catch-up
    // and the proposal round; T=128/2048 exercise the wide conv-add and prefill routes.
    constexpr std::array kN5120K4096{
        a16(1), a16(2), a16(4), a16(7), a16(8), a16(16), a16(17), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {5120, 4096, 151U, Comparison::Sampled, false, kN5120K4096});

    constexpr std::array kN5120K17408{
        a16(1), a16(2), a16(4), a16(7), a16(8), a16(16), a16(17), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {5120, 17408, 157U, Comparison::Sampled, false, kN5120K17408});

    constexpr std::array kN5120K25600{
        a16(1), a16(2), a16(4), a16(7), a16(8), a16(16), a16(17), a16(128), a16(2048),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {5120, 25600, 163U, Comparison::Sampled, false, kN5120K25600});

    constexpr std::array kN131072K2048{
        a16(1),   a16(2),   a16(8),   a16(9),   a16(16),  a16(17),  a16(20),  a16(21),  a16(32),
        a16(33),  a16(48),  a16(49),  a16(56),  a16(57),  a16(63),  a16(64),  a16(65),  a16(72),
        a16(73),  a16(80),  a16(81),  a16(96),  a16(97),  a16(103), a16(104), a16(105), a16(111),
        a16(112), a16(113), a16(119), a16(120), a16(121), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {131072, 2048, 131U, Comparison::Sampled, false, kN131072K2048});

    constexpr std::array kN3456K1152{
        a16(4),   a16(20),  a16(36),  a16(40),   a16(44),     a16(128),
        a16(320), a16(324), a16(328), a16(1024), a16(131072),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {3456, 1152, 137U, Comparison::Sampled, false, kN3456K1152});

    constexpr std::array kN4304K1152{
        a16(4),  a16(8),   a16(12),  a16(16),  a16(20),  a16(24),   a16(28),
        a16(32), a16(128), a16(320), a16(324), a16(328), a16(1024), a16(131072),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {4304, 1152, 139U, Comparison::Sampled, false, kN4304K1152});

    // Inkling-2-Mini geometries: K=2304 -> N in {1024, 3072, 6528, 201024} and K=3072 -> N=2304.
    constexpr std::array kInklingK2304{
        convenience(1), a16(2), a16(7), a16(8), a16(16), a16(17), a16(64), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {1024, 2304, 141U, Comparison::Full, true, kInklingK2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {3072, 2304, 143U, Comparison::Sampled, false, kInklingK2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {6528, 2304, 145U, Comparison::Sampled, false, kInklingK2304});

    constexpr std::array kN201024K2304{
        a16(1), a16(4), a16(8), a16(16), a16(17), a16(32), a16(33), a16(64), a16(65), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {201024, 2304, 147U, Comparison::Sampled, false, kN201024K2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {2304, 3072, 149U, Comparison::Sampled, false, kInklingK2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {4096, 2304, 153U, Comparison::Sampled, false, kInklingK2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {6912, 2304, 155U, Comparison::Sampled, false, kInklingK2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {13824, 2304, 157U, Comparison::Sampled, false, kInklingK2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {2304, 4096, 159U, Comparison::Sampled, false, kInklingK2304});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {2304, 6912, 161U, Comparison::Sampled, false, kInklingK2304});

    // Inkling-2-Mini vision geometries: K=1024 -> N in {3072, 4096}, including odd T_vis.
    constexpr std::array kInklingVisionK1024{
        a16(1), a16(5), a16(7), a16(16), a16(17), a16(65), a16(257), a16(321),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {3072, 1024, 163U, Comparison::Sampled, false, kInklingVisionK1024});
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4g64_f16s_weight,
                          {4096, 1024, 165U, Comparison::Sampled, false, kInklingVisionK1024});

    return failures;
}

int q4_a8_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    constexpr std::array kTokens{1, 16, 63, 64, 65, 128, 256, 300};
    const std::int32_t max_t = 300;
    std::vector<std::int32_t> rows;
    if (n <= 1024) {
        rows.resize(n);
        for (std::int32_t r = 0; r < n; ++r) rows[r] = r;
    } else {
        for (int s = 0; s < 32; ++s) rows.push_back(static_cast<std::int32_t>(std::int64_t(n - 1) * s / 31));
    }
    const auto packed = make_q4g64_f16s_weight(n, k, seed);
    const std::vector<float> oracle_w = ninfer::test::quantized_weight::materialize_rows_fp32(packed, rows);
    std::vector<std::uint16_t> act_bits(std::size_t(k) * max_t);
    for (std::size_t i = 0; i < act_bits.size(); ++i) {
        std::uint32_t c = seed ^ (static_cast<std::uint32_t>(i) * 0x9e3779b9U);
        c ^= c >> 16; c *= 0x7feb352dU; c ^= c >> 15; c *= 0x846ca68bU; c ^= c >> 16;
        act_bits[i] = ninfer::test::f32_to_bf16(static_cast<float>(int(c & 0xffU) - 128) * (1.0f / 256.0f));
        if (i % 64 == (i / 64) % 61) act_bits[i] = ninfer::test::f32_to_bf16(16.0f * ninfer::test::bf16_to_f32(act_bits[i]));
    }
    const auto quantized = ninfer::test::a8_g64_dequantized(act_bits, k, max_t);
    DeviceBuffer dw(packed.payload.size()), dx(act_bits.size() * 2);
    dw.copy_from_host(packed.payload.data(), dw.bytes);
    dx.copy_from_host(act_bits.data(), dx.bytes);
    const Weight w = packed.device_weight(dw.p);

    DeviceContext ctx;
    int failures = 0;
    for (std::int32_t t : kTokens) {
        const bool expect_a8 = (t >= 64);
        ninfer::test::GuardedDeviceBuffer out(std::size_t(n) * t * 2);
        Tensor tx(dx.p, DType::BF16, {k, t}), td(out.data(), DType::BF16, {n, t});
        std::vector<float> oracle_x(std::size_t(k) * t);
        for (std::size_t i = 0; i < oracle_x.size(); ++i)
            oracle_x[i] = expect_a8 ? static_cast<float>(quantized[i]) : ninfer::test::bf16_to_f32(act_bits[i]);
        std::vector<double> ref(rows.size() * t);
        cpu_linear_gemm_fp64(oracle_w.data(), oracle_x.data(), ref.data(), static_cast<std::int32_t>(rows.size()), k, t);
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::Q4G64_F16S, n, k, ops::LinearPolicy::AllowA8, t, t);
        failures += (capacity != 0) != expect_a8;
        WorkspaceArena ws(std::max<std::size_t>(capacity, 256));
        const auto launch = [&] { ops::linear(tx, w, td, ops::LinearPolicy::AllowA8, ws, ctx.stream); };
        if (t == 64 || t == 256) {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable graph;
            definition.capture(ctx.stream, launch);
            graph.instantiate(definition);
            graph.launch(ctx.stream);
        } else {
            launch();
        }
        std::vector<std::uint16_t> got_bits(std::size_t(n) * t);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        CUDA_CHECK(cudaMemcpy(got_bits.data(), out.data(), got_bits.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<double> actual(rows.size() * t);
        for (std::int32_t tk = 0; tk < t; ++tk)
            for (std::size_t ri = 0; ri < rows.size(); ++ri)
                actual[tk * rows.size() + ri] = ninfer::test::bf16_to_f32(got_bits[std::size_t(tk) * n + rows[ri]]);
        failures += ninfer::test::verify_reduction(
            "Q4_A8 [" + std::to_string(n) + "," + std::to_string(k) + "] T=" + std::to_string(t),
            actual, ref, ninfer::test::kDocumentedA8Criterion);
        failures += out.verify_guards("Q4_A8 guards");
    }
    return failures;
}

int q4_a8_conformance() {
    int failures = q4_a8_shape(1024, 2304, 201U);
    failures += q4_a8_shape(3072, 2304, 203U);
    failures += q4_a8_shape(6528, 2304, 205U);
    failures += q4_a8_shape(2304, 3072, 207U);
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q4_a16_conformance() + q4_a8_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4_A16/Q4_A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
