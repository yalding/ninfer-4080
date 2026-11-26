#include "ninfer/ops/inkling2_fused.h"
#include "core/cyclic_kv_cache.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/layout.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr ReductionCriterion kBF{1.85e-3, 1e-5, 3.95e-3};
constexpr ReductionCriterion kFP{2e-6, 2e-5, 5e-6};
constexpr ReductionCriterion kFPMma{1e-5, 2e-5, 2e-5};
std::vector<float> values(std::size_t n, int seed, float scale = 1, bool represented = true) {
    std::vector<float> out(n);
    fill_uniform(out, seed, -scale, scale);
    if (represented) round_to_bf16(out);
    return out;
}
void graph_run(DeviceContext& device, const std::function<void()>& body, bool graph) {
    // Test buffer fills use the default stream; DeviceContext uses a nonblocking stream.
    cuda_synchronize();
    if (graph) {
        DecodeGraphDefinition def; DecodeGraphExecutable exec;
        def.capture(device.stream, body); exec.instantiate(def); exec.launch(device.stream);
    } else body();
    device.synchronize();
}
int check(const char* label, const GuardedDeviceBuffer& out, const std::vector<double>& expected,
          bool fp32 = false, const ReductionCriterion* custom = nullptr) {
    std::vector<double> actual;
    if (fp32) {
        auto floats = from_device<float>(out.data(), expected.size());
        actual.assign(floats.begin(), floats.end());
    } else actual = from_device_bf16(out.data(), expected.size());
    const auto& crit = custom ? *custom : (fp32 ? kFP : kBF);
    return verify_reduction(label, actual, expected, crit) + out.verify_guards(label);
}
double invnorm(const float* data, int n) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(data[i]) * data[i];
    return 1 / std::sqrt(ss / n + 1e-6);
}

int projection(DeviceContext& dev, int t, int n, bool fp32, bool graph) {
    auto x = values(2304 * t, 1), w = values(2304 * n, 2);
    auto dx = to_device_bf16(x), dw = to_device_bf16(w);
    GuardedDeviceBuffer result(t * n * (fp32 ? 4 : 2));
    Tensor tx(dx.p, DType::BF16, {2304, t}), tw(dw.p, DType::BF16, {2304, n});
    Tensor out(result.data(), fp32 ? DType::FP32 : DType::BF16, {n, t});
    std::vector<double> expected(n * t);
    for (int i = 0; i < t; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < 2304; ++k)
                expected[i * n + j] += double(x[i * 2304 + k]) * w[j * 2304 + k];
    graph_run(dev, [&] { ops::inkling2_small_bf16_proj(tx, tw, out, dev.stream); }, graph);
    return check("inkling2 projection", result, expected, fp32, fp32 ? &kFPMma : nullptr);
}

int convolution(DeviceContext& dev, int c, int t, int dynamic_type, bool alias, bool residual,
                bool graph) {
    auto x = values(c * t, 11), old = values(c * 3, 12), w = values(c * 4, 13, .5, false);
    auto taps = values(32 * t, 14, .25, dynamic_type == 2), res = values(c * t, 15);
    auto dx = to_device_bf16(x), ds = to_device_bf16(old), dw = to_device(w);
    auto dd = dynamic_type == 2 ? to_device_bf16(taps) : to_device(taps);
    auto dr = to_device_bf16(res);
    GuardedDeviceBuffer result(c * t * 2), new_state(c * 3 * 2);
    Tensor tx(dx.p, DType::BF16, {c, t}), tw(dw.p, DType::FP32, {4, c});
    Tensor si(ds.p, DType::BF16, {c, 3});
    Tensor so(alias ? ds.p : new_state.data(), DType::BF16, {c, 3});
    Tensor out(result.data(), DType::BF16, {c, t});
    Tensor td, tr;
    if (dynamic_type) td = Tensor(dd.p, dynamic_type == 2 ? DType::BF16 : DType::FP32, {32, t});
    if (residual) tr = Tensor(dr.p, DType::BF16, {c, t});
    auto input = [&](int d, int token) -> double {
        return token >= 0 ? x[token * c + d] : old[(token + 3) * c + d];
    };
    std::vector<double> expected(c * t), expected_state(c * 3);
    for (int token = 0; token < t; ++token) for (int d = 0; d < c; ++d) {
        double sum = x[token * c + d];
        for (int tap = 0; tap < 4; ++tap) sum += w[d * 4 + tap] * input(d, token - 3 + tap);
        if (dynamic_type)
            for (int lag = 0; lag < 4; ++lag)
                sum += taps[token * 32 + (d / 128) * 4 + lag] * input(d, token - lag);
        sum /= dynamic_type ? 3 : std::sqrt(5.0);
        if (residual) sum += res[token * c + d];
        expected[token * c + d] = sum;
    }
    for (int lag = 0; lag < 3; ++lag) for (int d = 0; d < c; ++d)
        expected_state[lag * c + d] = input(d, t - 3 + lag);
    graph_run(dev, [&] { ops::inkling2_fused_sconv(tx, tw, td, si, so, tr, out, dev.stream); }, graph);
    int fail = check("inkling2 convolution", result, expected);
    fail += verify_exact("inkling2 convolution state", from_device_bf16(so.data, c * 3), expected_state);
    fail += new_state.verify_guards("convolution state guards");
    fail += verify_exact("convolution preserves input", from_device_bf16(dx, c * t), std::vector<double>(x.begin(), x.end()));
    return fail;
}

int mixing(DeviceContext& dev, int t, int m, int r, bool norm, bool graph, float scale = 1) {
    std::vector<std::vector<float>> x;
    std::vector<DeviceBuffer> dx;
    std::vector<Tensor> states;
    for (int j = 0; j < m; ++j) {
        x.push_back(values(2304 * t, 30 + j, scale)); dx.push_back(to_device_bf16(x.back()));
        states.emplace_back(dx.back().p, DType::BF16, std::initializer_list<int>{2304, t});
    }
    auto query = values(2304 * r, 45, .2, false), nw = values(2304 * r, 46);
    auto dq = to_device(query), dn = to_device_bf16(nw);
    Tensor tq(dq.p, DType::FP32, {2304, r});
    std::vector<Tensor> norms;
    if (norm) for (int j = 0; j < r; ++j)
        norms.emplace_back(static_cast<std::uint16_t*>(dn.p) + j * 2304, DType::BF16,
                           std::initializer_list<int>{2304});
    GuardedDeviceBuffer result(static_cast<std::size_t>(2304) * t * r * 2);
    Tensor out(result.data(), DType::BF16, {2304, t, r});
    std::vector<double> expected(static_cast<std::size_t>(2304) * t * r);
    for (int token = 0; token < t; ++token) for (int branch = 0; branch < r; ++branch) {
        std::vector<double> inverse(m), probability(m);
        double maximum = -INFINITY;
        for (int j = 0; j < m; ++j) {
            inverse[j] = invnorm(x[j].data() + token * 2304, 2304);
            double dot = 0;
            for (int d = 0; d < 2304; ++d) dot += query[branch * 2304 + d] * inverse[j] * x[j][token * 2304 + d];
            probability[j] = dot / 48;
            maximum = std::max(maximum, probability[j]);
        }
        double sum = 0;
        for (auto& p : probability) { p = std::exp(p - maximum); sum += p; }
        for (auto& p : probability) p /= sum;
        const auto base = (branch * t + token) * 2304;
        double ss = 0;
        for (int d = 0; d < 2304; ++d) {
            double value = 0;
            for (int j = 0; j < m; ++j) value += probability[j] * inverse[j] * x[j][token * 2304 + d];
            expected[base + d] = value; ss += value * value;
        }
        if (norm) for (int d = 0; d < 2304; ++d)
            expected[base + d] *= nw[branch * 2304 + d] / std::sqrt(ss / 2304 + 1e-6);
    }
    graph_run(dev, [&] { ops::inkling2_attn_res_mix(states, tq, norms, out, dev.stream); }, graph);
    return check("inkling2 AttnRes", result, expected);
}

int control(DeviceContext& dev, int t, int controllers, bool norm, bool graph, bool alias,
            bool broadcast = false) {
    const int cur_c = broadcast ? 1 : controllers;
    const int cur_size = 2304 * t * cur_c;
    const int size = 2304 * t * controllers;
    auto cur = values(cur_size, 61), msg = values(size, 62);
    auto down = values(4608 * 64 * controllers, 63, .005);
    auto gf = values(64 * 2304 * 2 * controllers, 64, .01);
    auto nw = values(2304 * controllers, 65);
    auto dc = to_device_bf16(cur), dm = to_device_bf16(msg), dd = to_device_bf16(down);
    auto dg = to_device_bf16(gf), dn = to_device_bf16(nw);
    GuardedDeviceBuffer result(size * 2);
    Tensor tc(dc.p, DType::BF16, {2304, t, cur_c}), tm(dm.p, DType::BF16, {2304, t, controllers});
    Tensor td(dd.p, DType::BF16, {4608, 64, controllers});
    Tensor tg(dg.p, DType::BF16, {64, 2304, 2, controllers});
    Tensor out(alias ? dc.p : result.data(), DType::BF16, {2304, t, controllers});
    std::vector<Tensor> norms;
    if (norm) for (int j = 0; j < controllers; ++j)
        norms.emplace_back(static_cast<std::uint16_t*>(dn.p) + j * 2304, DType::BF16, std::initializer_list<int>{2304});
    std::vector<double> expected(size);
    for (int controller = 0; controller < controllers; ++controller)
        for (int token = 0; token < t; ++token) {
            const int cur_base = ((broadcast ? 0 : controller) * t + token) * 2304;
            const int base     = (controller * t + token) * 2304;
            const double ic = invnorm(cur.data() + cur_base, 2304), im = invnorm(msg.data() + base, 2304);
            double z[64]{};
            for (int r = 0; r < 64; ++r)
                for (int d = 0; d < 2304; ++d) {
                    z[r] += down[(controller * 64 + r) * 4608 + d] * cur[cur_base + d] * ic;
                    z[r] += down[(controller * 64 + r) * 4608 + 2304 + d] * msg[base + d] * im;
                }
            double ss = 0;
            for (int d = 0; d < 2304; ++d) {
                double g = 0, f = 0;
                for (int r = 0; r < 64; ++r) {
                    g += gf[((controller * 2) * 2304 + d) * 64 + r] * z[r];
                    f += gf[((controller * 2 + 1) * 2304 + d) * 64 + r] * z[r];
                }
                const double y = cur[cur_base + d] * (1 - std::tanh(f)) + msg[base + d] * std::tanh(g);
                expected[base + d] = y; ss += y * y;
            }
            if (norm) for (int d = 0; d < 2304; ++d)
                expected[base + d] *= nw[controller * 2304 + d] / std::sqrt(ss / 2304 + 1e-6);
        }
    WorkspaceArena workspace(ops::inkling2_loop_control_workspace_capacity_bytes(controllers, 1, t));
    graph_run(dev, [&] { ops::inkling2_loop_control(tc, tm, td, tg, norms, out, workspace, dev.stream); }, graph);
    int fail = verify_reduction("inkling2 loop control", from_device_bf16(out.data, size), expected, kBF);
    fail += result.verify_guards("control output");
    if (workspace.used() != 0) ++fail;
    return fail;
}

int clipping(DeviceContext& dev, int t, bool tau, bool alias, bool graph, float scale = 5) {
    auto qgr = values(6528 * t, 71, scale), k = values(1024 * t, 72, scale), v = values(1024 * t, 73, scale);
    auto qc = values(128 * 24, 74, 2, false), rc = values(16 * 24, 75, 2, false);
    std::vector<int> positions(t);
    for (int i = 0; i < t; ++i) positions[i] = std::array<int, 4>{0, 8191, 8192, 131071}[i % 4];
    auto dq = to_device_bf16(qgr), dk = to_device_bf16(k), dv = to_device_bf16(v);
    auto dqc = to_device(qc), drc = to_device(rc), dp = to_device(positions);
    GuardedDeviceBuffer qo(3072 * t * 2), ko(1024 * t * 2), vo(1024 * t * 2), ro(384 * t * 4);
    Tensor tq(dq.p, DType::BF16, {6528, t}), tk(dk.p, DType::BF16, {1024, t}), tv(dv.p, DType::BF16, {1024, t});
    Tensor tqc(dqc.p, DType::FP32, {128, 24}), trc(drc.p, DType::FP32, {16, 24}), tp(dp.p, DType::I32, {t});
    Tensor tqo(qo.data(), DType::BF16, {128, 24, t});
    Tensor tko(alias ? dk.p : ko.data(), DType::BF16, {128, 8, t});
    Tensor tvo(alias ? dv.p : vo.data(), DType::BF16, {128, 8, t}), tro(ro.data(), DType::FP32, {16, 24, t});
    std::array<std::vector<double>, 4> ref{std::vector<double>(3072 * t), std::vector<double>(1024 * t), std::vector<double>(1024 * t), std::vector<double>(384 * t)};
    for (int token = 0; token < t; ++token) for (int which = 0; which < 4; ++which) {
        const int width = which == 3 ? 16 : 128, heads = (which == 0 || which == 3) ? 24 : 8;
        for (int h = 0; h < heads; ++h) {
            double ss = 0;
            for (int d = 0; d < width; ++d) {
                const auto i = (token * heads + h) * width + d;
                double value = which == 0 ? qgr[token * 6528 + h * 128 + d] * double(qc[h * 128 + d])
                             : which == 1 ? k[i] : which == 2 ? v[i]
                             : qgr[token * 6528 + 6144 + h * 16 + d] * double(rc[h * 16 + d]);
                ref[which][i] = value; ss += value * value;
            }
            double factor = which == 2 ? 1 / std::sqrt(ss / 128 + 1e-6)
                : std::min(1.0, (which == 3 ? 8 : 16 * std::sqrt(2.0)) / (std::sqrt(ss) + 1e-6));
            if (tau && (which == 0 || which == 3)) factor *= 1 + .1 * std::log(std::max(1.0, (positions[token] + 1.0) / 8192));
            for (int d = 0; d < width; ++d) ref[which][(token * heads + h) * width + d] *= factor;
        }
    }
    graph_run(dev, [&] { ops::inkling2_qkr_clip_v_norm(tq, tk, tv, tqc, trc, tp, tau, tqo, tko, tvo, tro, dev.stream); }, graph);
    int fail = check("inkling2 Q clip", qo, ref[0]);
    fail += verify_reduction("inkling2 K clip", from_device_bf16(tko.data, ref[1].size()), ref[1], kBF);
    fail += verify_reduction("inkling2 V norm", from_device_bf16(tvo.data, ref[2].size()), ref[2], kBF);
    fail += check("inkling2 R clip", ro, ref[3], true);
    return fail + ko.verify_guards("K guards") + vo.verify_guards("V guards");
}

int softcap(DeviceContext& dev, int t, bool padded, bool alias, bool graph) {
    constexpr int valid = 200058, stride = 201024;
    auto x = values(static_cast<std::size_t>(stride) * t, 81, 100);
    auto dx = to_device_bf16(x);
    const int rows = padded ? stride : valid;
    GuardedDeviceBuffer result(static_cast<std::size_t>(rows) * t * 2);
    result.fill(0);
    Tensor tx(dx.p, DType::BF16, {stride, t}), out(alias ? dx.p : result.data(), DType::BF16, {rows, t});
    std::vector<double> expected(static_cast<std::size_t>(rows) * t);
    for (int token = 0; token < t; ++token) for (int i = 0; i < rows; ++i)
        expected[token * rows + i] = i < valid ? 25 * std::tanh(x[token * stride + i] / 25.0)
                                                 : alias ? x[token * stride + i] : 0;
    graph_run(dev, [&] { ops::inkling2_softcap_slice(tx, 25, valid, out, dev.stream); }, graph);
    return verify_reduction("inkling2 softcap slice", from_device_bf16(out.data, expected.size()), expected, kBF)
        + result.verify_guards("softcap guards");
}

int dspark_feature_mix_test(DeviceContext& dev, int t, bool graph) {
    auto feat = values(2304 * t * 4, 91, 1.0f);
    auto fuse = values(16, 92, 1.0f, false);
    auto nw   = values(2304 * 4, 93, 1.0f);
    auto df   = to_device_bf16(feat), dfuse = to_device(fuse), dnw = to_device_bf16(nw);
    GuardedDeviceBuffer result(2304 * t * 4 * 2);
    Tensor tf(df.p, DType::BF16, {2304, t, 4}), tfw(dfuse.p, DType::FP32, {4, 4});
    Tensor out(result.data(), DType::BF16, {2304, t, 4});
    std::vector<Tensor> norms;
    for (int r = 0; r < 4; ++r)
        norms.emplace_back(static_cast<std::uint16_t*>(dnw.p) + r * 2304, DType::BF16,
                           std::initializer_list<int>{2304});
    std::vector<double> expected(2304 * t * 4);
    for (int r = 0; r < 4; ++r) {
        double p[4]{}, mx = -INFINITY, sum = 0;
        for (int j = 0; j < 4; ++j) mx = std::max(mx, double(fuse[r * 4 + j]));
        for (int j = 0; j < 4; ++j) { p[j] = std::exp(fuse[r * 4 + j] - mx); sum += p[j]; }
        for (int j = 0; j < 4; ++j) p[j] /= sum;
        for (int tok = 0; tok < t; ++tok) {
            double ss = 0;
            for (int d = 0; d < 2304; ++d) {
                double v = 0;
                for (int j = 0; j < 4; ++j) v += p[j] * feat[(j * t + tok) * 2304 + d];
                expected[(r * t + tok) * 2304 + d] = v;
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss / 2304.0 + 1e-6);
            for (int d = 0; d < 2304; ++d)
                expected[(r * t + tok) * 2304 + d] *= inv * nw[r * 2304 + d];
        }
    }
    graph_run(dev, [&] { ops::inkling2_dspark_feature_mix(tf, tfw, norms, out, dev.stream); }, graph);
    return check("dspark feature_mix", result, expected);
}

int dspark_rmsnorm_rope_test(DeviceContext& dev, int t, bool graph) {
    auto q = values(128 * 24 * t, 94, 0.5f), k = values(128 * 8 * t, 95, 0.5f);
    auto qn = values(128, 96, 1.0f), kn = values(128, 97, 1.0f);
    std::vector<int> pos(t);
    for (int i = 0; i < t; ++i) pos[i] = 1020 + i;
    auto dq = to_device_bf16(q), dk = to_device_bf16(k);
    auto dqn = to_device_bf16(qn), dkn = to_device_bf16(kn), dp = to_device(pos);
    Tensor tq(dq.p, DType::BF16, {128, 24, t}), tk(dk.p, DType::BF16, {128, 8, t});
    Tensor tqn(dqn.p, DType::BF16, {128}), tkn(dkn.p, DType::BF16, {128}), tp(dp.p, DType::I32, {t});
    auto oracle = [&](std::vector<float>& buf, const std::vector<float>& nw, int heads) {
        std::vector<double> out(buf.size());
        for (int tok = 0; tok < t; ++tok)
            for (int h = 0; h < heads; ++h) {
                const int base = (tok * heads + h) * 128;
                double ss = 0;
                for (int d = 0; d < 128; ++d) ss += double(buf[base + d]) * buf[base + d];
                const double inv = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
                for (int i = 0; i < 64; ++i) {
                    const double a = pos[tok] * std::pow(1e6, -2.0 * i / 128.0);
                    const double c = std::cos(a), s = std::sin(a);
                    const double n0 = buf[base + i] * inv * nw[i];
                    const double n1 = buf[base + i + 64] * inv * nw[i + 64];
                    out[base + i]      = n0 * c - n1 * s;
                    out[base + i + 64] = n1 * c + n0 * s;
                }
            }
        return out;
    };
    const auto rq = oracle(q, qn, 24), rk = oracle(k, kn, 8);
    graph_run(dev, [&] { ops::inkling2_dspark_rmsnorm_rope(tp, tqn, tkn, tq, tk, 1e6f, dev.stream); }, graph);
    int fail = verify_reduction("dspark rope q", from_device_bf16(dq.p, rq.size()), rq, kBF);
    fail += verify_reduction("dspark rope k", from_device_bf16(dk.p, rk.size()), rk, kBF);
    return fail;
}

int dspark_rnn_step_test(DeviceContext& dev, bool graph) {
    constexpr int V = 201024;
    auto w1 = values(256 * 64, 98, 0.25f);
    std::vector<float> full_w1(std::size_t(V) * 256, 0.0f);
    std::copy(w1.begin(), w1.end(), full_w1.begin() + 42 * 256);
    auto rw = values(768 * 5120, 99, 0.02f), rb = values(768, 100, 0.1f, false);
    auto hw = values(2304 * 256, 101, 0.05f), hb = values(2304, 102, 0.1f, false);
    auto hin = values(2304, 103, 0.5f), ch = values(2304, 104, 0.5f);
    auto st = values(256, 105, 0.5f, false);
    std::vector<int> tok{42};
    auto dt = to_device(tok), dw1 = to_device_bf16(full_w1);
    auto drw = to_device_bf16(rw), drb = to_device(rb);
    auto dhw = to_device_bf16(hw), dhb = to_device(hb);
    auto dhin = to_device_bf16(hin), dch = to_device_bf16(ch), dst = to_device(st);
    GuardedDeviceBuffer hout(2304 * 2);
    Tensor tt(dt.p, DType::I32, {1}), tw1(dw1.p, DType::BF16, {256, V});
    Tensor trw(drw.p, DType::BF16, {5120, 768}), trb(drb.p, DType::FP32, {768});
    Tensor thw(dhw.p, DType::BF16, {256, 2304}), thb(dhb.p, DType::FP32, {2304});
    Tensor thin(dhin.p, DType::BF16, {2304}), tch(dch.p, DType::BF16, {2304});
    Tensor tst(dst.p, DType::FP32, {256}), thout(hout.data(), DType::BF16, {2304});

    std::vector<double> x5120(5120);
    for (int i = 0; i < 256; ++i) x5120[i] = f32_to_bf16(st[i]) ? bf16_to_f32(f32_to_bf16(st[i])) : 0.0;
    for (int i = 0; i < 256; ++i) x5120[256 + i] = full_w1[42 * 256 + i];
    for (int i = 0; i < 2304; ++i) x5120[512 + i] = hin[i];
    for (int i = 0; i < 2304; ++i) x5120[2816 + i] = ch[i];
    double guz[768]{};
    for (int r = 0; r < 768; ++r) {
        double acc = rb[r];
        for (int k = 0; k < 5120; ++k) acc += rw[r * 5120 + k] * x5120[k];
        guz[r] = acc;
    }
    std::vector<double> exp_st(256), m(256), exp_h(2304);
    for (int i = 0; i < 256; ++i) {
        const double ut = std::tanh(guz[256 + i]);
        const double gs = 1.0 / (1.0 + std::exp(-guz[i]));
        exp_st[i]       = ut + gs * (st[i] - ut);
        m[i]            = std::tanh(guz[512 + i]);
    }
    for (int r = 0; r < 2304; ++r) {
        double acc = hb[r] + hin[r];
        for (int k = 0; k < 256; ++k) acc += hw[r * 256 + k] * m[k];
        exp_h[r] = acc;
    }
    WorkspaceArena ws(ops::inkling2_dspark_rnn_step_workspace_bytes());
    graph_run(dev, [&] {
        ops::inkling2_dspark_rnn_step(tt, tw1, trw, trb, thw, thb, thin, tch, tst, thout, ws, dev.stream);
    }, graph);
    int fail = check("dspark rnn h_out", hout, exp_h);
    auto got_st = from_device<float>(dst.p, 256);
    fail += verify_reduction("dspark rnn state", std::vector<double>(got_st.begin(), got_st.end()), exp_st, kFP);
    if (ws.used() != 0) ++fail;
    return fail;
}

struct OwnedQ4Weight {
    DeviceBuffer buf;
    Weight weight{};
    std::vector<double> decoded;
};

OwnedQ4Weight make_q4_weight(int n, int k, int seed) {
    OwnedQ4Weight out;
    const std::size_t qbytes = static_cast<std::size_t>(n) * k / 2U;
    const std::size_t sbytes = static_cast<std::size_t>(n) * (k / 64) * 2U;
    std::vector<std::uint8_t> raw(qbytes + sbytes, 0);
    out.decoded.resize(static_cast<std::size_t>(n) * k);
    for (int r = 0; r < n; ++r) {
        for (int g = 0; g < k / 64; ++g) {
            const float s_f32     = ((r + g + seed) % 7 + 1) * 0.03125f;
            const __half s_f16    = __float2half_rn(s_f32);
            const double s_d      = __half2float(s_f16);
            std::uint16_t s_bits  = 0;
            std::memcpy(&s_bits, &s_f16, 2);
            const std::size_t s_idx = (static_cast<std::size_t>(r) * (k / 64) + g) * 2U;
            raw[qbytes + s_idx]     = static_cast<std::uint8_t>(s_bits & 0xFF);
            raw[qbytes + s_idx + 1] = static_cast<std::uint8_t>(s_bits >> 8);
            for (int c = 0; c < 64; c += 2) {
                const int q0 = ((r * 13 + g * 5 + c + seed) % 15) - 7;
                const int q1 = ((r * 7 + g * 11 + c + 1 + seed) % 15) - 7;
                const std::uint8_t lo = static_cast<std::uint8_t>(q0 & 0x0F);
                const std::uint8_t hi = static_cast<std::uint8_t>(q1 & 0x0F);
                raw[(static_cast<std::size_t>(r) * k + g * 64 + c) / 2U] =
                    static_cast<std::uint8_t>(lo | (hi << 4));
                out.decoded[static_cast<std::size_t>(r) * k + g * 64 + c]     = q0 * s_d;
                out.decoded[static_cast<std::size_t>(r) * k + g * 64 + c + 1] = q1 * s_d;
            }
        }
    }
    out.buf = to_device(raw);
    out.weight.qtype = QType::Q4G64_F16S;
    out.weight.layout = QuantLayout::RowSplit;
    out.weight.scale_dtype = DType::FP16;
    out.weight.group = 64;
    out.weight.group_size = 64;
    out.weight.ndim = 2;
    out.weight.n = n;
    out.weight.k = k;
    out.weight.shape[0] = n;
    out.weight.shape[1] = k;
    out.weight.padded_shape[0] = n;
    out.weight.padded_shape[1] = k;
    out.weight.qdata = out.buf.p;
    out.weight.scales = static_cast<std::uint8_t*>(out.buf.p) + qbytes;
    out.weight.payload_bytes = qbytes + sbytes;
    return out;
}

int dflash2_kv_materialize_test(DeviceContext& dev, int w, int count, int base_pos, int slot,
                                bool graph) {
    auto ctx = values(2304 * w, 701 + w, 0.25f);
    std::vector<std::int32_t> pos(w), cnt{count}, sl{slot};
    for (int i = 0; i < w; ++i) pos[i] = base_pos + i;
    auto d_ctx = to_device_bf16(ctx);
    auto d_pos = to_device(pos);
    auto d_cnt = to_device(cnt);
    auto d_sl  = to_device(sl);

    LayoutBuilder lb;
    const auto kv_layout = plan_cyclic_kv_cache(lb, 5, 2048, 8, 128, 2);
    const std::size_t kv_bytes = lb.finish(256);
    DeviceBuffer kv_buf(kv_bytes);
    kv_buf.fill(0);
    CyclicKVCache kv(DeviceSpan{kv_buf.p, kv_bytes}, kv_layout);

    std::vector<OwnedQ4Weight> kw(5), vw(5);
    std::vector<std::vector<float>> kn_h(5);
    std::vector<DeviceBuffer> kn_d(5);
    std::array<ops::ContextKVMaterializeLayerView, 5> layers{};
    for (int l = 0; l < 5; ++l) {
        kw[l] = make_q4_weight(1024, 2304, 10 + l);
        vw[l] = make_q4_weight(1024, 2304, 20 + l);
        kn_h[l] = values(128, 30 + l, 0.5f);
        kn_d[l] = to_device_bf16(kn_h[l]);
        layers[l].key_weight = kw[l].weight;
        layers[l].value_weight = vw[l].weight;
        layers[l].key_norm_weight = Tensor(kn_d[l].p, DType::BF16, {128});
        layers[l].cache = kv.layer_view(l);
    }

    Tensor t_ctx(d_ctx.p, DType::BF16, {2304, w, 1});
    Tensor t_pos(d_pos.p, DType::I32, {w, 1});
    Tensor t_cnt(d_cnt.p, DType::I32, {1});
    Tensor t_sl(d_sl.p, DType::I32, {1});
    WorkspaceArena ws(ops::inkling2_dflash2_kv_materialize_workspace_bytes(w));
    ops::ContextKVMaterializeExecutionEnvelope env{0U, static_cast<std::uint32_t>(w)};
    graph_run(dev, [&] {
        ops::inkling2_dflash2_kv_materialize(t_ctx, t_pos, t_cnt, t_sl, layers, env, ws,
                                             dev.stream);
    }, graph);

    int fail = 0;
    for (int l = 0; l < 5; ++l) {
        const auto view = kv.layer_view(l);
        const std::size_t k_elems =
            128ULL * view.padded_capacity * 8U * static_cast<std::size_t>(view.lane_capacity);
        const auto got_k = from_device_bf16(view.k.data, k_elems);
        const auto got_v_raw = from_device<std::uint16_t>(view.v.data, k_elems);
        for (int i = 0; i < count; ++i) {
            const int ring = (pos[i]) & 2047;
            std::vector<double> k_raw(1024, 0.0), v_raw(1024, 0.0);
            for (int r = 0; r < 1024; ++r) {
                double sk = 0.0, sv = 0.0;
                for (int c = 0; c < 2304; ++c) {
                    const double x = ctx[static_cast<std::size_t>(i) * 2304 + c];
                    sk += kw[l].decoded[static_cast<std::size_t>(r) * 2304 + c] * x;
                    sv += vw[l].decoded[static_cast<std::size_t>(r) * 2304 + c] * x;
                }
                k_raw[r] = sk;
                v_raw[r] = sv;
            }
            for (int h = 0; h < 8; ++h) {
                double ss = 0.0;
                for (int d = 0; d < 128; ++d) ss += k_raw[h * 128 + d] * k_raw[h * 128 + d];
                const double inv = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
                std::vector<double> exp_k(128), act_k(128), exp_v(128), act_v(128);
                for (int d = 0; d < 64; ++d) {
                    const double a = static_cast<double>(pos[i]) *
                                     std::pow(10000.0, -2.0 * static_cast<double>(d) / 128.0);
                    const double co = std::cos(a), si = std::sin(a);
                    const double x0 = k_raw[h * 128 + d] * inv * kn_h[l][d];
                    const double x1 = k_raw[h * 128 + d + 64] * inv * kn_h[l][d + 64];
                    exp_k[d]      = x0 * co - x1 * si;
                    exp_k[d + 64] = x1 * co + x0 * si;
                }
                for (int d = 0; d < 128; ++d) {
                    const std::size_t idx =
                        static_cast<std::size_t>(d) +
                        128ULL * (ring + static_cast<std::size_t>(view.padded_capacity) *
                                             (h + 8ULL * slot));
                    act_k[d] = got_k[idx];
                    __half vh;
                    std::memcpy(&vh, &got_v_raw[idx], 2);
                    act_v[d] = __half2float(vh);
                    exp_v[d] = v_raw[h * 128 + d];
                }
                constexpr ReductionCriterion kBFComposed{4.0e-3, 1e-4, 8.0e-3};
                fail += verify_reduction("dflash2 kv K", act_k, exp_k, kBFComposed);
                fail += verify_reduction("dflash2 kv V", act_v, exp_v, kBFComposed);
                if (fail > 0) return fail;
            }
        }
    }
    return fail;
}

int dflash2_conv_test(DeviceContext& dev, int w, bool graph) {
    auto res_h = values(2304 * w, 801 + w, 0.5f);
    auto nw_h  = values(2304, 802 + w, 0.5f);
    auto bk_h  = values(2304 * 4, 803 + w, 0.25f);
    auto kp_h  = values(2304 * 576, 804 + w, 0.05f);
    auto po_h  = values(2304 * w, 805 + w, 0.5f);
    auto d_res = to_device_bf16(res_h);
    auto d_nw  = to_device_bf16(nw_h);
    auto d_bk  = to_device_bf16(bk_h);
    auto d_kp  = to_device_bf16(kp_h);
    auto d_po  = to_device_bf16(po_h);
    GuardedDeviceBuffer d_prep(2304ULL * w * 2U), d_delta(144ULL * 2U * w * 2U);
    Tensor t_res(d_res.p, DType::BF16, {2304, w});
    Tensor t_nw(d_nw.p, DType::BF16, {2304});
    Tensor t_bk(d_bk.p, DType::BF16, {2304, 2, 2});
    Tensor t_kp(d_kp.p, DType::BF16, {2304, 576});
    Tensor t_po(d_po.p, DType::BF16, {2304, w});
    Tensor t_prep(d_prep.data(), DType::BF16, {2304, w});
    Tensor t_delta(d_delta.data(), DType::BF16, {144, 2, w});

    std::vector<double> norm_d(2304 * w);
    for (int i = 0; i < w; ++i) {
        const double inv = invnorm(&res_h[i * 2304], 2304);
        for (int h = 0; h < 2304; ++h) {
            norm_d[i * 2304 + h] =
                static_cast<double>(__bfloat162float(__float2bfloat16_rn(
                    static_cast<float>(res_h[i * 2304 + h] * inv * nw_h[h]))));
        }
    }
    std::vector<double> d_mat(576 * w, 0.0);
    for (int i = 0; i < w; ++i)
        for (int r = 0; r < 576; ++r)
            for (int h = 0; h < 2304; ++h)
                d_mat[i * 576 + r] += kp_h[r * 2304 + h] * norm_d[i * 2304 + h];

    std::vector<double> exp_prep(2304 * w), exp_delta(144 * 2 * w), exp_res(2304 * w);
    for (int i = 0; i < w; ++i) {
        for (int g = 0; g < 144; ++g) {
            exp_delta[i * 288 + g]       = d_mat[i * 576 + 2 * 144 + g];
            exp_delta[i * 288 + 144 + g] = d_mat[i * 576 + 3 * 144 + g];
        }
        for (int h = 0; h < 2304; ++h) {
            const int g = h >> 4;
            double v = (bk_h[h] + d_mat[i * 576 + g]) * norm_d[i * 2304 + h];
            if (i > 0) v += (bk_h[2304 + h] + d_mat[i * 576 + 144 + g]) * norm_d[(i - 1) * 2304 + h];
            exp_prep[i * 2304 + h] = v;

            const double fd0 = __bfloat162float(__float2bfloat16_rn(
                static_cast<float>(d_mat[i * 576 + 2 * 144 + g])));
            const double fd1 = __bfloat162float(__float2bfloat16_rn(
                static_cast<float>(d_mat[i * 576 + 3 * 144 + g])));
            double r = res_h[i * 2304 + h] + (bk_h[2 * 2304 + h] + fd0) * po_h[i * 2304 + h];
            if (i > 0) r += (bk_h[3 * 2304 + h] + fd1) * po_h[(i - 1) * 2304 + h];
            exp_res[i * 2304 + h] = r;
        }
    }

    WorkspaceArena ws(ops::inkling2_dflash2_conv_prepare_workspace_bytes(w));
    graph_run(dev, [&] {
        ops::inkling2_dflash2_conv_prepare(t_res, t_nw, 1e-6f, t_bk, t_kp, t_prep, t_delta, ws,
                                           dev.stream);
        ops::inkling2_dflash2_conv_finish_add(t_po, t_bk, t_delta, t_res, dev.stream);
    }, graph);
    int fail = check("dflash2 conv_prepare", d_prep, exp_prep);
    fail += check("dflash2 conv_finish_delta", d_delta, exp_delta);
    fail += verify_reduction("dflash2 conv_finish_add", from_device_bf16(d_res.p, 2304 * w),
                             exp_res, kBF);
    return fail;
}

int dflash2_topk16_and_select_test(DeviceContext& dev, int k, float temp, bool graph) {
    auto mh_h  = values(2304 * k, 901 + k, 0.25f);
    auto hp_h  = values(2304 * 256, 902 + k, 0.05f);
    auto lg_h  = values(201024 * k, 903 + k, 5.0f);
    // Poison padded region [200058..201024) to verify exclusion.
    for (int i = 0; i < k; ++i)
        for (int v = 200058; v < 201024; ++v) lg_h[static_cast<std::size_t>(i) * 201024 + v] = 500.0f;
    // Plant ties on column 0.
    lg_h[100] = 20.0f;
    lg_h[50]  = 20.0f;

    auto pcb_h = values(256 * 201024, 904 + k, 0.1f);
    auto scb_h = values(256 * 201024, 905 + k, 0.1f);
    std::vector<std::int32_t> anc_h{42}, bpos_h{1000};
    ops::SamplingConfig cfg{};
    cfg.temperature = temp;
    cfg.seed        = 12345ULL;

    auto d_mh  = to_device_bf16(mh_h);
    auto d_lg  = to_device_bf16(lg_h);
    auto d_hp  = to_device_bf16(hp_h);
    auto d_anc = to_device(anc_h);
    auto d_pcb = to_device_bf16(pcb_h);
    auto d_scb = to_device_bf16(scb_h);
    auto d_bp  = to_device(bpos_h);
    auto d_cfg = to_device(std::vector<ops::SamplingConfig>{cfg});
    GuardedDeviceBuffer d_cids(16ULL * k * 4U), d_drafts(k * 4ULL), d_q(16ULL * k * 4U);

    Tensor t_mh(d_mh.p, DType::BF16, {2304, k});
    Tensor t_lg(d_lg.p, DType::BF16, {201024, k});
    Tensor t_hp(d_hp.p, DType::BF16, {2304, 256});
    Tensor t_anc(d_anc.p, DType::I32, {1});
    Tensor t_pcb(d_pcb.p, DType::BF16, {256, 201024});
    Tensor t_scb(d_scb.p, DType::BF16, {256, 201024});
    Tensor t_bp(d_bp.p, DType::I32, {1});
    Tensor t_cids(d_cids.data(), DType::I32, {16, k, 1});
    Tensor t_drafts(d_drafts.data(), DType::I32, {k, 1});
    Tensor t_q(d_q.data(), DType::FP32, {16, k, 1});

    WorkspaceArena ws(ops::inkling2_dflash2_topk16_and_select_workspace_bytes(k));
    graph_run(dev, [&] {
        ops::inkling2_dflash2_topk16_and_select(
            t_mh, t_lg, t_hp, t_anc, t_pcb, t_scb, t_bp,
            static_cast<const ops::SamplingConfig*>(d_cfg.p), t_cids, t_drafts, t_q, ws,
            dev.stream);
    }, graph);

    std::vector<std::int32_t> exp_cids(16 * k), exp_drafts(k);
    std::vector<double> exp_q(16 * k, 0.0);
    int pred = anc_h[0];
    for (int i = 0; i < k; ++i) {
        std::vector<std::pair<float, int>> items(200058);
        for (int v = 0; v < 200058; ++v)
            items[v] = {lg_h[static_cast<std::size_t>(i) * 201024 + v], v};
        std::partial_sort(items.begin(), items.begin() + 16, items.end(),
                          [](const auto& a, const auto& b) {
                              return a.first > b.first || (a.first == b.first && a.second < b.second);
                          });
        std::vector<double> sel_h(256, 0.0);
        for (int r = 0; r < 256; ++r) {
            double s = 0.0;
            for (int h = 0; h < 2304; ++h) s += hp_h[r * 2304 + h] * mh_h[i * 2304 + h];
            sel_h[r] = __bfloat162float(__float2bfloat16_rn(static_cast<float>(s)));
        }
        std::vector<double> edges(16);
        double max_e = -1e30;
        for (int c = 0; c < 16; ++c) {
            exp_cids[i * 16 + c] = items[c].second;
            double corr          = 0.0;
            for (int r = 0; r < 256; ++r)
                corr += pcb_h[static_cast<std::size_t>(pred) * 256 + r] * sel_h[r] *
                        scb_h[static_cast<std::size_t>(items[c].second) * 256 + r];
            edges[c] = 30.0 * std::tanh((items[c].first + corr) / 30.0);
            max_e    = std::max(max_e, edges[c]);
        }
        int best_c = 0;
        if (temp <= 0.0f) {
            for (int c = 1; c < 16; ++c)
                if (edges[c] > edges[best_c]) best_c = c;
            exp_q[i * 16 + best_c] = 1.0;
        } else {
            double sum_w = 0.0;
            for (int c = 0; c < 16; ++c) {
                exp_q[i * 16 + c] = std::exp((edges[c] - max_e) / temp);
                sum_w += exp_q[i * 16 + c];
            }
            for (int c = 0; c < 16; ++c) exp_q[i * 16 + c] /= sum_w;
            auto splitmix64 = [](std::uint64_t x) {
                x += 0x9E3779B97F4A7C15ull;
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
                x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
                return x ^ (x >> 31);
            };
            std::uint64_t key = cfg.seed;
            key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(bpos_h[0] + i)) * 0xD1B54A32D192ED03ull));
            key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ops::kSamplePurposeDFlash2Proposal)) << 21));
            const float u = static_cast<float>(static_cast<std::uint32_t>(key >> 40)) * (1.0f / 16777216.0f);
            double cdf = 0.0;
            best_c     = 15;
            for (int c = 0; c < 16; ++c) {
                cdf += exp_q[i * 16 + c];
                if (u < cdf) {
                    best_c = c;
                    break;
                }
            }
        }
        exp_drafts[i] = exp_cids[i * 16 + best_c];
        pred          = exp_drafts[i];
    }

    int fail = verify_exact("dflash2 topk16 candidate_ids",
                            from_device<std::int32_t>(d_cids.data(), 16 * k), exp_cids);
    fail += verify_exact("dflash2 drafts", from_device<std::int32_t>(d_drafts.data(), k),
                         exp_drafts);
    fail += check("dflash2 proposal_q", d_q, exp_q, true, &kFPMma);
    return fail;
}
} // namespace

int main() {
    try {
        if (cuda_unavailable()) return 77;
        DeviceContext device;
        int failures = 0;
        for (int t : {1, 3, 8, 15, 16, 17, 64, 129}) for (int n : {1, 24, 32, 64})
            for (bool fp32 : {false, true}) failures += projection(device, t, n, fp32, t != 1);
        for (int t : {1, 2, 3, 8, 15, 17, 1024}) for (int c : {1024, 2304})
            for (bool alias : {false, true}) {
                failures += convolution(device, c, t, 0, alias, true, t != 3);
                if (c == 1024) for (int dt : {1, 2}) failures += convolution(device, c, t, dt, alias, false, true);
            }
        for (int m : {1, 2, 5, 8}) for (int r : {1, 3, 4}) for (bool norm : {false, true})
            failures += mixing(device, 3, m, r, norm, true);
        failures += mixing(device, 1, 5, 4, true, false, 0);
        for (int t : {3, 8, 15, 16, 17, 64, 129})
            for (int c : {1, 3}) for (bool norm : {false, true}) for (bool alias : {false, true})
                failures += control(device, t, c, norm, true, alias);
        for (int t : {1, 2, 15, 16, 64})
            for (bool norm : {false, true})
                failures += control(device, t, 3, norm, true, false, true);
        failures += control(device, 1, 1, false, false, false);
        for (bool alias : {false, true}) for (bool tau : {false, true})
            for (float scale : {0.f, 1e-5f, 5.f}) failures += clipping(device, 4, tau, alias, true, scale);
        failures += clipping(device, 1, false, false, false);
        for (bool padded : {false, true}) failures += softcap(device, 3, padded, false, true);
        failures += softcap(device, 1, true, true, false);
        failures += dspark_feature_mix_test(device, 1, false);
        failures += dspark_feature_mix_test(device, 8, true);
        failures += dspark_rmsnorm_rope_test(device, 8, true);
        failures += dspark_rnn_step_test(device, false);
        for (int w : {1, 8, 16}) failures += dflash2_kv_materialize_test(device, w, w, 2043, 1, w != 1);
        failures += dflash2_kv_materialize_test(device, 8, 5, 2046, 0, true);
        for (int w : {2, 8, 16}) failures += dflash2_conv_test(device, w, w != 2);
        for (int k : {1, 7, 15}) {
            failures += dflash2_topk16_and_select_test(device, k, 0.0f, k != 1);
            failures += dflash2_topk16_and_select_test(device, k, 0.8f, true);
        }
        std::cout << (failures ? "FAIL" : "PASS") << " Inkling2 fused ops (" << failures << " failures)\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
