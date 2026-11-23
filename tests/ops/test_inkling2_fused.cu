#include "ninfer/ops/inkling2_fused.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

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
        std::cout << (failures ? "FAIL" : "PASS") << " Inkling2 fused ops (" << failures << " failures)\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
