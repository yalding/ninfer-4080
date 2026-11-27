#include "targets/inkling2_mini/impl/frontend/processor.h"

#include "media/decode/decode.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <list>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace ninfer::targets::inkling2_mini::frontend_internal {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kNativePatchSize = 20;
constexpr std::uint32_t kMergeSize       = 2;
constexpr std::uint32_t kTilePixels      = kNativePatchSize * kMergeSize;           // 40
constexpr std::uint32_t kPatchDim        = kNativePatchSize * kNativePatchSize * 3; // 1200
constexpr std::int32_t kPosembGrid       = 32;
constexpr float kPiF                     = 3.14159265358979323846F;
constexpr std::uint16_t kPadPixelBf16 =
    0xBF81U; // BF16((-1.0f / 255.0f - 0.5f) / 0.5f) = -1.0078125f < -1.0f

constexpr std::array<std::uint32_t, 64> kShaRound{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

std::uint32_t load_be32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24U) | (static_cast<std::uint32_t>(p[1]) << 16U) |
           (static_cast<std::uint32_t>(p[2]) << 8U) | static_cast<std::uint32_t>(p[3]);
}

void store_be32(std::uint32_t v, std::uint8_t* p) {
    p[0] = static_cast<std::uint8_t>(v >> 24U);
    p[1] = static_cast<std::uint8_t>(v >> 16U);
    p[2] = static_cast<std::uint8_t>(v >> 8U);
    p[3] = static_cast<std::uint8_t>(v);
}

void sha_block(std::array<std::uint32_t, 8>& s, const std::uint8_t* blk) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) { w[i] = load_be32(blk + 4 * i); }
    for (std::size_t i = 16; i < 64; ++i) {
        const auto s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
        const auto s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
        w[i]          = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = s;
    for (std::size_t i = 0; i < 64; ++i) {
        const auto s1  = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const auto ch  = (e & f) ^ (~e & g);
        const auto t1  = h + s1 + ch + kShaRound[i] + w[i];
        const auto s0  = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const auto maj = (a & b) ^ (a & c) ^ (b & c);
        const auto t2  = s0 + maj;
        h              = g;
        g              = f;
        f              = e;
        e              = d + t1;
        d              = c;
        c              = b;
        b              = a;
        a              = t1 + t2;
    }
    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
    s[5] += f;
    s[6] += g;
    s[7] += h;
}

void check_control(const PreparationControl& control) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "media preprocessing cancelled");
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::Cancelled, "media preprocessing deadline exceeded");
    }
}

std::uint16_t f32_to_bf16_rne(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    }
    const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float lanczos_sinc_f32(float x) noexcept {
    if (x == 0.0F) { return 1.0F; }
    const float pix = kPiF * x;
    return std::sin(pix) / pix;
}

struct LanczosFilter {
    std::vector<int> offsets;
    std::vector<int> counts;
    std::vector<int> src_indices;
    std::vector<float> weights;
};

LanczosFilter build_lanczos3_filter(int src_len, int dst_len) {
    LanczosFilter f;
    f.offsets.resize(static_cast<std::size_t>(dst_len));
    f.counts.resize(static_cast<std::size_t>(dst_len));
    constexpr float a  = 3.0F;
    const float scale  = static_cast<float>(dst_len) / static_cast<float>(src_len);
    const float radius = scale < 1.0F ? (a / scale) : a;

    for (int out_idx = 0; out_idx < dst_len; ++out_idx) {
        const float center = (static_cast<float>(out_idx) + 0.5F) / scale - 0.5F;
        const int left     = std::max(0, static_cast<int>(std::floor(center - radius)));
        const int right    = std::min(src_len - 1, static_cast<int>(std::ceil(center + radius)));
        f.offsets[static_cast<std::size_t>(out_idx)] = static_cast<int>(f.src_indices.size());
        float sum                                    = 0.0F;
        for (int s = left; s <= right; ++s) {
            const float dx = scale < 1.0F ? (static_cast<float>(s) - center) * scale
                                          : (static_cast<float>(s) - center);
            if (std::abs(dx) < a) {
                const float w = lanczos_sinc_f32(dx) * lanczos_sinc_f32(dx / a);
                f.src_indices.push_back(s);
                f.weights.push_back(w);
                sum += w;
            }
        }
        const int cnt =
            static_cast<int>(f.src_indices.size()) - f.offsets[static_cast<std::size_t>(out_idx)];
        f.counts[static_cast<std::size_t>(out_idx)] = cnt;
        if (sum > 0.0F) {
            const float inv = 1.0F / sum;
            for (int k = 0; k < cnt; ++k) {
                f.weights[static_cast<std::size_t>(f.offsets[out_idx] + k)] *= inv;
            }
        }
    }
    return f;
}

std::vector<std::uint8_t> resize_lanczos3_rgb8(std::span<const std::uint8_t> src, int src_w,
                                               int src_h, int dst_w, int dst_h,
                                               const PreparationControl& control) {
    const LanczosFilter fx = build_lanczos3_filter(src_w, dst_w);
    const LanczosFilter fy = build_lanczos3_filter(src_h, dst_h);
    std::vector<float> horiz(static_cast<std::size_t>(src_h) * dst_w * 3, 0.0F);
    for (int y = 0; y < src_h; ++y) {
        check_control(control);
        const std::uint8_t* row = src.data() + static_cast<std::size_t>(y) * src_w * 3;
        float* out_row          = horiz.data() + static_cast<std::size_t>(y) * dst_w * 3;
        for (int x = 0; x < dst_w; ++x) {
            const int off = fx.offsets[static_cast<std::size_t>(x)];
            const int cnt = fx.counts[static_cast<std::size_t>(x)];
            float r = 0.0F, g = 0.0F, b = 0.0F;
            for (int k = 0; k < cnt; ++k) {
                const int sx  = fx.src_indices[static_cast<std::size_t>(off + k)];
                const float w = fx.weights[static_cast<std::size_t>(off + k)];
                r += static_cast<float>(row[sx * 3 + 0]) * w;
                g += static_cast<float>(row[sx * 3 + 1]) * w;
                b += static_cast<float>(row[sx * 3 + 2]) * w;
            }
            out_row[x * 3 + 0] = r;
            out_row[x * 3 + 1] = g;
            out_row[x * 3 + 2] = b;
        }
    }

    std::vector<std::uint8_t> dst(static_cast<std::size_t>(dst_h) * dst_w * 3);
    for (int y = 0; y < dst_h; ++y) {
        check_control(control);
        const int off         = fy.offsets[static_cast<std::size_t>(y)];
        const int cnt         = fy.counts[static_cast<std::size_t>(y)];
        std::uint8_t* out_row = dst.data() + static_cast<std::size_t>(y) * dst_w * 3;
        for (int x = 0; x < dst_w; ++x) {
            float r = 0.0F, g = 0.0F, b = 0.0F;
            for (int k = 0; k < cnt; ++k) {
                const int sy   = fy.src_indices[static_cast<std::size_t>(off + k)];
                const float w  = fy.weights[static_cast<std::size_t>(off + k)];
                const float* p = horiz.data() + (static_cast<std::size_t>(sy) * dst_w + x) * 3;
                r += p[0] * w;
                g += p[1] * w;
                b += p[2] * w;
            }
            out_row[x * 3 + 0] =
                static_cast<std::uint8_t>(std::clamp(std::floor(r + 0.5F), 0.0F, 255.0F));
            out_row[x * 3 + 1] =
                static_cast<std::uint8_t>(std::clamp(std::floor(g + 0.5F), 0.0F, 255.0F));
            out_row[x * 3 + 2] =
                static_cast<std::uint8_t>(std::clamp(std::floor(b + 0.5F), 0.0F, 255.0F));
        }
    }
    return dst;
}

struct CacheKey {
    std::array<std::uint8_t, 32> digest{};
    ImageResizePolicy resize_policy = ImageResizePolicy::Downsize;

    bool operator==(const CacheKey&) const noexcept = default;
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const noexcept {
        std::size_t h = 0;
        for (std::uint8_t b : k.digest) {
            h ^= static_cast<std::size_t>(b) + 0x9e3779b9U + (h << 6U) + (h >> 2U);
        }
        return h ^ static_cast<std::size_t>(k.resize_policy);
    }
};

struct Flight {
    std::mutex mutex;
    std::condition_variable cv;
    PreprocessedImage result;
    std::exception_ptr error;
    std::size_t bytes = 0;
    bool done         = false;
};

struct CacheEntry {
    CacheKey key;
    PreprocessedImage value;
    std::size_t bytes = 0;
};

PreprocessedImage decode_and_pack_image(
    const OwnedMedia& media, const std::array<std::uint8_t, 32>& digest, std::uint32_t max_patches,
    std::uint32_t max_merged_tokens, const PreparationControl& control,
    const std::function<std::shared_ptr<PreparedMediaPayload>(std::size_t)>& reserve_payload) {
    media::decode::Policy policy;
    policy.max_bytes  = kMaximumPromptMediaBytes;
    policy.checkpoint = [&] { check_control(control); };

    media::decode::Image decoded;
    try {
        decoded = media::decode::decode_image(media.bytes, policy);
    } catch (const media::decode::Error& err) {
        if (err.kind() == media::decode::ErrorKind::BudgetExceeded) {
            throw RequestError(RequestErrorKind::MediaBudgetExceeded, err.what());
        }
        throw RequestError(RequestErrorKind::InvalidMedia, err.what());
    }
    if (decoded.width <= 0 || decoded.height <= 0 ||
        decoded.rgb.size() != static_cast<std::size_t>(decoded.width) * decoded.height * 3) {
        throw RequestError(RequestErrorKind::InvalidMedia, "decoded image has invalid dimensions");
    }

    int target_w          = decoded.width;
    int target_h          = decoded.height;
    const auto tile_count = [](int w, int h, int edge) {
        return static_cast<std::uint64_t>(1 + (w - 1) / edge) * (1 + (h - 1) / edge);
    };
    const auto fits = [&](int w, int h) {
        return tile_count(w, h, kNativePatchSize) <= max_patches &&
               tile_count(w, h, kTilePixels) <= max_merged_tokens;
    };

    if (!fits(target_w, target_h)) {
        if (media.image_resize_policy == ImageResizePolicy::RejectOversized) {
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                               "image exceeds Inkling-2-Mini vision token budget");
        }
        if (!fits(1, 1)) {
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                               "image cannot fit within Inkling-2-Mini vision token budget");
        }
        const int long_edge = std::max(decoded.width, decoded.height);
        int low             = 1;
        int high            = long_edge;
        while (high - low > 1) {
            const int mid = low + (high - low) / 2;
            const int w =
                std::max(1, static_cast<int>(std::int64_t{decoded.width} * mid / long_edge));
            const int h =
                std::max(1, static_cast<int>(std::int64_t{decoded.height} * mid / long_edge));
            if (fits(w, h))
                low = mid;
            else
                high = mid;
        }
        target_w = std::max(1, static_cast<int>(std::int64_t{decoded.width} * low / long_edge));
        target_h = std::max(1, static_cast<int>(std::int64_t{decoded.height} * low / long_edge));
    }

    std::vector<std::uint8_t> resized;
    std::span<const std::uint8_t> rgb = decoded.rgb;
    if (target_w != decoded.width || target_h != decoded.height) {
        resized = resize_lanczos3_rgb8(decoded.rgb, decoded.width, decoded.height, target_w,
                                       target_h, control);
        rgb     = resized;
    }

    const std::int32_t grid_h =
        (target_h + static_cast<int>(kTilePixels) - 1) / static_cast<int>(kTilePixels);
    const std::int32_t grid_w =
        (target_w + static_cast<int>(kTilePixels) - 1) / static_cast<int>(kTilePixels);
    const std::int32_t h20 = grid_h * static_cast<int>(kMergeSize);
    const std::int32_t w20 = grid_w * static_cast<int>(kMergeSize);
    const std::int32_t active_h20 =
        (target_h + static_cast<int>(kNativePatchSize) - 1) / static_cast<int>(kNativePatchSize);
    const std::int32_t active_w20 =
        (target_w + static_cast<int>(kNativePatchSize) - 1) / static_cast<int>(kNativePatchSize);
    const std::uint32_t native_patches = static_cast<std::uint32_t>(active_h20 * active_w20);
    const std::uint32_t merged_tokens  = static_cast<std::uint32_t>(grid_h * grid_w);

    const std::size_t patch_bytes =
        kPatchDim * sizeof(std::uint16_t) + 8 * sizeof(std::int32_t) + 4 * sizeof(float);
    const std::size_t bytes =
        static_cast<std::size_t>(native_patches) * patch_bytes +
        (static_cast<std::size_t>(merged_tokens) + 1) * 4 * sizeof(std::int32_t);
    auto payload            = reserve_payload(bytes);
    payload->native_patches = native_patches;
    payload->merged_tokens  = merged_tokens;
    payload->padded_h20     = h20;
    payload->padded_w20     = w20;
    payload->patches.resize(static_cast<std::size_t>(native_patches) * kPatchDim, kPadPixelBf16);
    payload->coords.resize(static_cast<std::size_t>(native_patches + 1) * 4);
    payload->pos_indices.resize(static_cast<std::size_t>(native_patches) * 4);
    payload->pos_weights.resize(static_cast<std::size_t>(native_patches) * 4);
    payload->merge_indices.assign(static_cast<std::size_t>(merged_tokens) * 4, -1);

    // Row 0 is [CLS].
    payload->coords[0] = -1;
    payload->coords[1] = -1;
    payload->coords[2] = h20;
    payload->coords[3] = w20;

    std::vector<std::int32_t> grid_to_patch(static_cast<std::size_t>(h20) * w20, -1);
    std::uint32_t p_idx = 0;
    for (std::int32_t y = 0; y < active_h20; ++y) {
        check_control(control);
        for (std::int32_t x = 0; x < active_w20; ++x) {
            grid_to_patch[static_cast<std::size_t>(y) * w20 + x] = static_cast<std::int32_t>(p_idx);
            const std::size_t t_row    = static_cast<std::size_t>(p_idx + 1) * 4;
            payload->coords[t_row + 0] = y;
            payload->coords[t_row + 1] = x;
            payload->coords[t_row + 2] = h20;
            payload->coords[t_row + 3] = w20;

            const float yf =
                std::clamp((static_cast<float>(y) + 0.5F) *
                                   (static_cast<float>(kPosembGrid) / static_cast<float>(h20)) -
                               0.5F,
                           0.0F, static_cast<float>(kPosembGrid - 1));
            const float xf =
                std::clamp((static_cast<float>(x) + 0.5F) *
                                   (static_cast<float>(kPosembGrid) / static_cast<float>(w20)) -
                               0.5F,
                           0.0F, static_cast<float>(kPosembGrid - 1));
            const int y0                 = static_cast<int>(std::floor(yf));
            const int x0                 = static_cast<int>(std::floor(xf));
            const int y1                 = std::min(y0 + 1, kPosembGrid - 1);
            const int x1                 = std::min(x0 + 1, kPosembGrid - 1);
            const float wy               = yf - static_cast<float>(y0);
            const float wx               = xf - static_cast<float>(x0);
            const std::size_t p4         = static_cast<std::size_t>(p_idx) * 4;
            payload->pos_indices[p4 + 0] = y0 * kPosembGrid + x0;
            payload->pos_indices[p4 + 1] = y0 * kPosembGrid + x1;
            payload->pos_indices[p4 + 2] = y1 * kPosembGrid + x0;
            payload->pos_indices[p4 + 3] = y1 * kPosembGrid + x1;
            payload->pos_weights[p4 + 0] = (1.0F - wy) * (1.0F - wx);
            payload->pos_weights[p4 + 1] = (1.0F - wy) * wx;
            payload->pos_weights[p4 + 2] = wy * (1.0F - wx);
            payload->pos_weights[p4 + 3] = wy * wx;

            std::uint16_t* dst =
                payload->patches.data() + static_cast<std::size_t>(p_idx) * kPatchDim;
            const int py0    = y * static_cast<int>(kNativePatchSize);
            const int px0    = x * static_cast<int>(kNativePatchSize);
            const int copy_h = std::min(static_cast<int>(kNativePatchSize), target_h - py0);
            const int copy_w = std::min(static_cast<int>(kNativePatchSize), target_w - px0);
            for (int dy = 0; dy < copy_h; ++dy) {
                const std::uint8_t* src_px =
                    rgb.data() + (static_cast<std::size_t>(py0 + dy) * target_w + px0) * 3;
                std::uint16_t* dst_px = dst + static_cast<std::size_t>(dy) * kNativePatchSize * 3;
                for (int dx = 0; dx < copy_w; ++dx) {
                    for (int c = 0; c < 3; ++c) {
                        const float norm =
                            static_cast<float>(src_px[dx * 3 + c]) * (2.0F / 255.0F) - 1.0F;
                        dst_px[dx * 3 + c] = f32_to_bf16_rne(norm);
                    }
                }
            }
            ++p_idx;
        }
    }

    for (std::int32_t r = 0; r < grid_h; ++r) {
        for (std::int32_t c = 0; c < grid_w; ++c) {
            const std::size_t v4     = static_cast<std::size_t>(r * grid_w + c) * 4;
            const int coords_q[4][2] = {
                {2 * r, 2 * c},
                {2 * r, 2 * c + 1},
                {2 * r + 1, 2 * c},
                {2 * r + 1, 2 * c + 1},
            };
            for (int q = 0; q < 4; ++q) {
                const int gp =
                    grid_to_patch[static_cast<std::size_t>(coords_q[q][0]) * w20 + coords_q[q][1]];
                payload->merge_indices[v4 + q] = gp >= 0 ? (1 + gp) : -1;
            }
        }
    }

    return PreprocessedImage{
        .grid_h         = grid_h,
        .grid_w         = grid_w,
        .content_digest = digest,
        .payload        = std::move(payload),
    };
}

} // namespace

std::array<std::uint8_t, 32> sha256_bytes(std::span<const std::uint8_t> input,
                                          const std::function<void()>& checkpoint) {
    std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::size_t offset = 0;
    while (input.size() - offset >= 64) {
        if (checkpoint && (offset % (1ULL << 20)) == 0) { checkpoint(); }
        sha_block(state, input.data() + offset);
        offset += 64;
    }
    std::array<std::uint8_t, 128> tail{};
    const std::size_t rem = input.size() - offset;
    std::copy_n(input.data() + offset, rem, tail.data());
    tail[rem]                   = 0x80U;
    const std::size_t tail_size = rem < 56 ? 64 : 128;
    const std::uint64_t bits    = static_cast<std::uint64_t>(input.size()) * 8ULL;
    for (std::size_t i = 0; i < 8; ++i) {
        tail[tail_size - 1 - i] = static_cast<std::uint8_t>(bits >> (8U * i));
    }
    sha_block(state, tail.data());
    if (tail_size == 128) { sha_block(state, tail.data() + 64); }
    std::array<std::uint8_t, 32> out{};
    for (std::size_t i = 0; i < 8; ++i) { store_be32(state[i], out.data() + 4 * i); }
    return out;
}

struct MediaPreprocessCache::Impl {
    struct LivePayload : PreparedMediaPayload {
        std::shared_ptr<std::atomic_size_t> live_bytes;
        std::size_t reserved = 0;

        ~LivePayload() {
            if (reserved != 0) live_bytes->fetch_sub(reserved);
        }
    };

    explicit Impl(std::uint32_t worker_count) : threads(worker_count), workers(worker_count) {}

    std::size_t capacity_bytes      = 0;
    std::size_t live_capacity_bytes = 0;
    std::uint32_t threads;
    std::counting_semaphore<> workers;
    std::shared_ptr<std::atomic_size_t> live_bytes = std::make_shared<std::atomic_size_t>(0);
    std::atomic_size_t queued_tasks{0}, active_tasks{0};
    std::uint32_t max_patches       = 0;
    std::uint32_t max_merged_tokens = 0;
    mutable std::mutex mutex;
    std::list<CacheEntry> lru;
    std::unordered_map<CacheKey, std::list<CacheEntry>::iterator, CacheKeyHash> index;
    std::unordered_map<CacheKey, std::shared_ptr<Flight>, CacheKeyHash> inflight;
    std::size_t retained_bytes       = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;

    void evict_oldest() {
        retained_bytes -= lru.back().bytes;
        index.erase(lru.back().key);
        lru.pop_back();
        ++evictions;
    }

    std::shared_ptr<PreparedMediaPayload> reserve_payload(std::size_t bytes) {
        if (bytes > live_capacity_bytes) {
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                               "image payload exceeds the live media budget");
        }
        auto payload = std::make_shared<LivePayload>();
        std::lock_guard lock(mutex);
        while (live_bytes->load() > live_capacity_bytes - bytes && !lru.empty()) evict_oldest();
        if (live_bytes->load() > live_capacity_bytes - bytes) {
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                               "live media payload budget exhausted");
        }
        live_bytes->fetch_add(bytes);
        payload->live_bytes = live_bytes;
        payload->reserved   = bytes;
        return payload;
    }
};

MediaPreprocessCache::MediaPreprocessCache(std::size_t capacity_bytes,
                                           std::size_t live_capacity_bytes,
                                           std::uint32_t preprocess_threads,
                                           std::uint32_t max_patches,
                                           std::uint32_t max_merged_tokens)
    : impl_(std::make_unique<Impl>(
          preprocess_threads != 0 ? preprocess_threads
                                  : std::clamp(std::thread::hardware_concurrency(), 1U, 16U))) {
    impl_->capacity_bytes      = capacity_bytes;
    impl_->live_capacity_bytes = live_capacity_bytes;
    impl_->max_patches         = max_patches;
    impl_->max_merged_tokens   = max_merged_tokens;
}

MediaPreprocessCache::~MediaPreprocessCache() = default;

PreprocessedImage MediaPreprocessCache::prepare_image(const OwnedMedia& media,
                                                      const PreparationControl& control,
                                                      MediaRequestMetrics& metrics) {
    check_control(control);
    const auto digest = sha256_bytes(media.bytes, [&] { check_control(control); });
    const CacheKey key{digest, media.image_resize_policy};

    std::shared_ptr<Flight> flight;
    for (;;) {
        check_control(control);
        {
            std::lock_guard lock(impl_->mutex);
            if (auto it = impl_->index.find(key); it != impl_->index.end()) {
                impl_->lru.splice(impl_->lru.begin(), impl_->lru, it->second);
                ++impl_->hits;
                ++metrics.hits;
                metrics.reused_patch_bytes += it->second->bytes;
                return it->second->value;
            }
            const auto fit = impl_->inflight.find(key);
            if (fit == impl_->inflight.end()) {
                flight = std::make_shared<Flight>();
                impl_->inflight.emplace(key, flight);
                ++impl_->misses;
                ++metrics.misses;
                break;
            }
            flight = fit->second;
            ++impl_->singleflight_waits;
            ++metrics.singleflight_waits;
        }

        std::unique_lock lock(flight->mutex);
        while (!flight->cv.wait_for(lock, std::chrono::milliseconds(10),
                                    [&] { return flight->done; })) {
            check_control(control);
        }
        check_control(control);
        if (flight->error) {
            try {
                std::rethrow_exception(flight->error);
            } catch (const RequestError& error) {
                if (error.kind() == RequestErrorKind::Cancelled) continue;
                throw;
            }
        }
        metrics.reused_patch_bytes += flight->bytes;
        return flight->result;
    }

    PreprocessedImage built;
    std::exception_ptr err;
    auto t0       = Clock::now();
    bool acquired = false;
    ++impl_->queued_tasks;
    try {
        while (!impl_->workers.try_acquire_for(std::chrono::milliseconds(10)))
            check_control(control);
        acquired = true;
        --impl_->queued_tasks;
        ++impl_->active_tasks;
        t0 = Clock::now();
        check_control(control);
        built = decode_and_pack_image(
            media, digest, impl_->max_patches, impl_->max_merged_tokens, control,
            [&](std::size_t bytes) { return impl_->reserve_payload(bytes); });
    } catch (...) { err = std::current_exception(); }
    if (acquired) {
        --impl_->active_tasks;
        impl_->workers.release();
    } else {
        --impl_->queued_tasks;
    }
    const double elapsed    = std::chrono::duration<double>(Clock::now() - t0).count();
    const std::size_t bytes = err ? 0 : built.payload->payload_bytes();

    {
        std::lock_guard lock(impl_->mutex);
        impl_->inflight.erase(key);
        if (!err) {
            if (bytes > impl_->capacity_bytes) {
                ++impl_->oversize_bypasses;
            } else {
                while (impl_->retained_bytes > impl_->capacity_bytes - bytes) impl_->evict_oldest();
                try {
                    const auto entry = impl_->index.try_emplace(key).first;
                    try {
                        impl_->lru.push_front(CacheEntry{key, built, bytes});
                    } catch (...) {
                        impl_->index.erase(entry);
                        throw;
                    }
                    entry->second = impl_->lru.begin();
                    impl_->retained_bytes += bytes;
                } catch (...) { err = std::current_exception(); }
            }
        }
    }

    {
        std::lock_guard flock(flight->mutex);
        flight->result = built;
        flight->error  = err;
        flight->bytes  = bytes;
        flight->done   = true;
    }
    flight->cv.notify_all();
    if (err) { std::rethrow_exception(err); }

    metrics.built_patch_bytes += bytes;
    metrics.build_seconds += elapsed;
    return built;
}

MediaCacheSummary MediaPreprocessCache::summary() const {
    std::lock_guard lock(impl_->mutex);
    return MediaCacheSummary{
        .capacity_bytes      = impl_->capacity_bytes,
        .live_capacity_bytes = impl_->live_capacity_bytes,
        .retained_bytes      = impl_->retained_bytes,
        .live_bytes          = impl_->live_bytes->load(),
        .entries             = impl_->lru.size(),
        .inflight            = impl_->inflight.size(),
        .queued_tasks        = impl_->queued_tasks.load(),
        .active_tasks        = impl_->active_tasks.load(),
        .preprocess_threads  = impl_->threads,
        .hits                = impl_->hits,
        .misses              = impl_->misses,
        .singleflight_waits  = impl_->singleflight_waits,
        .evictions           = impl_->evictions,
        .oversize_bypasses   = impl_->oversize_bypasses,
    };
}

} // namespace ninfer::targets::inkling2_mini::frontend_internal
