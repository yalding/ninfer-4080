#pragma once

#include "ninfer/ops/inkling2_fused.h"

namespace ninfer::ops::detail {
struct Inkling2Branches {
    const void* states[8]{};
    const void* norm[4]{};
    int count = 0;
};
void inkling2_proj_launch(const Tensor&, const Tensor&, Tensor&, cudaStream_t);
void inkling2_sconv_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&,
                          const Tensor&, Tensor&, cudaStream_t);
void inkling2_mix_launch(Inkling2Branches, const Tensor&, Tensor&, cudaStream_t);
void inkling2_control_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            Inkling2Branches, Tensor&, float*, float*, float*, cudaStream_t);
void inkling2_clip_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, bool, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t);
void inkling2_softcap_launch(const Tensor&, float, int, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
