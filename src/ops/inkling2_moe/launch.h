#pragma once
#include "ninfer/ops/inkling2_moe.h"

namespace ninfer::ops::inkling2_detail {
void moe_launch(const Tensor&, const Inkling2MoeWeights&, Tensor&, float*, int*, float*, float*,
                cudaStream_t);
}
