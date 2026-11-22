#pragma once
#include <ninfer/targets/inkling2_mini/frontend.h>
#include "targets/inkling2_mini/impl/frontend/tokenizer.h"

namespace ninfer::targets::inkling2_mini {
OutputSession make_output_session(std::shared_ptr<const frontend_internal::Tokenizer> tokenizer,
                                  const StopPolicy& policy, const OutputOptions& options,
                                  bool continuation = false, bool reasoning = false);
}
