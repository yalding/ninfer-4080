#include <ninfer/targets/inkling2_mini/frontend_resources.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <string>

namespace ninfer::targets::inkling2_mini {
namespace {

std::string take_string(artifact::MaterializedArtifact& materialized,
                        artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    FrontendResourcePlan plan{
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json =
            artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
    };
    if (binder.has_object("frontend/processor_config.json")) {
        plan.processor_config_json =
            artifact::bind_raw_resource(binder, "frontend/processor_config.json");
    }
    return plan;
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json         = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json  = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja    = take_string(materialized, plan.chat_template_jinja),
        .generation_config_json = take_string(materialized, plan.generation_config_json),
        .processor_config_json  = plan.processor_config_json
                                      ? take_string(materialized, *plan.processor_config_json)
                                      : std::string{},
    };
}

} // namespace ninfer::targets::inkling2_mini
