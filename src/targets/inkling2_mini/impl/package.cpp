#include <ninfer/targets/inkling2_mini/package.h>
#include <ninfer/targets/inkling2_mini/frontend_resources.h>
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/inkling2_mini/impl/load/bindings.h"
#include <stdexcept>
#include <utility>

namespace ninfer::targets::inkling2_mini::detail {

class LoadPlan::Impl {
public:
    Impl(ArtifactLoadPlan target_plan, EngineOptions engine_options)
        : value(std::move(target_plan)), options(std::move(engine_options)) {}

    ArtifactLoadPlan value;
    EngineOptions options;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->value.materialization;
}

class LoadedModel::Impl {
public:
    Impl(ArtifactLoadPlan target_plan, EngineOptions engine_options,
         artifact::MaterializedArtifact&& artifact)
        : backing(std::move(artifact)), view(materialize_model(backing, target_plan.bindings)),
          resources(take_frontend_resources(backing, target_plan.bindings.frontend)),
          options(std::move(engine_options)) {}

    artifact::MaterializedArtifact backing;
    RuntimeModelView view;
    FrontendResources resources;
    EngineOptions options;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::inkling2_mini::detail

namespace ninfer::targets::inkling2_mini {

ModelSamplingDefaults Package::sampling_defaults(std::string_view) {
    ModelSamplingDefaults out;
    out.thinking.temperature = out.non_thinking.temperature = 1;
    out.thinking.top_p = out.non_thinking.top_p = 1;
    out.thinking.top_k = out.non_thinking.top_k = 0;
    out.thinking.min_p = out.non_thinking.min_p = 0;
    return out;
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& id) {
    if (id.model_id != model_id || id.weights_id != "q3g128")
        throw std::invalid_argument("Inkling-2-Mini requires inkling2-mini/q3g128");
    return WeightsProfile::Q3G128;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile) {
    if (options.enable_vision)
        throw std::invalid_argument("Inkling-2-Mini supports text-only execution");
    const auto backend = options.speculative.backend;
    if (backend != SpeculativeBackend::None && backend != SpeculativeBackend::DFlash &&
        backend != SpeculativeBackend::Mtp && backend != SpeculativeBackend::DFlash2)
        throw std::invalid_argument(
            "Inkling-2-Mini supports DSpark (dflash/mtp) and DFlash 2 (dflash2) speculative decoding");
    if (backend == SpeculativeBackend::DFlash2) {
        if (options.speculative.draft_tokens > 15)
            throw std::invalid_argument("Inkling-2-Mini DFlash 2 supports draft_tokens in [0..15]");
        if (options.speculative.proposal_head == ProposalHead::Optimized)
            throw std::invalid_argument("Inkling-2-Mini DFlash 2 does not support --lm-head-draft");
    } else if (backend != SpeculativeBackend::None && options.speculative.draft_tokens > 7) {
        throw std::invalid_argument("Inkling-2-Mini DSpark supports draft_tokens in [0..7]");
    }
    const bool load_mtp =
        backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::Mtp;
    const bool load_draft_head =
        load_mtp && options.speculative.proposal_head == ProposalHead::Optimized;
    const bool load_dflash2 = backend == SpeculativeBackend::DFlash2;
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        detail::bind_artifact(binder, load_mtp, load_draft_head, load_dflash2), options));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& artifact) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        std::move(plan.impl_->value), std::move(plan.impl_->options), std::move(artifact));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return inkling2_mini::make_frontend(model.impl_->resources, {options.max_context});
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext&, const EngineOptions& options,
                                                        WeightsProfile) {
    return SequencePlanner(options);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model,
                                                          SequencePlan&& plan,
                                                          DeviceContext& device,
                                                          const StartupObserver&) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    auto options = std::move(plan.options);
    if (options.speculative.backend == SpeculativeBackend::DFlash2) {
        if (!model.impl_->view.has_dflash2)
            throw std::invalid_argument(
                "Inkling-2-Mini --spec dflash2 requires an artifact with dflash2/ weights");
        if (options.speculative.draft_tokens == 0) options.speculative.draft_tokens = 7;
    } else if (options.speculative.backend != SpeculativeBackend::None) {
        if (!model.impl_->view.has_mtp)
            throw std::invalid_argument(
                "Inkling-2-Mini speculative decoding requires an artifact with mtp/ DSpark weights");
        if (options.speculative.proposal_head == ProposalHead::Optimized &&
            !model.impl_->view.has_draft_head)
            throw std::invalid_argument(
                "Inkling-2-Mini --lm-head-draft requires an artifact with mtp/draft_head");
        if (options.speculative.draft_tokens == 0) options.speculative.draft_tokens = 2;
    }
    return std::make_unique<Program>(model.impl_->view, device, std::move(options));
}

} // namespace ninfer::targets::inkling2_mini
