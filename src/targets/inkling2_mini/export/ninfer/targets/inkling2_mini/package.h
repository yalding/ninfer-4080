#pragma once

#include "ninfer/types.h"
#include <ninfer/targets/inkling2_mini/frontend.h>
#include <ninfer/targets/inkling2_mini/runtime.h>

#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::inkling2_mini {

struct Package;

namespace detail {

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct inkling2_mini::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct inkling2_mini::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id = "inkling2-mini", target_key = "inkling2_mini";
    enum class WeightsProfile { Q3G128 };
    using LoadPlan                   = detail::LoadPlan;
    using LoadedModel                = detail::LoadedModel;
    using Frontend                   = inkling2_mini::Frontend;
    using PreparedPrompt             = inkling2_mini::PreparedPrompt;
    using OutputSession              = inkling2_mini::OutputSession;
    using PublishedOutput            = inkling2_mini::PublishedOutput;
    using SequencePlanner            = inkling2_mini::SequencePlanner;
    using SequencePlan               = inkling2_mini::SequencePlan;
    using RequestBasePlan            = inkling2_mini::RequestBasePlan;
    using AdmissionCandidate         = inkling2_mini::AdmissionCandidate;
    using ResourcePlan               = inkling2_mini::ResourcePlan;
    using PersistentBackfillProof    = inkling2_mini::PersistentBackfillProof;
    using SequenceHandle             = inkling2_mini::SequenceHandle;
    using ContinuationHandle         = inkling2_mini::ContinuationHandle;
    using SharedPrefixHandle         = inkling2_mini::SharedPrefixHandle;
    using CaptureOffer               = inkling2_mini::CaptureOffer;
    using CacheSessionKey            = PreparedSessionKey;
    using ContinuationSummary        = inkling2_mini::ContinuationSummary;
    using SharedPrefixSummary        = inkling2_mini::SharedPrefixSummary;
    using PressurePlanningSession    = inkling2_mini::PressurePlanningSession;
    using PressureTargetHandle       = inkling2_mini::PressureTargetHandle;
    using AssessedPressureTarget     = inkling2_mini::AssessedPressureTarget;
    using CapturePressurePlan        = inkling2_mini::CapturePressurePlan;
    using MaterializationResult      = inkling2_mini::MaterializationResult;
    using ContextTransactionProgress = inkling2_mini::ContextTransactionProgress;
    using CaptureAssessment          = inkling2_mini::CaptureAssessment;
    using ActiveCaptureResult        = inkling2_mini::ActiveCaptureResult;
    using PendingBatch               = inkling2_mini::PendingBatch;
    using StartResult                = inkling2_mini::StartResult;
    using PrefillProgress            = inkling2_mini::PrefillProgress;
    using CommitResult               = inkling2_mini::CommitResult;
    using DiscardResult              = inkling2_mini::DiscardResult;
    using FinishResult               = inkling2_mini::FinishResult;
    using AbortResult                = inkling2_mini::AbortResult;
    using ReleaseResult              = inkling2_mini::ReleaseResult;
    using Program                    = inkling2_mini::Program;
    static ModelSamplingDefaults sampling_defaults(std::string_view);
    static WeightsProfile resolve_weights(const artifact::ArtifactIdentity&);
    static LoadPlan plan_load(artifact::Binder&, const EngineOptions&, WeightsProfile);
    static std::unique_ptr<LoadedModel> construct_loaded_model(LoadPlan&&,
                                                               artifact::MaterializedArtifact&&);
    static Frontend make_frontend(const LoadedModel&, const EngineOptions&);
    static SequencePlanner make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                 WeightsProfile);
    static std::unique_ptr<Program> create_program(const LoadedModel&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

} // namespace targets::inkling2_mini
} // namespace ninfer
