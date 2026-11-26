#pragma once
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/inkling2_mini/prepared_prompt.h>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <variant>

namespace ninfer {
struct DeviceContext;
struct StartupObserver;
}

namespace ninfer::targets::inkling2_mini {
namespace detail {
struct RuntimeModelView;
}
class Program;

struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier = 0, identity_tag = 0;
    friend bool operator==(const PrefixShortlistKey&, const PrefixShortlistKey&) = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier = 0, backend_frontier = 0, main_pages = 0, backend_pages = 0;
    friend bool operator==(const TargetKVRequirement&, const TargetKVRequirement&) = default;
};

struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;
    friend bool operator==(const CheckpointSummary&, const CheckpointSummary&) = default;
};

struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint, rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references                                                = 0;
    friend bool operator==(const ContinuationSummary&, const ContinuationSummary&) = default;
};

struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references                                                = 0;
    friend bool operator==(const SharedPrefixSummary&, const SharedPrefixSummary&) = default;
};

struct PhysicalUsageSnapshot {
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t device_state_slots = 0, host_state_slots = 0, device_main_kv_pages = 0,
                  device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;
};

struct SequencePlan {
    EngineOptions options;
    std::size_t reservation = 0;

    std::uint32_t capacity() const noexcept { return options.max_context; }

    std::uint32_t kv_capacity() const noexcept { return options.kv_capacity.explicit_tokens; }

    std::size_t device_reservation_bytes() const noexcept { return reservation; }
};

class SequencePlanner {
public:
    SequencePlanner(EngineOptions options);

    const runtime::SequenceCapacityCurve& capacity_curve() const noexcept { return curve_; }

    SequencePlan finalize(std::uint32_t groups) &&;
private:
    EngineOptions options_;
    runtime::SequenceCapacityCurve curve_;
};

struct RequestBasePlan {
    runtime::RequestPlanSummary value;
    runtime::ResolvedExecutionOptions options;
    PreparedContextCache cache;
    std::vector<TokenId> ids;

    const runtime::RequestPlanSummary& summary() const noexcept { return value; }

    const PreparedContextCache& context_cache() const noexcept { return cache; }

    std::optional<PrefixShortlistKey> prefix_shortlist_key(std::uint32_t frontier) const noexcept;
    std::optional<runtime::PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept;
};

struct SequenceHandle {
    std::uint64_t id                                       = 0;
    friend bool operator==(SequenceHandle, SequenceHandle) = default;
};

struct ContinuationHandle {
    std::uint64_t id = 0;
};

struct SharedPrefixHandle {
    std::uint64_t id = 0;
};

struct CaptureOffer {
    std::uint64_t id = 0;
};

struct AdmissionCandidate {
    runtime::RequestPlanSummary value;
    runtime::ResolvedExecutionOptions options;
    runtime::IdentityMaterializationAssessment identity;
    runtime::LaneId lane;
    std::uint64_t source = 0;
    bool shared = false, host = false;
    std::uint32_t reused = 0;
    std::vector<runtime::ContextTransferRequirement> transfers;
    std::optional<runtime::CheckpointRef> selected_checkpoint;

    const runtime::RequestPlanSummary& summary() const noexcept { return value; }

    const runtime::IdentityMaterializationAssessment& identity_assessment() const noexcept {
        return identity;
    }
};

struct ResourcePlan {
    AdmissionCandidate candidate;
    runtime::ProgramResourceRevision revision;
    std::vector<std::pair<runtime::PlanningOwnerId, std::uint64_t>> victims, shared_victims;
    std::vector<std::uint32_t> capture_frontiers;

    const runtime::RequestPlanSummary& summary() const noexcept { return candidate.value; }

    bool needs_transfer() const noexcept { return candidate.host; }

    runtime::ProgramResourceRevision resource_revision() const noexcept { return revision; }
};

struct PersistentBackfillProof {
    runtime::ProgramResourceRevision revision;

    runtime::ProgramResourceRevision resource_revision() const noexcept { return revision; }
};

struct CaptureAssessment {
    PrefixShortlistKey shortlist_key;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
    runtime::PrefillWork protected_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> projected_recovery_work;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    bool publishes_private = false, publishes_shared = false, needs_transfer = false,
         physically_feasible = false;
    std::uint64_t source = 0, replacement = 0;
};

struct CaptureVictimAction {
    runtime::PlanningOwnerId owner;
    std::uint64_t sequence_id = 0;
    bool shared               = false;
    bool spill_to_host        = false;
};

struct CapturePressurePlan {
    runtime::ProgramResourceRevision revision;
    std::optional<CaptureVictimAction> victim;

    runtime::ProgramResourceRevision resource_revision() const noexcept { return revision; }
};

struct PressureTargetHandle {
    std::uint32_t candidate                                            = 0;
    bool maximal                                                       = false;
    friend bool operator==(PressureTargetHandle, PressureTargetHandle) = default;
};

struct AssessedPressureTarget {
    runtime::PressureTargetAssessment value;
    PressureTargetHandle handle;
    std::vector<runtime::PressureOwnerOutcome> outcomes;
    std::vector<runtime::PressureCheckpointRecoveryImpact> impacts;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> recovery;

    const runtime::PressureTargetAssessment& assessment() const noexcept { return value; }
};

struct PreparedPressureExpansion {
    std::uint32_t count = 0;

    std::uint32_t new_canonical_count() const noexcept { return count; }
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
    bool complete                     = true;
};

struct PressureConstructionCursor {
    PressureTargetHandle handle;
};

class PressurePlanningSession {
public:
    Program* program = nullptr;
    std::vector<const AdmissionCandidate*> candidates;
    std::vector<runtime::PlanningCandidateId> ids;
    std::vector<std::pair<runtime::PlanningOwnerId, std::uint64_t>> owners, shared_owners;
    PressureTargetHandle identity_target(runtime::PlanningCandidateId id) const;
    PressureTargetHandle root_maximal_target(runtime::PlanningCandidateId id) const;

    PressureTargetHandle maximal_target(runtime::PlanningCandidateId id) const {
        return root_maximal_target(id);
    }

    PressureConstructionCursor begin_construction(PressureTargetHandle t, bool = false) {
        return {t};
    }

    runtime::PressureConstructionStep next_construction_option(PressureConstructionCursor&) {
        return {.exhausted = true};
    }

    void choose_construction(PressureConstructionCursor&, runtime::PressureConstructionOptionId) {}

    std::optional<PressureTargetHandle> construction_target(const PressureConstructionCursor& c) {
        return c.handle;
    }

    runtime::PressureTargetGuidance guidance(PressureTargetHandle t);
    AssessedPressureTarget assess(PressureTargetHandle t);

    PreparedPressureExpansion prepare_expansion(PressureTargetHandle, std::uint32_t = ~0u) {
        return {};
    }

    PressureExpansionView commit_expansion(PreparedPressureExpansion&&) { return {}; }

    void discard_expansion(PreparedPressureExpansion&&) noexcept {}

    runtime::PrefillWork shared_capture_split_prefill_work(const AssessedPressureTarget&,
                                                           const PreparedPrompt&,
                                                           std::span<const std::uint32_t>) const;
    std::optional<ResourcePlan> seal(AssessedPressureTarget&&, const PreparedPrompt&,
                                     runtime::FinalScheduleIntent);

    std::optional<CapturePressurePlan> seal_capture(AssessedPressureTarget&&) {
        return std::nullopt;
    }
private:
    std::vector<runtime::PressureOwnerOutcome> outcomes_;
};

class CapturePressurePlanningSession {
public:
    Program* program = nullptr;
    CaptureAssessment capture;
    runtime::ProgramResourceRevision revision;
    std::vector<std::pair<runtime::PlanningOwnerId, std::uint64_t>> owners, shared_owners;

    static constexpr runtime::PlanningCandidateId candidate_id() noexcept { return {.value = 0}; }

    PressureTargetHandle identity_target() const { return {}; }

    runtime::PressureTargetGuidance guidance(PressureTargetHandle t);
    AssessedPressureTarget assess(PressureTargetHandle t);
    PreparedPressureExpansion prepare_expansion(PressureTargetHandle t);
    PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    std::optional<CapturePressurePlan> seal(AssessedPressureTarget&& assessed);

private:
    std::vector<CaptureVictimAction> children_actions_;
    std::vector<PressureTargetHandle> child_handles_;
    std::vector<runtime::PressureOwnerOutcome> guidance_outcomes_;
    bool expanded_      = false;
    bool scratch_ready_ = false;
};

struct PendingBatch {
    std::array<SequenceHandle, kMaximumConcurrency> rows{};
    std::vector<TokenId> values;
    std::vector<std::int32_t> counts;
    runtime::ExecutionTiming timing;
    std::uint64_t transaction = 0;
    std::uint32_t stride      = 1;

    std::size_t row_count() const noexcept { return counts.size(); }

    std::span<const TokenId> tokens() const noexcept { return values; }

    std::span<const std::int32_t> row_counts() const noexcept { return counts; }

    std::uint32_t row_stride() const noexcept { return stride; }

    runtime::ExecutionTiming execution_timing() const noexcept { return timing; }
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationSourceResult {
    runtime::PrivateSourceMode mode = runtime::PrivateSourceMode::Retain;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedSourceResult {
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult, ActiveCaptureResult>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

class Program {
public:
    Program(const detail::RuntimeModelView&, DeviceContext&, EngineOptions);
    Program(const detail::RuntimeModelView&, DeviceContext&, EngineOptions,
            const StartupObserver*);
    ~Program();
    RequestBasePlan plan_request(const PreparedPrompt&, const runtime::ResolvedExecutionOptions&);
    std::vector<float> causal_score(PreparedPrompt&&, std::uint32_t);
    std::optional<AdmissionCandidate>
    inspect_admission(const PreparedPrompt&, const RequestBasePlan&, runtime::LaneId,
                      const ContinuationHandle*, const SharedPrefixHandle*,
                      std::optional<runtime::CheckpointRef>, bool);
    std::optional<ResourcePlan> seal_identity(const AdmissionCandidate&, const PreparedPrompt&,
                                              runtime::FinalScheduleIntent);
    PressurePlanningSession begin_pressure_planning(std::span<const AdmissionCandidate* const>,
                                                    std::span<const runtime::PlanningCandidateId>,
                                                    std::span<const ContinuationHandle* const>,
                                                    std::span<const runtime::PlanningOwnerId>,
                                                    std::span<const SharedPrefixHandle* const>,
                                                    std::span<const runtime::PlanningOwnerId>);
    runtime::PrefillWork shared_capture_split_prefill_work(const AdmissionCandidate&,
                                                           const PreparedPrompt&,
                                                           std::span<const std::uint32_t>);
    runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&&, PreparedPrompt&&, runtime::CancellationFlagView);
    std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan&, const ResourcePlan&,
                              std::span<const SequenceHandle>) const;
    ContextTransactionProgress progress_context_transaction(runtime::CancellationFlagView);
    void finalize_context_transaction() noexcept;
    bool has_context_transaction() const noexcept;

    bool has_unsettled_state_fork() const noexcept { return false; }

    PrefillProgress advance_prefill(SequenceHandle, runtime::ExecutionTiming* = nullptr);
    CaptureAssessment inspect_capture(const CaptureOffer&, const SharedPrefixHandle*,
                                      const SharedPrefixHandle*,
                                      std::optional<runtime::CheckpointRef>, bool) const;
    std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle&, runtime::CheckpointRef) const;
    std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle&, runtime::CheckpointRef) const;
    CapturePressurePlanningSession begin_capture_pressure_planning(
        const CaptureAssessment&, std::span<const ContinuationHandle* const>,
        std::span<const runtime::PlanningOwnerId>, std::span<const SharedPrefixHandle* const>,
        std::span<const runtime::PlanningOwnerId>);
    bool shared_capture_matches(const CaptureOffer&, const SharedPrefixHandle&) const;
    void skip_capture(CaptureOffer&&);
    runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&&, const SharedPrefixHandle*, const SharedPrefixHandle*,
                           std::optional<runtime::CheckpointRef>, bool,
                           runtime::CancellationFlagView);
    runtime::ContextTransactionReserveStatus
    reserve_active_capture_with_pressure(CaptureOffer&&, const SharedPrefixHandle*,
                                         const SharedPrefixHandle*,
                                         std::optional<runtime::CheckpointRef>, bool,
                                         CapturePressurePlan&&, runtime::CancellationFlagView);
    PendingBatch decode(std::span<const SequenceHandle>, std::span<const runtime::RoundBudget>,
                        runtime::ExecutionTiming* = nullptr);
    runtime::ExecutionTiming append_forced_tokens(std::span<const SequenceHandle>,
                                                  std::span<const TokenId>, std::uint32_t,
                                                  std::span<const std::optional<std::uint32_t>>,
                                                  runtime::ExecutionTiming* = nullptr);
    CommitResult commit(PendingBatch&&, std::span<const runtime::CommitDecision>,
                        runtime::CommitObservation = runtime::CommitObservation::AllRows,
                        runtime::ExecutionTiming*  = nullptr);
    DiscardResult abort_pending(PendingBatch&&) noexcept;
    FinishResult finish(SequenceHandle) noexcept;
    AbortResult abort(SequenceHandle) noexcept;
    ReleaseResult release_continuation(ContinuationHandle&&) noexcept;
    ReleaseResult release_shared_prefix(SharedPrefixHandle&&) noexcept;
    void fail_all_cleanup() noexcept;
    std::uint32_t continuation_depth(const ContinuationHandle&) const noexcept;
    std::string continuation_digest(const ContinuationHandle&) const;
    std::vector<SlotCheckpoint> continuation_checkpoints(const ContinuationHandle&) const;
    ContinuationSummary continuation_summary(const ContinuationHandle&) const;
    runtime::SessionSnapshotTraffic session_snapshot_traffic() const noexcept;
    bool isolated_request_feasible(const RequestBasePlan&) const noexcept;
    runtime::ProgramResourceRevision resource_revision() const noexcept;
    PhysicalUsageSnapshot physical_usage() const noexcept;
    MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    bool physically_feasible(const AdmissionCandidate&, std::span<const std::uint64_t>) const;
    friend class PressurePlanningSession;
    friend class CapturePressurePlanningSession;
};
} // namespace ninfer::targets::inkling2_mini
