#include "targets/inkling2_mini/impl/runtime/program.h"
#include "targets/inkling2_mini/impl/runtime/executor.h"
#include "targets/inkling2_mini/impl/state/state_image.h"
#include "runtime/ngram_pool.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>

namespace ninfer::targets::inkling2_mini {
namespace {
std::uint32_t pages(std::uint32_t n) { return (n + 63) / 64; }

PrefixShortlistKey shortlist(std::span<const TokenId> ids, std::uint32_t frontier) noexcept {
    PrefixShortlistKey out;
    out.frontier = frontier;
    out.digests  = {1469598103934665603ull, 1099511628211ull};
    for (std::uint32_t i = 0; i < frontier; ++i)
        for (int b = 0; b < 4; ++b) {
            auto v         = (static_cast<std::uint32_t>(ids[i]) >> (8 * b)) & 255;
            out.digests[0] = (out.digests[0] ^ v) * 1099511628211ull;
            out.digests[1] = (out.digests[1] + v) * 0x9e3779b185ebca87ull;
        }
    return out;
}

std::string digest(std::span<const TokenId> ids) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx",
                  static_cast<unsigned long long>(shortlist(ids, ids.size()).digests[0]));
    return text;
}

void coalesce_transfer(runtime::CoalescedTransferWork& dst,
                       runtime::ContextTransferDirection direction, TransferWork work) noexcept {
    const auto idx = static_cast<std::size_t>(direction);
    if (idx >= dst.size() || (work.payload_bytes == 0 && work.copy_operations == 0)) return;
    dst[idx].payload_bytes =
        work.payload_bytes > std::numeric_limits<std::uint64_t>::max() - dst[idx].payload_bytes
            ? std::numeric_limits<std::uint64_t>::max()
            : dst[idx].payload_bytes + work.payload_bytes;
    const std::uint64_t ops =
        static_cast<std::uint64_t>(dst[idx].copy_operations) + work.copy_operations;
    dst[idx].copy_operations = ops > std::numeric_limits<std::uint32_t>::max()
                                   ? std::numeric_limits<std::uint32_t>::max()
                                   : static_cast<std::uint32_t>(ops);
}

std::uint32_t count_checkpoints(const ContinuationSummary& s) noexcept {
    return static_cast<std::uint32_t>(s.endpoint.has_value() + s.rewrite.has_value() +
                                      s.long_anchors.size());
}
} // namespace

SequencePlanner::SequencePlanner(EngineOptions options) : options_(std::move(options)) {
    if (!options_.max_context || options_.max_context > 131072)
        throw std::invalid_argument("Inkling-2-Mini max_context must be in [1,131072]");
    curve_.main_page_tokens                 = 64;
    curve_.minimum_main_page_groups         = 1;
    auto explicit_cap                       = options_.kv_capacity.mode == KvCapacityMode::Explicit
                                                  ? options_.kv_capacity.explicit_tokens
                                                  : options_.max_context * options_.max_concurrency;
    curve_.maximum_main_page_groups         = std::max(1u, pages(explicit_cap));
    curve_.minimum_device_reservation_bytes = detail::executor_device_bytes(options_, 64);
    curve_.bytes_per_additional_main_page_group =
        detail::executor_device_bytes(options_, 128) - curve_.minimum_device_reservation_bytes;
}

SequencePlan SequencePlanner::finalize(std::uint32_t groups) && {
    if (groups < curve_.minimum_main_page_groups || groups > curve_.maximum_main_page_groups)
        throw std::invalid_argument("Inkling-2 capacity outside plan");
    options_.kv_capacity = KvCapacityPolicy::explicit_capacity(groups * 64);
    return {std::move(options_), curve_.reservation_bytes(groups)};
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (!options.allow_prefix_reuse || frontier == 0 || frontier >= ids.size()) return std::nullopt;
    return shortlist(ids, frontier);
}

std::optional<runtime::PrefillWork>
RequestBasePlan::shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
    if (frontier == 0 || frontier > ids.size()) return std::nullopt;
    return runtime::make_prefill_work(0, frontier, 0, 0, 1024);
}

class Program::Impl {
public:
    struct Checkpoint {
        CheckpointSummary summary;
        std::shared_ptr<detail::HostStateImage> image;
    };

    struct Sequence {
        std::uint64_t id   = 0;
        std::uint32_t slot = 0;
        runtime::LaneId lane;
        bool active = true, publish = true;
        std::vector<TokenId> prompt, ledger, generated;
        runtime::ResolvedExecutionOptions options;
        runtime::BeginSummary begin;
        std::uint32_t progress = 0, entitlement = 0;
        GenerationTimings timings;
        SpeculativeStats speculative;
        bool shared = false;
        std::vector<std::uint32_t> captures, shared_captures;
        PreparedContextCache cache;
        PromptIdentity identity;
        std::vector<Checkpoint> checkpoints;
        std::shared_ptr<detail::HostStateImage> host;
    };

    Impl(const detail::RuntimeModelView& model, DeviceContext& device, EngineOptions options)
        : options(std::move(options)), executor(model, device, this->options),
          slots(detail::state_slot_count(this->options), 0) {
        if (this->options.speculative.backend == SpeculativeBackend::Mtp &&
            this->options.speculative.ngram.mode == NgramDraftMode::Chain) {
            const std::size_t entries = std::max<std::size_t>(
                1, this->options.speculative.ngram.pool_bytes / sizeof(std::uint32_t));
            ngram_pool = std::make_unique<runtime::NgramDraftPool>(
                runtime::NgramPoolSpec{this->options.speculative.ngram.match_tokens, entries, 200058});
        }
    }

    EngineOptions options;
    detail::Executor executor;
    std::unique_ptr<runtime::NgramDraftPool> ngram_pool;
    std::vector<std::uint64_t> slots;
    std::map<std::uint64_t, Sequence> sequences;
    std::uint64_t next = 1, revision = 1, pending = 0;
    std::optional<ResourcePlan> transaction;
    std::optional<PreparedPrompt> transaction_prompt;
    struct CaptureTransaction {
        std::uint64_t offer_id       = 0;
        std::uint64_t replacement_id = 0;
        std::optional<CaptureVictimAction> victim;
    };
    std::optional<CaptureTransaction> capture_transaction;
    runtime::SessionSnapshotTraffic traffic;

    void observe_external() {
        for (const auto& transfer : executor.take_transfer_observations()) {
            const bool state = transfer.resource == runtime::ContextResourceClass::State;
            switch (transfer.direction) {
            case runtime::ContextTransferDirection::DeviceToHost:
                if (state) {
                    traffic.state_d2h_count += transfer.units;
                    traffic.state_d2h_bytes += transfer.work.payload_bytes;
                } else {
                    traffic.main_kv_d2h_pages += transfer.page_count;
                    traffic.main_kv_d2h_bytes += transfer.work.payload_bytes;
                }
                break;
            case runtime::ContextTransferDirection::HostToDevice:
                if (state) {
                    traffic.state_h2d_count += transfer.units;
                    traffic.state_h2d_bytes += transfer.work.payload_bytes;
                } else {
                    traffic.main_kv_h2d_pages += transfer.page_count;
                    traffic.main_kv_h2d_bytes += transfer.work.payload_bytes;
                }
                break;
            case runtime::ContextTransferDirection::DeviceToDevice:
                if (state) {
                    traffic.state_d2d_count += transfer.units;
                    traffic.state_d2d_bytes += transfer.work.payload_bytes;
                } else {
                    traffic.main_kv_d2d_pages += transfer.page_count;
                    traffic.main_kv_d2d_bytes += transfer.work.payload_bytes;
                }
                break;
            }
        }
    }

    Sequence& get(SequenceHandle h) {
        auto it = sequences.find(h.id);
        if (it == sequences.end() || !it->second.active)
            throw std::logic_error("invalid Inkling-2 sequence capability");
        return it->second;
    }

    void erase(std::uint64_t id) {
        auto it = sequences.find(id);
        if (it == sequences.end()) return;
        if (!it->second.host) {
            executor.reset(it->second.slot);
            slots[it->second.slot] = 0;
        }
        sequences.erase(it);
        ++revision;
    }

    std::uint32_t depth(const Sequence& s) const {
        return s.host ? s.host->depth : executor.slot_depth(s.slot);
    }

    void spill(Sequence& s) {
        auto image = executor.snapshot_to_host(s.slot);
        executor.reset(s.slot);
        slots.at(s.slot) = 0;
        s.slot           = ~0u;
        s.host           = std::move(image);
        ++revision;
    }

    ContinuationSummary summary(const Sequence& s) const {
        const auto depth = this->depth(s);
        ContinuationSummary out;
        if (depth) {
            CheckpointSummary cp;
            cp.ref = {runtime::CheckpointKind::SessionEndpoint, depth, 0};
            if (s.shared) {
                cp.ref.kind = runtime::CheckpointKind::SharedStablePrefix;
                cp.scope    = runtime::CheckpointScope::Shared;
            }
            cp.shortlist_key             = shortlist(s.ledger, depth);
            cp.state_residency           = s.host ? runtime::ReplicaResidency::HostOnly
                                                  : runtime::ReplicaResidency::DeviceOnly;
            cp.required_kv.main_frontier = depth;
            cp.required_kv.main_pages    = pages(depth);
            cp.rebuild_work = runtime::make_prefill_work(0, depth, 0, 0, options.prefill_chunk);
            out.endpoint    = cp;
        }
        for (const auto& checkpoint : s.checkpoints) {
            if (checkpoint.summary.ref.kind == runtime::CheckpointKind::LongAnchor)
                out.long_anchors.push_back(checkpoint.summary);
            else
                out.rewrite = checkpoint.summary;
        }
        return out;
    }

    void capture_private(Sequence& s) {
        const auto frontier = executor.slot_depth(s.slot);
        if (!s.publish || !frontier) return;
        bool rewrite =
            s.identity.rewrite_checkpoint && s.identity.rewrite_checkpoint->frontier == frontier;
        auto marker = std::find_if(
            s.cache.opportunities.begin(), s.cache.opportunities.end(), [&](const auto& p) {
                return p.kind == PromptCacheMarkerKind::PrivateLongAnchor && p.frontier == frontier;
            });
        bool anchor = marker != s.cache.opportunities.end() &&
                      options.context_cache.max_long_anchors_per_continuation.value_or(0) > 0;
        if ((!rewrite && !anchor) || !executor.can_snapshot_to_host(s.slot)) return;
        Checkpoint cp;
        cp.image                   = executor.snapshot_to_host(s.slot);
        cp.summary                 = *summary(s).endpoint;
        cp.summary.state_residency = runtime::ReplicaResidency::HostOnly;
        if (rewrite) {
            cp.summary.ref.kind =
                s.identity.rewrite_checkpoint->kind == RewriteCheckpointKind::TurnClosure
                    ? runtime::CheckpointKind::TurnClosure
                    : runtime::CheckpointKind::ResponseReplay;
        } else {
            cp.summary.ref.kind   = runtime::CheckpointKind::LongAnchor;
            std::uint32_t ordinal = 1;
            for (const auto& previous : s.checkpoints)
                if (previous.summary.ref.kind == runtime::CheckpointKind::LongAnchor)
                    ordinal = std::max(ordinal, previous.summary.ref.ordinal + 1);
            if (ordinal > options.context_cache.max_long_anchors_per_continuation.value_or(0)) {
                auto oldest =
                    std::find_if(s.checkpoints.begin(), s.checkpoints.end(), [](const auto& p) {
                        return p.summary.ref.kind == runtime::CheckpointKind::LongAnchor;
                    });
                if (oldest == s.checkpoints.end()) return;
                ordinal = oldest->summary.ref.ordinal;
                s.checkpoints.erase(oldest);
            }
            cp.summary.ref.ordinal = ordinal;
        }
        s.checkpoints.erase(
            std::remove_if(s.checkpoints.begin(), s.checkpoints.end(),
                           [&](const auto& previous) {
                               return previous.summary.ref.kind == cp.summary.ref.kind &&
                                      cp.summary.ref.kind != runtime::CheckpointKind::LongAnchor;
                           }),
            s.checkpoints.end());
        s.checkpoints.push_back(std::move(cp));
        ++revision;
    }

    std::uint64_t reserved_growth() const {
        std::uint64_t out = 0;
        for (const auto& [id, s] : sequences)
            if (s.active) {
                auto owned = pages(executor.slot_depth(s.slot));
                if (s.entitlement > owned) out += s.entitlement - owned;
            }
        return out;
    }

    bool capture_feasible(std::uint32_t capture_frontier, std::uint64_t replacement_id,
                          const CaptureVictimAction* victim) const {
        std::size_t available = std::count(slots.begin(), slots.end(), 0);
        std::vector<std::uint32_t> reclaimed_slots;
        if (replacement_id) {
            auto rit = sequences.find(replacement_id);
            if (rit != sequences.end() && !rit->second.host) {
                ++available;
                reclaimed_slots.push_back(rit->second.slot);
            }
        }
        if (victim) {
            auto vit = sequences.find(victim->sequence_id);
            if (vit == sequences.end() || vit->second.active || vit->second.host ||
                vit->first == replacement_id) {
                return false;
            }
            if (victim->spill_to_host && !executor.can_snapshot_to_host(vit->second.slot)) {
                return false;
            }
            ++available;
            reclaimed_slots.push_back(vit->second.slot);
        }
        const std::uint32_t cow = (capture_frontier % 64) != 0 ? 1U : 0U;
        const std::uint64_t reclaimable =
            reclaimed_slots.empty() ? 0 : executor.reclaimable_pages(reclaimed_slots);
        return available != 0 && executor.free_pages() + reclaimable >= reserved_growth() + cow;
    }
};

Program::Program(const detail::RuntimeModelView& model, DeviceContext& device,
                 EngineOptions options)
    : impl_(std::make_unique<Impl>(model, device, std::move(options))) {}

Program::~Program() {
    try {
        impl_->executor.sync();
    } catch (...) {}
}

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    const auto& data = PreparedPromptAccess::view(prompt);
    RequestBasePlan plan;
    plan.options                    = options;
    plan.options.allow_prefix_reuse = options.allow_prefix_reuse && data.identity.reusable;
    plan.cache                      = data.context_cache;
    plan.ids                        = data.token_ids;
    auto& s                         = plan.value;
    s.prompt_tokens                 = data.token_ids.size();
    s.requested_output_tokens       = options.requested_output_tokens;
    if (s.prompt_tokens == 0 || s.prompt_tokens >= impl_->options.max_context)
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "Inkling-2 prompt leaves no generation capacity");
    s.effective_output_tokens =
        std::min(options.requested_output_tokens, impl_->options.max_context - s.prompt_tokens);
    s.effective_limit_reason = s.effective_output_tokens < options.requested_output_tokens
                                   ? FinishReason::ContextCapacity
                                   : FinishReason::OutputLimit;
    s.service_work_quanta    = std::uint64_t(s.prompt_tokens) + s.effective_output_tokens;
    s.publish_continuation   = impl_->options.context_cache.enabled && data.identity.reusable &&
                               options.allow_prefix_reuse;
    return plan;
}

bool Program::physically_feasible(const AdmissionCandidate& c,
                                  std::span<const std::uint64_t> victims) const {
    const auto& p      = *impl_;
    std::uint64_t free = p.executor.free_pages();
    std::size_t states = std::count(p.slots.begin(), p.slots.end(), 0);
    std::vector<std::uint32_t> dropped;
    for (auto id : victims) {
        auto it = p.sequences.find(id);
        if (it != p.sequences.end() && !it->second.active && id != c.source) {
            if (!it->second.host) {
                ++states;
                dropped.push_back(it->second.slot);
            }
        }
    }
    if (!dropped.empty()) free += p.executor.reclaimable_pages(dropped);
    std::uint32_t reuse = 0;
    if (c.source) {
        auto it = p.sequences.find(c.source);
        if (it == p.sequences.end() || it->second.active) return false;
        if (!c.host) {
            reuse = pages(c.reused);
            if (c.identity.source_mode == runtime::PrivateSourceMode::ConsumeToActive) ++states;
        }
    }
    if (states == 0) return false;
    std::uint64_t target = pages(c.value.prompt_tokens + c.value.effective_output_tokens);
    std::uint64_t growth = target - reuse;
    if (c.source && !c.host && c.identity.source_mode == runtime::PrivateSourceMode::Retain &&
        (c.reused % 64))
        ++growth;
    return free >= p.reserved_growth() + growth;
}

std::optional<AdmissionCandidate>
Program::inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base,
                           runtime::LaneId destination, const ContinuationHandle* source,
                           const SharedPrefixHandle* shared,
                           std::optional<runtime::CheckpointRef> checkpoint, bool retain) {
    if (destination.value >= impl_->options.max_concurrency) return std::nullopt;
    for (const auto& [id, s] : impl_->sequences)
        if (s.active && s.lane == destination) return std::nullopt;
    AdmissionCandidate out;
    out.value   = base.value;
    out.options = base.options;
    out.lane    = destination;
    out.identity.source_mode =
        retain ? runtime::PrivateSourceMode::Retain : runtime::PrivateSourceMode::ConsumeToActive;
    if (source || shared) {
        auto id = source ? source->id : shared->id;
        auto it = impl_->sequences.find(id);
        if (it == impl_->sequences.end() || it->second.active || !base.options.allow_prefix_reuse)
            return std::nullopt;
        auto depth = impl_->depth(it->second);
        if (checkpoint && checkpoint->kind != runtime::CheckpointKind::SessionEndpoint &&
            checkpoint->kind != runtime::CheckpointKind::SharedStablePrefix) {
            auto selected =
                std::find_if(it->second.checkpoints.begin(), it->second.checkpoints.end(),
                             [&](const auto& cp) { return cp.summary.ref == *checkpoint; });
            if (selected == it->second.checkpoints.end()) return std::nullopt;
            depth                    = selected->summary.ref.frontier;
            out.selected_checkpoint  = checkpoint;
            out.identity.source_mode = runtime::PrivateSourceMode::Retain;
        }
        if (depth == 0 || depth >= base.ids.size() || it->second.ledger.size() < depth ||
            !std::equal(it->second.ledger.begin(), it->second.ledger.begin() + depth,
                        base.ids.begin()) ||
            (checkpoint && checkpoint->frontier != depth))
            return std::nullopt;
        out.source = id;
        out.shared = shared != nullptr;
        out.host   = out.selected_checkpoint.has_value() || it->second.host != nullptr;
        out.reused = depth;
        out.value.reusable_prompt_tokens = depth;
        if (shared) {
            out.value.prefix_reuse_path = PrefixReusePath::SharedStablePrefix;
        } else if (!out.selected_checkpoint) {
            out.value.prefix_reuse_path = PrefixReusePath::PrivateEndpoint;
        } else {
            switch (out.selected_checkpoint->kind) {
            case runtime::CheckpointKind::TurnClosure:
                out.value.prefix_reuse_path = PrefixReusePath::PrivateTurnClosure;
                break;
            case runtime::CheckpointKind::ResponseReplay:
                out.value.prefix_reuse_path = PrefixReusePath::PrivateResponseReplay;
                break;
            case runtime::CheckpointKind::LongAnchor:
                out.value.prefix_reuse_path = PrefixReusePath::PrivateLongAnchor;
                break;
            case runtime::CheckpointKind::SharedStablePrefix:
                out.value.prefix_reuse_path = PrefixReusePath::SharedStablePrefix;
                break;
            case runtime::CheckpointKind::SessionEndpoint:
                out.value.prefix_reuse_path = PrefixReusePath::PrivateEndpoint;
                break;
            }
        }
        if (shared) out.identity.source_mode = runtime::PrivateSourceMode::Retain;
        if (out.host) {
            const TransferWork state_w   = impl_->executor.fixed_state_transfer_work();
            const std::uint32_t kv_pages = pages(out.reused);
            const TransferWork kv_w      = impl_->executor.host_kv_transfer_work(kv_pages);
            out.transfers.push_back({runtime::ContextResourceClass::State,
                                     runtime::ContextTransferDirection::HostToDevice, 1, 0,
                                     state_w});
            coalesce_transfer(out.identity.machine_work.candidate_transfers,
                              runtime::ContextTransferDirection::HostToDevice, state_w);
            if (kv_pages != 0) {
                out.transfers.push_back({runtime::ContextResourceClass::MainKV,
                                         runtime::ContextTransferDirection::HostToDevice,
                                         kv_w.payload_bytes, kv_pages, kv_w});
                coalesce_transfer(out.identity.machine_work.candidate_transfers,
                                  runtime::ContextTransferDirection::HostToDevice, kv_w);
            }
        } else if (out.identity.source_mode == runtime::PrivateSourceMode::Retain) {
            const TransferWork state_w = impl_->executor.fixed_state_transfer_work();
            out.transfers.push_back({runtime::ContextResourceClass::State,
                                     runtime::ContextTransferDirection::DeviceToDevice, 1, 0,
                                     state_w});
            coalesce_transfer(out.identity.machine_work.candidate_transfers,
                              runtime::ContextTransferDirection::DeviceToDevice, state_w);
            if ((out.reused % 64) != 0) {
                const TransferWork cow_w = impl_->executor.device_kv_cow_transfer_work();
                out.transfers.push_back({runtime::ContextResourceClass::MainKV,
                                         runtime::ContextTransferDirection::DeviceToDevice,
                                         cow_w.payload_bytes, 1, cow_w});
                coalesce_transfer(out.identity.machine_work.candidate_transfers,
                                  runtime::ContextTransferDirection::DeviceToDevice, cow_w);
            }
        }
        out.identity.machine_work.optimistic_candidate_transfers =
            out.identity.machine_work.candidate_transfers;
    }
    out.identity.machine_work.remaining_prefill_work = runtime::make_prefill_work(
        out.reused, base.ids.size() - out.reused, 0, 0, impl_->options.prefill_chunk);
    out.identity.machine_work.reused_prompt_tokens = out.reused;
    out.identity.physical_status = physically_feasible(out, {})
                                       ? runtime::MaterializationPhysicalStatus::Feasible
                                       : runtime::MaterializationPhysicalStatus::Infeasible;
    out.identity.expandable =
        out.identity.physical_status != runtime::MaterializationPhysicalStatus::Feasible;
    out.identity.projection_work = 1U + out.transfers.size();
    return out;
}

std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& candidate,
                                                   const PreparedPrompt&,
                                                   runtime::FinalScheduleIntent intent) {
    if (!physically_feasible(candidate, {})) return std::nullopt;
    ResourcePlan plan{candidate, resource_revision(), {}, {}};
    plan.capture_frontiers.assign(intent.shared_capture_frontiers.begin(),
                                  intent.shared_capture_frontiers.end());
    return plan;
}

runtime::PrefillWork Program::shared_capture_split_prefill_work(const AdmissionCandidate& c,
                                                                const PreparedPrompt&,
                                                                std::span<const std::uint32_t>) {
    return c.identity.machine_work.remaining_prefill_work;
}

PressurePlanningSession
Program::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const runtime::PlanningCandidateId> ids,
                                 std::span<const ContinuationHandle* const> owners,
                                 std::span<const runtime::PlanningOwnerId> ownerids,
                                 std::span<const SharedPrefixHandle* const> shared,
                                 std::span<const runtime::PlanningOwnerId> sharedids) {
    PressurePlanningSession out;
    out.program = this;
    out.candidates.assign(candidates.begin(), candidates.end());
    out.ids.assign(ids.begin(), ids.end());
    for (std::size_t i = 0; i < owners.size(); ++i)
        out.owners.emplace_back(ownerids[i], owners[i]->id);
    for (std::size_t i = 0; i < shared.size(); ++i)
        out.shared_owners.emplace_back(sharedids[i], shared[i]->id);
    return out;
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancel) {
    if (cancel.requested() || has_context_transaction() || impl_->pending ||
        plan.revision != resource_revision())
        return runtime::ContextTransactionReserveStatus::Aborted;
    std::vector<std::uint64_t> victims;
    for (auto [owner, id] : plan.victims) victims.push_back(id);
    for (auto [owner, id] : plan.shared_victims) victims.push_back(id);
    if (!physically_feasible(plan.candidate, victims))
        return runtime::ContextTransactionReserveStatus::Aborted;
    impl_->transaction.emplace(std::move(plan));
    impl_->transaction_prompt.emplace(std::move(prompt));
    return runtime::ContextTransactionReserveStatus::Reserved;
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancel) {
    MaterializationResult out;
    if (impl_->capture_transaction) {
        const auto tx = *impl_->capture_transaction;
        if (cancel.requested()) {
            ActiveCaptureResult result;
            if (tx.victim) {
                auto vit = impl_->sequences.find(tx.victim->sequence_id);
                if (vit != impl_->sequences.end()) {
                    if (tx.victim->shared) {
                        SharedPrefixSummary s;
                        if (auto ep = impl_->summary(vit->second).endpoint) s.checkpoint = *ep;
                        result.shared_victims.push_back(
                            {tx.victim->owner, runtime::VictimDisposition::Retained, false, s});
                    } else {
                        result.victims.push_back({tx.victim->owner,
                                                  runtime::VictimDisposition::Retained, false,
                                                  impl_->summary(vit->second)});
                    }
                }
            }
            return result;
        }
        ActiveCaptureResult result;
        if (tx.victim) {
            auto& victim_seq = impl_->sequences.at(tx.victim->sequence_id);
            if (tx.victim->spill_to_host) {
                impl_->spill(victim_seq);
                result.victims.push_back({tx.victim->owner, runtime::VictimDisposition::Retained,
                                          true, impl_->summary(victim_seq)});
            } else if (tx.victim->shared) {
                impl_->erase(tx.victim->sequence_id);
                result.shared_victims.push_back(
                    {tx.victim->owner, runtime::VictimDisposition::Evicted, true, std::nullopt});
            } else {
                impl_->erase(tx.victim->sequence_id);
                result.victims.push_back(
                    {tx.victim->owner, runtime::VictimDisposition::Evicted, true, std::nullopt});
            }
        }
        if (tx.replacement_id) impl_->erase(tx.replacement_id);
        auto& source = impl_->sequences.at(tx.offer_id);
        Impl::Sequence snapshot;
        snapshot.id = impl_->next++;
        snapshot.slot =
            std::find(impl_->slots.begin(), impl_->slots.end(), 0) - impl_->slots.begin();
        snapshot.active = false;
        snapshot.shared = true;
        snapshot.ledger.assign(source.ledger.begin(), source.ledger.begin() + source.progress);
        snapshot.progress = source.progress;
        impl_->executor.copy_state(source.slot, snapshot.slot);
        impl_->slots.at(snapshot.slot) = snapshot.id;
        auto key                       = shortlist(snapshot.ledger, snapshot.ledger.size());
        SharedPrefixSummary summary;
        summary.checkpoint.ref           = {runtime::CheckpointKind::SharedStablePrefix,
                                            static_cast<std::uint32_t>(snapshot.ledger.size()), 0};
        summary.checkpoint.scope         = runtime::CheckpointScope::Shared;
        summary.checkpoint.shortlist_key = key;
        summary.active_references        = 1;
        summary.checkpoint.required_kv.main_frontier = snapshot.ledger.size();
        summary.checkpoint.required_kv.main_pages    = pages(snapshot.ledger.size());
        summary.checkpoint.rebuild_work              = runtime::make_prefill_work(
            0, snapshot.ledger.size(), 0, 0, impl_->options.prefill_chunk);
        auto id = snapshot.id;
        impl_->sequences.emplace(id, std::move(snapshot));
        ++impl_->revision;
        result.status                         = runtime::ContextTransactionStatus::Published;
        result.capacity_preparation_committed = tx.replacement_id != 0;
        result.active_summary                 = impl_->summary(source);
        result.shared                         = SharedPrefixPublication{{id}, summary};
        result.transfer_observations          = impl_->executor.take_transfer_observations();
        return result;
    }
    if (!impl_->transaction) throw std::logic_error("Inkling-2 has no context transaction");
    auto& plan = *impl_->transaction;
    auto& c    = plan.candidate;
    if (cancel.requested()) {
        if (c.source) {
            if (c.shared)
                out.shared_source = MaterializationSharedSourceResult{std::nullopt};
            else
                out.source =
                    MaterializationSourceResult{runtime::PrivateSourceMode::Retain,
                                                impl_->summary(impl_->sequences.at(c.source))};
        }
        for (auto [owner, id] : plan.victims)
            out.victims.push_back({owner, runtime::VictimDisposition::Retained, false,
                                   impl_->summary(impl_->sequences.at(id))});
        for (auto [owner, id] : plan.shared_victims) {
            SharedPrefixSummary summary;
            summary.checkpoint = impl_->summary(impl_->sequences.at(id)).endpoint.value();
            out.shared_victims.push_back(
                {owner, runtime::VictimDisposition::Retained, false, summary});
        }
        return out;
    }
    for (auto [owner, id] : plan.victims) {
        impl_->erase(id);
        out.victims.push_back({owner, runtime::VictimDisposition::Evicted, true, std::nullopt});
    }
    for (auto [owner, id] : plan.shared_victims) {
        impl_->erase(id);
        out.shared_victims.push_back(
            {owner, runtime::VictimDisposition::Evicted, true, std::nullopt});
    }
    Impl::Sequence seq;
    seq.id              = impl_->next++;
    seq.lane            = c.lane;
    seq.options         = c.options;
    auto prepared       = PreparedPromptAccess::take(std::move(*impl_->transaction_prompt));
    seq.prompt          = std::move(prepared.token_ids);
    seq.cache           = std::move(prepared.context_cache);
    seq.identity        = std::move(prepared.identity);
    seq.captures        = plan.capture_frontiers;
    seq.shared_captures = plan.capture_frontiers;
    for (const auto& marker : seq.cache.opportunities)
        if (marker.kind == PromptCacheMarkerKind::PrivateLongAnchor)
            seq.captures.push_back(marker.frontier);
    for (auto frontier : seq.identity.rewrite_execution_frontiers) seq.captures.push_back(frontier);
    if (seq.identity.rewrite_checkpoint)
        seq.captures.push_back(seq.identity.rewrite_checkpoint->frontier);
    seq.entitlement = pages(c.value.prompt_tokens + c.value.effective_output_tokens);
    seq.publish     = c.value.publish_continuation;
    seq.begin       = {c.value.prompt_tokens, c.reused, c.value.prefix_reuse_path};
    seq.progress    = c.reused;
    if (c.source) {
        auto& source = impl_->sequences.at(c.source);
        seq.ledger.assign(source.ledger.begin(), source.ledger.begin() + c.reused);
        for (const auto& cp : source.checkpoints)
            if (cp.summary.ref.frontier <= c.reused) seq.checkpoints.push_back(cp);
        auto image = source.host;
        if (c.selected_checkpoint) {
            auto selected = std::find_if(
                source.checkpoints.begin(), source.checkpoints.end(),
                [&](const auto& cp) { return cp.summary.ref == *c.selected_checkpoint; });
            image = selected->image;
        }
        if (image) {
            seq.slot =
                std::find(impl_->slots.begin(), impl_->slots.end(), 0) - impl_->slots.begin();
            impl_->executor.restore_from_host(seq.slot, *image);
            if (c.identity.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
                impl_->sequences.erase(c.source);
                out.source = MaterializationSourceResult{
                    runtime::PrivateSourceMode::ConsumeToActive, std::nullopt};
            } else if (c.shared) {
                out.shared_source = MaterializationSharedSourceResult{std::nullopt};
            } else
                out.source = MaterializationSourceResult{runtime::PrivateSourceMode::Retain,
                                                         impl_->summary(source)};
        } else if (c.identity.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            seq.slot = source.slot;
            impl_->sequences.erase(c.source);
            out.source = MaterializationSourceResult{runtime::PrivateSourceMode::ConsumeToActive,
                                                     std::nullopt};
        } else {
            seq.slot =
                std::find(impl_->slots.begin(), impl_->slots.end(), 0) - impl_->slots.begin();
            impl_->executor.copy_state(source.slot, seq.slot);
            if (c.shared) {
                out.shared_source = MaterializationSharedSourceResult{std::nullopt};
            } else
                out.source = MaterializationSourceResult{runtime::PrivateSourceMode::Retain,
                                                         impl_->summary(source)};
        }
    } else {
        seq.slot = std::find(impl_->slots.begin(), impl_->slots.end(), 0) - impl_->slots.begin();
        impl_->executor.reset(seq.slot);
    }
    if (impl_->options.speculative.backend != SpeculativeBackend::None) {
        seq.speculative.backend       = impl_->options.speculative.backend;
        seq.speculative.enabled       = true;
        seq.speculative.draft_window  = impl_->options.speculative.draft_tokens;
        const std::uint32_t max_verify_drafts =
            impl_->ngram_pool
                ? std::min(7U, std::max(impl_->options.speculative.draft_tokens,
                                        impl_->options.speculative.ngram.max_drafts))
                : impl_->options.speculative.draft_tokens;
        seq.speculative.verify_window = max_verify_drafts + 1U;
        seq.speculative.accepted_per_position.assign(max_verify_drafts, 0);
    }
    impl_->slots.at(seq.slot) = seq.id;
    auto id                   = seq.id;
    impl_->sequences.emplace(id, std::move(seq));
    ++impl_->revision;
    out.status                = runtime::ContextTransactionStatus::Published;
    out.published             = StartResult{{id}};
    out.transfer_observations = impl_->executor.take_transfer_observations();
    return out;
}

void Program::finalize_context_transaction() noexcept {
    impl_->transaction.reset();
    impl_->transaction_prompt.reset();
    impl_->capture_transaction.reset();
}

bool Program::has_context_transaction() const noexcept {
    return impl_->transaction.has_value() || impl_->capture_transaction.has_value();
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan&, const ResourcePlan&,
                                   std::span<const SequenceHandle>) const {
    return std::nullopt;
}

PrefillProgress Program::advance_prefill(SequenceHandle handle, runtime::ExecutionTiming* failed) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    auto& seq = impl_->get(handle);
    PrefillProgress out;
    out.summary = seq.begin;
    const auto chunk =
        std::min<std::uint32_t>({impl_->options.prefill_chunk, 1024, impl_->options.max_context});
    auto count = std::min<std::size_t>(chunk, seq.prompt.size() - seq.progress);
    for (auto frontier : seq.captures)
        if (frontier > seq.progress && frontier < seq.progress + count)
            count = frontier - seq.progress;
    if (!count) throw std::logic_error("Inkling-2 prefill called past prompt");
    auto ids      = std::span(seq.prompt).subspan(seq.progress, count);
    Tensor logits = impl_->executor.forward(ids, seq.progress, seq.slot);
    const std::size_t prev_ledger_size = seq.ledger.size();
    seq.ledger.insert(seq.ledger.end(), ids.begin(), ids.end());
    if (impl_->ngram_pool) { impl_->ngram_pool->observe(seq.ledger, prev_ledger_size); }
    seq.progress += count;
    ++impl_->revision;
    out.processed_prompt_tokens = count;
    out.complete                = seq.progress == seq.prompt.size();
    impl_->capture_private(seq);
    impl_->observe_external();
    timing.begin_wait();
    impl_->executor.sync();
    timing.end_wait();
    if (out.complete) {
        PendingBatch pending;
        pending.transaction = impl_->next++;
        pending.rows[0]     = handle;
        pending.counts      = {1};
        pending.values = {impl_->executor.sample(logits, seq.options.sampling, 0, seq.generated)};
        impl_->pending = pending.transaction;
        out.pending.emplace(std::move(pending));
    }
    out.timing = timing.finish();
    seq.timings.prefill_seconds += out.timing.elapsed_ns() * 1e-9;
    if (out.pending) out.pending->timing = out.timing;
    if (!out.complete && std::find(seq.shared_captures.begin(), seq.shared_captures.end(),
                                   seq.progress) != seq.shared_captures.end())
        out.capture = CaptureOffer{seq.id};
    return out;
}

PendingBatch Program::decode(std::span<const SequenceHandle> handles,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed) {
    if (impl_->pending || handles.size() != budgets.size() || handles.empty() ||
        handles.size() > kMaximumConcurrency)
        throw std::logic_error("Inkling-2 invalid decode transaction");
    for (std::size_t i = 0; i < handles.size(); ++i) {
        const auto& seq = impl_->get(handles[i]);
        if (!budgets[i].generated_tokens_remaining ||
            impl_->executor.slot_depth(seq.slot) >= seq.ledger.size())
            throw std::logic_error("Inkling-2 decode row has no budget or input");
        for (std::size_t j = 0; j < i; ++j)
            if (handles[i] == handles[j]) throw std::logic_error("Inkling-2 duplicate decode row");
    }
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    PendingBatch pending;
    pending.transaction = impl_->next++;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        auto& seq = impl_->get(handles[i]);
        if (!budgets[i].generated_tokens_remaining)
            throw std::logic_error("Inkling-2 decode has zero budget");
        auto depth = impl_->executor.slot_depth(seq.slot);
        if (depth >= seq.ledger.size())
            throw std::logic_error("Inkling-2 decode has no committed input");
        auto input = std::span(seq.ledger).subspan(depth);
        pending.rows[i] = handles[i];
        const bool use_dspark = impl_->options.speculative.backend != SpeculativeBackend::None &&
                                input.size() == 1 && budgets[i].generated_tokens_remaining > 1 &&
                                depth + 2 <= impl_->options.max_context;
        if (use_dspark) {
            const auto k = std::min<std::uint32_t>({impl_->options.speculative.draft_tokens,
                                                    budgets[i].generated_tokens_remaining - 1,
                                                    impl_->options.max_context - depth - 1});
            std::vector<TokenId> licensed;
            std::uint32_t mtp_drafts   = k;
            std::uint32_t ngram_drafts = 0;
            if (impl_->options.speculative.backend == SpeculativeBackend::DFlash2) {
                timing.begin_wait();
                licensed = impl_->executor.speculative_step_dflash2(
                    seq.slot, input[0], k, depth, seq.options.sampling,
                    static_cast<std::uint32_t>(seq.generated.size()), seq.generated);
                timing.end_wait();
                timing.resume_submit();
            } else if (!impl_->ngram_pool) {
                timing.begin_wait();
                licensed = impl_->executor.speculative_step_dspark(
                    seq.slot, input[0], k, depth, seq.options.sampling,
                    static_cast<std::uint32_t>(seq.generated.size()), seq.generated);
                timing.end_wait();
                timing.resume_submit();
            } else {
                auto drafts = impl_->executor.draft_dspark(seq.slot, input[0], k);
                std::vector<TokenId> verify_ids;
                verify_ids.reserve(8);
                verify_ids.push_back(input[0]);
                verify_ids.insert(verify_ids.end(), drafts.begin(), drafts.end());
                const std::uint32_t max_total_drafts = std::min<std::uint32_t>(
                    {7U, impl_->options.speculative.ngram.max_drafts,
                     budgets[i].generated_tokens_remaining - 1U,
                     impl_->options.max_context - depth - 1U});
                if (max_total_drafts > k) {
                    const std::size_t n = impl_->ngram_pool->match_tokens();
                    std::vector<TokenId> ctx;
                    const std::size_t from_ledger = std::min(seq.ledger.size(), n);
                    ctx.reserve(from_ledger + drafts.size());
                    ctx.insert(ctx.end(),
                               seq.ledger.end() - static_cast<std::ptrdiff_t>(from_ledger),
                               seq.ledger.end());
                    ctx.insert(ctx.end(), drafts.begin(), drafts.end());
                    std::vector<TokenId> ext(max_total_drafts - k);
                    const std::size_t ext_len = impl_->ngram_pool->propose(ctx, ext);
                    const std::size_t min_ext =
                        std::max<std::size_t>(2U, impl_->options.speculative.ngram.min_drafts);
                    if (ext_len >= min_ext) {
                        ngram_drafts = static_cast<std::uint32_t>(ext_len);
                        verify_ids.insert(verify_ids.end(), ext.begin(), ext.begin() + ext_len);
                    }
                }
                timing.begin_wait();
                licensed = impl_->executor.verify_dspark(
                    seq.slot, verify_ids, depth, seq.options.sampling,
                    static_cast<std::uint32_t>(seq.generated.size()), seq.generated);
                timing.end_wait();
                timing.resume_submit();
            }
            ++impl_->revision;
            pending.counts.push_back(static_cast<std::uint32_t>(licensed.size()));
            pending.values.insert(pending.values.end(), licensed.begin(), licensed.end());
            ++seq.speculative.rounds;
            seq.speculative.drafted_tokens += mtp_drafts + ngram_drafts;
            const auto accepted_drafts = static_cast<std::uint32_t>(licensed.size() - 1);
            seq.speculative.accepted_tokens += accepted_drafts;
            seq.speculative.ngram_drafted_tokens += ngram_drafts;
            const auto mtp_acc = std::min(accepted_drafts, mtp_drafts);
            seq.speculative.ngram_accepted_tokens += accepted_drafts - mtp_acc;
            if (ngram_drafts > 0) ++seq.speculative.wide_rounds;
            for (std::uint32_t j = 0;
                 j < accepted_drafts && j < seq.speculative.accepted_per_position.size(); ++j)
                ++seq.speculative.accepted_per_position[j];
        } else {
            Tensor logits = impl_->executor.forward(input, depth, seq.slot);
            ++impl_->revision;
            pending.counts.push_back(1);
            timing.begin_wait();
            pending.values.push_back(impl_->executor.sample(logits, seq.options.sampling,
                                                            seq.generated.size(), seq.generated));
            timing.end_wait();
            timing.resume_submit();
            if (impl_->options.speculative.backend != SpeculativeBackend::None)
                ++seq.speculative.fallback_steps;
        }
    }
    impl_->observe_external();
    pending.timing = timing.finish();
    impl_->pending = pending.transaction;
    for (auto h : handles)
        impl_->get(h).timings.decode_seconds += pending.timing.elapsed_ns() * 1e-9;
    return pending;
}

CommitResult Program::commit(PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    if (!impl_->pending || pending.transaction != impl_->pending ||
        decisions.size() != pending.row_count())
        throw std::logic_error("Inkling-2 invalid pending commit");
    std::size_t expected_values = 0;
    for (std::size_t i = 0; i < decisions.size(); ++i) expected_values += pending.counts[i];
    if (pending.values.size() != expected_values)
        throw std::logic_error("Inkling-2 pending values truncated");
    for (std::size_t i = 0; i < decisions.size(); ++i) {
        if (decisions[i].accepted_tokens > pending.counts[i])
            throw std::logic_error("Inkling-2 commit exceeds pending row");
        (void)impl_->get(pending.rows[i]);
        for (std::size_t j = 0; j < i; ++j)
            if (pending.rows[i] == pending.rows[j])
                throw std::logic_error("Inkling-2 duplicate commit row");
    }
    CommitResult out;
    out.row_count = decisions.size();
    std::size_t value_offset = 0;
    for (std::size_t i = 0; i < decisions.size(); ++i) {
        const auto& d = decisions[i];
        auto& s       = impl_->get(pending.rows[i]);
        out.rows[i].timings     = s.timings;
        out.rows[i].speculative = s.speculative;
        if (d.cancelled) {
            impl_->erase(s.id);
            out.rows[i].disposition = runtime::CommitDisposition::CancelledReleased;
            value_offset += pending.counts[i];
            continue;
        }
        if (d.accepted_tokens) {
            if (s.generated.empty() && std::find(s.shared_captures.begin(), s.shared_captures.end(),
                                                 s.progress) != s.shared_captures.end())
                out.captures[i] = CaptureOffer{s.id};
            const std::size_t prev_ledger_size = s.ledger.size();
            for (std::uint32_t j = 0; j < d.accepted_tokens; ++j) {
                const auto tok = pending.values[value_offset + j];
                s.ledger.push_back(tok);
                s.generated.push_back(tok);
            }
            if (impl_->ngram_pool) { impl_->ngram_pool->observe(s.ledger, prev_ledger_size); }
            if (d.accepted_tokens < pending.counts[i] &&
                impl_->options.speculative.backend != SpeculativeBackend::None) {
                impl_->executor.truncate_speculative_commit(s.slot, d.accepted_tokens);
            }
        }
        value_offset += pending.counts[i];
        out.rows[i].disposition = d.terminal ? runtime::CommitDisposition::Finishable
                                             : runtime::CommitDisposition::Active;
    }
    impl_->pending = 0;
    return out;
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    if (!impl_->pending || pending.transaction != impl_->pending) return {};
    DiscardResult out{runtime::ConsumeStatus::Consumed, pending.row_count()};
    for (std::size_t i = 0; i < pending.row_count(); ++i) try {
            impl_->erase(pending.rows[i].id);
        } catch (...) {}
    impl_->pending = 0;
    return out;
}

FinishResult Program::finish(SequenceHandle handle) noexcept {
    FinishResult out;
    try {
        auto& seq       = impl_->get(handle);
        out.status      = runtime::ConsumeStatus::Consumed;
        out.timings     = seq.timings;
        out.speculative = seq.speculative;
        if (seq.publish) {
            seq.active = false;
            seq.ledger.resize(impl_->executor.slot_depth(seq.slot));
            seq.prompt.clear();
            seq.generated.clear();
            if (std::count(impl_->slots.begin(), impl_->slots.end(), 0) == 0 &&
                impl_->executor.can_snapshot_to_host(seq.slot))
                impl_->spill(seq);
            impl_->observe_external();
            out.summary      = impl_->summary(seq);
            out.continuation = ContinuationHandle{seq.id};
            out.disposition  = runtime::FinishDisposition::Catalogued;
            ++impl_->revision;
        } else {
            impl_->erase(handle.id);
            impl_->observe_external();
        }
    } catch (...) {}
    return out;
}

AbortResult Program::abort(SequenceHandle handle) noexcept {
    AbortResult out;
    try {
        auto& s         = impl_->get(handle);
        out.timings     = s.timings;
        out.speculative = s.speculative;
        impl_->erase(handle.id);
        out.status = runtime::ConsumeStatus::Consumed;
    } catch (...) {}
    return out;
}

ReleaseResult Program::release_continuation(ContinuationHandle&& h) noexcept {
    ReleaseResult out;
    try {
        auto it = impl_->sequences.find(h.id);
        if (it != impl_->sequences.end() && !it->second.active) {
            impl_->erase(h.id);
            h.id       = 0;
            out.status = runtime::ConsumeStatus::Consumed;
        }
    } catch (...) {}
    return out;
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& h) noexcept {
    ReleaseResult out;
    try {
        auto it = impl_->sequences.find(h.id);
        if (it != impl_->sequences.end() && it->second.shared) {
            impl_->erase(h.id);
            h.id       = 0;
            out.status = runtime::ConsumeStatus::Consumed;
        }
    } catch (...) {}
    return out;
}

void Program::fail_all_cleanup() noexcept {
    try {
        while (!impl_->sequences.empty()) impl_->erase(impl_->sequences.begin()->first);
        impl_->pending = 0;
        finalize_context_transaction();
    } catch (...) {}
}

std::uint32_t Program::continuation_depth(const ContinuationHandle& h) const noexcept {
    auto it = impl_->sequences.find(h.id);
    return it == impl_->sequences.end() ? 0 : impl_->depth(it->second);
}

std::string Program::continuation_digest(const ContinuationHandle& h) const {
    auto it = impl_->sequences.find(h.id);
    return it == impl_->sequences.end() ? std::string() : digest(it->second.ledger);
}

std::vector<SlotCheckpoint> Program::continuation_checkpoints(const ContinuationHandle& h) const {
    std::vector<SlotCheckpoint> out;
    auto it = impl_->sequences.find(h.id);
    if (it == impl_->sequences.end()) return out;
    for (const auto& cp : it->second.checkpoints)
        out.push_back({cp.summary.ref.frontier,
                       digest(std::span(it->second.ledger).first(cp.summary.ref.frontier))});
    auto d = continuation_depth(h);
    if (d) out.push_back({d, continuation_digest(h)});
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.frontier < b.frontier; });
    out.erase(std::unique(out.begin(), out.end(),
                          [](const auto& a, const auto& b) { return a.frontier == b.frontier; }),
              out.end());
    return out;
}

runtime::SessionSnapshotTraffic Program::session_snapshot_traffic() const noexcept {
    return impl_->traffic;
}

ContinuationSummary Program::continuation_summary(const ContinuationHandle& h) const {
    return impl_->summary(impl_->sequences.at(h.id));
}

bool Program::isolated_request_feasible(const RequestBasePlan& b) const noexcept {
    return pages(b.value.prompt_tokens + b.value.effective_output_tokens) <=
           pages(impl_->options.kv_capacity.explicit_tokens);
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return {impl_->revision};
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    return {resource_revision(),
            static_cast<std::uint32_t>(std::count_if(impl_->slots.begin(), impl_->slots.end(),
                                                     [](auto id) { return id != 0; })),
            impl_->executor.host_state_count(),
            impl_->executor.used_pages(),
            0,
            impl_->executor.host_kv_bytes()};
}

MemorySummary Program::memory_summary() const noexcept { return impl_->executor.memory_summary(); }

void Program::reset_memory_peaks() noexcept { impl_->executor.reset_memory_peaks(); }

std::vector<float> Program::causal_score(PreparedPrompt&& p, std::uint32_t first) {
    auto data = PreparedPromptAccess::take(std::move(p));
    return impl_->executor.causal_score(data.token_ids, first);
}

runtime::ExecutionTiming Program::append_forced_tokens(
    std::span<const SequenceHandle> handles, std::span<const TokenId> tokens, std::uint32_t stride,
    std::span<const std::optional<std::uint32_t>>, runtime::ExecutionTiming* failed) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    if (tokens.size() < handles.size() * stride)
        throw std::invalid_argument("Inkling-2 forced rows truncated");
    for (std::size_t i = 0; i < handles.size(); ++i) {
        auto& s  = impl_->get(handles[i]);
        auto row = tokens.subspan(i * stride, stride);
        s.ledger.insert(s.ledger.end(), row.begin(), row.end());
        auto depth = impl_->executor.slot_depth(s.slot);
        auto input = std::span(s.ledger).subspan(depth);
        (void)impl_->executor.forward(input, depth, s.slot);
        ++impl_->revision;
    }
    timing.begin_wait();
    impl_->executor.sync();
    timing.end_wait();
    impl_->observe_external();
    return timing.finish();
}

CaptureAssessment Program::inspect_capture(const CaptureOffer& offer,
                                           const SharedPrefixHandle* exact,
                                           const SharedPrefixHandle* replacement,
                                           std::optional<runtime::CheckpointRef>,
                                           bool permit) const {
    CaptureAssessment out;
    auto it = impl_->sequences.find(offer.id);
    if (it == impl_->sequences.end() || !it->second.active) return out;
    const auto& seq   = it->second;
    out.source        = seq.id;
    out.replacement   = replacement ? replacement->id : 0;
    out.shortlist_key = shortlist(seq.ledger, seq.progress);
    out.protected_rebuild_work =
        runtime::make_prefill_work(0, seq.progress, 0, 0, impl_->options.prefill_chunk);
    auto opportunity = std::find_if(seq.cache.opportunities.begin(), seq.cache.opportunities.end(),
                                    [&](const auto& p) { return p.frontier == seq.progress; });
    if (opportunity != seq.cache.opportunities.end()) out.shared_evidence = opportunity->evidence;
    out.publishes_shared = permit && !exact;
    if (out.publishes_shared) {
        const TransferWork state_w = impl_->executor.fixed_state_transfer_work();
        out.transfer_requirements.push_back({runtime::ContextResourceClass::State,
                                             runtime::ContextTransferDirection::DeviceToDevice, 1,
                                             0, state_w});
        if ((seq.progress % 64) != 0) {
            const TransferWork cow_w = impl_->executor.device_kv_cow_transfer_work();
            out.transfer_requirements.push_back({runtime::ContextResourceClass::MainKV,
                                                 runtime::ContextTransferDirection::DeviceToDevice,
                                                 cow_w.payload_bytes, 1, cow_w});
        }
        out.needs_transfer = !out.transfer_requirements.empty();
        out.projected_recovery_work.push_back({{}, out.protected_rebuild_work});
        out.projected_recovery_work.push_back({});
    } else {
        out.projected_recovery_work.push_back({});
    }
    out.physically_feasible = impl_->capture_feasible(seq.progress, out.replacement, nullptr);
    return out;
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const ContinuationHandle& h, runtime::CheckpointRef cp) const {
    auto it = impl_->sequences.find(h.id);
    if (it == impl_->sequences.end())
        throw std::invalid_argument("Inkling-2 continuation is not retained");
    const auto& seq = it->second;
    std::optional<CheckpointSummary> selected;
    bool host_only = false;
    if (cp.kind == runtime::CheckpointKind::SessionEndpoint ||
        cp.kind == runtime::CheckpointKind::SharedStablePrefix) {
        const auto sum = impl_->summary(seq);
        if (!sum.endpoint || sum.endpoint->ref != cp)
            throw std::invalid_argument("Inkling-2 endpoint changed");
        selected  = sum.endpoint;
        host_only = seq.host != nullptr;
    } else {
        auto found = std::find_if(seq.checkpoints.begin(), seq.checkpoints.end(),
                                  [&](const auto& c) { return c.summary.ref == cp; });
        if (found == seq.checkpoints.end())
            throw std::invalid_argument("Inkling-2 checkpoint is not retained");
        selected  = found->summary;
        host_only = true;
    }
    std::vector<runtime::CheckpointRecoveryAlternativeWork> out;
    out.push_back({{}, selected->rebuild_work});
    if (host_only) {
        runtime::CheckpointRecoveryAlternativeWork restore;
        coalesce_transfer(restore.transfers, runtime::ContextTransferDirection::HostToDevice,
                          impl_->executor.fixed_state_transfer_work());
        coalesce_transfer(restore.transfers, runtime::ContextTransferDirection::HostToDevice,
                          impl_->executor.host_kv_transfer_work(selected->required_kv.main_pages));
        out.push_back(restore);
    } else {
        out.push_back({});
    }
    return out;
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const SharedPrefixHandle& h, runtime::CheckpointRef cp) const {
    return checkpoint_recovery_work(ContinuationHandle{h.id}, cp);
}

CapturePressurePlanningSession Program::begin_capture_pressure_planning(
    const CaptureAssessment& assessment, std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_ids) {
    CapturePressurePlanningSession out;
    out.program  = this;
    out.capture  = assessment;
    out.revision = resource_revision();
    for (std::size_t i = 0; i < private_owners.size(); ++i)
        out.owners.emplace_back(private_ids[i], private_owners[i]->id);
    for (std::size_t i = 0; i < shared_owners.size(); ++i)
        out.shared_owners.emplace_back(shared_ids[i], shared_owners[i]->id);
    return out;
}

bool Program::shared_capture_matches(const CaptureOffer& offer,
                                     const SharedPrefixHandle& shared) const {
    auto a = impl_->sequences.find(offer.id), b = impl_->sequences.find(shared.id);
    return a != impl_->sequences.end() && b != impl_->sequences.end() &&
           a->second.progress == b->second.ledger.size() &&
           std::equal(b->second.ledger.begin(), b->second.ledger.end(), a->second.ledger.begin());
}

void Program::skip_capture(CaptureOffer&&) {}

runtime::ContextTransactionReserveStatus Program::reserve_active_capture(
    CaptureOffer&& offer, const SharedPrefixHandle* exact, const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> cp, bool permit, runtime::CancellationFlagView cancel) {
    auto assessment = inspect_capture(offer, exact, replacement, cp, permit);
    if (cancel.requested() || has_context_transaction() || !assessment.physically_feasible ||
        !assessment.publishes_shared)
        return runtime::ContextTransactionReserveStatus::Aborted;
    impl_->capture_transaction = Program::Impl::CaptureTransaction{
        offer.id, replacement ? replacement->id : 0, std::nullopt};
    return runtime::ContextTransactionReserveStatus::Reserved;
}

runtime::ContextTransactionReserveStatus Program::reserve_active_capture_with_pressure(
    CaptureOffer&& offer, const SharedPrefixHandle* exact, const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> cp, bool permit, CapturePressurePlan&& plan,
    runtime::CancellationFlagView cancel) {
    if (plan.revision != resource_revision() || cancel.requested() || has_context_transaction())
        return runtime::ContextTransactionReserveStatus::Aborted;
    auto assessment = inspect_capture(offer, exact, replacement, cp, permit);
    if (!assessment.publishes_shared) return runtime::ContextTransactionReserveStatus::Aborted;
    const std::uint64_t rep_id = replacement ? replacement->id : 0;
    const auto it              = impl_->sequences.find(offer.id);
    if (it == impl_->sequences.end() || !it->second.active)
        return runtime::ContextTransactionReserveStatus::Aborted;
    if (!impl_->capture_feasible(it->second.progress, rep_id,
                                 plan.victim ? &*plan.victim : nullptr))
        return runtime::ContextTransactionReserveStatus::Aborted;
    impl_->capture_transaction =
        Program::Impl::CaptureTransaction{offer.id, rep_id, std::move(plan.victim)};
    return runtime::ContextTransactionReserveStatus::Reserved;
}

runtime::PressureTargetGuidance
CapturePressurePlanningSession::guidance(PressureTargetHandle t) {
    runtime::PressureTargetGuidance out;
    out.candidate             = candidate_id();
    out.stable_target_ordinal = t.candidate;
    for (const auto& req : capture.transfer_requirements) {
        coalesce_transfer(out.estimated_machine_work.candidate_transfers, req.direction, req.work);
    }
    out.estimated_machine_work.optimistic_candidate_transfers =
        out.estimated_machine_work.candidate_transfers;
    if (t.candidate != 0) {
        const auto& a   = children_actions_.at(t.candidate - 1);
        const auto& seq = program->impl_->sequences.at(a.sequence_id);
        const auto s    = program->impl_->summary(seq);
        const std::uint32_t dropped = a.spill_to_host ? 0U : count_checkpoints(s);
        guidance_outcomes_.assign(1, {a.owner,
                                      a.spill_to_host ? runtime::VictimDisposition::Retained
                                                      : runtime::VictimDisposition::Evicted,
                                      1U, dropped});
        if (a.spill_to_host) {
            coalesce_transfer(out.estimated_machine_work.pressure_transfers,
                              runtime::ContextTransferDirection::DeviceToHost,
                              program->impl_->executor.fixed_state_transfer_work());
            coalesce_transfer(
                out.estimated_machine_work.pressure_transfers,
                runtime::ContextTransferDirection::DeviceToHost,
                program->impl_->executor.host_kv_transfer_work(pages(program->impl_->depth(seq))));
        }
        out.owner_outcomes      = guidance_outcomes_;
        out.degradation_units   = 1U;
        out.dropped_checkpoints = dropped;
    }
    return out;
}

AssessedPressureTarget CapturePressurePlanningSession::assess(PressureTargetHandle t) {
    AssessedPressureTarget out;
    out.handle                      = t;
    out.value.candidate             = candidate_id();
    out.value.stable_target_ordinal = t.candidate;
    for (const auto& req : capture.transfer_requirements) {
        coalesce_transfer(out.value.machine_work.candidate_transfers, req.direction, req.work);
    }
    out.value.machine_work.optimistic_candidate_transfers =
        out.value.machine_work.candidate_transfers;
    if (t.candidate == 0) {
        out.value.physical_status = capture.physically_feasible
                                        ? runtime::MaterializationPhysicalStatus::Feasible
                                        : runtime::MaterializationPhysicalStatus::Infeasible;
        out.value.projection_work = 1U + capture.transfer_requirements.size();
        const auto has_device     = [&](const auto& list, std::uint64_t exclude) {
            for (auto [owner, id] : list) {
                if (id == exclude) continue;
                auto it = program->impl_->sequences.find(id);
                if (it != program->impl_->sequences.end() && !it->second.active &&
                    !it->second.host) {
                    return true;
                }
            }
            return false;
        };
        out.value.expandable = !capture.physically_feasible &&
                               (has_device(owners, 0) || has_device(shared_owners, capture.replacement));
        return out;
    }
    const auto& a      = children_actions_.at(t.candidate - 1);
    const auto& victim = program->impl_->sequences.at(a.sequence_id);
    const bool feasible =
        program->impl_->capture_feasible(capture.shortlist_key.frontier, capture.replacement, &a);
    out.value.physical_status = feasible ? runtime::MaterializationPhysicalStatus::Feasible
                                         : runtime::MaterializationPhysicalStatus::Infeasible;
    if (a.spill_to_host) {
        coalesce_transfer(out.value.machine_work.pressure_transfers,
                          runtime::ContextTransferDirection::DeviceToHost,
                          program->impl_->executor.fixed_state_transfer_work());
        coalesce_transfer(out.value.machine_work.pressure_transfers,
                          runtime::ContextTransferDirection::DeviceToHost,
                          program->impl_->executor.host_kv_transfer_work(
                              pages(program->impl_->depth(victim))));
    }
    const auto s            = program->impl_->summary(victim);
    const std::uint32_t cps = count_checkpoints(s);
    out.recovery.reserve(cps * 2U);
    out.impacts.reserve(cps);
    const auto add_cp = [&](const CheckpointSummary& cp) {
        const std::size_t idx = out.recovery.size();
        out.recovery.push_back({{}, cp.rebuild_work});
        if (a.spill_to_host) {
            runtime::CheckpointRecoveryAlternativeWork restore;
            coalesce_transfer(restore.transfers, runtime::ContextTransferDirection::HostToDevice,
                              program->impl_->executor.fixed_state_transfer_work());
            coalesce_transfer(
                restore.transfers, runtime::ContextTransferDirection::HostToDevice,
                program->impl_->executor.host_kv_transfer_work(cp.required_kv.main_pages));
            out.recovery.push_back(restore);
            out.impacts.push_back({a.owner, cp.ref, std::span(out.recovery.data() + idx, 2), true});
        } else {
            out.impacts.push_back({a.owner, cp.ref, std::span(out.recovery.data() + idx, 1), false});
        }
    };
    if (s.endpoint) add_cp(*s.endpoint);
    if (s.rewrite) add_cp(*s.rewrite);
    for (const auto& cp : s.long_anchors) add_cp(cp);
    const std::uint32_t dropped = a.spill_to_host ? 0U : cps;
    out.outcomes.push_back({a.owner,
                            a.spill_to_host ? runtime::VictimDisposition::Retained
                                            : runtime::VictimDisposition::Evicted,
                            1U, dropped});
    out.value.degradation_units   = 1U;
    out.value.dropped_checkpoints = dropped;
    out.value.owner_outcomes      = out.outcomes;
    out.value.checkpoint_impacts  = out.impacts;
    out.value.projection_work     = 2U + capture.transfer_requirements.size();
    out.value.expandable          = false;
    return out;
}

PreparedPressureExpansion
CapturePressurePlanningSession::prepare_expansion(PressureTargetHandle t) {
    if (t.candidate != 0 || expanded_ || scratch_ready_ || capture.physically_feasible) {
        return {};
    }
    children_actions_.clear();
    for (auto [owner, id] : owners) {
        auto it = program->impl_->sequences.find(id);
        if (it == program->impl_->sequences.end() || it->second.active || it->second.host) continue;
        if (program->impl_->executor.can_snapshot_to_host(it->second.slot)) {
            children_actions_.push_back({owner, id, false, true});
        }
        children_actions_.push_back({owner, id, false, false});
    }
    for (auto [owner, id] : shared_owners) {
        if (id == capture.replacement) continue;
        auto it = program->impl_->sequences.find(id);
        if (it == program->impl_->sequences.end() || it->second.active || it->second.host) continue;
        children_actions_.push_back({owner, id, true, false});
    }
    scratch_ready_ = true;
    return {static_cast<std::uint32_t>(children_actions_.size())};
}

PressureExpansionView
CapturePressurePlanningSession::commit_expansion(PreparedPressureExpansion&&) {
    if (!scratch_ready_) return {};
    scratch_ready_ = false;
    expanded_      = true;
    child_handles_.clear();
    child_handles_.reserve(children_actions_.size());
    for (std::size_t i = 0; i < children_actions_.size(); ++i) {
        child_handles_.push_back({static_cast<std::uint32_t>(i + 1), false});
    }
    return {child_handles_, static_cast<std::uint32_t>(child_handles_.size()), true};
}

void CapturePressurePlanningSession::discard_expansion(PreparedPressureExpansion&&) noexcept {
    if (scratch_ready_) {
        children_actions_.clear();
        scratch_ready_ = false;
    }
}

std::optional<CapturePressurePlan>
CapturePressurePlanningSession::seal(AssessedPressureTarget&& assessed) {
    if (assessed.value.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        return std::nullopt;
    }
    if (assessed.handle.candidate == 0) return CapturePressurePlan{revision, std::nullopt};
    return CapturePressurePlan{revision, children_actions_.at(assessed.handle.candidate - 1)};
}

PressureTargetHandle
PressurePlanningSession::identity_target(runtime::PlanningCandidateId id) const {
    auto i = std::find(ids.begin(), ids.end(), id);
    if (i == ids.end()) throw std::invalid_argument("Inkling-2 pressure candidate");
    return {static_cast<std::uint32_t>(i - ids.begin()), false};
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(runtime::PlanningCandidateId id) const {
    auto t    = identity_target(id);
    t.maximal = true;
    return t;
}

runtime::PressureTargetGuidance PressurePlanningSession::guidance(PressureTargetHandle t) {
    const auto& c = *candidates.at(t.candidate);
    runtime::PressureTargetGuidance out;
    out.candidate              = ids[t.candidate];
    out.estimated_machine_work = c.identity.machine_work;
    out.stable_target_ordinal  = t.candidate * 2 + int(t.maximal);
    out.source_mode            = c.identity.source_mode;
    outcomes_.clear();
    if (t.maximal) {
        const auto append = [&](const auto& all) {
            for (auto [owner, id] : all)
                if (id != c.source) {
                    auto summary = program->continuation_summary(ContinuationHandle{id});
                    auto dropped = static_cast<std::uint32_t>(summary.endpoint.has_value() +
                                                              summary.rewrite.has_value() +
                                                              summary.long_anchors.size());
                    outcomes_.push_back({owner, runtime::VictimDisposition::Evicted, 1, dropped});
                }
        };
        append(owners);
        append(shared_owners);
    }
    out.owner_outcomes    = outcomes_;
    out.degradation_units = outcomes_.size();
    for (const auto& item : outcomes_) out.dropped_checkpoints += item.dropped_checkpoints;
    return out;
}

AssessedPressureTarget PressurePlanningSession::assess(PressureTargetHandle t) {
    const auto& c = *candidates.at(t.candidate);
    AssessedPressureTarget out;
    out.handle = t;
    std::vector<std::uint64_t> drops;
    const auto max_count =
        (owners.size() + shared_owners.size()) *
        (2 + program->impl_->options.context_cache.max_long_anchors_per_continuation.value_or(0));
    out.recovery.reserve(max_count);
    out.impacts.reserve(max_count);
    if (t.maximal) {
        const auto append = [&](const auto& all) {
            for (auto [owner, id] : all)
                if (id != c.source) {
                    drops.push_back(id);
                    auto summary          = program->continuation_summary(ContinuationHandle{id});
                    std::uint32_t count   = 0;
                    const auto checkpoint = [&](const auto& cp) {
                        runtime::CheckpointRecoveryAlternativeWork work;
                        work.prefill = cp.rebuild_work;
                        out.recovery.push_back(work);
                        out.impacts.push_back(
                            {owner, cp.ref, std::span(&out.recovery.back(), 1), false});
                        ++count;
                    };
                    if (summary.endpoint) checkpoint(*summary.endpoint);
                    if (summary.rewrite) checkpoint(*summary.rewrite);
                    for (const auto& cp : summary.long_anchors) checkpoint(cp);
                    out.outcomes.push_back({owner, runtime::VictimDisposition::Evicted, 1, count});
                }
        };
        append(owners);
        append(shared_owners);
    }
    out.value.physical_status       = program->physically_feasible(c, drops)
                                          ? runtime::MaterializationPhysicalStatus::Feasible
                                          : runtime::MaterializationPhysicalStatus::Infeasible;
    out.value.source_mode           = c.identity.source_mode;
    out.value.machine_work          = c.identity.machine_work;
    out.value.candidate             = ids[t.candidate];
    out.value.stable_target_ordinal = t.candidate * 2 + int(t.maximal);
    out.value.root_maximal          = t.maximal;
    out.value.degradation_units     = out.outcomes.size();
    for (const auto& item : out.outcomes) out.value.dropped_checkpoints += item.dropped_checkpoints;
    out.value.owner_outcomes     = out.outcomes;
    out.value.checkpoint_impacts = out.impacts;
    return out;
}

runtime::PrefillWork PressurePlanningSession::shared_capture_split_prefill_work(
    const AssessedPressureTarget& a, const PreparedPrompt&, std::span<const std::uint32_t>) const {
    return a.value.machine_work.remaining_prefill_work;
}

std::optional<ResourcePlan> PressurePlanningSession::seal(AssessedPressureTarget&& assessed,
                                                          const PreparedPrompt&,
                                                          runtime::FinalScheduleIntent intent) {
    auto t = assessed.handle;
    ResourcePlan out{*candidates.at(t.candidate), program->resource_revision(), {}, {}};
    if (t.maximal) {
        for (auto [owner, id] : owners)
            if (id != out.candidate.source) out.victims.emplace_back(owner, id);
        for (auto [owner, id] : shared_owners)
            if (id != out.candidate.source) out.shared_victims.emplace_back(owner, id);
    }
    std::vector<std::uint64_t> dropped;
    for (auto [owner, id] : out.victims) dropped.push_back(id);
    for (auto [owner, id] : out.shared_victims) dropped.push_back(id);
    if (!program->physically_feasible(out.candidate, dropped)) return std::nullopt;
    out.capture_frontiers.assign(intent.shared_capture_frontiers.begin(),
                                 intent.shared_capture_frontiers.end());
    return out;
}
} // namespace ninfer::targets::inkling2_mini
