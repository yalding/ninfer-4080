#include <ninfer/targets/inkling2_mini/runtime.h>
#include "targets/inkling2_mini/impl/load/bindings.h"
#include "core/device.h"
#include <atomic>
#include <iostream>
#include <numeric>
#include <stdexcept>

// A synthetic prompt exercises the real Program's admission/control path without reading any
// model weight. Numerical execution is qualified separately through the Op and product gates.
namespace ninfer::targets::inkling2_mini {
class FrontendTestAccess {
public:
    static PreparedPrompt tokens(std::size_t count, bool reusable = true) {
        auto data = std::make_unique<PreparedPromptData>();
        data->token_ids.assign(count, 1);
        data->positions.resize(count);
        std::iota(data->positions.begin(), data->positions.end(), 0);
        data->identity.reusable = reusable;
        return PreparedPrompt(std::move(data));
    }
};
} // namespace ninfer::targets::inkling2_mini

using namespace ninfer;
namespace ik = ninfer::targets::inkling2_mini;
namespace rt = ninfer::runtime;

namespace {
void require(bool condition, const char* label) {
    if (!condition) { throw std::runtime_error(label); }
}

template <class F>
void rejects(F operation, const char* label) {
    try {
        operation();
    } catch (const std::exception&) { return; }
    throw std::runtime_error(label);
}

void planner_tests() {
    EngineOptions options;
    options.max_context                          = 131072;
    options.kv_capacity                          = KvCapacityPolicy::explicit_capacity(131072);
    options.kv_cache                             = KvCacheStorage::Int8Group64;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    ik::SequencePlanner planner(options);
    const auto curve = planner.capacity_curve();
    require(curve.main_page_tokens == 64 && curve.minimum_main_page_groups == 1 &&
                curve.maximum_main_page_groups == 2048,
            "128K sequence page-capacity curve");
    require(curve.bytes_per_additional_main_page_group == 8ULL * 2 * 8 * 132 * 64,
            "int8 per-page reservation includes all eight full-attention layers");
    const auto plan = std::move(planner).finalize(2048);
    require(plan.capacity() == 131072 && plan.kv_capacity() == 131072 &&
                plan.device_reservation_bytes() == curve.reservation_bytes(2048),
            "finalized reservation matches planner curve");
    rejects(
        [&] {
            ik::SequencePlanner bad(options);
            (void)std::move(bad).finalize(2049);
        },
        "planner accepts capacity outside declared curve");
    options.max_context = 0;
    rejects([&] { ik::SequencePlanner bad(options); }, "planner accepts zero logical context");
}

void control_tests() {
    DeviceContext device;
    EngineOptions options;
    options.max_context                          = 256;
    options.kv_capacity                          = KvCapacityPolicy::explicit_capacity(256);
    options.kv_cache                             = KvCacheStorage::Int8Group64;
    options.prefill_chunk                        = 1;
    options.max_concurrency                      = 2;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.use_cuda_graph                       = false;
    const ik::detail::RuntimeModelView model;
    ik::Program program(model, device, options);
    auto prompt = ik::FrontendTestAccess::tokens(128);
    rt::ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 128;
    const auto base                   = program.plan_request(prompt, execution);
    require(base.summary().prompt_tokens == 128 && base.summary().effective_output_tokens == 128 &&
                base.summary().service_work_quanta == 256,
            "request prompt/output capacity accounting");
    require(program.isolated_request_feasible(base), "isolated request fits nominal KV capacity");
    require(base.prefix_shortlist_key(64).has_value() && !base.prefix_shortlist_key(0) &&
                !base.prefix_shortlist_key(128),
            "only proper nonempty prefixes are reusable");
    auto no_identity    = ik::FrontendTestAccess::tokens(128, false);
    const auto no_reuse = program.plan_request(no_identity, execution);
    require(!no_reuse.summary().publish_continuation && !no_reuse.prefix_shortlist_key(64),
            "non-reusable prompt suppresses prefix inspection and publication");
    auto too_full = ik::FrontendTestAccess::tokens(256);
    rejects([&] { (void)program.plan_request(too_full, execution); },
            "full-context prompt leaves no generation room");
    execution.requested_output_tokens = 129;
    const auto capped                 = program.plan_request(prompt, execution);
    require(capped.summary().effective_output_tokens == 128 &&
                capped.summary().effective_limit_reason == FinishReason::ContextCapacity,
            "logical context limits output");
    const auto revision = program.resource_revision();
    auto candidate      = program.inspect_admission(prompt, base, {0}, nullptr, nullptr, {}, true);
    require(candidate && candidate->identity_assessment().physical_status ==
                             rt::MaterializationPhysicalStatus::Feasible,
            "root candidate physically feasible");
    require(!program.inspect_admission(prompt, base, {2}, nullptr, nullptr, {}, true),
            "invalid lane rejected");
    auto resource = program.seal_identity(*candidate, prompt, {});
    require(resource.has_value(), "feasible root plan seals");
    auto stale = *resource;
    std::atomic<bool> cancelled{true};
    require(
        program.start_resource_transaction(std::move(*resource), std::move(prompt), {&cancelled}) ==
                rt::ContextTransactionReserveStatus::Aborted &&
            bool(prompt) && !program.has_context_transaction(),
        "cancelled reserve preserves prompt and owns no transaction");
    cancelled.store(false);
    resource = program.seal_identity(*candidate, prompt, {});
    require(program.start_resource_transaction(std::move(*resource), std::move(prompt), {}) ==
                    rt::ContextTransactionReserveStatus::Reserved &&
                program.has_context_transaction(),
            "resource transaction reserves");
    cancelled.store(true);
    const auto cancelled_progress = program.progress_context_transaction({&cancelled});
    require(std::get<ik::MaterializationResult>(cancelled_progress).status ==
                    rt::ContextTransactionStatus::Aborted &&
                program.physical_usage().device_state_slots == 0,
            "cancelled materialization publishes no sequence");
    program.finalize_context_transaction();
    require(!program.has_context_transaction() && program.resource_revision() == revision,
            "cancelled transaction leaves resource revision unchanged");
    prompt   = ik::FrontendTestAccess::tokens(128);
    resource = program.seal_identity(*candidate, prompt, {});
    require(program.start_resource_transaction(std::move(*resource), std::move(prompt), {}) ==
                rt::ContextTransactionReserveStatus::Reserved,
            "next transaction can reserve after cancellation");
    const auto progress   = program.progress_context_transaction({});
    const auto& published = std::get<ik::MaterializationResult>(progress);
    require(published.status == rt::ContextTransactionStatus::Published &&
                published.published.has_value(),
            "materialization publishes sequence capability");
    const auto sequence = published.published->sequence;
    program.finalize_context_transaction();
    require(program.physical_usage().device_state_slots == 1 &&
                program.resource_revision() != revision,
            "published admission owns exactly one state slot and advances revision");
    auto other = ik::FrontendTestAccess::tokens(128);
    require(!program.inspect_admission(other, base, {0}, nullptr, nullptr, {}, true),
            "active lane cannot admit a second sequence");
    require(program.start_resource_transaction(std::move(stale), std::move(other), {}) ==
                    rt::ContextTransactionReserveStatus::Aborted &&
                bool(other),
            "stale resource proof cannot mutate current ownership");
    auto second = program.inspect_admission(other, base, {1}, nullptr, nullptr, {}, true);
    require(second && second->identity_assessment().physical_status ==
                          rt::MaterializationPhysicalStatus::Infeasible,
            "active sequence's full future KV entitlement prevents over-admission");
    require(!program.seal_identity(*second, other, {}), "infeasible second request cannot seal");
    require(program.abort(sequence).status == rt::ConsumeStatus::Consumed &&
                program.physical_usage().device_state_slots == 0,
            "abort releases active slot and entitlement");
    require(program.abort(sequence).status == rt::ConsumeStatus::InvariantMismatch,
            "released sequence capability cannot be consumed twice");
    second = program.inspect_admission(other, base, {1}, nullptr, nullptr, {}, true);
    require(second && second->identity_assessment().physical_status ==
                          rt::MaterializationPhysicalStatus::Feasible,
            "released entitlement makes queued request feasible");
    program.fail_all_cleanup();
    require(program.physical_usage().device_state_slots == 0 && !program.has_context_transaction(),
            "failure cleanup owns no leaked slots or transactions");

    // Capture pressure: with 2 total device slots (C=1, device_state_slots=1), a catalogued
    // continuation + an active sequence leave 0 free slots, so explicit shared capture must
    // expand and evict the retained private continuation to succeed.
    {
        EngineOptions cap_opts                       = options;
        cap_opts.max_concurrency                     = 1;
        cap_opts.context_cache.device_state_slots    = 1;
        ik::Program cap_prog(model, device, cap_opts);
        auto p1     = ik::FrontendTestAccess::tokens(64);
        auto b1     = cap_prog.plan_request(p1, execution);
        auto c1     = cap_prog.inspect_admission(p1, b1, {0}, nullptr, nullptr, {}, false);
        auto r1     = cap_prog.seal_identity(*c1, p1, {});
        require(cap_prog.start_resource_transaction(std::move(*r1), std::move(p1), {}) ==
                    rt::ContextTransactionReserveStatus::Reserved,
                "first capture pressure request reserves");
        auto s1 = std::get<ik::MaterializationResult>(cap_prog.progress_context_transaction({}))
                      .published->sequence;
        cap_prog.finalize_context_transaction();
        auto f1 = cap_prog.finish(s1);
        require(f1.status == rt::ConsumeStatus::Consumed && f1.continuation.has_value(),
                "first request finishes and retains private continuation");

        auto p2 = ik::FrontendTestAccess::tokens(64);
        auto b2 = cap_prog.plan_request(p2, execution);
        auto c2 = cap_prog.inspect_admission(p2, b2, {0}, nullptr, nullptr, {}, false);
        auto r2 = cap_prog.seal_identity(*c2, p2, {});
        require(cap_prog.start_resource_transaction(std::move(*r2), std::move(p2), {}) ==
                    rt::ContextTransactionReserveStatus::Reserved,
                "second capture pressure request reserves");
        auto s2 = std::get<ik::MaterializationResult>(cap_prog.progress_context_transaction({}))
                      .published->sequence;
        cap_prog.finalize_context_transaction();
        require(cap_prog.physical_usage().device_state_slots == 2,
                "both device state slots are occupied");

        const ik::CaptureOffer offer{s2.id};
        const auto assessment = cap_prog.inspect_capture(offer, nullptr, nullptr, {}, true);
        require(assessment.publishes_shared && !assessment.physically_feasible &&
                    assessment.needs_transfer && !assessment.transfer_requirements.empty() &&
                    assessment.transfer_requirements[0].resource ==
                        rt::ContextResourceClass::State &&
                    assessment.transfer_requirements[0].direction ==
                        rt::ContextTransferDirection::DeviceToDevice &&
                    assessment.transfer_requirements[0].work.payload_bytes > 1000000ULL &&
                    assessment.projected_recovery_work.size() == 2,
                "full slots make shared capture infeasible with D2D transfer requirements");

        const ik::ContinuationHandle* owners[] = {&*f1.continuation};
        const rt::PlanningOwnerId owner_ids[]  = {{0}};
        auto session = cap_prog.begin_capture_pressure_planning(assessment, owners, owner_ids, {},
                                                                {});
        const auto id_target   = session.identity_target();
        const auto id_guide    = session.guidance(id_target);
        require(id_guide.candidate == session.candidate_id() &&
                    id_guide.estimated_machine_work.candidate_transfers[2].payload_bytes >
                        1000000ULL,
                "capture pressure guidance reports D2D transfer work");
        auto id_assessed = session.assess(id_target);
        require(id_assessed.assessment().physical_status ==
                        rt::MaterializationPhysicalStatus::Infeasible &&
                    id_assessed.assessment().expandable,
                "infeasible identity capture target is expandable");
        auto prepared = session.prepare_expansion(id_target);
        require(prepared.new_canonical_count() == 1, "capture expansion produces one victim");
        auto view = session.commit_expansion(std::move(prepared));
        require(view.children.size() == 1 && view.complete, "capture expansion commits child");
        const auto child_guide = session.guidance(view.children[0]);
        require(child_guide.owner_outcomes.size() == 1 &&
                    child_guide.owner_outcomes[0].owner == owner_ids[0] &&
                    child_guide.owner_outcomes[0].disposition == rt::VictimDisposition::Evicted,
                "capture child guidance records eviction outcome");
        auto child_assessed = session.assess(view.children[0]);
        require(child_assessed.assessment().physical_status ==
                        rt::MaterializationPhysicalStatus::Feasible &&
                    child_assessed.assessment().owner_outcomes.size() == 1 &&
                    child_assessed.assessment().owner_outcomes[0].disposition ==
                        rt::VictimDisposition::Evicted,
                "evicting unrelated continuation makes capture feasible");
        auto cap_plan = session.seal(std::move(child_assessed));
        require(cap_plan.has_value() && cap_plan->victim.has_value(), "capture pressure seals");

        std::atomic<bool> cap_cancel{true};
        auto cap_plan_copy = *cap_plan;
        require(cap_prog.reserve_active_capture_with_pressure(
                    ik::CaptureOffer{s2.id}, nullptr, nullptr, {}, true, std::move(cap_plan_copy),
                    {}) == rt::ContextTransactionReserveStatus::Reserved,
                "capture pressure transaction reserves");
        const auto cancelled_cap = std::get<ik::ActiveCaptureResult>(
            cap_prog.progress_context_transaction({&cap_cancel}));
        require(cancelled_cap.status == rt::ContextTransactionStatus::Aborted &&
                    cancelled_cap.victims.size() == 1 &&
                    cancelled_cap.victims[0].disposition == rt::VictimDisposition::Retained &&
                    !cancelled_cap.victims[0].pressure_committed,
                "cancelled capture pressure leaves victim retained and uncommitted");
        cap_prog.finalize_context_transaction();

        require(cap_prog.reserve_active_capture_with_pressure(
                    ik::CaptureOffer{s2.id}, nullptr, nullptr, {}, true, std::move(*cap_plan),
                    {}) == rt::ContextTransactionReserveStatus::Reserved,
                "capture pressure re-reserves after abort");
        const auto committed_cap =
            std::get<ik::ActiveCaptureResult>(cap_prog.progress_context_transaction({}));
        require(committed_cap.status == rt::ContextTransactionStatus::Published &&
                    committed_cap.victims.size() == 1 &&
                    committed_cap.victims[0].disposition == rt::VictimDisposition::Evicted &&
                    committed_cap.victims[0].pressure_committed && committed_cap.shared.has_value() &&
                    committed_cap.transfer_observations.size() == 1 &&
                    committed_cap.transfer_observations[0].resource ==
                        rt::ContextResourceClass::State &&
                    committed_cap.transfer_observations[0].direction ==
                        rt::ContextTransferDirection::DeviceToDevice &&
                    committed_cap.transfer_observations[0].work.payload_bytes > 1000000ULL,
                "capture pressure evicts victim, publishes shared prefix, and observes D2D bytes");
        cap_prog.finalize_context_transaction();
        cap_prog.fail_all_cleanup();
    }
}
} // namespace

int main() {
    try {
        planner_tests();
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
        control_tests();
        std::cout << "PASS: Inkling planner, request budgets, admission entitlements, "
                     "cancellation, stale proofs and slot ownership\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
