#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::runtime {

template <class Request>
class Scheduler {
public:
    using RequestPtr     = std::shared_ptr<Request>;
    using SequenceHandle = typename Request::SequenceHandle;

    enum class ExecutionAction : std::uint8_t {
        Prefill,
        Decode,
        Wait,
    };

    class AdmissionGrant {
    public:
        AdmissionGrant(AdmissionGrant&&) noexcept            = default;
        AdmissionGrant& operator=(AdmissionGrant&&) noexcept = default;
        AdmissionGrant(const AdmissionGrant&)                = delete;
        AdmissionGrant& operator=(const AdmissionGrant&)     = delete;

        [[nodiscard]] std::uint64_t request_id() const noexcept { return request_id_; }

        [[nodiscard]] BackfillClass backfill_class() const noexcept { return backfill_class_; }

        [[nodiscard]] std::uint64_t protection_epoch() const noexcept { return protection_epoch_; }

        [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept {
            return resource_revision_;
        }

        [[nodiscard]] std::uint64_t service_work_quanta() const noexcept {
            return service_work_quanta_;
        }

    private:
        AdmissionGrant(std::uint64_t request_id, BackfillClass backfill_class,
                       std::uint64_t protection_epoch, ProgramResourceRevision resource_revision,
                       std::uint64_t service_work_quanta) noexcept
            : request_id_(request_id), backfill_class_(backfill_class),
              protection_epoch_(protection_epoch), resource_revision_(resource_revision),
              service_work_quanta_(service_work_quanta) {}

        std::uint64_t request_id_       = 0;
        BackfillClass backfill_class_   = BackfillClass::None;
        std::uint64_t protection_epoch_ = 0;
        ProgramResourceRevision resource_revision_;
        std::uint64_t service_work_quanta_ = 0;

        friend class Scheduler;
    };

    class FifoSnapshot {
    public:
        [[nodiscard]] bool empty() const noexcept { return requests_.empty(); }

        [[nodiscard]] const RequestPtr& head() const {
            if (requests_.empty()) { throw std::logic_error("FIFO snapshot has no head"); }
            return requests_.front();
        }

        [[nodiscard]] std::span<const RequestPtr> backfill_candidates() const noexcept {
            if (requests_.size() <= 1) { return {}; }
            return {requests_.data() + 1, requests_.size() - 1};
        }

    private:
        explicit FifoSnapshot(const std::deque<RequestPtr>& pending)
            : requests_(pending.begin(), pending.end()) {}

        std::vector<RequestPtr> requests_;

        friend class Scheduler;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ControlMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::vector<TokenId> tokens;
        std::uint32_t row_stride = 0;
        std::size_t size         = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    [[nodiscard]] static FifoSnapshot fifo_snapshot(const std::deque<RequestPtr>& pending) {
        return FifoSnapshot(pending);
    }

    template <class Slots>
    [[nodiscard]] RoundMembership build_round_membership(const Slots& slots,
                                                         std::uint32_t max_concurrency) const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_decode_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            if (!request->sequence) {
                throw std::logic_error("decode-ready request has no sequence handle");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.budgets[membership.size]   = RoundBudget{
                  .generated_tokens_remaining =
                    request->output.model_token_budget_remaining(request->budget->remaining()),
            };
            if (membership.budgets[membership.size].generated_tokens_remaining == 0) {
                throw std::logic_error("decode-ready request has no licensed model tokens");
            }
            ++membership.size;
        }
        return membership;
    }

    template <class Slots>
    [[nodiscard]] ControlMembership build_control_membership(const Slots& slots,
                                                             std::uint32_t max_concurrency) const {
        ControlMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_control_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget || !request->sequence) {
                throw std::logic_error("control-ready request has no generation state");
            }
            const std::span<const TokenId> control = request->output.pending_control_tokens();
            if (control.empty() || control.size() > request->budget->remaining()) {
                throw std::logic_error("control-ready request has no admissible control span");
            }
            if (membership.row_stride == 0) {
                membership.row_stride = static_cast<std::uint32_t>(control.size());
                membership.tokens.reserve(static_cast<std::size_t>(membership.row_stride) *
                                          max_concurrency);
            } else if (control.size() != membership.row_stride) {
                throw std::logic_error("compact control membership has ragged target spans");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.tokens.insert(membership.tokens.end(), control.begin(), control.end());
            ++membership.size;
        }
        return membership;
    }

    template <class Slots>
    [[nodiscard]] ActiveAdmissionSet active_admission_set(const Slots& slots,
                                                          std::uint32_t max_concurrency) const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr) { continue; }
            if (request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id     = request->id,
                .backfill_epoch = request->backfill_epoch,
                .backfill_class = request->backfill_class,
            };
        }
        return active;
    }

    static void consume_service_work(Request& request, std::uint64_t work) {
        if (work == 0 || work > request.remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request.remaining_service_work) + " remaining");
        }
        request.remaining_service_work -= work;
    }

    [[nodiscard]] static std::size_t remaining_prefill_tokens(const Request& request) noexcept {
        if constexpr (requires { request.remaining_prefill_tokens(); }) {
            return request.remaining_prefill_tokens();
        } else if constexpr (requires {
                                 request.total_prompt_tokens;
                                 request.prefill_offset;
                             }) {
            return request.total_prompt_tokens > request.prefill_offset
                       ? request.total_prompt_tokens - request.prefill_offset
                       : 0;
        } else {
            return static_cast<std::size_t>(request.remaining_service_work);
        }
    }

    [[nodiscard]] bool should_attempt_admission(bool have_pending, bool admission_check_pending,
                                                bool have_decode, bool previous_unit_was_decode,
                                                bool context_transaction,
                                                bool unsettled_state_fork = false) const noexcept {
        return have_pending && admission_check_pending && !context_transaction &&
               !unsettled_state_fork && (!have_decode || previous_unit_was_decode);
    }

    [[nodiscard]] ExecutionAction
    choose_execution(std::size_t decode_ready_count, bool prefill_runnable,
                     std::uint32_t consecutive_decode_units,
                     bool prefill_single_chunk_remaining = false) const noexcept {
        if (prefill_runnable) {
            if (decode_ready_count == 0) { return ExecutionAction::Prefill; }
            if (prefill_single_chunk_remaining) {
                return consecutive_decode_units == 0 ? ExecutionAction::Decode
                                                     : ExecutionAction::Prefill;
            }
            const std::uint32_t decode_cadence = decode_ready_count >= 2 ? 2U : 1U;
            return consecutive_decode_units < decode_cadence ? ExecutionAction::Decode
                                                             : ExecutionAction::Prefill;
        }
        return decode_ready_count > 0 ? ExecutionAction::Decode : ExecutionAction::Wait;
    }

    [[nodiscard]] ExecutionAction choose_execution(bool have_decode, bool prefill_runnable,
                                                   bool previous_unit_was_decode) const noexcept {
        return choose_execution(have_decode ? 1U : 0U, prefill_runnable,
                                previous_unit_was_decode ? 1U : 0U, false);
    }

    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    select_prefill_lane(const Slots& slots, std::uint32_t max_concurrency) const {
        std::optional<std::uint32_t> best_lane;
        std::size_t best_remaining = 0;
        for (std::size_t i = 0; i < prefill_lane_count_; ++i) {
            const std::uint32_t lane = prefill_lanes_[i];
            if (lane >= max_concurrency || slots[lane] == nullptr ||
                !slots[lane]->is_prefilling()) {
                throw std::logic_error("prefill lane does not match an active prefilling request");
            }
            if (slots[lane]->capture_pending) { continue; }
            const std::size_t remaining = remaining_prefill_tokens(*slots[lane]);
            if (!best_lane || remaining < best_remaining) {
                best_lane      = lane;
                best_remaining = remaining;
            }
        }
        return best_lane;
    }

    [[nodiscard]] std::optional<std::uint32_t> prefill_lane() const noexcept {
        return prefill_lane_count_ > 0 ? std::optional<std::uint32_t>(prefill_lanes_[0])
                                       : std::nullopt;
    }

    [[nodiscard]] std::size_t prefill_lane_count() const noexcept { return prefill_lane_count_; }

    [[nodiscard]] bool owns_prefill_lane(std::uint32_t lane) const noexcept {
        for (std::size_t i = 0; i < prefill_lane_count_; ++i) {
            if (prefill_lanes_[i] == lane) { return true; }
        }
        return false;
    }

    [[nodiscard]] std::optional<std::uint64_t> protection_epoch() const noexcept {
        return protection_ ? std::optional<std::uint64_t>(protection_->epoch_id) : std::nullopt;
    }

    void set_prefill_lane(std::uint32_t lane) {
        if (lane >= kMaximumConcurrency) {
            throw std::logic_error("prefill lane exceeds maximum concurrency");
        }
        if (owns_prefill_lane(lane)) {
            throw std::logic_error("lane already owns staged prefill");
        }
        if (prefill_lane_count_ >= kMaximumConcurrency) {
            throw std::logic_error("staged prefill lane capacity exceeded");
        }
        prefill_lanes_[prefill_lane_count_++] = lane;
    }

    void clear_prefill_lane(std::uint32_t lane) {
        for (std::size_t i = 0; i < prefill_lane_count_; ++i) {
            if (prefill_lanes_[i] == lane) {
                for (std::size_t j = i + 1; j < prefill_lane_count_; ++j) {
                    prefill_lanes_[j - 1] = prefill_lanes_[j];
                }
                --prefill_lane_count_;
                return;
            }
        }
        throw std::logic_error("request does not own staged prefill");
    }

    void observe_fifo_head(std::optional<std::uint64_t> request_id) noexcept {
        if (fifo_head_id_ == request_id) { return; }
        fifo_head_id_ = request_id;
        protection_.reset();
    }

    void on_waiting_removed(std::uint64_t request_id) noexcept {
        if (fifo_head_id_ && *fifo_head_id_ == request_id) {
            fifo_head_id_.reset();
            protection_.reset();
        }
    }

    [[nodiscard]] AdmissionGrant grant_head(std::uint64_t request_id,
                                            std::uint64_t service_work_quanta) const {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id || request_id == 0 ||
            service_work_quanta == 0) {
            throw std::logic_error("head admission is not bound to the observed FIFO head");
        }
        return AdmissionGrant(request_id, BackfillClass::None, 0, {}, service_work_quanta);
    }

    [[nodiscard]] bool protect_blocked_head(std::uint64_t request_id,
                                            std::span<const ActiveAdmissionSnapshot> active,
                                            ProgramResourceRevision resource_revision) {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id) {
            throw std::logic_error("blocked admission does not match the observed FIFO head");
        }
        if (!protection_) {
            protection_.emplace(make_admission_protection(next_protection_epoch_++, request_id,
                                                          resource_revision, active));
        } else if (protection_->head_request_id != request_id) {
            throw std::logic_error("protected head changed without a FIFO transition");
        } else {
            rebind_admission_protection(*protection_, active, resource_revision);
        }
        return protection_has_live_donor(*protection_, active);
    }

    [[nodiscard]] std::optional<AdmissionGrant>
    qualify_backfill(std::uint64_t request_id, std::uint64_t service_work_quanta,
                     std::span<const ActiveAdmissionSnapshot> active,
                     ProgramResourceRevision program_proof_revision) const {
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_) {
            throw std::logic_error("backfill qualification has no open protected head");
        }
        if (request_id == 0 || request_id == *fifo_head_id_ || service_work_quanta == 0) {
            throw std::logic_error("backfill candidate has invalid scheduling identity");
        }
        if (persistent_backfill_is_authorized(*protection_, request_id, active,
                                              program_proof_revision)) {
            return AdmissionGrant(request_id, BackfillClass::Persistent, protection_->epoch_id,
                                  program_proof_revision, service_work_quanta);
        }
        return std::nullopt;
    }

    [[nodiscard]] bool validate_grant(const AdmissionGrant& grant) const noexcept {
        if (grant.request_id_ == 0 || grant.service_work_quanta_ == 0) { return false; }
        if (grant.backfill_class_ == BackfillClass::None) {
            return grant.protection_epoch_ == 0 && grant.resource_revision_.value == 0 &&
                   fifo_head_id_ && *fifo_head_id_ == grant.request_id_;
        }
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_ ||
            protection_->epoch_id != grant.protection_epoch_ ||
            protection_->resource_revision != grant.resource_revision_ ||
            grant.request_id_ == *fifo_head_id_) {
            return false;
        }
        return grant.backfill_class_ == BackfillClass::Persistent;
    }

    void commit_admission(AdmissionGrant&& grant) {
        if (!validate_grant(grant)) {
            throw std::logic_error("admission grant is stale or inconsistent");
        }
        if (grant.backfill_class_ == BackfillClass::None) {
            fifo_head_id_.reset();
            protection_.reset();
            grant.request_id_          = 0;
            grant.service_work_quanta_ = 0;
            return;
        }
        grant.request_id_          = 0;
        grant.service_work_quanta_ = 0;
    }

    void reset() noexcept {
        prefill_lane_count_ = 0;
        fifo_head_id_.reset();
        protection_.reset();
    }

private:
    std::array<std::uint32_t, kMaximumConcurrency> prefill_lanes_{};
    std::size_t prefill_lane_count_ = 0;
    std::optional<std::uint64_t> fifo_head_id_;
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
};

} // namespace ninfer::runtime
