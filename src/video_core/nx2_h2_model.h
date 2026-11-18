// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_profile.h"

// EDEN_NX2_H2_MODEL=1 (E04.T10): models the ceiling of a batched 50/50 GPUBack split between
// Side A (method dispatch, TextureCache, PipelineLookup, RenderTargets) and Side B (BufferCache,
// descriptor queue/binding, draw recording), with DynamicState tracked separately so both
// placements (Dyn on B vs Dyn on A) are evaluated simultaneously.
//
// Two barrier scenarios are accumulated in parallel:
// - C1 (conservative): every cross-side or shared-scheduler/descriptor-queue/RT synchronization.
// - C2 (optimistic): only real data/control barriers between BufferCache and TextureCache/host.
namespace VideoCommon::Nx2H2Model {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_H2_MODEL");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

enum class Side : u8 {
    Idle = 0,
    A = 1,
    B = 2,
    Dyn = 3,
};

enum class Barrier : u8 {
    // C2 barriers (real data/control barriers; also counted in C1)
    TexBufferBind = 0,
    DmaBufImg,
    Compute,
    FlushInvalBoth,
    Fence,
    Submit,
    Call,
    // C1-only extra barriers (conservative scheduler / descriptor queue / GPU-only coupling)
    Wfi,
    TexCacheSched,
    FeedbackBreak,
    PushImgDesc,
    RtOrClear,
    Count,
};

[[nodiscard]] constexpr bool IsC2Barrier(Barrier kind) {
    return static_cast<u8>(kind) <= static_cast<u8>(Barrier::Call);
}

[[nodiscard]] constexpr size_t BucketIndex(u64 draws) {
    if (draws <= 1) {
        return 0;
    }
    if (draws <= 3) {
        return 1;
    }
    if (draws <= 7) {
        return 2;
    }
    if (draws <= 15) {
        return 3;
    }
    if (draws <= 31) {
        return 4;
    }
    if (draws <= 63) {
        return 5;
    }
    return 6;
}

struct Scenario {
    u64 batch_a{};
    u64 batch_b{};
    u64 batch_dyn{};
    u64 batch_draws{};

    u64 sum_serial{};
    u64 sum_par_dyn_b{};
    u64 sum_par_dyn_a{};

    u64 batches_with_draws{};
    u64 draws_in_batches{};
    u64 draws_ge8{};
    u64 draws_ge16{};
    u64 barriers_total{};
    u64 total_drains{};
    std::array<u64, 7> hist{};

    void FlushBatch(bool count_barrier) {
        const u64 serial = batch_a + batch_b + batch_dyn;
        if (serial > 0) {
            sum_serial += serial;
            sum_par_dyn_b += (std::max)(batch_a, batch_b + batch_dyn);
            sum_par_dyn_a += (std::max)(batch_a + batch_dyn, batch_b);
        }
        if (batch_draws > 0) {
            ++batches_with_draws;
            draws_in_batches += batch_draws;
            if (batch_draws >= 8) {
                draws_ge8 += batch_draws;
            }
            if (batch_draws >= 16) {
                draws_ge16 += batch_draws;
            }
            ++hist[BucketIndex(batch_draws)];
            if (count_barrier) {
                ++total_drains;
            }
        }
        if (count_barrier) {
            ++barriers_total;
        }
        batch_a = 0;
        batch_b = 0;
        batch_dyn = 0;
        batch_draws = 0;
    }

    void Reset() {
        *this = Scenario{};
    }
};

struct ThreadState {
    bool is_back_thread{false};
    Side side{Side::Idle};
    u64 last_tick{};
    std::array<u64, 4> total_ticks{};
    u64 total_draws{};
    std::array<u64, static_cast<size_t>(Barrier::Count)> barrier_counts{};
    Scenario c1{};
    Scenario c2{};
    u64 window_start_ns{};
    u64 window_start_ticks{};
    const void* last_fb{nullptr};
};

inline thread_local ThreadState t_state{};

inline void MarkBackThread() {
    t_state.is_back_thread = true;
    if (Enabled()) [[unlikely]] {
        t_state.side = Side::Idle;
        t_state.last_tick = Nx2Profile::Ticks();
        t_state.window_start_ns = Nx2Profile::Now();
        t_state.window_start_ticks = t_state.last_tick;
    }
}

inline void AccumulateUntil(u64 now_tick) {
    auto& state = t_state;
    if (state.last_tick == 0) {
        state.last_tick = now_tick;
        return;
    }
    const u64 dt = now_tick - state.last_tick;
    state.last_tick = now_tick;
    state.total_ticks[static_cast<size_t>(state.side)] += dt;
    switch (state.side) {
    case Side::A:
        state.c1.batch_a += dt;
        state.c2.batch_a += dt;
        break;
    case Side::B:
        state.c1.batch_b += dt;
        state.c2.batch_b += dt;
        break;
    case Side::Dyn:
        state.c1.batch_dyn += dt;
        state.c2.batch_dyn += dt;
        break;
    case Side::Idle:
        break;
    }
}

inline Side Switch(Side new_side) {
    auto& state = t_state;
    const Side prev = state.side;
    if (prev == new_side) {
        return prev;
    }
    AccumulateUntil(Nx2Profile::Ticks());
    state.side = new_side;
    return prev;
}

class SideScope {
public:
    explicit SideScope(Side new_side) {
        if (Enabled()) [[unlikely]] {
            if (t_state.is_back_thread) {
                active = true;
                prev = Switch(new_side);
            }
        }
    }

    ~SideScope() {
        if (active) [[unlikely]] {
            Switch(prev);
        }
    }

    SideScope(const SideScope&) = delete;
    SideScope& operator=(const SideScope&) = delete;

private:
    Side prev{Side::Idle};
    bool active{false};
};

inline void NoteDraw() {
    if (!Enabled()) [[likely]] {
        return;
    }
    if (!t_state.is_back_thread) {
        return;
    }
    ++t_state.total_draws;
    ++t_state.c1.batch_draws;
    ++t_state.c2.batch_draws;
}

inline void NoteBarrier(Barrier kind) {
    if (!Enabled()) [[likely]] {
        return;
    }
    if (!t_state.is_back_thread) {
        return;
    }
    AccumulateUntil(Nx2Profile::Ticks());
    ++t_state.barrier_counts[static_cast<size_t>(kind)];
    t_state.c1.FlushBatch(true);
    if (IsC2Barrier(kind)) {
        t_state.c2.FlushBatch(true);
    }
}

inline void NoteFramebuffer(const void* fb) {
    if (!Enabled()) [[likely]] {
        return;
    }
    if (!t_state.is_back_thread) {
        return;
    }
    if (t_state.last_fb != nullptr && fb != t_state.last_fb) {
        NoteBarrier(Barrier::RtOrClear);
    }
    t_state.last_fb = fb;
}

inline void MaybeReport() {
    if (!Enabled()) [[likely]] {
        return;
    }
    auto& state = t_state;
    if (!state.is_back_thread) {
        return;
    }
    const u64 now_ns = Nx2Profile::Now();
    if (state.window_start_ns == 0) {
        state.window_start_ns = now_ns;
        state.window_start_ticks = Nx2Profile::Ticks();
        state.last_tick = state.window_start_ticks;
        return;
    }
    const u64 window_ns = now_ns - state.window_start_ns;
    if (window_ns < 10'000'000'000ULL) {
        return;
    }
    const u64 now_ticks = Nx2Profile::Ticks();
    AccumulateUntil(now_ticks);
    state.c1.FlushBatch(false);
    state.c2.FlushBatch(false);

    const u64 window_ticks = (std::max<u64>)(now_ticks - state.window_start_ticks, 1);
    const double inv_sec = 1e9 / static_cast<double>(window_ns);
    const double ticks_to_ms_per_s = 1000.0 / static_cast<double>(window_ticks);

    const u64 ticks_a = state.total_ticks[static_cast<size_t>(Side::A)];
    const u64 ticks_b = state.total_ticks[static_cast<size_t>(Side::B)];
    const u64 ticks_dyn = state.total_ticks[static_cast<size_t>(Side::Dyn)];
    const u64 ticks_active = ticks_a + ticks_b + ticks_dyn;

    const auto pct = [](u64 part, u64 total) -> double {
        return total == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(total);
    };
    const auto speedup_pct = [](u64 serial, u64 parallel) -> double {
        return parallel == 0
                   ? 0.0
                   : (static_cast<double>(serial) / static_cast<double>(parallel) - 1.0) * 100.0;
    };

    const u64 ideal_par_dyn_b = (std::max)(ticks_a, ticks_b + ticks_dyn);
    const u64 ideal_par_dyn_a = (std::max)(ticks_a + ticks_dyn, ticks_b);

    LOG_INFO(HW_GPU,
             "Nx2H2Model balance per second: active {:.1f} ms/s | A {:.1f} ({:.1f}%) B {:.1f} "
             "({:.1f}%) Dyn {:.1f} ({:.1f}%) | ideal DynB {:+.1f}% DynA {:+.1f}% | draws {:.0f}/s",
             static_cast<double>(ticks_active) * ticks_to_ms_per_s,
             static_cast<double>(ticks_a) * ticks_to_ms_per_s, pct(ticks_a, ticks_active),
             static_cast<double>(ticks_b) * ticks_to_ms_per_s, pct(ticks_b, ticks_active),
             static_cast<double>(ticks_dyn) * ticks_to_ms_per_s, pct(ticks_dyn, ticks_active),
             speedup_pct(ticks_active, ideal_par_dyn_b),
             speedup_pct(ticks_active, ideal_par_dyn_a),
             static_cast<double>(state.total_draws) * inv_sec);

    const auto format_scenario = [&](const char* label, const Scenario& sc) {
        const double avg_draws =
            sc.batches_with_draws == 0
                ? 0.0
                : static_cast<double>(sc.draws_in_batches) / static_cast<double>(sc.batches_with_draws);
        return fmt::format(
            "{} ceil_DynB {:+.1f}% ceil_DynA {:+.1f}% batches {:.0f}/s avg_draws {:.1f} "
            "ge8 {:.1f}% ge16 {:.1f}% (hist 1:{:.0f} 2-3:{:.0f} 4-7:{:.0f} 8-15:{:.0f} "
            "16-31:{:.0f} 32-63:{:.0f} 64+:{:.0f})",
            label, speedup_pct(sc.sum_serial, sc.sum_par_dyn_b),
            speedup_pct(sc.sum_serial, sc.sum_par_dyn_a),
            static_cast<double>(sc.batches_with_draws) * inv_sec, avg_draws,
            pct(sc.draws_ge8, sc.draws_in_batches), pct(sc.draws_ge16, sc.draws_in_batches),
            static_cast<double>(sc.hist[0]) * inv_sec, static_cast<double>(sc.hist[1]) * inv_sec,
            static_cast<double>(sc.hist[2]) * inv_sec, static_cast<double>(sc.hist[3]) * inv_sec,
            static_cast<double>(sc.hist[4]) * inv_sec, static_cast<double>(sc.hist[5]) * inv_sec,
            static_cast<double>(sc.hist[6]) * inv_sec);
    };

    LOG_INFO(HW_GPU, "Nx2H2Model scenarios: {} | {}", format_scenario("C1(all)", state.c1),
             format_scenario("C2(real)", state.c2));

    const auto b_rate = [&](Barrier b) -> double {
        return static_cast<double>(state.barrier_counts[static_cast<size_t>(b)]) * inv_sec;
    };

    LOG_INFO(HW_GPU,
             "Nx2H2Model barriers per second: C2[texbuf {:.0f} dmabi {:.0f} compute {:.0f} "
             "flushinval {:.0f} fence {:.0f} submit {:.0f} call {:.0f}] C1extra[wfi {:.0f} "
             "tcsched {:.0f} feedback {:.0f} pushimg {:.0f} rtclear {:.0f}]",
             b_rate(Barrier::TexBufferBind), b_rate(Barrier::DmaBufImg), b_rate(Barrier::Compute),
             b_rate(Barrier::FlushInvalBoth), b_rate(Barrier::Fence), b_rate(Barrier::Submit),
             b_rate(Barrier::Call), b_rate(Barrier::Wfi), b_rate(Barrier::TexCacheSched),
             b_rate(Barrier::FeedbackBreak), b_rate(Barrier::PushImgDesc),
             b_rate(Barrier::RtOrClear));

    state.total_ticks.fill(0);
    state.total_draws = 0;
    state.barrier_counts.fill(0);
    state.c1.Reset();
    state.c2.Reset();
    state.window_start_ns = now_ns;
    state.window_start_ticks = now_ticks;
    state.last_tick = now_ticks;
}

} // namespace VideoCommon::Nx2H2Model
