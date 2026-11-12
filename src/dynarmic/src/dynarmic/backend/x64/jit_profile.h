// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEST (EDEN_NX2_GPU_PROFILE): what the A64 JIT compiles, per host thread. Each compiled block is
// timed by phase (translate, optimize, emit) and checked against a global PC set: a block whose
// PC another thread already compiled counts as duplicated. Logged every second (Nx2JitEmit) and,
// for 10 ms windows with >= 1 ms of compiling, per window and thread (Nx2JitWin), so the cost can
// be matched against Nx2Stutter frames. Cache invalidations are counted too.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

#include <pthread.h>
#include <fmt/format.h>

#include "common/logging.h"

namespace Dynarmic::Backend::X64::JitProfile {

struct Slot {
    std::atomic<std::uint64_t> blocks{}, ns{}, dup_blocks{}, dup_ns{}, tr_ns{}, opt_ns{}, emit_ns{}, ir_insts{};
    std::atomic<std::uint64_t> w_new_ns{}, w_dup_ns{}, w_blocks{};
    char name[16]{};
};
inline constexpr std::size_t kSlots = 16;
inline std::array<Slot, kSlots> g_slots{};
inline std::atomic<std::size_t> g_next_slot{};
inline std::atomic<std::int64_t> g_last_sec{}, g_last_win{};
inline std::atomic<std::uint64_t> g_recompiles{};
inline std::atomic<std::uint64_t> g_invalidations{}, g_full_invalidations{};
inline std::mutex g_pcs_mutex;
inline std::unordered_set<std::uint64_t> g_pcs;

inline bool On() {
    static const bool on = [] {
        const char* v = std::getenv("EDEN_NX2_GPU_PROFILE");
        return v && *v && *v != '0';
    }();
    return on;
}

inline std::size_t SlotIndex() {
    thread_local const std::size_t slot = [] {
        const std::size_t i = g_next_slot.fetch_add(1) % kSlots;
        pthread_getname_np(pthread_self(), g_slots[i].name, sizeof(g_slots[i].name));
        return i;
    }();
    return slot;
}

inline std::int64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void Record(std::uint64_t pc, std::uint64_t tr, std::uint64_t opt, std::uint64_t emit, std::uint64_t ir_insts) {
    const std::size_t slot = SlotIndex();
    bool dup;
    {
        std::lock_guard lk{g_pcs_mutex};
        bool seen_other = false, seen_self = false;
        for (std::size_t i = 0; i < kSlots; ++i)
            if (g_pcs.count((pc << 4) | i)) (i == slot ? seen_self : seen_other) = true;
        dup = seen_other && !seen_self;
        if (seen_self) g_recompiles.fetch_add(1, std::memory_order_relaxed);
        g_pcs.insert((pc << 4) | slot);
    }
    const std::uint64_t ns = tr + opt + emit;
    auto& s = g_slots[slot];
    constexpr auto r = std::memory_order_relaxed;
    s.blocks.fetch_add(1, r); s.ns.fetch_add(ns, r);
    s.tr_ns.fetch_add(tr, r); s.opt_ns.fetch_add(opt, r); s.emit_ns.fetch_add(emit, r);
    s.ir_insts.fetch_add(ir_insts, r); s.w_blocks.fetch_add(1, r);
    if (dup) { s.dup_blocks.fetch_add(1, r); s.dup_ns.fetch_add(ns, r); s.w_dup_ns.fetch_add(ns, r); }
    else s.w_new_ns.fetch_add(ns, r);

    const std::int64_t now = NowNs();
    const std::int64_t win = now / 10'000'000;
    std::int64_t lw = g_last_win.load(r);
    if (win > lw && g_last_win.compare_exchange_strong(lw, win)) {
        std::uint64_t tot = 0;
        std::string per;
        for (std::size_t i = 0; i < kSlots; ++i) {
            auto& x = g_slots[i];
            const std::uint64_t b = x.w_blocks.exchange(0), n = x.w_new_ns.exchange(0), d = x.w_dup_ns.exchange(0);
            if (!b) continue;
            tot += n + d;
            per += fmt::format(" {}={}b/new{:.2f}/dup{:.2f}", x.name, b, n / 1e6, d / 1e6);
        }
        if (tot >= 1'000'000)
            LOG_INFO(Core, "Nx2JitWin steady_ms={} ms={:.2f}{}", lw * 10, tot / 1e6, per);
    }
    const std::int64_t sec = now / 1'000'000'000;
    std::int64_t ls = g_last_sec.load(r);
    if (sec > ls && g_last_sec.compare_exchange_strong(ls, sec)) {
        std::uint64_t tb = 0, tn = 0, tdb = 0, tdn = 0, ttr = 0, topt = 0, temit = 0, tir = 0;
        std::string per;
        for (std::size_t i = 0; i < kSlots; ++i) {
            auto& x = g_slots[i];
            const std::uint64_t b = x.blocks.exchange(0), n = x.ns.exchange(0);
            const std::uint64_t db = x.dup_blocks.exchange(0), dn = x.dup_ns.exchange(0);
            ttr += x.tr_ns.exchange(0); topt += x.opt_ns.exchange(0); temit += x.emit_ns.exchange(0);
            tir += x.ir_insts.exchange(0);
            if (!b) continue;
            tb += b; tn += n; tdb += db; tdn += dn;
            per += fmt::format(" {}={}/{:.2f}ms/dup{}/{:.2f}ms", x.name, b, n / 1e6, db, dn / 1e6);
        }
        LOG_INFO(Core, "Nx2JitEmit steady_ms={} blocks={} ms={:.2f} dup_blocks={} dup_ms={:.2f} dup_pct={:.1f} "
                       "translate_ms={:.2f} optimize_ms={:.2f} emit_ms={:.2f} ir_insts={} recompiles={} inval={} full_inval={}{}",
                 now / 1'000'000, tb, tn / 1e6, tdb, tdn / 1e6, tn ? 100.0 * tdn / tn : 0.0,
                 ttr / 1e6, topt / 1e6, temit / 1e6, tir, g_recompiles.exchange(0), g_invalidations.exchange(0),
                 g_full_invalidations.exchange(0), per);
    }
}

} // namespace Dynarmic::Backend::X64::JitProfile
