// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_list_draw.h"
#include "video_core/macro_turing.h"
#include "video_core/nx2_list_reader.h"

// List dispatches (nx2_list_dispatch.h): compute dispatches whose grid is GPU-written.
//
// The culling of Ocarina of Time writes the grid of its next dispatch (three words: x, y, z) into
// a command list on a GPU-written page, as the parameters of a compute macro call (method 0xe74).
// That macro only stores the words in the macro data RAM (0x20, 0x21, 0x22); the macro that
// uploads the QMD copies them into its grid words, and the dispatch reads them there. To parse
// the three words the front of the split waited for the back to run the dispatch that writes
// them and for the host GPU to copy the page back, about 13 times a frame, each one a round trip
// through the back and the host GPU.
// Here the front stores a marker instead of the words and sends the list's address; the dispatch
// whose QMD carries the marker takes its grid from that address on the host GPU
// (vkCmdDispatchIndirect, the path KeplerCompute::GetIndirectComputeAddress already has).
namespace Tegra::Nx2ListDispatch {

/// EDEN_NX2_LIST_DISPATCH, read once: 1 (the default) = indirect dispatch, 0 = off (the front
/// reads the words back).
[[nodiscard]] inline int Mode() {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_LIST_DISPATCH");
        return value != nullptr ? std::atoi(value) : 1;
    }();
    return mode;
}

// Where the macro stores the grid in the macro data RAM.
inline constexpr u32 DramGridX = 0x20;
inline constexpr u32 DramGridY = 0x21;
inline constexpr u32 DramGridZ = 0x22;

// The grid no dispatch has (above every host's maximum group count on x, y and z each fit the
// QMD's 16 bits): the front stores it for a grid it did not read.
inline constexpr u32 MarkerX = 0x7E57D15C;
inline constexpr u32 MarkerY = 0xD15C;
inline constexpr u32 MarkerZ = 0x7E57;

/// Whether QMD grid words 12 to 14 are the marker.
[[nodiscard]] constexpr bool IsMarker(u32 x, u32 y, u32 z) {
    return x == MarkerX && (y & 0xffff) == MarkerY && (z & 0xffff) == MarkerZ;
}

/// Whether the macro at `start` of `code` does nothing but store its three parameters in the data
/// RAM at DramGridX, DramGridY and DramGridZ, whatever they are.
[[nodiscard]] inline bool IsStoreGridMacro(std::span<const u32> code, u32 start) {
    // One path whatever the parameters are: no branch, loop, predicate, register or data RAM
    // read up to the instruction after the END.
    constexpr u32 DataRamWrite = 31;
    const u32 instructions = static_cast<u32>(code.size() / 3);
    bool ended = false;
    for (u32 pc = start;; ++pc) {
        if (pc >= instructions || pc - start > 8) {
            return false;
        }
        const auto decoded =
            TuringMacroInterpreter::Decode({code[pc * 3 + 2], code[pc * 3 + 1], code[pc * 3]});
        const auto allowed = [](u32 op) { return (op < 22 && op != 7) || op == DataRamWrite; };
        if (decoded.pred_mode != 0 || !allowed(decoded.op[0]) || !allowed(decoded.op[1])) {
            return false;
        }
        if (ended) {
            break;
        }
        ended = decoded.end;
    }
    // What it stores, for parameters that between them have every bit set and clear.
    constexpr std::array<std::array<u32, 3>, 3> samples{{
        {0xFFFFFFFFU, 0U, 0x55555555U},
        {0U, 0xFFFFFFFFU, 0xAAAAAAAAU},
        {0x12345678U, 0x9ABCDEF0U, 0x0FEDCBA9U}}};
    for (const auto& sample : samples) {
        TuringMacroInterpreter probe;
        constexpr u32 Untouched = 0xDEAD0000;
        for (u32 word = 0; word < probe.dram.size(); ++word) {
            probe.dram[word] = Untouched + word;
        }
        std::vector<u32> parameters{sample[0], sample[1], sample[2]};
        bool emitted = false;
        const auto result = probe.Execute(
            code, start, parameters, [](u32) { return 0U; },
            [&emitted](u32, u32) {
                emitted = true;
                return true;
            });
        if (!result.success || result.consumed != 3 || emitted) {
            return false;
        }
        for (u32 word = 0; word < probe.dram.size(); ++word) {
            const u32 expected = word == DramGridX   ? sample[0]
                                 : word == DramGridY ? sample[1]
                                 : word == DramGridZ ? sample[2]
                                                     : Untouched + word;
            if (probe.dram[word] != expected) {
                return false;
            }
        }
    }
    return true;
}

/// Whether the three grid words of the data RAM `ram` hold the marker.
template <typename Ram>
[[nodiscard]] inline bool HasMarker(const Ram& ram) {
    return ram[DramGridX] == MarkerX && ram[DramGridY] == MarkerY && ram[DramGridZ] == MarkerZ;
}

/// EDEN_NX2_LIST_GUARD, read once: 2 (the default) = the words a list draw or a list dispatch
/// left on the host GPU are read back before anything can use their values, 1 = what could use
/// them is only counted, 0 = off.
[[nodiscard]] inline int GuardMode() {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_LIST_GUARD");
        return value != nullptr ? std::atoi(value) : 2;
    }();
    return mode;
}

struct GuardStats {
    std::atomic<u64> dram_macros{};       ///< 3D macros that use the data RAM, on the marker
    std::atomic<u64> compute_unaffected{}; ///< Compute macros on the marker that do not read it
    std::atomic<u64> compute_copies{};    ///< ... that copy it into an upload (the QMD's grid)
    std::atomic<u64> compute_unsafe{};    ///< ... that use it otherwise, or cannot be followed
    std::atomic<u64> grid_resolves{};     ///< The grid words read back
    std::atomic<u64> stale_reads{};       ///< Macro reads of the two registers a list draw left
    std::atomic<u64> stale_draws{};       ///< Array draws that would use them
    std::atomic<u64> stale_back_macros{}; ///< Macros sent to the back while they are stale
    std::atomic<u64> draw_resolves{};     ///< The two words read back
    std::atomic<u64> failed{};            ///< Words that could not be read back
};
inline GuardStats g_guard;

/// Reads the grid words back into `ram` (the marker is gone). False if they could not be read.
template <typename Ram>
inline bool ResolveGrid(const ListReader& reader, Ram& ram) {
    std::array<u32, 3> words{};
    if (!reader.Read(words)) {
        g_guard.failed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    ram[DramGridX] = words[0];
    ram[DramGridY] = words[1];
    ram[DramGridZ] = words[2];
    g_guard.grid_resolves.fetch_add(1, std::memory_order_relaxed);
    return true;
}

struct Stats {
    std::atomic<u64> taken{};     ///< Front: lists sent as an address
    std::atomic<u64> refused{};   ///< Front: three-word compute lists of another macro or shape
    std::atomic<u64> indirect{};  ///< Back: dispatched from the list's words on the host GPU
    std::atomic<u64> orphan{};    ///< Back: a QMD with the marker and no list address: skipped
};
inline Stats g_stats;

inline void Count(std::atomic<u64>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

/// Front thread. Logged with EDEN_NX2_LIST_DIAG.
inline void MaybeLog() {
    if (!Nx2ListDraw::Diagnostics()) {
        return;
    }
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds{2}) {
        return;
    }
    window = now;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 taken = take(g_stats.taken);
    const u64 refused = take(g_stats.refused);
    const u64 indirect = take(g_stats.indirect);
    const u64 orphan = take(g_stats.orphan);
    if (taken + refused + indirect + orphan != 0) {
        LOG_INFO(HW_GPU,
                 "List dispatch per 2 s: {} lists taken, {} refused | {} indirect dispatches, {} "
                 "markers without a list",
                 taken, refused, indirect, orphan);
    }
    if (GuardMode() != 0) {
        LOG_INFO(HW_GPU,
                 "List guard per 2 s: grid: {} 3D macros with data RAM, compute macros {} "
                 "unaffected / {} copies / {} unsafe, {} read back | draws: {} register reads, {} "
                 "array draws, {} back macros, {} read back | {} failed",
                 take(g_guard.dram_macros), take(g_guard.compute_unaffected),
                 take(g_guard.compute_copies), take(g_guard.compute_unsafe),
                 take(g_guard.grid_resolves), take(g_guard.stale_reads),
                 take(g_guard.stale_draws), take(g_guard.stale_back_macros),
                 take(g_guard.draw_resolves), take(g_guard.failed));
    }
}

} // namespace Tegra::Nx2ListDispatch
