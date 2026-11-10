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
#include "video_core/macro_turing.h"

// List draws (nx2_list_draw.h): array draws whose vertex count and first vertex are GPU-written.
//
// The culling of Ocarina of Time writes, for each object, the two parameters of a macro call (3D
// method 0xe62: draw.begin, vertex_buffer.first and count, draw.end) into a command list on a
// GPU-written page. To parse them the front of the split waited for the back to run the work that
// writes them and for the host GPU to copy the page back (about 1,900 times a second), and every
// copy was submitted on its own. The macro only moves the words into the two registers, and only
// the draw reads those: the front sends the address of the list instead, and the back draws
// indirectly from arguments the host GPU builds from the two words (RasterizerVulkan).
namespace Tegra::Nx2ListDraw {

/// EDEN_NX2_LIST_DRAW, read once: 1 (the default) = indirect draw, 0 = off (the front reads the
/// words back), 2 = the front does not read the words, the back does at the draw (a check).
[[nodiscard]] inline int Mode() {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_LIST_DRAW");
        return value != nullptr ? std::atoi(value) : 1;
    }();
    return mode;
}

/// EDEN_NX2_LIST_DIAG: the counters below are logged every 2 s.
[[nodiscard]] inline bool Diagnostics() {
    static const bool enabled = std::getenv("EDEN_NX2_LIST_DIAG") != nullptr;
    return enabled;
}

// The 3D registers the macro writes (checked against Maxwell3D::Regs where it is used).
inline constexpr u32 VertexFirst = 0x35d;
inline constexpr u32 VertexCount = 0x35e;
inline constexpr u32 DrawEnd = 0x585;
inline constexpr u32 DrawBegin = 0x586;

/// Whether the macro at `start` of `code` does nothing but write 0 to draw.begin, its second
/// parameter to vertex_buffer.first, its first to vertex_buffer.count and 0 to draw.end, in that
/// order, whatever the two parameters are.
[[nodiscard]] inline bool IsArrayDrawMacro(std::span<const u32> code, u32 start) {
    const auto effects = TuringMacroInterpreter::AnalyzeEffects(code, start);
    constexpr std::array<u32, 4> written{VertexFirst, VertexCount, DrawEnd, DrawBegin};
    if (!effects.methods_known || effects.data_ram ||
        !std::ranges::equal(effects.methods, written)) {
        return false;
    }
    // One path whatever the parameters are: no branch, loop, predicate, register or data RAM
    // read up to the instruction after the END.
    const u32 instructions = static_cast<u32>(code.size() / 3);
    bool ended = false;
    for (u32 pc = start;; ++pc) {
        if (pc >= instructions || pc - start > 16) {
            return false;
        }
        const auto decoded =
            TuringMacroInterpreter::Decode({code[pc * 3 + 2], code[pc * 3 + 1], code[pc * 3]});
        if (decoded.pred_mode != 0 || decoded.op[0] >= 22 || decoded.op[1] >= 22 ||
            decoded.op[0] == 7 || decoded.op[1] == 7) {
            return false;
        }
        if (ended) {
            break;
        }
        ended = decoded.end;
    }
    // What it writes, for parameters that between them have every bit set and clear.
    TuringMacroInterpreter probe;
    constexpr std::array<std::array<u32, 2>, 3> samples{{
        {0xFFFFFFFFU, 0U}, {0U, 0xFFFFFFFFU}, {0x12345678U, 0x9ABCDEF0U}}};
    for (const auto& sample : samples) {
        std::vector<u32> parameters{sample[0], sample[1]};
        std::vector<std::pair<u32, u32>> emits;
        const auto result = probe.Execute(
            code, start, parameters, [](u32) { return 0U; },
            [&emits](u32 address, u32 value) {
                emits.emplace_back(address, value);
                return true;
            });
        const std::array<std::pair<u32, u32>, 4> expected{{
            {DrawBegin, 0U}, {VertexFirst, sample[1]}, {VertexCount, sample[0]}, {DrawEnd, 0U}}};
        if (!result.success || result.consumed != 2 || !std::ranges::equal(emits, expected)) {
            return false;
        }
    }
    return true;
}

struct Stats {
    std::atomic<u64> taken{};         ///< Front: lists sent as an address
    std::atomic<u64> refused{};       ///< Front: two-word lists of another macro or call shape
    std::atomic<u64> indirect{};      ///< Back: drawn indirectly, vertex pipelines
    std::atomic<u64> indirect_mesh{}; ///< Back: drawn indirectly, task/mesh pipelines
    std::atomic<u64> read_back{};     ///< Back: words read back at the draw
    std::atomic<u64> read_zero{};     ///< Back: of those, no vertices
    std::atomic<u64> read_first{};    ///< Back: of those, first vertex not 0
    std::atomic<u64> read_mesh{};     ///< Back: of those, task/mesh pipelines
    std::atomic<u64> not_recorded{};  ///< Back: no pipeline or buffer, nothing drawn
};
inline Stats g_stats;

inline void Count(std::atomic<u64>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

/// Front thread.
inline void MaybeLog() {
    if (!Diagnostics()) {
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
    const u64 indirect_mesh = take(g_stats.indirect_mesh);
    const u64 read_back = take(g_stats.read_back);
    const u64 read_zero = take(g_stats.read_zero);
    const u64 read_first = take(g_stats.read_first);
    const u64 read_mesh = take(g_stats.read_mesh);
    const u64 not_recorded = take(g_stats.not_recorded);
    LOG_INFO(HW_GPU,
             "list draw 2s: {} lists taken, {} refused | back: {} indirect, {} indirect mesh, {} "
             "read back ({} with no vertices, {} with a first vertex, {} mesh), {} not recorded",
             taken, refused, indirect, indirect_mesh, read_back, read_zero, read_first, read_mesh,
             not_recorded);
}

} // namespace Tegra::Nx2ListDraw
