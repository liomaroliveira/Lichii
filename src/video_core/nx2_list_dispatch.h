// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <optional>
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
// That macro only stores the words in three words of the macro data RAM; the macros that upload
// the QMD copy them into its grid words, and the dispatch reads them there. To parse
// the three words the front of the split waited for the back to run the dispatch that writes
// them and for the host GPU to copy the page back, about 13 times a frame, each one a round trip
// through the back and the host GPU.
// Here the front stores a marker instead of the words and sends the list's address; the dispatch
// whose QMD carries the marker takes its grid from that address on the host GPU
// (vkCmdDispatchIndirect, the path KeplerCompute::GetIndirectComputeAddress already has). The back
// also keeps a copy of the words at the list's place in the stream (Nx2KeepListWords), which is
// what anything that needs their values after all reads.
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

/// The data RAM words a macro stores a grid in, x, y and z: read from its code (StoreGridCells).
using GridCells = std::array<u32, 3>;

// The grid no dispatch has: the front stores it for a grid it did not read. Its x carries the
// list's keep id in its low 16 bits (KeepId), so that a dispatch finds its own list even when
// later lists were taken before it launched; its high 16 bits, y and z are fixed. An x of
// 0x7E570000 or more groups is far above every host's maximum; y and z fill the QMD's 16 bits.
inline constexpr u32 MarkerXBase = 0x7E570000;
inline constexpr u32 MarkerY = 0xD15C;
inline constexpr u32 MarkerZ = 0x7E57;

/// The marker of the list kept as `keep_id`, x, y and z.
[[nodiscard]] constexpr std::array<u32, 3> MarkerWords(u64 keep_id) {
    return {MarkerXBase | static_cast<u32>(keep_id & 0xFFFF), MarkerY, MarkerZ};
}

/// Whether QMD grid words 12 to 14 are a marker.
[[nodiscard]] constexpr bool IsMarker(u32 x, u32 y, u32 z) {
    return (x & 0xFFFF0000U) == MarkerXBase && (y & 0xffff) == MarkerY && (z & 0xffff) == MarkerZ;
}

/// The low 16 bits of the keep id a marker's x carries.
[[nodiscard]] constexpr u32 MarkerKeepBits(u32 x) {
    return x & 0xFFFF;
}

/// Whether `value` is word `i` (x, y or z) of a marker, whichever list it stands for.
[[nodiscard]] constexpr bool IsMarkerWord(size_t i, u32 value) {
    switch (i) {
    case 0:
        return (value & 0xFFFF0000U) == MarkerXBase;
    case 1:
        return value == MarkerY;
    case 2:
        return value == MarkerZ;
    default:
        return false;
    }
}

/// Ids for the list words the back keeps (RasterizerInterface::Nx2KeepListWords), shared by
/// every channel. 0 is no id.
inline std::atomic<u64> g_next_keep_id{1};

[[nodiscard]] inline u64 NextKeepId() {
    return g_next_keep_id.fetch_add(1, std::memory_order_relaxed);
}

namespace Detail {

/// A value the store macro computes: a constant of its code, one of its parameters unchanged, or
/// anything else.
struct Value {
    bool known{};
    u32 value{};
    int parameter{-1};
};

/// The ALU result of `op` on two constants, as TuringMacroInterpreter::Execute computes it. False
/// for the operations a store macro may not use.
[[nodiscard]] inline bool Constant(u32 op, u32 x, u32 y, u32 imm, u32& carry, bool& carry_known,
                                   u32& out) {
    switch (op) {
    case 0:
        out = x + y;
        carry = out < x;
        carry_known = true;
        return true;
    case 1:
        if (!carry_known) {
            return false;
        }
        out = x + y + carry;
        return true;
    case 2:
        out = x - y;
        carry = out > x;
        carry_known = true;
        return true;
    case 3:
        if (!carry_known) {
            return false;
        }
        out = x - y - carry;
        return true;
    case 9:
        out = x << (y & 31);
        return true;
    case 10:
        out = x >> (y & 31);
        return true;
    case 11:
        out = static_cast<u32>(static_cast<s32>(x) >> (y & 31));
        return true;
    case 12:
        out = x & y;
        return true;
    case 13:
        out = ~(x & y);
        return true;
    case 14:
        out = x | y;
        return true;
    case 15:
        out = x ^ y;
        return true;
    case 16: {
        const u32 dest = (imm >> 10) & 63, count = (imm >> 5) & 31, source = imm & 31;
        if (dest >= 32) {
            return false;
        }
        const u32 mask = (1U << count) - 1;
        out = (x & ~(mask << dest)) | (((y >> source) & mask) << dest);
        return true;
    }
    default:
        return false;
    }
}

} // namespace Detail

/// If the macro at `start` of `code` does nothing but store its three parameters, unchanged, in
/// three data RAM words: those words, in parameter order. Which words is the macro's own
/// business. The macro is followed value by value, not run on samples: a parameter may only be
/// moved (an addition of a constant zero of the code) and stored; any operation on it, any store
/// of another value, any method write or branch, and the store addresses must be constants of the
/// code. Proven for every parameter value.
[[nodiscard]] inline std::optional<GridCells> StoreGridCells(std::span<const u32> code, u32 start) {
    using Detail::Value;
    constexpr u32 DataRamWrite = 31;
    const u32 instructions = static_cast<u32>(code.size() / 3);
    std::array<Value, 24> regs{};
    std::array<bool, 24> initialized{};
    u32 carry{};
    bool carry_known = true;
    int consumed = 0;
    GridCells cells{~0U, ~0U, ~0U};
    int stores = 0;
    bool ended = false;
    for (u32 pc = start;; ++pc) {
        if (pc >= instructions || pc - start > 16) {
            return std::nullopt;
        }
        const auto d =
            TuringMacroInterpreter::Decode({code[pc * 3 + 2], code[pc * 3 + 1], code[pc * 3]});
        // One path whatever the parameters are: no predicate, branch, loop, register or data RAM
        // read.
        const auto allowed = [](u32 op) { return (op < 22 && op != 7) || op == DataRamWrite; };
        if (d.pred_mode != 0 || !allowed(d.op[0]) || !allowed(d.op[1])) {
            return std::nullopt;
        }
        std::array<Value, 2> load{};
        for (u32 i = 0; i < 2; ++i) {
            if (d.load[i]) {
                if (consumed >= 3) {
                    return std::nullopt;
                }
                load[i] = {false, 0, consumed++};
            }
        }
        const auto get = [&](u32 reg, u32 lane) -> std::optional<Value> {
            if (reg < 24) {
                if (!initialized[reg]) {
                    return std::nullopt;
                }
                return regs[reg];
            }
            switch (reg) {
            case 24:
                return Value{true, 0, -1};
            case 25:
                return Value{true, static_cast<u32>(static_cast<s32>(static_cast<s16>(d.imm[lane]))), -1};
            case 26:
                return Value{true, static_cast<u32>(static_cast<s32>(static_cast<s16>(d.imm[1 - lane]))),
                             -1};
            case 27:
                return Value{true, (d.imm[0] << 16) | d.imm[1], -1};
            case 28:
                return load[0];
            case 29:
                return load[1];
            default:
                return std::nullopt;
            }
        };
        std::array<Value, 2> results{};
        for (u32 lane = 0; lane < 2; ++lane) {
            const auto x = get(d.src_a[lane], lane);
            const auto y = get(d.src_b[lane], lane);
            if (!x || !y) {
                return std::nullopt;
            }
            const u32 op = d.op[lane];
            if (op == DataRamWrite) {
                // The address a constant, the value one parameter as it is, each stored once.
                if (!x->known || y->parameter < 0 || cells[y->parameter] != ~0U ||
                    std::ranges::find(cells, x->value) != cells.end()) {
                    return std::nullopt;
                }
                cells[y->parameter] = x->value;
                ++stores;
                results[lane] = {};
            } else if (op == 0 && ((x->parameter >= 0 && y->known && y->value == 0 && y->parameter < 0) ||
                                   (y->parameter >= 0 && x->known && x->value == 0 &&
                                    x->parameter < 0))) {
                // A move of a parameter.
                results[lane] = x->parameter >= 0 ? *x : *y;
                carry = 0;
                carry_known = true;
            } else if (x->known && y->known) {
                u32 value{};
                if (!Detail::Constant(op, x->value, y->value, d.imm[lane], carry, carry_known,
                                      value)) {
                    return std::nullopt;
                }
                results[lane] = {true, value, -1};
            } else {
                // Anything computed from a parameter: never stored, never an address.
                results[lane] = {false, 0, -1};
                if (op == 0 || op == 2) {
                    carry_known = false;
                }
            }
        }
        for (u32 lane = 0; lane < 2; ++lane) {
            if (d.dst[lane] < 24) {
                regs[d.dst[lane]] = results[lane];
                initialized[d.dst[lane]] = true;
            } else if (d.dst[lane] != 24) {
                return std::nullopt;
            }
            if (d.out_emit[lane] != 0) {
                return std::nullopt;
            }
        }
        if (ended) {
            break;
        }
        ended = d.end;
    }
    if (consumed != 3 || stores != 3) {
        return std::nullopt;
    }
    return cells;
}

/// The grid a channel left on the host GPU: the list its words are kept from, the data RAM words
/// that hold the marker in their place and the marker itself. Shared by the channel's 3D and
/// compute engines, which share the data RAM. Each word is followed on its own: a macro that
/// stores over one of them leaves the others marked.
struct GridMarker {
    ListReader list;
    GridCells cells{};
    std::array<u32, 3> words{};
    bool set{};
    /// The lists taken last, by keep id: a marker copied elsewhere (the engine's copy of a QMD's
    /// grid) finds its own list by the keep id its x carries.
    std::array<ListReader, 64> recent{};

    void Remember(const ListReader& reader) {
        recent[reader.keep_id % recent.size()] = reader;
    }

    /// The list a marker whose x is `x` stands for, if it is still among the recent ones.
    [[nodiscard]] const ListReader* Find(u32 x) const {
        const u32 bits = MarkerKeepBits(x);
        const ListReader& reader = recent[bits % recent.size()];
        return reader.keep_id != 0 && (reader.keep_id & 0xFFFF) == bits ? &reader : nullptr;
    }

    /// Which of the three words of `ram` still hold their marker.
    template <typename Ram>
    [[nodiscard]] std::array<bool, 3> Marked(const Ram& ram) const {
        return {Holds(ram, 0), Holds(ram, 1), Holds(ram, 2)};
    }

    /// Whether word `i` of `ram` still holds its marker.
    template <typename Ram>
    [[nodiscard]] bool Holds(const Ram& ram, size_t i) const {
        return set && ram[cells[i]] == words[i];
    }

    /// Whether any word of `ram` still holds its marker.
    template <typename Ram>
    [[nodiscard]] bool In(const Ram& ram) const {
        return Holds(ram, 0) || Holds(ram, 1) || Holds(ram, 2);
    }
};

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

/// Reads the grid words back into the marker's words of `ram` (the marker is gone). False if they
/// could not be read.
template <typename Ram>
inline bool ResolveGrid(GridMarker& marker, Ram& ram) {
    std::array<u32, 3> words{};
    if (!marker.list.Read(words)) {
        g_guard.failed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // Only the words that still hold the marker: a macro may have stored over the others.
    for (size_t i = 0; i < 3; ++i) {
        if (marker.Holds(ram, i)) {
            ram[marker.cells[i]] = words[i];
        }
    }
    marker.set = false;
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
