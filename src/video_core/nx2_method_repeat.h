// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEST (E15.T1, go/no-go for method memoization): how often the run of methods the split's back
// replays between two draws (a "segment") repeats one from the previous frame, and how much of
// the method replay time those repeats take. EDEN_NX2_METHOD_REPEAT=1 turns it on; off it costs
// one predictable branch per entry and per draw.
//
// A segment is every Engine, Method and Multi entry from one draw up to and including the method
// that triggers the next one. Two hashes per segment:
//   full   = engines, methods, arguments and Multi data (what a memo of the resolved state needs)
//   struct = the same without the constant buffer data words (const_buffer.buffer[0..15])
// Batching details are left out (IsLast, the DMA segment word, Multi's methods pending). Call
// entries add their thunk (not the payload) to the hash and their time to a separate total.
// Each frame (RendererVulkan::Composite) the current set of hashes becomes the previous one.
// Time is per entry with the TSC (two reads per entry: the replay runs slower while on, the
// shares are what count). Every 10 s the log prints the shares; at the end of the run, the
// cumulative ones.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::MethodRepeat {

inline const bool enabled = [] {
    const char* value = std::getenv("EDEN_NX2_METHOD_REPEAT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}();

inline bool Enabled() {
    return enabled;
}

/// Frames composited so far (any thread).
inline std::atomic<u64> frames{0};

inline void FrameEnd() {
    if (enabled) [[unlikely]] {
        frames.fetch_add(1, std::memory_order_relaxed);
    }
}

class Counter {
public:
    /// Constant buffer data methods: Maxwell3D's const_buffer.buffer[0..15].
    void SetConstBufferData(u32 first) {
        cb_data = first;
    }

    /// Wraps one entry of the back's ring. kind/flags as GpuSplit's: 0 Wrap, 1 Engine, 2 Method,
    /// 3 Multi, 4 Call.
    template <typename Run>
    size_t Execute(const u64* entry, Run&& run) {
        const u64 head = entry[0];
        const u32 kind = static_cast<u32>(head & 0xf);
        if (kind == 0) {
            return run();
        }
        HashBefore(entry, kind);
        in_entry = true;
        entry_open = true;
        entry_kind = kind;
        entry_start = Ticks();
        const size_t consumed = run();
        const u64 end = Ticks();
        in_entry = false;
        if (entry_open) {
            AddTicks(kind, end - entry_start);
        }
        // Constant buffer words in a row go up in one entry (fast path): hash the rest of them.
        if (kind == 2 && consumed > 1 && (head & HasSegment) == 0) {
            for (size_t i = 1; i < consumed; ++i) {
                HashMethodWord(entry[i]);
            }
        }
        MaybeReport(end);
        return consumed;
    }

    /// A draw starts: close the segment (the entry running now stops counting time).
    void Draw() {
        if (!in_entry) {
            return;
        }
        const u64 now = Ticks();
        if (entry_open) {
            AddTicks(entry_kind, now - entry_start);
            entry_open = false;
        }
        CloseSegment();
    }

private:
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr size_t TableSize = 16384; // power of two
    static constexpr size_t MaxPerFrame = TableSize / 2;

    struct Table {
        std::vector<u64> slots = std::vector<u64>(TableSize, 0);
        size_t used = 0;

        bool Contains(u64 key) const {
            for (size_t i = key & (TableSize - 1);; i = (i + 1) & (TableSize - 1)) {
                if (slots[i] == key) {
                    return true;
                }
                if (slots[i] == 0) {
                    return false;
                }
            }
        }
        bool Insert(u64 key) {
            if (used >= MaxPerFrame) {
                return false;
            }
            for (size_t i = key & (TableSize - 1);; i = (i + 1) & (TableSize - 1)) {
                if (slots[i] == key) {
                    return true;
                }
                if (slots[i] == 0) {
                    slots[i] = key;
                    ++used;
                    return true;
                }
            }
        }
        void Clear() {
            std::fill(slots.begin(), slots.end(), 0);
            used = 0;
        }
    };

    struct Totals {
        u64 segments{}, methods{}, ticks{}, call_ticks{};
        u64 full_prev{}, full_prev_methods{}, full_prev_ticks{};
        u64 full_same{}, full_same_ticks{};
        u64 struct_prev{}, struct_prev_ticks{};
        u64 overflow{}, frames{};
    };

    static u64 Ticks() {
#if defined(__x86_64__) || defined(_M_X64)
        return __rdtsc();
#else
        return NowNs();
#endif
    }
    static u64 NowNs() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }

    static u64 Mix(u64 h, u64 w) {
        h ^= w + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h * 0xff51afd7ed558ccdULL;
    }

    void HashBoth(u64 w) {
        full = Mix(full, w);
        structure = Mix(structure, w);
    }

    bool IsConstBufferData(u32 method) const {
        return cb_data != 0 && method - cb_data < 16;
    }

    void HashMethodWord(u64 head) {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u64 key = head & ~(IsLast | HasSegment);
        full = Mix(full, key);
        structure = Mix(structure, IsConstBufferData(method) ? (key & 0xffffffffULL) : key);
        ++methods;
    }

    void HashBefore(const u64* entry, u32 kind) {
        const u64 head = entry[0];
        switch (kind) {
        case 1: // Engine
            HashBoth(0xE0ULL ^ entry[1]);
            break;
        case 2: // Method
            HashMethodWord(head);
            break;
        case 3: { // Multi
            const u32 method = static_cast<u32>((head >> 8) & 0xffff);
            const u32 amount = static_cast<u32>(head >> 32);
            const size_t data = (head & HasSegment) != 0 ? 3 : 2;
            HashBoth(head & ~(IsLast | HasSegment));
            const u32* words = reinterpret_cast<const u32*>(entry + data);
            const bool cb = IsConstBufferData(method);
            for (u32 i = 0; i < amount; ++i) {
                full = Mix(full, words[i]);
                if (!cb) {
                    structure = Mix(structure, words[i]);
                }
            }
            methods += amount;
            break;
        }
        case 4: // Call: the thunk only
            HashBoth(0xCA11ULL ^ entry[1]);
            break;
        default:
            break;
        }
    }

    void AddTicks(u32 kind, u64 ticks) {
        if (kind == 4) {
            call_ticks += ticks;
        } else {
            seg_ticks += ticks;
        }
    }

    void CloseSegment() {
        const u64 frame = frames.load(std::memory_order_relaxed);
        if (frame != current_frame) {
            std::swap(prev_full, cur_full);
            std::swap(prev_struct, cur_struct);
            cur_full.Clear();
            cur_struct.Clear();
            window.frames += frame - current_frame;
            current_frame = frame;
        }
        const u64 kf = full | 1; // 0 marks an empty slot
        const u64 ks = structure | 1;
        Totals& t = window;
        ++t.segments;
        t.methods += methods;
        t.ticks += seg_ticks;
        t.call_ticks += call_ticks;
        if (prev_full.Contains(kf)) {
            ++t.full_prev;
            t.full_prev_methods += methods;
            t.full_prev_ticks += seg_ticks;
        } else if (cur_full.Contains(kf)) {
            ++t.full_same;
            t.full_same_ticks += seg_ticks;
        }
        if (prev_struct.Contains(ks)) {
            ++t.struct_prev;
            t.struct_prev_ticks += seg_ticks;
        }
        if (!cur_full.Insert(kf) || !cur_struct.Insert(ks)) {
            ++t.overflow;
        }
        full = 0;
        structure = 0;
        methods = 0;
        seg_ticks = 0;
        call_ticks = 0;
    }

    static void Add(Totals& into, const Totals& from) {
        into.segments += from.segments;
        into.methods += from.methods;
        into.ticks += from.ticks;
        into.call_ticks += from.call_ticks;
        into.full_prev += from.full_prev;
        into.full_prev_methods += from.full_prev_methods;
        into.full_prev_ticks += from.full_prev_ticks;
        into.full_same += from.full_same;
        into.full_same_ticks += from.full_same_ticks;
        into.struct_prev += from.struct_prev;
        into.struct_prev_ticks += from.struct_prev_ticks;
        into.overflow += from.overflow;
        into.frames += from.frames;
    }

    void Log(const char* label, const Totals& t, double seconds, double ticks_per_ns) const {
        const auto pct = [](u64 part, u64 whole) {
            return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
        };
        const auto ms_s = [&](u64 ticks) {
            return static_cast<double>(ticks) / ticks_per_ns / 1e6 / seconds;
        };
        const u64 struct_full = t.full_prev; // full repeats are also struct repeats
        LOG_INFO(HW_GPU,
                 "MethodRepeat {}: {:.0f} segments/s, {:.0f} frames/s, {:.0f} methods/s, "
                 "replay {:.1f} ms/s (calls {:.1f}) | full repeat of previous frame {:.1f}% "
                 "of segments, {:.1f}% of methods, {:.1f} ms/s | repeat within frame only "
                 "{:.1f}%, {:.1f} ms/s | struct repeat {:.1f}%, {:.1f} ms/s, of which same "
                 "cbuf data {:.1f}% | overflow {}",
                 label, t.segments / seconds, t.frames / seconds, t.methods / seconds,
                 ms_s(t.ticks), ms_s(t.call_ticks), pct(t.full_prev, t.segments),
                 pct(t.full_prev_methods, t.methods), ms_s(t.full_prev_ticks),
                 pct(t.full_same, t.segments), ms_s(t.full_same_ticks),
                 pct(t.struct_prev, t.segments), ms_s(t.struct_prev_ticks),
                 pct(struct_full, t.struct_prev), t.overflow);
    }

    void MaybeReport(u64 ticks_now) {
        if (window_ticks_start == 0) {
            window_ticks_start = ticks_now;
            window_ns_start = NowNs();
            first_ns = window_ns_start;
            first_ticks = ticks_now;
            current_frame = frames.load(std::memory_order_relaxed);
            return;
        }
        if (++entries_since_check < 4096) {
            return;
        }
        entries_since_check = 0;
        const u64 ns = NowNs();
        if (ns - window_ns_start < 10'000'000'000ULL) {
            return;
        }
        const double seconds = static_cast<double>(ns - window_ns_start) / 1e9;
        const double ticks_per_ns =
            static_cast<double>(ticks_now - window_ticks_start) / static_cast<double>(ns - window_ns_start);
        Log("10 s", window, seconds, ticks_per_ns);
        Add(total, window);
        const double total_seconds = static_cast<double>(ns - first_ns) / 1e9;
        const double total_tpn =
            static_cast<double>(ticks_now - first_ticks) / static_cast<double>(ns - first_ns);
        Log("total", total, total_seconds, total_tpn);
        window = {};
        window_ns_start = ns;
        window_ticks_start = ticks_now;
    }

    u32 cb_data{};
    // Segment being built.
    u64 full{};
    u64 structure{};
    u64 methods{};
    u64 seg_ticks{};
    u64 call_ticks{};
    // Entry being run.
    bool in_entry{};
    bool entry_open{};
    u32 entry_kind{};
    u64 entry_start{};
    // Frames.
    u64 current_frame{};
    Table prev_full, cur_full, prev_struct, cur_struct;
    // Reports.
    Totals window{}, total{};
    u64 window_ns_start{}, window_ticks_start{}, first_ns{}, first_ticks{};
    u32 entries_since_check{};
};

/// Touched only by the split's back thread (Execute wraps its entries; draws run inside them).
inline Counter& Get() {
    static Counter counter;
    return counter;
}

inline void Draw() {
    if (enabled) [[unlikely]] {
        Get().Draw();
    }
}

} // namespace VideoCommon::MethodRepeat
