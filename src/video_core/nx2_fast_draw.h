// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

/// A draw that follows a draw of the same pipeline with only light state changes in between takes
/// the delta path. It keeps the previous draw's pipeline, image views, samplers and texture buffer
/// bindings. Buffers are resolved and bound again, except small streamed uniforms whose ranges
/// were not rebound or written: those can keep the previous stream allocation in the same command
/// buffer. This assumes no unsynchronized guest CPU writes unknown to the caches between draws;
/// EDEN_NX2_UBO_KEEP_VERIFY checks that assumption. Kept image descriptors and rescaling words are
/// replayed while the views remain valid; EDEN_NX2_DELTA_DESCRIPTORS_VERIFY checks the replay.
/// Light changes are draw parameters, index buffer and vertex stream registers, constant buffer
/// binds and inline constant buffer data (collected in g_delta: the rasterizer resolves in full
/// when they reach the pipeline's texture handles at their GPU addresses or through another
/// mapping of their device memory). Anything else that could change what a resolve produces bumps
/// a serial: a 3D register that changes, any 3D method with side effects (uploads, reports,
/// clears, barriers, ...), constant buffer data reaching the texture or sampler headers (the same
/// way), any other engine, deferred calls on the split's back (except the ones marked
/// GpuSplit::FastDrawNeutral), GPU::TickWork requests, texture cache writes, unmaps and
/// deletions, shader removals, fence releases, HLE macros and frames. The rasterizer takes the
/// delta path only while the serial, the GPU memory map, the command buffer, the render pass and
/// the bound pipeline are the ones the previous draw left. A draw of another pipeline (or after a
/// draw that can't be followed) takes it with what that pipeline kept at its own last draw while
/// the memory serial, the map, the command buffer, the render pass and the texture pool registers
/// are the ones that draw left, reading every texture handle again (the rasterizer's RevisitDraw).
/// EDEN_NX2_FAST_DRAW_VERIFY=1 resolves those draws in full and counts where the pipeline, views,
/// samplers or kept buffer bindings differ.
namespace VideoCommon::Nx2FastDraw {

enum class Reason : u8 {
    Register,
    Method,
    ConstantData,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    Map,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",      "method",    "constant buffer data", "inline data",
    "other engine",  "deferred call", "tick work",        "invalidation",
    "fence release", "indirect reset", "frame",           "texture header data",
    "memory map",
};

inline std::atomic<bool> g_enabled{true};

/// Whether vertex stream strides and enables are dynamic state (extended dynamic state and
/// dynamic vertex input): then no pipeline depends on them and they are draw parameters. Set
/// by the pipeline cache.
inline bool g_vertex_streams_dynamic{};

/// Any thread: whether fast draws run (always, unless a test turned them off). The rasterizer
/// latches it per draw (on the thread that draws) and starts with a full resolve when it turns on.
[[nodiscard]] inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

/// Tests only.
inline void SetEnabledForTest(bool on) {
    g_enabled.store(on, std::memory_order_relaxed);
}

/// EDEN_NX2_FAST_DRAW_VERIFY=1, read once.
[[nodiscard]] inline bool VerifyEnabled() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_FAST_DRAW_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// On their own cache lines, away from what every thread reads (g_enabled).
alignas(64) inline std::atomic<u64> g_serial{1};
/// Bumped by every break but a 3D register change: register writes change no memory, so draws
/// with only those in between can keep uniforms as delta draws do (MemorySerial).
inline std::atomic<u64> g_memory_serial{1};
/// The reason of the break that bumped g_serial last and, for Register and Method, its 3D method
/// (diagnostics; on the thread that draws, the first break since it read the serial).
inline std::atomic<u32> g_last_break{};

/// The thread that draws reads the serials before each draw (Serial, MemorySerial). Once one of
/// its breaks bumped a serial since it last read it, more change nothing it compares: they skip
/// the atomic add (~1.4 million other engine methods a second in Ocarina of Time's water). Other
/// threads always bump.
inline thread_local bool t_reads_serials{};
inline thread_local bool t_serial_bumped{};
inline thread_local bool t_memory_serial_bumped{};

/// E04.T6 step 3g: stage A of the back split (GPUBack with EDEN_NX2_BACK_SPLIT=1). Its breaks go
/// to plain counters it ships with each draw (SplitSendFastDraw); GPUBackB, which draws, reads
/// the global serials plus the shipped ones, so it never sees breaks of draws A is ahead on.
inline thread_local bool t_stage_a{};
struct StageSerials {
    u64 serial{};
    u64 memory_serial{};
    u32 last_break{};
};
/// Stage A only.
inline StageSerials g_stage_a;
/// GPUBackB only: stage A's serials as of the draw it runs; `moved` when A broke since the
/// previous shipment (LastBreak).
inline StageSerials g_shipped;
inline bool g_shipped_moved{};

inline void Break(Reason reason, u32 method = 0) {
    if (Enabled()) {
        if (t_stage_a) {
            StageSerials& a = g_stage_a;
            a.memory_serial += reason != Reason::Register ? 1 : 0;
            ++a.serial;
            a.last_break = static_cast<u32>(reason) | (method << 8);
            return;
        }
        if (reason != Reason::Register && !(t_reads_serials && t_memory_serial_bumped)) {
            g_memory_serial.fetch_add(1, std::memory_order_relaxed);
            t_memory_serial_bumped = t_reads_serials;
        }
        if (!(t_reads_serials && t_serial_bumped)) {
            g_serial.fetch_add(1, std::memory_order_relaxed);
            g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
            t_serial_bumped = t_reads_serials;
        }
    }
}

/// The thread that draws: what it compares fast draws with (and, from then on, its next break
/// bumps the serial again).
inline u64 Serial() {
    t_reads_serials = true;
    t_serial_bumped = false;
    return g_serial.load(std::memory_order_relaxed) + g_shipped.serial;
}

inline u64 MemorySerial() {
    t_reads_serials = true;
    t_memory_serial_bumped = false;
    return g_memory_serial.load(std::memory_order_relaxed) + g_shipped.memory_serial;
}

/// The thread that draws: the break that changed the serial last (diagnostics).
inline u32 LastBreak() {
    return g_shipped_moved ? g_shipped.last_break : g_last_break.load(std::memory_order_relaxed);
}

/// GPUBackB: stage A's serials shipped before a draw.
inline void ApplyShipped(const StageSerials& shipped) {
    g_shipped_moved = shipped.serial != g_shipped.serial;
    g_shipped = shipped;
}

/// The device memory a GPU range maps to. A write through another GPU address that maps the
/// same device memory changes the range too.
class DeviceSpans {
public:
    /// Translates [address, address + size) again unless that range of memory was translated last
    /// and memory's mappings haven't changed since.
    void Update(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

    /// Whether [begin, end) of device memory overlaps the range's.
    [[nodiscard]] bool Intersects(DAddr begin, DAddr end) const;

    /// The GPU address the range starts at.
    [[nodiscard]] GPUVAddr Address() const noexcept {
        return address;
    }

    /// Whether the range maps to one run of device memory in GPU order, starting at LinearBase.
    [[nodiscard]] bool IsLinear() const noexcept {
        return linear;
    }
    [[nodiscard]] DAddr LinearBase() const noexcept {
        return linear_base;
    }

private:
    const Tegra::MemoryManager* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    bool linear{};
    DAddr linear_base{};
    std::vector<std::pair<DAddr, DAddr>> spans; ///< Sorted, disjoint [begin, end) pairs
};

/// Light changes since the last draw that can reach texture handles. Written and read on the
/// thread that runs the 3D engine and the rasterizer (the split's back).
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    /// Constant buffer data written, by GPU address (unmapped addresses included)...
    std::array<GPUVAddr, MaxWrites> write_begin{};
    std::array<GPUVAddr, MaxWrites> write_end{};
    u32 num_writes{};
    /// ...and by the device address it landed on.
    std::array<DAddr, MaxWrites> device_begin{};
    std::array<DAddr, MaxWrites> device_end{};
    u32 num_device_writes{};
    /// Set when the ranges were widened (AppendRange) or reached a buffer of texture handles
    /// between its handles: the draw keeps no uniforms, as when such writes ended draw reuse.
    bool no_uniform_keeps{};

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    /// Whether data written reaches [address, address + size), whose device memory is spans: at
    /// one of its GPU addresses, or through any GPU address that maps its device memory.
    [[nodiscard]] bool Reaches(GPUVAddr address, u64 size, const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (spans.Intersects(device_begin[write], device_end[write])) {
                return true;
            }
        }
        return false;
    }

    /// Reaches for only the bytes [base + first, base + second) of each of words, all inside the
    /// range spans was updated with. Device writes are compared word by word when that range maps
    /// linearly, else with the whole range.
    [[nodiscard]] bool ReachesWords(GPUVAddr base, std::span<const std::pair<u32, u32>> words,
                                    const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            for (const auto& [first, second] : words) {
                if (write_begin[write] < base + second && base + first < write_end[write]) {
                    return true;
                }
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (!spans.Intersects(device_begin[write], device_end[write])) {
                continue;
            }
            if (!spans.IsLinear()) {
                return true;
            }
            for (const auto& [first, second] : words) {
                const DAddr begin = spans.LinearBase() + (base + first - spans.Address());
                const DAddr end = begin + (second - first);
                if (device_begin[write] < end && begin < device_end[write]) {
                    return true;
                }
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// Moves g_delta into `out` and clears it, copying only the ranges in use (Delta is ~0.5 KiB and
/// taken for every draw).
inline void TakeDelta(Delta& out) {
    Delta& delta = g_delta;
    out.binds = delta.binds;
    out.num_writes = delta.num_writes;
    std::copy_n(delta.write_begin.begin(), delta.num_writes, out.write_begin.begin());
    std::copy_n(delta.write_end.begin(), delta.num_writes, out.write_end.begin());
    out.num_device_writes = delta.num_device_writes;
    std::copy_n(delta.device_begin.begin(), delta.num_device_writes, out.device_begin.begin());
    std::copy_n(delta.device_end.begin(), delta.num_device_writes, out.device_end.begin());
    out.no_uniform_keeps = delta.no_uniform_keeps;
    delta.binds = {};
    delta.num_writes = 0;
    delta.num_device_writes = 0;
    delta.no_uniform_keeps = false;
}

/// Adds [begin, end) to a delta's range list: extends a range it touches or overlaps, else
/// appends it. A full list widens the range that grows least instead (true then), which only
/// makes Reaches more conservative.
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (begin <= ends[i] && begins[i] <= end) {
            begins[i] = (std::min)(begins[i], begin);
            ends[i] = (std::max)(ends[i], end);
            return false;
        }
    }
    if (count < Delta::MaxWrites) {
        begins[count] = begin;
        ends[count] = end;
        ++count;
        return false;
    }
    u32 best = 0;
    u64 best_growth = ~u64{0};
    for (u32 i = 0; i < count; ++i) {
        const u64 growth = ((std::max)(ends[i], end) - (std::min)(begins[i], begin)) -
                           (ends[i] - begins[i]);
        if (growth < best_growth) {
            best = i;
            best_growth = growth;
        }
    }
    begins[best] = (std::min)(begins[best], begin);
    ends[best] = (std::max)(ends[best], end);
    return true;
}

/// Stage A of the back split: its delta, shipped with each draw and merged into g_delta on
/// GPUBackB (MergeDelta).
inline Delta g_stage_a_delta;

/// The delta notes go to on this thread.
inline Delta& NoteDelta() {
    return t_stage_a ? g_stage_a_delta : g_delta;
}

/// GPUBackB: adds a delta stage A shipped to g_delta.
inline void MergeDelta(const std::array<u32, 5>& binds, std::span<const u64> writes,
                       std::span<const u64> device_writes, bool no_uniform_keeps) {
    Delta& delta = g_delta;
    for (size_t stage = 0; stage < binds.size(); ++stage) {
        delta.binds[stage] |= binds[stage];
    }
    for (size_t i = 0; i + 1 < writes.size(); i += 2) {
        delta.no_uniform_keeps |= AppendRange(delta.write_begin, delta.write_end, delta.num_writes,
                                              writes[i], writes[i + 1]);
    }
    for (size_t i = 0; i + 1 < device_writes.size(); i += 2) {
        delta.no_uniform_keeps |= AppendRange(delta.device_begin, delta.device_end,
                                              delta.num_device_writes, device_writes[i],
                                              device_writes[i + 1]);
    }
    delta.no_uniform_keeps |= no_uniform_keeps;
}

inline void NoteBind(size_t stage, u32 slot) {
    if (Enabled()) {
        NoteDelta().binds[stage] |= 1U << slot;
    }
}

/// Inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(GPUVAddr begin, GPUVAddr end) {
    if (Enabled()) {
        Delta& delta = NoteDelta();
        delta.no_uniform_keeps |=
            AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end);
    }
}

/// The device addresses [begin, end) that inline constant buffer data landed on.
inline void NoteDeviceData(DAddr begin, DAddr end) {
    if (Enabled()) {
        Delta& delta = NoteDelta();
        delta.no_uniform_keeps |= AppendRange(delta.device_begin, delta.device_end,
                                              delta.num_device_writes, begin, end);
    }
}

/// Inline constant buffer writes left out of the delta because they wrote the words memory
/// held (EDEN_NX2_GPU_PROFILE logs them).
inline std::atomic<u64> g_unchanged_writes{};

/// Notes data about to be written to [begin, begin + bytes) in GPU memory: inline constant buffer
/// `words`, unless memory holds those words already, or with `words` null data not known yet
/// (reports). Texture and sampler headers are checked by the caller.
void NoteWrittenData(Tegra::MemoryManager& memory, GPUVAddr begin, const void* words, u64 bytes);
/// NoteWrittenData for inline constant buffer `words` the caller is about to write: true when
/// the write changes nothing and can be left out (memory holds the words, and neither the GPU
/// caches nor a download for a guest read may hold others there). Fast draws off: false.
[[nodiscard]] bool WriteData(Tegra::MemoryManager& memory, GPUVAddr begin, const void* words,
                             u64 bytes);

} // namespace VideoCommon::Nx2FastDraw
