// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include "common/common_types.h"
#include "common/logging.h"
#include "common/thread.h"
#include "video_core/gpu_wait_stats.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace VideoCommon::BackSplit {

/// E04.T6d F2 (EDEN_NX2_BACK_SPLIT_TASK_PROF=1): GPUBackB's time by task function, in TSC
/// ticks, written and reported by GPUBackB alone (OnBLoopTick). Async tasks are sampled 1 in 4.
struct BTaskStats {
    struct Slot {
        const void* fn{};
        u64 ticks{};
        u64 count{};
    };
    static constexpr size_t Slots = 256;
    const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_BACK_SPLIT_TASK_PROF");
        return value != nullptr && value[0] == '1';
    }();
    std::array<Slot, Slots> async_slots{};
    std::array<Slot, Slots> sync_slots{};
    u64 busy_ticks{};
    u64 async_total{}; ///< every async task, timed or not
    u32 sample{};

    void Reset() noexcept {
        async_slots = {};
        sync_slots = {};
        busy_ticks = 0;
        async_total = 0;
    }

    static u64 Tsc() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        return __rdtsc();
#else
        return 0;
#endif
    }
    static void Add(std::array<Slot, Slots>& slots, const void* fn, u64 ticks) noexcept {
        size_t i = (reinterpret_cast<uintptr_t>(fn) * 0x9E3779B97F4A7C15ULL) >> 56;
        for (size_t probe = 0; probe < Slots; ++probe, i = (i + 1) & (Slots - 1)) {
            Slot& slot = slots[i];
            if (slot.fn == fn || slot.fn == nullptr) {
                slot.fn = fn;
                slot.ticks += ticks;
                ++slot.count;
                return;
            }
        }
    }
};
inline BTaskStats g_btask;

inline bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_BACK_SPLIT");
    return value != nullptr && value[0] == '1';
}();

inline bool g_h2_async = [] {
    const char* value = std::getenv("EDEN_NX2_H2_SPLIT");
    return value != nullptr && value[0] == '2';
}();

/// Enables the Stage A -> Stage B SPSC prototype ring (E04.T5+ legacy mode).
inline bool Enabled() noexcept {
    return g_enabled;
}

/// Enables H2 Passo 2 asynchronous Side B execution on GPUBackB (EDEN_NX2_H2_SPLIT=2).
inline bool H2Async() noexcept {
    return g_h2_async;
}

/// True whenever the A -> B SPSC ring and maxwell_3d_b shadow are active (legacy or H2 async).
inline bool RingEnabled() noexcept {
    return g_enabled || g_h2_async;
}

/// E04.T12: set on GPUBack (Stage A) whenever RingEnabled() is active.
inline thread_local bool t_stage_a{};
/// E04.T12: nesting depth of H2DrainScope on Stage A. While > 0, GPUBackB is drained and
/// Stage A is executing a synchronous C2 barrier, so nested ops must run synchronously rather
/// than enqueueing onto GPUBackB.
inline thread_local u32 t_h2_drained_depth{};

/// Whether the calling thread is Stage A (GPUBack with the ring active).
inline bool OnStageA() noexcept {
    return t_stage_a;
}

/// E04.T14: Dynamic bypass of H2 split under light draw load (< EDEN_NX2_H2_BYPASS_THRESHOLD draws/s).
inline bool g_h2_dynamic_bypass_enabled = [] {
    const char* val = std::getenv("EDEN_NX2_H2_DYNAMIC_BYPASS");
    return val == nullptr || (val[0] != '0');
}();
inline std::atomic<bool> g_h2_dynamic_bypass_active{false};

inline bool H2DynamicBypassActive() noexcept {
    return g_h2_dynamic_bypass_active.load(std::memory_order_relaxed);
}

/// True on Stage A under H2Async() when not inside a drained C2 barrier scope.
inline bool UseH2AsyncQueue() noexcept {
    return g_h2_async && t_stage_a && t_h2_drained_depth == 0 && !H2DynamicBypassActive();
}

/// E04.T6: set on GPUBackB, the thread that owns the rasterizer with the split on.
inline thread_local bool t_on_b{};

/// Whether the calling thread is GPUBackB.
inline bool OnB() noexcept {
    return t_on_b;
}

/// E04.T12: reasons Stage A drains GPUBackB under EDEN_NX2_H2_SPLIT=2.
enum class H2DrainReason : u8 {
    DrawFallback = 0,
    DrawIndirect,
    DrawTexture,
    Clear,
    Compute,
    Query,
    Fence,
    Flush,
    Invalidation,
    Channel,
    DmaDisplay,
    Call,
    TexOp,
    GuardRead,
    Lockstep,
    Other,
    Count,
};

/// Optional callback invoked on Stage A after DrainAsync/RunOnB completes under H2Async() so
/// RasterizerVulkan can synchronize H2SchedulerMirror from Scheduler.
inline void (*g_h2_post_drain_cb)(void*){};
inline void* g_h2_post_drain_arg{};
inline u64 g_h2_draws_enqueued{};
inline u64 g_h2_computes_enqueued{};
inline u64 g_h2_clears_enqueued{};
inline u64 g_h2_stlf_hits{};
inline u64 g_h2_stlf_misses{};
inline std::array<u64, static_cast<size_t>(H2DrainReason::Count)> g_h2_drain_reasons{};

/// E04.T12b: return value of PushTask2WithPtr / EmitTask2WithPtr so Stage A can record
/// zero-copy Store-to-Load Forwarding (STLF) pointers directly into the SPSC ring.
struct EmitTask2Result {
    u64 start_pos;
    u64 end_pos;
    const u8* tail_ptr;
};

/// E04.T12b: pending guest writes sent to GPUBackB (Stage A only, FIFO ring without coalescing).
struct PendingWrite {
    u64 begin;
    u32 bytes;
    u32 _pad;
    u64 start_word_pos;
    u64 end_word_pos;
    const u8* ring_data; ///< nullptr when no copyable payload exists in the ring (e.g., Query)
};
static constexpr u32 PendingWriteCap = 1024;
inline std::array<PendingWrite, PendingWriteCap> g_pending_writes{};
inline u32 g_pending_write_head{};
inline u32 g_pending_write_tail{};
inline bool g_pending_write_overflow{};
inline u64 g_consumed_cache{};

enum class ForwardResult : u8 { NoPending, Forwarded, NeedDrain };

/// GPUBackB, once at its start: marks it as a back thread for GpuSplit (OnBackThread and
/// OwnsScheduler hold on both stages). Defined in nx2_back_split_rasterizer.cpp.
void OnBThreadStart();
/// GPUBackB, each loop turn: prints its zones every 10 s with EDEN_NX2_GPU_PROFILE (E04.T6d F1;
/// the profiler keeps zones per thread and only the thread itself reports them).
void OnBLoopTick();

/// Counters the A side keeps for the report (Call entries and deferred map/unmap closures).
inline std::atomic<u64> g_calls{};
inline std::atomic<u64> g_deferred{};

inline void CpuPause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

// Bits 0..11 store register index (< 0xE00) when bits 12..15 == 0; commands use kinds 1..8 << 12.
enum class Kind : u8 {
    RegDelta = 0,
    Wrap = 1,
    CbUpload = 2,
    Draw = 3,
    DrawTexture = 4,
    Clear = 5,
    LaunchDma = 6,
    InlineUpload = 7,
    Query = 8,
    /// E04.T6: a call GPUBackB runs in ring order, without stage A waiting (aux = payload words).
    Task = 9,
    /// E04.T6 step 2: the Maxwell3D whose register deltas follow (aux 0, word 1 = engine).
    Bind = 10,
    /// E04.T6 step 2: an engine's whole register file (word 1 = engine, aux = SnapshotInit or
    /// SnapshotCheck). Init seeds GPUBackB's shadow; Check compares it with the shadow.
    Snapshot = 11,
};

constexpr u32 SnapshotInit = 0;
constexpr u32 SnapshotCheck = 1;

constexpr size_t RegCount = 0xE00;
constexpr size_t RingWords = size_t{1} << 20; // 8 MiB SPSC ring
constexpr size_t RingMask = RingWords - 1;
constexpr u64 PublishBatchWords = 4096;
constexpr u64 H2PublishBatchWords = 512;
constexpr size_t SnapshotWords = 2 + RegCount / 2;

/// Draws between two shadow checks (EDEN_NX2_BACK_SPLIT_CHECK; 0 = no checks). Power of two.
inline u32 g_check_mask = [] {
    const char* value = std::getenv("EDEN_NX2_BACK_SPLIT_CHECK");
    u32 every = value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 256U;
    if (every == 0) {
        return ~0U;
    }
    u32 pow2 = 1;
    while (pow2 < every) {
        pow2 <<= 1;
    }
    return pow2 - 1;
}();

inline constexpr u64 MakeHeader(Kind kind, u32 aux = 0) noexcept {
    return (static_cast<u64>(kind) << 12) | (u64{aux} << 16);
}

/// Computes the total u64 word span of a packet from its first word (0 means wrap to ring[0]).
inline constexpr size_t PacketWords(u64 word0) noexcept {
    const auto kind = static_cast<Kind>((word0 >> 12) & 0xfU);
    if (kind == Kind::RegDelta) {
        return 1;
    }
    const u32 aux = static_cast<u32>(word0 >> 16);
    switch (kind) {
    case Kind::RegDelta:
        return 1;
    case Kind::Wrap:
        return 0;
    case Kind::CbUpload:
        return 2 + (static_cast<size_t>(aux) + 1) / 2;
    case Kind::Draw:
        return 2;
    case Kind::DrawTexture:
    case Kind::Clear:
        return 1;
    case Kind::LaunchDma:
        return 4;
    case Kind::InlineUpload:
        return 2 + (static_cast<size_t>(aux) + sizeof(u64) - 1) / sizeof(u64);
    case Kind::Query:
        return 4;
    case Kind::Task:
        return 2 + static_cast<size_t>(aux);
    case Kind::Bind:
        return 2;
    case Kind::Snapshot:
        return SnapshotWords;
    }
    return 1;
}

class RingState final {
public:
    RingState() = default;
    ~RingState() {
        Stop();
    }

    void EnsureStarted() {
        if (started.load(std::memory_order_acquire)) [[likely]] {
            return;
        }
        StartSlow();
    }

    using TaskFn = void (*)(void*);

    /// Stage A: runs fn(arg) on GPUBackB after the ring entries sent so far, and waits for it
    /// (E04.T6 locked mode).
    void RunOnB(TaskFn fn, void* arg, bool hard = true) {
        EnsureStarted();
        const u64 wait_start = NowNs();
        if (hard) {
            ModelSegmentEnd(wait_start);
        }
        PublishNow();
        drained_position = write_position;
        task_fn = fn;
        task_arg = arg;
        const u64 seq = ++task_posted;
        task_seq.store(seq, std::memory_order_seq_cst);
        WakeConsumer();
        ++stat_run_on_b;
        for (u32 spin = 0; task_done.load(std::memory_order_acquire) != seq; ++spin) {
            if (spin < 1U << 16) {
                CpuPause();
                continue;
            }
            ++stat_run_sleeps;
            a_waiting.store(true, std::memory_order_seq_cst);
            if (task_done.load(std::memory_order_seq_cst) != seq) {
                task_done.wait(seq - 1, std::memory_order_seq_cst);
            }
            a_waiting.store(false, std::memory_order_relaxed);
        }
        const u64 wait_end = NowNs();
        a_wait_ns += wait_end - wait_start;
        if (hard) {
            ModelSegmentClose(wait_end);
        }
        g_pending_write_tail = 0;
        g_pending_write_head = 0;
        g_pending_write_overflow = false;
        g_consumed_cache = write_position;
        if (H2Async() && OnStageA() && g_h2_post_drain_cb != nullptr) {
            g_h2_post_drain_cb(g_h2_post_drain_arg);
        }
    }

    /// E04.T6 step 5 model (refutation test before async draws): between two "hard" sync points
    /// (those that stay synchronous in step 5) stage A and GPUBackB would run side by side, so a
    /// segment would take max(A busy, B busy) instead of their sum. A busy = wall time minus A's
    /// waits for B and its idle time; B busy = time GPUBackB spent running packets and calls.
    void ModelSegmentEnd(u64 now) {
        const u64 idle = GpuWaitStats::g_back_idle_ns.load(std::memory_order_relaxed);
        const u64 wall = now - model_last;
        const u64 excluded = (a_wait_ns - model_a_wait) + (idle - model_idle);
        model_seg_a = wall > excluded ? wall - excluded : 0;
    }
    void ModelSegmentClose(u64 now) {
        const u64 b = b_busy_ns.load(std::memory_order_acquire);
        const u64 seg_b = b - model_b;
        if (model_last != 0) {
            model_sum += model_seg_a + seg_b;
            model_max += (std::max)(model_seg_a, seg_b);
            model_a += model_seg_a;
            model_bb += seg_b;
            ++model_hard;
        }
        model_last = now;
        model_a_wait = a_wait_ns;
        model_idle = GpuWaitStats::g_back_idle_ns.load(std::memory_order_relaxed);
        model_b = b;
    }

    using AsyncFn = void (*)(const u64* payload);

    /// Stage A: has GPUBackB run fn(payload) in ring order, without waiting. `bytes` of `data`
    /// are copied into the ring.
    void PushTask(AsyncFn fn, const void* data, size_t bytes) {
        const size_t payload = (bytes + sizeof(u64) - 1) / sizeof(u64);
        const size_t words = 2 + payload;
        u64* dst = Reserve(words);
        dst[0] = MakeHeader(Kind::Task, static_cast<u32>(payload));
        dst[1] = reinterpret_cast<u64>(fn);
        if (payload != 0) {
            dst[words - 1] = 0;
            std::memcpy(dst + 2, data, bytes);
        }
        Commit(words);
        ++stat_async;
    }

    /// Stage A: zero-copy two-segment PushTask (header + inline tail array) directly into the ring,
    /// returning ring positions and pointer to the inline tail payload for STLF.
    EmitTask2Result PushTask2WithPtr(AsyncFn fn, const void* head, size_t head_bytes,
                                     const void* tail, size_t tail_bytes) {
        const size_t total = head_bytes + tail_bytes;
        const size_t payload = (total + sizeof(u64) - 1) / sizeof(u64);
        const size_t words = 2 + payload;
        u64* dst = Reserve(words);
        const u64 start_pos = write_position;
        dst[0] = MakeHeader(Kind::Task, static_cast<u32>(payload));
        dst[1] = reinterpret_cast<u64>(fn);
        const u8* tail_ptr = nullptr;
        if (payload != 0) {
            dst[words - 1] = 0;
            auto* out = reinterpret_cast<u8*>(dst + 2);
            if (head_bytes != 0) {
                std::memcpy(out, head, head_bytes);
            }
            if (tail_bytes != 0) {
                std::memcpy(out + head_bytes, tail, tail_bytes);
                tail_ptr = out + head_bytes;
            }
        }
        Commit(words);
        ++stat_async;
        return EmitTask2Result{start_pos, write_position, tail_ptr};
    }

    /// Stage A: zero-copy two-segment PushTask (header + inline tail array) directly into the ring.
    void PushTask2(AsyncFn fn, const void* head, size_t head_bytes, const void* tail,
                   size_t tail_bytes) {
        (void)PushTask2WithPtr(fn, head, head_bytes, tail, tail_bytes);
    }

    /// Stage A: zero-copy three-segment PushTask (header + two inline tail arrays) directly into
    /// the ring.
    void PushTask3(AsyncFn fn, const void* head, size_t head_bytes, const void* tail1,
                   size_t tail1_bytes, const void* tail2, size_t tail2_bytes) {
        const size_t total = head_bytes + tail1_bytes + tail2_bytes;
        const size_t payload = (total + sizeof(u64) - 1) / sizeof(u64);
        const size_t words = 2 + payload;
        u64* dst = Reserve(words);
        dst[0] = MakeHeader(Kind::Task, static_cast<u32>(payload));
        dst[1] = reinterpret_cast<u64>(fn);
        if (payload != 0) {
            dst[words - 1] = 0;
            auto* out = reinterpret_cast<u8*>(dst + 2);
            if (head_bytes != 0) {
                std::memcpy(out, head, head_bytes);
            }
            if (tail1_bytes != 0) {
                std::memcpy(out + head_bytes, tail1, tail1_bytes);
            }
            if (tail2_bytes != 0) {
                std::memcpy(out + head_bytes + tail1_bytes, tail2, tail2_bytes);
            }
        }
        Commit(words);
        ++stat_async;
    }

    /// Stage A: waits until GPUBackB consumed everything sent since the last wait: tasks sent
    /// without waiting and register deltas (a Call closure on stage A reads B's shadow).
    /// Stage A (step 4d): the ring position after everything sent so far, and the one GPUBackB
    /// is known to have run up to (the last RunOnB).
    [[nodiscard]] u64 WritePosition() const {
        return write_position;
    }
    [[nodiscard]] u64 DrainedPosition() const {
        return drained_position;
    }
    [[nodiscard]] u64 ConsumedPosition() const noexcept {
        return consumed.load(std::memory_order_acquire);
    }

    /// Stage A: makes everything sent so far visible to GPUBackB now (and wakes it).
    void Flush() {
        PublishNow();
    }

    void CountGuardDrain(size_t site) {
        ++stat_guard_drains[site];
    }

    void DrainAsync() {
        if (write_position != drained_position) {
            ++stat_drains;
            RunOnB([](void*) {}, nullptr);
        } else {
            g_pending_write_tail = 0;
            g_pending_write_head = 0;
            g_pending_write_overflow = false;
            g_consumed_cache = write_position;
        }
    }

    /// `engine` is the Maxwell3D writing, `regs` its register file before this write.
    void PushReg(const void* engine, const u32* regs, u32 method, u32 argument) {
        if (method >= RegCount) [[unlikely]] {
            return;
        }
        if (engine != a_engine) [[unlikely]] {
            BindEngine(engine, regs);
        }
        u64* dst = Reserve(1);
        dst[0] = (u64{argument} << 32) | u64{method};
        Commit(1);
        ++stat_delta_regs;
    }

    /// Stage A, on each draw: every (mask + 1) draws, sends the drawing engine's registers for
    /// GPUBackB to compare with its shadow.
    /// Returns whether it sent a check (the caller then sends the engine's other state).
    bool MaybeCheck(const void* engine, const u32* regs) {
        if ((++a_draws & g_check_mask) != 0) [[likely]] {
            return false;
        }
        if (engine != a_engine) {
            BindEngine(engine, regs);
        }
        PushSnapshot(engine, regs, SnapshotCheck);
        return true;
    }

    void PushCbUpload(u64 gpu_addr, const u32* data, u32 count) {
        if (count == 0) {
            return;
        }
        const size_t words = 2 + (static_cast<size_t>(count) + 1) / 2;
        if (words >= RingWords / 4) {
            return;
        }
        u64* dst = Reserve(words);
        dst[0] = MakeHeader(Kind::CbUpload, count);
        dst[1] = gpu_addr;
        dst[words - 1] = 0;
        std::memcpy(dst + 2, data, static_cast<size_t>(count) * sizeof(u32));
        Commit(words);
        ++stat_packets;
        stat_cb_words += count;
    }

    void PushDraw(bool draw_indexed, bool indirect, u32 instance_count) {
        u64* dst = Reserve(2);
        const u32 flags = (draw_indexed ? 1U : 0U) | (indirect ? 2U : 0U);
        dst[0] = MakeHeader(Kind::Draw, flags);
        dst[1] = instance_count;
        Commit(2);
        ++stat_packets;
    }

    void PushDrawTexture() {
        u64* dst = Reserve(1);
        dst[0] = MakeHeader(Kind::DrawTexture, 0);
        Commit(1);
        ++stat_packets;
    }

    void PushClear(u32 layer_count) {
        u64* dst = Reserve(1);
        dst[0] = MakeHeader(Kind::Clear, layer_count);
        Commit(1);
        ++stat_packets;
    }

    void PushLaunchDma(u64 src_addr, u64 dst_addr, u64 byte_size_and_flags) {
        u64* dst = Reserve(4);
        dst[0] = MakeHeader(Kind::LaunchDma, 0);
        dst[1] = src_addr;
        dst[2] = dst_addr;
        dst[3] = byte_size_and_flags;
        Commit(4);
        ++stat_packets;
    }

    void PushInlineUpload(u64 dst_addr, const void* data, size_t bytes) {
        if (bytes == 0) {
            return;
        }
        const size_t clamped = std::min<size_t>(bytes, (RingWords / 4) * sizeof(u64));
        const size_t payload_words = (clamped + sizeof(u64) - 1) / sizeof(u64);
        const size_t words = 2 + payload_words;
        u64* dst = Reserve(words);
        dst[0] = MakeHeader(Kind::InlineUpload, static_cast<u32>(clamped));
        dst[1] = dst_addr;
        dst[words - 1] = 0;
        std::memcpy(dst + 2, data, clamped);
        Commit(words);
        ++stat_packets;
    }

    void PushQuery(u64 gpu_addr, u64 payload, u64 meta) {
        u64* dst = Reserve(4);
        dst[0] = MakeHeader(Kind::Query, 0);
        dst[1] = gpu_addr;
        dst[2] = payload;
        dst[3] = meta;
        Commit(4);
        ++stat_packets;
    }

    void BindEngine(const void* engine, const u32* regs) {
        a_engine = engine;
        for (const void* known : a_known) {
            if (known == engine) {
                u64* dst = Reserve(2);
                dst[0] = MakeHeader(Kind::Bind, 0);
                dst[1] = reinterpret_cast<u64>(engine);
                Commit(2);
                return;
            }
        }
        a_known.push_back(engine);
        PushSnapshot(engine, regs, SnapshotInit);
    }

    void PushSnapshot(const void* engine, const u32* regs, u32 kind) {
        u64* dst = Reserve(SnapshotWords);
        dst[0] = MakeHeader(Kind::Snapshot, kind);
        dst[1] = reinterpret_cast<u64>(engine);
        std::memcpy(dst + 2, regs, RegCount * sizeof(u32));
        Commit(SnapshotWords);
    }

    void MaybeReport() {
        const u64 now = NowNs();
        if (last_report_ns == 0) {
            last_report_ns = now;
            return;
        }
        if (now - last_report_ns < 10'000'000'000ULL) {
            return;
        }
        const double sec = static_cast<double>(now - last_report_ns) / 1e9;
        last_report_ns = now;
        LOG_INFO(HW_GPU,
                 "Nx2BackSplit: cmds {:.0f}/s, delta_regs {:.0f}/s, cb_words {:.0f}/s, "
                 "words {:.0f}/s, A_stalls {}, run_on_b {:.0f}/s, run_sleeps {}, b_sleeps {}, "
                 "calls {:.0f}/s, deferred {:.0f}/s, async {:.0f}/s, drains {:.0f}/s, "
                 "read guard drains {:.0f}/s (macro {}, query {}, other {})",
                 static_cast<double>(stat_packets) / sec,
                 static_cast<double>(stat_delta_regs) / sec,
                 static_cast<double>(stat_cb_words) / sec,
                 static_cast<double>(write_position - last_report_words) / sec,
                 stat_stalls, static_cast<double>(stat_run_on_b) / sec, stat_run_sleeps,
                 b_sleeps.exchange(0, std::memory_order_relaxed),
                 static_cast<double>(g_calls.exchange(0, std::memory_order_relaxed)) / sec,
                 static_cast<double>(g_deferred.exchange(0, std::memory_order_relaxed)) / sec,
                 static_cast<double>(stat_async) / sec, static_cast<double>(stat_drains) / sec,
                 static_cast<double>(stat_guard_drains[0] + stat_guard_drains[1] +
                                     stat_guard_drains[2]) / sec,
                 stat_guard_drains[0], stat_guard_drains[1], stat_guard_drains[2]);
        const u64 b_bad_win = b_bad_checks.exchange(0, std::memory_order_relaxed);
        LOG_INFO(HW_GPU,
                 "Nx2BackSplit shadow: engines {}, checks {}, diverged checks {}, diverged regs {} "
                 "(total since boot: checks {}, diverged checks {}); state checks {}, diverged "
                 "state {} (total {})",
                 b_engines.load(std::memory_order_relaxed),
                 b_checks.exchange(0, std::memory_order_relaxed), b_bad_win,
                 b_bad_regs.exchange(0, std::memory_order_relaxed),
                 b_checks_total.load(std::memory_order_relaxed),
                 b_bad_checks_total.load(std::memory_order_relaxed),
                 b_state_checks.exchange(0, std::memory_order_relaxed),
                 b_bad_state.exchange(0, std::memory_order_relaxed),
                 b_bad_state_total.load(std::memory_order_relaxed));
        const u64 cur_b_busy_ns = b_busy_ns.load(std::memory_order_relaxed);
        const double a_wait_ms_s =
            static_cast<double>(a_wait_ns - last_report_a_wait_ns) / 1e6 / sec;
        const double b_busy_ms_s =
            static_cast<double>(cur_b_busy_ns - last_report_b_busy_ns) / 1e6 / sec;
        last_report_a_wait_ns = a_wait_ns;
        last_report_b_busy_ns = cur_b_busy_ns;
        if (H2Async()) {
            static constexpr std::array<const char*, static_cast<size_t>(H2DrainReason::Count)>
                reason_names{
                    "DrawFallback", "DrawIndirect", "DrawTexture", "Clear",     "Compute",
                    "Query",        "Fence",        "Flush",       "Inval",     "Channel",
                    "DmaDisplay",   "Call",         "TexOp",       "GuardRead", "Lockstep",
                    "Other",
                };
            std::string reasons;
            for (size_t i = 0; i < g_h2_drain_reasons.size(); ++i) {
                if (g_h2_drain_reasons[i] != 0) {
                    reasons += fmt::format("{}{} {:.0f}/s", reasons.empty() ? "" : ", ",
                                           reason_names[i],
                                           static_cast<double>(g_h2_drain_reasons[i]) / sec);
                }
            }
            LOG_INFO(HW_GPU,
                     "Nx2BackSplit H2: draws_enqueued {:.0f}/s, computes_enqueued {:.0f}/s, "
                     "clears_enqueued {:.0f}/s, async {:.0f}/s, drains {:.0f}/s, "
                     "a_wait {:.1f} ms/s, b_busy {:.1f} ms/s, stlf_hits {:.0f}/s, "
                     "stlf_misses {:.0f}/s, b_bad_checks {}; drain reasons: {}",
                     static_cast<double>(g_h2_draws_enqueued) / sec,
                     static_cast<double>(g_h2_computes_enqueued) / sec,
                     static_cast<double>(g_h2_clears_enqueued) / sec,
                     static_cast<double>(stat_async) / sec, static_cast<double>(stat_drains) / sec,
                     a_wait_ms_s, b_busy_ms_s, static_cast<double>(g_h2_stlf_hits) / sec,
                     static_cast<double>(g_h2_stlf_misses) / sec, b_bad_win,
                     reasons.empty() ? "none" : reasons);
            g_h2_draws_enqueued = 0;
            g_h2_computes_enqueued = 0;
            g_h2_clears_enqueued = 0;
            g_h2_stlf_hits = 0;
            g_h2_stlf_misses = 0;
            g_h2_drain_reasons.fill(0);
        }
        stat_async = 0;
        stat_drains = 0;
        LOG_INFO(HW_GPU,
                 "Nx2BackSplit step 5 model: hard syncs {:.0f}/s, A busy {:.1f} ms/s, B busy {:.1f} "
                 "ms/s, sum {:.1f} ms/s, max {:.1f} ms/s, ceiling {:+.1f}%",
                 static_cast<double>(model_hard) / sec, static_cast<double>(model_a) / 1e6 / sec,
                 static_cast<double>(model_bb) / 1e6 / sec, static_cast<double>(model_sum) / 1e6 / sec,
                 static_cast<double>(model_max) / 1e6 / sec,
                 model_max == 0 ? 0.0
                                : (static_cast<double>(model_sum) / static_cast<double>(model_max) - 1.0) *
                                      100.0);
        model_sum = model_max = model_a = model_bb = model_hard = 0;
        stat_guard_drains = {};
        if (report_hook != nullptr) {
            report_hook(sec);
        }
        stat_run_on_b = 0;
        stat_run_sleeps = 0;
        last_report_words = write_position;
        stat_packets = 0;
        stat_delta_regs = 0;
        stat_cb_words = 0;
        stat_stalls = 0;
    }

    /// Extra lines for the report (the rasterizer proxy's per-call counts).
    void (*report_hook)(double seconds){};
    /// Step 3: checks of the shadow's constant buffer bindings and root tables (Maxwell3D),
    /// written by its tasks on GPUBackB.
    std::atomic<u64> b_state_checks{0};
    std::atomic<u64> b_bad_state{0};
    std::atomic<u64> b_bad_state_total{0};
    u32 b_state_report_lines = 0;

private:
    void StartSlow() {
        bool expected = false;
        if (!starting.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            while (!started.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            return;
        }
        ring = std::make_unique<u64[]>(RingWords);
        safe_end = RingWords;
        consumer = std::jthread([this](std::stop_token st) { ConsumerLoop(st); });
        started.store(true, std::memory_order_release);
    }

    void Stop() {
        if (!started.load(std::memory_order_acquire)) {
            return;
        }
        if (consumer.joinable()) {
            consumer.request_stop();
            WakeConsumer();
            consumer.join();
        }
        started.store(false, std::memory_order_release);
    }

    static u64 NowNs() noexcept {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }

    u64* Reserve(size_t words) {
        if (write_position + words <= safe_end) [[likely]] {
            return &ring[write_position & RingMask];
        }
        return ReserveSlow(words);
    }

    u64* ReserveSlow(size_t words) {
        EnsureStarted();
        const size_t index = write_position & RingMask;
        if (index + words > RingWords) {
            const size_t pad = RingWords - index;
            WaitForSpace(pad);
            ring[index] = MakeHeader(Kind::Wrap, 0);
            write_position += pad;
            PublishNow();
        }
        WaitForSpace(words);
        const u64 wrap_limit = (write_position & ~u64{RingMask}) + RingWords;
        safe_end = std::min(wrap_limit, consumed_cache + RingWords);
        return &ring[write_position & RingMask];
    }

    void WaitForSpace(size_t words) {
        while (write_position + words - consumed_cache > RingWords) {
            consumed_cache = consumed.load(std::memory_order_acquire);
            if (write_position + words - consumed_cache <= RingWords) {
                return;
            }
            PublishNow();
            ++stat_stalls;
            producer_waiting.store(true, std::memory_order_seq_cst);
            const u32 seen = producer_wake.load(std::memory_order_seq_cst);
            consumed_cache = consumed.load(std::memory_order_seq_cst);
            if (write_position + words - consumed_cache > RingWords) {
                producer_wake.wait(seen, std::memory_order_seq_cst);
            }
            producer_waiting.store(false, std::memory_order_relaxed);
        }
    }

    void Commit(size_t words) {
        write_position += words;
        const u64 batch = H2Async() ? H2PublishBatchWords : PublishBatchWords;
        if (write_position - published_local >= batch) [[unlikely]] {
            PublishNow();
        }
    }

    void PublishNow() {
        if (write_position == published_local) {
            return;
        }
        published_local = write_position;
        published.store(write_position, std::memory_order_release);
        WakeConsumer();
    }

    void WakeConsumer() {
        if (consumer_waiting.load(std::memory_order_relaxed)) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (consumer_waiting.load(std::memory_order_relaxed)) {
                consumer_wake.fetch_add(1, std::memory_order_seq_cst);
                consumer_wake.notify_one();
            }
        }
    }

    void NotifyProducer() {
        if (producer_waiting.load(std::memory_order_seq_cst)) {
            producer_wake.fetch_add(1, std::memory_order_seq_cst);
            producer_wake.notify_one();
        }
    }

    void ConsumerLoop(std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPUBackB");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
        Common::SetCurrentThreadToPerformanceCores();
        t_on_b = true;
        OnBThreadStart();
        u64 read_pos = 0;
        u64 sink = 0;
        u64 done = 0;
        while (!stop_token.stop_requested()) {
            OnBLoopTick();
            // The task first: A published its ring entries before posting it.
            u64 task = task_seq.load(std::memory_order_acquire);
            u64 available = published.load(std::memory_order_acquire);
            if (available == read_pos && task == done) {
                // Locked mode: A posts a task every few microseconds; stay awake (~50-100 us).
                for (u32 spin = 0; spin < 4096 && available == read_pos && task == done;
                     ++spin) {
                    CpuPause();
                    task = task_seq.load(std::memory_order_acquire);
                    available = published.load(std::memory_order_acquire);
                }
                if (available == read_pos && task == done) {
                    consumer_waiting.store(true, std::memory_order_seq_cst);
                    const u32 seen = consumer_wake.load(std::memory_order_seq_cst);
                    task = task_seq.load(std::memory_order_seq_cst);
                    available = published.load(std::memory_order_seq_cst);
                    if (available == read_pos && task == done && !stop_token.stop_requested()) {
                        b_sleeps.fetch_add(1, std::memory_order_relaxed);
                        consumer_wake.wait(seen, std::memory_order_seq_cst);
                    }
                    consumer_waiting.store(false, std::memory_order_relaxed);
                    continue;
                }
            }
            const u64 busy_start = NowNs();
            const u64 busy_start_tsc = g_btask.enabled ? BTaskStats::Tsc() : 0;
            bool busy_counted = false;
            u64 last_store = read_pos;
            while (read_pos != available) {
                const size_t index = read_pos & RingMask;
                const u64 word0 = ring[index];
                const size_t words = PacketWords(word0);
                if (words == 0) {
                    read_pos += RingWords - index;
                } else if ((word0 & 0xf000U) == 0) {
                    // RegDelta: bits 0..11 register, 32..63 value.
                    if (b_regs != nullptr) [[likely]] {
                        b_regs[word0 & 0xfffU] = static_cast<u32>(word0 >> 32);
                    }
                    read_pos += 1;
                } else if (((word0 >> 12) & 0xfU) == static_cast<u64>(Kind::Task)) {
                    const auto fn = reinterpret_cast<AsyncFn>(ring[index + 1]);
                    if (g_btask.enabled) [[unlikely]] {
                        ++g_btask.async_total;
                        if ((++g_btask.sample & 3U) == 0) {
                            const u64 t0 = BTaskStats::Tsc();
                            fn(&ring[index + 2]);
                            BTaskStats::Add(g_btask.async_slots,
                                            reinterpret_cast<const void*>(fn),
                                            BTaskStats::Tsc() - t0);
                        } else {
                            fn(&ring[index + 2]);
                        }
                    } else {
                        fn(&ring[index + 2]);
                    }
                    read_pos += words;
                } else if (((word0 >> 12) & 0xfU) == static_cast<u64>(Kind::Bind)) {
                    b_regs = ShadowOf(reinterpret_cast<const void*>(ring[index + 1]));
                    read_pos += words;
                } else if (((word0 >> 12) & 0xfU) == static_cast<u64>(Kind::Snapshot)) {
                    OnSnapshot(reinterpret_cast<const void*>(ring[index + 1]),
                               static_cast<u32>(word0 >> 16),
                               reinterpret_cast<const u32*>(&ring[index + 2]));
                    read_pos += words;
                } else {
                    for (size_t i = 0; i < words; ++i) {
                        sink ^= ring[index + i];
                    }
                    read_pos += words;
                }
                if (read_pos - last_store >= 8192 && read_pos != available) {
                    consumed.store(read_pos, std::memory_order_release);
                    last_store = read_pos;
                    NotifyProducer();
                }
            }
            consumed.store(read_pos, std::memory_order_release);
            NotifyProducer();
            if (task != done) {
                if (g_btask.enabled) [[unlikely]] {
                    const u64 t0 = BTaskStats::Tsc();
                    task_fn(task_arg);
                    BTaskStats::Add(g_btask.sync_slots, reinterpret_cast<const void*>(task_fn),
                                    BTaskStats::Tsc() - t0);
                } else {
                    task_fn(task_arg);
                }
                done = task;
                b_busy_ns.fetch_add(NowNs() - busy_start, std::memory_order_release);
                busy_counted = true;
                task_done.store(task, std::memory_order_seq_cst);
                if (a_waiting.load(std::memory_order_seq_cst)) {
                    task_done.notify_one();
                }
            }
            if (!busy_counted) {
                b_busy_ns.fetch_add(NowNs() - busy_start, std::memory_order_release);
            }
            if (g_btask.enabled) [[unlikely]] {
                g_btask.busy_ticks += BTaskStats::Tsc() - busy_start_tsc;
            }
        }
        consumer_sink.store(sink, std::memory_order_relaxed);
    }

    /// GPUBackB: the shadow register file named by `engine` on the ring. Since step 3 the ring
    /// names an engine by its shadow's registers (Maxwell3D::split_shadow), owned by the channel.
    static u32* ShadowOf(const void* engine) {
        return static_cast<u32*>(const_cast<void*>(engine));
    }

    /// GPUBackB: seeds (init) or checks (check) the shadow of `engine` against stage A's
    /// registers; a check logs each register's first divergence and then takes A's values.
    void OnSnapshot(const void* engine, u32 kind, const u32* a_regs) {
        u32* const shadow = ShadowOf(engine);
        b_regs = shadow;
        if (kind == SnapshotInit) {
            b_engines.fetch_add(1, std::memory_order_relaxed);
        }
        if (kind == SnapshotCheck) {
            b_checks.fetch_add(1, std::memory_order_relaxed);
            b_checks_total.fetch_add(1, std::memory_order_relaxed);
            if (std::memcmp(shadow, a_regs, RegCount * sizeof(u32)) != 0) [[unlikely]] {
                u32 bad = 0;
                for (u32 reg = 0; reg < RegCount; ++reg) {
                    if (shadow[reg] == a_regs[reg]) {
                        continue;
                    }
                    ++bad;
                    if (!b_reported[reg] && b_report_lines < 128) {
                        b_reported[reg] = true;
                        ++b_report_lines;
                        LOG_WARNING(HW_GPU,
                                    "Nx2BackSplit shadow diverged: check {}, engine {}, reg {:#x} "
                                    "A={:#x} B={:#x}",
                                    b_checks_total.load(std::memory_order_relaxed), engine, reg,
                                    a_regs[reg], shadow[reg]);
                    }
                }
                b_bad_checks.fetch_add(1, std::memory_order_relaxed);
                b_bad_checks_total.fetch_add(1, std::memory_order_relaxed);
                b_bad_regs.fetch_add(bad, std::memory_order_relaxed);
            }
        }
        std::memcpy(shadow, a_regs, RegCount * sizeof(u32));
    }

    std::atomic<bool> starting{false};
    std::atomic<bool> started{false};
    std::unique_ptr<u64[]> ring;

    u64 write_position = 0;
    u64 safe_end = 0;
    u64 published_local = 0;
    u64 consumed_cache = 0;
    u64 last_report_ns = 0;
    u64 last_report_words = 0;
    u64 stat_packets = 0;
    u64 stat_delta_regs = 0;
    u64 stat_cb_words = 0;
    u64 stat_stalls = 0;
    u64 stat_run_on_b = 0;
    u64 stat_run_sleeps = 0;
    u64 stat_async = 0;
    u64 stat_drains = 0;
    // Step 5 model (A side, but b_busy_ns written by GPUBackB).
    u64 a_wait_ns = 0;
    u64 last_report_a_wait_ns = 0;
    u64 last_report_b_busy_ns = 0;
    std::atomic<u64> b_busy_ns{0};
    u64 model_last = 0, model_a_wait = 0, model_idle = 0, model_b = 0, model_seg_a = 0;
    u64 model_sum = 0, model_max = 0, model_a = 0, model_bb = 0, model_hard = 0;
    std::array<u64, 3> stat_guard_drains{}; ///< step 4d, by GuardRead site
    u64 drained_position = 0; ///< write_position at the last RunOnB
    // Stage A: the engine the deltas belong to, the engines GPUBackB has a shadow of, draws.
    const void* a_engine{};
    std::vector<const void*> a_known;
    u32 a_draws = 0;
    // GPUBackB: the shadow of the bound engine and every engine's shadow (E04.T6 step 2).
    u32* b_regs{};
    std::array<bool, RegCount> b_reported{};
    u32 b_report_lines = 0;
    std::atomic<u64> b_engines{0};
    std::atomic<u64> b_checks{0};
    std::atomic<u64> b_bad_checks{0};
    std::atomic<u64> b_bad_regs{0};
    std::atomic<u64> b_checks_total{0};
    std::atomic<u64> b_bad_checks_total{0};
    u64 task_posted = 0;
    TaskFn task_fn{};
    void* task_arg{};

    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    alignas(64) std::atomic<u32> consumer_wake{0};
    alignas(64) std::atomic<bool> consumer_waiting{false};
    alignas(64) std::atomic<u32> producer_wake{0};
    alignas(64) std::atomic<bool> producer_waiting{false};
    alignas(64) std::atomic<u64> task_seq{0};
    alignas(64) std::atomic<u64> task_done{0};
    alignas(64) std::atomic<bool> a_waiting{false};
    std::atomic<u64> b_sleeps{0};
    std::atomic<u64> consumer_sink{0};
    std::jthread consumer;
};

inline RingState g_state;

inline void EmitReg(const void* engine, const u32* regs, u32 method, u32 argument) {
    if (H2DynamicBypassActive()) [[unlikely]] {
        if (method < RegCount) [[likely]] {
            static_cast<u32*>(const_cast<void*>(engine))[method] = argument;
        }
        return;
    }
    g_state.PushReg(engine, regs, method, argument);
}

/// Stage A, at each draw of `engine` (E04.T6 step 2 shadow check).
inline bool CheckRegs(const void* engine, const u32* regs) {
    return g_state.MaybeCheck(engine, regs);
}

/// Stage A: has GPUBackB run fn(payload) in ring order without waiting (`bytes` copied).
inline void EmitTask(RingState::AsyncFn fn, const void* data, size_t bytes) {
    g_state.PushTask(fn, data, bytes);
}

/// Stage A: zero-copy two-segment EmitTask directly into the ring.
inline void EmitTask2(RingState::AsyncFn fn, const void* head, size_t head_bytes, const void* tail,
                      size_t tail_bytes) {
    g_state.PushTask2(fn, head, head_bytes, tail, tail_bytes);
}

/// Stage A: zero-copy three-segment EmitTask directly into the ring.
inline void EmitTask3(RingState::AsyncFn fn, const void* head, size_t head_bytes, const void* tail1,
                      size_t tail1_bytes, const void* tail2, size_t tail2_bytes) {
    g_state.PushTask3(fn, head, head_bytes, tail1, tail1_bytes, tail2, tail2_bytes);
}

/// Stage A: zero-copy two-segment EmitTask returning ring positions and inline tail pointer.
inline EmitTask2Result EmitTask2WithPtr(RingState::AsyncFn fn, const void* head, size_t head_bytes,
                                        const void* tail, size_t tail_bytes) {
    return g_state.PushTask2WithPtr(fn, head, head_bytes, tail, tail_bytes);
}

/// E04.T6d F2: the packets below are dead (GPUBackB only XORs them); they are sent only with
/// EDEN_NX2_BACK_SPLIT_DEAD_PACKETS=1 (the old behavior, as a control).
inline const bool g_dead_packets = [] {
    const char* value = std::getenv("EDEN_NX2_BACK_SPLIT_DEAD_PACKETS");
    return value != nullptr && value[0] == '1';
}();

inline void EmitCbUpload(u64 gpu_addr, const u32* data, u32 count) {
    if (!g_dead_packets) [[likely]] {
        return;
    }
    g_state.PushCbUpload(gpu_addr, data, count);
}

inline void EmitDraw(bool draw_indexed, bool indirect, u32 instance_count) {
    if (!g_dead_packets) [[likely]] {
        return;
    }
    g_state.PushDraw(draw_indexed, indirect, instance_count);
}

inline void EmitDrawTexture() {
    if (!g_dead_packets) [[likely]] {
        return;
    }
    g_state.PushDrawTexture();
}

inline void EmitClear(u32 layer_count) {
    if (!g_dead_packets) [[likely]] {
        return;
    }
    g_state.PushClear(layer_count);
}

inline void EmitLaunchDma(u64 src_addr, u64 dst_addr, u64 byte_size_and_flags) {
    if (!g_dead_packets) [[likely]] {
        return;
    }
    g_state.PushLaunchDma(src_addr, dst_addr, byte_size_and_flags);
}

inline void EmitInlineUpload(u64 dst_addr, const void* data, size_t bytes) {
    if (!g_dead_packets) [[likely]] {
        return;
    }
    g_state.PushInlineUpload(dst_addr, data, bytes);
}

inline void EmitQuery(u64 gpu_addr, u64 payload, u64 meta) {
    if (!g_dead_packets) [[likely]] {
        return;
    }
    g_state.PushQuery(gpu_addr, payload, meta);
}

inline void RunOnB(RingState::TaskFn fn, void* arg, bool hard = true) {
    g_state.RunOnB(fn, arg, hard);
}

inline void DrainAsync() {
    g_state.DrainAsync();
}

/// Stage A under EDEN_NX2_H2_SPLIT=2: drains GPUBackB and accounts the barrier reason.
inline void DrainAsyncH2(H2DrainReason reason) {
    if (H2Async() && OnStageA()) [[unlikely]] {
        if (g_state.WritePosition() != g_state.DrainedPosition()) {
            ++g_h2_drain_reasons[static_cast<size_t>(reason)];
            g_state.DrainAsync();
        }
    }
}

inline u32 g_h2_bypass_threshold = [] {
    const char* val = std::getenv("EDEN_NX2_H2_BYPASS_THRESHOLD");
    if (val != nullptr) {
        const u32 parsed = static_cast<u32>(std::strtoul(val, nullptr, 10));
        if (parsed > 0) {
            return parsed;
        }
    }
    return 65000U;
}();
inline std::atomic<u32> g_h2_current_draw_rate{140000};
inline std::atomic<u64> g_h2_bypass_frames{0};
inline std::atomic<u64> g_h2_split_frames{0};
inline u32 g_h2_bypass_cooldown_frames{0};

enum class H2BypassTransition : u8 {
    None = 0,
    EnteredBypass,
    ExitedBypass,
};

inline H2BypassTransition UpdateH2BypassState(u32 rate, const char* source) {
    if (g_h2_bypass_cooldown_frames > 0) {
        --g_h2_bypass_cooldown_frames;
        return H2BypassTransition::None;
    }
    const u32 low_thresh = g_h2_bypass_threshold;
    const u32 high_thresh = low_thresh == 65000U ? 95000U : (low_thresh * 145) / 100;
    const bool currently_bypassed = g_h2_dynamic_bypass_active.load(std::memory_order_relaxed);
    if (!currently_bypassed && rate < low_thresh) {
        g_h2_dynamic_bypass_active.store(true, std::memory_order_relaxed);
        g_h2_bypass_cooldown_frames = 30;
        LOG_INFO(HW_GPU, "Nx2BackSplit H2: bypass ACTIVATED via {} (draw_rate={}/s < {})",
                 source, rate, low_thresh);
        return H2BypassTransition::EnteredBypass;
    } else if (currently_bypassed && rate > high_thresh) {
        g_h2_dynamic_bypass_active.store(false, std::memory_order_relaxed);
        g_h2_bypass_cooldown_frames = 30;
        LOG_INFO(HW_GPU, "Nx2BackSplit H2: bypass DEACTIVATED via {} (draw_rate={}/s > {})",
                 source, rate, high_thresh);
        return H2BypassTransition::ExitedBypass;
    }
    return H2BypassTransition::None;
}

inline H2BypassTransition NoteH2FrameDrawCount(u32 frame_draws, u64 frame_duration_ns) {
    if (!g_h2_async || !g_h2_dynamic_bypass_enabled) {
        return H2BypassTransition::None;
    }
    // Filter out micro-intervals (< 5ms) to prevent division explosions during rapid presentation bursts
    if (frame_duration_ns < 5'000'000ULL) {
        return H2BypassTransition::None;
    }
    const u32 rate = static_cast<u32>((static_cast<u64>(frame_draws) * 1'000'000'000ULL) / frame_duration_ns);
    const u32 prev_rate = g_h2_current_draw_rate.load(std::memory_order_relaxed);
    const u32 new_rate = (prev_rate * 15 + rate) / 16;
    g_h2_current_draw_rate.store(new_rate, std::memory_order_relaxed);

    const H2BypassTransition transition = UpdateH2BypassState(new_rate, "FrameTick");

    if (g_h2_dynamic_bypass_active.load(std::memory_order_relaxed)) {
        g_h2_bypass_frames.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_h2_split_frames.fetch_add(1, std::memory_order_relaxed);
    }
    return transition;
}

/// RAII C2 barrier under H2Async(): drains GPUBackB on entry, suppresses async queueing while
/// active, and re-synchronizes H2SchedulerMirror on exit of the outermost scope.
struct H2DrainScope {
    bool active{};

    explicit H2DrainScope(H2DrainReason reason) {
        if (H2Async() && OnStageA() &&
            (!H2DynamicBypassActive() ||
             g_state.WritePosition() != g_state.DrainedPosition())) [[unlikely]] {
            DrainAsyncH2(reason);
            ++t_h2_drained_depth;
            active = true;
        }
    }
    /// Conditional variant: when `do_drain` is false, suppresses async queueing (depth increment)
    /// but skips the actual DrainAsync call. Use when the caller has determined that the drain
    /// can be elided (e.g. SkipEmptyFence fast-path), but the scope must still prevent GPUBackB
    /// from enqueuing new tasks while the synchronous operation runs on Stage A.
    explicit H2DrainScope(H2DrainReason reason, bool do_drain) {
        if (H2Async() && OnStageA() &&
            (!H2DynamicBypassActive() ||
             g_state.WritePosition() != g_state.DrainedPosition())) [[unlikely]] {
            if (do_drain) {
                DrainAsyncH2(reason);
            }
            ++t_h2_drained_depth;
            active = true;
        }
    }
    ~H2DrainScope() {
        if (active) [[unlikely]] {
            --t_h2_drained_depth;
            if (t_h2_drained_depth == 0 && g_h2_post_drain_cb != nullptr) {
                g_h2_post_drain_cb(g_h2_post_drain_arg);
            }
        }
    }
};

inline void Flush() {
    g_state.Flush();
}

/// Largest inline write sent asynchronously to GPUBackB.
inline constexpr size_t MaxAsyncInlineBytes = 4096;

/// E04.T6 step 4d: per GPU page (hashed), the ring position after the last guest write stage A
/// sent to GPUBackB there. Stage A only.
inline std::array<u64, 4096> g_pending_pages{};

/// Stage A: guest memory [gpu_addr, gpu_addr + bytes) is written by GPUBackB in ring order,
/// optionally with a zero-copy pointer to its inline payload inside the SPSC ring.
inline void NotePendingWriteData(u64 gpu_addr, u64 bytes, u64 start_pos, u64 end_pos,
                                 const u8* ring_data) {
    if (bytes == 0) {
        return;
    }
    for (u64 page = gpu_addr >> 12; page <= (gpu_addr + bytes - 1) >> 12; ++page) {
        g_pending_pages[page & 4095] = end_pos;
    }
    if (H2Async() && !g_pending_write_overflow) {
        if (g_pending_write_head - g_pending_write_tail >= PendingWriteCap) [[unlikely]] {
            g_consumed_cache = g_state.ConsumedPosition();
            while (g_pending_write_tail != g_pending_write_head &&
                   g_pending_writes[g_pending_write_tail & (PendingWriteCap - 1)].end_word_pos <=
                       g_consumed_cache) {
                ++g_pending_write_tail;
            }
        }
        if (g_pending_write_head - g_pending_write_tail >= PendingWriteCap ||
            bytes > UINT32_MAX) [[unlikely]] {
            g_pending_write_overflow = true;
        } else {
            g_pending_writes[g_pending_write_head++ & (PendingWriteCap - 1)] = PendingWrite{
                .begin = gpu_addr,
                .bytes = static_cast<u32>(bytes),
                ._pad = 0,
                .start_word_pos = start_pos,
                .end_word_pos = end_pos,
                .ring_data = ring_data,
            };
        }
    }
}

/// Stage A: guest memory [gpu_addr, gpu_addr + bytes) is written by GPUBackB in ring order
/// without a copyable payload in the ring (e.g., Query).
inline void NotePendingWrite(u64 gpu_addr, u64 bytes) {
    const u64 pos = g_state.WritePosition();
    NotePendingWriteData(gpu_addr, bytes, pos, pos, nullptr);
}

/// Stage A under H2Async(): zero-copy Store-to-Load Forwarding (STLF) from the SPSC ring.
/// Returns NoPending when RAM is already up to date, Forwarded when `dst` was populated directly
/// from the ring without draining GPUBackB, or NeedDrain when Stage A must drain before reading.
inline ForwardResult TryForwardRead(u64 gpu_addr, void* dst, size_t bytes) {
    if (bytes == 0 || (g_pending_write_head == g_pending_write_tail && !g_pending_write_overflow))
        [[likely]] {
        return ForwardResult::NoPending;
    }
    const u64 first_page = gpu_addr >> 12;
    const u64 last_page = (gpu_addr + bytes - 1) >> 12;
    auto any_page_after = [&](u64 pos) {
        for (u64 page = first_page; page <= last_page; ++page) {
            if (g_pending_pages[page & 4095] > pos) [[unlikely]] {
                return true;
            }
        }
        return false;
    };
    if (!any_page_after(g_consumed_cache)) [[likely]] {
        return ForwardResult::NoPending;
    }
    g_consumed_cache = g_state.ConsumedPosition();
    if (!any_page_after(g_consumed_cache)) [[likely]] {
        return ForwardResult::NoPending;
    }
    if (g_pending_write_overflow) [[unlikely]] {
        ++g_h2_stlf_misses;
        return ForwardResult::NeedDrain;
    }
    while (g_pending_write_tail != g_pending_write_head &&
           g_pending_writes[g_pending_write_tail & (PendingWriteCap - 1)].end_word_pos <=
               g_consumed_cache) {
        ++g_pending_write_tail;
    }
    const u64 read_end = gpu_addr + bytes;
    const u64 cur_write_pos = g_state.WritePosition();
    for (u32 i = g_pending_write_head; i != g_pending_write_tail;) {
        --i;
        const auto& w = g_pending_writes[i & (PendingWriteCap - 1)];
        if (w.end_word_pos <= g_consumed_cache) {
            break;
        }
        const u64 w_end = w.begin + w.bytes;
        if (gpu_addr >= w_end || read_end <= w.begin) {
            continue;
        }
        if (w.ring_data != nullptr && w.begin <= gpu_addr && read_end <= w_end &&
            (cur_write_pos - w.start_word_pos) < (RingWords - 4096)) [[likely]] {
            std::memcpy(dst, w.ring_data + (gpu_addr - w.begin), bytes);
            ++g_h2_stlf_hits;
            return ForwardResult::Forwarded;
        }
        ++g_h2_stlf_misses;
        return ForwardResult::NeedDrain;
    }
    return ForwardResult::NoPending;
}

/// Stage A helper: reads a scalar `T` from guest GPU memory using STLF under H2Async(),
/// draining GPUBackB only on an unforwardable overlap.
template <typename T, typename Memory>
inline T ReadWithForward(const Memory& gpu_memory, u64 gpu_addr) {
    if (H2Async() && OnStageA()) [[unlikely]] {
        T val{};
        const auto fwd = TryForwardRead(gpu_addr, &val, sizeof(T));
        if (fwd == ForwardResult::Forwarded) {
            return val;
        }
        if (fwd == ForwardResult::NeedDrain) {
            DrainAsyncH2(H2DrainReason::GuardRead);
        }
    }
    return gpu_memory.template Read<T>(gpu_addr);
}

/// Stage A under H2Async(): consumed-aware fast check against pending writes, draining only on
/// exact byte range overlap with an unconsumed GPUBackB guest write.
inline void GuardReadExact(u64 gpu_addr, u64 bytes) {
    if (bytes == 0 || (g_pending_write_head == g_pending_write_tail && !g_pending_write_overflow))
        [[likely]] {
        return;
    }
    const u64 first_page = gpu_addr >> 12;
    const u64 last_page = (gpu_addr + bytes - 1) >> 12;
    auto any_page_after = [&](u64 pos) {
        for (u64 page = first_page; page <= last_page; ++page) {
            if (g_pending_pages[page & 4095] > pos) [[unlikely]] {
                return true;
            }
        }
        return false;
    };
    if (!any_page_after(g_consumed_cache)) [[likely]] {
        return;
    }
    g_consumed_cache = g_state.ConsumedPosition();
    if (!any_page_after(g_consumed_cache)) [[likely]] {
        return;
    }
    if (g_pending_write_overflow) [[unlikely]] {
        DrainAsyncH2(H2DrainReason::GuardRead);
        return;
    }
    while (g_pending_write_tail != g_pending_write_head &&
           g_pending_writes[g_pending_write_tail & (PendingWriteCap - 1)].end_word_pos <=
               g_consumed_cache) {
        ++g_pending_write_tail;
    }
    const u64 read_end = gpu_addr + bytes;
    for (u32 i = g_pending_write_head; i != g_pending_write_tail;) {
        --i;
        const auto& w = g_pending_writes[i & (PendingWriteCap - 1)];
        if (w.end_word_pos <= g_consumed_cache) {
            break;
        }
        if (gpu_addr < w.begin + w.bytes && w.begin < read_end) [[unlikely]] {
            DrainAsyncH2(H2DrainReason::GuardRead);
            return;
        }
    }
}

enum class GuardSite : u8 { Macro = 0, Query = 1, Other = 2 };

/// Stage A, before reading guest memory: waits for GPUBackB when a write it was sent there may
/// not have landed yet (a hashed page can drain needlessly, never miss).
inline void GuardRead(u64 gpu_addr, u64 bytes, GuardSite site) {
    if (bytes == 0) {
        return;
    }
    if (H2Async()) {
        GuardReadExact(gpu_addr, bytes);
        return;
    }
    const u64 drained = g_state.DrainedPosition();
    for (u64 page = gpu_addr >> 12; page <= (gpu_addr + bytes - 1) >> 12; ++page) {
        if (g_pending_pages[page & 4095] > drained) [[unlikely]] {
            g_state.CountGuardDrain(static_cast<size_t>(site));
            g_state.DrainAsync();
            return;
        }
    }
}

inline void MaybeReport() {
    if (RingEnabled()) [[unlikely]] {
        g_state.MaybeReport();
    }
}

} // namespace VideoCommon::BackSplit
