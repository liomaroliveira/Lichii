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

inline bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_BACK_SPLIT");
    return value != nullptr && value[0] == '1';
}();

/// Enables the Stage A -> Stage B SPSC prototype ring (E04.T5+).
inline bool Enabled() noexcept {
    return g_enabled;
}

/// E04.T6: set on GPUBackB, the thread that owns the rasterizer with the split on.
inline thread_local bool t_on_b{};

/// Whether the calling thread is GPUBackB.
inline bool OnB() noexcept {
    return t_on_b;
}

/// GPUBackB, once at its start: marks it as a back thread for GpuSplit (OnBackThread and
/// OwnsScheduler hold on both stages). Defined in nx2_back_split_rasterizer.cpp.
void OnBThreadStart();

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
        LOG_INFO(HW_GPU,
                 "Nx2BackSplit shadow: engines {}, checks {}, diverged checks {}, diverged regs {} "
                 "(total since boot: checks {}, diverged checks {}); state checks {}, diverged "
                 "state {} (total {})",
                 b_engines.load(std::memory_order_relaxed),
                 b_checks.exchange(0, std::memory_order_relaxed),
                 b_bad_checks.exchange(0, std::memory_order_relaxed),
                 b_bad_regs.exchange(0, std::memory_order_relaxed),
                 b_checks_total.load(std::memory_order_relaxed),
                 b_bad_checks_total.load(std::memory_order_relaxed),
                 b_state_checks.exchange(0, std::memory_order_relaxed),
                 b_bad_state.exchange(0, std::memory_order_relaxed),
                 b_bad_state_total.load(std::memory_order_relaxed));
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
        if (write_position - published_local >= PublishBatchWords) [[unlikely]] {
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
                    reinterpret_cast<AsyncFn>(ring[index + 1])(&ring[index + 2]);
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
                task_fn(task_arg);
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

inline void EmitCbUpload(u64 gpu_addr, const u32* data, u32 count) {
    g_state.PushCbUpload(gpu_addr, data, count);
}

inline void EmitDraw(bool draw_indexed, bool indirect, u32 instance_count) {
    g_state.PushDraw(draw_indexed, indirect, instance_count);
}

inline void EmitDrawTexture() {
    g_state.PushDrawTexture();
}

inline void EmitClear(u32 layer_count) {
    g_state.PushClear(layer_count);
}

inline void EmitLaunchDma(u64 src_addr, u64 dst_addr, u64 byte_size_and_flags) {
    g_state.PushLaunchDma(src_addr, dst_addr, byte_size_and_flags);
}

inline void EmitInlineUpload(u64 dst_addr, const void* data, size_t bytes) {
    g_state.PushInlineUpload(dst_addr, data, bytes);
}

inline void EmitQuery(u64 gpu_addr, u64 payload, u64 meta) {
    g_state.PushQuery(gpu_addr, payload, meta);
}

inline void RunOnB(RingState::TaskFn fn, void* arg, bool hard = true) {
    g_state.RunOnB(fn, arg, hard);
}

inline void DrainAsync() {
    g_state.DrainAsync();
}

inline void Flush() {
    g_state.Flush();
}

/// E04.T6 step 4d: per GPU page (hashed), the ring position after the last guest write stage A
/// sent to GPUBackB there. Stage A only.
inline std::array<u64, 4096> g_pending_pages{};

/// Stage A: guest memory [gpu_addr, gpu_addr + bytes) is written by GPUBackB in ring order.
inline void NotePendingWrite(u64 gpu_addr, u64 bytes) {
    if (bytes == 0) {
        return;
    }
    const u64 position = g_state.WritePosition();
    for (u64 page = gpu_addr >> 12; page <= (gpu_addr + bytes - 1) >> 12; ++page) {
        g_pending_pages[page & 4095] = position;
    }
}

enum class GuardSite : u8 { Macro = 0, Query = 1, Other = 2 };

/// Stage A, before reading guest memory: waits for GPUBackB when a write it was sent there may
/// not have landed yet (a hashed page can drain needlessly, never miss).
inline void GuardRead(u64 gpu_addr, u64 bytes, GuardSite site) {
    if (bytes == 0) {
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
    if (Enabled()) [[unlikely]] {
        g_state.MaybeReport();
    }
}

} // namespace VideoCommon::BackSplit
