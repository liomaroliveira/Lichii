// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include "common/common_types.h"
#include "common/logging.h"
#include "common/thread.h"

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
};

constexpr size_t RegCount = 0xE00;
constexpr size_t RingWords = size_t{1} << 20; // 8 MiB SPSC ring
constexpr size_t RingMask = RingWords - 1;
constexpr u64 PublishBatchWords = 4096;

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
    void RunOnB(TaskFn fn, void* arg) {
        EnsureStarted();
        PublishNow();
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
    }

    void PushReg(u32 method, u32 argument) {
        if (method >= RegCount) [[unlikely]] {
            return;
        }
        u64* dst = Reserve(1);
        dst[0] = (u64{argument} << 32) | u64{method};
        Commit(1);
        ++stat_delta_regs;
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
                 "calls {:.0f}/s, deferred {:.0f}/s",
                 static_cast<double>(stat_packets) / sec,
                 static_cast<double>(stat_delta_regs) / sec,
                 static_cast<double>(stat_cb_words) / sec,
                 static_cast<double>(write_position - last_report_words) / sec,
                 stat_stalls, static_cast<double>(stat_run_on_b) / sec, stat_run_sleeps,
                 b_sleeps.exchange(0, std::memory_order_relaxed),
                 static_cast<double>(g_calls.exchange(0, std::memory_order_relaxed)) / sec,
                 static_cast<double>(g_deferred.exchange(0, std::memory_order_relaxed)) / sec);
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
            u64 last_store = read_pos;
            while (read_pos != available) {
                const size_t index = read_pos & RingMask;
                const u64 word0 = ring[index];
                const size_t words = PacketWords(word0);
                if (words == 0) {
                    read_pos += RingWords - index;
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
                task_done.store(task, std::memory_order_seq_cst);
                if (a_waiting.load(std::memory_order_seq_cst)) {
                    task_done.notify_one();
                }
            }
        }
        consumer_sink.store(sink, std::memory_order_relaxed);
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

inline void EmitReg(u32 method, u32 argument) {
    g_state.PushReg(method, argument);
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

inline void RunOnB(RingState::TaskFn fn, void* arg) {
    g_state.RunOnB(fn, arg);
}

inline void MaybeReport() {
    if (Enabled()) [[unlikely]] {
        g_state.MaybeReport();
    }
}

} // namespace VideoCommon::BackSplit
