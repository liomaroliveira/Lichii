// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include "common/common_types.h"
#include "common/logging.h"
#include "common/thread.h"

namespace VideoCommon::BackSplit {

inline bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_BACK_SPLIT");
    return value != nullptr && value[0] == '1';
}();

/// Enables the Stage A -> Stage B SPSC prototype ring (E04.T5+).
inline bool Enabled() noexcept {
    return g_enabled;
}

enum class Kind : u8 {
    Wrap = 0,
    RegDeltas = 1,
    CbUpload = 2,
    Draw = 3,
    DrawTexture = 4,
    Clear = 5,
    LaunchDma = 6,
    InlineUpload = 7,
    Query = 8,
};

constexpr size_t RegCount = 0xE00;
constexpr size_t MaskWords = RegCount / 64;
constexpr size_t RingWords = size_t{1} << 20; // 8 MiB SPSC ring
constexpr size_t RingMask = RingWords - 1;
constexpr u64 PublishBatchWords = 4096;
constexpr u32 CbCoalesceCap = 256;

inline constexpr u64 MakeHeader(Kind kind, u32 aux = 0) noexcept {
    return static_cast<u64>(kind) | (u64{aux} << 8);
}

/// Computes the total u64 word span of a packet from its header word (0 means wrap to ring[0]).
inline constexpr size_t PacketWords(u64 header) noexcept {
    const auto kind = static_cast<Kind>(header & 0xffU);
    const u32 aux = static_cast<u32>(header >> 8);
    switch (kind) {
    case Kind::Wrap:
        return 0;
    case Kind::RegDeltas:
        return 1 + static_cast<size_t>(aux);
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
        if (started.load(std::memory_order_acquire)) {
            return;
        }
        bool expected = false;
        if (!starting.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            while (!started.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            return;
        }
        ring = std::make_unique<u64[]>(RingWords);
        shadow_regs.fill(0);
        dirty_mask.fill(0);
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

    void NoteRegisterWrite(u32 method) noexcept {
        if (method < RegCount) {
            dirty_mask[method >> 6] |= 1ULL << (method & 63U);
        }
    }

    void FlushPendingCb() {
        if (pending_cb_count == 0) {
            return;
        }
        EmitCbPacket(pending_cb_addr, pending_cb_words.data(), pending_cb_count);
        pending_cb_count = 0;
        pending_cb_addr = 0;
    }

    void FlushRegDeltas(const u32* reg_array) {
        u32 count = 0;
        for (size_t w = 0; w < MaskWords; ++w) {
            u64 bits = dirty_mask[w];
            if (bits == 0) {
                continue;
            }
            dirty_mask[w] = 0;
            const u32 base = static_cast<u32>(w * 64);
            while (bits != 0) {
                const u32 bit = static_cast<u32>(std::countr_zero(bits));
                bits &= bits - 1;
                const u32 reg = base + bit;
                const u32 val = reg_array[reg];
                if (shadow_regs[reg] != val) {
                    shadow_regs[reg] = val;
                    scratch_deltas[count++] = (u64{val} << 32) | u64{reg};
                }
            }
        }
        if (count == 0) {
            return;
        }
        const size_t words = 1 + count;
        u64* dst = Reserve(words);
        dst[0] = MakeHeader(Kind::RegDeltas, count);
        std::memcpy(dst + 1, scratch_deltas.data(), count * sizeof(u64));
        Commit(words);
        ++stat_packets;
        stat_delta_regs += count;
    }

    void PushCbUpload(u64 gpu_addr, const u32* data, u32 count) {
        if (count == 0) {
            return;
        }
        EnsureStarted();
        stat_cb_words += count;
        const u64 expected_next = pending_cb_addr + u64{pending_cb_count} * sizeof(u32);
        if (pending_cb_count > 0 && gpu_addr == expected_next &&
            pending_cb_count + count <= CbCoalesceCap) {
            std::memcpy(&pending_cb_words[pending_cb_count], data, static_cast<size_t>(count) * sizeof(u32));
            pending_cb_count += count;
            return;
        }
        FlushPendingCb();
        if (count <= CbCoalesceCap) {
            pending_cb_addr = gpu_addr;
            pending_cb_count = count;
            std::memcpy(pending_cb_words.data(), data, static_cast<size_t>(count) * sizeof(u32));
            return;
        }
        EmitCbPacket(gpu_addr, data, count);
    }

    void PushDraw(const u32* reg_array, bool draw_indexed, bool indirect, u32 instance_count) {
        EnsureStarted();
        FlushPendingCb();
        FlushRegDeltas(reg_array);
        u64* dst = Reserve(2);
        const u32 flags = (draw_indexed ? 1U : 0U) | (indirect ? 2U : 0U);
        dst[0] = MakeHeader(Kind::Draw, flags);
        dst[1] = instance_count;
        Commit(2);
        ++stat_packets;
    }

    void PushDrawTexture(const u32* reg_array) {
        EnsureStarted();
        FlushPendingCb();
        FlushRegDeltas(reg_array);
        u64* dst = Reserve(1);
        dst[0] = MakeHeader(Kind::DrawTexture, 0);
        Commit(1);
        ++stat_packets;
    }

    void PushClear(const u32* reg_array, u32 layer_count) {
        EnsureStarted();
        FlushPendingCb();
        FlushRegDeltas(reg_array);
        u64* dst = Reserve(1);
        dst[0] = MakeHeader(Kind::Clear, layer_count);
        Commit(1);
        ++stat_packets;
    }

    void PushLaunchDma(u64 src_addr, u64 dst_addr, u64 byte_size_and_flags) {
        EnsureStarted();
        FlushPendingCb();
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
        EnsureStarted();
        FlushPendingCb();
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

    void PushQuery(const u32* reg_array, u64 gpu_addr, u64 payload, u64 meta) {
        EnsureStarted();
        FlushPendingCb();
        FlushRegDeltas(reg_array);
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
                 "Nx2BackSplit: packets {:.0f}/s, delta_regs {:.0f}/s, cb_words {:.0f}/s, "
                 "words {:.0f}/s, A_stalls {}",
                 static_cast<double>(stat_packets) / sec,
                 static_cast<double>(stat_delta_regs) / sec,
                 static_cast<double>(stat_cb_words) / sec,
                 static_cast<double>(stat_words) / sec,
                 stat_stalls);
        stat_packets = 0;
        stat_delta_regs = 0;
        stat_cb_words = 0;
        stat_words = 0;
        stat_stalls = 0;
    }

private:
    void EmitCbPacket(u64 gpu_addr, const u32* data, u32 count) {
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
    }

    static u64 NowNs() noexcept {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }

    u64* Reserve(size_t words) {
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }

    u64* ReserveSlow(size_t words) {
        const size_t index = write_position & RingMask;
        if (index + words > RingWords) {
            const size_t pad = RingWords - index;
            WaitForSpace(pad);
            ring[index] = MakeHeader(Kind::Wrap, 0);
            write_position += pad;
            PublishNow();
        }
        WaitForSpace(words);
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
        stat_words += words;
        if (write_position - published_local >= PublishBatchWords ||
            consumer_waiting.load(std::memory_order_relaxed)) {
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
        if (consumer_waiting.load(std::memory_order_seq_cst)) {
            consumer_wake.fetch_add(1, std::memory_order_seq_cst);
            consumer_wake.notify_one();
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
        u64 read_pos = 0;
        u64 sink = 0;
        while (!stop_token.stop_requested()) {
            u64 available = published.load(std::memory_order_acquire);
            if (available == read_pos) {
                for (u32 spin = 0; spin < 2048 && available == read_pos; ++spin) {
                    available = published.load(std::memory_order_acquire);
                }
                if (available == read_pos) {
                    consumer_waiting.store(true, std::memory_order_seq_cst);
                    const u32 seen = consumer_wake.load(std::memory_order_seq_cst);
                    available = published.load(std::memory_order_seq_cst);
                    if (available == read_pos && !stop_token.stop_requested()) {
                        consumer_wake.wait(seen, std::memory_order_seq_cst);
                    }
                    consumer_waiting.store(false, std::memory_order_relaxed);
                    continue;
                }
            }
            u64 last_store = read_pos;
            while (read_pos != available) {
                const size_t index = read_pos & RingMask;
                const u64 head = ring[index];
                const size_t words = PacketWords(head);
                if (words == 0) {
                    read_pos += RingWords - index;
                } else {
                    for (size_t i = 0; i < words; ++i) {
                        sink ^= ring[index + i];
                    }
                    read_pos += words;
                }
                if (read_pos - last_store >= 4096 && read_pos != available) {
                    consumed.store(read_pos, std::memory_order_release);
                    last_store = read_pos;
                    NotifyProducer();
                }
            }
            consumed.store(read_pos, std::memory_order_release);
            NotifyProducer();
        }
        consumer_sink.store(sink, std::memory_order_relaxed);
    }

    std::atomic<bool> starting{false};
    std::atomic<bool> started{false};
    std::unique_ptr<u64[]> ring;
    std::array<u32, RegCount> shadow_regs{};
    std::array<u64, MaskWords> dirty_mask{};
    std::array<u64, RegCount> scratch_deltas{};
    std::array<u32, CbCoalesceCap> pending_cb_words{};
    u64 pending_cb_addr = 0;
    u32 pending_cb_count = 0;

    u64 write_position = 0;
    u64 published_local = 0;
    u64 consumed_cache = 0;
    u64 last_report_ns = 0;
    u64 stat_packets = 0;
    u64 stat_delta_regs = 0;
    u64 stat_cb_words = 0;
    u64 stat_words = 0;
    u64 stat_stalls = 0;

    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    alignas(64) std::atomic<u32> consumer_wake{0};
    alignas(64) std::atomic<bool> consumer_waiting{false};
    alignas(64) std::atomic<u32> producer_wake{0};
    alignas(64) std::atomic<bool> producer_waiting{false};
    std::atomic<u64> consumer_sink{0};
    std::jthread consumer;
};

inline RingState g_state;

inline void NoteRegisterWrite(u32 method) noexcept {
    g_state.NoteRegisterWrite(method);
}

inline void EmitCbUpload(u64 gpu_addr, const u32* data, u32 count) {
    g_state.PushCbUpload(gpu_addr, data, count);
}

inline void EmitDraw(const u32* reg_array, bool draw_indexed, bool indirect, u32 instance_count) {
    g_state.PushDraw(reg_array, draw_indexed, indirect, instance_count);
}

inline void EmitDrawTexture(const u32* reg_array) {
    g_state.PushDrawTexture(reg_array);
}

inline void EmitClear(const u32* reg_array, u32 layer_count) {
    g_state.PushClear(reg_array, layer_count);
}

inline void EmitLaunchDma(u64 src_addr, u64 dst_addr, u64 byte_size_and_flags) {
    g_state.PushLaunchDma(src_addr, dst_addr, byte_size_and_flags);
}

inline void EmitInlineUpload(u64 dst_addr, const void* data, size_t bytes) {
    g_state.PushInlineUpload(dst_addr, data, bytes);
}

inline void EmitQuery(const u32* reg_array, u64 gpu_addr, u64 payload, u64 meta) {
    g_state.PushQuery(reg_array, gpu_addr, payload, meta);
}

inline void MaybeReport() {
    if (Enabled()) [[unlikely]] {
        g_state.MaybeReport();
    }
}

} // namespace VideoCommon::BackSplit
