// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace Common {

/// EDEN_NX2_GPU_PROFILE: contended SpinRecursiveMutex acquisitions, their wait and how many slept.
struct SpinMutexContention {
    std::uint64_t count{};
    std::uint64_t ns{};
    std::uint64_t sleeps{};
};
struct SharedSpinMutexContention {
    std::atomic<std::uint64_t> count{};
    std::atomic<std::uint64_t> ns{};
    std::atomic<std::uint64_t> sleeps{};
};
inline bool g_count_spin_contention{};
inline thread_local SpinMutexContention t_spin_contention{};
inline SharedSpinMutexContention g_spin_contention{};

/// EDEN_NX2_GPU_PROFILE: a waiter's contended time by lock and by the thread that held it.
enum class ContentionLock : std::uint8_t { Other, BufferCache, TextureCache, Count };
inline constexpr std::size_t MaxContentionHolders = 64;
/// Indexed by the id NameContentionHolder gave a thread (0: a thread without a name).
inline std::array<std::atomic<const char*>, MaxContentionHolders> g_contention_holder_names{};
inline thread_local std::uint32_t t_contention_holder{};
inline thread_local std::array<std::array<SpinMutexContention, MaxContentionHolders>,
                               static_cast<std::size_t>(ContentionLock::Count)>
    t_contention_by_holder{};
/// Names the calling thread as a lock holder (SetCurrentThreadName calls it).
void NameContentionHolder(const char* name);

/// EDEN_NX2_GPU_PROFILE: contention totals by lock (all threads) and per-core / per-holder on CPU cores.
inline constexpr std::size_t MaxWatchedCpuCores = 8;
inline thread_local std::uint8_t t_watched_cpu_core{0xFF};
inline std::array<SharedSpinMutexContention, static_cast<std::size_t>(ContentionLock::Count)>
    g_spin_contention_by_lock{};
inline std::array<std::array<SharedSpinMutexContention, static_cast<std::size_t>(ContentionLock::Count)>,
                  MaxWatchedCpuCores>
    g_cpu_contention_by_core{};
inline std::array<std::array<SharedSpinMutexContention, MaxContentionHolders>,
                  static_cast<std::size_t>(ContentionLock::Count)>
    g_cpu_contention_by_holder{};

/// One busy-wait step: tells the core this thread spins (less power, more for its sibling).
inline void SpinPause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

/// A recursive mutex that spins briefly before sleeping. The GPU caches' locks are taken tens of
/// thousands of times a second by the GPU thread, the fencing thread and CPU write traps, and
/// held for microseconds: std::recursive_mutex put a waiting thread to sleep at once, and its
/// wake-up took longer than the critical section (title 1's GPU thread spent ~8% asleep there).
class SpinRecursiveMutex {
public:
    SpinRecursiveMutex() = default;
    explicit SpinRecursiveMutex(ContentionLock lock_) : contention_lock{lock_} {}

    void lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return;
        }
        std::uint32_t expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            LockContended();
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
        if (g_count_spin_contention) {
            holder.store(t_contention_holder, std::memory_order_relaxed);
        }
    }

    bool try_lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return true;
        }
        std::uint32_t expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            return false;
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
        if (g_count_spin_contention) {
            holder.store(t_contention_holder, std::memory_order_relaxed);
        }
        return true;
    }

    void unlock() {
        if (--depth != 0) {
            return;
        }
        owner.store(std::thread::id{}, std::memory_order_relaxed);
        if (state.exchange(0, std::memory_order_release) == 2) {
            state.notify_one();
        }
    }

private:
    // About 10 microseconds of spinning before sleeping.
    static constexpr int SpinCount = 1024;

    static void Pause() {
        SpinPause();
    }

    void LockContended() {
        if (g_count_spin_contention) {
            const std::uint32_t held_by = holder.load(std::memory_order_relaxed);
            const auto start = std::chrono::steady_clock::now();
            const bool slept = Acquire();
            const auto ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count());
            ++t_spin_contention.count;
            t_spin_contention.ns += ns;
            t_spin_contention.sleeps += slept ? 1 : 0;
            const auto lock_idx = static_cast<std::size_t>(contention_lock);
            const auto held_idx = held_by < MaxContentionHolders ? held_by : 0;
            const std::uint64_t sleep_add = slept ? 1 : 0;
            auto& by_holder = t_contention_by_holder[lock_idx][held_idx];
            ++by_holder.count;
            by_holder.ns += ns;
            by_holder.sleeps += sleep_add;
            auto add_shared = [ns, sleep_add](SharedSpinMutexContention& dst) {
                dst.count.fetch_add(1, std::memory_order_relaxed);
                dst.ns.fetch_add(ns, std::memory_order_relaxed);
                if (sleep_add != 0) {
                    dst.sleeps.fetch_add(sleep_add, std::memory_order_relaxed);
                }
            };
            add_shared(g_spin_contention);
            add_shared(g_spin_contention_by_lock[lock_idx]);
            if (t_watched_cpu_core < MaxWatchedCpuCores) {
                add_shared(g_cpu_contention_by_core[t_watched_cpu_core][lock_idx]);
                add_shared(g_cpu_contention_by_holder[lock_idx][held_idx]);
            }
            return;
        }
        Acquire();
    }

    /// Returns whether it slept.
    bool Acquire() {
        for (int spin = 0; spin < SpinCount; ++spin) {
            Pause();
            if (state.load(std::memory_order_relaxed) == 0) {
                std::uint32_t expected = 0;
                if (state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
                    return false;
                }
            }
        }
        // 2: held with (possible) sleepers; taken this way it stays marked, so its unlock wakes
        // the next sleeper.
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            state.wait(2, std::memory_order_relaxed);
        }
        return true;
    }

    std::atomic<std::uint32_t> state{0}; ///< 0 free, 1 held, 2 held with sleepers
    std::atomic<std::thread::id> owner{};
    std::uint32_t depth{}; ///< Recursion depth, touched only by the owner
    /// EDEN_NX2_GPU_PROFILE: the holder's t_contention_holder, for the waits it causes.
    std::atomic<std::uint32_t> holder{};
    ContentionLock contention_lock{ContentionLock::Other};
};

} // namespace Common
