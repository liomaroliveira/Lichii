// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEST (E02.T2, sensitivity study): burns a fixed share of a thread's wall time, spread over its
// events, to see how much the frame rate depends on that thread.
//   EDEN_NX2_DELAY_BACK_MS  = ms per second burnt on the split's back (per executed entry)
//   EDEN_NX2_DELAY_FRONT_MS = ms per second burnt on the front (per command batch)
// Time owed accrues with wall time and is paid at the next event, at most MaxSpinNs at once, so
// an idle thread does not pay it later in one block. Every 10 s the log says what was burnt.

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::Sensitivity {

class Burner {
public:
    explicit Burner(const char* env, const char* name_) : name{name_} {
        if (const char* value = std::getenv(env)) {
            ms_per_second = std::strtoull(value, nullptr, 10);
        }
    }

    void Event() {
        if (ms_per_second == 0) [[likely]] {
            return;
        }
        const u64 now = Now();
        if (start == 0) {
            start = now;
            window_start = now;
        }
        ++events;
        // ns owed so far: elapsed * ms_per_second / 1000.
        const u64 owed = (now - start) / 1000 * ms_per_second;
        if (owed > burnt) {
            u64 spin = owed - burnt;
            if (spin > MaxSpinNs) {
                // Idle time is not paid back: drop what exceeds one spin.
                dropped += spin - MaxSpinNs;
                burnt += spin - MaxSpinNs;
                spin = MaxSpinNs;
            }
            const u64 end = now + spin;
            u64 t = now;
            while (t < end) {
                t = Now();
            }
            burnt += spin;
            window_burnt += spin;
        }
        if (now - window_start >= 10'000'000'000ULL) {
            const double seconds = static_cast<double>(now - window_start) / 1e9;
            LOG_INFO(HW_GPU, "Sensitivity {}: burnt {:.1f} ms/s over {:.0f} events/s (target {} ms/s)",
                     name, static_cast<double>(window_burnt) / 1e6 / seconds,
                     static_cast<double>(events) / seconds, ms_per_second);
            window_start = now;
            window_burnt = 0;
            events = 0;
        }
    }

private:
    static constexpr u64 MaxSpinNs = 50'000;

    static u64 Now() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }

    const char* name;
    u64 ms_per_second{};
    u64 start{};
    u64 burnt{};
    u64 dropped{};
    u64 window_start{};
    u64 window_burnt{};
    u64 events{};
};

/// One per thread: each is touched only by its own thread.
inline Burner& Back() {
    static Burner burner{"EDEN_NX2_DELAY_BACK_MS", "back"};
    return burner;
}

inline Burner& Front() {
    static Burner burner{"EDEN_NX2_DELAY_FRONT_MS", "front"};
    return burner;
}

} // namespace VideoCommon::Sensitivity
