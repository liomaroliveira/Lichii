// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <optional>

#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_download_area.h"

namespace VideoCommon::Nx2FastDraw {

void DeviceSpans::Update(const Tegra::MemoryManager& memory_, GPUVAddr address_, u64 size_) {
    const u64 serial = memory_.MapSerial();
    if (memory == &memory_ && address == address_ && size == size_ && map_serial == serial) {
        return;
    }
    memory = &memory_;
    address = address_;
    size = size_;
    map_serial = serial;
    spans.clear();
    const auto submapped = memory_.GetSubmappedDeviceRange(address_, size_);
    // Segments come in GPU order, joined where device memory continues.
    linear = submapped.size() == 1 && submapped[0].second == size_;
    linear_base = linear ? submapped[0].first : 0;
    for (const auto& [device, bytes] : submapped) {
        spans.emplace_back(device, device + bytes);
    }
    std::ranges::sort(spans);
    size_t merged = 0;
    for (const auto& span : spans) {
        if (merged != 0 && span.first <= spans[merged - 1].second) {
            spans[merged - 1].second = (std::max)(spans[merged - 1].second, span.second);
        } else {
            spans[merged++] = span;
        }
    }
    spans.resize(merged);
}

bool DeviceSpans::Intersects(DAddr begin, DAddr end) const {
    const auto span = std::ranges::upper_bound(spans, begin, {}, &std::pair<DAddr, DAddr>::second);
    return span != spans.end() && span->first < end;
}

void NoteWrittenData(Tegra::MemoryManager& memory, GPUVAddr begin, const void* words, u64 bytes) {
    static_cast<void>(WriteData(memory, begin, words, bytes));
}

bool WriteData(Tegra::MemoryManager& memory, GPUVAddr begin, const void* words, u64 bytes) {
    if (!Enabled() || bytes == 0) {
        return false;
    }
    constexpr u64 PageSize = 0x1000; // the smallest GPU page
    const GPUVAddr end = begin + bytes;
    // Within one page the words are contiguous in host memory. Words memory holds already leave
    // memory, and so every handle and uniform read from it, the same: they stay out of the delta.
    if (words != nullptr && (begin & (PageSize - 1)) + bytes <= PageSize) {
        if (const u8* const held = memory.GetPointer<u8>(begin);
            held != nullptr && std::memcmp(held, words, bytes) == 0) {
            g_unchanged_writes.fetch_add(1, std::memory_order_relaxed);
            // Writing them again changes nothing either, unless the GPU caches may hold the
            // words newer than memory, or a download for a guest read may still land over them:
            // the write must win then.
            const std::optional<DAddr> device = memory.GpuToCpuAddress(begin);
            return device && !Tegra::Nx2MayBeGpuWritten(*device, bytes) &&
                   !VideoCore::ReadDownloadsMayCover(*device, bytes);
        }
    }
    NoteConstantData(begin, end);
    // The same device memory can be mapped at other GPU addresses: note where it lands.
    for (GPUVAddr page = begin; page < end;) {
        const GPUVAddr page_end = (std::min)(end, (page & ~(PageSize - 1)) + PageSize);
        if (const std::optional<DAddr> device = memory.GpuToCpuAddress(page)) {
            NoteDeviceData(*device, *device + (page_end - page));
        }
        page = page_end;
    }
    return false;
}

} // namespace VideoCommon::Nx2FastDraw
