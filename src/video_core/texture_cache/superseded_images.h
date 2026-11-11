// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <utility>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "video_core/texture_cache/image_base.h"

namespace VideoCommon {

enum class Supersession : u8 {
    None,       ///< Keep the image.
    Unproven,   ///< Newer images cover it, but its GPU-written bytes may outlive their writes.
    Clean,      ///< Newer images cover it and it holds no GPU data a guest read could want.
    Rewritten,  ///< Newer images cover it and each cleared itself whole after its last GPU write.
};

/// Whether the memory of a bad-overlapping image belongs entirely to images used after it, so
/// that deleting it without a download changes no byte a guest read, or a join, can return.
///
/// A guest read downloads the overlapping images in modification_tick order and a join copies
/// them in that order, each over its whole range: an older image is only seen where no newer one
/// reaches. An image holding GPU data is therefore dropped only when every byte of its range lies
/// in newer images that still hold GPU data (downloadable) and cleared their whole surface after
/// its last write (full_write_tick): those bytes were rewritten, whatever was drawn since. One
/// that only covers a part, or whose cover was not cleared whole, is Unproven and stays.
template <typename Images>
[[nodiscard]] Supersession ClassifySuperseded(const Images& images, const ImageBase& image) {
    constexpr ImageFlagBits Excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::Unbacked | ImageFlagBits::UnwrittenSparse | ImageFlagBits::Remapped |
        ImageFlagBits::IsDecoding | ImageFlagBits::TrimmedLevels;
    if (image.overlapping_images.empty() || !image.aliased_images.empty() ||
        True(image.flags & Excluded) || image.cpu_addr >= image.cpu_addr_end) {
        return Supersession::None;
    }
    const bool holds_gpu_data = image.IsSafeGpuCopy();
    using Span = std::pair<VAddr, VAddr>;
    boost::container::small_vector<Span, 8> used_after;
    boost::container::small_vector<Span, 8> rewritten_after;
    for (const ImageId overlap_id : image.overlapping_images) {
        const ImageBase& other = images[overlap_id];
        if (other.last_use_frame <= image.last_use_frame ||
            True(other.flags & (Excluded | ImageFlagBits::CpuModified)) ||
            False(other.flags & ImageFlagBits::Registered)) {
            continue;
        }
        const Span span{(std::max)(other.cpu_addr, image.cpu_addr),
                        (std::min)(other.cpu_addr_end, image.cpu_addr_end)};
        if (span.first >= span.second) {
            continue;
        }
        used_after.push_back(span);
        if (True(other.flags & ImageFlagBits::GpuModified) &&
            other.modification_tick > image.modification_tick &&
            other.full_write_tick > image.modification_tick) {
            rewritten_after.push_back(span);
        }
    }
    const auto covers = [&image](boost::container::small_vector<Span, 8>& spans) {
        std::ranges::sort(spans);
        VAddr reached = image.cpu_addr;
        for (const auto& [begin, end] : spans) {
            if (begin > reached) {
                return false;
            }
            reached = (std::max)(reached, end);
        }
        return reached >= image.cpu_addr_end;
    };
    if (!holds_gpu_data) {
        return covers(used_after) ? Supersession::Clean : Supersession::None;
    }
    if (covers(rewritten_after)) {
        return Supersession::Rewritten;
    }
    return covers(used_after) ? Supersession::Unproven : Supersession::None;
}

} // namespace VideoCommon
