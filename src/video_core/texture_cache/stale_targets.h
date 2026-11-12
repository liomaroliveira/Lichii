// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/stale_pages.h"

namespace VideoCommon {

// Match the age at which critical collection drops bad overlaps without downloading them.
inline constexpr u64 STALE_TARGET_AGE = 10;

[[nodiscard]] inline bool GpuWrittenRenderTarget(const ImageBase& image) {
    return True(image.flags & ImageFlagBits::RenderTargetOrigin) && image.info.num_samples == 1 &&
           !image.info.is_sparse && !image.HasScaled() &&
           True(image.flags & ImageFlagBits::GpuModified) &&
           False(image.flags &
                 (ImageFlagBits::CpuModified | ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                  ImageFlagBits::SparseTiles | ImageFlagBits::Unbacked | ImageFlagBits::Rescaled |
                  ImageFlagBits::Remapped | ImageFlagBits::TrimmedLevels));
}

[[nodiscard]] inline bool StaleMarkerCovers(const ImageBase& image) {
    // MarkStale retains only whole pages; partial edge pages could upload obsolete guest bytes.
    constexpr u64 page_mask = StalePages::PAGE_SIZE - 1;
    return image.guest_size_bytes != 0 && (image.cpu_addr & page_mask) == 0 &&
           (image.guest_size_bytes & page_mask) == 0;
}

} // namespace VideoCommon
