// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <initializer_list>

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/stale_targets.h"

namespace {

using VideoCommon::GpuWrittenRenderTarget;
using VideoCommon::ImageBase;
using VideoCommon::ImageFlagBits;
using VideoCommon::NullImageParams;
using VideoCommon::StaleMarkerCovers;
using VideoCommon::StalePages;

ImageBase Target() {
    ImageBase image{NullImageParams{}};
    image.flags = ImageFlagBits::RenderTargetOrigin | ImageFlagBits::GpuModified;
    image.cpu_addr = 0x10000;
    image.guest_size_bytes = 4 * StalePages::PAGE_SIZE;
    return image;
}

} // namespace

TEST_CASE("Stale targets require render-target origin and GPU contents",
          "[video_core][texture_cache]") {
    ImageBase image = Target();
    REQUIRE(GpuWrittenRenderTarget(image));
    image.flags &= ~ImageFlagBits::RenderTargetOrigin;
    CHECK_FALSE(GpuWrittenRenderTarget(image));
    image = Target();
    image.flags &= ~ImageFlagBits::GpuModified;
    CHECK_FALSE(GpuWrittenRenderTarget(image));

    for (const ImageFlagBits flag :
         {ImageFlagBits::CpuModified, ImageFlagBits::Sparse, ImageFlagBits::SparseLayers,
          ImageFlagBits::SparseTiles, ImageFlagBits::Unbacked, ImageFlagBits::Rescaled,
          ImageFlagBits::Remapped, ImageFlagBits::TrimmedLevels}) {
        image = Target();
        image.flags |= flag;
        CHECK_FALSE(GpuWrittenRenderTarget(image));
    }
    image = Target();
    image.info.is_sparse = true;
    CHECK_FALSE(GpuWrittenRenderTarget(image));
    image = Target();
    image.info.num_samples = 4;
    CHECK_FALSE(GpuWrittenRenderTarget(image));
    image = Target();
    image.has_scaled = true;
    CHECK_FALSE(GpuWrittenRenderTarget(image));
}

TEST_CASE("Stale target eligibility guarantees complete page markers",
          "[video_core][texture_cache]") {
    ImageBase image = Target();
    REQUIRE(StaleMarkerCovers(image));
    StalePages pages;
    pages.Mark(image.cpu_addr, image.guest_size_bytes, [](DAddr, u64) {});
    REQUIRE(pages.IsAll(image.cpu_addr, image.guest_size_bytes));
    pages.Clear(image.cpu_addr + 1, 1, [](DAddr, u64) {});
    CHECK_FALSE(pages.IsAll(image.cpu_addr, image.guest_size_bytes));

    // Sub-page and unaligned allocations must stay: marking cannot cover their edge pages.
    for (const u32 size : {1536U, 4096U}) {
        for (const u64 offset : {0ULL, 512ULL}) {
            image = Target();
            image.cpu_addr += offset;
            image.guest_size_bytes = size;
            StalePages marked;
            marked.Mark(image.cpu_addr, image.guest_size_bytes, [](DAddr, u64) {});
            CHECK(StaleMarkerCovers(image) == marked.IsAll(image.cpu_addr, image.guest_size_bytes));
        }
    }
    image.guest_size_bytes = 0;
    CHECK_FALSE(StaleMarkerCovers(image));
}
