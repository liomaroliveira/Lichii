// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/slot_vector.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/superseded_images.h"

namespace {

using VideoCommon::ClassifySuperseded;
using VideoCommon::ImageBase;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageId;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::Supersession;
using VideoCore::Surface::PixelFormat;

constexpr VAddr Base = 0x100000;
constexpr size_t PoolSize = 0x80000;
constexpr u8 GuestByte = 0xEE;

ImageInfo Info() {
    ImageInfo info;
    info.format = PixelFormat::D32_FLOAT;
    info.type = ImageType::e2D;
    info.size = {64, 64, 1};
    info.resources.levels = 1;
    info.resources.layers = 1;
    return info;
}

/// The images of a memory pool and the guest memory under them, as the texture cache keeps them:
/// every image holds `value` over its whole range, and a guest read (TextureCache::DownloadMemory)
/// writes the GPU-written images back in modification_tick order.
struct Pool {
    Common::SlotVector<ImageBase> images;
    std::vector<u8> values;
    u64 tick = 0;
    u64 frame = 0;

    /// An image created (bad overlap with every image it intersects) and cleared whole.
    ImageId Cleared(VAddr begin, VAddr end, u8 value) {
        const ImageId id = Create(begin, end, value);
        ImageBase& image = images[id];
        image.flags |= ImageFlagBits::GpuModified;
        image.flags &= ~ImageFlagBits::CpuModified;
        image.modification_tick = ++tick;
        image.full_write_tick = image.modification_tick;
        return id;
    }

    /// An image only drawn into: GPU-written, but nothing says it wrote all of its range.
    ImageId Drawn(VAddr begin, VAddr end, u8 value) {
        const ImageId id = Cleared(begin, end, value);
        images[id].full_write_tick = 0;
        return id;
    }

    ImageId Create(VAddr begin, VAddr end, u8 value) {
        const ImageId id = images.insert(Info(), begin, begin);
        ImageBase& image = images[id];
        image.cpu_addr_end = end;
        image.flags = ImageFlagBits::Registered;
        image.last_use_frame = frame;
        values.resize((std::max<size_t>)(values.size(), id.Value() + 1));
        values[id.Value()] = value;
        for (const ImageId other_id : AllIds()) {
            if (other_id != id && images[other_id].Overlaps(begin, end - begin)) {
                images[id].overlapping_images.push_back(other_id);
                images[other_id].overlapping_images.push_back(id);
            }
        }
        return id;
    }

    std::vector<ImageId> AllIds() {
        std::vector<ImageId> ids;
        for (const auto [id, image] : images) {
            static_cast<void>(image);
            ids.push_back(id);
        }
        return ids;
    }

    void Delete(ImageId id) {
        for (const ImageId other : images[id].overlapping_images) {
            std::erase(images[other].overlapping_images, id);
        }
        images.erase(id);
    }

    /// DeleteSupersededImages' pass: classify everything first, then delete.
    std::vector<ImageId> Prune() {
        std::vector<ImageId> deleted;
        for (const ImageId id : AllIds()) {
            const Supersession kind = ClassifySuperseded(images, images[id]);
            if (kind == Supersession::Clean || kind == Supersession::Rewritten) {
                deleted.push_back(id);
            }
        }
        for (const ImageId id : deleted) {
            Delete(id);
        }
        return deleted;
    }

    /// What a guest read of [begin, end) returns.
    std::vector<u8> GuestRead(VAddr begin, VAddr end) {
        std::vector<u8> memory(PoolSize, GuestByte);
        std::vector<ImageId> downloads;
        for (const ImageId id : AllIds()) {
            const ImageBase& image = images[id];
            if (image.Overlaps(begin, end - begin) && image.IsSafeGpuCopy()) {
                downloads.push_back(id);
            }
        }
        std::ranges::sort(downloads, [this](ImageId lhs, ImageId rhs) {
            return images[lhs].modification_tick < images[rhs].modification_tick;
        });
        for (const ImageId id : downloads) {
            const ImageBase& image = images[id];
            std::fill(memory.begin() + (image.cpu_addr - Base),
                      memory.begin() + (image.cpu_addr_end - Base), values[id.Value()]);
        }
        return {memory.begin() + (begin - Base), memory.begin() + (end - Base)};
    }
};

} // namespace

TEST_CASE("A render target pool moved by a few KiB frees the old targets once covered",
          "[video_core][texture_cache]") {
    Pool pool;
    const ImageId a = pool.Cleared(Base, Base + 0x10000, 1);
    const ImageId b = pool.Cleared(Base + 0x10000, Base + 0x20000, 2);
    const ImageId c = pool.Cleared(Base + 0x20000, Base + 0x30000, 3);
    pool.frame = 10;
    // The next frame's targets start 0x4000 later: the head of `a` is not theirs yet.
    pool.Cleared(Base + 0x4000, Base + 0x14000, 4);
    pool.Cleared(Base + 0x14000, Base + 0x24000, 5);
    pool.Cleared(Base + 0x24000, Base + 0x34000, 6);

    const auto before = pool.GuestRead(Base, Base + 0x40000);
    const auto deleted = pool.Prune();
    CHECK(deleted == std::vector{b, c});
    CHECK(pool.GuestRead(Base, Base + 0x40000) == before);
    CHECK(ClassifySuperseded(pool.images, pool.images[a]) == Supersession::None);

    // A later target takes the head of `a`: now nothing of it is left.
    pool.frame = 11;
    pool.Cleared(Base, Base + 0x4000, 7);
    const auto after_head = pool.GuestRead(Base, Base + 0x40000);
    CHECK(pool.Prune() == std::vector{a});
    CHECK(pool.GuestRead(Base, Base + 0x40000) == after_head);
    // Any read of the pool, whole or in parts, is what it was before the deletions.
    CHECK(pool.GuestRead(Base + 0x8000, Base + 0x9000) == std::vector<u8>(0x1000, 4));
}

TEST_CASE("An image only partly covered by newer images is kept", "[video_core][texture_cache]") {
    Pool pool;
    const ImageId old = pool.Cleared(Base, Base + 0x10000, 1);
    pool.frame = 10;
    pool.Cleared(Base, Base + 0x8000, 2);
    const auto before = pool.GuestRead(Base, Base + 0x10000);
    CHECK(pool.Prune().empty());
    CHECK(pool.GuestRead(Base, Base + 0x10000) == before);
    // The half nobody rewrote is still the old image's: a guest read returns it.
    CHECK(pool.GuestRead(Base + 0x8000, Base + 0x10000) == std::vector<u8>(0x8000, 1));
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::None);
}

TEST_CASE("A newer image that was only drawn into does not prove a rewrite",
          "[video_core][texture_cache]") {
    Pool pool;
    const ImageId old = pool.Cleared(Base, Base + 0x10000, 1);
    pool.frame = 10;
    // Covers all of it but its draw may have touched only a part, the rest being whatever it
    // was loaded with: the old image stays, a read after the next frames still finds its data.
    pool.Drawn(Base, Base + 0x10000, 2);
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::Unproven);
    CHECK(pool.Prune().empty());
    CHECK(pool.AllIds().size() == 2);
}

TEST_CASE("A newer image cleared before the old one's last write proves nothing",
          "[video_core][texture_cache]") {
    Pool pool;
    const ImageId newer = pool.Cleared(Base, Base + 0x10000, 2);
    pool.frame = 10;
    // Created first but written after: the old image holds the later data of the two.
    const ImageId old = pool.Create(Base, Base + 0x10000, 1);
    pool.images[old].flags |= ImageFlagBits::GpuModified;
    pool.images[old].modification_tick = ++pool.tick;
    pool.images[newer].last_use_frame = 11;
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::Unproven);
    CHECK(pool.Prune().empty());
}

TEST_CASE("Covers must be used after the image, and still hold GPU data",
          "[video_core][texture_cache]") {
    Pool pool;
    pool.frame = 10;
    const ImageId old = pool.Cleared(Base, Base + 0x10000, 1);
    const ImageId cover = pool.Cleared(Base, Base + 0x10000, 2);
    // Not used after the old one.
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::None);
    pool.images[cover].last_use_frame = 12;
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::Rewritten);
    // The guest wrote over the cover since: its GPU data is gone, so it proves nothing.
    pool.images[cover].flags |= ImageFlagBits::CpuModified;
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::None);
    pool.images[cover].flags &= ~ImageFlagBits::CpuModified;
    // Several covers together do.
    pool.Delete(cover);
    pool.Cleared(Base, Base + 0x8000, 3);
    pool.Cleared(Base + 0x8000, Base + 0x10000, 4);
    for (const ImageId id : pool.AllIds()) {
        pool.images[id].last_use_frame = id == old ? 10 : 12;
    }
    const auto before = pool.GuestRead(Base, Base + 0x10000);
    CHECK(pool.Prune() == std::vector{old});
    CHECK(pool.GuestRead(Base, Base + 0x10000) == before);
}

TEST_CASE("An image that holds no GPU data goes once newer images cover it",
          "[video_core][texture_cache]") {
    Pool pool;
    // A texture loaded from guest memory: nothing in it to download.
    const ImageId texture = pool.Create(Base, Base + 0x10000, 1);
    pool.frame = 10;
    const ImageId cover = pool.Create(Base, Base + 0x10000, 2);
    CHECK(ClassifySuperseded(pool.images, pool.images[texture]) == Supersession::Clean);
    // Written by the guest after the GPU wrote it: nothing a guest read would write back either.
    pool.images[texture].flags |= ImageFlagBits::GpuModified | ImageFlagBits::CpuModified;
    CHECK(ClassifySuperseded(pool.images, pool.images[texture]) == Supersession::Clean);
    // The cover being a plain upload does not matter, only that it is used after.
    pool.images[texture].flags &= ~ImageFlagBits::CpuModified;
    pool.images[texture].flags &= ~ImageFlagBits::GpuModified;
    pool.images[cover].last_use_frame = pool.frame;
    CHECK(pool.Prune() == std::vector{texture});
}

TEST_CASE("Aliased, sparse and unbacked images are left to the collector",
          "[video_core][texture_cache]") {
    Pool pool;
    const ImageId old = pool.Cleared(Base, Base + 0x10000, 1);
    pool.frame = 10;
    pool.Cleared(Base, Base + 0x10000, 2);
    REQUIRE(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::Rewritten);
    for (const ImageFlagBits flag : {ImageFlagBits::Sparse, ImageFlagBits::SparseLayers,
                                     ImageFlagBits::SparseTiles, ImageFlagBits::Unbacked,
                                     ImageFlagBits::UnwrittenSparse, ImageFlagBits::Remapped,
                                     ImageFlagBits::IsDecoding, ImageFlagBits::TrimmedLevels}) {
        pool.images[old].flags |= flag;
        CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::None);
        pool.images[old].flags &= ~flag;
    }
    pool.images[old].aliased_images.push_back({{}, ImageId{}});
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::None);
}

TEST_CASE("An image registered but never prepared is not a cover", "[video_core][texture_cache]") {
    Pool pool;
    const ImageId old = pool.Cleared(Base, Base + 0x10000, 1);
    pool.images[old].last_use_frame = 5;
    // Created by a lookup and cleared, but no draw or sample prepared it: last_use_frame is 0.
    const ImageId cover = pool.Cleared(Base, Base + 0x10000, 2);
    pool.images[cover].last_use_frame = 0;
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::None);
    pool.images[cover].last_use_frame = 6;
    CHECK(ClassifySuperseded(pool.images, pool.images[old]) == Supersession::Rewritten);
}
