// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace {

using Tegra::Engines::Maxwell3D;

class NoDma final : public Tegra::Engines::AccelerateDMAInterface {
public:
    bool BufferCopy(GPUVAddr, GPUVAddr, u64) override { return true; }
    bool BufferClear(GPUVAddr, u64, u32) override { return true; }
    bool ImageToBuffer(const Tegra::DMA::ImageCopy&, const Tegra::DMA::ImageOperand&,
                       const Tegra::DMA::BufferOperand&) override { return true; }
    bool BufferToImage(const Tegra::DMA::ImageCopy&, const Tegra::DMA::BufferOperand&,
                       const Tegra::DMA::ImageOperand&) override { return true; }
};

/// Records what each draw is asked to draw.
class DrawRecorder final : public VideoCore::RasterizerInterface {
public:
    std::vector<std::pair<bool, u32>> draws; ///< indexed, instances

    void Draw(bool is_indexed, u32 instances) override {
        draws.emplace_back(is_indexed, instances);
    }
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&& func) override { func(); }
    void SyncOperation(std::function<void()>&& func) override { func(); }
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override { return false; }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, false};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(PAddr, u64) override {}
    bool OnCPUWrite(PAddr, u64) override { return false; }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override { return dma; }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}

private:
    NoDma dma;
};

struct LoopFixture {
    LoopFixture() : memory{system, device_memory, 32, 1ULL << 31}, maxwell3d{memory} {
        maxwell3d.BindRasterizer(&rasterizer);
    }

    void Write(u32 method, u32 value) {
        maxwell3d.CallMethod(system, method, value, true);
    }

    /// The first instance of an array draw: draw.begin with instance id First.
    void FirstInstance() {
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.first), 0);
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.count), 3);
        Write(MAXWELL3D_REG_INDEX(draw.begin), Triangles);
        Write(MAXWELL3D_REG_INDEX(draw.end), 0);
    }

    /// A later instance, as the game's macro writes it.
    void LaterInstance() {
        Write(MAXWELL3D_REG_INDEX(draw.begin), Triangles | Subsequent);
        Write(MAXWELL3D_REG_INDEX(draw.end), 0);
    }

    /// A state change: the instances counted so far are drawn.
    void EndLoop() {
        constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
        Write(blend, maxwell3d.regs.reg_array[blend] ^ 1);
    }

    static constexpr u32 Triangles = 0x4;
    static constexpr u32 Subsequent = 1U << 26;

    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory;
    DrawRecorder rasterizer;
    Maxwell3D maxwell3d;
};

} // namespace

TEST_CASE("Folded instances draw what the loop's writes draw", "[instance_fold]") {
    for (const u32 later : {1U, 2U, 5U, 40U}) {
        LoopFixture written;
        written.FirstInstance();
        for (u32 i = 0; i < later; ++i) {
            written.LaterInstance();
        }
        written.EndLoop();

        // The front sends the first later instance (the register changes) and counts the rest.
        LoopFixture folded;
        folded.FirstInstance();
        folded.LaterInstance();
        if (later > 1) {
            folded.maxwell3d.draw_manager.AddLoopInstances(folded.maxwell3d, later - 1);
        }
        folded.EndLoop();

        REQUIRE(written.rasterizer.draws.size() >= 2);
        CHECK(written.rasterizer.draws == folded.rasterizer.draws);
        CHECK(written.rasterizer.draws.back().second >= later);
    }
}

TEST_CASE("Folded instances keep counting across a split loop", "[instance_fold]") {
    LoopFixture written;
    LoopFixture folded;
    for (LoopFixture* f : {&written, &folded}) {
        f->FirstInstance();
        f->LaterInstance();
    }
    written.LaterInstance();
    written.LaterInstance();
    folded.maxwell3d.draw_manager.AddLoopInstances(folded.maxwell3d, 2);
    for (LoopFixture* f : {&written, &folded}) {
        f->EndLoop();
        // A new loop starts clean.
        f->FirstInstance();
        f->LaterInstance();
        f->EndLoop();
    }
    CHECK(written.rasterizer.draws == folded.rasterizer.draws);
    CHECK(written.rasterizer.draws.size() == 4);
}
