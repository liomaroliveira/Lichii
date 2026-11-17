// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include "video_core/engines/maxwell_dma.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon::BackSplit {

/// E04.T6 step 1 (EDEN_NX2_BACK_SPLIT=1): GPUBackB owns the rasterizer. Stage A (the split's back,
/// GPUBack) runs each rasterizer call on GPUBackB and waits for it (locked mode); every other
/// thread calls the rasterizer as before. The thread-local state those calls read or write
/// (Nx2FastDraw serial flags, query fence flag, writer tag, wait stats...) moves to GPUBackB for
/// the call and back to stage A after it.
class RasterizerProxy final : public VideoCore::RasterizerInterface,
                              public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit RasterizerProxy(VideoCore::RasterizerInterface& inner_);
    ~RasterizerProxy() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force = true) override;
    void FlushQueryResults(DAddr addr, u64 size) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void FlushRegionForRead(DAddr addr, u64 size) override;
    void FinishRead(DAddr addr, u64 size) override;
    void TraceRead(VideoCommon::ReadTrace::Read& read, VideoCommon::ReadTrace::Step step) override;
    bool NoteReportPending(DAddr addr, u64 size) override;
    void FinishReport(DAddr addr, u64 size) override;
    GpuWrittenList ReadGpuWrittenCommandList(DAddr addr, std::span<u8> out, bool past_sync,
                                             u64* pending_tick = nullptr) override;
    bool IsGpuTickDone(u64 tick) override;
    void NoteGpuWrittenCommandListSynced(DAddr page) override;
    [[nodiscard]] bool Nx2ListSlotsSupported() const override;
    [[nodiscard]] std::optional<u32> Nx2AcquireListSlot() override;
    void Nx2ReleaseListSlot(u32 slot) override;
    void Nx2SnapshotListWords(DAddr device_addr, u32 count, u32 slot) override;
    void Nx2NoteOwedListWord(DAddr device_addr, u32 value) override;
    void Nx2NoteInlineQmdGrid(DAddr device_addr, u32 x, u32 y, u32 z) override;
    void Nx2ReadListSlots(std::span<const u32> slots, std::span<u32> words) override;
    void Nx2CheckListSlot(u32 slot, std::span<const u32> words) override;
    void FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    [[nodiscard]] bool HasPendingQueries(DAddr addr, u64 size) override;
    [[nodiscard]] bool HoldsQueries(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(PAddr addr, u64 size) override;
    void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) override;
    bool TryCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(PAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateConditionalRendering() override;
    [[nodiscard]] bool AccelerateSurfaceCopy(
        const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
        const Tegra::Engines::Fermi2D::Config& copy_config) override;
    [[nodiscard]] bool AccelerateTemporalUpscale(
        const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
        const Tegra::Engines::Fermi2D::Surface& motion_vectors,
        const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
        std::array<f32, 2> motion_scale, bool use_history = true,
        std::optional<std::array<u32, 2>> render_region = std::nullopt) override;
    [[nodiscard]] bool SupportsGameDlssNetwork() const override;
    [[nodiscard]] bool UsesNvidiaDlss() const override;
    [[nodiscard]] bool CanProbeGameDlssOutput() const override;
    [[nodiscard]] std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Surface& dst) override;
    [[nodiscard]] VideoCommon::OotNetwork::BypassState* OotNetworkBypass() override;
    [[nodiscard]] Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;
    void InitializeChannel(Tegra::Control::ChannelState& channel) override;
    void BindChannel(Tegra::Control::ChannelState& channel) override;
    void ReleaseChannel(s32 channel_id) override;
    void RegisterTransformFeedback(GPUVAddr tfb_object_addr) override;
    bool HasDrawTransformFeedback() override;

    // AccelerateDMAInterface (the inner rasterizer's, run the same way)
    bool BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) override;
    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;
    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;
    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                       const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    VideoCore::RasterizerInterface& inner;
};

/// Runs `fn(arg)` on GPUBackB with stage A's thread-local state when called on stage A with the
/// split on, else on the calling thread (the split's idle and tick hooks).
void RunHook(void (*fn)(void*), void* arg);

} // namespace VideoCommon::BackSplit
