// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <exception>
#include <numeric>
#include <utility>

#include "common/logging.h"
#include "video_core/buffer_cache/word_manager.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_stats.h"
#include "video_core/nx2_back_split.h"
#include "video_core/nx2_back_split_rasterizer.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_list_stats.h"
#include "video_core/nx2_read_trace.h"

namespace VideoCommon::BackSplit {

void OnBThreadStart() {
    VideoCommon::GpuSplit::MarkBackThread();
}

namespace {

enum class Call : u8 {
    Draw, DrawIndirect, DrawTexture, Clear, DispatchCompute, ResetCounter, Query,
    BindUniform, DisableUniform, SignalFence, SyncOperation, SignalSyncPoint, SignalReference,
    ReleaseFences, FlushQueryResults, FlushAll, FlushRegion, FlushRegionForRead, FinishRead,
    TraceRead, NoteReportPending, FinishReport, ReadGpuWrittenList, IsGpuTickDone,
    NoteListSynced, AcquireListSlot, ReleaseListSlot, SnapshotListWords, NoteOwedListWord,
    NoteInlineQmdGrid, ReadListSlots, CheckListSlot, FlushListPages, MustFlushRegion,
    GetFlushArea, HasPendingQueries, HoldsQueries, InvalidateRegion, PrepareRegionWrite,
    InnerInvalidation, OnCacheInvalidation, OnCacheInvalidations, TryCacheInvalidation,
    OnCPUWrite, InvalidateGPUCache, UnmapMemory, ModifyGPUMemory, FlushAndInvalidateRegion,
    WaitForIdle, FragmentBarrier, TiledCacheBarrier, FlushCommands, TickFrame,
    ConditionalRendering, SurfaceCopy, TemporalUpscale, ProbeDlss, InlineToMemory,
    InitializeChannel, BindChannel, ReleaseChannel, RegisterTfb, HasDrawTfb, DmaBufferCopy,
    DmaBufferClear, DmaImageToBuffer, DmaBufferToImage, Hook, Count,
};

constexpr std::array<const char*, static_cast<size_t>(Call::Count)> call_names{
    "Draw", "DrawIndirect", "DrawTexture", "Clear", "DispatchCompute", "ResetCounter", "Query",
    "BindUniform", "DisableUniform", "SignalFence", "SyncOperation", "SignalSyncPoint",
    "SignalReference", "ReleaseFences", "FlushQueryResults", "FlushAll", "FlushRegion",
    "FlushRegionForRead", "FinishRead", "TraceRead", "NoteReportPending", "FinishReport",
    "ReadGpuWrittenList", "IsGpuTickDone", "NoteListSynced", "AcquireListSlot",
    "ReleaseListSlot", "SnapshotListWords", "NoteOwedListWord", "NoteInlineQmdGrid",
    "ReadListSlots", "CheckListSlot", "FlushListPages", "MustFlushRegion", "GetFlushArea",
    "HasPendingQueries", "HoldsQueries", "InvalidateRegion", "PrepareRegionWrite",
    "InnerInvalidation", "OnCacheInvalidation", "OnCacheInvalidations", "TryCacheInvalidation",
    "OnCPUWrite", "InvalidateGPUCache", "UnmapMemory", "ModifyGPUMemory",
    "FlushAndInvalidateRegion", "WaitForIdle", "FragmentBarrier", "TiledCacheBarrier",
    "FlushCommands", "TickFrame", "ConditionalRendering", "SurfaceCopy", "TemporalUpscale",
    "ProbeDlss", "InlineToMemory", "InitializeChannel", "BindChannel", "ReleaseChannel",
    "RegisterTfb", "HasDrawTfb", "DmaBufferCopy", "DmaBufferClear", "DmaImageToBuffer",
    "DmaBufferToImage", "Hook",
};

/// Stage A only (the counts are not atomic).
std::array<u64, static_cast<size_t>(Call::Count)> g_counts{};

/// The thread-local state a rasterizer call reads or writes for its calling thread.
struct Context {
    bool fd_reads_serials;
    bool fd_serial_bumped;
    bool fd_memory_serial_bumped;
    bool signaling_query_report_fence;
    Tegra::Nx2ListStats::WriterTag writer;
    Tegra::Nx2ListStats::ServedDetail served;
    bool reference;
    GpuWaitStats::HostWaits host_waits;
    u64 download_bytes;
    ReadTrace::Read* read;
    DeferredPageCaching* deferred_page_caching;

    static Context Capture() {
        return Context{
            .fd_reads_serials = Nx2FastDraw::t_reads_serials,
            .fd_serial_bumped = Nx2FastDraw::t_serial_bumped,
            .fd_memory_serial_bumped = Nx2FastDraw::t_memory_serial_bumped,
            .signaling_query_report_fence = VideoCore::g_signaling_query_report_fence,
            .writer = Tegra::Nx2ListStats::t_writer,
            .served = Tegra::Nx2ListStats::t_served,
            .reference = Tegra::Nx2ListStats::t_reference,
            .host_waits = GpuWaitStats::t_host_waits,
            .download_bytes = GpuWaitStats::t_download_bytes,
            .read = ReadTrace::t_read,
            .deferred_page_caching = t_deferred_page_caching,
        };
    }

    void Apply() const {
        Nx2FastDraw::t_reads_serials = fd_reads_serials;
        Nx2FastDraw::t_serial_bumped = fd_serial_bumped;
        Nx2FastDraw::t_memory_serial_bumped = fd_memory_serial_bumped;
        VideoCore::g_signaling_query_report_fence = signaling_query_report_fence;
        Tegra::Nx2ListStats::t_writer = writer;
        Tegra::Nx2ListStats::t_served = served;
        Tegra::Nx2ListStats::t_reference = reference;
        GpuWaitStats::t_host_waits = host_waits;
        GpuWaitStats::t_download_bytes = download_bytes;
        ReadTrace::t_read = read;
        t_deferred_page_caching = deferred_page_caching;
    }
};

/// Stage A: the split's back thread, with the back split on (GPUBackB is a back thread too).
bool OnStageA() {
    return Enabled() && VideoCommon::GpuSplit::OnBackThread() && !OnB();
}

template <typename F>
struct Task {
    F* func;
    Context context;
    /// What the call threw on GPUBackB (e.g. FireEmblemUpscale::DispatchReplaced, which the
    /// shader cache throws for KeplerCompute to catch), rethrown on stage A.
    std::exception_ptr exception;

    static void Run(void* argument) {
        auto& task = *static_cast<Task*>(argument);
        task.context.Apply();
        try {
            (*task.func)();
        } catch (...) {
            task.exception = std::current_exception();
        }
        task.context = Context::Capture();
    }
};

template <typename F>
void RunWithContext(Call call, F&& func) {
    ++g_counts[static_cast<size_t>(call)];
    using Func = std::remove_reference_t<F>;
    Task<Func> task{&func, Context::Capture(), nullptr};
    RunOnB(&Task<Func>::Run, &task);
    task.context.Apply();
    if (task.exception) [[unlikely]] {
        std::rethrow_exception(task.exception);
    }
}

/// Runs `func` on GPUBackB from stage A, on the calling thread otherwise.
template <typename F>
void Route(Call call, F&& func) {
    if (!OnStageA()) {
        func();
        return;
    }
    RunWithContext(call, std::forward<F>(func));
}

void ReportCounts(double seconds) {
    std::array<size_t, static_cast<size_t>(Call::Count)> order{};
    std::iota(order.begin(), order.end(), size_t{0});
    std::partial_sort(order.begin(), order.begin() + 8, order.end(),
                      [](size_t a, size_t b) { return g_counts[a] > g_counts[b]; });
    std::string line;
    for (size_t i = 0; i < 8 && g_counts[order[i]] != 0; ++i) {
        line += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ", call_names[order[i]],
                            static_cast<double>(g_counts[order[i]]) / seconds);
    }
    LOG_INFO(HW_GPU, "Nx2BackSplit RunOnB by call: {}", line);
    g_counts.fill(0);
}

} // Anonymous namespace

void RunHook(void (*fn)(void*), void* arg) {
    if (!OnStageA()) {
        fn(arg);
        return;
    }
    RunWithContext(Call::Hook, [fn, arg] { fn(arg); });
}

RasterizerProxy::RasterizerProxy(VideoCore::RasterizerInterface& inner_) : inner{inner_} {
    g_state.report_hook = &ReportCounts;
    LOG_INFO(HW_GPU, "Nx2BackSplit: E04.T6 locked mode, GPUBackB owns the rasterizer");
}

RasterizerProxy::~RasterizerProxy() {
    g_state.report_hook = nullptr;
}

void RasterizerProxy::Draw(bool is_indexed, u32 instance_count) {
    Route(Call::Draw, [&] { inner.Draw(is_indexed, instance_count); });
}
void RasterizerProxy::DrawIndirect() {
    Route(Call::DrawIndirect, [&] { inner.DrawIndirect(); });
}
void RasterizerProxy::DrawTexture() {
    Route(Call::DrawTexture, [&] { inner.DrawTexture(); });
}
void RasterizerProxy::Clear(u32 layer_count) {
    Route(Call::Clear, [&] { inner.Clear(layer_count); });
}
void RasterizerProxy::DispatchCompute() {
    Route(Call::DispatchCompute, [&] { inner.DispatchCompute(); });
}
void RasterizerProxy::ResetCounter(VideoCommon::QueryType type) {
    Route(Call::ResetCounter, [&] { inner.ResetCounter(type); });
}
void RasterizerProxy::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    Route(Call::Query, [&] { inner.Query(gpu_addr, type, flags, payload, subreport); });
}
void RasterizerProxy::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {
    Route(Call::BindUniform, [&] { inner.BindGraphicsUniformBuffer(stage, index, gpu_addr, size); });
}
void RasterizerProxy::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    Route(Call::DisableUniform, [&] { inner.DisableGraphicsUniformBuffer(stage, index); });
}
void RasterizerProxy::SignalFence(std::function<void()>&& func) {
    Route(Call::SignalFence, [&] { inner.SignalFence(std::move(func)); });
}
void RasterizerProxy::SyncOperation(std::function<void()>&& func) {
    Route(Call::SyncOperation, [&] { inner.SyncOperation(std::move(func)); });
}
void RasterizerProxy::SignalSyncPoint(u32 value) {
    Route(Call::SignalSyncPoint, [&] { inner.SignalSyncPoint(value); });
}
void RasterizerProxy::SignalReference() {
    Route(Call::SignalReference, [&] { inner.SignalReference(); });
}
void RasterizerProxy::ReleaseFences(bool force) {
    Route(Call::ReleaseFences, [&] { inner.ReleaseFences(force); });
}
void RasterizerProxy::FlushQueryResults(DAddr addr, u64 size) {
    Route(Call::FlushQueryResults, [&] { inner.FlushQueryResults(addr, size); });
}
void RasterizerProxy::FlushAll() {
    Route(Call::FlushAll, [&] { inner.FlushAll(); });
}
void RasterizerProxy::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    Route(Call::FlushRegion, [&] { inner.FlushRegion(addr, size, which); });
}
void RasterizerProxy::FlushRegionForRead(DAddr addr, u64 size) {
    Route(Call::FlushRegionForRead, [&] { inner.FlushRegionForRead(addr, size); });
}
void RasterizerProxy::FinishRead(DAddr addr, u64 size) {
    Route(Call::FinishRead, [&] { inner.FinishRead(addr, size); });
}
void RasterizerProxy::TraceRead(VideoCommon::ReadTrace::Read& read,
                                VideoCommon::ReadTrace::Step step) {
    Route(Call::TraceRead, [&] { inner.TraceRead(read, step); });
}
bool RasterizerProxy::NoteReportPending(DAddr addr, u64 size) {
    bool result{};
    Route(Call::NoteReportPending, [&] { result = inner.NoteReportPending(addr, size); });
    return result;
}
void RasterizerProxy::FinishReport(DAddr addr, u64 size) {
    Route(Call::FinishReport, [&] { inner.FinishReport(addr, size); });
}
VideoCore::RasterizerInterface::GpuWrittenList RasterizerProxy::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out, bool past_sync, u64* pending_tick) {
    GpuWrittenList result{};
    Route(Call::ReadGpuWrittenList,
          [&] { result = inner.ReadGpuWrittenCommandList(addr, out, past_sync, pending_tick); });
    return result;
}
bool RasterizerProxy::IsGpuTickDone(u64 tick) {
    bool result{};
    Route(Call::IsGpuTickDone, [&] { result = inner.IsGpuTickDone(tick); });
    return result;
}
void RasterizerProxy::NoteGpuWrittenCommandListSynced(DAddr page) {
    Route(Call::NoteListSynced, [&] { inner.NoteGpuWrittenCommandListSynced(page); });
}
bool RasterizerProxy::Nx2ListSlotsSupported() const {
    return inner.Nx2ListSlotsSupported();
}
std::optional<u32> RasterizerProxy::Nx2AcquireListSlot() {
    std::optional<u32> result;
    Route(Call::AcquireListSlot, [&] { result = inner.Nx2AcquireListSlot(); });
    return result;
}
void RasterizerProxy::Nx2ReleaseListSlot(u32 slot) {
    Route(Call::ReleaseListSlot, [&] { inner.Nx2ReleaseListSlot(slot); });
}
void RasterizerProxy::Nx2SnapshotListWords(DAddr device_addr, u32 count, u32 slot) {
    Route(Call::SnapshotListWords, [&] { inner.Nx2SnapshotListWords(device_addr, count, slot); });
}
void RasterizerProxy::Nx2NoteOwedListWord(DAddr device_addr, u32 value) {
    Route(Call::NoteOwedListWord, [&] { inner.Nx2NoteOwedListWord(device_addr, value); });
}
void RasterizerProxy::Nx2NoteInlineQmdGrid(DAddr device_addr, u32 x, u32 y, u32 z) {
    Route(Call::NoteInlineQmdGrid, [&] { inner.Nx2NoteInlineQmdGrid(device_addr, x, y, z); });
}
void RasterizerProxy::Nx2ReadListSlots(std::span<const u32> slots, std::span<u32> words) {
    Route(Call::ReadListSlots, [&] { inner.Nx2ReadListSlots(slots, words); });
}
void RasterizerProxy::Nx2CheckListSlot(u32 slot, std::span<const u32> words) {
    Route(Call::CheckListSlot, [&] { inner.Nx2CheckListSlot(slot, words); });
}
void RasterizerProxy::FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
    Route(Call::FlushListPages, [&] { inner.FlushGpuWrittenCommandListPages(pages); });
}
bool RasterizerProxy::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    bool result{};
    Route(Call::MustFlushRegion, [&] { result = inner.MustFlushRegion(addr, size, which); });
    return result;
}
VideoCore::RasterizerDownloadArea RasterizerProxy::GetFlushArea(DAddr addr, u64 size) {
    VideoCore::RasterizerDownloadArea result{};
    Route(Call::GetFlushArea, [&] { result = inner.GetFlushArea(addr, size); });
    return result;
}
bool RasterizerProxy::HasPendingQueries(DAddr addr, u64 size) {
    bool result{};
    Route(Call::HasPendingQueries, [&] { result = inner.HasPendingQueries(addr, size); });
    return result;
}
bool RasterizerProxy::HoldsQueries(DAddr addr, u64 size) {
    bool result{};
    Route(Call::HoldsQueries, [&] { result = inner.HoldsQueries(addr, size); });
    return result;
}
void RasterizerProxy::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    Route(Call::InvalidateRegion, [&] { inner.InvalidateRegion(addr, size, which); });
}
void RasterizerProxy::PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) {
    Route(Call::PrepareRegionWrite, [&] { inner.PrepareRegionWrite(addr, size, which); });
}
void RasterizerProxy::InnerInvalidation(
    std::span<const std::pair<DAddr, std::size_t>> sequences) {
    Route(Call::InnerInvalidation, [&] { inner.InnerInvalidation(sequences); });
}
void RasterizerProxy::OnCacheInvalidation(PAddr addr, u64 size) {
    Route(Call::OnCacheInvalidation, [&] { inner.OnCacheInvalidation(addr, size); });
}
void RasterizerProxy::OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
    Route(Call::OnCacheInvalidations, [&] { inner.OnCacheInvalidations(ranges); });
}
bool RasterizerProxy::TryCacheInvalidation(DAddr addr, u64 size) {
    bool result{};
    Route(Call::TryCacheInvalidation, [&] { result = inner.TryCacheInvalidation(addr, size); });
    return result;
}
bool RasterizerProxy::OnCPUWrite(PAddr addr, u64 size) {
    bool result{};
    Route(Call::OnCPUWrite, [&] { result = inner.OnCPUWrite(addr, size); });
    return result;
}
void RasterizerProxy::InvalidateGPUCache() {
    Route(Call::InvalidateGPUCache, [&] { inner.InvalidateGPUCache(); });
}
void RasterizerProxy::UnmapMemory(DAddr addr, u64 size) {
    Route(Call::UnmapMemory, [&] { inner.UnmapMemory(addr, size); });
}
void RasterizerProxy::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    Route(Call::ModifyGPUMemory, [&] { inner.ModifyGPUMemory(as_id, addr, size); });
}
void RasterizerProxy::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                               VideoCommon::CacheType which) {
    Route(Call::FlushAndInvalidateRegion,
          [&] { inner.FlushAndInvalidateRegion(addr, size, which); });
}
void RasterizerProxy::WaitForIdle() {
    Route(Call::WaitForIdle, [&] { inner.WaitForIdle(); });
}
void RasterizerProxy::FragmentBarrier() {
    Route(Call::FragmentBarrier, [&] { inner.FragmentBarrier(); });
}
void RasterizerProxy::TiledCacheBarrier() {
    Route(Call::TiledCacheBarrier, [&] { inner.TiledCacheBarrier(); });
}
void RasterizerProxy::FlushCommands() {
    Route(Call::FlushCommands, [&] { inner.FlushCommands(); });
}
void RasterizerProxy::TickFrame() {
    Route(Call::TickFrame, [&] { inner.TickFrame(); });
}
bool RasterizerProxy::AccelerateConditionalRendering() {
    bool result{};
    Route(Call::ConditionalRendering, [&] { result = inner.AccelerateConditionalRendering(); });
    return result;
}
bool RasterizerProxy::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    bool result{};
    Route(Call::SurfaceCopy, [&] { result = inner.AccelerateSurfaceCopy(src, dst, copy_config); });
    return result;
}
bool RasterizerProxy::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history,
    std::optional<std::array<u32, 2>> render_region) {
    bool result{};
    Route(Call::TemporalUpscale, [&] {
        result = inner.AccelerateTemporalUpscale(src, dst, motion_vectors, depth, jitter,
                                                 motion_scale, use_history, render_region);
    });
    return result;
}
bool RasterizerProxy::SupportsGameDlssNetwork() const {
    return inner.SupportsGameDlssNetwork();
}
bool RasterizerProxy::UsesNvidiaDlss() const {
    return inner.UsesNvidiaDlss();
}
bool RasterizerProxy::CanProbeGameDlssOutput() const {
    return inner.CanProbeGameDlssOutput();
}
std::optional<VideoCore::GameDlssProbe> RasterizerProxy::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst) {
    std::optional<VideoCore::GameDlssProbe> result;
    Route(Call::ProbeDlss, [&] { result = inner.ProbeGameDlssOutput(src, dst); });
    return result;
}
VideoCommon::OotNetwork::BypassState* RasterizerProxy::OotNetworkBypass() {
    return inner.OotNetworkBypass();
}
Tegra::Engines::AccelerateDMAInterface& RasterizerProxy::AccessAccelerateDMA() {
    return *this;
}
void RasterizerProxy::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {
    Route(Call::InlineToMemory,
          [&] { inner.AccelerateInlineToMemory(address, copy_size, memory); });
}
void RasterizerProxy::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                        const VideoCore::DiskResourceLoadCallback& callback) {
    inner.LoadDiskResources(title_id, stop_loading, callback);
}
void RasterizerProxy::InitializeChannel(Tegra::Control::ChannelState& channel) {
    Route(Call::InitializeChannel, [&] { inner.InitializeChannel(channel); });
}
void RasterizerProxy::BindChannel(Tegra::Control::ChannelState& channel) {
    Route(Call::BindChannel, [&] { inner.BindChannel(channel); });
}
void RasterizerProxy::ReleaseChannel(s32 channel_id) {
    Route(Call::ReleaseChannel, [&] { inner.ReleaseChannel(channel_id); });
}
void RasterizerProxy::RegisterTransformFeedback(GPUVAddr tfb_object_addr) {
    Route(Call::RegisterTfb, [&] { inner.RegisterTransformFeedback(tfb_object_addr); });
}
bool RasterizerProxy::HasDrawTransformFeedback() {
    bool result{};
    Route(Call::HasDrawTfb, [&] { result = inner.HasDrawTransformFeedback(); });
    return result;
}

bool RasterizerProxy::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    bool result{};
    Route(Call::DmaBufferCopy, [&] {
        result = inner.AccessAccelerateDMA().BufferCopy(src_address, dest_address, amount);
    });
    return result;
}
bool RasterizerProxy::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    bool result{};
    Route(Call::DmaBufferClear,
          [&] { result = inner.AccessAccelerateDMA().BufferClear(src_address, amount, value); });
    return result;
}
bool RasterizerProxy::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                    const Tegra::DMA::ImageOperand& src,
                                    const Tegra::DMA::BufferOperand& dst) {
    bool result{};
    Route(Call::DmaImageToBuffer,
          [&] { result = inner.AccessAccelerateDMA().ImageToBuffer(copy_info, src, dst); });
    return result;
}
bool RasterizerProxy::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                    const Tegra::DMA::BufferOperand& src,
                                    const Tegra::DMA::ImageOperand& dst) {
    bool result{};
    Route(Call::DmaBufferToImage,
          [&] { result = inner.AccessAccelerateDMA().BufferToImage(copy_info, src, dst); });
    return result;
}

} // namespace VideoCommon::BackSplit
