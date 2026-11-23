// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_multi_range_buffer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

// EDEN_NX2_MULTI_RANGE_SPARSE=0 forces the gathered-copy path; anything else keeps sparse aliasing.
bool SparseAllowedByEnv() {
    static const bool allowed = [] {
        const char* value = std::getenv("EDEN_NX2_MULTI_RANGE_SPARSE");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return allowed;
}

// EDEN_NX2_VA_TRACE=1: bounded "VA trace:" logs of multi-range buffer lifetimes.
bool VaTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_VA_TRACE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

constexpr u64 TRACE_VA_BEGIN = 0xffff800100000000ULL;
constexpr u64 TRACE_VA_END = 0xffff800100400000ULL;

enum class TraceKind : size_t {
    Create,
    Retire,
    Free,
    Drop,
    Delete,
    BufferRetire,
    BufferFree,
    Count,
};

// Events inside the watched VA window always log; the rest log 1 line per 64 events. A summary of
// event counts is logged once per second.
template <typename... Args>
void VaTrace(TraceKind kind, u64 address, fmt::format_string<Args...> format, Args&&... args) {
    static std::mutex mutex;
    static std::array<u64, size_t(TraceKind::Count)> counts{};
    static std::array<u64, size_t(TraceKind::Count)> last_counts{};
    static u64 total = 0;
    static auto last_summary = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++counts[size_t(kind)];
    const bool in_window = address >= TRACE_VA_BEGIN && address < TRACE_VA_END;
    if (in_window || (total++ % 64) == 0) {
        LOG_INFO(Render_Vulkan, "VA trace: {}", fmt::format(format, std::forward<Args>(args)...));
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_summary >= std::chrono::seconds(1)) {
        last_summary = now;
        LOG_INFO(Render_Vulkan,
                 "VA trace: summary/s creates={} retires={} frees={} drops={} deletes={} "
                 "buffer_retires={} buffer_frees={}",
                 counts[0] - last_counts[0], counts[1] - last_counts[1],
                 counts[2] - last_counts[2], counts[3] - last_counts[3],
                 counts[4] - last_counts[4], counts[5] - last_counts[5],
                 counts[6] - last_counts[6]);
        last_counts = counts;
    }
}

} // namespace

bool MultiRangeBufferCache::TraceEnabled() {
    return VaTraceEnabled();
}

void MultiRangeBufferCache::TraceBufferDeleted(VkBuffer handle, VkDeviceSize size, u64 frame,
                                               bool multi_range_owner) {
    if (!VaTraceEnabled()) {
        return;
    }
    VaTrace(TraceKind::Delete, 0, "delete buffer={} size={} frame={} multi_range_owner={}",
            fmt::ptr(handle), u64(size), frame, multi_range_owner);
}

void MultiRangeBufferCache::TraceBufferRetired(VkBuffer handle, u64 tick, u64 current_tick) {
    if (!VaTraceEnabled()) {
        return;
    }
    VaTrace(TraceKind::BufferRetire, 0, "retire source buffer={} until_tick={} current_tick={}",
            fmt::ptr(handle), tick, current_tick);
}

void MultiRangeBufferCache::TraceBufferFreed(VkBuffer handle, u64 tick, u64 current_tick) {
    if (!VaTraceEnabled()) {
        return;
    }
    VaTrace(TraceKind::BufferFree, 0, "free source buffer={} retire_tick={} current_tick={}",
            fmt::ptr(handle), tick, current_tick);
}

MultiRangeBufferCache::MultiRangeBufferCache(const Device& device) {
    sparse_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (device.IsBufferDeviceAddressSupported()) {
        sparse_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    const bool env_allows = SparseAllowedByEnv();
    const bool supported = device.IsSparseBindingSupported();
    if (!env_allows || !supported) {
        LOG_INFO(Render_Vulkan, "MultiRange: sparse aliasing {}",
                 !supported ? "unsupported" : "disabled by EDEN_NX2_MULTI_RANGE_SPARSE=0");
        return;
    }
    u32 memory_type_bits = 0;
    const VkDeviceSize queried = QueryBlockSize(device, memory_type_bits);
    if (queried == 0 || memory_type_bits == 0) {
        return;
    }
    block_size = queried;
    sparse_memory_type_bits = memory_type_bits;
    use_sparse = true;
    LOG_INFO(Render_Vulkan,
             "MultiRange: sparse aliasing enabled (default; EDEN_NX2_MULTI_RANGE_SPARSE=0 disables)");
}

VkDeviceSize MultiRangeBufferCache::QueryBlockSize(const Device& device,
                                                   u32& memory_type_bits) const {
    const VkDevice logical = *device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    const VkBufferCreateInfo probe_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
        .size = DEFAULT_BLOCK_SIZE,
        .usage = sparse_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer probe{};
    if (dld.vkCreateBuffer(logical, &probe_ci, nullptr, &probe) != VK_SUCCESS) {
        return 0;
    }
    const SparseBuffer owned{probe, logical, dld};
    const VkBufferMemoryRequirementsInfo2 reqs_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = nullptr,
        .buffer = probe,
    };
    VkMemoryRequirements2 reqs2{
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = nullptr,
        .memoryRequirements = {},
    };
    dld.vkGetBufferMemoryRequirements2(logical, &reqs_info, &reqs2);
    memory_type_bits = reqs2.memoryRequirements.memoryTypeBits;
    return reqs2.memoryRequirements.alignment;
}

u64 MultiRangeBufferCache::HashSources(std::span<const MultiRangeSource> sources) const {
    u64 hash = 0xcbf29ce484222325ULL;
    const auto mix = [&hash](u64 value) {
        hash ^= value;
        hash *= 0x100000001b3ULL;
    };
    for (const MultiRangeSource& source : sources) {
        mix(u64(source.handle));
        // A VkBuffer handle value reused with other memory must not match a stale entry.
        mix(u64(source.memory));
        mix(u64(source.offset));
        mix(u64(source.size));
    }
    return hash;
}

u64 MultiRangeBufferCache::HashContent(std::span<const MultiRangeSource> sources) const {
    u64 hash = 0xcbf29ce484222325ULL;
    for (const MultiRangeSource& source : sources) {
        hash ^= source.write_tick;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

bool MultiRangeBufferCache::CanBindSparse(std::span<const MultiRangeSource> sources) const {
    return use_sparse &&
           std::none_of(sources.begin(), sources.end(),
                        [block = block_size, bits = sparse_memory_type_bits](auto const& e) {
                            const VkDeviceSize memory_offset = e.memory_offset + e.offset;
                            return e.memory == VK_NULL_HANDLE || e.memory_type >= 32 ||
                                   ((bits >> e.memory_type) & 1) == 0 ||
                                   (memory_offset % block) != 0 || (e.size % block) != 0;
                        });
}

SparseBuffer MultiRangeBufferCache::CreateSparse(const Device& device, Scheduler& scheduler,
                                                 std::span<const MultiRangeSource> sources,
                                                 VkDeviceSize total) {
    const VkDevice logical = *device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
        .size = total,
        .usage = sparse_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer raw{};
    if (dld.vkCreateBuffer(logical, &buffer_ci, nullptr, &raw) != VK_SUCCESS) {
        return SparseBuffer{};
    }
    SparseBuffer handle{raw, logical, dld};
    std::vector<VkSparseMemoryBind> binds;
    binds.reserve(sources.size());
    VkDeviceSize resource_offset = 0;
    for (const MultiRangeSource& source : sources) {
        binds.push_back(VkSparseMemoryBind{
            .resourceOffset = resource_offset,
            .size = source.size,
            .memory = source.memory,
            .memoryOffset = source.memory_offset + source.offset,
            .flags = 0,
        });
        resource_offset += source.size;
    }
    const VkSparseBufferMemoryBindInfo buffer_bind{
        .buffer = raw,
        .bindCount = static_cast<u32>(binds.size()),
        .pBinds = binds.data(),
    };
    const VkBindSparseInfo bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .bufferBindCount = 1,
        .pBufferBinds = &buffer_bind,
        .imageOpaqueBindCount = 0,
        .pImageOpaqueBinds = nullptr,
        .imageBindCount = 0,
        .pImageBinds = nullptr,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    const VkFenceCreateInfo fence_ci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    vk::Fence fence = device.GetLogical().CreateFence(fence_ci);
    VkResult bind_result = VK_ERROR_UNKNOWN;
    {
        std::scoped_lock lock{scheduler.submit_mutex};
        bind_result = device.GetGraphicsQueue().BindSparse(bind_info, *fence);
    }
    if (bind_result != VK_SUCCESS) {
        return SparseBuffer{};
    }
    fence.Wait();
    if (VaTraceEnabled()) {
        u64 address = 0;
        bool have_address = false;
        if ((sparse_usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
            address = device.GetLogical().GetBufferDeviceAddress(raw);
            have_address = true;
        }
        const MultiRangeSource& first = sources.front();
        std::string address_text = "n/a";
        if (have_address) {
            address_text = fmt::format("{:#x}", address);
        }
        VaTrace(TraceKind::Create, address,
                "create va={} total={} sources={} first_buffer={} first_memory={} "
                "first_memory_offset={:#x} tick={}",
                address_text, u64(total), sources.size(), fmt::ptr(first.handle),
                fmt::ptr(first.memory), u64(first.memory_offset + first.offset),
                scheduler.CurrentTick());
    }
    return handle;
}

void MultiRangeBufferCache::RetireEntry(Scheduler& scheduler, Entry& entry, VkBuffer owner) {
    if (!entry.sparse_handle && !entry.gathered) {
        return;
    }
    if (VaTraceEnabled()) {
        if (owner == VK_NULL_HANDLE && !entry.owners.empty()) {
            owner = entry.owners.front();
        }
        VaTrace(TraceKind::Retire, entry.address,
                "retire va={:#x} retire_tick={} current_tick={} owner={}", u64(entry.address),
                scheduler.CurrentTick(), scheduler.CurrentTick(), fmt::ptr(owner));
    }
    // Grows past MAX_RETIRED instead of waiting for the oldest: this runs while a draw binds its
    // buffers, and a wait on the current tick submits it, leaving the draw's earlier bindings in
    // the submitted command buffer and its stamped buffers free before the draw reads them.
    if (retired.size() >= MAX_RETIRED) {
        DrainRetired(scheduler);
    }
    retired.push_back(Retired{
        .handle = std::move(entry.sparse_handle),
        .gathered = std::move(entry.gathered),
        .tick = scheduler.CurrentTick(),
        .address = entry.address,
        .owner = owner,
        .owners = std::move(entry.owners),
    });
}

void MultiRangeBufferCache::DrainRetired(Scheduler& scheduler) {
    size_t index = 0;
    while (index < retired.size()) {
        if (scheduler.IsFree(retired[index].tick)) {
            if (VaTraceEnabled()) {
                const Retired& freed = retired[index];
                VaTrace(TraceKind::Free, freed.address,
                        "free va={:#x} retire_tick={} current_tick={} owner={}",
                        u64(freed.address), freed.tick, scheduler.CurrentTick(),
                        fmt::ptr(freed.owner));
            }
            if (index + 1 != retired.size()) {
                retired[index] = std::move(retired.back());
            }
            retired.pop_back();
        } else {
            ++index;
        }
    }
}

void MultiRangeBufferCache::TickFrame(Scheduler& scheduler) {
    DrainRetired(scheduler);
}

MultiRangeRef MultiRangeBufferCache::Get(const Device& device, Scheduler& scheduler,
                                         MemoryAllocator& memory_allocator, u64 key,
                                         std::span<const MultiRangeSource> sources,
                                         VkDeviceSize total) {
    if (sources.empty() || total == 0) {
        return MultiRangeRef{};
    }
    if (!retired.empty()) {
        DrainRetired(scheduler);
    }
    const u64 geometry = HashSources(sources);
    const u64 content = HashContent(sources);
    const auto it = entries.find(key);
    if (it != entries.end() && it->second.geometry == geometry && it->second.size == total) {
        Entry& entry = it->second;
        if (entry.content != content) {
            entry.content = content;
            entry.dirty = true;
        }
        MultiRangeRef ref{
            .handle = *entry.sparse_handle,
            .address = entry.address,
            .size = entry.size,
            .sparse = true,
            .needs_gather = false,
        };
        if (!entry.sparse_handle) {
            ref.handle = *entry.gathered;
            ref.sparse = false;
            ref.needs_gather = entry.dirty;
        }
        return ref;
    }
    if (it != entries.end()) {
        RetireEntry(scheduler, it->second);
        entries.erase(it);
    }

    Entry entry{};
    entry.geometry = geometry;
    entry.content = content;
    entry.size = total;
    entry.owners.reserve(sources.size());
    for (const MultiRangeSource& source : sources) {
        entry.owners.push_back(source.handle);
    }
    if (CanBindSparse(sources)) {
        entry.sparse_handle = CreateSparse(device, scheduler, sources, total);
    }
    if (!entry.sparse_handle) {
        VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (device.IsBufferDeviceAddressSupported()) {
            flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }
        const VkBufferCreateInfo gather_ci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = total,
            .usage = flags,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        };
        entry.gathered = memory_allocator.CreateBuffer(gather_ci, MemoryUsage::DeviceLocal);
        entry.dirty = true;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        VkBuffer address_handle = *entry.sparse_handle;
        if (!entry.sparse_handle) {
            address_handle = *entry.gathered;
        }
        entry.address = device.GetLogical().GetBufferDeviceAddress(address_handle);
    }

    MultiRangeRef ref{
        .handle = *entry.sparse_handle,
        .address = entry.address,
        .size = entry.size,
        .sparse = true,
        .needs_gather = false,
    };
    if (!entry.sparse_handle) {
        ref.handle = *entry.gathered;
        ref.sparse = false;
        ref.needs_gather = true;
    }
    entries.emplace(key, std::move(entry));
    return ref;
}

void MultiRangeBufferCache::MarkGathered(u64 key) {
    if (auto const it = entries.find(key); it != entries.end()) {
        it->second.dirty = false;
    }
}

void MultiRangeBufferCache::DropOwner(Scheduler& scheduler, VkBuffer owner) {
    if (owner == VK_NULL_HANDLE) {
        return;
    }
    for (auto it = entries.begin(); it != entries.end();) {
        Entry& entry = it->second;
        bool owned = false;
        for (const VkBuffer handle : entry.owners) {
            if (handle == owner) {
                owned = true;
                break;
            }
        }
        if (!owned) {
            ++it;
            continue;
        }
        if (VaTraceEnabled()) {
            VaTrace(TraceKind::Drop, entry.address,
                    "drop va={:#x} retire_tick={} current_tick={} owner={}", u64(entry.address),
                    scheduler.CurrentTick(), scheduler.CurrentTick(), fmt::ptr(owner));
        }
        RetireEntry(scheduler, entry, owner);
        it = entries.erase(it);
    }
}

bool MultiRangeBufferCache::IsOwner(VkBuffer owner) const {
    for (const auto& [key, entry] : entries) {
        if (std::ranges::find(entry.owners, owner) != entry.owners.end()) {
            return true;
        }
    }
    // A retired sparse entry still aliases its sources' memory until the GPU is past it.
    for (const Retired& pending : retired) {
        if (pending.handle && std::ranges::find(pending.owners, owner) != pending.owners.end()) {
            return true;
        }
    }
    return false;
}

void MultiRangeBufferCache::Invalidate(u64 key) {
    if (auto const it = entries.find(key); it != entries.end()) {
        it->second.dirty = true;
    }
}

} // namespace Vulkan
