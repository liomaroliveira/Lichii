// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

using SparseBuffer = vk::Handle<VkBuffer, VkDevice, vk::DeviceDispatch>;

class Device;
class Scheduler;

struct MultiRangeSource {
    VkBuffer handle{};
    VkDeviceMemory memory{};
    VkDeviceSize memory_offset{};
    VkDeviceSize offset{};
    VkDeviceSize size{};
    u64 write_tick{};
    u32 memory_type{};
};

struct MultiRangeRef {
    VkBuffer handle{};
    VkDeviceAddress address{};
    VkDeviceSize size{};
    bool sparse{};
    bool needs_gather{};
};

class MultiRangeBufferCache final {
public:
    static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = 64 * 1024;
    static constexpr size_t MAX_RETIRED = 256;

    explicit MultiRangeBufferCache(const Device& device);

    LICHII_NON_COPYABLE(MultiRangeBufferCache);

    [[nodiscard]] MultiRangeRef Get(const Device& device, Scheduler& scheduler,
                                    MemoryAllocator& memory_allocator, u64 key,
                                    std::span<const MultiRangeSource> sources,
                                    VkDeviceSize total);

    void MarkGathered(u64 key);

    void Invalidate(u64 key);

    void DropOwner(Scheduler& scheduler, VkBuffer owner);

    void TickFrame(Scheduler& scheduler);

    [[nodiscard]] static bool TraceEnabled();

    // EDEN_NX2_VA_TRACE log for a deleted cache buffer; no-op when the trace is off.
    static void TraceBufferDeleted(VkBuffer handle, VkDeviceSize size, u64 frame,
                                   bool multi_range_owner);

    // EDEN_NX2_VA_TRACE logs for a deleted multi-range source kept until the GPU is past `tick`,
    // and for its final release; no-ops when the trace is off.
    static void TraceBufferRetired(VkBuffer handle, u64 tick, u64 current_tick);
    static void TraceBufferFreed(VkBuffer handle, u64 tick, u64 current_tick);

    [[nodiscard]] bool IsOwner(VkBuffer owner) const;

    VkDeviceSize block_size{DEFAULT_BLOCK_SIZE};
    bool use_sparse{};

private:
    struct Retired {
        SparseBuffer handle;
        vk::Buffer gathered;
        u64 tick{};
        VkDeviceAddress address{};
        VkBuffer owner{};
        std::vector<VkBuffer> owners;
    };

    struct Entry {
        vk::Buffer gathered;
        SparseBuffer sparse_handle;
        std::vector<VkBuffer> owners;
        VkDeviceAddress address{};
        VkDeviceSize size{};
        u64 geometry{};
        u64 content{};
        bool dirty{true};
    };

    [[nodiscard]] u64 HashSources(std::span<const MultiRangeSource> sources) const;

    [[nodiscard]] u64 HashContent(std::span<const MultiRangeSource> sources) const;

    [[nodiscard]] bool CanBindSparse(std::span<const MultiRangeSource> sources) const;

    [[nodiscard]] SparseBuffer CreateSparse(const Device& device, Scheduler& scheduler,
                                            std::span<const MultiRangeSource> sources,
                                            VkDeviceSize total);

    [[nodiscard]] VkDeviceSize QueryBlockSize(const Device& device, u32& memory_type_bits) const;

    void RetireEntry(Scheduler& scheduler, Entry& entry, VkBuffer owner = VK_NULL_HANDLE);

    void DrainRetired(Scheduler& scheduler);

    ::Common::unordered_map<u64, Entry> entries;
    boost::container::small_vector<Retired, MAX_RETIRED> retired;
    u32 sparse_memory_type_bits{};
    VkBufferUsageFlags sparse_usage{};
};

} // namespace Vulkan
