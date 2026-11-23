// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/present/vk_oot_ui.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

struct ImDrawData;
struct ImGuiContext;

namespace Vulkan {

class Device;
class MemoryAllocator;
class Swapchain;

class OotOverlay {
public:
    OotOverlay(const vk::Instance& instance, const Device& device,
               MemoryAllocator& memory_allocator, Swapchain& swapchain,
               std::mutex& queue_submit_mutex);
    ~OotOverlay();

    OotOverlay(const OotOverlay&) = delete;
    OotOverlay& operator=(const OotOverlay&) = delete;

    /// Called on frames where nothing is drawn so previously retired staging buffers / textures
    /// are freed safely once their GPU fences have completed.
    void Idle();

    /// Records the ImGui render pass onto `cmdbuf` while `swapchain_image` is in
    /// `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`, leaving it back in `TRANSFER_DST_OPTIMAL`.
    void Record(const vk::CommandBuffer& cmdbuf, VkImage swapchain_image, std::size_t image_index,
                VkExtent2D extent, VkFormat swapchain_format, bool is_hdr,
                const std::function<void(OotUi::Ctx&)>& draw_fn);

private:
    struct GpuTex {
        vk::Image image;
        vk::ImageView view;
        VkDescriptorSet ds{VK_NULL_HANDLE};
    };

    struct RetiredUpload {
        vk::Buffer buffer;
        u64 serial{};
    };

    struct RetiredTex {
        std::unique_ptr<GpuTex> tex;
        u64 serial{};
    };

    bool EnsureInitialized(VkFormat swapchain_format, VkExtent2D extent);
    void BuildFontAtlas();
    void CleanupRetired(bool force_all, bool lock_held = false);
    void ProcessBackendTextures(ImDrawData* draw_data);
    void UploadTextures(const vk::CommandBuffer& cmdbuf, ImDrawData* draw_data);
    void Shutdown();

    const vk::Instance& instance;
    const Device& device;
    MemoryAllocator& memory_allocator;
    Swapchain& swapchain;
    std::mutex& queue_submit_mutex_; ///< Scheduler::submit_mutex: guards the graphics queue

    ImGuiContext* ctx = nullptr;
    OotUi::Fonts fonts_{};
    bool backend_ready_ = false;
    bool failed_ = false;
    bool loaded_functions_ = false;
    bool fonts_from_romfs_ = false;

    vk::DescriptorPool descriptor_pool_;
    vk::RenderPass render_pass_;
    VkFormat render_pass_format_{VK_FORMAT_UNDEFINED};
    VkExtent2D fb_extent_{};
    std::vector<vk::ImageView> image_views_;
    std::vector<vk::Framebuffer> framebuffers_;
    std::vector<VkImage> fb_images_; ///< Swapchain images the views were made for


    std::vector<std::unique_ptr<GpuTex>> live_textures_;
    std::vector<RetiredUpload> retired_uploads_;
    std::vector<RetiredTex> retired_textures_;

    u64 present_serial_ = 0;
    std::chrono::steady_clock::time_point last_frame_time_{};

    // E47.T12 / E61.T2: the running game's icon (user texture uploaded like the font atlas) and labels
    void SyncGameInfo();
    void UploadGameIcon(const vk::CommandBuffer& cmdbuf);
    std::unique_ptr<ImTextureData> game_icon_;
    bool icon_failed_ = false;
    u64 game_info_serial_ = 0;
    u64 backend_tex_events_ = 0;
    std::string game_title_;
    std::string game_id_;
};

} // namespace Vulkan
