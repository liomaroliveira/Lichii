// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_vulkan/present/vk_oot_imgui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

#include <fmt/format.h>

#include "backends/imgui_impl_vulkan.h"
#include "common/logging.h"
#include "common/oot_menu_bridge.h"
#include "common/settings.h"
#include "imgui.h"
#include "imgui_internal.h" // RegisterUserTexture (E47.T12)
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

extern const unsigned longshot_font_body_size;
extern const unsigned longshot_font_strong_size;
extern const unsigned longshot_font_heading_size;
extern const unsigned char longshot_font_body[];
extern const unsigned char longshot_font_strong[];
extern const unsigned char longshot_font_heading[];

namespace Vulkan {
namespace {

struct LoaderCtx {
    VkInstance instance;
    VkDevice device;
    const vk::DeviceDispatch* dld;
};

PFN_vkVoidFunction LoadVulkanFunc(const char* name, void* user_data) {
    const auto* ctx = static_cast<const LoaderCtx*>(user_data);
    if (!ctx || !ctx->dld || !name) {
        return nullptr;
    }
    if (ctx->dld->vkGetDeviceProcAddr && ctx->device) {
        if (PFN_vkVoidFunction fn = ctx->dld->vkGetDeviceProcAddr(ctx->device, name)) {
            return fn;
        }
    }
    if (ctx->dld->vkGetInstanceProcAddr) {
        return ctx->dld->vkGetInstanceProcAddr(ctx->instance, name);
    }
    return nullptr;
}

void CheckVk(VkResult err) {
    if (err != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "OotOverlay Vulkan error: {}", static_cast<int>(err));
    }
}

bool IsSrgbFormat(VkFormat format) noexcept {
    switch (format) {
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return true;
    default:
        return false;
    }
}

u8 SrgbToLinearByte(u8 c) noexcept {
    static const auto kTable = []() {
        std::array<u8, 256> t{};
        for (int i = 0; i < 256; ++i) {
            const double x = static_cast<double>(i) / 255.0;
            const double lin =
                (x <= 0.04045) ? (x / 12.92) : std::pow((x + 0.055) / 1.055, 2.4);
            t[static_cast<std::size_t>(i)] =
                static_cast<u8>(std::clamp(std::lround(lin * 255.0), 0L, 255L));
        }
        return t;
    }();
    return kTable[c];
}

double SrgbByteToLinear(u8 c) noexcept {
    const double x = static_cast<double>(c) / 255.0;
    return (x <= 0.04045) ? (x / 12.92) : std::pow((x + 0.055) / 1.055, 2.4);
}

u8 LinearToPqByte(double lin_nits_norm) noexcept {
    const double y = std::clamp(lin_nits_norm, 0.0, 1.0);
    constexpr double m1 = 2610.0 / 16384.0;
    constexpr double m2 = (2523.0 / 4096.0) * 128.0;
    constexpr double c1 = 3424.0 / 4096.0;
    constexpr double c2 = (2413.0 / 4096.0) * 32.0;
    constexpr double c3 = (2392.0 / 4096.0) * 32.0;
    const double ym1 = std::pow(y, m1);
    const double num = c1 + c2 * ym1;
    const double den = 1.0 + c3 * ym1;
    const double pq = std::pow(num / den, m2);
    return static_cast<u8>(std::clamp(std::lround(pq * 255.0), 0L, 255L));
}

void RemapDrawDataColors(ImDrawData* draw_data, bool is_srgb, bool is_hdr, double white_nits) {
    if (!draw_data || (!is_srgb && !is_hdr)) {
        return;
    }
    const double ref_scale = std::clamp(white_nits, 80.0, 1000.0) / 10000.0;
    for (ImDrawList* dl : draw_data->CmdLists) {
        for (ImDrawVert& v : dl->VtxBuffer) {
            const u32 col = v.col;
            u8 r = static_cast<u8>((col >> IM_COL32_R_SHIFT) & 0xFFu);
            u8 g = static_cast<u8>((col >> IM_COL32_G_SHIFT) & 0xFFu);
            u8 b = static_cast<u8>((col >> IM_COL32_B_SHIFT) & 0xFFu);
            const u8 a = static_cast<u8>((col >> IM_COL32_A_SHIFT) & 0xFFu);
            if (is_hdr) {
                const double lr = SrgbByteToLinear(r);
                const double lg = SrgbByteToLinear(g);
                const double lb = SrgbByteToLinear(b);
                // Convert BT.709 linear -> BT.2020 linear and scale to HDR10 PQ reference white
                const double r2020 = 0.627404 * lr + 0.329283 * lg + 0.043313 * lb;
                const double g2020 = 0.069097 * lr + 0.919540 * lg + 0.011362 * lb;
                const double b2020 = 0.016391 * lr + 0.088013 * lg + 0.895595 * lb;
                r = LinearToPqByte(r2020 * ref_scale);
                g = LinearToPqByte(g2020 * ref_scale);
                b = LinearToPqByte(b2020 * ref_scale);
            } else if (is_srgb) {
                r = SrgbToLinearByte(r);
                g = SrgbToLinearByte(g);
                b = SrgbToLinearByte(b);
            }
            v.col = IM_COL32(r, g, b, a);
        }
    }
}

void TransitionSwapchainImage(const vk::CommandBuffer& cmdbuf, VkImage image, VkImageLayout old_layout,
                              VkImageLayout new_layout, VkAccessFlags src_access,
                              VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                              VkPipelineStageFlags dst_stage) {
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    cmdbuf.PipelineBarrier(src_stage, dst_stage, 0, barrier);
}

} // namespace

OotOverlay::OotOverlay(const vk::Instance& instance_, const Device& device_,
                       MemoryAllocator& memory_allocator_, Swapchain& swapchain_)
    : instance(instance_), device(device_), memory_allocator(memory_allocator_),
      swapchain(swapchain_) {}

OotOverlay::~OotOverlay() {
    Shutdown();
}

void OotOverlay::Shutdown() {
    if (!ctx && !backend_ready_) {
        return;
    }
    ImGui::SetCurrentContext(ctx);
    if (backend_ready_) {
        (void)device.GetLogical().WaitIdle(); // nothing of ours may still be in flight
    }
    CleanupRetired(true);
    for (auto& tex : live_textures_) {
        if (tex && tex->ds != VK_NULL_HANDLE && backend_ready_) {
            ImGui_ImplVulkan_RemoveTexture(tex->ds);
            tex->ds = VK_NULL_HANDLE;
        }
    }
    live_textures_.clear();
    // E47: the atlas textures carry our GpuTex in BackendUserData (uploaded here, not by the
    // backend). ImGui_ImplVulkan_Shutdown would read it as an ImGui_ImplVulkan_Texture and destroy
    // garbage handles (SIGSEGV in RADV on every exit since pack-ab), so detach them first, as the
    // RomFS font rebuild above already does.
    if (ctx) {
        for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
            tex->SetTexID(ImTextureID_Invalid);
            tex->BackendUserData = nullptr;
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    }
    framebuffers_.clear();
    image_views_.clear();
    render_pass_.reset();
    render_pass_format_ = VK_FORMAT_UNDEFINED;
    if (backend_ready_) {
        ImGui_ImplVulkan_Shutdown();
        backend_ready_ = false;
    }
    descriptor_pool_.reset();
    if (ctx && game_icon_) {
        ImGui::UnregisterUserTexture(game_icon_.get());
    }
    game_icon_.reset();
    game_info_serial_ = 0;
    if (ctx) {
        ImGui::DestroyContext(ctx);
        ctx = nullptr;
    }
}

void OotOverlay::CleanupRetired(bool force_all) {
    constexpr u64 kKeepFrames = 10;
    std::erase_if(retired_uploads_, [&](const RetiredUpload& item) {
        return force_all || (present_serial_ - item.serial >= kKeepFrames);
    });
    for (auto it = retired_textures_.begin(); it != retired_textures_.end();) {
        if (force_all || (present_serial_ - it->serial >= kKeepFrames)) {
            if (it->tex && it->tex->ds != VK_NULL_HANDLE && backend_ready_) {
                ImGui_ImplVulkan_RemoveTexture(it->tex->ds);
                it->tex->ds = VK_NULL_HANDLE;
            }
            it = retired_textures_.erase(it);
        } else {
            ++it;
        }
    }
}

void OotOverlay::BuildFontAtlas() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    const auto& bridge = Common::OotMenu::Bridge::Instance();
    const auto rom_body = bridge.CopyRomFsBodyFont();
    const auto rom_strong = bridge.CopyRomFsStrongFont();
    const auto rom_heading = bridge.CopyRomFsHeadingFont();

    auto add_rom_font = [&](const std::vector<u8>& bytes, const char* name,
                            const unsigned char* fallback_bytes, unsigned fallback_size) {
        void* copy = IM_ALLOC(bytes.size());
        std::memcpy(copy, bytes.data(), bytes.size());
        ImFontConfig cfg;
        std::snprintf(cfg.Name, sizeof(cfg.Name), "%s", name);
        ImFont* primary =
            io.Fonts->AddFontFromMemoryTTF(copy, static_cast<int>(bytes.size()), 18.0f, &cfg);
        if (primary && fallback_bytes && fallback_size > 0) {
            ImFontConfig merge;
            merge.MergeMode = true;
            merge.FontDataOwnedByAtlas = false;
            std::snprintf(merge.Name, sizeof(merge.Name), "%s Fallback", name);
            io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(fallback_bytes),
                                           static_cast<int>(fallback_size), 18.0f, &merge);
        }
        return primary;
    };

    auto add_bundled = [&](const unsigned char* bytes, unsigned size, const char* name) {
        ImFontConfig cfg;
        cfg.FontDataOwnedByAtlas = false;
        std::snprintf(cfg.Name, sizeof(cfg.Name), "%s", name);
        return io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(bytes),
                                              static_cast<int>(size), 18.0f, &cfg);
    };

    if (!rom_body.empty() && !rom_strong.empty() && !rom_heading.empty()) {
        fonts_.body = add_rom_font(rom_body, "ClarimoUDPE-Regular", longshot_font_body,
                                   longshot_font_body_size);
        fonts_.strong = add_rom_font(rom_strong, "ClarimoUDPE-DemiBold", longshot_font_strong,
                                     longshot_font_strong_size);
        fonts_.heading = add_rom_font(rom_heading, "CetraDisplayPro-Medium", longshot_font_heading,
                                      longshot_font_heading_size);
        if (fonts_.body && fonts_.strong && fonts_.heading) {
            fonts_from_romfs_ = true;
            LOG_INFO(Render_Vulkan, "OotOverlay: loaded game RomFS menu fonts with Libertinus fallback");
            return;
        }
        io.Fonts->Clear();
    }

    fonts_.body = add_bundled(longshot_font_body, longshot_font_body_size, "Libertinus Sans");
    fonts_.strong =
        add_bundled(longshot_font_strong, longshot_font_strong_size, "Libertinus Sans Bold");
    fonts_.heading =
        add_bundled(longshot_font_heading, longshot_font_heading_size, "Libertinus Serif Semibold");
    fonts_from_romfs_ = false;
    LOG_INFO(Render_Vulkan, "OotOverlay: loaded bundled Libertinus menu fonts");
}

bool OotOverlay::EnsureInitialized(VkFormat swapchain_format, VkExtent2D extent) {
    if (failed_ || extent.width == 0 || extent.height == 0) {
        return false;
    }

    if (!ctx) {
        ctx = ImGui::CreateContext();
        if (!ctx) {
            failed_ = true;
            return false;
        }
        ImGui::SetCurrentContext(ctx);
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_NoKeyboard;

        BuildFontAtlas();

        if (!loaded_functions_) {
            LoaderCtx loader_ctx{
                .instance = *instance,
                .device = *device.GetLogical(),
                .dld = &device.GetDispatchLoader(),
            };
            loaded_functions_ =
                ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_1, &LoadVulkanFunc, &loader_ctx);
            if (!loaded_functions_) {
                LOG_ERROR(Render_Vulkan, "OotOverlay: ImGui_ImplVulkan_LoadFunctions failed");
                failed_ = true;
                return false;
            }
        }

        const std::array<VkDescriptorPoolSize, 1> pool_sizes{{
            {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 64},
        }};
        descriptor_pool_ = device.GetLogical().CreateDescriptorPool({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 64,
            .poolSizeCount = static_cast<u32>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        });
    } else {
        ImGui::SetCurrentContext(ctx);
        if (!fonts_from_romfs_ && !Common::OotMenu::Bridge::Instance().CopyRomFsBodyFont().empty() &&
            !Common::OotMenu::Bridge::Instance().CopyRomFsStrongFont().empty() &&
            !Common::OotMenu::Bridge::Instance().CopyRomFsHeadingFont().empty()) {
            // Destroy old ImGui font textures safely at idle before rebuilding atlas with RomFS fonts
            (void)device.GetLogical().WaitIdle();
            for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
                if (tex->TexID != ImTextureID_Invalid) {
                    auto* gpu_tex = static_cast<GpuTex*>(tex->BackendUserData);
                    if (gpu_tex && gpu_tex->ds != VK_NULL_HANDLE && backend_ready_) {
                        ImGui_ImplVulkan_RemoveTexture(gpu_tex->ds);
                        gpu_tex->ds = VK_NULL_HANDLE;
                    }
                    tex->SetTexID(ImTextureID_Invalid);
                    tex->SetStatus(ImTextureStatus_Destroyed);
                    tex->BackendUserData = nullptr;
                }
            }
            live_textures_.clear();
            BuildFontAtlas();
        }
    }

    if (!render_pass_ || render_pass_format_ != swapchain_format) {
        if (backend_ready_) {
            (void)device.GetLogical().WaitIdle();
            for (auto& tex : live_textures_) {
                if (tex && tex->ds != VK_NULL_HANDLE) {
                    ImGui_ImplVulkan_RemoveTexture(tex->ds);
                    tex->ds = VK_NULL_HANDLE;
                }
            }
            live_textures_.clear();
            for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
                tex->SetTexID(ImTextureID_Invalid);
                tex->SetStatus(ImTextureStatus_WantCreate);
                tex->BackendUserData = nullptr;
            }
            ImGui_ImplVulkan_Shutdown();
            backend_ready_ = false;
        }
        framebuffers_.clear();
        image_views_.clear();
        render_pass_.reset();

        const VkAttachmentDescription attachment{
            .flags = 0,
            .format = swapchain_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        const VkAttachmentReference color_ref{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        const VkSubpassDescription subpass{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_ref,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };
        const VkSubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = 0,
        };
        render_pass_ = device.GetLogical().CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = 1,
            .pAttachments = &attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        });
        render_pass_format_ = swapchain_format;
        fb_extent_ = {};

        const u32 swap_count = static_cast<u32>(std::max<std::size_t>(2, swapchain.GetImageCount()));
        ImGui_ImplVulkan_InitInfo init_info{};
        init_info.ApiVersion = VK_API_VERSION_1_1;
        init_info.Instance = *instance;
        init_info.PhysicalDevice = device.GetPhysical();
        init_info.Device = *device.GetLogical();
        init_info.QueueFamily = device.GetGraphicsFamily();
        init_info.Queue = *device.GetGraphicsQueue();
        init_info.DescriptorPool = *descriptor_pool_;
        init_info.MinImageCount = swap_count;
        // Keep >= 8 ring slots so in-flight frames never overwrite vertex/index buffers
        init_info.ImageCount = std::max<u32>(8U, swap_count);
        init_info.PipelineInfoMain.RenderPass = *render_pass_;
        init_info.PipelineInfoMain.Subpass = 0;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.CheckVkResultFn = &CheckVk;

        if (!ImGui_ImplVulkan_Init(&init_info)) {
            LOG_ERROR(Render_Vulkan, "OotOverlay: ImGui_ImplVulkan_Init failed");
            failed_ = true;
            return false;
        }
        backend_ready_ = true;
    }

    const std::size_t count = swapchain.GetImageCount();
    if (count == 0) {
        return false;
    }
    bool images_changed = fb_images_.size() != count;
    for (std::size_t i = 0; !images_changed && i < count; ++i) {
        images_changed = fb_images_[i] != swapchain.GetImageIndex(i);
    }
    if (fb_extent_.width != extent.width || fb_extent_.height != extent.height ||
        framebuffers_.size() != count || images_changed) {
        framebuffers_.clear();
        image_views_.clear();
        fb_images_.clear();
        for (std::size_t i = 0; i < count; ++i) {
            fb_images_.push_back(swapchain.GetImageIndex(i));
        }
        image_views_.reserve(count);
        framebuffers_.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto view = device.GetLogical().CreateImageView({
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = swapchain.GetImageIndex(i),
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swapchain_format,
                .components =
                    {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            });
            const VkImageView raw_view = *view;
            auto fb = device.GetLogical().CreateFramebuffer({
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderPass = *render_pass_,
                .attachmentCount = 1,
                .pAttachments = &raw_view,
                .width = extent.width,
                .height = extent.height,
                .layers = 1,
            });
            image_views_.push_back(std::move(view));
            framebuffers_.push_back(std::move(fb));
        }
        fb_extent_ = extent;
    }
    return true;
}

void OotOverlay::SyncGameInfo() {
    const auto& bridge = Common::OotMenu::Bridge::Instance();
    const u64 serial = bridge.GameInfoSerial();
    if (serial == game_info_serial_) {
        return;
    }
    game_info_serial_ = serial;
    const auto info = bridge.GetGameInfo();
    game_title_ = info.title;
    game_id_ = info.program_id ? fmt::format("{:016X}", info.program_id) : std::string{};
    if (info.icon_rgba.empty()) {
        return; // keep the last icon rather than churn textures; a new game brings its own
    }
    if (game_icon_ && (game_icon_->Width != info.icon_width ||
                       game_icon_->Height != info.icon_height)) {
        return; // the core always sends 128x128; a different size is not expected
    }
    if (!game_icon_) {
        game_icon_ = std::make_unique<ImTextureData>();
        game_icon_->Create(ImTextureFormat_RGBA32, info.icon_width, info.icon_height);
        ImGui::RegisterUserTexture(game_icon_.get());
    }
    std::memcpy(game_icon_->GetPixels(), info.icon_rgba.data(), info.icon_rgba.size());
    if (game_icon_->Status == ImTextureStatus_OK) {
        game_icon_->UpdateRect = {0, 0, static_cast<unsigned short>(info.icon_width),
                                  static_cast<unsigned short>(info.icon_height)};
        game_icon_->SetStatus(ImTextureStatus_WantUpdates);
    }
}

void OotOverlay::UploadTextures(const vk::CommandBuffer& cmdbuf, ImDrawData* draw_data) {
    if (!draw_data || !draw_data->Textures) {
        return;
    }
    for (ImTextureData* tex : *draw_data->Textures) {
        if (tex->Status == ImTextureStatus_OK) {
            continue;
        }
        if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0) {
            auto* ptr = static_cast<GpuTex*>(tex->BackendUserData);
            if (ptr) {
                auto it = std::ranges::find_if(
                    live_textures_, [&](const auto& p) { return p.get() == ptr; });
                if (it != live_textures_.end()) {
                    retired_textures_.push_back(
                        {.tex = std::move(*it), .serial = present_serial_});
                    live_textures_.erase(it);
                }
            }
            tex->SetTexID(ImTextureID_Invalid);
            tex->SetStatus(ImTextureStatus_Destroyed);
            tex->BackendUserData = nullptr;
            continue;
        }

        const bool creating = (tex->Status == ImTextureStatus_WantCreate);
        if (creating) {
            auto gpu_tex = std::make_unique<GpuTex>();
            gpu_tex->image = memory_allocator.CreateImage({
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .extent =
                    {
                        .width = static_cast<u32>(tex->Width),
                        .height = static_cast<u32>(tex->Height),
                        .depth = 1,
                    },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            });
            gpu_tex->view = device.GetLogical().CreateImageView({
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = *gpu_tex->image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .components =
                    {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            });
            gpu_tex->ds = ImGui_ImplVulkan_AddTexture(*gpu_tex->view,
                                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            tex->SetTexID(reinterpret_cast<ImTextureID>(gpu_tex->ds));
            tex->BackendUserData = gpu_tex.get();
            live_textures_.push_back(std::move(gpu_tex));
        }

        if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates) {
            auto* gpu_tex = static_cast<GpuTex*>(tex->BackendUserData);
            if (!gpu_tex) {
                continue;
            }
            const int ux = creating ? 0 : static_cast<int>(tex->UpdateRect.x);
            const int uy = creating ? 0 : static_cast<int>(tex->UpdateRect.y);
            const int uw = creating ? tex->Width : static_cast<int>(tex->UpdateRect.w);
            const int uh = creating ? tex->Height : static_cast<int>(tex->UpdateRect.h);
            const std::size_t pitch =
                static_cast<std::size_t>(uw) * static_cast<std::size_t>(tex->BytesPerPixel);
            const VkDeviceSize upload_size = static_cast<VkDeviceSize>(uh) * pitch;

            auto upload_buf = memory_allocator.CreateBuffer(
                {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = upload_size,
                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                MemoryUsage::Upload);
            auto mapped = upload_buf.Mapped();
            for (int y = 0; y < uh; ++y) {
                std::memcpy(mapped.data() + static_cast<std::size_t>(y) * pitch,
                            tex->GetPixelsAt(ux, uy + y), pitch);
            }
            upload_buf.Flush();

            const VkImageMemoryBarrier to_dst{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = creating ? VkAccessFlags{0} : VK_ACCESS_SHADER_READ_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = creating ? VK_IMAGE_LAYOUT_UNDEFINED
                                      : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *gpu_tex->image,
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, to_dst);

            const std::array<VkBufferImageCopy, 1> region{{
                {
                    .bufferOffset = 0,
                    .bufferRowLength = static_cast<u32>(uw),
                    .bufferImageHeight = static_cast<u32>(uh),
                    .imageSubresource =
                        {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .mipLevel = 0,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                        },
                    .imageOffset = {.x = ux, .y = uy, .z = 0},
                    .imageExtent =
                        {
                            .width = static_cast<u32>(uw),
                            .height = static_cast<u32>(uh),
                            .depth = 1,
                        },
                },
            }};
            cmdbuf.CopyBufferToImage(*upload_buf, *gpu_tex->image,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

            const VkImageMemoryBarrier to_read{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *gpu_tex->image,
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, to_read);

            tex->SetStatus(ImTextureStatus_OK);
            retired_uploads_.push_back({.buffer = std::move(upload_buf), .serial = present_serial_});
        }
    }
    // Disable ImGui's synchronous Upload/Destroy fallback inside RenderDrawData since we uploaded
    // onto our own asynchronous present command buffer.
    draw_data->Textures = nullptr;
}

void OotOverlay::Idle() {
    ++present_serial_;
    if (ctx) {
        CleanupRetired(false);
    }
}

void OotOverlay::Record(const vk::CommandBuffer& cmdbuf, VkImage swapchain_image, std::size_t image_index,
                        VkExtent2D extent, VkFormat swapchain_format, bool is_hdr,
                        const std::function<void(OotUi::Ctx&)>& draw_fn) {
    ++present_serial_;
    if (!EnsureInitialized(swapchain_format, extent)) {
        return;
    }
    if (image_index >= framebuffers_.size()) {
        return;
    }
    CleanupRetired(false);

    const auto now = std::chrono::steady_clock::now();
    float dt = 1.0f / 60.0f;
    if (last_frame_time_ != std::chrono::steady_clock::time_point{}) {
        const float elapsed =
            std::chrono::duration<float>(now - last_frame_time_).count();
        dt = std::clamp(elapsed, 0.001f, 0.25f);
    }
    last_frame_time_ = now;

    ImGui::SetCurrentContext(ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize =
        ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
    io.DeltaTime = dt;

    SyncGameInfo(); // before NewFrame, which gathers the user textures
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    const auto& bridge = Common::OotMenu::Bridge::Instance();
    const auto opts = bridge.GetOptions();
    const auto bindings = bridge.GetPadBindings();

    OotUi::SetLang(opts.menu_lang);
    OotUi::Ctx ui_ctx{};
    ui_ctx.d = ImGui::GetForegroundDrawList();
    ui_ctx.f = fonts_;
    ui_ctx.w = static_cast<float>(extent.width);
    ui_ctx.h = static_cast<float>(extent.height);
    ui_ctx.s = OotUi::Unit(ui_ctx.h, opts.menu_size, &ui_ctx.safe);
    ui_ctx.pad = bindings.pad_connected && !bindings.last_input_keyboard;
    const bool nintendo_pad = (opts.pad_layout == static_cast<int>(Common::OotMenu::PadLayout::Nintendo));
    ui_ctx.confirm_dot = nintendo_pad ? 1 : 3;
    ui_ctx.back_dot = nintendo_pad ? 3 : 1;
    ui_ctx.has_game_icon = game_icon_ != nullptr;
    if (game_icon_) {
        ui_ctx.game_icon = game_icon_->GetTexRef();
    }
    ui_ctx.game_title = game_title_.c_str();
    ui_ctx.game_id = game_id_.c_str();

    draw_fn(ui_ctx);

    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (static int logged = 0; logged < 3) {
        ++logged;
        int tex_ok = 0, tex_total = 0;
        if (draw_data && draw_data->Textures) {
            for (const ImTextureData* t : *draw_data->Textures) {
                ++tex_total;
                tex_ok += t->Status == ImTextureStatus_OK ? 1 : 0;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "OotOverlay: serial {} lists {} vtx {} textures {}/{} ok, {}x{} fmt {} img {}/{}",
                 present_serial_, draw_data ? draw_data->CmdListsCount : -1,
                 draw_data ? draw_data->TotalVtxCount : -1, tex_ok, tex_total, extent.width,
                 extent.height, static_cast<int>(swapchain_format), image_index,
                 framebuffers_.size());
    }
    if (!draw_data || draw_data->CmdListsCount == 0) {
        return;
    }

    const double white_nits = Settings::values.hdr_reference_white.GetValue() > 0
                                  ? static_cast<double>(Settings::values.hdr_reference_white.GetValue())
                                  : 200.0;
    RemapDrawDataColors(draw_data, IsSrgbFormat(swapchain_format), is_hdr, white_nits);
    UploadTextures(cmdbuf, draw_data);

    TransitionSwapchainImage(cmdbuf, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    const VkRenderPassBeginInfo rp_begin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = *render_pass_,
        .framebuffer = *framebuffers_[image_index],
        .renderArea =
            {
                .offset = {.x = 0, .y = 0},
                .extent = extent,
            },
        .clearValueCount = 0,
        .pClearValues = nullptr,
    };
    cmdbuf.BeginRenderPass(rp_begin, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw_data, *cmdbuf);
    cmdbuf.EndRenderPass();

    TransitionSwapchainImage(cmdbuf, swapchain_image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT);
}

} // namespace Vulkan
