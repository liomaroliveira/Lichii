// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

vk::DebugUtilsMessenger CreateDebugUtilsCallback(const vk::Instance& instance);

/// EDEN_NX2_ADDR_REPORT=1[,<hex address>]: logs every VK_EXT_device_address_binding_report bind or
/// unbind that covers the address (default 0x800100200000, the RDNA2 device-loss page), with the
/// object and the call stack (module+offset, for addr2line). Diagnostic only.
bool AddressReportEnabled();
vk::DebugUtilsMessenger CreateAddressBindingCallback(const vk::Instance& instance);

} // namespace Vulkan
