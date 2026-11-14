// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "frontend_common/shader_disk_cache.h"

namespace FrontendCommon {

void SetUpShaderDiskCaches() {
#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
    if (std::getenv("__GL_SHADER_DISK_CACHE_PATH") == nullptr) {
        const char* const xdg_cache = std::getenv("XDG_CACHE_HOME");
        const char* const home = std::getenv("HOME");
        std::filesystem::path base;
        if (xdg_cache != nullptr && xdg_cache[0] == '/') {
            base = xdg_cache;
        } else if (home != nullptr && home[0] == '/') {
            base = std::filesystem::path{home} / ".cache";
        }
        if (!base.empty()) {
            const std::filesystem::path folder = base / "lichii" / "nvidia";
            std::error_code ec;
            std::filesystem::create_directories(folder, ec);
            if (!ec) {
                setenv("__GL_SHADER_DISK_CACHE_PATH", folder.c_str(), 0);
            }
        }
    }
    setenv("__GL_SHADER_DISK_CACHE_SIZE", "10737418240", 0);
    // Mesa defaults to 1G; full-game shader caches such as Zelda OoT exceed 1 GiB on disk.
    setenv("MESA_SHADER_CACHE_MAX_SIZE", "10G", 0);
#endif
}

} // namespace FrontendCommon
