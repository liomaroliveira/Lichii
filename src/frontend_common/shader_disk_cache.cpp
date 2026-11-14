// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "frontend_common/shader_disk_cache.h"

namespace FrontendCommon {

namespace {

#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
std::filesystem::path ResolveEnvPath(const char* name) {
    const char* const raw = std::getenv(name);
    if (raw != nullptr && raw[0] == '/') {
        return std::filesystem::path{raw};
    }
    return {};
}

std::filesystem::path ResolveCacheBase() {
    if (const auto xdg_cache = ResolveEnvPath("XDG_CACHE_HOME"); !xdg_cache.empty()) {
        return xdg_cache;
    }
    if (const auto home = ResolveEnvPath("HOME"); !home.empty()) {
        return home / ".cache";
    }
    return {};
}

void SetEnvIfUnset(const char* name, const char* value) {
    const char* const current = std::getenv(name);
    if (current == nullptr || current[0] == '\0') {
        setenv(name, value, 1);
    }
}
#endif

} // namespace

ShaderDiskCacheBudget ComputeShaderDiskCacheBudget(std::uint64_t available_bytes,
                                                   bool has_space_info) {
    constexpr std::uint64_t MiB = 1ULL << 20;
    constexpr std::uint64_t GiB = 1ULL << 30;
    constexpr std::uint64_t LowDiskThreshold = 2ULL * GiB;
    constexpr std::uint64_t EmergencyBytes = 512ULL * MiB;
    constexpr std::uint64_t MinGiB = 1;
    constexpr std::uint64_t MaxGiB = 32;

    if (!has_space_info) {
        return {};
    }
    if (available_bytes < LowDiskThreshold) {
        return {
            .bytes = EmergencyBytes,
            .nvidia_bytes = std::to_string(EmergencyBytes),
            .mesa_size = "512M",
        };
    }
    const std::uint64_t gib = std::clamp<std::uint64_t>((available_bytes / 4) / GiB, MinGiB, MaxGiB);
    const std::uint64_t bytes = gib * GiB;
    return {
        .bytes = bytes,
        .nvidia_bytes = std::to_string(bytes),
        .mesa_size = std::to_string(gib) + "G",
    };
}

ShaderDiskCacheBudget ComputeShaderDiskCacheBudgetForPath(const std::filesystem::path& probe_path) {
    std::filesystem::path cur = probe_path;
    std::error_code ec;
    while (!cur.empty() && !std::filesystem::exists(cur, ec)) {
        ec.clear();
        const auto parent = cur.parent_path();
        if (parent == cur) {
            break;
        }
        cur = parent;
    }
    if (cur.empty()) {
        return ComputeShaderDiskCacheBudget(0, false);
    }
    const std::filesystem::space_info info = std::filesystem::space(cur, ec);
    if (ec || info.available == static_cast<std::uintmax_t>(-1)) {
        return ComputeShaderDiskCacheBudget(0, false);
    }
    return ComputeShaderDiskCacheBudget(static_cast<std::uint64_t>(info.available), true);
}

void SetUpShaderDiskCaches() {
#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
    const std::filesystem::path base = ResolveCacheBase();
    std::filesystem::path nv_probe = ResolveEnvPath("__GL_SHADER_DISK_CACHE_PATH");
    if (nv_probe.empty() && !base.empty()) {
        const std::filesystem::path folder = base / "lichii" / "nvidia";
        std::error_code ec;
        std::filesystem::create_directories(folder, ec);
        if (!ec) {
            setenv("__GL_SHADER_DISK_CACHE_PATH", folder.c_str(), 0);
            nv_probe = folder;
        } else {
            nv_probe = base;
        }
    }

    const ShaderDiskCacheBudget nv_budget = nv_probe.empty()
                                                ? ComputeShaderDiskCacheBudget(0, false)
                                                : ComputeShaderDiskCacheBudgetForPath(nv_probe);
    SetEnvIfUnset("__GL_SHADER_DISK_CACHE_SIZE", nv_budget.nvidia_bytes.c_str());

    std::filesystem::path mesa_probe = ResolveEnvPath("MESA_SHADER_CACHE_DIR");
    if (mesa_probe.empty()) {
        mesa_probe = base;
    }
    const ShaderDiskCacheBudget mesa_budget = mesa_probe.empty()
                                                  ? ComputeShaderDiskCacheBudget(0, false)
                                                  : ComputeShaderDiskCacheBudgetForPath(mesa_probe);
    // Mesa defaults to 1G; full-game shader caches such as Zelda OoT exceed 1 GiB on disk,
    // so scale cleanly with available disk space unless the user configured a cap explicitly.
    SetEnvIfUnset("MESA_SHADER_CACHE_MAX_SIZE", mesa_budget.mesa_size.c_str());
#endif
}

} // namespace FrontendCommon
