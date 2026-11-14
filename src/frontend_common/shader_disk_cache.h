// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace FrontendCommon {

struct ShaderDiskCacheBudget {
    std::uint64_t bytes = 10ULL << 30;
    std::string nvidia_bytes = "10737418240";
    std::string mesa_size = "10G";
};

/// Computes an adaptive driver shader disk cache budget from filesystem free space:
/// 25% of `available_bytes` rounded down to whole GiB, clamped to [1 GiB, 32 GiB] (512 MiB
/// emergency floor below 2 GiB free; 10 GiB fallback when filesystem space is unavailable).
ShaderDiskCacheBudget ComputeShaderDiskCacheBudget(std::uint64_t available_bytes,
                                                   bool has_space_info = true);

/// Queries free space on the filesystem holding `probe_path` (walking up to the closest existing
/// ancestor) and returns the resulting adaptive shader disk cache budget.
ShaderDiskCacheBudget ComputeShaderDiskCacheBudgetForPath(const std::filesystem::path& probe_path);

/// Configures driver shader disk cache defaults dynamically from free disk space on each driver's
/// cache filesystem without overriding explicit user environment variables.
void SetUpShaderDiskCaches();

} // namespace FrontendCommon
