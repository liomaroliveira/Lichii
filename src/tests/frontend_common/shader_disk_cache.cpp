// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

#include "frontend_common/shader_disk_cache.h"

namespace {

#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
struct ScopedEnvVar {
    explicit ScopedEnvVar(const char* name_) : name(name_) {
        if (const char* current = std::getenv(name.c_str())) {
            original = current;
        }
    }

    ~ScopedEnvVar() {
        if (original) {
            setenv(name.c_str(), original->c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }

    std::string name;
    std::optional<std::string> original;
};
#endif

} // namespace

TEST_CASE("FrontendCommon: ComputeShaderDiskCacheBudget scales with free space and clamps",
          "[frontend_common]") {
    constexpr std::uint64_t MiB = 1ULL << 20;
    constexpr std::uint64_t GiB = 1ULL << 30;

    const auto fallback = FrontendCommon::ComputeShaderDiskCacheBudget(0, false);
    CHECK(fallback.bytes == 10ULL * GiB);
    CHECK(fallback.nvidia_bytes == "10737418240");
    CHECK(fallback.mesa_size == "10G");

    const auto tight = FrontendCommon::ComputeShaderDiskCacheBudget(1ULL * GiB, true);
    CHECK(tight.bytes == 512ULL * MiB);
    CHECK(tight.nvidia_bytes == "536870912");
    CHECK(tight.mesa_size == "512M");

    const auto small = FrontendCommon::ComputeShaderDiskCacheBudget(3ULL * GiB, true);
    CHECK(small.bytes == 1ULL * GiB);
    CHECK(small.mesa_size == "1G");

    const auto medium = FrontendCommon::ComputeShaderDiskCacheBudget(78ULL * GiB, true);
    CHECK(medium.bytes == 19ULL * GiB);
    CHECK(medium.nvidia_bytes == std::to_string(19ULL * GiB));
    CHECK(medium.mesa_size == "19G");

    const auto large = FrontendCommon::ComputeShaderDiskCacheBudget(200ULL * GiB, true);
    CHECK(large.bytes == 32ULL * GiB);
    CHECK(large.nvidia_bytes == "34359738368");
    CHECK(large.mesa_size == "32G");
}

TEST_CASE("FrontendCommon: SetUpShaderDiskCaches configures dynamic budget and preserves overrides",
          "[frontend_common]") {
#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
    ScopedEnvVar keep_xdg("XDG_CACHE_HOME");
    ScopedEnvVar keep_mesa_dir("MESA_SHADER_CACHE_DIR");
    ScopedEnvVar keep_nv_path("__GL_SHADER_DISK_CACHE_PATH");
    ScopedEnvVar keep_nv_size("__GL_SHADER_DISK_CACHE_SIZE");
    ScopedEnvVar keep_mesa_size("MESA_SHADER_CACHE_MAX_SIZE");

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto temp_cache =
        std::filesystem::temp_directory_path() / ("lichii-disk-cache-test-" + std::to_string(stamp));
    setenv("XDG_CACHE_HOME", temp_cache.c_str(), 1);
    unsetenv("MESA_SHADER_CACHE_DIR");
    unsetenv("__GL_SHADER_DISK_CACHE_PATH");
    unsetenv("__GL_SHADER_DISK_CACHE_SIZE");
    unsetenv("MESA_SHADER_CACHE_MAX_SIZE");

    const auto expected_budget = FrontendCommon::ComputeShaderDiskCacheBudgetForPath(temp_cache);

    FrontendCommon::SetUpShaderDiskCaches();

    const char* nv_path = std::getenv("__GL_SHADER_DISK_CACHE_PATH");
    const char* nv_size = std::getenv("__GL_SHADER_DISK_CACHE_SIZE");
    const char* mesa_size = std::getenv("MESA_SHADER_CACHE_MAX_SIZE");
    REQUIRE(nv_path != nullptr);
    CHECK(std::string(nv_path) == (temp_cache / "lichii" / "nvidia").string());
    REQUIRE(nv_size != nullptr);
    CHECK(std::string(nv_size) == expected_budget.nvidia_bytes);
    REQUIRE(mesa_size != nullptr);
    CHECK(std::string(mesa_size) == expected_budget.mesa_size);

    // Explicit user overrides must remain untouched.
    setenv("MESA_SHADER_CACHE_MAX_SIZE", "4G", 1);
    setenv("__GL_SHADER_DISK_CACHE_SIZE", "2147483648", 1);
    FrontendCommon::SetUpShaderDiskCaches();
    CHECK(std::string(std::getenv("MESA_SHADER_CACHE_MAX_SIZE")) == "4G");
    CHECK(std::string(std::getenv("__GL_SHADER_DISK_CACHE_SIZE")) == "2147483648");

    std::error_code ec;
    std::filesystem::remove_all(temp_cache, ec);
#else
    FrontendCommon::SetUpShaderDiskCaches();
    SUCCEED();
#endif
}
