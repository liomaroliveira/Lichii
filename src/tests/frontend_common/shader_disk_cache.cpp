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

TEST_CASE("FrontendCommon: SetUpShaderDiskCaches configures defaults and preserves overrides",
          "[frontend_common]") {
#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
    ScopedEnvVar keep_xdg("XDG_CACHE_HOME");
    ScopedEnvVar keep_nv_path("__GL_SHADER_DISK_CACHE_PATH");
    ScopedEnvVar keep_nv_size("__GL_SHADER_DISK_CACHE_SIZE");
    ScopedEnvVar keep_mesa_size("MESA_SHADER_CACHE_MAX_SIZE");

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto temp_cache =
        std::filesystem::temp_directory_path() / ("lichii-disk-cache-test-" + std::to_string(stamp));
    setenv("XDG_CACHE_HOME", temp_cache.c_str(), 1);
    unsetenv("__GL_SHADER_DISK_CACHE_PATH");
    unsetenv("__GL_SHADER_DISK_CACHE_SIZE");
    unsetenv("MESA_SHADER_CACHE_MAX_SIZE");

    FrontendCommon::SetUpShaderDiskCaches();

    const char* nv_path = std::getenv("__GL_SHADER_DISK_CACHE_PATH");
    const char* nv_size = std::getenv("__GL_SHADER_DISK_CACHE_SIZE");
    const char* mesa_size = std::getenv("MESA_SHADER_CACHE_MAX_SIZE");
    REQUIRE(nv_path != nullptr);
    CHECK(std::string(nv_path) == (temp_cache / "lichii" / "nvidia").string());
    REQUIRE(nv_size != nullptr);
    CHECK(std::string(nv_size) == "10737418240");
    REQUIRE(mesa_size != nullptr);
    CHECK(std::string(mesa_size) == "10G");

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
