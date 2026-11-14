// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
#include <getopt.h>
#endif

#include "common/settings.h"
#include "core/core.h"
#include "core/launch_params.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

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

TEST_CASE("Vulkan::IsPrecompileAllEnv accepts truthy values and rejects false/empty",
          "[video_core]") {
    REQUIRE_FALSE(Vulkan::IsPrecompileAllEnv(nullptr));
    REQUIRE_FALSE(Vulkan::IsPrecompileAllEnv(""));
    REQUIRE_FALSE(Vulkan::IsPrecompileAllEnv("0"));
    REQUIRE_FALSE(Vulkan::IsPrecompileAllEnv("false"));
    REQUIRE_FALSE(Vulkan::IsPrecompileAllEnv("no"));
    REQUIRE_FALSE(Vulkan::IsPrecompileAllEnv("off"));

    REQUIRE(Vulkan::IsPrecompileAllEnv("1"));
    REQUIRE(Vulkan::IsPrecompileAllEnv("true"));
    REQUIRE(Vulkan::IsPrecompileAllEnv("TRUE"));
    REQUIRE(Vulkan::IsPrecompileAllEnv("yes"));
    REQUIRE(Vulkan::IsPrecompileAllEnv("YES"));
    REQUIRE(Vulkan::IsPrecompileAllEnv("on"));
    REQUIRE(Vulkan::IsPrecompileAllEnv("ON"));
}

#if defined(__unix__) && !defined(__APPLE__) && !defined(__ANDROID__)
TEST_CASE("Vulkan::PrecompileAllRequested tracks EDEN_NX2_PRECOMPILE_ALL", "[video_core]") {
    ScopedEnvVar keep_env("EDEN_NX2_PRECOMPILE_ALL");

    unsetenv("EDEN_NX2_PRECOMPILE_ALL");
    REQUIRE_FALSE(Vulkan::PrecompileAllRequested());

    setenv("EDEN_NX2_PRECOMPILE_ALL", "0", 1);
    REQUIRE_FALSE(Vulkan::PrecompileAllRequested());

    setenv("EDEN_NX2_PRECOMPILE_ALL", "1", 1);
    REQUIRE(Vulkan::PrecompileAllRequested());
}

TEST_CASE("Core::ParseLaunchParams and ApplyLaunchParams handle --precompile-all", "[core]") {
    const bool prev_disk = Settings::values.use_disk_shader_cache.GetValue();
    const bool prev_vk = Settings::values.use_vulkan_driver_pipeline_cache.GetValue();
    Settings::values.use_disk_shader_cache = false;
    Settings::values.use_vulkan_driver_pipeline_cache = false;

    Core::System system;
    std::string arg0 = "lichii-cmd";
    std::string arg1 = "--precompile-all";
    std::string arg2 = "/tmp/game.nup";
    std::vector<char*> argv{arg0.data(), arg1.data(), arg2.data()};

    optind = 1;
    const Core::LaunchParams lp =
        Core::ParseLaunchParams(system, static_cast<int>(argv.size()), argv.data(), nullptr);
    optind = 1;

    REQUIRE(lp.precompile_all);
    REQUIRE(lp.filepath == "/tmp/game.nup");

    Core::ApplyLaunchParams(lp);
    REQUIRE(Settings::values.use_disk_shader_cache.GetValue());
    REQUIRE(Settings::values.use_vulkan_driver_pipeline_cache.GetValue());

    Settings::values.use_disk_shader_cache = prev_disk;
    Settings::values.use_vulkan_driver_pipeline_cache = prev_vk;
}
#endif
