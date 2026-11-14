// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "shader_recompiler/exception.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_warm/shader_archives.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/shader_warm/warm_records.h"

namespace {

namespace ShaderWarm = VideoCommon::ShaderWarm;
using Settings::PrecacheShaders;

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-shader-precache-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void WriteBytes(const std::filesystem::path& file, std::string_view bytes) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

} // Anonymous namespace

TEST_CASE("ShaderPrecache: the setting decides whether the boot precaches", "[video_core]") {
    STATIC_REQUIRE_FALSE(ShaderWarm::PrecacheBeforeStart(PrecacheShaders::Off, false));
    STATIC_REQUIRE_FALSE(ShaderWarm::PrecacheBeforeStart(PrecacheShaders::Off, true));
    STATIC_REQUIRE_FALSE(ShaderWarm::PrecacheBeforeStart(PrecacheShaders::Ask, false));
    STATIC_REQUIRE(ShaderWarm::PrecacheBeforeStart(PrecacheShaders::Ask, true));
    STATIC_REQUIRE(ShaderWarm::PrecacheBeforeStart(PrecacheShaders::Always, false));
    STATIC_REQUIRE(ShaderWarm::PrecacheBeforeStart(PrecacheShaders::Always, true));
}

TEST_CASE("ShaderPrecache: launch requests are taken once", "[video_core]") {
    ShaderWarm::RequestWarmBeforeStart(true);
    REQUIRE(ShaderWarm::TakeWarmBeforeStart());
    REQUIRE_FALSE(ShaderWarm::TakeWarmBeforeStart());

    ShaderWarm::RequestSkipWarm();
    REQUIRE(ShaderWarm::TakeSkipWarm());
    REQUIRE_FALSE(ShaderWarm::TakeSkipWarm());

    // A skip pressed during an earlier launch does not skip the next one.
    ShaderWarm::RequestSkipWarm();
    ShaderWarm::RequestWarmBeforeStart(true);
    REQUIRE_FALSE(ShaderWarm::TakeSkipWarm());
    REQUIRE(ShaderWarm::TakeWarmBeforeStart());
}

TEST_CASE("ShaderPrecache: the marker matches only the same file and driver", "[video_core]") {
    TempDir dir;
    const auto warm = dir.path / "warm-compute.bin";
    REQUIRE_FALSE(ShaderWarm::CurrentWarmStamp(warm, 0, 1, 2, 3).has_value());
    REQUIRE_FALSE(ShaderWarm::ReadWarmMarker(warm).has_value());

    WriteBytes(warm, "precache");
    const auto stamp = ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53);
    REQUIRE(stamp.has_value());
    REQUIRE(stamp->size == 8);
    REQUIRE(ShaderWarm::WriteWarmMarker(warm, *stamp));
    REQUIRE(ShaderWarm::WarmMarkerPath(warm) == dir.path / "warm-compute.done");

    const auto marker = ShaderWarm::ReadWarmMarker(warm);
    REQUIRE(marker.has_value());
    REQUIRE(marker->entries == 12);
    // The entry count is informative: a stamp taken before reading the file still matches.
    REQUIRE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 0, 0x1234, 0x2206, 53)));
    REQUIRE_FALSE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1235, 0x2206, 53)));
    REQUIRE_FALSE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2207, 53)));
    REQUIRE_FALSE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 54)));
    // Another variant (an update that changed the SPIR-V) builds the programs again.
    REQUIRE_FALSE(
        marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53, 0x77)));

    // Built again with the same contents: still done.
    WriteBytes(warm, "precache");
    REQUIRE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53)));

    // The same size with other contents: only the content hash tells them apart.
    WriteBytes(warm, "precachf");
    REQUIRE_FALSE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53)));

    WriteBytes(warm, "a rebuilt precache");
    REQUIRE_FALSE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53)));
}

TEST_CASE("ShaderPrecache: a rebuild with the same contents keeps the marker", "[video_core]") {
    TempDir dir;
    const auto warm = dir.path / "warm-compute.bin";
    const auto rebuilt = dir.path / "warm-compute.tmp";
    WriteBytes(warm, "precache");
    REQUIRE(ShaderWarm::WriteWarmMarker(
        warm, *ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53)));

    WriteBytes(rebuilt, "precache");
    ShaderWarm::DropStaleWarmMarker(warm, rebuilt);
    REQUIRE(ShaderWarm::ReadWarmMarker(warm).has_value());

    WriteBytes(rebuilt, "a rebuilt precache");
    ShaderWarm::DropStaleWarmMarker(warm, rebuilt);
    REQUIRE_FALSE(std::filesystem::exists(ShaderWarm::WarmMarkerPath(warm)));

    // No marker left and nothing to replace the file with: nothing to do.
    ShaderWarm::DropStaleWarmMarker(warm, dir.path / "missing.tmp");
    REQUIRE_FALSE(ShaderWarm::ReadWarmMarker(warm).has_value());
}

TEST_CASE("ShaderPrecache: the marker matches only the same variant", "[video_core]") {
    TempDir dir;
    const auto warm = dir.path / "warm-compute.bin";
    WriteBytes(warm, "precache");
    const u64 variant = 0xfedcba9876543210ULL;
    const auto stamp = ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53, variant);
    REQUIRE(stamp.has_value());
    REQUIRE(ShaderWarm::WriteWarmMarker(warm, *stamp));

    const auto marker = ShaderWarm::ReadWarmMarker(warm);
    REQUIRE(marker.has_value());
    REQUIRE(marker->variant == variant);
    REQUIRE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 0, 0x1234, 0x2206, 53, variant)));
    REQUIRE_FALSE(
        marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53, variant + 1)));
    REQUIRE_FALSE(marker->Matches(*ShaderWarm::CurrentWarmStamp(warm, 12, 0x1234, 0x2206, 53)));
    REQUIRE(ShaderWarm::EmissionStamp() != 0);
}

TEST_CASE("ShaderPrecache: a marker of an older format is not read", "[video_core]") {
    TempDir dir;
    const auto warm = dir.path / "warm-compute.bin";
    WriteBytes(warm, "precache");
    // Size, last-write time, entries, driver, device and cache version, then the line end.
    WriteBytes(ShaderWarm::WarmMarkerPath(warm), "8 133981234567890000 12 4660 8710 53\n");
    REQUIRE_FALSE(ShaderWarm::ReadWarmMarker(warm).has_value());
    // The same with a variant but no format tag.
    WriteBytes(ShaderWarm::WarmMarkerPath(warm),
               "8 133981234567890000 12 4660 8710 53 18364758544493064720\n");
    REQUIRE_FALSE(ShaderWarm::ReadWarmMarker(warm).has_value());
    // Tagged v2 but cut short.
    WriteBytes(ShaderWarm::WarmMarkerPath(warm), "v2 8 1234 12 4660 8710 53\n");
    REQUIRE_FALSE(ShaderWarm::ReadWarmMarker(warm).has_value());

    REQUIRE(ShaderWarm::WriteWarmMarker(warm, *ShaderWarm::CurrentWarmStamp(warm, 12, 1, 2, 3)));
    REQUIRE(ShaderWarm::ReadWarmMarker(warm).has_value());
}

TEST_CASE("ShaderPrecache: the entry count belongs to a file of one size", "[video_core]") {
    TempDir dir;
    const auto warm = dir.path / "warm-compute.bin";
    WriteBytes(warm, "0123456789");
    REQUIRE(ShaderWarm::ReadWarmCount(warm) == 0);

    WriteBytes(ShaderWarm::WarmCountPath(warm), "345 10\n");
    REQUIRE(ShaderWarm::ReadWarmCount(warm) == 345);

    WriteBytes(warm, "01234567890");
    REQUIRE(ShaderWarm::ReadWarmCount(warm) == 0);
}

TEST_CASE("ShaderPrecache: a file from an older builder or another cache version is stale",
          "[video_core]") {
    TempDir dir;
    const auto warm = dir.path / "warm-compute.bin";
    REQUIRE(ShaderWarm::WarmFilesStale(warm).has_value());

    const auto header = [](u32 version) {
        std::string bytes(VideoCommon::PipelineCacheMagic.begin(),
                          VideoCommon::PipelineCacheMagic.end());
        bytes.append(reinterpret_cast<const char*>(&version), sizeof(version));
        return bytes;
    };
    WriteBytes(warm, header(ShaderWarm::WarmCacheVersion));
    const auto count = ShaderWarm::WarmCountPath(warm);
    // Count files written before the builder version was recorded have two fields.
    WriteBytes(count, "0 12\n");
    REQUIRE(ShaderWarm::WarmFilesStale(warm).has_value());
    WriteBytes(count, fmt::format("0 12 {}\n", ShaderWarm::WarmBuilderVersion - 1));
    REQUIRE(ShaderWarm::WarmFilesStale(warm).has_value());
    WriteBytes(count, fmt::format("0 12 {}\n", ShaderWarm::WarmBuilderVersion));
    REQUIRE_FALSE(ShaderWarm::WarmFilesStale(warm).has_value());

    WriteBytes(count, fmt::format("0 13 {}\n", ShaderWarm::WarmBuilderVersion));
    REQUIRE(ShaderWarm::WarmFilesStale(warm).has_value());
    WriteBytes(count, fmt::format("0 12 {}\n", ShaderWarm::WarmBuilderVersion));
    // The emulator loads a file of a compatible older version, so it is not built again.
    const auto& compatible = VideoCommon::CompatibleVulkanPipelineCacheVersions;
    WriteBytes(warm, header(compatible.front()));
    REQUIRE_FALSE(ShaderWarm::WarmFilesStale(warm).has_value());
    WriteBytes(warm, header(compatible.back() - 1));
    REQUIRE(ShaderWarm::WarmFilesStale(warm).has_value());
}

TEST_CASE("ShaderPrecache: a compute entry reads back like a pipeline cache entry",
          "[video_core]") {
    ShaderWarm::LoweredCode lowered;
    lowered.code.resize(16 * sizeof(u64));
    for (size_t i = 0; i < lowered.code.size(); ++i) {
        lowered.code[i] = static_cast<u8>(i * 7 + 1);
    }
    lowered.texture_buffer = 3;
    const ShaderWarm::ComputeControl control{{64, 2, 1}, 1024, 256};
    ShaderWarm::ComputeKey key_bytes{};
    const auto entry = ShaderWarm::ComputeEntry(lowered, control, key_bytes);

    TempDir dir;
    const auto file = dir.path / "entry.bin";
    WriteBytes(file, std::string_view{reinterpret_cast<const char*>(entry.data()), entry.size()});

    std::ifstream in(file, std::ios::binary);
    in.exceptions(std::ifstream::failbit);
    u32 num_envs{};
    in.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
    REQUIRE(num_envs == 1);
    VideoCommon::FileEnvironment env;
    env.Deserialize(in, static_cast<std::streamoff>(entry.size()));
    Vulkan::ComputePipelineCacheKey key{};
    static_assert(sizeof(key) == sizeof(key_bytes));
    in.read(reinterpret_cast<char*>(&key), sizeof(key));
    REQUIRE(in.tellg() == static_cast<std::streamoff>(entry.size()));

    REQUIRE(env.ShaderStage() == Shader::Stage::Compute);
    REQUIRE(env.WorkgroupSize() == control.workgroup);
    REQUIRE(env.SharedMemorySize() == control.shared_memory);
    REQUIRE(env.LocalMemorySize() == control.local_memory);
    REQUIRE(env.TextureBoundBuffer() == 3);
    REQUIRE(std::memcmp(&key, key_bytes.data(), sizeof(key)) == 0);
    REQUIRE(key.unique_hash == ShaderWarm::StageHash(lowered.code));
    REQUIRE(key.shared_memory_size == control.shared_memory);
    REQUIRE(key.workgroup_size == control.workgroup);

    // The entry has no recorded reads: without guesses they are errors, with them the guesses.
    REQUIRE_THROWS_AS(env.ReadCbufValue(0, 0x40), Shader::LogicError);
    static const VideoCommon::GuessMissing guess{};
    env.SetGuessMissing(&guess);
    REQUIRE(env.ReadCbufValue(0, 0x40) == guess.cbuf_bank0);
    REQUIRE(env.ReadCbufValue(3, 0x40) == guess.cbuf_other);
    REQUIRE(env.ReadTextureType(5) == guess.texture_type);
    REQUIRE(env.IsCbufStorageBufferPointer(0, 0x10) == guess.storage_pointer_bank0);
    REQUIRE(env.IsCbufStorageBufferPointer(1, 0x10) == guess.storage_pointer_other);
}

TEST_CASE("ShaderPrecache: a program naming no texture bank reads bank 0", "[video_core]") {
    ShaderWarm::LoweredCode lowered;
    lowered.code.resize(16 * sizeof(u64));
    ShaderWarm::ComputeKey key{};
    const auto entry = ShaderWarm::ComputeEntry(lowered, {{1, 1, 1}, 0, 0}, key);
    u32 texture_bound{};
    // num_envs, then six u64 counts and local memory before the texture bank.
    std::memcpy(&texture_bound, entry.data() + 4 + 6 * 8 + 4, sizeof(texture_bound));
    REQUIRE(texture_bound == 0);
}

TEST_CASE("ShaderPrecache: every container the archive walker reads is discovered",
          "[video_core]") {
    using ShaderWarm::ClassifyGameFile;
    using ShaderWarm::GameFileKind;
    for (const char* path : {"Shader/Model.product.nin_ounce_nvn2.bfsha.zs",
                             "Shader/Model.bfsha", "Pack/Shaders.nin_ounce_nvn2.release.sarc.zs",
                             "Pack/Shaders.release.sarc", "Shader/Loose.bnsh",
                             "Shader/Loose.bnsh.zs", "Shader/Bundle.cgsb", "Shader/Bundle.cgsb.zs",
                             "Shader/Program.shn", "Shader/Program.shn.zs", "Shader/Archive.sharcb",
                             "Shader/Archive.sharcb.zs", "Shader/UPPER.BNSH", "Effect/Fx.bfsha.zs",
                             "Shaders/Model.bfsha.zst", "Lib/agl/agl_ndsr.pck",
                             "Lib/agl/agl_ndsr.pck.zs", "Movie/Ndve/Ndve-Nvn2-vid_seq5.dat.zs",
                             "Movie/Ndve/Network.pck"}) {
        INFO(path);
        REQUIRE(ClassifyGameFile(path) == GameFileKind::ShaderArchive);
    }
    REQUIRE(ClassifyGameFile("Effect/Fx.nin_ounce_nvn2.vsetb.byml.zs") == GameFileKind::Effect);
    REQUIRE(ClassifyGameFile("Effect/static.Nin_Ounce_NVN2.esetb.byml.zs") == GameFileKind::Effect);
    REQUIRE(ClassifyGameFile("Effect/Fx.ESETB.BYML.ZS") == GameFileKind::Effect);
    REQUIRE(ClassifyGameFile("Effect/Particle/Magic.ptcl.zst") == GameFileKind::Effect);
    REQUIRE(ClassifyGameFile("Effect/Particle/Magic.ptcl") == GameFileKind::Effect);
    REQUIRE(ClassifyGameFile("Pack/ZsDic.pack.zs") == GameFileKind::ZstdDictionaries);
    for (const char* path : {"Effect/Sub/Fx.vsetb.byml", "Model/Fx.vsetb.byml.zs",
                             "Effect/Fx.vsetb.byml", "Effect/Sub/Fx.esetb.byml.zs",
                             "Model/Tex.bntx.zs", "Model/Mesh.bfres.zs", "Asset/assets_a000.dat",
                             "Audio/Sound.pck", "Bin/Nvn2-upper_left.ndsr", "Shader/notes.shn.txt",
                             "Shader/bnsh"}) {
        INFO(path);
        REQUIRE(ClassifyGameFile(path) == GameFileKind::Other);
    }
}

TEST_CASE("ShaderPrecache: a particle effect without shader programs is not an error",
          "[video_core]") {
    std::vector<u8> effect(64, 0);
    std::memcpy(effect.data(), "VFXB", 4);
    size_t records = 0;
    size_t errors = 0;
    REQUIRE_NOTHROW(ShaderWarm::WalkArchive(
        effect, "Magic.ptcl", nullptr, [&](const ShaderWarm::ShaderRecord&) { ++records; },
        [&](const std::string&) { ++errors; }));
    REQUIRE(records == 0);
    REQUIRE(errors == 0);
}

TEST_CASE("ShaderPrecache: programs that cannot be lowered are refused", "[video_core]") {
    ShaderWarm::LoweredCode lowered;
    REQUIRE(ShaderWarm::LowerProgram({}, lowered).has_value());
    const std::array<u8, 15> partial{};
    REQUIRE(ShaderWarm::LowerProgram(partial, lowered).has_value());
    // One instruction with no exit: not a complete program.
    const std::array<u8, 16> no_exit{};
    REQUIRE(ShaderWarm::LowerProgram(no_exit, lowered).has_value());
    REQUIRE(lowered.code.empty());
}

namespace {

void PutLe32(std::vector<u8>& buf, size_t offset, u32 value) {
    std::memcpy(buf.data() + offset, &value, sizeof(value));
}

std::vector<u8> MakeControl(u32 stage_type, u32 declared, u32 allocation) {
    std::vector<u8> control(0xe0, 0);
    constexpr std::array<u8, 8> magic{0x34, 0x12, 0x76, 0x98, 0xc9, 0x00, 0x00, 0x00};
    std::memcpy(control.data(), magic.data(), magic.size());
    PutLe32(control, 56, declared);
    PutLe32(control, 68, allocation);
    PutLe32(control, 84, stage_type << 16);
    PutLe32(control, 88, 8);
    PutLe32(control, 92, 8);
    PutLe32(control, 96, 1);
    return control;
}

std::vector<u8> MakeSampleNdsr() {
    struct Entry {
        u32 type;
        std::vector<u8> control;
        std::vector<u8> code;
    };
    std::vector<Entry> entries;
    {
        std::vector<u8> vcode(128 + 32, 0);
        PutLe32(vcode, 0, 1U << 10); // GraphicsStages[0] = Vertex
        entries.push_back({0, MakeControl(0, 32, 128 + 32), std::move(vcode)});
    }
    {
        std::vector<u8> fcode(128 + 48, 0);
        PutLe32(fcode, 0, 5U << 10); // GraphicsStages[4] = Fragment
        entries.push_back({3, MakeControl(1, 48, 128 + 48), std::move(fcode)});
    }
    {
        std::vector<u8> ccode(64, 0);
        ccode[0] = 0x19;
        entries.push_back({6, MakeControl(5, 48, 64), std::move(ccode)});
    }

    // 1 metadata item (type 11) + 6 control/code items = 7 items
    const size_t item_count = 1 + entries.size() * 2;
    const size_t table_end = 0x14 + item_count * 8;
    std::vector<u8> out(table_end + 16, 0);
    std::memcpy(out.data(), "NDSR\x00\x01\x01\x00", 8);
    PutLe32(out, 0x0c, static_cast<u32>(item_count));
    PutLe32(out, 0x14, static_cast<u32>(table_end));
    PutLe32(out, 0x18, (16U << 4) | 11U);

    size_t idx = 1;
    for (const auto& entry : entries) {
        const u32 ctrl_off = static_cast<u32>(out.size());
        out.insert(out.end(), entry.control.begin(), entry.control.end());
        const u32 code_off = static_cast<u32>(out.size());
        out.insert(out.end(), entry.code.begin(), entry.code.end());
        PutLe32(out, 0x14 + idx * 8, ctrl_off);
        PutLe32(out, 0x14 + idx * 8 + 4,
                (static_cast<u32>(entry.control.size()) << 4) | entry.type);
        PutLe32(out, 0x14 + (idx + 1) * 8, code_off);
        PutLe32(out, 0x14 + (idx + 1) * 8 + 4,
                (static_cast<u32>(entry.code.size()) << 4) | (entry.type + 1));
        idx += 2;
    }
    return out;
}

} // namespace

TEST_CASE("ShaderPrecache: NDSR, NDVE and AAMP containers walk cleanly", "[video_core]") {
    const std::vector<u8> ndsr = MakeSampleNdsr();
    std::vector<ShaderWarm::ProgramStage> ndsr_stages;
    std::vector<size_t> ndsr_sizes;
    size_t errors = 0;
    ShaderWarm::WalkArchive(
        ndsr, "agl_ndsr.pck", nullptr,
        [&](const ShaderWarm::ShaderRecord& r) {
            ndsr_stages.push_back(r.stage);
            ndsr_sizes.push_back(r.code.size());
        },
        [&](const std::string&) { ++errors; });
    REQUIRE(errors == 0);
    REQUIRE(ndsr_stages == std::vector<ShaderWarm::ProgramStage>{
                               ShaderWarm::ProgramStage::Vertex,
                               ShaderWarm::ProgramStage::Fragment,
                               ShaderWarm::ProgramStage::Compute});
    REQUIRE(ndsr_sizes[2] == 48); // Compute uses declared size (48), not padded allocation (64)

    // Build an NDVE wrapping the NDSR + one table (2 compute shaders) + a 1ALR trailer.
    std::vector<u8> ndve(0x28, 0);
    std::memcpy(ndve.data(), "NDVE\x05\x00\xf0\x0f", 8);
    PutLe32(ndve, 0x0c, static_cast<u32>(ndsr.size()));
    ndve.insert(ndve.end(), ndsr.begin(), ndsr.end());

    const size_t table_start = ndve.size();
    const u32 table_hdr_size = 8 + 16 * 1;
    ndve.resize(table_start + table_hdr_size, 0);
    PutLe32(ndve, table_start, 1); // 1 stage
    PutLe32(ndve, table_start + 4, 0);
    const auto table_ctrl = MakeControl(5, 64, 80);
    const std::vector<u8> table_code(80, 0x2a);
    const u32 to_ctrl = static_cast<u32>(ndve.size() - table_start);
    ndve.insert(ndve.end(), table_ctrl.begin(), table_ctrl.end());
    const u32 to_code = static_cast<u32>(table_ctrl.size());
    ndve.insert(ndve.end(), table_code.begin(), table_code.end());
    PutLe32(ndve, table_start + 8, to_ctrl);
    PutLe32(ndve, table_start + 12, to_code);
    PutLe32(ndve, table_start + 16, static_cast<u32>(table_code.size()));

    const u32 rla_rel = static_cast<u32>(ndve.size() - table_start);
    ndve.resize(ndve.size() + 8, 0);
    PutLe32(ndve, ndve.size() - 8, rla_rel);
    std::memcpy(ndve.data() + ndve.size() - 4, "1ALR", 4);
    PutLe32(ndve, 0x10, static_cast<u32>(ndve.size() - table_start));

    std::vector<ShaderWarm::ProgramStage> ndve_stages;
    std::vector<size_t> ndve_sizes;
    ShaderWarm::WalkArchive(
        ndve, "Ndve-test.dat", nullptr,
        [&](const ShaderWarm::ShaderRecord& r) {
            ndve_stages.push_back(r.stage);
            ndve_sizes.push_back(r.code.size());
        },
        [&](const std::string&) { ++errors; });
    REQUIRE(errors == 0);
    REQUIRE(ndve_stages.size() == 4);
    REQUIRE(ndve_stages[3] == ShaderWarm::ProgramStage::Compute);
    REQUIRE(ndve_sizes[3] == 64);

    // AAMP parameter files are skipped quietly.
    std::vector<u8> aamp(32, 0);
    std::memcpy(aamp.data(), "AAMP\x02\x00\x00\x00", 8);
    size_t aamp_records = 0;
    ShaderWarm::WalkArchive(
        aamp, "gsys.bgmsconf", nullptr, [&](const ShaderWarm::ShaderRecord&) { ++aamp_records; },
        [&](const std::string&) { ++errors; });
    REQUIRE(aamp_records == 0);
    REQUIRE(errors == 0);

    // Truncated NDSR header throws ArchiveError cleanly.
    REQUIRE_THROWS_AS(
        ShaderWarm::WalkArchive(
            std::vector<u8>{'N', 'D', 'S', 'R', 0, 0, 0, 0}, "bad.pck", nullptr,
            [&](const ShaderWarm::ShaderRecord&) {}, [&](const std::string&) {}),
        ShaderWarm::ArchiveError);
}
