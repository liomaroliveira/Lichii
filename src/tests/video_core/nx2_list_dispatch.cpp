// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/nx2_list_dispatch.h"

namespace {

namespace ListDispatch = Tegra::Nx2ListDispatch;

// Ocarina of Time's compute macros, as its command stream uploads them: the macro that stores a
// dispatch's grid (two instructions), then the one that uploads the QMD with that grid.
constexpr std::array<u32, 51> GameCode{
    0x00000087, 0xb9c7c008, 0x399c7f01, 0x00000003, 0x18c00008, 0xb99c7f00,
    0x00000087, 0x190f8008, 0x31917b00, 0x00000003, 0x1c280008, 0xb1907b00,
    0x00000420, 0xb81c0040, 0x0b824300, 0x00540840, 0xb8c40418, 0x318c0300,
    0x30a00003, 0x18c00000, 0x718c0300, 0x6a600107, 0x18c0001b, 0x318c0300,
    0x001801b7, 0x18c00000, 0xf04c5f00, 0x40600003, 0x18c00000, 0x318c0300,
    0x40600003, 0x18c00000, 0x318c0300, 0x20200003, 0x01c00000, 0x302c0300,
    0x60400003, 0x00c00000, 0xf03c5f00, 0x40600003, 0x18c00000, 0x318c0300,
    0x40600003, 0x18c00000, 0x318c0300, 0x00000003, 0x18c00000, 0x318c0301,
    0x00000003, 0x18c00000, 0x318c0300,
};
constexpr u32 StoreGrid = 0;
constexpr u32 UploadQmd = 2;
// Where the game's macro stores the grid: found by running it, used here to check the result.
constexpr ListDispatch::GridCells GameCells{0x20, 0x21, 0x22};

} // namespace

TEST_CASE("List dispatch recognizes the macro that stores a grid", "[list_dispatch]") {
    const auto cells = ListDispatch::StoreGridCells(GameCode, StoreGrid);
    REQUIRE(cells.has_value());
    CHECK(*cells == GameCells);
}

TEST_CASE("List dispatch stores what the macro stores", "[list_dispatch]") {
    Tegra::TuringMacroInterpreter interpreter;
    std::vector<u32> parameters{60, 270, 1};
    const auto result = interpreter.Execute(
        GameCode, StoreGrid, parameters, [](u32) { return 0U; }, [](u32, u32) { return true; });
    REQUIRE(result.success);
    CHECK(result.consumed == 3);
    CHECK(interpreter.dram[GameCells[0]] == 60);
    CHECK(interpreter.dram[GameCells[1]] == 270);
    CHECK(interpreter.dram[GameCells[2]] == 1);
}

TEST_CASE("List dispatch refuses other macros", "[list_dispatch]") {
    // Reads the grid back and writes methods.
    using ListDispatch::StoreGridCells;
    CHECK_FALSE(StoreGridCells(GameCode, UploadQmd));
    // The second half of the macro alone: one word stored.
    CHECK_FALSE(StoreGridCells(GameCode, StoreGrid + 1));
    // Code that ends before the macro does.
    CHECK_FALSE(StoreGridCells(std::span<const u32>{GameCode}.first(3), StoreGrid));
    CHECK_FALSE(StoreGridCells(std::span<const u32>{}, StoreGrid));
    // Past the code.
    CHECK_FALSE(StoreGridCells(GameCode, 17));
}

TEST_CASE("List dispatch finds the grid wherever a macro stores it", "[list_dispatch]") {
    // The immediate of the first store is the data RAM address (0x20 in the game's macro).
    std::vector<u32> code(GameCode.begin(), GameCode.end());
    const auto cells = [&code] { return ListDispatch::StoreGridCells(code, StoreGrid); };
    REQUIRE(cells() == GameCells);
    // Lane 0's immediate is bits 30 to 45 of the instruction: its low two bits end word 2
    // (the instruction's first 32 bits), the rest starts word 1. Moved to 0x24, x is stored
    // elsewhere: the same macro for another layout.
    code[1] ^= 0x00000001U; // 0x20 -> 0x24
    const auto moved = cells();
    REQUIRE(moved.has_value());
    CHECK(*moved == ListDispatch::GridCells{0x24, 0x21, 0x22});
    // x stored over y: two parameters to one word, no grid.
    code[1] ^= 0x00000001U;
    code[2] ^= 0x40000000U; // 0x20 -> 0x21
    CHECK_FALSE(cells());
}

TEST_CASE("List dispatch marker", "[list_dispatch]") {
    using ListDispatch::IsMarker;
    CHECK(IsMarker(ListDispatch::MarkerX, ListDispatch::MarkerY, ListDispatch::MarkerZ));
    // The QMD's y and z are 16 bits: what is above them is not the grid.
    CHECK(IsMarker(ListDispatch::MarkerX, ListDispatch::MarkerY | 0x10000,
                   ListDispatch::MarkerZ | 0xABCD0000));
    CHECK_FALSE(IsMarker(1, 1, 1));
    CHECK_FALSE(IsMarker(ListDispatch::MarkerX, 1, 1));
    CHECK_FALSE(IsMarker(0, ListDispatch::MarkerY, ListDispatch::MarkerZ));
    // No host dispatches a grid that large, so no game's grid is the marker.
    CHECK(ListDispatch::MarkerX > 0xffffff);
}

TEST_CASE("List dispatch: the grid words are read back over the marker", "[list_dispatch]") {
    // Cells that are no game's in particular.
    constexpr ListDispatch::GridCells Cells{0x105, 0x40, 0x7};
    Tegra::TuringMacroInterpreter::DataRam ram{};
    CHECK_FALSE(ListDispatch::HasMarker(ram, Cells));
    ram[Cells[0]] = ListDispatch::MarkerX;
    ram[Cells[1]] = ListDispatch::MarkerY;
    CHECK_FALSE(ListDispatch::HasMarker(ram, Cells));
    ram[Cells[2]] = ListDispatch::MarkerZ;
    CHECK(ListDispatch::HasMarker(ram, Cells));

    // No reader: the marker stays, and the failure is counted.
    ListDispatch::GridMarker marker{{}, Cells, true};
    CHECK(marker.In(ram));
    const u64 failed = ListDispatch::g_guard.failed.load();
    CHECK_FALSE(ListDispatch::ResolveGrid(marker, ram));
    CHECK(marker.In(ram));
    CHECK(ListDispatch::g_guard.failed.load() == failed + 1);

    struct Source {
        GPUVAddr asked{};
        std::array<u32, 3> words{60, 270, 1};
    } source;
    const ListDispatch::ListReader reader{
        0x2272dce010,
        [](void* context, GPUVAddr address, std::span<u32> words) {
            auto& from = *static_cast<Source*>(context);
            from.asked = address;
            std::ranges::copy(from.words, words.begin());
            return words.size() == from.words.size();
        },
        &source};
    marker.list = reader;
    CHECK(ListDispatch::ResolveGrid(marker, ram));
    CHECK(source.asked == 0x2272dce010);
    CHECK(ram[Cells[0]] == 60);
    CHECK(ram[Cells[1]] == 270);
    CHECK(ram[Cells[2]] == 1);
    CHECK_FALSE(ListDispatch::HasMarker(ram, Cells));
    CHECK_FALSE(marker.set);
}
