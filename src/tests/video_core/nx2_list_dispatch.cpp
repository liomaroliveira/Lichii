// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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

} // namespace

TEST_CASE("List dispatch recognizes the macro that stores a grid", "[list_dispatch]") {
    CHECK(ListDispatch::IsStoreGridMacro(GameCode, StoreGrid));
}

TEST_CASE("List dispatch stores what the macro stores", "[list_dispatch]") {
    Tegra::TuringMacroInterpreter interpreter;
    std::vector<u32> parameters{60, 270, 1};
    const auto result = interpreter.Execute(
        GameCode, StoreGrid, parameters, [](u32) { return 0U; }, [](u32, u32) { return true; });
    REQUIRE(result.success);
    CHECK(result.consumed == 3);
    CHECK(interpreter.dram[ListDispatch::DramGridX] == 60);
    CHECK(interpreter.dram[ListDispatch::DramGridY] == 270);
    CHECK(interpreter.dram[ListDispatch::DramGridZ] == 1);
}

TEST_CASE("List dispatch refuses other macros", "[list_dispatch]") {
    // Reads the grid back and writes methods.
    CHECK_FALSE(ListDispatch::IsStoreGridMacro(GameCode, UploadQmd));
    // The second half of the macro alone: one word stored.
    CHECK_FALSE(ListDispatch::IsStoreGridMacro(GameCode, StoreGrid + 1));
    // Code that ends before the macro does.
    CHECK_FALSE(ListDispatch::IsStoreGridMacro(std::span<const u32>{GameCode}.first(3), StoreGrid));
    CHECK_FALSE(ListDispatch::IsStoreGridMacro(std::span<const u32>{}, StoreGrid));
    // Past the code.
    CHECK_FALSE(ListDispatch::IsStoreGridMacro(GameCode, 17));
}

TEST_CASE("List dispatch refuses a macro that stores the grid elsewhere", "[list_dispatch]") {
    // The immediate of the first store is the data RAM address (0x20 in the game's macro).
    std::vector<u32> code(GameCode.begin(), GameCode.end());
    const auto stores = [&code] { return ListDispatch::IsStoreGridMacro(code, StoreGrid); };
    REQUIRE(stores());
    // Lane 0's immediate is bits 30 to 45 of the instruction: its low two bits end word 2
    // (the instruction's first 32 bits), the rest starts word 1.
    code[2] ^= 0x40000000U; // 0x20 -> 0x21
    CHECK_FALSE(stores());
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
