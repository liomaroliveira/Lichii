// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/nx2_grid_taint.h"

namespace {

namespace GridTaint = Tegra::Nx2GridTaint;
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
constexpr u32 DataUpload = 0x6d;

// Where the game's store macro puts the grid (ListDispatch::StoreGridCells finds it).
constexpr ListDispatch::GridCells GameCells{0x20, 0x21, 0x22};

GridTaint::DataRam MarkerRam(const ListDispatch::GridCells& cells = GameCells) {
    GridTaint::DataRam ram{};
    for (u32 word = 0; word < ram.size(); ++word) {
        ram[word] = 0x1000 + word;
    }
    ram[cells[0]] = ListDispatch::MarkerX;
    ram[cells[1]] = ListDispatch::MarkerY;
    ram[cells[2]] = ListDispatch::MarkerZ;
    return ram;
}

GridTaint::Result Run(std::span<const u32> code, u32 start, std::span<const u32> parameters) {
    return GridTaint::DryRun(code, start, parameters, MarkerRam(), GameCells,
                             [](u32) { return 0U; });
}

// One macro instruction, as TuringMacroInterpreter::Decode reads its 96 bits.
struct Lane {
    u32 op{};     ///< 0 = add
    u32 dst{24};  ///< 24 = none
    u32 a{24};    ///< 24 = zero, 25 = this lane's immediate, 28/29 = the next parameters
    u32 b{24};
    u32 imm{};
    u32 method{}; ///< Output selectors: 0 none, 1/2 ALU, 3/4 loads, 5/6 immediates
    u32 emit{};
};

struct Instruction {
    Lane lane0;
    Lane lane1;
    bool end{};
    u32 pred_mode{};
    u32 pred_reg{};
};

void Append(std::vector<u32>& code, const Instruction& instruction) {
    std::array<u32, 3> bits{};
    const auto set = [&bits](u32 low, u32 count, u32 value) {
        for (u32 i = 0; i < count; ++i) {
            if ((value >> i) & 1) {
                bits[(low + i) / 32] |= 1U << ((low + i) % 32);
            }
        }
    };
    set(0, 1, instruction.end ? 1 : 0);
    set(1, 4, instruction.pred_mode);
    set(5, 5, instruction.pred_reg);
    const std::array<const Lane*, 2> lanes{&instruction.lane0, &instruction.lane1};
    for (u32 lane = 0; lane < 2; ++lane) {
        const Lane& l = *lanes[lane];
        set(10 + lane * 36, 5, l.op);
        set(15 + lane * 36, 5, l.dst);
        set(20 + lane * 36, 5, l.a);
        set(25 + lane * 36, 5, l.b);
        set(30 + lane * 36, 16, l.imm);
        set(82 + lane * 7, 3, l.method);
        set(85 + lane * 7, 4, l.emit);
    }
    code.push_back(bits[2]);
    code.push_back(bits[1]);
    code.push_back(bits[0]);
}

constexpr u32 Imm = 25;
constexpr u32 DataRead = 30;
constexpr u32 DataWrite = 31;

// r0 = the grid's x, then `second` with the macro's END, then an empty instruction.
std::vector<u32> AfterGridRead(const Instruction& second) {
    std::vector<u32> code;
    Append(code, {.lane0 = {.op = DataRead, .dst = 0, .a = Imm, .imm = GameCells[0]}});
    Instruction last = second;
    last.end = true;
    Append(code, last);
    Append(code, {});
    return code;
}

} // namespace

TEST_CASE("Grid taint: the game's QMD upload only copies the grid", "[grid_taint]") {
    // Parameter 0 packs the counts: 3 groups of four words before the grid, 1 after, and the
    // upload's line length. Then the upload's two address words and the QMD words themselves.
    std::vector<u32> parameters{(0x50U << 16) | (1U << 8) | 3U, 0x21, 0x145a9800};
    for (u32 word = 0; word < 16; ++word) {
        parameters.push_back(0xA000 + word);
    }
    const auto result = Run(GameCode, UploadQmd, parameters);
    REQUIRE(result.completed);
    CHECK(result.safe);
    CHECK(result.read_grid);
    CHECK(result.consumed == parameters.size());
    std::vector<u32> data;
    std::vector<bool> grid;
    for (const auto& emit : result.emits) {
        if (emit.method == DataUpload) {
            data.push_back(emit.value);
            grid.push_back(emit.grid);
        } else {
            CHECK_FALSE(emit.grid);
        }
    }
    // The QMD's words 12 to 14 are the grid.
    REQUIRE(data.size() == 20);
    CHECK(data[12] == ListDispatch::MarkerX);
    CHECK(data[13] == ListDispatch::MarkerY);
    CHECK(data[14] == ListDispatch::MarkerZ);
    CHECK(std::ranges::count(grid, true) == 3);
    CHECK((grid[12] && grid[13] && grid[14]));
    CHECK(GridTaint::EmitsGridOnly(result, DataUpload));
    CHECK(GridTaint::MayRunOnMarker(result, DataUpload));
    // The same words sent as anything else are not a grid upload.
    CHECK_FALSE(GridTaint::EmitsGridOnly(result, DataUpload + 1));
}

TEST_CASE("Grid taint: a macro that does not read the grid is untouched", "[grid_taint]") {
    const std::vector<u32> parameters{7, 8, 9};
    const auto result = Run(GameCode, StoreGrid, parameters);
    REQUIRE(result.completed);
    CHECK(result.safe);
    CHECK_FALSE(result.read_grid);
    CHECK(result.emits.empty());
    CHECK(GridTaint::MayRunOnMarker(result, DataUpload));
}

TEST_CASE("Grid taint: a moved grid word emitted elsewhere is safe but no grid upload",
          "[grid_taint]") {
    const auto code = AfterGridRead({.lane0 = {.a = 0, .imm = 0x44, .method = 5, .emit = 1}});
    const auto result = Run(code, 0, {});
    REQUIRE(result.completed);
    CHECK(result.safe);
    CHECK(result.read_grid);
    REQUIRE(result.emits.size() == 1);
    CHECK(result.emits[0].method == 0x44);
    CHECK(result.emits[0].value == ListDispatch::MarkerX);
    CHECK(result.emits[0].grid);
    CHECK_FALSE(GridTaint::MayRunOnMarker(result, DataUpload));
}

TEST_CASE("Grid taint: anything computed from a grid word is unsafe", "[grid_taint]") {
    const auto unsafe = [](const Instruction& second) {
        const auto result = Run(AfterGridRead(second), 0, {});
        return !result.completed && !result.safe && result.read_grid &&
               !GridTaint::MayRunOnMarker(result, DataUpload);
    };
    // grid + 1
    CHECK(unsafe({.lane0 = {.op = 0, .dst = 1, .a = 0, .b = Imm, .imm = 1}}));
    // grid & mask
    CHECK(unsafe({.lane0 = {.op = 12, .dst = 1, .a = 0, .b = Imm, .imm = 0xff}}));
    // grid == 0, as a value
    CHECK(unsafe({.lane0 = {.op = 21, .dst = 1, .a = 0}}));
    // a branch on the grid
    CHECK(unsafe({.lane0 = {.op = 29, .a = 0, .imm = 0x8002}}));
    // a loop the grid counts
    CHECK(unsafe({.lane0 = {.op = 23, .a = 0, .imm = 2}}));
    // the grid stored elsewhere in the data RAM
    CHECK(unsafe({.lane0 = {.op = DataWrite, .a = Imm, .b = 0, .imm = 0x30}}));
    // the grid as a data RAM address
    CHECK(unsafe({.lane0 = {.op = DataRead, .dst = 1, .a = 0}}));
    // the grid as a register number to read
    CHECK(unsafe({.lane0 = {.op = 22, .dst = 1, .a = 0}}));
    // the grid as a method number
    CHECK(unsafe({.lane0 = {.a = 0, .method = 1}}));
    // a predicate on the grid
    CHECK(unsafe({.lane0 = {.dst = 1, .a = Imm, .imm = 5}, .pred_mode = 1, .pred_reg = 0}));
}

TEST_CASE("Grid taint: a parameter that happens to be zero is no move", "[grid_taint]") {
    // grid + parameter: the parameter is 0 here, another call's is not.
    auto code = AfterGridRead({.lane0 = {.op = 0, .dst = 1, .a = 0, .b = 28}});
    const std::vector<u32> parameters{0};
    const auto result = Run(code, 0, parameters);
    CHECK_FALSE(result.safe);
    // grid + the code's constant 0 is a move.
    code = AfterGridRead({.lane0 = {.op = 0, .dst = 1, .a = 0, .b = Imm, .imm = 0}});
    const auto moved = Run(code, 0, {});
    CHECK(moved.completed);
    CHECK(moved.safe);
}

TEST_CASE("Grid taint: a macro's own value replaces the grid word", "[grid_taint]") {
    // The macro stores 5 at the grid's x, then reads it back and adds 1.
    std::vector<u32> code;
    Append(code, {.lane0 = {.op = DataWrite, .a = Imm, .b = 28, .imm = GameCells[0]}});
    Append(code, {.lane0 = {.op = DataRead, .dst = 0, .a = Imm, .imm = GameCells[0]}});
    Append(code, {.lane0 = {.op = 0, .dst = 1, .a = 0, .b = Imm, .imm = 1, .method = 5, .emit = 1},
                  .end = true});
    Append(code, {});
    const std::vector<u32> parameters{5};
    const auto result = Run(code, 0, parameters);
    REQUIRE(result.completed);
    CHECK(result.safe);
    CHECK_FALSE(result.read_grid);
    REQUIRE(result.emits.size() == 1);
    CHECK(result.emits[0].value == 6);
    CHECK_FALSE(result.emits[0].grid);
}

TEST_CASE("Grid taint: a call that cannot be followed is not run on the marker", "[grid_taint]") {
    // Parameters missing: what a memory read would append is not known in a dry run.
    const std::vector<u32> parameters{(0x50U << 16) | (1U << 8) | 3U, 0x21};
    const auto result = Run(GameCode, UploadQmd, parameters);
    CHECK_FALSE(result.completed);
    CHECK_FALSE(GridTaint::MayRunOnMarker(result, DataUpload));
    // Code that ends before the macro does.
    const auto cut = Run(std::span<const u32>{GameCode}.first(9), UploadQmd, parameters);
    CHECK_FALSE(cut.completed);
}

TEST_CASE("Grid taint: the grid copied to the QMD's registers is a grid copy", "[grid_taint]") {
    constexpr u32 GridRegister = 0xc8 + 12;
    // r0, r1, r2 = the grid; then x, y, z to the three registers (the method counts up).
    std::vector<u32> code;
    Append(code, {.lane0 = {.op = DataRead, .dst = 0, .a = Imm, .imm = GameCells[0]},
                  .lane1 = {.op = DataRead, .dst = 1, .a = Imm, .imm = GameCells[1]}});
    Append(code, {.lane0 = {.op = DataRead, .dst = 2, .a = Imm, .imm = GameCells[2]},
                  .lane1 = {.imm = 0x1000 | GridRegister, .method = 6}});
    Append(code, {.lane0 = {.a = 0, .emit = 1}, .lane1 = {.a = 1, .emit = 2}});
    Append(code, {.lane0 = {.a = 2, .emit = 1}, .end = true});
    Append(code, {});
    const auto result = Run(code, 0, {});
    REQUIRE(result.completed);
    CHECK(result.safe);
    REQUIRE(result.emits.size() == 3);
    CHECK(result.emits[0].method == GridRegister);
    CHECK(result.emits[2].method == GridRegister + 2);
    CHECK(GridTaint::MayRunOnMarker(result, DataUpload, GridRegister));
    // Not a grid copy for an engine that keeps no such registers, or keeps them elsewhere.
    CHECK_FALSE(GridTaint::MayRunOnMarker(result, DataUpload));
    CHECK_FALSE(GridTaint::MayRunOnMarker(result, DataUpload, GridRegister + 1));
}

TEST_CASE("Grid taint: a grid register read back is a grid word", "[grid_taint]") {
    constexpr u32 GridRegister = 0xc8 + 12;
    const auto registers = [](u32 address) {
        return address == GridRegister ? ListDispatch::MarkerX : 0U;
    };
    const auto run = [&](const Instruction& second) {
        std::vector<u32> code;
        // r0 = the register's value (MME_READ, op 22).
        Append(code, {.lane0 = {.op = 22, .dst = 0, .a = Imm, .imm = GridRegister}});
        Instruction last = second;
        last.end = true;
        Append(code, last);
        Append(code, {});
        return GridTaint::DryRun(code, 0, {}, MarkerRam(), GameCells, registers, GridRegister);
    };
    // Emitted as it is, elsewhere: followed, and not a grid copy.
    const auto moved = run({.lane0 = {.a = 0, .imm = 0x44, .method = 5, .emit = 1}});
    REQUIRE(moved.completed);
    CHECK(moved.read_grid);
    REQUIRE(moved.emits.size() == 1);
    CHECK(moved.emits[0].grid);
    CHECK_FALSE(GridTaint::MayRunOnMarker(moved, DataUpload, GridRegister));
    // Computed with: unsafe.
    const auto added = run({.lane0 = {.op = 0, .dst = 1, .a = 0, .b = Imm, .imm = 4}});
    CHECK_FALSE(added.safe);
    // A register that does not hold the marker is a plain value.
    std::vector<u32> code;
    Append(code, {.lane0 = {.op = 22, .dst = 0, .a = Imm, .imm = GridRegister + 1}});
    Append(code, {.lane0 = {.op = 0, .dst = 1, .a = 0, .b = Imm, .imm = 4}, .end = true});
    Append(code, {});
    const auto plain =
        GridTaint::DryRun(code, 0, {}, MarkerRam(), GameCells, registers, GridRegister);
    CHECK(plain.completed);
    CHECK(plain.safe);
    CHECK_FALSE(plain.read_grid);
}

TEST_CASE("Grid taint: a grid the game stored itself is a plain value", "[grid_taint]") {
    // The same data RAM without the marker: the upload macro's grid words are values like any
    // other, whatever is done with them.
    auto ram = MarkerRam();
    ram[GameCells[0]] = 160;
    ram[GameCells[1]] = 90;
    ram[GameCells[2]] = 1;
    std::vector<u32> parameters{(0x50U << 16) | (1U << 8) | 3U, 0x21, 0x145a9800};
    for (u32 word = 0; word < 16; ++word) {
        parameters.push_back(0xA000 + word);
    }
    const auto result =
        GridTaint::DryRun(GameCode, UploadQmd, parameters, ram, GameCells, [](u32) { return 0U; });
    REQUIRE(result.completed);
    CHECK(result.safe);
    CHECK_FALSE(result.read_grid);
    CHECK(std::ranges::none_of(result.emits, &GridTaint::Emit::grid));
    CHECK(GridTaint::MayRunOnMarker(result, DataUpload));
    // And arithmetic on them is the game's business.
    std::vector<u32> code;
    Append(code, {.lane0 = {.op = DataRead, .dst = 0, .a = Imm, .imm = GameCells[0]}});
    Append(code, {.lane0 = {.op = 0, .dst = 1, .a = 0, .b = Imm, .imm = 1}, .end = true});
    Append(code, {});
    const auto added = GridTaint::DryRun(code, 0, {}, ram, GameCells, [](u32) { return 0U; });
    CHECK(added.completed);
    CHECK(added.safe);
}

TEST_CASE("Grid taint: the marker is followed wherever it is stored", "[grid_taint]") {
    // The same reads at cells no game in particular uses.
    constexpr ListDispatch::GridCells Cells{0x105, 0x40, 0x7};
    std::vector<u32> code;
    Append(code, {.lane0 = {.op = DataRead, .dst = 0, .a = Imm, .imm = Cells[0]}});
    Append(code, {.lane0 = {.op = 0, .dst = 1, .a = 0, .b = Imm, .imm = 1}, .end = true});
    Append(code, {});
    const auto at = [&](const ListDispatch::GridCells& marked) {
        return GridTaint::DryRun(code, 0, {}, MarkerRam(marked), Cells, [](u32) { return 0U; });
    };
    const auto marked = at(Cells);
    CHECK(marked.read_grid);
    CHECK_FALSE(marked.safe);
    // The same code with the marker elsewhere reads a plain value.
    const auto other = at(GameCells);
    CHECK_FALSE(other.read_grid);
    CHECK(other.safe);
}
