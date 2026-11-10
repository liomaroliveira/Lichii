// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/nx2_list_draw.h"

namespace {

// One macro instruction, as TuringMacroInterpreter::Decode reads its 96 bits.
struct Lane {
    u32 op{};       ///< 0 = add
    u32 dst{24};    ///< 24 = none
    u32 a{24};      ///< 24 = zero, 28/29 = the next parameters
    u32 b{24};
    u32 imm{};
    u32 method{};   ///< Output selectors: 0 none, 1/2 ALU, 3/4 loads, 5/6 immediates
    u32 emit{};     ///< ... 8/9 immediate >> 12
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
    // The interpreter reads an instruction's words last to first.
    code.push_back(bits[2]);
    code.push_back(bits[1]);
    code.push_back(bits[0]);
}

constexpr u32 Zero = 24;
constexpr u32 Load0 = 28;

// The array draw macro of Ocarina of Time: draw.begin = 0, vertex_buffer.first = parameter 1,
// vertex_buffer.count = parameter 0, draw.end = 0.
std::vector<u32> ArrayDraw(u32 first_register = 0x135d, u32 count_source = 0) {
    std::vector<u32> code;
    Append(code, {.lane0 = {.dst = 0, .a = Load0, .imm = 0x586, .method = 5, .emit = 8},
                  .lane1 = {.imm = first_register, .method = 6, .emit = 4},
                  .end = true});
    Append(code, {.lane0 = {.a = count_source, .imm = 0x585, .emit = 1},
                  .lane1 = {.method = 5, .emit = 8}});
    return code;
}

} // namespace

TEST_CASE("List draw: the array draw macro is recognised", "[video_core]") {
    const auto code = ArrayDraw();
    // What the interpreter makes of it.
    Tegra::TuringMacroInterpreter interpreter;
    std::vector<u32> parameters{36, 1200};
    std::vector<std::pair<u32, u32>> emits;
    const auto result = interpreter.Execute(
        code, 0, parameters, [](u32) { return 0U; },
        [&emits](u32 address, u32 value) {
            emits.emplace_back(address, value);
            return true;
        });
    REQUIRE(result.success);
    REQUIRE(result.consumed == 2);
    REQUIRE((emits == std::vector<std::pair<u32, u32>>{
                          {0x586, 0}, {0x35d, 1200}, {0x35e, 36}, {0x585, 0}}));
    REQUIRE(Tegra::Nx2ListDraw::IsArrayDrawMacro(code, 0));
}

TEST_CASE("List draw: other macros are refused", "[video_core]") {
    using Tegra::Nx2ListDraw::IsArrayDrawMacro;
    // The second parameter goes to another register.
    REQUIRE_FALSE(IsArrayDrawMacro(ArrayDraw(0x135c), 0));
    // The count is a register the macro never set, then zero instead of parameter 0.
    REQUIRE_FALSE(IsArrayDrawMacro(ArrayDraw(0x135d, 1), 0));
    REQUIRE_FALSE(IsArrayDrawMacro(ArrayDraw(0x135d, Zero), 0));
    // The count is masked (and with an immediate).
    {
        std::vector<u32> code;
        Append(code, {.lane0 = {.dst = 0, .a = Load0, .imm = 0x586, .method = 5, .emit = 8},
                      .lane1 = {.imm = 0x135d, .method = 6, .emit = 4},
                      .end = true});
        Append(code, {.lane0 = {.op = 12, .a = 0, .b = 25, .imm = 0x585, .emit = 1},
                      .lane1 = {.method = 5, .emit = 8}});
        REQUIRE_FALSE(IsArrayDrawMacro(code, 0));
    }
    // A predicate on the count.
    {
        std::vector<u32> code;
        Append(code, {.lane0 = {.dst = 0, .a = Load0, .imm = 0x586, .method = 5, .emit = 8},
                      .lane1 = {.imm = 0x135d, .method = 6, .emit = 4},
                      .end = true});
        Append(code, {.lane0 = {.a = 0, .imm = 0x585, .emit = 1},
                      .lane1 = {.method = 5, .emit = 8},
                      .pred_mode = 1,
                      .pred_reg = 0});
        REQUIRE_FALSE(IsArrayDrawMacro(code, 0));
    }
    // A register read (op 22) in place of the count.
    {
        std::vector<u32> code;
        Append(code, {.lane0 = {.dst = 0, .a = Load0, .imm = 0x586, .method = 5, .emit = 8},
                      .lane1 = {.imm = 0x135d, .method = 6, .emit = 4},
                      .end = true});
        Append(code, {.lane0 = {.op = 22, .a = 0, .imm = 0x585, .emit = 1},
                      .lane1 = {.method = 5, .emit = 8}});
        REQUIRE_FALSE(IsArrayDrawMacro(code, 0));
    }
    // Code that ends before the macro does.
    {
        auto code = ArrayDraw();
        code.resize(3);
        REQUIRE_FALSE(IsArrayDrawMacro(code, 0));
    }
}
