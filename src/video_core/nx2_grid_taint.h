// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "video_core/macro_turing.h"
#include "video_core/nx2_list_dispatch.h"

// List dispatches (nx2_list_dispatch.h): where the grid a list dispatch left on the host GPU goes.
//
// While the three data RAM words of a dispatch's grid hold the marker (nx2_list_dispatch.h), the
// values the game's macros would read there are not known to the front. A macro may run on the
// marker only if nothing it does depends on those values, except copying them into the data words
// it uploads (the QMD's grid, where the dispatch finds the marker). DryRun runs a macro call as
// TuringMacroInterpreter::Execute does, on a copy of the data RAM and without side effects, and
// follows the three words through it: every register, output and data RAM word is either clean or
// one of the grid words unchanged. Anything else done with a grid word (arithmetic, a comparison,
// a branch, a loop count, an address, a method number, a store in the data RAM) makes the call
// unsafe: the caller reads the real words first.
namespace Tegra::Nx2GridTaint {

using Word = TuringMacroInterpreter::Word;
using DataRam = TuringMacroInterpreter::DataRam;

struct Emit {
    Word method;
    Word value;
    bool grid; ///< The value is one of the grid words, unchanged
};

struct Result {
    bool completed{};  ///< The call ran to its end as Execute would
    bool safe{};       ///< Nothing but copies and emitted values depends on the grid words
    bool read_grid{};  ///< The call read at least one of the three words
    std::size_t consumed{};
    std::vector<Emit> emits;
    std::string_view reason; ///< Why the call is not completed or not safe
};

/// Runs the macro at `start` of `code` on a copy of `dram`, with `parameters` and register reads
/// through `read`. Emits are collected, not applied. `grid_register` is the first of three
/// registers that hold the grid words when they hold the marker (the engine's copy of the QMD's
/// grid), ~0 for none.
template <typename Read>
[[nodiscard]] Result DryRun(std::span<const Word> code, Word start,
                            std::span<const Word> parameters, const DataRam& dram, Read read,
                            Word grid_register = ~Word{0}) {
    using Decoded = TuringMacroInterpreter::Decoded;
    Result result;
    DataRam ram = dram;
    // The data RAM words that hold a grid word: the three at first, if they hold the marker
    // (words the game's own macro stored there are values like any other).
    const bool marked = Nx2ListDispatch::HasMarker(dram);
    std::array<bool, 3> grid_cell{marked, marked, marked};
    const auto cell_is_grid = [&](Word address) {
        return address >= Nx2ListDispatch::DramGridX && address <= Nx2ListDispatch::DramGridZ &&
               grid_cell[address - Nx2ListDispatch::DramGridX];
    };
    std::array<Word, 24> regs{};
    std::array<bool, 24> reg_grid{};
    Word initialized{}, carry{}, method{}, increment{};
    Word pc = start, loop_count{}, loop_start{}, loop_end{};
    bool end_next{}, stopped{};
    const auto stop = [&](std::string_view reason, bool unsafe) {
        if (result.reason.empty()) {
            result.reason = reason;
        }
        result.safe = result.safe && !unsafe;
    };
    result.safe = true;
    bool failed = false;
    const auto fail = [&](std::string_view reason) {
        stop(reason, false);
        failed = true;
    };
    const auto unsafe = [&](std::string_view reason) {
        stop(reason, true);
        failed = true;
    };
    for (Word steps = 0; steps < 1000000 && !failed; ++steps) {
        if (static_cast<std::size_t>(pc) * 3 + 2 >= code.size()) {
            fail("instruction outside uploaded RAM");
            break;
        }
        const Decoded d = TuringMacroInterpreter::Decode(
            {code[pc * 3 + 2], code[pc * 3 + 1], code[pc * 3]});
        std::array<Word, 2> load{}, alu{};
        std::array<bool, 2> alu_grid{};
        for (Word i = 0; i < 2 && !failed; ++i) {
            if (d.load[i]) {
                if (result.consumed >= parameters.size()) {
                    // Parameters a memory read would append (MME_DMA_READ_FIFOED) are not known
                    // here either.
                    fail("parameter FIFO underflow");
                    break;
                }
                load[i] = parameters[result.consumed++];
            }
        }
        if (failed) {
            break;
        }
        struct Operand {
            Word value;
            bool grid;
            bool zero_source; ///< A constant zero of the code, not a value that happens to be 0
        };
        const auto get = [&](Word reg, Word lane) -> Operand {
            if (reg < 24) {
                if (!(initialized & (1U << reg))) {
                    fail("uninitialized register read");
                }
                return {regs[reg], reg_grid[reg], false};
            }
            switch (reg) {
            case 24:
                return {0, false, true};
            case 25:
                return {Word(std::int32_t(std::int16_t(d.imm[lane]))), false, d.imm[lane] == 0};
            case 26:
                return {Word(std::int32_t(std::int16_t(d.imm[1 - lane]))), false,
                        d.imm[1 - lane] == 0};
            case 27:
                return {(d.imm[0] << 16) | d.imm[1], false, (d.imm[0] | d.imm[1]) == 0};
            case 28:
                return {load[0], false, false};
            case 29:
                return {load[1], false, false};
            default:
                fail("invalid source register");
                return {0, false, false};
            }
        };
        Word mask = 15;
        if (d.pred_mode) {
            const Operand predicate = get(d.pred_reg, 0);
            if (predicate.grid) {
                unsafe("a predicate tests a grid word");
                break;
            }
            mask = predicate.value ? d.mask_true : d.mask_false;
        }
        std::array<Operand, 2> xs{}, ys{};
        for (Word lane = 0; lane < 2; ++lane) {
            if (mask & (1U << lane)) {
                xs[lane] = get(d.src_a[lane], lane);
                ys[lane] = get(d.src_b[lane], lane);
            }
        }
        if (failed) {
            break;
        }
        Word next_pc = pc + 1;
        bool branch_cancels_end{};
        for (Word lane = 0; lane < 2 && !failed; ++lane) {
            if (!(mask & (1U << lane))) {
                continue;
            }
            const Word x = xs[lane].value, y = ys[lane].value;
            const bool any_grid = xs[lane].grid || ys[lane].grid;
            const Word imm = d.imm[lane];
            Word value{};
            bool value_grid{};
            const auto condition = [&](Word comparison) {
                switch (comparison) {
                case 17:
                case 25:
                    return std::int32_t(x) < std::int32_t(y);
                case 18:
                case 26:
                    return x < y;
                case 19:
                case 27:
                    return std::int32_t(x) <= std::int32_t(y);
                case 20:
                case 28:
                    return x <= y;
                default:
                    return x == y;
                }
            };
            const Word op = d.op[lane];
            if (any_grid) {
                // The one thing a grid word may go through: an addition with a constant zero,
                // which is how the macro code moves a value.
                const bool move = op == 0 && ((xs[lane].grid && ys[lane].zero_source) ||
                                              (ys[lane].grid && xs[lane].zero_source));
                if (!move) {
                    unsafe(op == 31   ? "a grid word is stored in the data RAM"
                           : op == 30 ? "a grid word is a data RAM address"
                           : op >= 23 ? "a branch or a loop depends on a grid word"
                                      : "an operation takes a grid word");
                    break;
                }
                value = x + y;
                carry = 0;
                value_grid = true;
            } else {
                switch (op) {
                case 0:
                    value = x + y;
                    carry = value < x;
                    break;
                case 1:
                    value = x + y + carry;
                    break;
                case 2:
                    value = x - y;
                    carry = value > x;
                    break;
                case 3:
                    value = x - y - carry;
                    break;
                case 4:
                case 6: {
                    const std::uint64_t a =
                        op == 4 ? std::uint64_t(std::int64_t(std::int32_t(x))) : x;
                    const std::uint64_t b =
                        op == 4 ? std::uint64_t(std::int64_t(std::int32_t(y))) : y;
                    const auto product = a * b;
                    value = Word(product);
                    carry = Word(product >> 32);
                    break;
                }
                case 5:
                    value = carry;
                    break;
                case 7:
                    if (x != 0x1000) {
                        fail("unsupported extended operation");
                    }
                    break;
                case 8:
                    value = Word(std::countl_zero(x));
                    break;
                case 9:
                    value = x << (y & 31);
                    break;
                case 10:
                    value = x >> (y & 31);
                    break;
                case 11:
                    value = Word(std::int32_t(x) >> (y & 31));
                    break;
                case 12:
                    value = x & y;
                    break;
                case 13:
                    value = ~(x & y);
                    break;
                case 14:
                    value = x | y;
                    break;
                case 15:
                    value = x ^ y;
                    break;
                case 16: {
                    const Word dest = (imm >> 10) & 63, count = (imm >> 5) & 31;
                    const Word source = imm & 31;
                    if (dest >= 32) {
                        fail("invalid merge destination");
                        break;
                    }
                    const Word bits_mask = (1U << count) - 1;
                    value = (x & ~(bits_mask << dest)) | (((y >> source) & bits_mask) << dest);
                    break;
                }
                case 17:
                case 18:
                case 19:
                case 20:
                case 21:
                    value = condition(op) ? ~0U : 0;
                    break;
                case 22: {
                    const Word address = (x + y) & 0xffff;
                    value = read(address);
                    constexpr std::array<Word, 3> marker{Nx2ListDispatch::MarkerX,
                                                         Nx2ListDispatch::MarkerY,
                                                         Nx2ListDispatch::MarkerZ};
                    if (grid_register != ~Word{0} && address >= grid_register &&
                        address < grid_register + 3 && value == marker[address - grid_register]) {
                        value_grid = true;
                        result.read_grid = true;
                    }
                    break;
                }
                case 23:
                    if (loop_count || imm < 2) {
                        fail("invalid loop");
                        break;
                    }
                    if (x == 0) {
                        next_pc = pc + imm;
                        break;
                    }
                    loop_count = x - 1;
                    loop_start = pc;
                    loop_end = pc + imm - 1;
                    break;
                case 24:
                    next_pc = pc + (imm & 0x7fff);
                    break;
                case 25:
                case 26:
                case 27:
                case 28:
                case 29:
                    if (condition(op) == bool(imm & 0x8000)) {
                        const auto offset = std::int32_t(imm << 19) >> 19;
                        if (offset == -4096) {
                            stopped = true;
                        } else {
                            next_pc = Word(std::int32_t(pc) + offset);
                            branch_cancels_end = (imm & 0x4000) != 0;
                        }
                    }
                    break;
                case 30:
                    if (x >= ram.size()) {
                        fail("data RAM read out of bounds");
                        break;
                    }
                    value = ram[x];
                    if (cell_is_grid(x)) {
                        value_grid = true;
                        result.read_grid = true;
                    }
                    break;
                case 31:
                    if (x >= ram.size()) {
                        fail("data RAM write out of bounds");
                        break;
                    }
                    ram[x] = y;
                    if (x >= Nx2ListDispatch::DramGridX && x <= Nx2ListDispatch::DramGridZ) {
                        // A value of the macro's own replaces the grid word.
                        grid_cell[x - Nx2ListDispatch::DramGridX] = false;
                    }
                    break;
                default:
                    break;
                }
            }
            if (failed) {
                break;
            }
            alu[lane] = value;
            alu_grid[lane] = value_grid;
            if (d.dst[lane] < 24) {
                regs[d.dst[lane]] = value;
                reg_grid[d.dst[lane]] = value_grid;
                initialized |= 1U << d.dst[lane];
            } else if (d.dst[lane] != 24) {
                fail("invalid destination register");
            }
        }
        if (failed) {
            break;
        }
        struct Output {
            Word value;
            bool grid;
        };
        const auto output = [&](Word selector) -> Output {
            switch (selector) {
            case 1:
                return {alu[0], alu_grid[0]};
            case 2:
                return {alu[1], alu_grid[1]};
            case 3:
                return {load[0], false};
            case 4:
                return {load[1], false};
            case 5:
                return {d.imm[0], false};
            case 6:
                return {d.imm[1], false};
            case 8:
                return {d.imm[0] >> 12, false};
            case 9:
                return {d.imm[1] >> 12, false};
            case 10:
                return {(d.imm[0] << 16) | d.imm[1], false};
            default:
                fail("invalid output selector");
                return {0, false};
            }
        };
        for (Word lane = 0; lane < 2 && !failed; ++lane) {
            if (!(mask & (4U << lane))) {
                continue;
            }
            if (d.out_method[lane]) {
                const Output address = output(d.out_method[lane]);
                if (address.grid) {
                    unsafe("a grid word is a method number");
                    break;
                }
                method = address.value & 0xfff;
                increment = (address.value >> 12) & 15;
            }
            if (d.out_emit[lane]) {
                const Output value = output(d.out_emit[lane]);
                if (!failed) {
                    result.emits.push_back({method, value.value, value.grid});
                }
                method += increment;
            }
        }
        if (failed) {
            break;
        }
        if (end_next || stopped) {
            result.completed = true;
            return result;
        }
        end_next = d.end && !branch_cancels_end;
        if (loop_count && pc == loop_end) {
            --loop_count;
            next_pc = loop_start + 1;
        }
        pc = next_pc;
    }
    if (result.reason.empty()) {
        result.reason = "instruction limit exceeded";
    }
    return result;
}

/// Whether the grid words `result` emitted are a QMD's grid and nothing else. They come three at
/// a time, x then y then z, consecutive: either as data words of an upload (`data_method`), or
/// written to the three registers from `grid_register` (the engine's copy of the QMD's grid).
[[nodiscard]] inline bool EmitsGridOnly(const Result& result, Word data_method,
                                        Word grid_register = ~Word{0}) {
    constexpr std::array<Word, 3> marker{Nx2ListDispatch::MarkerX, Nx2ListDispatch::MarkerY,
                                         Nx2ListDispatch::MarkerZ};
    for (std::size_t i = 0; i < result.emits.size(); ++i) {
        if (!result.emits[i].grid) {
            continue;
        }
        if (i + 3 > result.emits.size()) {
            return false;
        }
        const bool to_registers =
            grid_register != ~Word{0} && result.emits[i].method == grid_register;
        for (std::size_t word = 0; word < 3; ++word) {
            const Emit& emit = result.emits[i + word];
            const Word method = to_registers ? grid_register + static_cast<Word>(word)
                                             : data_method;
            if (!emit.grid || emit.method != method || emit.value != marker[word]) {
                return false;
            }
        }
        i += 2;
    }
    return true;
}

/// Whether a macro call that ran as `result` may run for real on the marker: it completed, used
/// the grid words only as copies, and emitted them only as a QMD's grid.
[[nodiscard]] inline bool MayRunOnMarker(const Result& result, Word data_method,
                                         Word grid_register = ~Word{0}) {
    return result.completed && result.safe && EmitsGridOnly(result, data_method, grid_register);
}

} // namespace Tegra::Nx2GridTaint
