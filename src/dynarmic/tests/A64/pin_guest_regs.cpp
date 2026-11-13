// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <oaknut/oaknut.hpp>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/tests/native/testenv.h"

using namespace Dynarmic;
using namespace oaknut::util;

namespace {

class PinTestEnv final : public A64::UserCallbacks {
public:
    std::vector<u32> code_mem;
    ::Common::unordered_map<u64, u8> mem;
    u64 ticks_left = 10000;
    A64::Jit* jit = nullptr;
    std::function<void(u32)> on_svc;
    std::function<void(A64::VAddr, A64::Exception)> on_exception;

    std::optional<u32> MemoryReadCode(A64::VAddr vaddr) override {
        if ((vaddr & 3) != 0) {
            return std::nullopt;
        }
        const size_t idx = static_cast<size_t>(vaddr / 4);
        if (idx >= code_mem.size()) {
            return std::nullopt;
        }
        return code_mem[idx];
    }

    u64 MemoryRead(A64::VAddr vaddr, size_t size) override {
        u64 val = 0;
        for (size_t i = 0; i < size; ++i) {
            val |= static_cast<u64>(mem[vaddr + i]) << (i * 8);
        }
        return val;
    }

    A64::Vector MemoryRead128(A64::VAddr vaddr) override {
        return {MemoryRead(vaddr, sizeof(u64)), MemoryRead(vaddr + sizeof(u64), sizeof(u64))};
    }

    void MemoryWrite(A64::VAddr vaddr, u64 value, size_t size) override {
        for (size_t i = 0; i < size; ++i) {
            mem[vaddr + i] = static_cast<u8>((value >> (i * 8)) & 0xFF);
        }
    }

    void MemoryWrite128(A64::VAddr vaddr, A64::Vector value) override {
        MemoryWrite(vaddr, value[0], sizeof(u64));
        MemoryWrite(vaddr + sizeof(u64), value[1], sizeof(u64));
    }

    void CallSVC(u32 swi) override {
        if (on_svc) {
            on_svc(swi);
            return;
        }
        if (jit) {
            jit->HaltExecution();
        }
    }

    void ExceptionRaised(A64::VAddr pc, A64::Exception exception) override {
        if (on_exception) {
            on_exception(pc, exception);
            return;
        }
        if (jit) {
            jit->HaltExecution();
        }
    }

    void AddTicks(u64 ticks) override {
        ticks_left = ticks > ticks_left ? 0 : ticks_left - ticks;
    }

    u64 GetTicksRemaining() override {
        return ticks_left;
    }

    u64 GetCNTPCT() override {
        return 0;
    }
};

struct PinResult {
    HaltReason halt = HaltReason{};
    std::array<u64, 31> regs{};
    u64 sp = 0;
    u64 pc = 0;
    u32 pstate = 0;
    u64 ticks_left = 0;

    bool operator==(const PinResult&) const = default;
};

A64::UserConfig MakeEdenLikeConfig(PinTestEnv& env, u32 pin_guest_regs) {
    A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.enable_cycle_counting = false;
    conf.spill_to_free_gpr = true;
    conf.scratch_copy_to_free_reg = true;
    conf.elide_redundant_zext = true;
    conf.fold_address_offsets = true;
    conf.omit_block_frame = true;
    conf.fast_dispatch_table_bits = 16;
    conf.fast_dispatch_set_associative = true;
    conf.rsb_size = 32;
    conf.compact_block_exits = true;
    conf.far_code = true;
    conf.reuse_host_flags = true;
    conf.halt_check_on_back_edges = true;
    conf.host_call_return = true;
    conf.pin_guest_regs = pin_guest_regs;
    return conf;
}

PinResult CaptureResult(const A64::Jit& jit, const PinTestEnv& env, HaltReason halt) {
    return PinResult{
        .halt = halt,
        .regs = jit.GetRegisters(),
        .sp = jit.GetSP(),
        .pc = jit.GetPC(),
        .pstate = jit.GetPstate(),
        .ticks_left = env.ticks_left,
    };
}

}  // namespace

TEST_CASE("A64: pin_guest_regs nested calls, host_call_return, and Step", "[a64][pin_guest_regs]") {
    auto run = [](u32 pin) {
        PinTestEnv env;
        A64::UserConfig conf = MakeEdenLikeConfig(env, pin);
        A64::Jit jit{conf};
        env.jit = &jit;

        oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
        oaknut::Label loop, helper, inner, done;

        code.MOVZ(X19, 6);
        code.l(loop);
        code.BL(helper);
        code.SUBS(X19, X19, 1);
        code.B(NE, loop);
        code.SVC(1);
        code.l(done);
        code.B(done);

        code.l(helper);
        code.STP(X29, X30, SP, PRE_INDEXED, -16);
        code.ADD(X8, X8, X0);
        code.BL(inner);
        code.ADD(X0, X0, 5);
        code.LDP(X29, X30, SP, POST_INDEXED, 16);
        code.RET();

        code.l(inner);
        code.ADD(X0, X0, X8);
        code.ADD(W8, W8, 3);
        code.RET();

        jit.SetPC(0);
        jit.SetSP(0x8000);
        jit.SetRegister(0, 7);
        jit.SetRegister(8, 10);

        // Single-step the first two instructions before full Run() to exercise StepCode with pinned regs.
        HaltReason step1 = HaltReason{};
        CheckedRun([&]() { step1 = jit.Step(); });
        REQUIRE((step1 & HaltReason::Step) != HaltReason{});
        REQUIRE(jit.GetRegister(0) == 7);
        REQUIRE(jit.GetRegister(8) == 10);

        HaltReason hr = HaltReason{};
        CheckedRun([&]() { hr = jit.Run(); });
        return CaptureResult(jit, env, hr);
    };

    const PinResult ref = run(0);
    const PinResult pinned = run(2);
    REQUIRE(pinned == ref);
    REQUIRE(pinned.regs[0] != 7);
    REQUIRE(pinned.regs[8] != 10);
}

TEST_CASE("A64: pin_guest_regs SetW zero-extension and live flags", "[a64][pin_guest_regs]") {
    auto run = [](u32 pin) {
        PinTestEnv env;
        A64::UserConfig conf = MakeEdenLikeConfig(env, pin);
        A64::Jit jit{conf};
        env.jit = &jit;

        oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
        oaknut::Label eq_ok, after_eq;

        // 1. Live flags must survive SetW(R0, 0), while upper 32 bits of X0 must clear.
        code.CMP(W1, W2);
        code.MOV(W0, WZR);
        code.B(EQ, eq_ok);
        code.MOVZ(X9, 0xBAD);
        code.B(after_eq);
        code.l(eq_ok);
        code.MOVZ(X9, 0x111);
        code.l(after_eq);

        // 2. Xmm -> W8 (`IsInXmm` in EmitA64SetW) must zero-extend even with bit 31 = 1.
        code.FMOV(W8, S0);
        code.MOV(X11, X8);

        // 3. Xmm -> X0 (`IsInXmm` in EmitA64SetX) must keep all 64 bits.
        code.FMOV(X0, D1);
        code.MOV(X12, X0);

        // 4. Bit-31-set 32-bit MOV + negative flag -> CSEL must see upper 32 bits == 0 and MI flag intact.
        code.SUBS(W4, WZR, W5);  // negative when W5 == 1
        code.MOV(W8, 0x80000000);
        code.CSEL(X10, X8, X0, MI);

        // 5. Negative s32 immediate to SetX (`FitsInImmediateS32`), plus 32-bit zero to W8 while preserving X13 check.
        code.MOV(X0, -1);
        code.MOV(W8, WZR);
        code.SVC(0);

        jit.SetPC(0);
        jit.SetRegister(0, 0xCAFEBABE11112222ULL);
        jit.SetRegister(8, 0xDEADBEEF33334444ULL);
        jit.SetRegister(1, 42);
        jit.SetRegister(2, 42);
        jit.SetRegister(5, 1);
        jit.SetVector(0, {0xDEADBEEF90000001ULL, 0});
        jit.SetVector(1, {0x123456789ABCDEF0ULL, 0});

        HaltReason hr = HaltReason{};
        CheckedRun([&]() { hr = jit.Run(); });
        return CaptureResult(jit, env, hr);
    };

    const PinResult ref = run(0);
    const PinResult pinned = run(2);
    REQUIRE(pinned == ref);
    REQUIRE(pinned.regs[9] == 0x111);
    REQUIRE(pinned.regs[11] == 0x0000000090000001ULL);
    REQUIRE(pinned.regs[12] == 0x123456789ABCDEF0ULL);
    REQUIRE(pinned.regs[10] == 0x0000000080000000ULL);
    REQUIRE(pinned.regs[0] == 0xFFFFFFFFFFFFFFFFULL);
    REQUIRE(pinned.regs[8] == 0);
}

TEST_CASE("A64: pin_guest_regs CallSVC reads/writes X0 and X8 and halts mid-block", "[a64][pin_guest_regs]") {
    auto run = [](u32 pin) {
        PinTestEnv env;
        A64::UserConfig conf = MakeEdenLikeConfig(env, pin);
        A64::Jit jit{conf};
        env.jit = &jit;

        std::array<u64, 4> svc_seen{};
        env.on_svc = [&](u32 swi) {
            if (swi == 0x29) {
                svc_seen[0] = jit.GetRegister(0);
                svc_seen[1] = jit.GetRegister(8);
                jit.SetRegister(0, 0x1000000000000005ULL);
                jit.SetRegister(8, 0x2000000000000003ULL);
            } else if (swi == 0x2A) {
                svc_seen[2] = jit.GetRegister(0);
                svc_seen[3] = jit.GetRegister(8);
                jit.SetRegister(0, 0xCAFE000000000001ULL);
                jit.SetRegister(8, 0xBEEF000000000002ULL);
                jit.HaltExecution(HaltReason::UserDefined1);
            }
        };

        oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
        code.ADD(X0, X0, 1);
        code.ADD(X8, X8, 2);
        code.SVC(0x29);
        code.ADD(X2, X0, X8);
        code.SUB(W8, W0, W8);
        code.SVC(0x2A);
        code.MOVZ(X3, 99);

        jit.SetPC(0);
        jit.SetRegister(0, 0x1122334455667787ULL);
        jit.SetRegister(8, 0x887766554433220FULL);

        HaltReason hr = HaltReason{};
        CheckedRun([&]() { hr = jit.Run(); });
        return std::pair{CaptureResult(jit, env, hr), svc_seen};
    };

    const auto [ref, ref_seen] = run(0);
    const auto [pinned, pin_seen] = run(2);
    REQUIRE(pin_seen == ref_seen);
    REQUIRE(pinned == ref);
    REQUIRE(pin_seen[0] == 0x1122334455667788ULL);
    REQUIRE(pin_seen[1] == 0x8877665544332211ULL);
    REQUIRE(pinned.regs[2] == 0x3000000000000008ULL);
    REQUIRE(pinned.regs[0] == 0xCAFE000000000001ULL);
    REQUIRE(pinned.regs[8] == 0xBEEF000000000002ULL);
}

TEST_CASE("A64: pin_guest_regs ExceptionRaised reads X0, writes X8, and resumes", "[a64][pin_guest_regs]") {
    auto run = [](u32 pin) {
        PinTestEnv env;
        A64::UserConfig conf = MakeEdenLikeConfig(env, pin);
        A64::Jit jit{conf};
        env.jit = &jit;

        u64 seen_x0 = 0;
        u64 seen_x8_before = 0;
        u64 seen_x8_after = 0;
        env.on_exception = [&](A64::VAddr pc, A64::Exception ex) {
            REQUIRE(ex == A64::Exception::Breakpoint);
            seen_x0 = jit.GetRegister(0);
            seen_x8_before = jit.GetRegisters()[8];
            jit.SetRegister(8, 0x0000000100000002ULL);
            seen_x8_after = jit.GetRegisters()[8];
            jit.SetPC(pc + 4);
        };

        oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
        code.ADD(X0, X0, 4);
        code.ADD(X8, X8, 8);
        code.BRK(1);
        code.ADD(X4, X8, X0);
        code.ADD(X0, X0, 1);
        code.SVC(0);

        jit.SetPC(0);
        jit.SetRegister(0, 0x123456789ABCDEF0ULL);
        jit.SetRegister(8, 0x1111222233334444ULL);

        HaltReason hr = HaltReason{};
        CheckedRun([&]() { hr = jit.Run(); });
        return std::tuple{CaptureResult(jit, env, hr), seen_x0, seen_x8_before, seen_x8_after};
    };

    const auto [ref, ref_x0, ref_x8_b, ref_x8_a] = run(0);
    const auto [pinned, pin_x0, pin_x8_b, pin_x8_a] = run(2);
    REQUIRE(pin_x0 == ref_x0);
    REQUIRE(pin_x8_b == ref_x8_b);
    REQUIRE(pin_x8_a == ref_x8_a);
    REQUIRE(pinned == ref);
    REQUIRE(pinned.regs[4] == 0x123456799ABCDEF6ULL);
    REQUIRE(pinned.regs[0] == 0x123456789ABCDEF5ULL);
}

TEST_CASE("A64: pin_guest_regs non-fastmem page_table lookup with X0 and X8", "[a64][pin_guest_regs]") {
    for (const bool absolute_offset : {false, true}) {
        auto run = [&](u32 pin) {
            PinTestEnv env;
            alignas(16) std::array<u8, 4096> page_buf{};
            std::array<void*, 1024> ptable{};
            constexpr u64 guest_page_vaddr = 0x10000;
            ptable[guest_page_vaddr >> 12] =
                absolute_offset ? reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(page_buf.data()) - guest_page_vaddr)
                                : static_cast<void*>(page_buf.data());

            A64::UserConfig conf = MakeEdenLikeConfig(env, pin);
            conf.fastmem_pointer = std::nullopt;
            conf.page_table = ptable.data();
            conf.page_table_address_space_bits = 22;
            conf.page_table_pointer_mask = 0;
            conf.page_table_log2_stride = 3;
            conf.absolute_offset_page_table = absolute_offset;
            conf.silently_mirror_page_table = true;

            A64::Jit jit{conf};
            env.jit = &jit;

            oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
            code.STR(X8, X0);
            code.ADD(X8, X8, 7);
            code.STR(W8, X0, 8);
            code.MOV(X9, X0);
            code.LDR(W0, X9);
            code.LDR(X8, X9);
            code.LDR(W10, X9, 8);
            code.SVC(0);

            jit.SetPC(0);
            jit.SetRegister(0, guest_page_vaddr);
            jit.SetRegister(8, 0xABCDEF0123456789ULL);

            HaltReason hr = HaltReason{};
            CheckedRun([&]() { hr = jit.Run(); });
            return std::pair{CaptureResult(jit, env, hr), page_buf};
        };

        const auto [ref, ref_page] = run(0);
        const auto [pinned, pin_page] = run(2);
        REQUIRE(pinned == ref);
        REQUIRE(pin_page == ref_page);
        REQUIRE(pinned.regs[0] == 0x23456789ULL);
        REQUIRE(pinned.regs[8] == 0xABCDEF0123456789ULL);
        REQUIRE(pinned.regs[10] == 0x23456790ULL);
    }
}

TEST_CASE("A64: pin_guest_regs halt-on-entry and shared code cache preserve X0 and X8", "[a64][pin_guest_regs]") {
    PinTestEnv env;
    A64::UserConfig conf = MakeEdenLikeConfig(env, 2);
    conf.enable_cycle_counting = true;
    env.ticks_left = 0;

    A64::Jit halt_jit{conf};
    env.jit = &halt_jit;
    env.code_mem.push_back(0xD4000001);  // SVC #0
    halt_jit.SetPC(0);
    halt_jit.SetRegister(0, 0x0123456789ABCDEFULL);
    halt_jit.SetRegister(8, 0xFEDCBA9876543210ULL);
    halt_jit.HaltExecution();
    CheckedRun([&]() { halt_jit.Run(); });
    REQUIRE(halt_jit.GetRegister(0) == 0x0123456789ABCDEFULL);
    REQUIRE(halt_jit.GetRegister(8) == 0xFEDCBA9876543210ULL);

    // SharedCodeCache with two independent JIT instances under pin_guest_regs = 2.
    PinTestEnv shared_env;
    A64::UserConfig shared_conf{};
    shared_conf.callbacks = &shared_env;
    shared_conf.enable_cycle_counting = false;
    shared_conf.omit_block_frame = true;
    shared_conf.pin_guest_regs = 2;

    auto cache = A64::MakeSharedCodeCache(shared_conf);
    shared_conf.shared_cache = cache;

    oaknut::VectorCodeGenerator code{shared_env.code_mem, nullptr};
    code.ADD(X0, X0, X8);
    code.ADD(X8, X8, 10);
    code.SVC(0);

    A64::Jit core0{shared_conf};
    A64::Jit core1{shared_conf};
    core0.SetPC(0);
    core1.SetPC(0);
    core0.SetRegister(0, 100);
    core0.SetRegister(8, 5);
    core1.SetRegister(0, 1000);
    core1.SetRegister(8, 50);

    shared_env.jit = &core0;
    CheckedRun([&]() { core0.Run(); });
    shared_env.jit = &core1;
    CheckedRun([&]() { core1.Run(); });

    REQUIRE(core0.GetRegister(0) == 105);
    REQUIRE(core0.GetRegister(8) == 15);
    REQUIRE(core1.GetRegister(0) == 1050);
    REQUIRE(core1.GetRegister(8) == 60);
}
