// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include <boost/icl/interval_set.hpp>
#include "common/assert.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include <bit>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::A64 {

using namespace Backend::X64;

static RunCodeCallbacks GenRunCodeCallbacks(A64::UserCallbacks* cb, CodePtr (*LookupBlock)(void* lookup_block_arg), void* arg, const A64::UserConfig& conf, bool shared) {
    if (shared) {
        return RunCodeCallbacks{
            ArgCallback::FromJitState(reinterpret_cast<u64>(LookupBlock), offsetof(A64JitState, jit_impl)),
            DevirtualizeFromJitState<&A64::UserCallbacks::AddTicks>(cb, offsetof(A64JitState, callbacks)),
            DevirtualizeFromJitState<&A64::UserCallbacks::GetTicksRemaining>(cb, offsetof(A64JitState, callbacks)),
            conf.enable_cycle_counting,
        };
    }
    return RunCodeCallbacks{
        ArgCallback(LookupBlock, reinterpret_cast<u64>(arg)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::AddTicks>(cb)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::GetTicksRemaining>(cb)),
        conf.enable_cycle_counting,
    };
}

static std::function<void(BlockOfCode&)> GenRCP(const A64::UserConfig& conf) {
    return [conf](BlockOfCode& code) {
        if (conf.page_table) {
            code.mov(code.r14, std::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.mov(code.r13, *conf.fastmem_pointer);
        }
    };
}

static Optimization::PolyfillOptions GenPolyfillOptions(const BlockOfCode& code) {
    return Optimization::PolyfillOptions{
        .sha256 = !code.HasHostFeature(HostFeature::SHA),
        .vector_multiply_widen = true,
    };
}

struct SharedCodeCache final {
    SharedCodeCache(UserConfig conf, bool is_shared, CodePtr (*LookupBlock)(void*), void* single_core_arg)
        : conf(conf)
        , is_shared(is_shared)
        , block_of_code(GenRunCodeCallbacks(conf.callbacks, LookupBlock, single_core_arg, conf, is_shared),
                        JitStateInfo{dummy_state}, conf.code_cache_size, GenRCP(conf))
        , emitter(block_of_code, conf, nullptr, is_shared)
        , polyfill_options(GenPolyfillOptions(block_of_code))
    {
        ASSERT(conf.page_table_address_space_bits >= 12 && conf.page_table_address_space_bits <= 64);
    }

    CodePtr LookupOrCompile(IR::LocationDescriptor descriptor, UserCallbacks* caller_cb, IR::Block& local_ir_block, bool* need_clear_out = nullptr) {
        if (is_shared) {
            std::shared_lock slk{cache_mutex};
            if (!emitter.HasPendingFastmemRecompiles()) {
                if (auto block = emitter.GetBasicBlock(descriptor)) {
                    return block->entrypoint;
                }
            }
        } else {
            if (auto block = emitter.GetBasicBlock(descriptor)) {
                return block->entrypoint;
            }
        }

        // Translate + Optimize outside any cache lock using caller's IR::Block + code-page cache.
        const auto get_code = [caller_cb](u64 vaddr) { return caller_cb->MemoryReadCode(vaddr); };
        auto const arch_descriptor = A64::LocationDescriptor{descriptor};
        local_ir_block.Reset(arch_descriptor);
        A64::Translate(local_ir_block, arch_descriptor, get_code, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});
        Optimization::Optimize(local_ir_block, conf, polyfill_options);

        std::unique_lock lk{cache_mutex, std::defer_lock};
        if (is_shared) {
            lk.lock();
        }
        if (auto invalidated = emitter.DrainPendingFastmemRecompiles(); !invalidated.empty()) {
            RecordInvalidatedLocked(invalidated);
            if (is_shared) {
                HaltAllCoresLocked();
            }
        }
        if (auto block = emitter.GetBasicBlock(descriptor)) {
            return block->entrypoint;
        }
        constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        if (block_of_code.SpaceRemaining() < MINIMUM_REMAINING_CODESIZE) {
            if (!is_shared) {
                if (need_clear_out) {
                    *need_clear_out = true;
                }
                return nullptr;
            }
            if (active_runs.load(std::memory_order_acquire) == 0) {
                ClearAllUnderLock();
            } else {
                pending_full_clear = true;
                HaltAllCoresLocked();
                return block_of_code.GetReturnFromRunCodeAddress();
            }
        }
        block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);

        return emitter.Emit(local_ir_block).entrypoint;
    }

    void RecordInvalidatedLocked(const std::vector<IR::LocationDescriptor>& locs) {
        for (const auto& loc : locs) {
            ++epoch_seq;
            invalidation_ring[epoch_seq & (INVALIDATION_RING_SIZE - 1)] = {epoch_seq, loc.Value()};
        }
        cache_epoch.store(epoch_seq, std::memory_order_release);
    }

    void ClearAllUnderLock() {
        block_of_code.ClearCache();
        emitter.ClearCache();
        pending_full_clear = false;
        clear_all_seq = ++epoch_seq;
        cache_epoch.store(epoch_seq, std::memory_order_release);
    }

    void InvalidateSharedRange(u64 start_address, size_t length) {
        std::lock_guard lk{cache_mutex};
        boost::icl::interval_set<u64> ranges;
        const auto end_address = static_cast<u64>(start_address + length - 1);
        ranges.add(boost::icl::discrete_interval<u64>::closed(start_address, end_address));
        if (auto invalidated = emitter.InvalidateCacheRanges(ranges); !invalidated.empty()) {
            RecordInvalidatedLocked(invalidated);
            HaltAllCoresLocked();
        }
    }

    void RequestSharedClear() {
        std::lock_guard lk{cache_mutex};
        if (active_runs.load(std::memory_order_acquire) == 0) {
            ClearAllUnderLock();
        } else {
            pending_full_clear = true;
            HaltAllCoresLocked();
        }
    }

    void TryFinishPendingClear() {
        if (!pending_full_clear) {
            return;
        }
        std::lock_guard lk{cache_mutex};
        if (pending_full_clear && active_runs.load(std::memory_order_acquire) == 0) {
            ClearAllUnderLock();
        }
    }

    void RegisterCore(volatile u32* halt_ptr) {
        std::lock_guard lk{cache_mutex};
        cores_halt.push_back(halt_ptr);
    }

    void UnregisterCore(volatile u32* halt_ptr) {
        std::lock_guard lk{cache_mutex};
        std::erase(cores_halt, halt_ptr);
    }

    void HaltAllCoresLocked() {
        for (volatile u32* halt_ptr : cores_halt) {
            Atomic::Or(halt_ptr, static_cast<u32>(HaltReason::CacheInvalidation));
        }
    }

    static constexpr size_t INVALIDATION_RING_SIZE = 4096;
    static constexpr u64 MAX_INCREMENTAL_SYNC = 256;
    struct InvalidationRecord {
        u64 seq = 0;
        u64 loc = 0;
    };

    A64JitState dummy_state{};
    const UserConfig conf;
    const bool is_shared;
    BlockOfCode block_of_code;
    A64EmitX64 emitter;
    IR::Block ir_block = {LocationDescriptor(0, FP::FPCR(0), false)};
    Optimization::PolyfillOptions polyfill_options;
    std::shared_mutex cache_mutex;
    std::atomic<u64> cache_epoch{0};
    u64 epoch_seq = 0;
    u64 clear_all_seq = 0;
    std::array<InvalidationRecord, INVALIDATION_RING_SIZE> invalidation_ring{};
    std::atomic<u32> active_runs{0};
    bool pending_full_clear = false;
    std::vector<volatile u32*> cores_halt;
};

struct Jit::Impl final {
public:
    Impl(Jit*, UserConfig conf)
        : conf(conf)
        , cache(conf.shared_cache ? conf.shared_cache
                                  : std::make_shared<SharedCodeCache>(conf, false, &GetCurrentBlockThunk, this))
    {
        if (cache->is_shared && conf.HasOptimization(OptimizationFlag::FastDispatch)) {
            fast_dispatch_table.resize(A64EmitX64::fast_dispatch_table_size);
        }
        InitJitStatePerCore();
        if (cache->is_shared) {
            cache->RegisterCore(&jit_state.halt_reason);
        }
    }

    ~Impl() {
        if (cache->is_shared) {
            cache->UnregisterCore(&jit_state.halt_reason);
        }
    }

    HaltReason Run() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        SyncEpoch();
        is_executing = true;
        cache->active_runs.fetch_add(1, std::memory_order_acq_rel);
        const CodePtr current_code_ptr = [this] {
            // RSB optimization
            const u32 new_rsb_ptr = (jit_state.rsb_ptr - 1) & A64JitState::RSB_PTR_MASK;
            if (jit_state.GetUniqueHash() == jit_state.rsb_location_descriptors[new_rsb_ptr]) {
                jit_state.rsb_ptr = new_rsb_ptr;
                return CodePtr(jit_state.rsb_codeptrs[new_rsb_ptr]);
            }
            return CodePtr((uintptr_t(GetCurrentBlock()) + 15) & ~uintptr_t(15));
        }();
        const HaltReason hr = cache->block_of_code.RunCode(&jit_state, current_code_ptr);
        cache->active_runs.fetch_sub(1, std::memory_order_acq_rel);
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        return hr;
    }

    HaltReason Step() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        SyncEpoch();
        is_executing = true;
        cache->active_runs.fetch_add(1, std::memory_order_acq_rel);
        const HaltReason hr = cache->block_of_code.StepCode(&jit_state, GetCurrentSingleStep());
        cache->active_runs.fetch_sub(1, std::memory_order_acq_rel);
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        return hr;
    }

    void ClearCache() {
        if (cache->is_shared) {
            cache->RequestSharedClear();
            return;
        }
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(u64 start_address, size_t length) {
        if (cache->is_shared) {
            cache->InvalidateSharedRange(start_address, length);
            return;
        }
        std::unique_lock lock{invalidation_mutex};
        const auto end_address = static_cast<u64>(start_address + length - 1);
        const auto range = boost::icl::discrete_interval<u64>::closed(start_address, end_address);
        invalid_cache_ranges.add(range);
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!is_executing);
        jit_state = {};
        InitJitStatePerCore();
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&jit_state.halt_reason, static_cast<u32>(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&jit_state.halt_reason, ~static_cast<u32>(hr));
    }

    u64 GetSP() const {
        return jit_state.sp;
    }

    void SetSP(u64 value) {
        jit_state.sp = value;
    }

    u64 GetPC() const {
        return jit_state.pc;
    }

    void SetPC(u64 value) {
        jit_state.pc = value;
    }

    u64 GetRegister(size_t index) const {
        if (index == 31)
            return GetSP();
        return jit_state.reg.at(index);
    }

    void SetRegister(size_t index, u64 value) {
        if (index == 31)
            return SetSP(value);
        jit_state.reg.at(index) = value;
    }

    std::array<u64, 31> GetRegisters() const {
        return jit_state.reg;
    }

    void SetRegisters(const std::array<u64, 31>& value) {
        jit_state.reg = value;
    }

    Vector GetVector(size_t index) const {
        return {jit_state.vec.at(index * 2), jit_state.vec.at(index * 2 + 1)};
    }

    void SetVector(size_t index, Vector value) {
        jit_state.vec.at(index * 2) = value[0];
        jit_state.vec.at(index * 2 + 1) = value[1];
    }

    std::array<Vector, 32> GetVectors() const {
        std::array<Vector, 32> ret;
        static_assert(sizeof(ret) == sizeof(jit_state.vec));
        std::memcpy(ret.data(), jit_state.vec.data(), sizeof(jit_state.vec));
        return ret;
    }

    void SetVectors(const std::array<Vector, 32>& value) {
        static_assert(sizeof(value) == sizeof(jit_state.vec));
        std::memcpy(jit_state.vec.data(), value.data(), sizeof(jit_state.vec));
    }

    u32 GetFpcr() const {
        return jit_state.GetFpcr();
    }

    void SetFpcr(u32 value) {
        jit_state.SetFpcr(value);
    }

    u32 GetFpsr() const {
        return jit_state.GetFpsr();
    }

    void SetFpsr(u32 value) {
        jit_state.SetFpsr(value);
    }

    u32 GetPstate() const {
        return jit_state.GetPstate();
    }

    void SetPstate(u32 value) {
        jit_state.SetPstate(value);
    }

    void ClearExclusiveState() {
        jit_state.exclusive_state = 0;
    }

    bool IsExecuting() const {
        return is_executing;
    }

    std::string Disassemble() const {
        const size_t size = reinterpret_cast<const char*>(cache->block_of_code.getCurr()) - reinterpret_cast<const char*>(cache->block_of_code.GetCodeBegin());
        auto const* p = reinterpret_cast<const char*>(cache->block_of_code.GetCodeBegin());
        return Common::DisassembleX64(p, p + size);
    }

private:
    void InitJitStatePerCore() {
        jit_state.callbacks = conf.callbacks;
        jit_state.tpidrro_el0 = conf.tpidrro_el0;
        jit_state.tpidr_el0 = conf.tpidr_el0;
        jit_state.jit_impl = this;
        jit_state.global_monitor = conf.global_monitor;
        jit_state.processor_id = conf.processor_id;
        jit_state.exclusive_address_ptr = conf.global_monitor ? GetExclusiveMonitorAddressPointer(conf.global_monitor, conf.processor_id) : nullptr;
        jit_state.exclusive_value_ptr = conf.global_monitor ? GetExclusiveMonitorValuePointer(conf.global_monitor, conf.processor_id) : nullptr;
        jit_state.fast_dispatch_table = fast_dispatch_table.empty() ? nullptr : fast_dispatch_table.data();
    }

    void SyncEpoch() {
        if (!cache->is_shared) {
            return;
        }
        const u64 epoch = cache->cache_epoch.load(std::memory_order_acquire);
        if (local_epoch == epoch) {
            return;
        }
        std::shared_lock slk{cache->cache_mutex};
        const u64 cur_seq = cache->epoch_seq;
        if (local_epoch == cur_seq) {
            return;
        }
        jit_state.ResetRSB();
        if (!fast_dispatch_table.empty()) {
            const bool full_wipe = local_epoch < cache->clear_all_seq ||
                                   (cur_seq - local_epoch) > SharedCodeCache::MAX_INCREMENTAL_SYNC;
            if (full_wipe) {
                std::fill(fast_dispatch_table.begin(), fast_dispatch_table.end(), A64EmitX64::FastDispatchEntry{});
            } else if (cache->emitter.shared_fast_dispatch_lookup) {
                auto* table = fast_dispatch_table.data();
                for (u64 s = local_epoch + 1; s <= cur_seq; ++s) {
                    const auto& rec = cache->invalidation_ring[s & (SharedCodeCache::INVALIDATION_RING_SIZE - 1)];
                    if (rec.seq == s) {
                        auto& entry = (*cache->emitter.shared_fast_dispatch_lookup)(rec.loc, table);
                        if (entry.location_descriptor == rec.loc) {
                            entry = {};
                        }
                    }
                }
            }
        }
        local_epoch = cur_seq;
    }

    static CodePtr GetCurrentBlockThunk(void* thisptr) {
        Jit::Impl* this_ = static_cast<Jit::Impl*>(thisptr);
        this_->SyncEpoch();
        return this_->GetCurrentBlock();
    }

    IR::LocationDescriptor GetCurrentLocation() const {
        return IR::LocationDescriptor{jit_state.GetUniqueHash()};
    }

    CodePtr GetCurrentBlock() {
        return GetBlock(GetCurrentLocation());
    }

    CodePtr GetCurrentSingleStep() {
        return GetBlock(A64::LocationDescriptor{GetCurrentLocation()}.SetSingleStepping(true));
    }

    CodePtr GetBlock(IR::LocationDescriptor descriptor) {
        bool need_clear = false;
        if (CodePtr entry = cache->LookupOrCompile(descriptor, conf.callbacks, ir_block, &need_clear)) {
            return entry;
        }
        if (need_clear) {
            invalidate_entire_cache = true;
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
            return cache->LookupOrCompile(descriptor, conf.callbacks, ir_block, nullptr);
        }
        return nullptr;
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            if (cache->is_shared) {
                ClearHalt(HaltReason::CacheInvalidation);
                cache->TryFinishPendingClear();
                SyncEpoch();
                return;
            }
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (!invalidate_entire_cache && invalid_cache_ranges.empty()) {
                return;
            }

            jit_state.ResetRSB();
            if (invalidate_entire_cache) {
                cache->block_of_code.ClearCache();
                cache->emitter.ClearCache();
            } else {
                cache->emitter.InvalidateCacheRanges(invalid_cache_ranges);
            }
            invalid_cache_ranges.clear();
            invalidate_entire_cache = false;
        }
    }

    const UserConfig conf;
    IR::Block ir_block = {LocationDescriptor(0, FP::FPCR(0), false)};
    A64JitState jit_state;
    std::shared_ptr<SharedCodeCache> cache;
    std::vector<A64EmitX64::FastDispatchEntry> fast_dispatch_table;
    u64 local_epoch = 0;
    bool is_executing = false;
    bool invalidate_entire_cache = false;
    boost::icl::interval_set<u64> invalid_cache_ranges;
    std::mutex invalidation_mutex;

    friend std::shared_ptr<SharedCodeCache> MakeSharedCodeCache(UserConfig conf);
};

std::shared_ptr<SharedCodeCache> MakeSharedCodeCache(UserConfig conf) {
    return std::make_shared<SharedCodeCache>(conf, true, &Jit::Impl::GetCurrentBlockThunk, nullptr);
}

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Jit::Impl>(this, conf)) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

void Jit::InvalidateCacheRange(u64 start_address, size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

u64 Jit::GetSP() const {
    return impl->GetSP();
}

void Jit::SetSP(u64 value) {
    impl->SetSP(value);
}

u64 Jit::GetPC() const {
    return impl->GetPC();
}

void Jit::SetPC(u64 value) {
    impl->SetPC(value);
}

u64 Jit::GetRegister(size_t index) const {
    return impl->GetRegister(index);
}

void Jit::SetRegister(size_t index, u64 value) {
    impl->SetRegister(index, value);
}

std::array<u64, 31> Jit::GetRegisters() const {
    return impl->GetRegisters();
}

void Jit::SetRegisters(const std::array<u64, 31>& value) {
    impl->SetRegisters(value);
}

Vector Jit::GetVector(size_t index) const {
    return impl->GetVector(index);
}

void Jit::SetVector(size_t index, Vector value) {
    impl->SetVector(index, value);
}

std::array<Vector, 32> Jit::GetVectors() const {
    return impl->GetVectors();
}

void Jit::SetVectors(const std::array<Vector, 32>& value) {
    impl->SetVectors(value);
}

u32 Jit::GetFpcr() const {
    return impl->GetFpcr();
}

void Jit::SetFpcr(u32 value) {
    impl->SetFpcr(value);
}

u32 Jit::GetFpsr() const {
    return impl->GetFpsr();
}

void Jit::SetFpsr(u32 value) {
    impl->SetFpsr(value);
}

u32 Jit::GetPstate() const {
    return impl->GetPstate();
}

void Jit::SetPstate(u32 value) {
    impl->SetPstate(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

bool Jit::IsExecuting() const {
    return impl->IsExecuting();
}

std::string Jit::Disassemble() const {
    return impl->Disassemble();
}

}  // namespace Dynarmic::A64
