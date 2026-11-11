// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"

namespace Tegra::Nx2ListDispatch {

/// List draws and list dispatches: the words a command list left on the host GPU, and how
/// the front reads them back when something needs their values after all
/// (DmaPusher::ReadListWords): from the copy the back kept at the list's place in the stream
/// (`keep_id`), or, without one, from the list itself.
struct ListReader {
    GPUVAddr address{};
    u64 keep_id{};
    bool (*read)(void* context, GPUVAddr address, u64 keep_id, std::span<u32> words){};
    void* context{};

    [[nodiscard]] bool Read(std::span<u32> words) const {
        return read != nullptr && read(context, address, keep_id, words);
    }
};

} // namespace Tegra::Nx2ListDispatch
