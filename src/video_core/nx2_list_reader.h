// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"

namespace Tegra::Nx2ListDispatch {

/// List draws and list dispatches: the words a command list left on the host GPU, and how
/// the front reads them back when something needs their values after all
/// (DmaPusher::ReadListWords).
struct ListReader {
    GPUVAddr address{};
    bool (*read)(void* context, GPUVAddr address, std::span<u32> words){};
    void* context{};

    [[nodiscard]] bool Read(std::span<u32> words) const {
        return read != nullptr && read(context, address, words);
    }
};

} // namespace Tegra::Nx2ListDispatch
