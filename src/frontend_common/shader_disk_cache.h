// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace FrontendCommon {

/// Configures driver shader disk cache defaults (NVIDIA 10 GiB + dedicated dir, Mesa 10 GiB)
/// without overriding explicit user environment variables.
void SetUpShaderDiskCaches();

} // namespace FrontendCommon
