// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/channel_state_cache.inc"
#include "video_core/nx2_back_split.h"

namespace VideoCommon {

ChannelInfo::ChannelInfo(Tegra::Control::ChannelState& channel_state, bool use_side_a)
    : maxwell3d{(VideoCommon::BackSplit::H2Async() && use_side_a)
                    ? channel_state.payload->maxwell_3d
                    : channel_state.RasterizerMaxwell3D()}
    , kepler_compute{channel_state.payload->kepler_compute}
    , gpu_memory{*channel_state.memory_manager}
    , program_id{channel_state.program_id}
  {}

template class VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo>;

} // namespace VideoCommon
