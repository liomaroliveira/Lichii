// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "common/oot_menu_bridge.h"
#include "video_core/renderer_vulkan/present/vk_oot_ui.h"

namespace Vulkan::OotUi {

class OotMenuController {
public:
    OotMenuController();

    /// Returns true if any overlay (full menu, compact live bar, or corner FPS HUD) needs rendering
    /// on this frame. When false, the caller can skip ImGui completely (`Idle()`).
    [[nodiscard]] bool ShouldDraw(const Common::OotMenu::Bridge& bridge) const;

    /// Processes queued keyboard/mouse/gamepad navigation inputs and renders the active overlays.
    void TickAndDraw(Ctx& c);

    // Row tables are built by a free function in vk_oot_menu.cpp.
    enum class Tab : int {
        Graphics = 0,
        Performance = 1,
        Controls = 2,
        AudioUi = 3,
        Count = 4,
    };

    enum class RowKind : int {
        Heading = 0,
        Choice,
        Toggle,
        Slider,
        Action,
    };

    enum class HitKind : int {
        Tab = 0,
        TabPrev,
        TabNext,
        Row,
        SliderTrack,
        ApplyRestart,
        BackToCompact,
        DropItem,
    };

    struct HitBox {
        float x0, y0, x1, y1;
        HitKind kind;
        int index;
        float vx0, vx1;
        float tx0, tx1;
    };

    struct RowDef {
        Tab tab{};
        RowKind kind{RowKind::Choice};
        const char* id{};
        const char* label{};
        const char* help{};
        bool requires_restart{false};
        int lo{0};
        int hi{1};
        int step{1};
    };

    struct RestartBaseline {
        bool captured{false};
        int ocarina_fsr_render{};
        bool ocarina_fsr{};
        int resolution_setup{};
        int use_docked_mode{};
        int frame_rate_limit{};
        bool occlusion_mod{};
        bool disk_shader_cache{};
        int precache_shaders{};
        bool read_areas{};
        bool gamemode{};
        bool pin_ccx{};
        int sink_id{};
    };

private:
    void CaptureBaselineOnce(const Common::OotMenu::PcOptions& opts);
    [[nodiscard]] bool IsRowPendingRestart(const char* id,
                                           const Common::OotMenu::PcOptions& opts) const;
    [[nodiscard]] std::vector<const char*> CollectPendingRowNames(
        const Common::OotMenu::PcOptions& opts) const;

    [[nodiscard]] int ReadRowValue(const char* id, const Common::OotMenu::PcOptions& opts) const;
    void WriteRowValue(const char* id, int val, Common::OotMenu::PcOptions& opts,
                       Common::OotMenu::Bridge& bridge);
    void TriggerRowAction(const char* id, Common::OotMenu::Bridge& bridge);
    [[nodiscard]] std::string FormatRowValue(const RowDef& r, int val,
                                             const Common::OotMenu::TelemetrySnapshot& tel) const;
    [[nodiscard]] bool IsRowDisabled(const char* id,
                                     const Common::OotMenu::TelemetrySnapshot& tel) const;

    [[nodiscard]] std::vector<const RowDef*> RowsForTab(Tab t) const;
    void ClampSelection(const std::vector<const RowDef*>& tab_rows);
    void MoveSelection(int delta, const std::vector<const RowDef*>& tab_rows);
    void AdjustRow(const RowDef& r, int delta, Common::OotMenu::Bridge& bridge);

    // Drop-down list for Choice rows (T6a): A opens, up/down picks, A applies, B closes.
    void OpenDropdown(const RowDef& r, const Common::OotMenu::PcOptions& opts);
    void CloseDropdown();
    void ApplyDropdown(Common::OotMenu::Bridge& bridge);
    void MoveDropdown(int delta);
    void DrawDropdown(Ctx& c, ImVec2 anchor_a, ImVec2 anchor_b, bool prefer_up,
                      const Common::OotMenu::TelemetrySnapshot& tel,
                      const Common::OotMenu::PcOptions& opts);
    [[nodiscard]] bool DropdownOpen() const noexcept {
        return dd_row_ != nullptr;
    }

    void ProcessInput(Common::OotMenu::Bridge& bridge, float dt_sec);
    void ProcessPointer(Common::OotMenu::Bridge& bridge, const std::vector<const RowDef*>& tab_rows);

    void DrawFullMenu(Ctx& c, Common::OotMenu::Bridge& bridge);
    void DrawCompactBar(Ctx& c, Common::OotMenu::Bridge& bridge);
    void DrawControlsBindings(Ctx& c, float x0, float x1, float top, float bottom,
                              const Common::OotMenu::PadBindingSnapshot& snap);
    void DrawFpsHud(Ctx& c, const Common::OotMenu::Bridge& bridge, bool menu_open, bool compact);

    // EDEN_NX2_OOT_MENU_SELFTEST=1 (T6d): writes every value of every row, reads it back, logs
    // PASS/FAIL and restores it; then cycles the tabs, a drop-down, the compact bar and the HUD
    // (8 s each, labelled) for the harness's screenshots.
    void RunValueSelfTest(Common::OotMenu::Bridge& bridge);
    void DriveSelfTest(Ctx& c, Common::OotMenu::Bridge& bridge);

    RestartBaseline baseline_{};
    Tab cur_tab_{Tab::Graphics};
    std::array<int, static_cast<int>(Tab::Count)> tab_sel_{1, 1, 1, 1};
    std::array<float, static_cast<int>(Tab::Count)> scroll_at_{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, static_cast<int>(Tab::Count)> scroll_want_{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<int, static_cast<int>(Tab::Count)> last_scrolled_{0, 0, 0, 0};

    std::string confirm_action_id_;
    int hover_row_{-1};
    int hover_tab_{-1};
    bool hover_apply_{false};
    bool hover_compact_{false};
    float menu_ox_{0.0f};
    int dragging_slider_row_{-1};
    std::vector<HitBox> hits_;

    int rebind_pick_{0}; // Switch button chosen in "Trocar um botão" (T6b)
    const RowDef* dd_row_{nullptr};
    int dd_sel_{0};
    int dd_scroll_{0};
    int dd_hover_{-1};

    u64 prev_pad_mask_{0};
    int held_dir_{0}; // 1=up, 2=down, 3=left, 4=right
    float dir_hold_timer_{0.0f};
    float dir_repeat_timer_{0.0f};
    std::chrono::steady_clock::time_point last_tick_{};

    std::vector<float> graph_frames_;
    std::chrono::steady_clock::time_point selftest_start_{};
    bool selftest_values_done_{false};
    int selftest_phase_{-1};
    int selftest_saved_hud_{-1};
};

} // namespace Vulkan::OotUi
