// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/settings.h"

namespace Common::OotMenu {

constexpr u64 kOcarinaProgramId = 0x0400486024DEC000ULL;

enum class Key : int {
    Up = 0,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
    Confirm,
    Back,
    TabPrev,
    TabNext,
    Clear,
    Esc,
    Count,
};

enum class PromptFace : int {
    Auto = 0,
    Switch = 1,
    Xbox = 2,
    Ps4 = 3,
    Ps5 = 4,
    SteamDeck = 5,
    Keyboard = 6,
    Count = 7,
};

enum class PadLayout : int {
    Nintendo = 0,
    Xbox = 1,
    Custom = 2,
};

enum class MenuLang : int {
    PtBr = 0,
    En = 1,
};

enum class MenuSize : int {
    Auto = 0,
    Small = 1,
    Medium = 2,
    Large = 3,
};

enum class FpsHudLevel : int {
    Off = 0,
    Simple = 1,
    Frametime = 2,
    Graph = 3,
    Detailed = 4,
    Dev = 5,
};

enum class FpsCorner : int {
    TopRight = 0,
    TopLeft = 1,
    BottomRight = 2,
    BottomLeft = 3,
};

enum class MenuHotkey : int {
    Insert = 0,
    F1 = 1,
    Both = 2,
};

enum class CompactHotkey : int {
    RightShift = 0,
    RightCtrl = 1,
    F4 = 2,
};

// Switch button order matching prompt synthesis (A B X Y LS RS L R ZL ZR + - Left Up Right Down)
constexpr std::size_t kSwitchButtonCount = 16;
// Pad input slot order matching prompt glyphs:
// 0 South, 1 East, 2 West, 3 North, 4 Back, 5 Guide, 6 Start, 7 LStick, 8 RStick,
// 9 LShoulder, 10 RShoulder, 11 Up, 12 Down, 13 Left, 14 Right, 15 LTrigger, 16 RTrigger
constexpr int kPadInputCount = 17;

struct KeyEvent {
    Key key;
    bool down;
};

struct PointerSnapshot {
    float x = -1.0f;
    float y = -1.0f;
    bool moved = false;
    u32 held_mask = 0;     // bit 0 = left, bit 1 = right
    u32 pressed_mask = 0;  // rising edges consumed per tick
    float wheel_y = 0.0f;
};

struct PcOptions {
    int menu_size = static_cast<int>(MenuSize::Auto);
    int menu_lang = static_cast<int>(MenuLang::PtBr);
    int fps_hud = static_cast<int>(FpsHudLevel::Off);
    int fps_corner = static_cast<int>(FpsCorner::TopRight);
    int button_prompts = static_cast<int>(PromptFace::Auto);
    int pad_layout = static_cast<int>(PadLayout::Xbox);
    bool keyboard_layer = true;
    bool occlusion_mod = true;
    int idle_spin_us = -1; // -1 = profile/default
    bool read_areas = true;
    bool gamemode = false;
    bool pin_ccx = false;
    int menu_key = static_cast<int>(MenuHotkey::Both);
    int compact_key = static_cast<int>(CompactHotkey::RightShift);
    bool esc_opens = false;
    // E47.T13 (Longshot parity): on top of the emulator's own UI settings
    bool mute_in_background = false;
    bool pause_in_background = false;
};

struct PadBindingSnapshot {
    // Physical pad input (0..16) mapped to each Switch button index (0..15)
    std::array<int, kSwitchButtonCount> pad_input{
        1, 0, 3, 2, 7, 8, 9, 10, 15, 16, 6, 4, 13, 11, 14, 12};
    // Win32-compatible VK code (0x45='E', 0x10=Shift, etc.) matching prompt_keys.py
    std::array<int, kSwitchButtonCount> first_vk{
        0x43, 0x58, 0x56, 0x5A, 0x11, 0x20, 0x51, 0x45, 0x52, 0x54, 0x4E, 0x4D, 0x25, 0x26, 0x27, 0x28};
    std::array<std::string, kSwitchButtonCount> pad_names;
    std::array<std::string, kSwitchButtonCount> key_names;
    std::string active_pad_name;
    u16 active_pad_vid = 0;
    u16 active_pad_pid = 0;
    bool pad_connected = false;
    bool last_input_keyboard = false;
};

struct TelemetrySnapshot {
    double fps = 0.0;
    double base_fps = 0.0;
    double frametime_ms = 0.0;
    double frametime_p99_ms = 0.0;
    double gpu_time_ms = 0.0;
    u32 render_width = 0;
    u32 render_height = 0;
    u32 output_width = 0;
    u32 output_height = 0;
    bool fsr4_available = false;
    bool fsr4_int8 = false;
    bool fsr4_failed = false;
    bool fsr3_supported = true;
    bool hdr_active = false;
    bool has_prompts = false;
    bool needs_restart = false;
    std::string upscaler_name = "Native";
    // T6c performance overlay
    double frametime_min_ms = 0.0;
    double frametime_max_ms = 0.0;
    u64 total_frames = 0;
    u64 stutters = 0;          // frames over 50 ms since boot
    bool host_stats_valid = false;
    double emulation_speed = 0.0; // 1.0 = 100%
    double game_fps = 0.0;        // the guest's own frame rate (PerfStats)
    int shaders_building = 0;
    u64 vram_used = 0;
    u64 vram_budget = 0;
    std::vector<std::string> restart_reasons;
};

struct HostCallbacks {
    std::function<void()> save_config;
    std::function<void()> restart_game;
    std::function<void()> quit_game;
    // Stops the game and keeps the emulator open (back to the game list), without asking (E47.T5)
    std::function<void()> stop_game;
    std::function<void(int layout_mode)> apply_pad_layout;
    std::function<void()> reset_controls;
    std::function<void()> toggle_gamemode;
    // Button-by-button remap (T6b): the host waits for the next press on player 1's controller
    // and maps it to the Switch button `switch_button` (0..15, NativeButton order).
    std::function<void(int switch_button)> begin_rebind;
    std::function<void()> cancel_rebind;
};

/// The running game, for the menu banner (E47.T12): read from its NACP when it starts.
struct GameInfo {
    u64 serial = 0; ///< bumps on every SetGameInfo, so the overlay re-uploads the icon
    u64 program_id = 0;
    std::string title;
    std::string version;
    int icon_width = 0;
    int icon_height = 0;
    std::vector<u8> icon_rgba; ///< icon_width * icon_height * 4, empty without an icon
};

class Bridge {
public:
    static Bridge& Instance();

    void SetProgramId(u64 program_id);
    [[nodiscard]] u64 ProgramId() const noexcept {
        return program_id_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool IsOcarina() const noexcept {
        return ProgramId() == kOcarinaProgramId;
    }

    // Menu visibility and mode
    [[nodiscard]] bool IsOpen() const noexcept {
        return open_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool IsCompact() const noexcept {
        return compact_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool ShouldSuppressGameInput() const noexcept {
        return IsOcarina() && IsOpen();
    }
    void SetOpen(bool open, bool compact = false);
    void ToggleMenu();
    void ToggleCompact();
    void CloseMenu();

    // Hotkey helpers used by Qt and SDL3 windows
    [[nodiscard]] bool MatchesMenuHotkey(bool is_insert, bool is_f1) const;
    [[nodiscard]] bool MatchesCompactHotkey(bool is_rshift, bool is_rctrl, bool is_f4) const;
    [[nodiscard]] bool EscOpensMenu() const;

    // Input from host windows (Qt / SDL3)
    void PushKeyEvent(Key key, bool down);
    void DrainKeyEvents(std::vector<KeyEvent>& out);

    void SetMousePosition(float x, float y);
    void SetMouseButton(int button_index, bool down);
    void AddMouseWheel(float delta_y);
    [[nodiscard]] PointerSnapshot ConsumePointer();

    // Gamepad navigation input from Player 1 EmulatedController (bitmask mirrors Switch NpadButton)
    void UpdateGamepadNav(u64 nav_mask, bool any_pad_connected, bool from_keyboard,
                          std::string_view device_name = {}, u16 vid = 0, u16 pid = 0);
    [[nodiscard]] u64 GamepadNavMask(bool* from_keyboard = nullptr) const;
    [[nodiscard]] bool ConsumeGamepadToggleCombo(u64 raw_buttons);

    // Options persistence (`oot_menu.ini` + `oot-controles/layout.txt`)
    void LoadFromDisk();
    void SaveToDisk() const;
    [[nodiscard]] PcOptions GetOptions() const;
    void SetOptions(const PcOptions& opts, bool bump_prompts = false);

    // Effective idle spin override for PhysicalCore (-1 = let env / default decide)
    [[nodiscard]] int IdleSpinOverrideUs() const noexcept {
        return idle_spin_override_us_.load(std::memory_order_relaxed);
    }
    void RefreshIdleSpinOverride();

    // Live generation counters
    void BumpSettingsGeneration();
    [[nodiscard]] u64 SettingsGeneration() const noexcept {
        return settings_gen_.load(std::memory_order_acquire);
    }
    void BumpPromptsGeneration();
    [[nodiscard]] u64 PromptsGeneration() const noexcept {
        return prompts_gen_.load(std::memory_order_acquire);
    }

    // Pad bindings & device info snapshot for prompts + Controls tab
    void SetPadBindings(const PadBindingSnapshot& snap);
    // Only the readable names in the Controls tab (keeps pad_input / first_vk for prompts)
    void SetBindingNames(const std::array<std::string, kSwitchButtonCount>& pad_names,
                         const std::array<std::string, kSwitchButtonCount>& key_names);

    // Button-by-button remap state (-1 = not capturing)
    void BeginRebind(int switch_button);
    void CancelRebind();
    void FinishRebind(std::string message);
    [[nodiscard]] int RebindTarget() const noexcept {
        return rebind_target_.load(std::memory_order_acquire);
    }
    // Last remap result and how long ago it was set, for a short notice in the menu
    [[nodiscard]] std::string RebindMessage(double* age_sec) const;
    [[nodiscard]] PadBindingSnapshot GetPadBindings() const;
    [[nodiscard]] int EffectivePromptFace() const;

    // Telemetry updated from Vulkan / Present / Rasterizer
    void UpdateRenderStats(u32 render_w, u32 render_h, u32 output_w, u32 output_h,
                           std::string_view upscaler_name, bool fsr4_avail, bool fsr4_int8,
                           bool fsr4_failed, bool fsr3_supported, bool hdr_active);
    void RecordPresentSample(double frame_ms, bool is_generated = false);
    [[nodiscard]] TelemetrySnapshot GetTelemetry() const;
    // The last frames' present-to-present times, oldest first (up to 240), for the graph
    void CopyRecentFrames(std::vector<float>& out) const;
    // From the Qt status bar update (emulation speed, guest FPS, shaders being built)
    void UpdateHostStats(double emulation_speed, double game_fps, int shaders_building);
    // From the present thread when the overlay shows VRAM (VK_EXT_memory_budget)
    void UpdateVramStats(u64 used_bytes, u64 budget_bytes);

    // RomFS fonts + prompts directory discovered at boot
    void SetRomFsFontData(std::vector<u8> body_otf_bytes,
                          std::vector<u8> strong_otf_bytes = {},
                          std::vector<u8> heading_otf_bytes = {});
    [[nodiscard]] std::vector<u8> CopyRomFsBodyFont() const;
    [[nodiscard]] std::vector<u8> CopyRomFsStrongFont() const;
    [[nodiscard]] std::vector<u8> CopyRomFsHeadingFont() const;

    // E47.T12: title, version and icon of the running game (core, when the application changes)
    void SetGameInfo(GameInfo info);
    [[nodiscard]] GameInfo GetGameInfo() const;
    [[nodiscard]] u64 GameInfoSerial() const noexcept {
        return game_info_serial_.load(std::memory_order_acquire);
    }

    void SetPromptsAvailable(bool available, std::string path);
    [[nodiscard]] bool HasPrompts() const noexcept {
        return has_prompts_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::string PromptsPath() const;

    // Host frontend callbacks (registered once by Qt MainWindow or SDL3 loop)
    void RegisterCallbacks(HostCallbacks callbacks);
    void ClearCallbacks();
    void RequestSaveConfig();
    void RequestRestartGame();
    void RequestQuitGame();
    void RequestStopGame();
    void RequestApplyPadLayout(int layout_mode);
    void RequestResetControls();
    void RequestToggleGamemode();

    // Deferred action pump for frontends whose Qt/SDL API must run on main/GUI thread
    void MarkRestartNeeded(std::string_view reason);
    void ClearRestartNeeded();
    void SyncEnvSwitchesSnapshot() const;

private:
    Bridge();
    ~Bridge() = default;

    std::atomic<u64> program_id_{0};
    std::atomic<bool> open_{false};
    std::atomic<bool> compact_{false};
    bool shortcut_session_{false}; ///< started by play/zelda.sh (LICHII_SHORTCUT_*)
    std::atomic<u64> settings_gen_{1};
    std::atomic<u64> prompts_gen_{1};
    std::atomic<int> idle_spin_override_us_{-1};
    std::atomic<bool> has_prompts_{false};
    std::atomic<bool> loaded_from_disk_{false};
    std::atomic<int> rebind_target_{-1};
    std::string rebind_message_;
    std::chrono::steady_clock::time_point rebind_message_at_{};

    mutable std::mutex input_mutex_;
    std::vector<KeyEvent> key_queue_;
    PointerSnapshot pointer_;
    u64 gamepad_nav_mask_ = 0;
    bool gamepad_last_kbd_ = false;
    bool combo_held_ = false;

    mutable std::mutex state_mutex_;
    PcOptions options_;
    PadBindingSnapshot bindings_;
    TelemetrySnapshot telemetry_;
    std::vector<u8> romfs_body_font_;
    std::vector<u8> romfs_strong_font_;
    std::vector<u8> romfs_heading_font_;
    std::string prompts_dir_;
    HostCallbacks callbacks_;
    GameInfo game_info_;
    std::atomic<u64> game_info_serial_{0};

    mutable std::mutex perf_mutex_;
    std::array<double, 240> recent_frame_ms_{};
    std::size_t perf_head_ = 0;
    std::size_t perf_count_ = 0;
    std::chrono::steady_clock::time_point window_start_{};
    u64 window_frames_ = 0;
    u64 window_real_frames_ = 0;
    double last_fps_ = 0.0;
    double last_base_fps_ = 0.0;
    u64 total_frames_ = 0;
    u64 stutters_ = 0;
};

} // namespace Common::OotMenu
