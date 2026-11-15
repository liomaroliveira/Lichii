// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/oot_menu_bridge.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"

namespace Common::OotMenu {

namespace {

std::string ToLowerAscii(std::string_view s) {
    std::string out(s);
    for (char& ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool MenuSaveDisabled() {
    if (const char* v = std::getenv("EDEN_NX2_MENU_NO_SAVE"); v && v[0] == '1') {
        return true;
    }
    if (const char* h = std::getenv("EDEN_NX2_HIDE_WINDOW"); h && h[0] == '1') {
        return true;
    }
    return false;
}

std::filesystem::path ConfigFilePath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "oot_menu.ini";
}

std::filesystem::path LayoutFilePath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "oot-controles" /
           "layout.txt";
}

std::string Trim(std::string_view sv) {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        sv.remove_suffix(1);
    }
    return std::string(sv);
}

bool ParseBool(const std::string& val, bool fallback) {
    const std::string lo = ToLowerAscii(val);
    if (lo == "1" || lo == "true" || lo == "yes" || lo == "on") {
        return true;
    }
    if (lo == "0" || lo == "false" || lo == "no" || lo == "off") {
        return false;
    }
    return fallback;
}

int ParseInt(const std::string& val, int fallback) {
    try {
        std::size_t pos = 0;
        const int parsed = std::stoi(val, &pos);
        return pos > 0 ? parsed : fallback;
    } catch (...) {
        return fallback;
    }
}

void SyncOcclusionAddonSetting(bool occlusion_mod_enabled) {
    auto& disabled = Settings::values.disabled_addons[kOcarinaProgramId];
    static constexpr std::array<std::string_view, 2> kOcclusionNames{
        "Occlusion Off",
        "DisableOcclusion",
    };
    if (occlusion_mod_enabled) {
        std::erase_if(disabled, [&](const std::string& name) {
            return std::find(kOcclusionNames.begin(), kOcclusionNames.end(), name) !=
                   kOcclusionNames.end();
        });
    } else {
        const bool already_disabled = std::ranges::any_of(disabled, [&](const std::string& name) {
            return std::find(kOcclusionNames.begin(), kOcclusionNames.end(), name) !=
                   kOcclusionNames.end();
        });
        if (!already_disabled) {
            disabled.emplace_back("Occlusion Off");
            std::sort(disabled.begin(), disabled.end());
        }
    }
}

bool IsOcclusionAddonEnabledFromSettings() {
    const auto it = Settings::values.disabled_addons.find(kOcarinaProgramId);
    if (it == Settings::values.disabled_addons.end()) {
        return true;
    }
    for (const auto& name : it->second) {
        if (name == "Occlusion Off" || name == "DisableOcclusion") {
            return false;
        }
    }
    return true;
}

} // namespace

Bridge& Bridge::Instance() {
    static Bridge instance;
    return instance;
}

Bridge::Bridge() {
    LoadFromDisk();
}

void Bridge::SetProgramId(u64 program_id) {
    program_id_.store(program_id, std::memory_order_release);
    // EDEN_NX2_OOT_MENU_OPEN=full|compact: open the menu when the game starts (screenshots, tests)
    if (program_id == kOcarinaProgramId) {
        if (const char* open = std::getenv("EDEN_NX2_OOT_MENU_OPEN"); open != nullptr) {
            const std::string_view mode{open};
            if (mode == "full" || mode == "compact") {
                SetOpen(true, mode == "compact");
            }
        }
    }
    if (program_id == kOcarinaProgramId) {
        std::scoped_lock lk{state_mutex_};
        options_.occlusion_mod = IsOcclusionAddonEnabledFromSettings();
    }
}

void Bridge::SetOpen(bool open, bool compact) {
    open_.store(open, std::memory_order_release);
    compact_.store(open ? compact : false, std::memory_order_release);
    if (open) {
        std::scoped_lock lk{state_mutex_};
        options_.occlusion_mod = IsOcclusionAddonEnabledFromSettings();
    }
}

void Bridge::ToggleMenu() {
    const bool was_open = open_.load(std::memory_order_acquire);
    const bool was_compact = compact_.load(std::memory_order_acquire);
    if (was_open && !was_compact) {
        SetOpen(false, false);
    } else {
        SetOpen(true, false);
    }
}

void Bridge::ToggleCompact() {
    const bool was_open = open_.load(std::memory_order_acquire);
    const bool was_compact = compact_.load(std::memory_order_acquire);
    if (was_open && was_compact) {
        SetOpen(false, false);
    } else {
        SetOpen(true, true);
    }
}

void Bridge::CloseMenu() {
    SetOpen(false, false);
}

bool Bridge::MatchesMenuHotkey(bool is_insert, bool is_f1) const {
    if (!IsOcarina()) {
        return false;
    }
    std::scoped_lock lk{state_mutex_};
    switch (static_cast<MenuHotkey>(options_.menu_key)) {
    case MenuHotkey::Insert:
        return is_insert;
    case MenuHotkey::F1:
        return is_f1;
    case MenuHotkey::Both:
    default:
        return is_insert || is_f1;
    }
}

bool Bridge::MatchesCompactHotkey(bool is_rshift, bool is_rctrl, bool is_f4) const {
    if (!IsOcarina()) {
        return false;
    }
    std::scoped_lock lk{state_mutex_};
    switch (static_cast<CompactHotkey>(options_.compact_key)) {
    case CompactHotkey::RightShift:
        return is_rshift;
    case CompactHotkey::RightCtrl:
        return is_rctrl;
    case CompactHotkey::F4:
        return is_f4;
    default:
        return false;
    }
}

bool Bridge::EscOpensMenu() const {
    if (!IsOcarina()) {
        return false;
    }
    std::scoped_lock lk{state_mutex_};
    return options_.esc_opens;
}

void Bridge::PushKeyEvent(Key key, bool down) {
    std::scoped_lock lk{input_mutex_};
    if (key_queue_.size() < 256) {
        key_queue_.push_back({key, down});
    }
}

void Bridge::DrainKeyEvents(std::vector<KeyEvent>& out) {
    std::scoped_lock lk{input_mutex_};
    out.swap(key_queue_);
    key_queue_.clear();
}

void Bridge::SetMousePosition(float x, float y) {
    std::scoped_lock lk{input_mutex_};
    if (pointer_.x != x || pointer_.y != y) {
        pointer_.x = x;
        pointer_.y = y;
        pointer_.moved = true;
    }
}

void Bridge::SetMouseButton(int button_index, bool down) {
    if (button_index < 0 || button_index > 2) {
        return;
    }
    const u32 bit = 1u << static_cast<u32>(button_index);
    std::scoped_lock lk{input_mutex_};
    if (down) {
        if ((pointer_.held_mask & bit) == 0) {
            pointer_.pressed_mask |= bit;
        }
        pointer_.held_mask |= bit;
    } else {
        pointer_.held_mask &= ~bit;
    }
}

void Bridge::AddMouseWheel(float delta_y) {
    std::scoped_lock lk{input_mutex_};
    pointer_.wheel_y += delta_y;
}

PointerSnapshot Bridge::ConsumePointer() {
    std::scoped_lock lk{input_mutex_};
    PointerSnapshot snap = pointer_;
    pointer_.moved = false;
    pointer_.pressed_mask = 0;
    pointer_.wheel_y = 0.0f;
    return snap;
}

void Bridge::UpdateGamepadNav(u64 nav_mask, bool any_pad_connected, bool from_keyboard,
                              std::string_view device_name, u16 vid, u16 pid) {
    {
        std::scoped_lock lk{input_mutex_};
        gamepad_nav_mask_ = nav_mask;
        gamepad_last_kbd_ = from_keyboard;
    }
    std::scoped_lock slk{state_mutex_};
    const bool prev_kbd = bindings_.last_input_keyboard;
    const bool prev_conn = bindings_.pad_connected;
    const std::string prev_name = bindings_.active_pad_name;
    bindings_.pad_connected = any_pad_connected;
    bindings_.last_input_keyboard = from_keyboard || !any_pad_connected;
    if (!device_name.empty()) {
        bindings_.active_pad_name = std::string(device_name);
    }
    bindings_.active_pad_vid = vid;
    bindings_.active_pad_pid = pid;
    if (prev_kbd != bindings_.last_input_keyboard || prev_conn != bindings_.pad_connected ||
        prev_name != bindings_.active_pad_name) {
        prompts_gen_.fetch_add(1, std::memory_order_release);
    }
}

u64 Bridge::GamepadNavMask(bool* from_keyboard) const {
    std::scoped_lock lk{input_mutex_};
    if (from_keyboard) {
        *from_keyboard = gamepad_last_kbd_;
    }
    return gamepad_nav_mask_;
}

bool Bridge::ConsumeGamepadToggleCombo(u64 raw_buttons) {
    // Allow Plus + Minus OR L3 + R3 combo on gamepad to toggle menu cleanly from the couch
    // Bit indices follow Core::HID::NpadButton:
    // StickL=4, StickR=5, Plus=10, Minus=11
    constexpr u64 kPlusMinus = (1ULL << 10) | (1ULL << 11);
    constexpr u64 kSticks = (1ULL << 4) | (1ULL << 5);
    const bool held = ((raw_buttons & kPlusMinus) == kPlusMinus) || ((raw_buttons & kSticks) == kSticks);
    std::scoped_lock lk{input_mutex_};
    const bool triggered = held && !combo_held_;
    combo_held_ = held;
    return triggered;
}

void Bridge::LoadFromDisk() {
    if (loaded_from_disk_.exchange(true)) {
        return;
    }
    PcOptions opts;
    bool has_layout_in_ini = false;
    const auto ini_path = ConfigFilePath();
    if (std::filesystem::exists(ini_path)) {
        std::ifstream in(ini_path);
        std::string line;
        while (std::getline(in, line)) {
            const auto hash = line.find('#');
            if (hash != std::string::npos) {
                line.erase(hash);
            }
            const auto eq = line.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            const std::string key = Trim(std::string_view(line).substr(0, eq));
            const std::string val = Trim(std::string_view(line).substr(eq + 1));
            if (key == "menu_size") opts.menu_size = std::clamp(ParseInt(val, 0), 0, 3);
            else if (key == "menu_lang") opts.menu_lang = std::clamp(ParseInt(val, 0), 0, 1);
            // fps_hud: levels before T6c (3 = Detailed); fps_hud2: Off..Dev (Graph = 3)
            else if (key == "fps_hud") {
                const int old = std::clamp(ParseInt(val, 0), 0, 3);
                opts.fps_hud = old == 3 ? static_cast<int>(FpsHudLevel::Detailed) : old;
            }
            else if (key == "fps_hud2") opts.fps_hud = std::clamp(ParseInt(val, 0), 0, 5);
            else if (key == "fps_corner") opts.fps_corner = std::clamp(ParseInt(val, 0), 0, 3);
            else if (key == "button_prompts") opts.button_prompts = std::clamp(ParseInt(val, 0), 0, 6);
            else if (key == "pad_layout") {
                opts.pad_layout = std::clamp(ParseInt(val, 1), 0, 2);
                has_layout_in_ini = true;
            } else if (key == "keyboard_layer") opts.keyboard_layer = ParseBool(val, true);
            else if (key == "occlusion_mod") opts.occlusion_mod = ParseBool(val, true);
            else if (key == "idle_spin_us") opts.idle_spin_us = std::clamp(ParseInt(val, -1), -1, 500);
            else if (key == "read_areas") opts.read_areas = ParseBool(val, true);
            else if (key == "gamemode") opts.gamemode = ParseBool(val, false);
            else if (key == "pin_ccx") opts.pin_ccx = ParseBool(val, false);
            else if (key == "menu_key") opts.menu_key = std::clamp(ParseInt(val, 2), 0, 2);
            else if (key == "compact_key") opts.compact_key = std::clamp(ParseInt(val, 0), 0, 2);
            else if (key == "esc_opens") opts.esc_opens = ParseBool(val, false);
            else if (key == "mute_in_background") opts.mute_in_background = ParseBool(val, false);
            else if (key == "pause_in_background") opts.pause_in_background = ParseBool(val, false);
        }
    }

    if (!has_layout_in_ini) {
        std::ifstream lin(LayoutFilePath());
        std::string txt;
        if (std::getline(lin, txt)) {
            const std::string t = ToLowerAscii(Trim(txt));
            if (t == "nintendo") {
                opts.pad_layout = static_cast<int>(PadLayout::Nintendo);
            } else if (t == "xbox") {
                opts.pad_layout = static_cast<int>(PadLayout::Xbox);
            }
        }
    }

    // E47.T8: the shortcut (play/zelda.sh) exports what this session really runs with, so the
    // menu shows the truth and its saved file can be read back by the shortcut next time.
    const auto env_flag = [](const char* name, bool* out) {
        if (const char* v = std::getenv(name); v != nullptr && v[0] != '\0') {
            *out = !(std::strcmp(v, "0") == 0 || std::strcmp(v, "none") == 0);
            return true;
        }
        return false;
    };
    bool from_shortcut = env_flag("LICHII_SHORTCUT_PIN", &opts.pin_ccx);
    from_shortcut = env_flag("LICHII_SHORTCUT_GAMEMODE", &opts.gamemode) || from_shortcut;
    env_flag("EDEN_NX2_READ_AREAS", &opts.read_areas);
    shortcut_session_ = from_shortcut;

    {
        std::scoped_lock lk{state_mutex_};
        options_ = opts;
    }
    RefreshIdleSpinOverride();
}

void Bridge::SaveToDisk() const {
    if (MenuSaveDisabled()) {
        return;
    }
    PcOptions opts;
    {
        std::scoped_lock lk{state_mutex_};
        opts = options_;
    }

    std::error_code ec;
    const auto ini_path = ConfigFilePath();
    std::filesystem::create_directories(ini_path.parent_path(), ec);
    const auto tmp = ini_path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            return;
        }
        out << "# Ocarina PC Menu options\n"
            << "menu_size=" << opts.menu_size << "\n"
            << "menu_lang=" << opts.menu_lang << "\n"
            << "fps_hud=" << std::min(opts.fps_hud, 3) << "\n"
            << "fps_hud2=" << opts.fps_hud << "\n"
            << "fps_corner=" << opts.fps_corner << "\n"
            << "button_prompts=" << opts.button_prompts << "\n"
            << "pad_layout=" << opts.pad_layout << "\n"
            << "keyboard_layer=" << (opts.keyboard_layer ? 1 : 0) << "\n"
            << "occlusion_mod=" << (opts.occlusion_mod ? 1 : 0) << "\n"
            << "idle_spin_us=" << opts.idle_spin_us << "\n"
            << "read_areas=" << (opts.read_areas ? 1 : 0) << "\n"
            << "gamemode=" << (opts.gamemode ? 1 : 0) << "\n"
            << "pin_ccx=" << (opts.pin_ccx ? 1 : 0) << "\n"
            << "menu_key=" << opts.menu_key << "\n"
            << "compact_key=" << opts.compact_key << "\n"
            << "esc_opens=" << (opts.esc_opens ? 1 : 0) << "\n"
            << "mute_in_background=" << (opts.mute_in_background ? 1 : 0) << "\n"
            << "pause_in_background=" << (opts.pause_in_background ? 1 : 0) << "\n";
        if (shortcut_session_) {
            // pin_ccx / gamemode / read_areas above were seeded by the shortcut: it may apply them
            out << "shortcut_sync=1\n";
        }
    }
    std::filesystem::rename(tmp, ini_path, ec);

    if (opts.pad_layout == static_cast<int>(PadLayout::Nintendo) ||
        opts.pad_layout == static_cast<int>(PadLayout::Xbox)) {
        const auto lpath = LayoutFilePath();
        std::filesystem::create_directories(lpath.parent_path(), ec);
        std::ofstream lout(lpath, std::ios::trunc);
        if (lout) {
            lout << (opts.pad_layout == static_cast<int>(PadLayout::Nintendo) ? "nintendo\n"
                                                                             : "xbox\n");
        }
    }
}

PcOptions Bridge::GetOptions() const {
    std::scoped_lock lk{state_mutex_};
    return options_;
}

void Bridge::SetOptions(const PcOptions& opts, bool bump_prompts) {
    bool occ_changed = false;
    {
        std::scoped_lock lk{state_mutex_};
        occ_changed = (options_.occlusion_mod != opts.occlusion_mod);
        options_ = opts;
    }
    if (occ_changed) {
        SyncOcclusionAddonSetting(opts.occlusion_mod);
    }
    RefreshIdleSpinOverride();
    if (bump_prompts) {
        prompts_gen_.fetch_add(1, std::memory_order_release);
    }
    SaveToDisk();
}

void Bridge::RefreshIdleSpinOverride() {
    int val = -1;
    {
        std::scoped_lock lk{state_mutex_};
        val = options_.idle_spin_us;
    }
    idle_spin_override_us_.store(val, std::memory_order_release);
}

void Bridge::BumpSettingsGeneration() {
    settings_gen_.fetch_add(1, std::memory_order_release);
}

void Bridge::BumpPromptsGeneration() {
    prompts_gen_.fetch_add(1, std::memory_order_release);
}

void Bridge::SetPadBindings(const PadBindingSnapshot& snap) {
    std::scoped_lock lk{state_mutex_};
    bindings_.pad_input = snap.pad_input;
    bindings_.first_vk = snap.first_vk;
    bindings_.pad_names = snap.pad_names;
    bindings_.key_names = snap.key_names;
    if (!snap.active_pad_name.empty()) {
        bindings_.active_pad_name = snap.active_pad_name;
    }
    if (snap.active_pad_vid != 0) {
        bindings_.active_pad_vid = snap.active_pad_vid;
    }
    if (snap.active_pad_pid != 0) {
        bindings_.active_pad_pid = snap.active_pad_pid;
    }
    bindings_.pad_connected = snap.pad_connected;
}

PadBindingSnapshot Bridge::GetPadBindings() const {
    std::scoped_lock lk{state_mutex_};
    return bindings_;
}

int Bridge::EffectivePromptFace() const {
    std::scoped_lock lk{state_mutex_};
    if (options_.button_prompts >= static_cast<int>(PromptFace::Switch) &&
        options_.button_prompts < static_cast<int>(PromptFace::Count)) {
        return options_.button_prompts;
    }
    if (!bindings_.pad_connected || bindings_.last_input_keyboard) {
        return static_cast<int>(PromptFace::Keyboard);
    }
    const std::string lower = ToLowerAscii(bindings_.active_pad_name);
    if (bindings_.active_pad_vid == 0x28DE || lower.find("steam deck") != std::string::npos ||
        lower.find("steam virtual") != std::string::npos) {
        return static_cast<int>(PromptFace::SteamDeck);
    }
    if (bindings_.active_pad_vid == 0x054C || lower.find("dualsense") != std::string::npos ||
        lower.find("dualshock") != std::string::npos || lower.find("playstation") != std::string::npos) {
        if (bindings_.active_pad_pid == 0x0CE6 || bindings_.active_pad_pid == 0x0DF2 ||
            lower.find("dualsense") != std::string::npos || lower.find("ps5") != std::string::npos) {
            return static_cast<int>(PromptFace::Ps5);
        }
        return static_cast<int>(PromptFace::Ps4);
    }
    if (bindings_.active_pad_vid == 0x057E || lower.find("nintendo") != std::string::npos ||
        lower.find("pro controller") != std::string::npos || lower.find("joy-con") != std::string::npos) {
        return static_cast<int>(PromptFace::Switch);
    }
    return static_cast<int>(PromptFace::Xbox);
}

void Bridge::UpdateRenderStats(u32 render_w, u32 render_h, u32 output_w, u32 output_h,
                               std::string_view upscaler_name, bool fsr4_avail, bool fsr4_int8,
                               bool fsr4_failed, bool fsr3_supported, bool hdr_active) {
    std::scoped_lock lk{state_mutex_};
    telemetry_.render_width = render_w;
    telemetry_.render_height = render_h;
    telemetry_.output_width = output_w;
    telemetry_.output_height = output_h;
    telemetry_.upscaler_name = std::string(upscaler_name);
    telemetry_.fsr4_available = fsr4_avail;
    telemetry_.fsr4_int8 = fsr4_int8;
    telemetry_.fsr4_failed = fsr4_failed;
    telemetry_.fsr3_supported = fsr3_supported;
    telemetry_.hdr_active = hdr_active;
}

void Bridge::RecordPresentSample(double frame_ms, bool is_generated) {
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lk{perf_mutex_};
    if (window_start_.time_since_epoch().count() == 0) {
        window_start_ = now;
    }
    if (frame_ms > 0.01 && frame_ms < 2000.0) {
        ++total_frames_;
        if (frame_ms > 50.0) {
            ++stutters_;
        }
        recent_frame_ms_[perf_head_] = frame_ms;
        perf_head_ = (perf_head_ + 1) % recent_frame_ms_.size();
        if (perf_count_ < recent_frame_ms_.size()) {
            ++perf_count_;
        }
    }
    ++window_frames_;
    if (!is_generated) {
        ++window_real_frames_;
    }
    const double sec = std::chrono::duration<double>(now - window_start_).count();
    if (sec >= 0.25) {
        last_fps_ = static_cast<double>(window_frames_) / sec;
        last_base_fps_ = static_cast<double>(window_real_frames_) / sec;
        window_start_ = now;
        window_frames_ = 0;
        window_real_frames_ = 0;
    }
}

void Bridge::RecordUpscale(u32 in_w, u32 in_h, u32 out_w, u32 out_h, int fsr_version) {
    const u64 packed = (u64{in_w & 0xFFFF} << 48) | (u64{in_h & 0xFFFF} << 32) |
                       (u64{out_w & 0xFFFF} << 16) | u64{out_h & 0xFFFF};
    upscale_sizes_.store(packed, std::memory_order_relaxed);
    upscale_version_.store(fsr_version, std::memory_order_relaxed);
    upscale_time_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count(),
                           std::memory_order_release);
}

TelemetrySnapshot Bridge::GetTelemetry() const {
    TelemetrySnapshot snap;
    {
        std::scoped_lock lk{state_mutex_};
        snap = telemetry_;
        snap.has_prompts = has_prompts_.load(std::memory_order_acquire);
    }
    if (const s64 t = upscale_time_ns_.load(std::memory_order_acquire); t != 0) {
        const u64 packed = upscale_sizes_.load(std::memory_order_relaxed);
        snap.fsr_in_width = static_cast<u32>((packed >> 48) & 0xFFFF);
        snap.fsr_in_height = static_cast<u32>((packed >> 32) & 0xFFFF);
        snap.fsr_out_width = static_cast<u32>((packed >> 16) & 0xFFFF);
        snap.fsr_out_height = static_cast<u32>(packed & 0xFFFF);
        snap.fsr_version = upscale_version_.load(std::memory_order_relaxed);
        const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
        snap.fsr_age_ms = static_cast<double>(now - t) / 1e6;
    }
    std::scoped_lock plk{perf_mutex_};
    snap.fps = last_fps_;
    snap.base_fps = last_base_fps_;
    if (perf_count_ > 0) {
        std::array<double, 240> sorted{};
        for (std::size_t i = 0; i < perf_count_; ++i) {
            sorted[i] = recent_frame_ms_[i];
        }
        std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(perf_count_));
        double sum = 0.0;
        for (std::size_t i = 0; i < perf_count_; ++i) {
            sum += sorted[i];
        }
        snap.frametime_ms = sum / static_cast<double>(perf_count_);
        const std::size_t p99_idx = std::min(perf_count_ - 1, (perf_count_ * 99) / 100);
        snap.frametime_p99_ms = sorted[p99_idx];
        snap.frametime_min_ms = sorted[0];
        snap.frametime_max_ms = sorted[perf_count_ - 1];
    }
    snap.total_frames = total_frames_;
    snap.stutters = stutters_;
    return snap;
}

void Bridge::CopyRecentFrames(std::vector<float>& out) const {
    std::scoped_lock lk{perf_mutex_};
    out.resize(perf_count_);
    const std::size_t n = recent_frame_ms_.size();
    const std::size_t first = (perf_head_ + n - perf_count_) % n;
    for (std::size_t i = 0; i < perf_count_; ++i) {
        out[i] = static_cast<float>(recent_frame_ms_[(first + i) % n]);
    }
}

void Bridge::UpdateHostStats(double emulation_speed, double game_fps, int shaders_building) {
    std::scoped_lock lk{state_mutex_};
    telemetry_.host_stats_valid = true;
    telemetry_.emulation_speed = emulation_speed;
    telemetry_.game_fps = game_fps;
    telemetry_.shaders_building = shaders_building;
}

void Bridge::UpdateVramStats(u64 used_bytes, u64 budget_bytes) {
    std::scoped_lock lk{state_mutex_};
    telemetry_.vram_used = used_bytes;
    telemetry_.vram_budget = budget_bytes;
}

void Bridge::SetGameInfo(GameInfo info) {
    std::scoped_lock lk{state_mutex_};
    info.serial = game_info_serial_.load(std::memory_order_relaxed) + 1;
    if (info.icon_rgba.size() !=
        static_cast<std::size_t>(info.icon_width) * static_cast<std::size_t>(info.icon_height) * 4) {
        info.icon_rgba.clear();
        info.icon_width = info.icon_height = 0;
    }
    game_info_ = std::move(info);
    game_info_serial_.store(game_info_.serial, std::memory_order_release);
}

GameInfo Bridge::GetGameInfo() const {
    std::scoped_lock lk{state_mutex_};
    return game_info_;
}

void Bridge::SetRomFsFontData(std::vector<u8> body_otf_bytes,
                              std::vector<u8> strong_otf_bytes,
                              std::vector<u8> heading_otf_bytes) {
    std::scoped_lock lk{state_mutex_};
    if (!body_otf_bytes.empty()) {
        romfs_body_font_ = std::move(body_otf_bytes);
    }
    if (!strong_otf_bytes.empty()) {
        romfs_strong_font_ = std::move(strong_otf_bytes);
    }
    if (!heading_otf_bytes.empty()) {
        romfs_heading_font_ = std::move(heading_otf_bytes);
    }
}

std::vector<u8> Bridge::CopyRomFsBodyFont() const {
    std::scoped_lock lk{state_mutex_};
    return romfs_body_font_;
}

std::vector<u8> Bridge::CopyRomFsStrongFont() const {
    std::scoped_lock lk{state_mutex_};
    return romfs_strong_font_;
}

std::vector<u8> Bridge::CopyRomFsHeadingFont() const {
    std::scoped_lock lk{state_mutex_};
    return romfs_heading_font_;
}

void Bridge::SetPromptsAvailable(bool available, std::string path) {
    std::scoped_lock lk{state_mutex_};
    has_prompts_.store(available, std::memory_order_release);
    prompts_dir_ = std::move(path);
}

std::string Bridge::PromptsPath() const {
    std::scoped_lock lk{state_mutex_};
    return prompts_dir_;
}

void Bridge::RegisterCallbacks(HostCallbacks callbacks) {
    std::scoped_lock lk{state_mutex_};
    callbacks_ = std::move(callbacks);
}

void Bridge::ClearCallbacks() {
    std::scoped_lock lk{state_mutex_};
    callbacks_ = {};
}

void Bridge::RequestSaveConfig() {
    SaveToDisk();
    if (MenuSaveDisabled()) {
        return;
    }
    std::function<void()> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.save_config;
    }
    if (cb) {
        cb();
    }
}

void Bridge::RequestRestartGame() {
    RequestSaveConfig();
    std::function<void()> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.restart_game;
    }
    if (cb) {
        cb();
    }
}

void Bridge::RequestQuitGame() {
    RequestSaveConfig();
    std::function<void()> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.quit_game;
    }
    if (cb) {
        cb();
    }
}

void Bridge::RequestStopGame() {
    RequestSaveConfig();
    std::function<void()> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.stop_game;
    }
    if (cb) {
        cb();
    }
}

void Bridge::RequestApplyPadLayout(int layout_mode) {
    {
        std::scoped_lock lk{state_mutex_};
        options_.pad_layout = layout_mode;
    }
    SaveToDisk();
    std::function<void(int)> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.apply_pad_layout;
    }
    if (cb) {
        cb(layout_mode);
    }
    prompts_gen_.fetch_add(1, std::memory_order_release);
}

void Bridge::RequestResetControls() {
    std::function<void()> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.reset_controls;
    }
    if (cb) {
        cb();
    }
    prompts_gen_.fetch_add(1, std::memory_order_release);
}

void Bridge::SetBindingNames(const std::array<std::string, kSwitchButtonCount>& pad_names,
                             const std::array<std::string, kSwitchButtonCount>& key_names) {
    std::scoped_lock lk{state_mutex_};
    bindings_.pad_names = pad_names;
    bindings_.key_names = key_names;
}

void Bridge::BeginRebind(int switch_button) {
    if (switch_button < 0 || switch_button >= static_cast<int>(kSwitchButtonCount)) {
        return;
    }
    std::function<void(int)> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.begin_rebind;
        rebind_message_.clear();
    }
    if (!cb) {
        FinishRebind("Troca de botão indisponível nesta janela");
        return;
    }
    rebind_target_.store(switch_button, std::memory_order_release);
    cb(switch_button);
}

void Bridge::CancelRebind() {
    if (rebind_target_.exchange(-1, std::memory_order_acq_rel) < 0) {
        return;
    }
    std::function<void()> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.cancel_rebind;
    }
    if (cb) {
        cb();
    }
    FinishRebind("Troca cancelada");
}

void Bridge::FinishRebind(std::string message) {
    rebind_target_.store(-1, std::memory_order_release);
    std::scoped_lock lk{state_mutex_};
    rebind_message_ = std::move(message);
    rebind_message_at_ = std::chrono::steady_clock::now();
}

std::string Bridge::RebindMessage(double* age_sec) const {
    std::scoped_lock lk{state_mutex_};
    if (age_sec) {
        *age_sec = rebind_message_.empty()
                       ? 1e9
                       : std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                       rebind_message_at_)
                             .count();
    }
    return rebind_message_;
}

void Bridge::RequestToggleGamemode() {
    std::function<void()> cb;
    {
        std::scoped_lock lk{state_mutex_};
        cb = callbacks_.toggle_gamemode;
    }
    if (cb) {
        cb();
    }
}

void Bridge::MarkRestartNeeded(std::string_view reason) {
    std::scoped_lock lk{state_mutex_};
    telemetry_.needs_restart = true;
    const std::string s(reason);
    if (std::ranges::find(telemetry_.restart_reasons, s) == telemetry_.restart_reasons.end()) {
        telemetry_.restart_reasons.push_back(s);
    }
}

void Bridge::ClearRestartNeeded() {
    std::scoped_lock lk{state_mutex_};
    telemetry_.needs_restart = false;
    telemetry_.restart_reasons.clear();
}

void Bridge::SyncEnvSwitchesSnapshot() const {
    if (MenuSaveDisabled()) {
        return;
    }
    PcOptions opts;
    {
        std::scoped_lock lk{state_mutex_};
        opts = options_;
    }
    const auto sw_path =
        Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "nx2_menu_switches.txt";
    std::error_code ec;
    std::filesystem::create_directories(sw_path.parent_path(), ec);
    std::ofstream out(sw_path, std::ios::trunc);
    if (!out) {
        return;
    }
    if (opts.idle_spin_us >= 0) {
        out << "EDEN_NX2_IDLE_SPIN_US=" << opts.idle_spin_us << "\n";
    }
    out << "EDEN_NX2_READ_AREAS=" << (opts.read_areas ? 1 : 0) << "\n";
    out << "EDEN_NX2_KEYBOARD_LAYER=" << (opts.keyboard_layer ? 1 : 0) << "\n";
}

} // namespace Common::OotMenu
