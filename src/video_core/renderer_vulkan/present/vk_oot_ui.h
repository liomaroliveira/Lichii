// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#define IMGUI_DEFINE_MATH_OPERATORS
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

#include "imgui.h"
#include "video_core/renderer_vulkan/present/vk_oot_i18n.h"

namespace Vulkan::OotUi {

struct Fonts {
    ImFont* body = nullptr;
    ImFont* strong = nullptr;
    ImFont* heading = nullptr;
};

constexpr ImU32 Rgba(int r, int g, int b, int a = 255) noexcept {
    return IM_COL32(r, g, b, a);
}

inline ImU32 Alpha(ImU32 c, float k) noexcept {
    const int a = static_cast<int>(static_cast<float>((c >> IM_COL32_A_SHIFT) & 0xFF) * k);
    return (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(std::clamp(a, 0, 255)) << IM_COL32_A_SHIFT);
}

// Ocarina of Time dark wood / bronze / gold palette with WCAG >= 4.5:1 contrast.
constexpr ImU32 kInk = Rgba(240, 232, 214);
constexpr ImU32 kInkBright = Rgba(255, 250, 240);
constexpr ImU32 kSoft = Rgba(214, 204, 184);
constexpr ImU32 kDim = Rgba(168, 156, 136);
constexpr ImU32 kGold = Rgba(222, 152, 40);
constexpr ImU32 kGoldHi = Rgba(238, 168, 52);
constexpr ImU32 kGoldInk = Rgba(36, 24, 8);
constexpr ImU32 kAccent = Rgba(214, 170, 96);
constexpr ImU32 kLiveGreen = Rgba(106, 214, 138);
constexpr ImU32 kEdge = Rgba(122, 98, 64, 150);
constexpr ImU32 kCellDark = Rgba(30, 25, 19, 246);
constexpr ImU32 kCellName = Rgba(60, 47, 32, 240);
constexpr ImU32 kCellNameSel = Rgba(92, 70, 42, 248);
constexpr ImU32 kBronze = Rgba(84, 64, 38);
constexpr ImU32 kColumn = Rgba(34, 26, 17, 230);
constexpr ImU32 kBannerTop = Rgba(184, 146, 96, 246);
constexpr ImU32 kBanner = Rgba(176, 138, 88, 246);
constexpr ImU32 kBannerInk = Rgba(46, 30, 12);
constexpr ImU32 kBannerEdge = Rgba(70, 48, 24);
constexpr ImU32 kTabInset = Rgba(84, 56, 30, 250);
constexpr ImU32 kBox = Rgba(36, 28, 19, 248);
constexpr ImU32 kBoxEdge = Rgba(122, 98, 64, 220);
constexpr ImU32 kBar = Rgba(22, 17, 11, 232);
constexpr int kDimAlpha = 185;

inline float Unit(float h, int size, float* margin) noexcept {
    if (size < 1 || size > 3) {
        size = 2;
    }
    const float k = size == 1 ? 0.85f : (size == 3 ? 1.25f : 1.0f);
    *margin = h * (size == 3 ? 0.045f : 0.025f);
    return (h / 720.0f) * k;
}

struct Ctx {
    ImDrawList* d = nullptr;
    Fonts f{};
    float s = 1.0f;
    float w = 1280.0f;
    float h = 720.0f;
    float safe = 18.0f;
    bool pad = false;
    int confirm_dot = 3; // 0 top, 1 right, 2 left, 3 bottom (Xbox South = 3, Nintendo East = 1)
    int back_dot = 1;
};

inline ImVec2 SizeOf(ImFont* font, float px, const char* raw_text, float wrap = 0.0f) {
    return font->CalcTextSizeA(px, FLT_MAX, wrap, Tr(raw_text));
}

inline ImVec2 SizeOfRaw(ImFont* font, float px, const char* text, float wrap = 0.0f) {
    return font->CalcTextSizeA(px, FLT_MAX, wrap, text ? text : "");
}

template <class Width>
float FitPx(float px, float min_px, float step, float room, Width&& width) {
    const float w = width(px);
    if (w <= room) {
        return px;
    }
    if (room <= 0.0f || w <= 0.0f || step <= 0.0f) {
        return 0.0f;
    }
    float f = std::floor(px * room / w / step) * step;
    while (f + step < px && width(f + step) <= room) {
        f += step;
    }
    while (f >= min_px && width(f) > room) {
        f -= step;
    }
    return f >= min_px ? f : 0.0f;
}

inline float TextRaw(Ctx& c, ImFont* font, float px, ImVec2 p, ImU32 col, const char* text,
                     float wrap = 0.0f) {
    const char* s = text ? text : "";
    c.d->AddText(font, px, p, col, s, nullptr, wrap);
    return font->CalcTextSizeA(px, FLT_MAX, wrap, s).x;
}

inline float Text(Ctx& c, ImFont* font, float px, ImVec2 p, ImU32 col, const char* raw_text,
                  float wrap = 0.0f) {
    return TextRaw(c, font, px, p, col, Tr(raw_text), wrap);
}

inline float Para(Ctx& c, ImFont* font, float px, ImVec2 p, ImU32 col, const char* raw_text,
                  float wrap, bool draw = true) {
    const char* t = Tr(raw_text);
    const char* end = t + std::strlen(t);
    float y = 0.0f;
    while (t < end) {
        const char* cut = font->CalcWordWrapPosition(px, t, end, wrap);
        if (cut == t) {
            cut = t + 1;
            while (cut < end && (static_cast<unsigned char>(*cut) & 0xC0u) == 0x80u) {
                cut++;
            }
        }
        if (draw) {
            c.d->AddText(font, px, ImVec2(p.x, p.y + y), col, t, cut);
        }
        y += px * 1.3f;
        t = cut;
        while (t < end && (*t == ' ' || *t == '\n')) {
            t++;
        }
    }
    return y > 0.0f ? y - px * 0.3f : 0.0f;
}

// Top of a line whose letters look centred on y (E47.T6). Measures the loaded font: the centre is
// halfway between the baseline and the average of the cap and x heights, so text that mixes
// capitals and lowercase sits in the middle of its box whatever the font's ascent/descent are.
inline float MidTop(ImFont* font, float px, float y) {
    ImFontBaked* baked = font->GetFontBaked(px);
    // Copy the values: the second lookup may bake 'x' and reallocate baked->Glyphs
    const ImFontGlyph* cap = baked->FindGlyphNoFallback(static_cast<ImWchar>('H'));
    if (cap == nullptr || cap->Y1 <= cap->Y0 || baked->Size <= 0.0f) {
        return y - baked->Ascent + 0.35f * px;
    }
    const float cap_y0 = cap->Y0;
    const float cap_y1 = cap->Y1;
    const ImFontGlyph* low = baked->FindGlyphNoFallback(static_cast<ImWchar>('x'));
    if (low == nullptr) {
        return y - baked->Ascent + 0.35f * px;
    }
    const float scale = px / baked->Size;
    const float top = 0.5f * (cap_y0 + low->Y0) * scale;
    const float base = cap_y1 * scale;
    return y - 0.5f * (top + base);
}

inline float TextMidRaw(Ctx& c, ImFont* font, float px, float x, float y, ImU32 col,
                        const char* text) {
    return TextRaw(c, font, px, ImVec2(x, MidTop(font, px, y)), col, text);
}

inline float TextMid(Ctx& c, ImFont* font, float px, float x, float y, ImU32 col,
                     const char* raw_text) {
    return TextMidRaw(c, font, px, x, y, col, Tr(raw_text));
}

inline void TextMidCentreRaw(Ctx& c, ImFont* font, float px, float cx, float y, ImU32 col,
                             const char* text) {
    TextMidRaw(c, font, px, cx - SizeOfRaw(font, px, text).x * 0.5f, y, col, text);
}

inline void TextMidCentre(Ctx& c, ImFont* font, float px, float cx, float y, ImU32 col,
                          const char* raw_text) {
    TextMidCentreRaw(c, font, px, cx, y, col, Tr(raw_text));
}

inline void TextMidRight(Ctx& c, ImFont* font, float px, float x, float y, ImU32 col,
                         const char* raw_text) {
    const char* s = Tr(raw_text);
    TextMidRaw(c, font, px, x - SizeOfRaw(font, px, s).x, y, col, s);
}

inline void Shadow(Ctx& c, ImVec2 a, ImVec2 b, float spread, float strength, float dy) {
    const int n = 14;
    for (int i = n; i >= 1; i--) {
        const float k = static_cast<float>(i) / static_cast<float>(n);
        const float e = spread * k;
        const int alpha_val = static_cast<int>(255.0f * strength * (1.0f - k) * (1.0f - k) / 3.5f);
        c.d->AddRectFilled(a - ImVec2(e, e) + ImVec2(0.0f, dy),
                           b + ImVec2(e, e) + ImVec2(0.0f, dy), Rgba(0, 0, 0, alpha_val), e);
    }
}

inline void Triangle(Ctx& c, ImVec2 m, float size, int dir, ImU32 col) {
    const float a = size * 0.5f;
    const float b = size * 0.42f;
    switch (dir) {
    case 0:
        c.d->AddTriangleFilled(m + ImVec2(b, -a), m + ImVec2(-b, 0.0f), m + ImVec2(b, a), col);
        break;
    case 1:
        c.d->AddTriangleFilled(m + ImVec2(-b, -a), m + ImVec2(b, 0.0f), m + ImVec2(-b, a), col);
        break;
    case 2:
        c.d->AddTriangleFilled(m + ImVec2(-a, b), m + ImVec2(0.0f, -b), m + ImVec2(a, b), col);
        break;
    default:
        c.d->AddTriangleFilled(m + ImVec2(-a, -b), m + ImVec2(a, -b), m + ImVec2(0.0f, b), col);
        break;
    }
}

inline void Diamond(Ctx& c, ImVec2 m, float r, ImU32 col) {
    c.d->AddQuadFilled(m + ImVec2(0.0f, -r), m + ImVec2(r, 0.0f), m + ImVec2(0.0f, r),
                       m + ImVec2(-r, 0.0f), col);
}

inline void Glow(Ctx& c, ImVec2 a0, ImVec2 b0) {
    const float s = c.s;
    for (int i = 6; i >= 1; i--) {
        const float fi = static_cast<float>(i);
        c.d->AddRect(a0 - ImVec2(fi * 1.4f * s, fi * 1.4f * s),
                     b0 + ImVec2(fi * 1.4f * s, fi * 1.4f * s),
                     Rgba(255, 244, 220, 12 + 4 * (6 - i)), 2.0f * s, 2.0f * s);
    }
    c.d->AddRect(a0 - ImVec2(1.5f * s, 1.5f * s), b0 + ImVec2(1.5f * s, 1.5f * s), kInkBright,
                 2.0f * s, 2.5f * s);
}

inline void GoldFill(Ctx& c, ImVec2 a0, ImVec2 b0) {
    c.d->AddRectFilledMultiColor(a0, b0, kGoldHi, kGoldHi, kGold, kGold);
}

enum class Glyph { UpDown, LeftRight, Tabs, Confirm, Back, Key, Compact };

struct Prompt {
    Glyph g;
    const char* what;
    const char* key = nullptr;
};

inline float DrawGlyph(Ctx& c, float px, ImVec2 p, const Prompt& pr, bool draw) {
    const float s = c.s;
    const float hh = px + 10.0f * s;
    const float my = p.y + hh * 0.5f;
    const float kpx = px * 0.84f;
    const ImU32 cap = kInk;
    const ImU32 ink = Rgba(40, 30, 20);
    const ImU32 dim = Rgba(150, 140, 126);

    auto draw_key = [&](float x, const char* label) {
        const char* tr = label ? Tr(label) : nullptr;
        const float w = tr ? std::max(hh, SizeOfRaw(c.f.strong, kpx, tr).x + 14.0f * s) : hh;
        if (draw) {
            c.d->AddRectFilled(ImVec2(x, p.y), ImVec2(x + w, p.y + hh), cap, 4.0f * s);
            if (tr) {
                TextMidCentreRaw(c, c.f.strong, kpx, x + w * 0.5f, my, ink, tr);
            }
        }
        return w;
    };

    if (pr.g == Glyph::Key) {
        return draw_key(p.x, pr.key);
    }
    if (pr.g == Glyph::Compact) {
        // Keyboard: the chosen key; controller: R3 (E47.T4, vk_oot_menu.cpp ProcessInput)
        return draw_key(p.x, c.pad ? "R3" : (pr.key ? pr.key : "RShift"));
    }
    if (!c.pad) {
        switch (pr.g) {
        case Glyph::UpDown:
        case Glyph::LeftRight: {
            const bool v = pr.g == Glyph::UpDown;
            const float w = draw_key(p.x, nullptr);
            draw_key(p.x + w + 3.0f * s, nullptr);
            if (draw) {
                Triangle(c, ImVec2(p.x + w * 0.5f, my), 8.0f * s, v ? 2 : 0, ink);
                Triangle(c, ImVec2(p.x + w * 1.5f + 3.0f * s, my), 8.0f * s, v ? 3 : 1, ink);
            }
            return 2.0f * w + 3.0f * s;
        }
        case Glyph::Tabs: {
            const float w = draw_key(p.x, "Q");
            return w + 3.0f * s + draw_key(p.x + w + 3.0f * s, "E");
        }
        case Glyph::Confirm:
            return draw_key(p.x, "Enter");
        default:
            return draw_key(p.x, "Esc");
        }
    }

    const float r = hh * 0.5f;
    const ImVec2 m(p.x + r, my);
    if (pr.g == Glyph::Tabs) {
        float x = p.x;
        for (const char* l : {"L", "R"}) {
            const float w = hh * 1.25f;
            if (draw) {
                c.d->AddRectFilled(ImVec2(x, p.y + 2.0f * s), ImVec2(x + w, p.y + hh - 2.0f * s),
                                   cap, (hh - 4.0f * s) * 0.5f);
                TextMidCentreRaw(c, c.f.strong, px * 0.8f, x + w * 0.5f, my, ink, l);
            }
            x += w + 3.0f * s;
        }
        return x - p.x - 3.0f * s;
    }
    if (!draw) {
        return 2.0f * r;
    }
    c.d->AddCircleFilled(m, r, cap);
    if (pr.g == Glyph::UpDown || pr.g == Glyph::LeftRight) {
        const float arm = r * 0.26f;
        const float len = r * 0.68f;
        const bool v = pr.g == Glyph::UpDown;
        c.d->AddRectFilled(m - ImVec2(arm, len), m + ImVec2(arm, len), v ? ink : dim, arm * 0.5f);
        c.d->AddRectFilled(m - ImVec2(len, arm), m + ImVec2(len, arm), v ? dim : ink, arm * 0.5f);
        c.d->AddRectFilled(m - ImVec2(arm, arm), m + ImVec2(arm, arm), ink);
    } else {
        const int lit = pr.g == Glyph::Confirm ? c.confirm_dot : c.back_dot;
        const ImVec2 at[4] = {m + ImVec2(0.0f, -r * 0.5f), m + ImVec2(r * 0.5f, 0.0f),
                              m + ImVec2(-r * 0.5f, 0.0f), m + ImVec2(0.0f, r * 0.5f)};
        for (int i = 0; i < 4; i++) {
            if (i == lit) {
                c.d->AddCircleFilled(at[i], r * 0.27f, ink);
            } else {
                c.d->AddCircle(at[i], r * 0.22f, dim, 0, std::max(1.0f, s));
            }
        }
    }
    return 2.0f * r;
}

inline float Prompts(Ctx& c, ImVec2 p, const Prompt* list, int n, bool draw) {
    const float s = c.s;
    const float px = 17.0f * s;
    const float hh = px + 10.0f * s;
    const float x0 = p.x;
    for (int i = 0; i < n; i++) {
        if (draw) {
            TextMid(c, c.f.body, px, p.x, p.y + hh * 0.5f, kInk, list[i].what);
        }
        p.x += SizeOf(c.f.body, px, list[i].what).x + 8.0f * s;
        p.x += DrawGlyph(c, px, p, list[i], draw) + 22.0f * s;
    }
    return n > 0 ? p.x - x0 - 22.0f * s : 0.0f;
}

inline float PromptBarTop(const Ctx& c) noexcept {
    return c.h - c.safe - (17.0f * c.s + 10.0f * c.s) - 12.0f * c.s;
}

inline float PromptBar(Ctx& c, const Prompt* list, int n) {
    const float s = c.s;
    const float px = 17.0f * s;
    const float ph = px + 10.0f * s;
    const float py = c.h - c.safe - ph;
    Ctx k = c;
    const float room = c.w - 2.0f * c.safe - 20.0f * s;
    const float full = Prompts(c, ImVec2(0.0f, 0.0f), list, n, false);
    if (full > room && full > 0.0f) {
        k.s = s * room / full;
    }
    const float pw = Prompts(k, ImVec2(0.0f, 0.0f), list, n, false);
    const float x0 = std::max(0.0f, c.w - c.safe - pw - 34.0f * s);
    if (x0 > 0.0f) {
        c.d->AddRectFilledMultiColor(ImVec2(std::max(0.0f, x0 - 80.0f * s), py - 12.0f * s),
                                     ImVec2(x0, c.h), Alpha(kBar, 0.0f), kBar, kBar,
                                     Alpha(kBar, 0.0f));
    }
    c.d->AddRectFilled(ImVec2(x0, py - 12.0f * s), ImVec2(c.w, c.h), kBar);
    c.d->AddLine(ImVec2(x0, py - 12.0f * s), ImVec2(c.w, py - 12.0f * s), Alpha(kEdge, 0.8f),
                 std::max(1.0f, s));
    const float kh = 17.0f * k.s + 10.0f * k.s;
    Prompts(k, ImVec2(c.w - c.safe - pw - 10.0f * s, py + (ph - kh) * 0.5f), list, n, true);
    return py - 12.0f * s;
}

inline void Banner(Ctx& c, float hh) {
    const float s = c.s;
    c.d->AddRectFilledMultiColor(ImVec2(0.0f, 0.0f), ImVec2(c.w, hh), kBannerTop, kBannerTop,
                                 kBanner, kBanner);
    c.d->AddRectFilled(ImVec2(0.0f, hh - 3.0f * s), ImVec2(c.w, hh), kBannerEdge);
    c.d->AddLine(ImVec2(0.0f, hh - 4.0f * s), ImVec2(c.w, hh - 4.0f * s),
                 Rgba(226, 192, 138, 140), std::max(1.0f, s));
}

inline void Box(Ctx& c, ImVec2 a, ImVec2 b) {
    Shadow(c, a, b, 18.0f * c.s, 0.8f, 4.0f * c.s);
    c.d->AddRectFilled(a, b, kBox);
    c.d->AddRect(a, b, kBoxEdge, 0.0f, 1.5f * c.s);
}

inline void Button(Ctx& c, ImVec2 a0, ImVec2 b0, const char* raw_label, bool sel) {
    const float s = c.s;
    if (sel) {
        GoldFill(c, a0, b0);
    } else {
        c.d->AddRectFilled(a0, b0, kCellName);
    }
    c.d->AddRect(a0 + ImVec2(4.0f * s, 4.0f * s), b0 - ImVec2(4.0f * s, 4.0f * s),
                 sel ? Rgba(36, 24, 8, 60) : Rgba(190, 154, 98, 90), 0.0f, std::max(1.0f, s));
    c.d->AddRect(a0, b0, kBannerEdge, 0.0f, std::max(1.0f, s));
    ImFont* f = sel ? c.f.strong : c.f.body;
    TextMidCentre(c, f, 20.0f * s, (a0.x + b0.x) * 0.5f, (a0.y + b0.y) * 0.5f,
                  sel ? kGoldInk : kInk, raw_label);
    if (sel) {
        Glow(c, a0, b0);
    }
}

} // namespace Vulkan::OotUi
