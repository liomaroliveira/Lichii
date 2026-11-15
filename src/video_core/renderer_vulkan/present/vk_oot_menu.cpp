// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_vulkan/present/vk_oot_menu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/logging.h"
#include "common/settings.h"
#include "common/settings_enums.h"

namespace Vulkan::OotUi {
namespace {

using Bridge = Common::OotMenu::Bridge;
using Key = Common::OotMenu::Key;
using PcOptions = Common::OotMenu::PcOptions;
using TelemetrySnapshot = Common::OotMenu::TelemetrySnapshot;

template <typename SettingT, typename ValT>
void ApplySettingValue(SettingT& s, const ValT& val) {
    if constexpr (requires {
                      s.UsingGlobal();
                      s.SetGlobal(false);
                  }) {
        if (!Settings::IsConfiguringGlobal() && s.UsingGlobal()) {
            s.SetGlobal(false);
        }
    }
    s.SetValue(val);
}

constexpr std::array<u16, 7> kNx2FpsChoices{30, 60, 90, 120, 144, 165, 240};
constexpr std::array<u16, 6> kSpeedChoices{0, 100, 150, 200, 240, 400};
constexpr std::array<Settings::ResolutionSetup, 7> kResScaleChoices{
    Settings::ResolutionSetup::Res1_2X, Settings::ResolutionSetup::Res3_4X,
    Settings::ResolutionSetup::Res1X,   Settings::ResolutionSetup::Res5_4X,
    Settings::ResolutionSetup::Res3_2X, Settings::ResolutionSetup::Res2X,
    Settings::ResolutionSetup::Res3X,
};
constexpr std::array<Settings::ScalingFilter, 8> kScalingFilterChoices{
    Settings::ScalingFilter::NearestNeighbor, Settings::ScalingFilter::Bilinear,
    Settings::ScalingFilter::Bicubic,         Settings::ScalingFilter::Lanczos,
    Settings::ScalingFilter::ScaleForce,      Settings::ScalingFilter::Fsr,
    Settings::ScalingFilter::Area,            Settings::ScalingFilter::Sgsr,
};
constexpr std::array<int, 6> kIdleSpinChoices{-1, 0, 5, 15, 30, 50};
// NVIDIA DLSS Frame Generation only runs with Streamline (Windows NVIDIA builds); elsewhere it
// falls back to the optical flow interpolation, so the menu does not offer it (E47.T3).
#ifdef EGLANTINE_STREAMLINE
constexpr int kFrameGenMax = 3;
#else
constexpr int kFrameGenMax = 2;
#endif
constexpr std::array<Settings::OcarinaFsrOutput, 4> kFsrOutputChoices{
    Settings::OcarinaFsrOutput::Output1080p,
    Settings::OcarinaFsrOutput::Output1440p,
    Settings::OcarinaFsrOutput::Output4K,
    Settings::OcarinaFsrOutput::OutputGame,
};
constexpr std::array<Settings::OcarinaFsrRender, 3> kFsrRenderChoices{
    Settings::OcarinaFsrRender::Render720p,
    Settings::OcarinaFsrRender::Render1080p,
    Settings::OcarinaFsrRender::Render1440p,
};

// Presets mirror the measured project profiles (bench/profiles/packaa_*). Indices into
// upscaler (0 Native, 1 FSR 3.1, 2 FSR 4.1), kFsrOutputChoices and kFsrRenderChoices.
struct UpscalePreset {
    const char* label;
    int upscaler;
    int output_idx;
    int render_idx;
};
constexpr std::array<UpscalePreset, 7> kPresets{{
    {"Atalho: FSR 4.1 > 1440p no passo do jogo", 2, 3, 1},
    {"FSR 4.1 Full HD", 2, 0, 1},
    {"FSR 3.1 Full HD", 1, 0, 1},
    {"FSR 4.1 1440p", 2, 1, 2},
    {"FSR 4.1 saída 4K", 2, 2, 1},
    {"Leve: FSR 3.1 720p > 1080p", 1, 0, 0},
    {"Nativo (sem FSR)", 0, -1, -1},
}};

template <typename T, std::size_t N>
int IndexIn(const std::array<T, N>& arr, const T& val, int fallback = 0) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        if (arr[i] == val) {
            return static_cast<int>(i);
        }
    }
    return fallback;
}

const std::array<const char*, 4> kTabTitles{"Gráficos", "Desempenho", "Controles", "Áudio & UI"};

// Switch button order matching PadBindingSnapshot (0..15)
constexpr std::array<const char*, 16> kSwitchActionLabels{
    "A (Ação)",     "B (Espada)",   "X (Item E)",  "Y (Item C)",
    "LS (Analóg.)", "RS (Câmera)",  "L (Esq.)",    "R (Dir.)",
    "ZL (Mira Z)",  "ZR (Escudo)",  "+ (Pausa)",   "- (Mapa)",
    "D-Esq.",       "D-Cima",       "D-Dir.",      "D-Baixo",
};

constexpr int kRowListSize = 37;
const std::array<const char*, 4>& TabTitles() noexcept {
    return kTabTitles;
}

float BannerHeight(const Ctx& c) noexcept {
    return 56.0f * c.s;
}

void Columns(const Ctx& c, float* x0, float* x1, float* hx0, float* hx1, bool* narrow) noexcept {
    const float s = c.s;
    *x0 = c.safe + 6.0f * s;
    *x1 = std::min(c.w - c.safe, std::max(c.w * 0.585f, *x0 + 560.0f * s));
    *hx0 = *x1 + 28.0f * s;
    *hx1 = c.w - c.safe;
    *narrow = (*hx1 - *hx0) < 260.0f * s;
    if (*narrow) {
        *hx0 = *x0 + 20.0f * s;
        *hx1 = *x1 - 20.0f * s;
    }
}

void DrawColumn(Ctx& c, float x0, float x1, float top) {
    const float s = c.s;
    c.d->AddRectFilled(ImVec2(x0, top), ImVec2(x1, c.h), kColumn);
    c.d->AddLine(ImVec2(x0, top), ImVec2(x0, c.h), Alpha(kEdge, 1.0f), 2.0f * s);
    c.d->AddLine(ImVec2(x1, top), ImVec2(x1, c.h), Alpha(kEdge, 1.0f), 2.0f * s);
}

std::string FitEllipsis(const Ctx& c, ImFont* f, float px, float room, const char* text) {
    const std::string s(text ? text : "");
    if (room <= 0.0f || SizeOfRaw(f, px, s.c_str()).x <= room) {
        return s;
    }
    std::vector<std::size_t> cut{0};
    for (std::size_t i = 0; i < s.size();) {
        ++i;
        while (i < s.size() && (static_cast<unsigned char>(s[i]) & 0xC0u) == 0x80u) {
            ++i;
        }
        cut.push_back(i);
    }
    std::size_t lo = 0;
    std::size_t hi = cut.size() - 1;
    while (lo < hi) {
        const std::size_t mid = (lo + hi + 1) / 2;
        const std::string candidate = s.substr(0, cut[mid]) + "...";
        if (SizeOfRaw(f, px, candidate.c_str()).x <= room) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return s.substr(0, cut[lo]) + "...";
}

void DrawNameCell(Ctx& c, ImVec2 a, ImVec2 b, const char* raw_name, bool sel, bool pending_restart,
                  bool live_tag) {
    const float s = c.s;
    const float mid = (a.y + b.y) * 0.5f;
    c.d->AddRectFilled(a, b, sel ? kCellNameSel : kCellName);
    c.d->AddRect(a, b, kEdge, 0.0f, std::max(1.0f, s));
    if (sel) {
        c.d->AddRectFilled(a, ImVec2(a.x + 4.0f * s, b.y), kGoldHi);
    }
    float room = b.x - a.x - 36.0f * s;
    if (pending_restart) {
        Diamond(c, ImVec2(b.x - 13.0f * s, mid), 4.2f * s, kGoldHi);
        room -= 14.0f * s;
    } else if (live_tag && sel) {
        c.d->AddCircleFilled(ImVec2(b.x - 13.0f * s, mid), 3.0f * s, kLiveGreen);
        room -= 12.0f * s;
    }
    ImFont* f = sel ? c.f.strong : c.f.body;
    const std::string shown = FitEllipsis(c, f, 18.0f * s, room, Tr(raw_name));
    TextMidRaw(c, f, 18.0f * s, a.x + 18.0f * s, mid, sel ? kInkBright : kInk, shown.c_str());
}

void DrawChoiceCell(Ctx& c, ImVec2 a, ImVec2 b, const std::string& val, bool sel, bool left,
                    bool right, bool fixed) {
    const float s = c.s;
    const float mid = (a.y + b.y) * 0.5f;
    const float in = 4.0f * s;
    const float px = 17.5f * s;
    c.d->AddRectFilled(a, b, kCellDark);
    c.d->AddRect(a, b, kEdge, 0.0f, std::max(1.0f, s));
    if (sel && !fixed) {
        GoldFill(c, a + ImVec2(in, in), b - ImVec2(in, in));
    }
    const ImU32 ink = sel && !fixed ? kGoldInk : (fixed ? kDim : kInk);
    const ImU32 arrow = sel && !fixed ? kGoldInk : kDim;
    if (left) {
        Triangle(c, ImVec2(a.x + in + 11.0f * s, mid), 9.0f * s, 0, arrow);
    }
    if (right) {
        Triangle(c, ImVec2(b.x - in - 11.0f * s, mid), 9.0f * s, 1, arrow);
    }
    ImFont* f = sel ? c.f.strong : c.f.body;
    const float text_room = b.x - a.x - 2.0f * in - ((left || right) ? 44.0f : 16.0f) * s;
    const std::string shown = FitEllipsis(c, f, px, text_room, val.c_str());
    TextMidCentreRaw(c, f, px, (a.x + b.x) * 0.5f, mid, ink, shown.c_str());
    if (sel) {
        Glow(c, a, b);
    }
}

void DrawHalvesCell(Ctx& c, ImVec2 a, ImVec2 b, const char* lo_raw, const char* hi_raw, bool high,
                    bool sel, bool fixed) {
    const float s = c.s;
    const float mid = (a.y + b.y) * 0.5f;
    const float in = 4.0f * s;
    const float px = 17.5f * s;
    const float xm = (a.x + b.x) * 0.5f;
    c.d->AddRectFilled(a, b, kCellDark);
    c.d->AddRect(a, b, kEdge, 0.0f, std::max(1.0f, s));
    const ImVec2 fa = high ? ImVec2(xm, a.y + in) : a + ImVec2(in, in);
    const ImVec2 fb = high ? b - ImVec2(in, in) : ImVec2(xm, b.y - in);
    if (sel && !fixed) {
        GoldFill(c, fa, fb);
    } else {
        c.d->AddRectFilled(fa, fb, kBronze);
    }
    const ImU32 on = sel && !fixed ? kGoldInk : kInk;
    TextMidCentre(c, high ? c.f.body : c.f.strong, px, (a.x + xm) * 0.5f, mid, high ? kDim : on,
                  lo_raw);
    TextMidCentre(c, high ? c.f.strong : c.f.body, px, (xm + b.x) * 0.5f, mid, high ? on : kDim,
                  hi_raw);
    if (sel) {
        Glow(c, a, b);
    }
}

void DrawSliderCell(Ctx& c, ImVec2 a, ImVec2 b, const std::string& val, int permille, bool sel,
                    float* t0, float* t1) {
    const float s = c.s;
    const float mid = (a.y + b.y) * 0.5f;
    const float in = 4.0f * s;
    const float px = 17.5f * s;
    c.d->AddRectFilled(a, b, kCellDark);
    c.d->AddRect(a, b, kEdge, 0.0f, std::max(1.0f, s));
    if (sel) {
        GoldFill(c, a + ImVec2(in, in), b - ImVec2(in, in));
    }
    const float x0 = a.x + in + 24.0f * s;
    const float x1 = b.x - in - 24.0f * s;
    const float frac = std::clamp(static_cast<float>(permille) / 1000.0f, 0.0f, 1.0f);
    const float xv = x0 + (x1 - x0) * frac;
    const float ty = b.y - in - 5.5f * s;
    const ImU32 track = sel ? Rgba(36, 24, 8, 80) : Rgba(255, 255, 255, 40);
    const ImU32 done = sel ? kGoldInk : kAccent;
    c.d->AddRectFilled(ImVec2(x0, ty - 1.5f * s), ImVec2(x1, ty + 1.5f * s), track, 1.5f * s);
    c.d->AddRectFilled(ImVec2(x0, ty - 1.5f * s), ImVec2(xv, ty + 1.5f * s), done, 1.5f * s);
    Diamond(c, ImVec2(xv, ty), 4.5f * s, done);
    TextMidCentreRaw(c, c.f.strong, px, (a.x + b.x) * 0.5f, mid - 3.0f * s,
                     sel ? kGoldInk : kInk, val.c_str());
    const ImU32 arrow = sel ? kGoldInk : kDim;
    Triangle(c, ImVec2(a.x + in + 11.0f * s, mid), 9.0f * s, 0, arrow);
    Triangle(c, ImVec2(b.x - in - 11.0f * s, mid), 9.0f * s, 1, arrow);
    if (sel) {
        Glow(c, a, b);
    }
    *t0 = x0;
    *t1 = x1;
}

void OutlinedText(Ctx& c, ImFont* font, float px, float x, float y, ImU32 col, const char* text) {
    const float o = std::max(1.0f, c.s);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx != 0 || dy != 0) {
                c.d->AddText(font, px, ImVec2(x + static_cast<float>(dx) * o,
                                              y + static_cast<float>(dy) * o),
                             Rgba(0, 0, 0, 225), text);
            }
        }
    }
    c.d->AddText(font, px, ImVec2(x, y), col, text);
}

// Small clickable pill (E47.T4: switch full menu <-> compact bar with the mouse)
void PillButton(Ctx& c, ImVec2 a, ImVec2 b, const char* raw_label, bool hover) {
    const float s = c.s;
    c.d->AddRectFilled(a, b, hover ? kGoldHi : kBannerInk, 4.0f * s);
    c.d->AddRect(a, b, kEdge, 4.0f * s, std::max(1.0f, s));
    TextMidCentre(c, c.f.strong, 14.0f * s, (a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f,
                  hover ? kGoldInk : kInk, raw_label);
}

float PillWidth(const Ctx& c, const char* raw_label) {
    return SizeOf(c.f.strong, 14.0f * c.s, raw_label).x + 26.0f * c.s;
}

const std::vector<OotMenuController::RowDef>& AllRows() {
    using RK = OotMenuController::RowKind;
    using T = OotMenuController::Tab;
    static const std::vector<OotMenuController::RowDef> kRows{
        // Tab 0: Graphics
        {T::Graphics, RK::Heading, "h_preset", "Predefinições", "", false, 0, 0, 1},
        {T::Graphics, RK::Choice, "preset", "Predefinição (perfis)",
         "Aplica de uma vez o upscaler e as resoluções dos perfis medidos do projeto (pack-aa). A resolução interna só muda após reiniciar.",
         false, 1, static_cast<int>(kPresets.size()), 1},
        {T::Graphics, RK::Heading, "h_upscale", "Reconstrução e Nitidez", "", false, 0, 0, 1},
        {T::Graphics, RK::Choice, "upscaler", "Upscaler (OoT)",
         "Reconstrução da imagem do jogo: Nativo, AMD FSR 3.1 ou AMD FSR 4.1 (INT8 nas Radeon RX 6000/7000). Entre FSR 3.1 e 4.1 a troca é ao vivo; ligar ou desligar o FSR (Nativo) vale depois de reiniciar.",
         false, 0, 2, 1},
        {T::Graphics, RK::Choice, "fsr_output", "Resolução Saída FSR",
         "Resolução que o FSR 3.1/4.1 reconstrói. \"1440p no passo do jogo\": com o jogo em 1080p, o FSR reconstrói até 1440p no lugar do passo de escala do próprio jogo, e os passos seguintes do jogo (nitidez e HUD) rodam normalmente; é o mais nítido. Full HD, 1440p e 4K reconstroem e depois esticam a imagem para 4K. Muda ao vivo (o FSR recomeça do zero por um instante).",
         false, 0, 3, 1},
        {T::Graphics, RK::Choice, "fsr_render", "Resolução Interna FSR",
         "Resolução interna renderizada pelo jogo antes do upscaler FSR. 1080p ou 720p aliviam bastante GPU/Templo do Fogo.",
         true, 0, 2, 1},
        {T::Graphics, RK::Choice, "res_scale", "Escala de Resolução",
         "Fator de escala geral do emulador quando o upscaler de cena está desligado ou em jogos gerais.",
         true, 0, 6, 1},
        {T::Graphics, RK::Slider, "rcas_sharp", "Nitidez RCAS (FSR)",
         "Força do filtro de nitidez AMD RCAS aplicado depois do FSR 3.1/4.1, na mesma escala da janela de configurações do Lichii: 0% desliga, 100% é o máximo. Muda ao vivo.",
         false, 0, 200, 10},
        {T::Graphics, RK::Toggle, "oot_aa", "Nitidez do Jogo (Nativo)",
         "Só vale com o Upscaler em Nativo: mantém a reconstrução com nitidez original do jogo ou a troca por uma versão sem nitidez (sem halos). Com FSR 3.1/4.1 ligado, o FSR substitui esse passo do jogo e a nitidez passa a ser a do RCAS acima.",
         false, 0, 1, 1},

        {T::Graphics, RK::Heading, "h_win", "Janela e Pós-Processamento", "", false, 0, 0, 1},
        {T::Graphics, RK::Choice, "win_aa", "Antisserrilhamento Janela",
         "Suavização aplicada na apresentação final da janela (FXAA ou SMAA).", false, 0, 2, 1},
        {T::Graphics, RK::Choice, "win_filter", "Filtro da Janela",
         "Filtro usado ao esticar ou reduzir a imagem do emulador até o tamanho da janela.",
         false, 0, 7, 1},
        {T::Graphics, RK::Choice, "aniso", "Filtro Anisotrópico",
         "Qualidade da filtragem de texturas inclinadas no chão e paredes. Vale por inteiro depois de reiniciar (as texturas já carregadas mantêm o valor antigo).",
         true, 0, 5, 1},
        {T::Graphics, RK::Toggle, "docked", "Modo Console",
         "Define se o jogo inicia em perfil Dock (alta resolução) ou Portátil (720p).", true, 0,
         1, 1},
        {T::Graphics, RK::Choice, "vsync", "Sincronização (VSync)",
         "Controla a sincronização vertical da janela. Mailbox elimina tearing sem travar em múltiplos inteiros.",
         false, 0, 3, 1},
        {T::Graphics, RK::Choice, "hdr", "HDR (Monitor)",
         "Ativa saída HDR10 na tela quando suportada pelo monitor e compositor.", false, 0, 2, 1},
        {T::Graphics, RK::Choice, "frame_gen", "Geração de Quadros",
         "Cria quadros intermediários para a imagem ficar mais fluida que o FPS do jogo. Fluxo óptico: interpolação própria do Lichii feita na GPU (no Lichii aparece como \"FSR 3\", mas não é a biblioteca da AMD). LSFG: usa a DLL do Lossless Scaling, se ela estiver instalada."
#ifdef EGLANTINE_STREAMLINE
         " DLSS Frame Gen: só em placas NVIDIA."
#endif
         ,
         false, 0, kFrameGenMax, 1},
        {T::Graphics, RK::Choice, "fg_mult", "Multiplicador FG",
         "Multiplicador de quadros gerados quando Frame Gen está ativo (2x = dobra o FPS exibido).",
         false, 2, 4, 1},

        // Tab 1: Performance
        {T::Performance, RK::Heading, "h_perf", "Taxa de Quadros e GPU", "", false, 0, 0, 1},
        {T::Performance, RK::Choice, "nx2_fps", "Taxa de Quadros (NX2)",
         "Teto de FPS do motor NX2 do Switch 2. Define a cadência alvo antes de reiniciar.", true, 0,
         6, 1},
        {T::Performance, RK::Choice, "speed_limit", "Limite Velocidade (Live)",
         "Aplica ou remove o limite de velocidade ao vivo em tempo real durante a gameplay.", false,
         0, 5, 1},
        {T::Performance, RK::Toggle, "gpu_acc", "Precisão da GPU",
         "Precisão da GPU emulada. Baixa aumenta performance; Alta corrige efeitos complexos quando necessário.",
         false, 0, 1, 1},
        {T::Performance, RK::Toggle, "vram", "Uso de VRAM",
         "Estratégia de retenção de buffers e texturas na VRAM da GPU. Lida só ao iniciar o jogo.",
         true, 0, 1, 1},

        {T::Performance, RK::Heading, "h_fork", "Otimizações Lichii & Perfil", "", false, 0, 0, 1},
        {T::Performance, RK::Toggle, "occlusion_mod", "Mod Occlusion Off",
         "Mantém ou desativa o mod Occlusion Off (ganho grande de CPU/GPU no Templo do Fogo). Requer reinício.",
         true, 0, 1, 1},
        {T::Performance, RK::Toggle, "async_shaders", "Shaders Assíncronos",
         "Compila pipelines em background para evitar engasgos no primeiro uso de um shader. Lida só ao iniciar o jogo.",
         true, 0, 1, 1},
        {T::Performance, RK::Toggle, "disk_cache", "Cache Shader em Disco",
         "Salva shaders compilados em disco entre sessões para abertura instantânea.", true, 0, 1,
         1},
        {T::Performance, RK::Choice, "precache", "Pré-compilar Shaders",
         "Pré-carrega e compila os shaders salvos antes de entrar no jogo.", true, 0, 2, 1},
        {T::Performance, RK::Choice, "idle_spin", "Idle Spin CPU (us)",
         "Tempo de spin ativo (microsegundos) antes de dormir em Core::Idle. Valores baixos reduzem consumo e latência.",
         false, 0, 5, 1},
        {T::Performance, RK::Toggle, "read_areas", "Read Areas Direto (NX2)",
         "Lê direto da RAM as áreas que a GPU já escreveu, em vez de esperar uma cópia (ganho grande no Templo do Fogo). Vale na próxima abertura pelo atalho.",
         true, 0, 1, 1},
        {T::Performance, RK::Toggle, "gamemode", "GameMode (Linux)",
         "Liga o GameMode e o clock alto da GPU ao abrir o jogo pelo atalho. Vale na próxima abertura pelo atalho.", true, 0, 1,
         1},
        {T::Performance, RK::Toggle, "pin_ccx", "Fixar 1 CCX (Ryzen)",
         "Põe os núcleos do jogo num CCX e a GPU no outro (Ryzen com 2 CCX, +4% no 3800X). Vale na próxima abertura pelo atalho.", true, 0, 1,
         1},
        {T::Performance, RK::Action, "apply_restart", "Aplicar e Reiniciar",
         "Salva todas as configurações no disco e reinicia a execução do jogo imediatamente.", false,
         0, 0, 1},

        // Tab 2: Controls
        {T::Controls, RK::Heading, "h_prompts", "Botões na Tela e Layout", "", false, 0, 0, 1},
        {T::Controls, RK::Choice, "prompts", "Ícones de Botão (Prompts)",
         "Estilo visual dos ícones de botões mostrados dentro do jogo: Automático segue o controle/teclado conectado!",
         false, 0, 6, 1},
        {T::Controls, RK::Choice, "pad_layout", "Layout do Controle",
         "Alterna ao vivo entre botões estilo Xbox PC (A embaixo / B direita) e Nintendo posicional, sincronizando com os prompts.",
         false, 0, 2, 1},
        {T::Controls, RK::Toggle, "kbd_layer", "Camada Teclado NX2",
         "Ativa a camada de teclado/mouse do NX2 ao vivo para câmera e movimento sem controle.", false,
         0, 1, 1},
        {T::Controls, RK::Choice, "rebind", "Trocar um botão",
         "Escolha o botão do Switch e aperte no controle o botão que vai fazer essa função. Fica salvo no perfil deste controle (oot-controles).",
         false, 0, 15, 1},
        {T::Controls, RK::Action, "reset_controls", "Resetar Controles",
         "Mostra os botões e teclas atuais do Jogador 1 e permite restaurar os padrões.", false, 0,
         0, 1},

        // Tab 3: Audio & UI
        {T::AudioUi, RK::Heading, "h_audio", "Áudio", "", false, 0, 0, 1},
        {T::AudioUi, RK::Slider, "volume", "Volume Principal",
         "Volume geral da saída de áudio do emulador (0% a 100%).", false, 0, 100, 5},
        {T::AudioUi, RK::Toggle, "mute", "Silenciar (Mudo)",
         "Silencia instantaneamente todo o som do jogo.", false, 0, 1, 1},
        {T::AudioUi, RK::Choice, "audio_sink", "Backend de Áudio",
         "Driver de saída de áudio utilizado pelo emulador.", true, 0, 3, 1},

        {T::AudioUi, RK::Heading, "h_hud", "Medidor FPS e Interface", "", false, 0, 0, 1},
        {T::AudioUi, RK::Choice, "fps_hud", "Contador de FPS",
         "Contador no canto da tela, como no Longshot: FPS, Frame time + 1% low, Gráfico (pior quadro por coluna), Detalhado (resolução, velocidade, shaders, VRAM) ou Dev (mín/máx, travadas, ajustes).",
         false, 0, 5, 1},
        {T::AudioUi, RK::Choice, "fps_corner", "Posição do FPS",
         "Canto da tela onde o contador de FPS é exibido.", false, 0, 3, 1},
        {T::AudioUi, RK::Choice, "menu_size", "Tamanho do Menu",
         "Escala dos textos e caixas do menu.", false, 0, 3, 1},
        {T::AudioUi, RK::Choice, "menu_lang", "Idioma do Menu",
         "Idioma exibido em todas as abas e descrições deste menu.", false, 0, 1, 1},

        {T::AudioUi, RK::Heading, "h_keys", "Atalhos e Sessão", "", false, 0, 0, 1},
        {T::AudioUi, RK::Choice, "menu_key", "Tecla Menu Completo",
         "Atalho de teclado para abrir/fechar este menu completo (no controle: L3+R3 ou +&-).",
         false, 0, 2, 1},
        {T::AudioUi, RK::Choice, "compact_key", "Tecla Barra Compacta",
         "Tecla para abrir diretamente ou alternar com a Barra Compacta ao vivo.", false, 0, 2, 1},
        {T::AudioUi, RK::Toggle, "mute_bg", "Mudo em 2º Plano",
         "Silencia o jogo enquanto a janela do emulador não está em foco e volta o som ao voltar o foco. Ao vivo.",
         false, 0, 1, 1},
        {T::AudioUi, RK::Toggle, "pause_bg", "Pausar em 2º Plano",
         "Pausa a emulação enquanto a janela não está em foco e continua ao voltar. O controle segue funcionando com a janela em 2º plano quando esta opção está desligada. Ao vivo (só na interface gráfica).",
         false, 0, 1, 1},
        {T::AudioUi, RK::Toggle, "esc_opens", "Esc Abre o Menu",
         "Permite usar a tecla Esc também para abrir o menu completo (além de fechá-lo).", false, 0,
         1, 1},
        {T::AudioUi, RK::Action, "restart_game", "Reiniciar jogo",
         "Salva as configurações e reinicia o jogo, sem a pergunta do emulador.", false, 0, 0, 1},
        {T::AudioUi, RK::Action, "stop_game", "Encerrar e voltar à lista",
         "Salva as configurações, encerra o jogo e deixa o emulador aberto na lista de jogos.",
         false, 0, 0, 1},
        {T::AudioUi, RK::Action, "quit_game", "Fechar o jogo",
         "Salva as configurações e fecha o jogo e o emulador, sem a pergunta do emulador.", false,
         0, 0, 1},
    };
    return kRows;
}

} // namespace

OotMenuController::OotMenuController() = default;

bool OotMenuController::ShouldDraw(const Common::OotMenu::Bridge& bridge) const {
    if (!bridge.IsOcarina()) {
        return false;
    }
    if (bridge.IsOpen()) {
        return true;
    }
    // E47.T9: keep ticking the self-test until its live upscaler switches ran, menu open or not
    static const bool selftest = [] {
        const char* v = std::getenv("EDEN_NX2_OOT_MENU_SELFTEST");
        return v != nullptr && v[0] == '1';
    }();
    if (selftest && selftest_live_step_ < 2) {
        return true;
    }
    return bridge.GetOptions().fps_hud > 0;
}

void OotMenuController::CaptureBaselineOnce(const PcOptions& opts) {
    if (baseline_.captured) {
        return;
    }
    baseline_.captured = true;
    baseline_.ocarina_fsr_render = static_cast<int>(Settings::values.ocarina_fsr_render.GetValue());
    baseline_.ocarina_fsr = Settings::values.ocarina_fsr.GetValue();
    baseline_.max_anisotropy = static_cast<int>(Settings::values.max_anisotropy.GetValue());
    baseline_.vram_usage_mode = static_cast<int>(Settings::values.vram_usage_mode.GetValue());
    baseline_.async_shaders = Settings::values.use_asynchronous_shaders.GetValue();
    baseline_.resolution_setup = static_cast<int>(Settings::values.resolution_setup.GetValue());
    baseline_.use_docked_mode = static_cast<int>(Settings::values.use_docked_mode.GetValue());
    baseline_.frame_rate_limit = static_cast<int>(Settings::values.frame_rate_limit.GetValue());
    baseline_.occlusion_mod = opts.occlusion_mod;
    baseline_.disk_shader_cache = Settings::values.use_disk_shader_cache.GetValue();
    baseline_.precache_shaders = static_cast<int>(Settings::values.precache_shaders.GetValue());
    baseline_.read_areas = opts.read_areas;
    baseline_.gamemode = opts.gamemode;
    baseline_.pin_ccx = opts.pin_ccx;
    baseline_.sink_id = static_cast<int>(Settings::values.sink_id.GetValue());
}

bool OotMenuController::IsRowPendingRestart(const char* id, const PcOptions& opts) const {
    if (!baseline_.captured || !id) {
        return false;
    }
    if (std::strcmp(id, "fsr_render") == 0) {
        return static_cast<int>(Settings::values.ocarina_fsr_render.GetValue()) !=
               baseline_.ocarina_fsr_render;
    }
    if (std::strcmp(id, "aniso") == 0) {
        return static_cast<int>(Settings::values.max_anisotropy.GetValue()) !=
               baseline_.max_anisotropy;
    }
    if (std::strcmp(id, "vram") == 0) {
        return static_cast<int>(Settings::values.vram_usage_mode.GetValue()) !=
               baseline_.vram_usage_mode;
    }
    if (std::strcmp(id, "async_shaders") == 0) {
        return Settings::values.use_asynchronous_shaders.GetValue() != baseline_.async_shaders;
    }
    if (std::strcmp(id, "upscaler") == 0 || std::strcmp(id, "preset") == 0) {
        // Native <-> FSR changes the render scale patched into the game at boot (E47.T2)
        return Settings::values.ocarina_fsr.GetValue() != baseline_.ocarina_fsr;
    }
    if (std::strcmp(id, "res_scale") == 0) {
        return static_cast<int>(Settings::values.resolution_setup.GetValue()) !=
               baseline_.resolution_setup;
    }
    if (std::strcmp(id, "docked") == 0) {
        return static_cast<int>(Settings::values.use_docked_mode.GetValue()) !=
               baseline_.use_docked_mode;
    }
    if (std::strcmp(id, "nx2_fps") == 0) {
        return static_cast<int>(Settings::values.frame_rate_limit.GetValue()) !=
               baseline_.frame_rate_limit;
    }
    if (std::strcmp(id, "occlusion_mod") == 0) {
        return opts.occlusion_mod != baseline_.occlusion_mod;
    }
    if (std::strcmp(id, "disk_cache") == 0) {
        return Settings::values.use_disk_shader_cache.GetValue() != baseline_.disk_shader_cache;
    }
    if (std::strcmp(id, "precache") == 0) {
        return static_cast<int>(Settings::values.precache_shaders.GetValue()) !=
               baseline_.precache_shaders;
    }
    if (std::strcmp(id, "read_areas") == 0) {
        return opts.read_areas != baseline_.read_areas;
    }
    if (std::strcmp(id, "gamemode") == 0) {
        return opts.gamemode != baseline_.gamemode;
    }
    if (std::strcmp(id, "pin_ccx") == 0) {
        return opts.pin_ccx != baseline_.pin_ccx;
    }
    if (std::strcmp(id, "audio_sink") == 0) {
        return static_cast<int>(Settings::values.sink_id.GetValue()) != baseline_.sink_id;
    }
    return false;
}

std::vector<const char*> OotMenuController::CollectPendingRowNames(const PcOptions& opts) const {
    std::vector<const char*> names;
    for (const auto& r : AllRows()) {
        if ((r.requires_restart || std::strcmp(r.id, "upscaler") == 0) &&
            IsRowPendingRestart(r.id, opts)) {
            names.push_back(r.label);
        }
    }
    return names;
}

std::vector<const OotMenuController::RowDef*> OotMenuController::RowsForTab(Tab t) const {
    std::vector<const RowDef*> out;
    for (const auto& r : AllRows()) {
        if (r.tab == t) {
            out.push_back(&r);
        }
    }
    return out;
}

int OotMenuController::ReadRowValue(const char* id, const PcOptions& opts) const {
    if (std::strcmp(id, "rebind") == 0) {
        return rebind_pick_;
    }
    if (std::strcmp(id, "preset") == 0) {
        const int up = ReadRowValue("upscaler", opts);
        const int out = ReadRowValue("fsr_output", opts);
        const int ren = ReadRowValue("fsr_render", opts);
        for (int i = 0; i < static_cast<int>(kPresets.size()); ++i) {
            const auto& p = kPresets[static_cast<std::size_t>(i)];
            if (p.upscaler != up) {
                continue;
            }
            if (p.upscaler == 0 || (p.output_idx == out && p.render_idx == ren)) {
                return i + 1;
            }
        }
        return 0; // Custom
    }
    if (std::strcmp(id, "upscaler") == 0) {
        if (!Settings::values.ocarina_fsr.GetValue()) {
            return 0;
        }
        return Settings::values.amd_fsr4.GetValue() ? 2 : 1;
    }
    if (std::strcmp(id, "fsr_output") == 0) {
        return IndexIn(kFsrOutputChoices, Settings::values.ocarina_fsr_output.GetValue(), 0);
    }
    if (std::strcmp(id, "fsr_render") == 0) {
        return IndexIn(kFsrRenderChoices, Settings::values.ocarina_fsr_render.GetValue(), 1);
    }
    if (std::strcmp(id, "res_scale") == 0) {
        return IndexIn(kResScaleChoices, Settings::values.resolution_setup.GetValue(), 2);
    }
    if (std::strcmp(id, "rcas_sharp") == 0) {
        // Stored as RCAS "stops" (0 = sharpest, 200 = off); the menu shows sharpness (E47.T1)
        return 200 - std::clamp(Settings::values.fsr_sharpening_slider.GetValue(), 0, 200);
    }
    if (std::strcmp(id, "oot_aa") == 0) {
        return Settings::values.oot_anti_aliasing.GetValue() ==
                       Settings::OotAntiAliasing::NoSharpening
                   ? 1
                   : 0;
    }
    if (std::strcmp(id, "win_aa") == 0) {
        return static_cast<int>(Settings::values.anti_aliasing.GetValue());
    }
    if (std::strcmp(id, "win_filter") == 0) {
        return IndexIn(kScalingFilterChoices, Settings::values.scaling_filter.GetValue(), 1);
    }
    if (std::strcmp(id, "aniso") == 0) {
        return static_cast<int>(Settings::values.max_anisotropy.GetValue());
    }
    if (std::strcmp(id, "docked") == 0) {
        return Settings::values.use_docked_mode.GetValue() == Settings::ConsoleMode::Docked ? 1 : 0;
    }
    if (std::strcmp(id, "vsync") == 0) {
        return static_cast<int>(Settings::values.vsync_mode.GetValue());
    }
    if (std::strcmp(id, "hdr") == 0) {
        return static_cast<int>(Settings::values.hdr_mode.GetValue());
    }
    if (std::strcmp(id, "frame_gen") == 0) {
        if (!Settings::values.frame_gen.GetValue()) {
            return 0;
        }
        switch (Settings::values.frame_gen_type.GetValue()) {
        case Settings::FrameGenType::Lsfg:
            return 1;
        case Settings::FrameGenType::Fsr3:
            return 2;
        case Settings::FrameGenType::NvidiaDlssG:
            return kFrameGenMax == 3 ? 3 : 2; // without Streamline it runs the optical flow
        }
        return 1;
    }
    if (std::strcmp(id, "fg_mult") == 0) {
        return std::clamp(static_cast<int>(Settings::values.frame_gen_multiplier.GetValue()), 2, 4);
    }
    if (std::strcmp(id, "nx2_fps") == 0) {
        const u16 cur = Settings::values.frame_rate_limit.GetValue();
        int best = 1;
        int best_diff = 9999;
        for (int i = 0; i < static_cast<int>(kNx2FpsChoices.size()); ++i) {
            const int d = std::abs(static_cast<int>(cur) - static_cast<int>(kNx2FpsChoices[i]));
            if (d < best_diff) {
                best_diff = d;
                best = i;
            }
        }
        return best;
    }
    if (std::strcmp(id, "speed_limit") == 0) {
        if (!Settings::values.use_speed_limit.GetValue()) {
            return 0;
        }
        const u16 lim = Settings::values.speed_limit.GetValue();
        for (int i = 1; i < static_cast<int>(kSpeedChoices.size()); ++i) {
            if (lim <= kSpeedChoices[i]) {
                return i;
            }
        }
        return 5;
    }
    if (std::strcmp(id, "gpu_acc") == 0) {
        return Settings::values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::High ? 1 : 0;
    }
    if (std::strcmp(id, "vram") == 0) {
        return Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive
                   ? 1
                   : 0;
    }
    if (std::strcmp(id, "occlusion_mod") == 0) {
        return opts.occlusion_mod ? 1 : 0;
    }
    if (std::strcmp(id, "async_shaders") == 0) {
        return Settings::values.use_asynchronous_shaders.GetValue() ? 1 : 0;
    }
    if (std::strcmp(id, "disk_cache") == 0) {
        return Settings::values.use_disk_shader_cache.GetValue() ? 1 : 0;
    }
    if (std::strcmp(id, "precache") == 0) {
        return static_cast<int>(Settings::values.precache_shaders.GetValue());
    }
    if (std::strcmp(id, "idle_spin") == 0) {
        return IndexIn(kIdleSpinChoices, opts.idle_spin_us, 0);
    }
    if (std::strcmp(id, "read_areas") == 0) {
        return opts.read_areas ? 1 : 0;
    }
    if (std::strcmp(id, "gamemode") == 0) {
        return opts.gamemode ? 1 : 0;
    }
    if (std::strcmp(id, "pin_ccx") == 0) {
        return opts.pin_ccx ? 1 : 0;
    }
    if (std::strcmp(id, "prompts") == 0) {
        return opts.button_prompts;
    }
    if (std::strcmp(id, "pad_layout") == 0) {
        return opts.pad_layout;
    }
    if (std::strcmp(id, "kbd_layer") == 0) {
        return opts.keyboard_layer ? 1 : 0;
    }
    if (std::strcmp(id, "volume") == 0) {
        return static_cast<int>(Settings::values.volume.GetValue());
    }
    if (std::strcmp(id, "mute") == 0) {
        return Settings::values.audio_muted.GetValue() ? 1 : 0;
    }
    if (std::strcmp(id, "audio_sink") == 0) {
        return static_cast<int>(Settings::values.sink_id.GetValue());
    }
    if (std::strcmp(id, "fps_hud") == 0) {
        return opts.fps_hud;
    }
    if (std::strcmp(id, "fps_corner") == 0) {
        return opts.fps_corner;
    }
    if (std::strcmp(id, "menu_size") == 0) {
        return opts.menu_size;
    }
    if (std::strcmp(id, "menu_lang") == 0) {
        return opts.menu_lang;
    }
    if (std::strcmp(id, "menu_key") == 0) {
        return opts.menu_key;
    }
    if (std::strcmp(id, "compact_key") == 0) {
        return opts.compact_key;
    }
    if (std::strcmp(id, "mute_bg") == 0) {
        return opts.mute_in_background ? 1 : 0;
    }
    if (std::strcmp(id, "pause_bg") == 0) {
        return opts.pause_in_background ? 1 : 0;
    }
    if (std::strcmp(id, "esc_opens") == 0) {
        return opts.esc_opens ? 1 : 0;
    }
    return 0;
}

void OotMenuController::WriteRowValue(const char* id, int val, PcOptions& opts, Bridge& bridge) {
    bool dirty_settings = false;
    bool dirty_bridge = false;
    bool bump_prompts = false;
    bool request_save = true;

    if (std::strcmp(id, "rebind") == 0) {
        rebind_pick_ = std::clamp(val, 0, static_cast<int>(kSwitchActionLabels.size()) - 1);
        bridge.BeginRebind(rebind_pick_);
        return;
    }
    if (std::strcmp(id, "preset") == 0) {
        if (val < 1 || val > static_cast<int>(kPresets.size())) {
            return;
        }
        const auto& p = kPresets[static_cast<std::size_t>(val - 1)];
        ApplySettingValue(Settings::values.ocarina_fsr, p.upscaler > 0);
        ApplySettingValue(Settings::values.amd_fsr4, p.upscaler == 2);
        if (p.output_idx >= 0) {
            ApplySettingValue(Settings::values.ocarina_fsr_output,
                              kFsrOutputChoices[static_cast<std::size_t>(p.output_idx)]);
        }
        if (p.render_idx >= 0) {
            ApplySettingValue(Settings::values.ocarina_fsr_render,
                              kFsrRenderChoices[static_cast<std::size_t>(p.render_idx)]);
        }
        dirty_settings = true;
    } else if (std::strcmp(id, "upscaler") == 0) {
        val = std::clamp(val, 0, 2);
        ApplySettingValue(Settings::values.ocarina_fsr, val > 0);
        ApplySettingValue(Settings::values.amd_fsr4, val == 2);
        dirty_settings = true;
    } else if (std::strcmp(id, "fsr_output") == 0) {
        val = std::clamp(val, 0, 3);
        ApplySettingValue(Settings::values.ocarina_fsr_output,
                          kFsrOutputChoices[static_cast<std::size_t>(val)]);
        dirty_settings = true;
    } else if (std::strcmp(id, "fsr_render") == 0) {
        val = std::clamp(val, 0, 2);
        ApplySettingValue(Settings::values.ocarina_fsr_render,
                          kFsrRenderChoices[static_cast<std::size_t>(val)]);
    } else if (std::strcmp(id, "res_scale") == 0) {
        val = std::clamp(val, 0, static_cast<int>(kResScaleChoices.size()) - 1);
        ApplySettingValue(Settings::values.resolution_setup,
                          kResScaleChoices[static_cast<std::size_t>(val)]);
    } else if (std::strcmp(id, "rcas_sharp") == 0) {
        val = std::clamp(val, 0, 200);
        ApplySettingValue(Settings::values.fsr_sharpening_slider, 200 - val);
        dirty_settings = true;
    } else if (std::strcmp(id, "oot_aa") == 0) {
        ApplySettingValue(Settings::values.oot_anti_aliasing,
                          val != 0 ? Settings::OotAntiAliasing::NoSharpening
                                   : Settings::OotAntiAliasing::Game);
        dirty_settings = true;
    } else if (std::strcmp(id, "win_aa") == 0) {
        ApplySettingValue(Settings::values.anti_aliasing,
                          static_cast<Settings::AntiAliasing>(std::clamp(val, 0, 2)));
        dirty_settings = true;
    } else if (std::strcmp(id, "win_filter") == 0) {
        val = std::clamp(val, 0, static_cast<int>(kScalingFilterChoices.size()) - 1);
        ApplySettingValue(Settings::values.scaling_filter,
                          kScalingFilterChoices[static_cast<std::size_t>(val)]);
        dirty_settings = true;
    } else if (std::strcmp(id, "aniso") == 0) {
        ApplySettingValue(Settings::values.max_anisotropy,
                          static_cast<Settings::AnisotropyMode>(std::clamp(val, 0, 5)));
        dirty_settings = true;
    } else if (std::strcmp(id, "docked") == 0) {
        ApplySettingValue(Settings::values.use_docked_mode,
                          val != 0 ? Settings::ConsoleMode::Docked
                                   : Settings::ConsoleMode::Handheld);
    } else if (std::strcmp(id, "vsync") == 0) {
        ApplySettingValue(Settings::values.vsync_mode,
                          static_cast<Settings::VSyncMode>(std::clamp(val, 0, 3)));
        dirty_settings = true;
    } else if (std::strcmp(id, "hdr") == 0) {
        ApplySettingValue(Settings::values.hdr_mode,
                          static_cast<Settings::HdrMode>(std::clamp(val, 0, 2)));
        dirty_settings = true;
    } else if (std::strcmp(id, "frame_gen") == 0) {
        val = std::clamp(val, 0, kFrameGenMax);
        if (kFrameGenMax == 2 && val == 2 && Settings::values.frame_gen.GetValue() &&
            Settings::values.frame_gen_type.GetValue() == Settings::FrameGenType::NvidiaDlssG) {
            return; // already running the optical flow fallback; keep the user's DLSS-G choice
        }
        ApplySettingValue(Settings::values.frame_gen, val > 0);
        if (val == 1) {
            ApplySettingValue(Settings::values.frame_gen_type, Settings::FrameGenType::Lsfg);
        } else if (val == 2) {
            ApplySettingValue(Settings::values.frame_gen_type, Settings::FrameGenType::Fsr3);
        } else if (val == 3) {
            ApplySettingValue(Settings::values.frame_gen_type,
                              Settings::FrameGenType::NvidiaDlssG);
        }
        dirty_settings = true;
    } else if (std::strcmp(id, "fg_mult") == 0) {
        ApplySettingValue(Settings::values.frame_gen_multiplier,
                          static_cast<u32>(std::clamp(val, 2, 4)));
        dirty_settings = true;
    } else if (std::strcmp(id, "nx2_fps") == 0) {
        val = std::clamp(val, 0, static_cast<int>(kNx2FpsChoices.size()) - 1);
        ApplySettingValue(Settings::values.frame_rate_limit,
                          kNx2FpsChoices[static_cast<std::size_t>(val)]);
    } else if (std::strcmp(id, "speed_limit") == 0) {
        val = std::clamp(val, 0, static_cast<int>(kSpeedChoices.size()) - 1);
        ApplySettingValue(Settings::values.use_speed_limit, val > 0);
        if (val > 0) {
            ApplySettingValue(Settings::values.speed_limit,
                              kSpeedChoices[static_cast<std::size_t>(val)]);
        }
        dirty_settings = true;
    } else if (std::strcmp(id, "gpu_acc") == 0) {
        ApplySettingValue(Settings::values.gpu_accuracy,
                          val != 0 ? Settings::GpuAccuracy::High : Settings::GpuAccuracy::Low);
        Settings::UpdateGPUAccuracy();
        dirty_settings = true;
    } else if (std::strcmp(id, "vram") == 0) {
        ApplySettingValue(Settings::values.vram_usage_mode,
                          val != 0 ? Settings::VramUsageMode::Aggressive
                                   : Settings::VramUsageMode::Conservative);
        dirty_settings = true;
    } else if (std::strcmp(id, "occlusion_mod") == 0) {
        opts.occlusion_mod = (val != 0);
        dirty_bridge = true;
    } else if (std::strcmp(id, "async_shaders") == 0) {
        ApplySettingValue(Settings::values.use_asynchronous_shaders, val != 0);
        dirty_settings = true;
    } else if (std::strcmp(id, "disk_cache") == 0) {
        ApplySettingValue(Settings::values.use_disk_shader_cache, val != 0);
    } else if (std::strcmp(id, "precache") == 0) {
        ApplySettingValue(Settings::values.precache_shaders,
                          static_cast<Settings::PrecacheShaders>(std::clamp(val, 0, 2)));
    } else if (std::strcmp(id, "idle_spin") == 0) {
        val = std::clamp(val, 0, static_cast<int>(kIdleSpinChoices.size()) - 1);
        opts.idle_spin_us = kIdleSpinChoices[static_cast<std::size_t>(val)];
        dirty_bridge = true;
    } else if (std::strcmp(id, "read_areas") == 0) {
        opts.read_areas = (val != 0);
        dirty_bridge = true;
    } else if (std::strcmp(id, "gamemode") == 0) {
        opts.gamemode = (val != 0);
        dirty_bridge = true;
        bridge.RequestToggleGamemode();
    } else if (std::strcmp(id, "pin_ccx") == 0) {
        opts.pin_ccx = (val != 0);
        dirty_bridge = true;
    } else if (std::strcmp(id, "prompts") == 0) {
        opts.button_prompts = std::clamp(val, 0, 6);
        dirty_bridge = true;
        bump_prompts = true;
    } else if (std::strcmp(id, "pad_layout") == 0) {
        val = std::clamp(val, 0, 2);
        opts.pad_layout = val;
        bridge.SetOptions(opts, true);
        bridge.RequestApplyPadLayout(val);
        request_save = false;
    } else if (std::strcmp(id, "kbd_layer") == 0) {
        opts.keyboard_layer = (val != 0);
        dirty_bridge = true;
        bump_prompts = true;
    } else if (std::strcmp(id, "volume") == 0) {
        ApplySettingValue(Settings::values.volume, static_cast<u8>(std::clamp(val, 0, 100)));
        dirty_settings = true;
    } else if (std::strcmp(id, "mute") == 0) {
        ApplySettingValue(Settings::values.audio_muted, val != 0);
        dirty_settings = true;
    } else if (std::strcmp(id, "audio_sink") == 0) {
        ApplySettingValue(Settings::values.sink_id,
                          static_cast<Settings::AudioEngine>(std::clamp(val, 0, 3)));
    } else if (std::strcmp(id, "fps_hud") == 0) {
        opts.fps_hud = std::clamp(val, 0, 5);
        dirty_bridge = true;
    } else if (std::strcmp(id, "fps_corner") == 0) {
        opts.fps_corner = std::clamp(val, 0, 3);
        dirty_bridge = true;
    } else if (std::strcmp(id, "menu_size") == 0) {
        opts.menu_size = std::clamp(val, 0, 3);
        dirty_bridge = true;
    } else if (std::strcmp(id, "menu_lang") == 0) {
        opts.menu_lang = std::clamp(val, 0, 1);
        SetLang(opts.menu_lang);
        dirty_bridge = true;
    } else if (std::strcmp(id, "menu_key") == 0) {
        opts.menu_key = std::clamp(val, 0, 2);
        dirty_bridge = true;
    } else if (std::strcmp(id, "compact_key") == 0) {
        opts.compact_key = std::clamp(val, 0, 2);
        dirty_bridge = true;
    } else if (std::strcmp(id, "esc_opens") == 0) {
        opts.esc_opens = (val != 0);
        dirty_bridge = true;
    } else if (std::strcmp(id, "mute_bg") == 0) {
        opts.mute_in_background = (val != 0);
        dirty_bridge = true;
    } else if (std::strcmp(id, "pause_bg") == 0) {
        opts.pause_in_background = (val != 0);
        dirty_bridge = true;
    }

    if (dirty_bridge) {
        bridge.SetOptions(opts, bump_prompts);
    }
    if (dirty_settings) {
        bridge.BumpSettingsGeneration();
    }
    if (request_save) {
        bridge.RequestSaveConfig();
    }
}

void OotMenuController::TriggerRowAction(const char* id, Bridge& bridge) {
    if (std::strcmp(id, "apply_restart") == 0) {
        confirm_action_id_.clear();
        bridge.RequestRestartGame();
        return;
    }
    if (std::strcmp(id, "reset_controls") == 0) {
        if (confirm_action_id_ == id) {
            confirm_action_id_.clear();
            bridge.RequestResetControls();
        } else {
            confirm_action_id_ = id;
        }
        return;
    }
    if (std::strcmp(id, "restart_game") == 0 || std::strcmp(id, "quit_game") == 0 ||
        std::strcmp(id, "stop_game") == 0) {
        if (confirm_action_id_ == id) {
            confirm_action_id_.clear();
            if (std::strcmp(id, "restart_game") == 0) {
                bridge.RequestRestartGame();
            } else if (std::strcmp(id, "stop_game") == 0) {
                bridge.RequestStopGame();
            } else {
                bridge.RequestQuitGame();
            }
        } else {
            confirm_action_id_ = id;
        }
    }
}

bool OotMenuController::IsRowDisabled(const char* id, const TelemetrySnapshot& tel) const {
    if (std::strcmp(id, "fsr_output") == 0 || std::strcmp(id, "fsr_render") == 0 ||
        std::strcmp(id, "rcas_sharp") == 0) {
        return !Settings::values.ocarina_fsr.GetValue();
    }
    if (std::strcmp(id, "oot_aa") == 0) {
        // The FSR replaces the game's output pass, so this toggle has no effect with it on (E47)
        return Settings::values.ocarina_fsr.GetValue();
    }
    if (std::strcmp(id, "fg_mult") == 0) {
        return !Settings::values.frame_gen.GetValue();
    }
    if (std::strcmp(id, "apply_restart") == 0) {
        return CollectPendingRowNames(Bridge::Instance().GetOptions()).empty() && !tel.needs_restart;
    }
    return false;
}

std::string OotMenuController::FormatRowValue(const RowDef& r, int val,
                                              const TelemetrySnapshot& tel) const {
    const char* id = r.id;
    if (r.kind == RowKind::Action) {
        if (confirm_action_id_ == id) {
            return Tr("Confirmar novamente");
        }
        return "";
    }
    if (std::strcmp(id, "rebind") == 0) {
        return Tr(kSwitchActionLabels[static_cast<std::size_t>(std::clamp(val, 0, 15))]);
    }
    if (std::strcmp(id, "preset") == 0) {
        if (val < 1 || val > static_cast<int>(kPresets.size())) {
            return Tr("Personalizado");
        }
        return Tr(kPresets[static_cast<std::size_t>(val - 1)].label);
    }
    if (std::strcmp(id, "upscaler") == 0) {
        if (val == 0) {
            return Tr("Nativo (Desligado)");
        }
        if (val == 2) {
            if (!tel.fsr4_available) {
                return std::string(Tr("FSR 4.1")) + " (GPU N/A)";
            }
            return tel.fsr4_int8 ? "FSR 4.1 (INT8)" : "FSR 4.1";
        }
        return Tr("FSR 3.1");
    }
    if (std::strcmp(id, "fsr_output") == 0) {
        static constexpr std::array<const char*, 4> kLabs{"1080p (Full HD)", "1440p (QHD)",
                                                         "4K (UHD)", "1440p no passo do jogo"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 3))]);
    }
    if (std::strcmp(id, "fsr_render") == 0) {
        static constexpr std::array<const char*, 3> kLabs{"720p (Leve)", "1080p (Equilibrado)",
                                                         "1440p (Máxima)"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 2))]);
    }
    if (std::strcmp(id, "res_scale") == 0) {
        static constexpr std::array<const char*, 7> kLabs{
            "0.5x (Metade)", "0.75x", "1x (Nativa)", "1.25x", "1.5x", "2x", "3x"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 6))]);
    }
    if (std::strcmp(id, "rcas_sharp") == 0) {
        if (val <= 0) {
            return Tr("Desligada (0%)");
        }
        char buf[16];
        if (val % 2 == 0) {
            std::snprintf(buf, sizeof(buf), "%d%%", val / 2);
        } else {
            std::snprintf(buf, sizeof(buf), "%d,5%%", val / 2);
        }
        return buf;
    }
    if (std::strcmp(id, "volume") == 0) {
        return std::to_string(val) + "%";
    }
    if (std::strcmp(id, "oot_aa") == 0) {
        return Tr(val != 0 ? "Sem Nitidez" : "Nitidez Original");
    }
    if (std::strcmp(id, "win_aa") == 0) {
        static constexpr std::array<const char*, 3> kLabs{"Nenhum", "FXAA", "SMAA"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 2))]);
    }
    if (std::strcmp(id, "win_filter") == 0) {
        static constexpr std::array<const char*, 8> kLabs{
            "Mais Próximo", "Bilinear", "Bicúbico", "Lanczos",
            "ScaleForce",   "FSR",      "Area",     "SGSR"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 7))]);
    }
    if (std::strcmp(id, "aniso") == 0) {
        static constexpr std::array<const char*, 6> kLabs{"Automático", "Padrão", "2x",
                                                         "4x",         "8x",     "16x"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 5))]);
    }
    if (std::strcmp(id, "docked") == 0) {
        return Tr(val != 0 ? "Dock (Alta Res.)" : "Portátil (720p)");
    }
    if (std::strcmp(id, "vsync") == 0) {
        static constexpr std::array<const char*, 4> kLabs{
            "Imediato (Desligado)", "Mailbox (Sem Tearing)", "FIFO (VSync)", "FIFO Relaxado"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 3))]);
    }
    if (std::strcmp(id, "hdr") == 0) {
        static constexpr std::array<const char*, 3> kLabs{"Desligado", "Ligado", "Automático"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 2))]);
    }
    if (std::strcmp(id, "frame_gen") == 0) {
        static constexpr std::array<const char*, 4> kLabs{
            "Desligado", "LSFG (Lossless Scaling)", "Fluxo óptico (Lichii)", "DLSS Frame Gen (NVIDIA)"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 3))]);
    }
    if (std::strcmp(id, "fg_mult") == 0) {
        return std::to_string(val) + "x";
    }
    if (std::strcmp(id, "nx2_fps") == 0) {
        return std::to_string(kNx2FpsChoices[static_cast<std::size_t>(std::clamp(val, 0, 6))]) +
               " FPS";
    }
    if (std::strcmp(id, "speed_limit") == 0) {
        static constexpr std::array<const char*, 6> kLabs{
            "Ilimitado",        "100% (60 FPS base)",  "150% (90 FPS base)",
            "200% (120 FPS base)", "240% (144 FPS base)", "400% (240 FPS base)"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 5))]);
    }
    if (std::strcmp(id, "gpu_acc") == 0) {
        return Tr(val != 0 ? "Alta" : "Baixa (Normal)");
    }
    if (std::strcmp(id, "vram") == 0) {
        return Tr(val != 0 ? "Agressivo" : "Conservador");
    }
    if (std::strcmp(id, "precache") == 0) {
        static constexpr std::array<const char*, 3> kLabs{"Desligado", "Perguntar", "Sempre"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 2))]);
    }
    if (std::strcmp(id, "idle_spin") == 0) {
        const int us = kIdleSpinChoices[static_cast<std::size_t>(std::clamp(val, 0, 5))];
        if (us < 0) {
            return Tr("Padrão (-1)");
        }
        return std::to_string(us) + " us";
    }
    if (std::strcmp(id, "prompts") == 0) {
        static constexpr std::array<const char*, 7> kLabs{
            "Automático", "Switch", "Xbox", "PS4", "PS5", "Steam Deck", "Teclado"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 6))]);
    }
    if (std::strcmp(id, "pad_layout") == 0) {
        static constexpr std::array<const char*, 3> kLabs{
            "Nintendo Posicional", "Xbox PC (A=Baixo, B=Dir)", "Personalizado"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 2))]);
    }
    if (std::strcmp(id, "audio_sink") == 0) {
        static constexpr std::array<const char*, 4> kLabs{"Automático", "Cubeb", "SDL3", "Null"};
        return kLabs[static_cast<std::size_t>(std::clamp(val, 0, 3))];
    }
    if (std::strcmp(id, "fps_hud") == 0) {
        static constexpr std::array<const char*, 6> kLabs{
            "Desligado", "Simples", "Frame time + 1%", "Gráfico", "Detalhado", "Dev"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 5))]);
    }
    if (std::strcmp(id, "fps_corner") == 0) {
        static constexpr std::array<const char*, 4> kLabs{
            "Superior Direito", "Superior Esquerdo", "Inferior Direito", "Inferior Esquerdo"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 3))]);
    }
    if (std::strcmp(id, "menu_size") == 0) {
        static constexpr std::array<const char*, 4> kLabs{"Automático", "Pequeno", "Médio", "Grande"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 3))]);
    }
    if (std::strcmp(id, "menu_lang") == 0) {
        return val == 1 ? "English" : "Português (BR)";
    }
    if (std::strcmp(id, "menu_key") == 0) {
        static constexpr std::array<const char*, 3> kLabs{"Insert ou F1", "Insert", "F1"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 2))]);
    }
    if (std::strcmp(id, "compact_key") == 0) {
        static constexpr std::array<const char*, 3> kLabs{"RShift", "RCtrl", "F4"};
        return Tr(kLabs[static_cast<std::size_t>(std::clamp(val, 0, 2))]);
    }
    if (r.kind == RowKind::Toggle) {
        return Tr(val != 0 ? "Ligado" : "Desligado");
    }
    return std::to_string(val);
}

void OotMenuController::ClampSelection(const std::vector<const RowDef*>& tab_rows) {
    if (tab_rows.empty()) {
        return;
    }
    int& sel = tab_sel_[static_cast<std::size_t>(cur_tab_)];
    sel = std::clamp(sel, 0, static_cast<int>(tab_rows.size()) - 1);
    if (tab_rows[static_cast<std::size_t>(sel)]->kind == RowKind::Heading) {
        for (int i = 0; i < static_cast<int>(tab_rows.size()); ++i) {
            if (tab_rows[static_cast<std::size_t>(i)]->kind != RowKind::Heading) {
                sel = i;
                break;
            }
        }
    }
}

void OotMenuController::MoveSelection(int delta, const std::vector<const RowDef*>& tab_rows) {
    if (tab_rows.empty()) {
        return;
    }
    confirm_action_id_.clear();
    const int n = static_cast<int>(tab_rows.size());
    int& sel = tab_sel_[static_cast<std::size_t>(cur_tab_)];
    int next = sel;
    for (int step = 0; step < n; ++step) {
        next = (next + delta + n) % n;
        if (tab_rows[static_cast<std::size_t>(next)]->kind != RowKind::Heading) {
            sel = next;
            return;
        }
    }
}

void OotMenuController::AdjustRow(const RowDef& r, int delta, Bridge& bridge) {
    const auto tel = bridge.GetTelemetry();
    if (IsRowDisabled(r.id, tel)) {
        return;
    }
    auto opts = bridge.GetOptions();
    if (r.kind == RowKind::Action) {
        if (delta == 0) {
            TriggerRowAction(r.id, bridge);
        }
        return;
    }
    int cur = ReadRowValue(r.id, opts);
    int next = cur;
    if (std::strcmp(r.id, "rebind") == 0) {
        // Arrows only pick the button; A (or a click) starts the capture.
        if (delta == 0) {
            WriteRowValue(r.id, cur, opts, bridge);
        } else {
            rebind_pick_ = r.lo + ((cur - r.lo + delta + (r.hi - r.lo + 1)) % (r.hi - r.lo + 1));
        }
        return;
    }
    if (r.kind == RowKind::Slider) {
        next = std::clamp(cur + delta * r.step, r.lo, r.hi);
    } else if (r.kind == RowKind::Toggle) {
        next = (delta == 0) ? (1 - cur) : std::clamp(cur + delta, 0, 1);
    } else {
        const int span = r.hi - r.lo + 1;
        if (delta == 0) {
            next = r.lo + ((cur - r.lo + 1) % span);
        } else {
            next = r.lo + ((cur - r.lo + delta + span) % span);
        }
    }
    if (next != cur) {
        confirm_action_id_.clear();
        WriteRowValue(r.id, next, opts, bridge);
    }
}

void OotMenuController::OpenDropdown(const RowDef& r, const PcOptions& opts) {
    dd_row_ = &r;
    const int cur = ReadRowValue(r.id, opts);
    dd_sel_ = (cur >= r.lo && cur <= r.hi) ? cur : r.lo;
    dd_scroll_ = 0;
    dd_hover_ = -1;
    confirm_action_id_.clear();
}

void OotMenuController::CloseDropdown() {
    dd_row_ = nullptr;
    dd_hover_ = -1;
}

void OotMenuController::MoveDropdown(int delta) {
    if (dd_row_ == nullptr) {
        return;
    }
    dd_sel_ = std::clamp(dd_sel_ + delta, dd_row_->lo, dd_row_->hi);
}

void OotMenuController::ApplyDropdown(Bridge& bridge) {
    if (dd_row_ == nullptr) {
        return;
    }
    const RowDef& r = *dd_row_;
    CloseDropdown();
    if (IsRowDisabled(r.id, bridge.GetTelemetry())) {
        return;
    }
    auto opts = bridge.GetOptions();
    if (ReadRowValue(r.id, opts) != dd_sel_ || std::strcmp(r.id, "rebind") == 0) {
        WriteRowValue(r.id, dd_sel_, opts, bridge);
    }
}

void OotMenuController::ProcessInput(Bridge& bridge, float dt_sec) {
    if (!bridge.IsOpen()) {
        prev_pad_mask_ = 0;
        held_dir_ = 0;
        CloseDropdown();
        return;
    }
    if (bridge.RebindTarget() >= 0) {
        // Capturing a button for the remap: the press belongs to the host, not to the menu.
        std::vector<Common::OotMenu::KeyEvent> keys;
        bridge.DrainKeyEvents(keys);
        for (const auto& ev : keys) {
            if (ev.down && (ev.key == Key::Back || ev.key == Key::Esc)) {
                bridge.CancelRebind();
            }
        }
        prev_pad_mask_ = bridge.GamepadNavMask();
        held_dir_ = 0;
        const auto ptr = bridge.ConsumePointer();
        if ((ptr.pressed_mask & 2u) != 0) {
            bridge.CancelRebind();
        }
        return;
    }
    auto tab_rows = RowsForTab(cur_tab_);
    ClampSelection(tab_rows);

    auto switch_tab = [&](int delta) {
        confirm_action_id_.clear();
        const int count = static_cast<int>(Tab::Count);
        cur_tab_ = static_cast<Tab>((static_cast<int>(cur_tab_) + delta + count) % count);
        tab_rows = RowsForTab(cur_tab_);
        ClampSelection(tab_rows);
    };

    auto apply_nav = [&](int up_down, int left_right, bool confirm, bool back, int tab_delta) {
        if (DropdownOpen()) {
            if (tab_delta != 0) {
                CloseDropdown();
            } else {
                MoveDropdown(up_down + left_right);
                if (confirm) {
                    ApplyDropdown(bridge);
                } else if (back) {
                    CloseDropdown();
                }
                return;
            }
        }
        if (tab_delta != 0) {
            switch_tab(tab_delta);
        }
        if (up_down != 0) {
            MoveSelection(up_down, tab_rows);
        }
        if (left_right != 0 && !tab_rows.empty()) {
            const int sel = tab_sel_[static_cast<std::size_t>(cur_tab_)];
            AdjustRow(*tab_rows[static_cast<std::size_t>(sel)], left_right, bridge);
        }
        if (confirm && !tab_rows.empty()) {
            const int sel = tab_sel_[static_cast<std::size_t>(cur_tab_)];
            const RowDef& row = *tab_rows[static_cast<std::size_t>(sel)];
            if (row.kind == RowKind::Choice && !IsRowDisabled(row.id, bridge.GetTelemetry())) {
                OpenDropdown(row, bridge.GetOptions());
            } else {
                AdjustRow(row, 0, bridge);
            }
        }
        if (back) {
            if (!confirm_action_id_.empty()) {
                confirm_action_id_.clear();
            } else {
                bridge.CloseMenu();
            }
        }
    };

    std::vector<Common::OotMenu::KeyEvent> keys;
    bridge.DrainKeyEvents(keys);
    bool kbd_confirm_or_back = false;
    for (const auto& ev : keys) {
        if (ev.down && (ev.key == Key::Confirm || ev.key == Key::Back || ev.key == Key::Esc)) {
            kbd_confirm_or_back = true;
        }
        if (!ev.down) {
            continue;
        }
        switch (ev.key) {
        case Key::Up:
            apply_nav(-1, 0, false, false, 0);
            break;
        case Key::Down:
            apply_nav(1, 0, false, false, 0);
            break;
        case Key::Left:
            apply_nav(0, -1, false, false, 0);
            break;
        case Key::Right:
            apply_nav(0, 1, false, false, 0);
            break;
        case Key::PageUp:
            if (DropdownOpen()) {
                MoveDropdown(-5);
            } else {
                MoveSelection(-5, tab_rows);
            }
            break;
        case Key::PageDown:
            if (DropdownOpen()) {
                MoveDropdown(5);
            } else {
                MoveSelection(5, tab_rows);
            }
            break;
        case Key::Home:
            if (DropdownOpen()) {
                MoveDropdown(-1000);
                break;
            }
            tab_sel_[static_cast<std::size_t>(cur_tab_)] = 0;
            ClampSelection(tab_rows);
            break;
        case Key::End:
            if (DropdownOpen()) {
                MoveDropdown(1000);
                break;
            }
            tab_sel_[static_cast<std::size_t>(cur_tab_)] =
                static_cast<int>(tab_rows.size()) - 1;
            MoveSelection(0, tab_rows);
            break;
        case Key::Confirm:
            apply_nav(0, 0, true, false, 0);
            break;
        case Key::Back:
        case Key::Esc:
            apply_nav(0, 0, false, true, 0);
            break;
        case Key::TabPrev:
            apply_nav(0, 0, false, false, -1);
            break;
        case Key::TabNext:
            apply_nav(0, 0, false, false, 1);
            break;
        default:
            break;
        }
    }

    // Gamepad navigation with smooth initial hold delay + fast repeat
    bool from_kbd = false;
    const u64 mask = bridge.GamepadNavMask(&from_kbd);
    const u64 pressed = mask & ~prev_pad_mask_;
    prev_pad_mask_ = mask;

    constexpr u64 kBitA = 1ULL << 0;
    constexpr u64 kBitB = 1ULL << 1;
    constexpr u64 kBitL = 1ULL << 6;
    constexpr u64 kBitR = 1ULL << 7;
    constexpr u64 kBitPlus = 1ULL << 10;
    constexpr u64 kBitX = 1ULL << 2;
    constexpr u64 kBitY = 1ULL << 3;
    // E47.T4: the left face button by position switches full menu <-> compact bar, as in
    // Longshot (HOS_PAD_WEST). Switch Y is on the left in the Nintendo layout; with the Xbox PC
    // layout the buttons follow the Xbox labels, so the left one is X.
    const u64 kBitWest =
        bridge.GetOptions().pad_layout == static_cast<int>(Common::OotMenu::PadLayout::Nintendo)
            ? kBitY
            : kBitX;
    constexpr u64 kBitLeft = (1ULL << 12) | (1ULL << 16);
    constexpr u64 kBitUp = (1ULL << 13) | (1ULL << 17);
    constexpr u64 kBitRight = (1ULL << 14) | (1ULL << 18);
    constexpr u64 kBitDown = (1ULL << 15) | (1ULL << 19);

    int cur_dir = 0;
    if (mask & kBitUp) {
        cur_dir = 1;
    } else if (mask & kBitDown) {
        cur_dir = 2;
    } else if (mask & kBitLeft) {
        cur_dir = 3;
    } else if (mask & kBitRight) {
        cur_dir = 4;
    }

    bool trigger_dir = false;
    if (cur_dir == 0) {
        held_dir_ = 0;
        dir_hold_timer_ = 0.0f;
        dir_repeat_timer_ = 0.0f;
    } else if (cur_dir != held_dir_) {
        held_dir_ = cur_dir;
        dir_hold_timer_ = 0.0f;
        dir_repeat_timer_ = 0.0f;
        trigger_dir = true;
    } else {
        dir_hold_timer_ += dt_sec;
        if (dir_hold_timer_ >= 0.34f) {
            dir_repeat_timer_ += dt_sec;
            if (dir_repeat_timer_ >= 0.065f) {
                dir_repeat_timer_ = 0.0f;
                trigger_dir = true;
            }
        }
    }

    if (trigger_dir) {
        apply_nav(held_dir_ == 1 ? -1 : (held_dir_ == 2 ? 1 : 0),
                  held_dir_ == 3 ? -1 : (held_dir_ == 4 ? 1 : 0), false, false, 0);
    }
    if ((pressed & kBitWest) != 0 && !from_kbd && !DropdownOpen()) {
        // ToggleCompact alone would close the menu from the compact bar
        bridge.IsCompact() ? bridge.ToggleMenu() : bridge.ToggleCompact();
    }
    if ((pressed & kBitL) != 0) {
        apply_nav(0, 0, false, false, -1);
    }
    if ((pressed & kBitR) != 0) {
        apply_nav(0, 0, false, false, 1);
    }
    // A keyboard key mapped to Player 1 can arrive here as A/B in the same tick as the key event;
    // handling both would open and apply a drop-down in one press.
    if ((pressed & kBitA) != 0 && !kbd_confirm_or_back) {
        apply_nav(0, 0, true, false, 0);
    }
    if ((pressed & (kBitB | kBitPlus)) != 0 && !kbd_confirm_or_back) {
        apply_nav(0, 0, false, true, 0);
    }

    ProcessPointer(bridge, tab_rows);
}

void OotMenuController::ProcessPointer(Bridge& bridge, const std::vector<const RowDef*>& tab_rows) {
    const auto ptr = bridge.ConsumePointer();
    const float mx = ptr.x - menu_ox_;
    const float my = ptr.y;
    const std::size_t tab_idx = static_cast<std::size_t>(cur_tab_);

    if (DropdownOpen()) {
        hover_tab_ = -1;
        hover_apply_ = false;
        hover_compact_ = false;
        if ((ptr.pressed_mask & 2u) != 0) {
            CloseDropdown();
            return;
        }
        if (ptr.wheel_y != 0.0f) {
            MoveDropdown(ptr.wheel_y > 0.0f ? -1 : 1);
        }
        const bool pressed = (ptr.pressed_mask & 1u) != 0;
        for (const auto& hb : hits_) {
            if (hb.kind != HitKind::DropItem) {
                continue;
            }
            if (mx >= hb.x0 && mx <= hb.x1 && my >= hb.y0 && my <= hb.y1) {
                if (ptr.moved) {
                    dd_sel_ = hb.index;
                }
                if (pressed) {
                    dd_sel_ = hb.index;
                    ApplyDropdown(bridge);
                }
                return;
            }
        }
        if (pressed) {
            CloseDropdown();
        }
        return;
    }

    if (ptr.wheel_y != 0.0f && !bridge.IsCompact()) {
        const float max_scroll = std::max(0.0f, static_cast<float>(tab_rows.size()) - 8.0f);
        scroll_want_[tab_idx] =
            std::clamp(scroll_want_[tab_idx] - ptr.wheel_y * 2.0f, 0.0f, max_scroll);
    }

    hover_tab_ = -1;
    hover_apply_ = false;
    hover_compact_ = false;
    if (ptr.moved) {
        hover_row_ = -1;
    }

    const bool l_down = (ptr.held_mask & 1u) != 0;
    const bool l_pressed = (ptr.pressed_mask & 1u) != 0;
    const bool r_pressed = (ptr.pressed_mask & 2u) != 0;

    if (!l_down) {
        dragging_slider_row_ = -1;
    }

    if (r_pressed) {
        bridge.CloseMenu();
        return;
    }

    for (const auto& hb : hits_) {
        const bool inside = (mx >= hb.x0 && mx <= hb.x1 && my >= hb.y0 && my <= hb.y1);
        if (hb.kind == HitKind::SliderTrack && dragging_slider_row_ == hb.index && l_down &&
            !tab_rows.empty() && hb.index >= 0 && hb.index < static_cast<int>(tab_rows.size())) {
            const float denom = std::max(1.0f, hb.tx1 - hb.tx0);
            const float norm = std::clamp((mx - hb.tx0) / denom, 0.0f, 1.0f);
            const auto& r = *tab_rows[static_cast<std::size_t>(hb.index)];
            const int raw =
                r.lo + static_cast<int>(std::lround(norm * static_cast<float>(r.hi - r.lo)));
            const int snapped = ((raw - r.lo + r.step / 2) / r.step) * r.step + r.lo;
            auto opts = bridge.GetOptions();
            if (ReadRowValue(r.id, opts) != snapped) {
                WriteRowValue(r.id, snapped, opts, bridge);
            }
            return;
        }
        if (!inside) {
            continue;
        }
        switch (hb.kind) {
        case HitKind::Tab:
            hover_tab_ = hb.index;
            if (l_pressed) {
                cur_tab_ = static_cast<Tab>(hb.index);
                confirm_action_id_.clear();
                const auto next_rows = RowsForTab(cur_tab_);
                ClampSelection(next_rows);
            }
            return;
        case HitKind::TabPrev:
        case HitKind::TabNext:
            if (l_pressed) {
                const int d = (hb.kind == HitKind::TabPrev) ? -1 : 1;
                const int count = static_cast<int>(Tab::Count);
                cur_tab_ =
                    static_cast<Tab>((static_cast<int>(cur_tab_) + d + count) % count);
                confirm_action_id_.clear();
                const auto next_rows = RowsForTab(cur_tab_);
                ClampSelection(next_rows);
            }
            return;
        case HitKind::ApplyRestart:
            hover_apply_ = true;
            if (l_pressed) {
                bridge.RequestRestartGame();
            }
            return;
        case HitKind::BackToCompact:
            hover_compact_ = true;
            if (l_pressed) {
                bridge.IsCompact() ? bridge.ToggleMenu() : bridge.ToggleCompact();
            }
            return;
        case HitKind::DropItem:
            continue;
        case HitKind::Row:
        case HitKind::SliderTrack:
            if (hb.index >= 0 && hb.index < static_cast<int>(tab_rows.size())) {
                const auto& r = *tab_rows[static_cast<std::size_t>(hb.index)];
                if (r.kind != RowKind::Heading) {
                    if (ptr.moved) {
                        hover_row_ = hb.index;
                        tab_sel_[tab_idx] = hb.index;
                        last_scrolled_[tab_idx] = hb.index;
                    }
                    if (l_pressed) {
                        tab_sel_[tab_idx] = hb.index;
                        if (r.kind == RowKind::Slider && hb.tx1 > hb.tx0) {
                            dragging_slider_row_ = hb.index;
                            const float norm =
                                std::clamp((mx - hb.tx0) / (hb.tx1 - hb.tx0), 0.0f, 1.0f);
                            const int raw =
                                r.lo +
                                static_cast<int>(std::lround(norm * static_cast<float>(r.hi - r.lo)));
                            const int snapped =
                                ((raw - r.lo + r.step / 2) / r.step) * r.step + r.lo;
                            auto opts = bridge.GetOptions();
                            WriteRowValue(r.id, snapped, opts, bridge);
                        } else if (mx >= hb.vx0 && mx <= hb.vx1) {
                            const float vmid = (hb.vx0 + hb.vx1) * 0.5f;
                            if (r.kind == RowKind::Toggle) {
                                auto opts = bridge.GetOptions();
                                WriteRowValue(r.id, mx >= vmid ? 1 : 0, opts, bridge);
                            } else if (r.kind == RowKind::Action) {
                                TriggerRowAction(r.id, bridge);
                            } else {
                                // Arrow zones step the value; the centre opens the list.
                                const float arrow_w = 38.0f * (hb.y1 - hb.y0) / 37.0f;
                                if (mx < hb.vx0 + arrow_w) {
                                    AdjustRow(r, -1, bridge);
                                } else if (mx > hb.vx1 - arrow_w) {
                                    AdjustRow(r, 1, bridge);
                                } else if (r.kind == RowKind::Choice &&
                                           !IsRowDisabled(r.id, bridge.GetTelemetry())) {
                                    OpenDropdown(r, bridge.GetOptions());
                                } else {
                                    AdjustRow(r, mx < vmid ? -1 : 1, bridge);
                                }
                            }
                        } else if (r.kind == RowKind::Action) {
                            TriggerRowAction(r.id, bridge);
                        }
                    }
                }
            }
            return;
        }
    }
}

void OotMenuController::DrawDropdown(Ctx& c, ImVec2 a, ImVec2 b, bool prefer_up,
                                     const TelemetrySnapshot& tel, const PcOptions& opts) {
    if (dd_row_ == nullptr) {
        return;
    }
    const RowDef& r = *dd_row_;
    const float s = c.s;
    const int count = r.hi - r.lo + 1;
    const int vis = std::min(count, 8);
    const float item_h = 32.0f * s;
    const float pad = 4.0f * s;
    const float px = 16.5f * s;
    const int cur = ReadRowValue(r.id, opts);

    std::vector<std::string> labels;
    labels.reserve(static_cast<std::size_t>(count));
    float widest = 0.0f;
    for (int v = r.lo; v <= r.hi; ++v) {
        labels.push_back(FormatRowValue(r, v, tel));
        widest = std::max(widest, SizeOfRaw(c.f.strong, px, labels.back().c_str()).x);
    }
    const float w = std::clamp(widest + 56.0f * s, b.x - a.x, c.w - 2.0f * c.safe);
    float x0 = std::min(a.x, c.w - c.safe - w);
    x0 = std::max(x0, c.safe);
    const float x1 = x0 + w;
    const float h = static_cast<float>(vis) * item_h + 2.0f * pad;
    const bool fits_below = b.y + 4.0f * s + h <= c.h - c.safe;
    const bool fits_above = a.y - 4.0f * s - h >= c.safe;
    const bool up = (prefer_up && fits_above) || (!fits_below && fits_above);
    const float y0 = up ? (a.y - 4.0f * s - h) : (b.y + 4.0f * s);
    const float y1 = y0 + h;

    const int sel_idx = dd_sel_ - r.lo;
    if (sel_idx < dd_scroll_) {
        dd_scroll_ = sel_idx;
    }
    if (sel_idx >= dd_scroll_ + vis) {
        dd_scroll_ = sel_idx - vis + 1;
    }
    dd_scroll_ = std::clamp(dd_scroll_, 0, std::max(0, count - vis));

    Shadow(c, ImVec2(x0, y0), ImVec2(x1, y1), 14.0f * s, 0.55f, 4.0f * s);
    c.d->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), kBox, 4.0f * s);
    c.d->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), kBoxEdge, 4.0f * s, std::max(1.0f, 1.5f * s));

    std::vector<HitBox> items;
    for (int k = 0; k < vis; ++k) {
        const int idx = dd_scroll_ + k;
        const int v = r.lo + idx;
        const ImVec2 ia(x0 + pad, y0 + pad + static_cast<float>(k) * item_h);
        const ImVec2 ib(x1 - pad, ia.y + item_h);
        const bool is_sel = (v == dd_sel_);
        if (is_sel) {
            GoldFill(c, ia, ib);
        } else if ((k & 1) == 0) {
            c.d->AddRectFilled(ia, ib, Rgba(255, 255, 255, 10));
        }
        if (v == cur) {
            Diamond(c, ImVec2(ia.x + 14.0f * s, (ia.y + ib.y) * 0.5f), 4.5f * s,
                    is_sel ? kGoldInk : kGoldHi);
        }
        const std::string shown = FitEllipsis(c, c.f.strong, px, ib.x - ia.x - 40.0f * s,
                                              labels[static_cast<std::size_t>(idx)].c_str());
        TextMidRaw(c, is_sel ? c.f.strong : c.f.body, px, ia.x + 30.0f * s, (ia.y + ib.y) * 0.5f,
                   is_sel ? kGoldInk : kInk, shown.c_str());
        items.push_back({ia.x, ia.y, ib.x, ib.y, HitKind::DropItem, v, 0, 0, 0, 0});
    }
    if (count > vis) {
        const float sx = x1 - 6.0f * s;
        const float span = y1 - y0 - 2.0f * pad;
        const float t0 = y0 + pad + span * static_cast<float>(dd_scroll_) / static_cast<float>(count);
        const float t1 =
            y0 + pad + span * static_cast<float>(dd_scroll_ + vis) / static_cast<float>(count);
        c.d->AddRectFilled(ImVec2(sx - 1.5f * s, t0), ImVec2(sx + 1.5f * s, t1), kAccent, s);
    }
    // Drop-down items must win hit tests over the rows below them.
    hits_.insert(hits_.begin(), items.begin(), items.end());
}

void OotMenuController::DrawCompactBar(Ctx& c, Bridge& bridge) {
    const float s = c.s;
    const auto opts = bridge.GetOptions();
    const auto tel = bridge.GetTelemetry();
    const auto tab_rows = RowsForTab(cur_tab_);
    ClampSelection(tab_rows);

    const int sel = tab_sel_[static_cast<std::size_t>(cur_tab_)];
    const auto& r = *tab_rows[static_cast<std::size_t>(sel)];
    const int val = ReadRowValue(r.id, opts);
    const bool pending = (r.requires_restart || std::strcmp(r.id, "upscaler") == 0) &&
                             IsRowPendingRestart(r.id, opts);
    const bool disabled = IsRowDisabled(r.id, tel);

    const float bw = std::min(c.w - 2.0f * c.safe, 760.0f * s);
    const float bx0 = (c.w - bw) * 0.5f;
    const float bx1 = bx0 + bw;
    const float bh = 112.0f * s;
    const float by1 = c.h - c.safe - 14.0f * s;
    const float by0 = by1 - bh;

    c.d->AddRectFilled(ImVec2(bx0, by0), ImVec2(bx1, by1), Rgba(14, 10, 6, 236), 6.0f * s);
    c.d->AddRect(ImVec2(bx0, by0), ImVec2(bx1, by1), Alpha(kEdge, 0.9f), 6.0f * s,
                 std::max(1.0f, s));

    {
        const char* lab = "Menu completo";
        const float pw = PillWidth(c, lab);
        const ImVec2 a(bx1 - pw, by0 - 36.0f * s);
        const ImVec2 b(bx1, by0 - 8.0f * s);
        PillButton(c, a, b, lab, hover_compact_);
        hits_.push_back({a.x, a.y, b.x, b.y, HitKind::BackToCompact, -1, 0, 0, 0, 0});
    }

    const float header_y = by0 + 14.0f * s;
    const char* cur_name = TabTitles()[static_cast<std::size_t>(cur_tab_)];
    const float nw = TextMid(c, c.f.heading, 14.5f * s, bx0 + 16.0f * s, header_y, kGoldHi, cur_name);
    const char* tab_hint = c.pad ? " (L / R)" : " (Q / E)";
    TextMidRaw(c, c.f.body, 12.5f * s, bx0 + 18.0f * s + nw, header_y, kDim, tab_hint);

    if (pending || r.requires_restart) {
        Diamond(c, ImVec2(bx1 - 20.0f * s, header_y), 4.0f * s, kGoldHi);
        TextMidRight(c, c.f.body, 12.5f * s, bx1 - 30.0f * s, header_y, kGoldHi,
                     pending ? "Após reiniciar" : "Requer reinício");
    } else {
        c.d->AddCircleFilled(ImVec2(bx1 - 20.0f * s, header_y), 3.5f * s, kLiveGreen);
        TextMidRight(c, c.f.body, 12.5f * s, bx1 - 30.0f * s, header_y, kLiveGreen, "Ao Vivo");
    }

    const float ry0 = by0 + 28.0f * s;
    const float ry1 = ry0 + 38.0f * s;
    const float rx0 = bx0 + 14.0f * s;
    const float rx1 = bx1 - 14.0f * s;
    const float split = rx0 + (rx1 - rx0) * 0.45f;

    DrawNameCell(c, ImVec2(rx0, ry0), ImVec2(split, ry1), r.label, true, pending,
                 !r.requires_restart);

    const ImVec2 va(split, ry0);
    const ImVec2 vb(rx1, ry1);
    float tx0 = 0.0f;
    float tx1 = 0.0f;
    const std::string val_str = FormatRowValue(r, val, tel);
    if (r.kind == RowKind::Slider) {
        DrawSliderCell(c, va, vb, val_str,
                       static_cast<int>((val - r.lo) * 1000LL / std::max(1, r.hi - r.lo)), true,
                       &tx0, &tx1);
    } else if (r.kind == RowKind::Toggle) {
        // Same labels as the full menu (E47: the compact bar said Off/On for the game sharpening)
        const bool oot_aa = std::strcmp(r.id, "oot_aa") == 0;
        const char* lo_l = std::strcmp(r.id, "gpu_acc") == 0
                               ? "Baixa (Normal)"
                               : (std::strcmp(r.id, "vram") == 0
                                      ? "Conservador"
                                      : (oot_aa ? "Nitidez Original" : "Desligado"));
        const char* hi_l = std::strcmp(r.id, "gpu_acc") == 0
                               ? "Alta"
                               : (std::strcmp(r.id, "vram") == 0 ? "Agressivo"
                                                                  : (oot_aa ? "Sem Nitidez" : "Ligado"));
        DrawHalvesCell(c, va, vb, lo_l, hi_l, val != 0, true, disabled);
    } else {
        DrawChoiceCell(c, va, vb, val_str, true, !disabled && r.kind != RowKind::Action,
                       !disabled && r.kind != RowKind::Action, disabled);
    }
    hits_.push_back({rx0, ry0, rx1, ry1,
                     r.kind == RowKind::Slider ? HitKind::SliderTrack : HitKind::Row, sel, va.x,
                     vb.x, tx0, tx1});

    if (dd_row_ == &r) {
        DrawDropdown(c, va, vb, true, tel, opts);
    } else if (dd_row_ != nullptr) {
        CloseDropdown();
    }

    const char* compact_k =
        opts.compact_key == 1 ? "RCtrl" : (opts.compact_key == 2 ? "F4" : "RShift");
    const std::array<Prompt, 4> pr{{
        {Glyph::UpDown, "Escolher"},
        {r.kind == RowKind::Action ? Glyph::Confirm : Glyph::LeftRight, "Alterar"},
        {Glyph::Compact, "Menu completo", compact_k},
        {Glyph::Back, "Fechar"},
    }};
    const float pw = Prompts(c, ImVec2(0.0f, 0.0f), pr.data(), 4, false);
    Prompts(c, ImVec2(bx0 + (bw - pw) * 0.5f, by1 - 33.0f * s), pr.data(), 4, true);
}

void OotMenuController::DrawControlsBindings(Ctx& c, float x0, float x1, float top, float bottom,
                                             const Common::OotMenu::PadBindingSnapshot& snap) {
    const float s = c.s;
    const float pad_top = top + 12.0f * s;
    const float row_h = 22.0f * s;
    const float col1_x = x0 + 18.0f * s;
    const float col2_x = x0 + (x1 - x0) * 0.42f;
    const float col3_x = x0 + (x1 - x0) * 0.72f;

    const std::string header = std::string(Tr("Mapeamento Jogador 1")) +
                               (snap.pad_connected && !snap.active_pad_name.empty()
                                    ? (" · " + snap.active_pad_name)
                                    : "");
    TextMidRaw(c, c.f.heading, 15.0f * s, col1_x, pad_top + 10.0f * s, kAccent, header.c_str());

    const auto& bridge = Common::OotMenu::Bridge::Instance();
    const int target = bridge.RebindTarget();
    double msg_age = 1e9;
    const std::string msg = bridge.RebindMessage(&msg_age);
    float list_top = pad_top + 26.0f * s;
    if (target >= 0 || msg_age < 5.0) {
        const ImVec2 na(x0 + 10.0f * s, list_top);
        const ImVec2 nb(x1 - 10.0f * s, list_top + 30.0f * s);
        std::string line;
        if (target >= 0) {
            GoldFill(c, na, nb);
            line = std::string(Tr("Aperte no controle o botão para")) + " " +
                   Tr(kSwitchActionLabels[static_cast<std::size_t>(target)]) + "  ·  " +
                   Tr("Esc ou botão direito cancela");
        } else {
            c.d->AddRectFilled(na, nb, kCellNameSel);
            line = Tr(msg.c_str());
        }
        TextMidRaw(c, c.f.strong, 14.5f * s, na.x + 10.0f * s, (na.y + nb.y) * 0.5f,
                   target >= 0 ? kGoldInk : kInkBright, line.c_str());
        list_top += 34.0f * s;
    }
    const int fit = std::max(1, static_cast<int>((bottom - list_top) / row_h));
    const int first = (target >= fit) ? (target - fit + 1) : 0;
    const int n = std::min(fit, static_cast<int>(kSwitchActionLabels.size()) - first);
    for (int k = 0; k < n; ++k) {
        const int i = first + k;
        const float ry = list_top + static_cast<float>(k) * row_h;
        if (i == target) {
            c.d->AddRect(ImVec2(x0 + 10.0f * s, ry), ImVec2(x1 - 10.0f * s, ry + row_h - 2.0f * s),
                         kGoldHi, 0.0f, std::max(1.5f, 2.0f * s));
        }
        if ((i & 1) == 0) {
            c.d->AddRectFilled(ImVec2(x0 + 10.0f * s, ry), ImVec2(x1 - 10.0f * s, ry + row_h - 2.0f * s),
                               Rgba(24, 18, 12, 170));
        }
        const float ym = ry + row_h * 0.5f;
        TextMid(c, c.f.body, 14.0f * s, col1_x, ym, kInk, kSwitchActionLabels[static_cast<std::size_t>(i)]);
        const char* p_str = snap.pad_names[static_cast<std::size_t>(i)].empty()
                                ? "-"
                                : snap.pad_names[static_cast<std::size_t>(i)].c_str();
        const char* k_str = snap.key_names[static_cast<std::size_t>(i)].empty()
                                ? "-"
                                : snap.key_names[static_cast<std::size_t>(i)].c_str();
        TextMidRaw(c, c.f.strong, 14.0f * s, col2_x, ym, kGoldHi, p_str);
        TextMidRaw(c, c.f.body, 14.0f * s, col3_x, ym, kSoft, k_str);
    }
}

void OotMenuController::DrawFullMenu(Ctx& c, Bridge& bridge) {
    const float s = c.s;
    const float hh = BannerHeight(c);
    const auto opts = bridge.GetOptions();
    const auto tel = bridge.GetTelemetry();
    const auto bindings = bridge.GetPadBindings();

    Banner(c, hh);

    // Centered tabs in banner
    std::array<float, 4> tw{};
    float total = 0.0f;
    for (int i = 0; i < 4; ++i) {
        tw[static_cast<std::size_t>(i)] =
            SizeOf(c.f.heading, 22.0f * s, TabTitles()[static_cast<std::size_t>(i)]).x + 52.0f * s;
        total += tw[static_cast<std::size_t>(i)];
    }
    float tx = (c.w - total) * 0.5f;
    const float ty = hh * 0.5f + s;
    const std::array<const char*, 2> tab_keys{c.pad ? "L" : "Q", c.pad ? "R" : "E"};
    for (int i = 0; i < 2; ++i) {
        const float cw = 26.0f * s;
        const float cx = (i == 1) ? (tx + total + 20.0f * s) : (tx - 20.0f * s - cw);
        c.d->AddRectFilled(ImVec2(cx, ty - 13.0f * s), ImVec2(cx + cw, ty + 13.0f * s), kBannerInk,
                           3.0f * s);
        TextMidCentreRaw(c, c.f.strong, 14.0f * s, cx + cw * 0.5f, ty, kInk,
                         tab_keys[static_cast<std::size_t>(i)]);
        hits_.push_back({(i == 1) ? cx - 10.0f * s : 0.0f, 0.0f, (i == 1) ? c.w : cx + cw + 10.0f * s,
                         hh, (i == 1) ? HitKind::TabNext : HitKind::TabPrev, -1, 0, 0, 0, 0});
    }

    for (int i = 0; i < 4; ++i) {
        const bool shown = (i == static_cast<int>(cur_tab_));
        const bool hover = (i == hover_tab_);
        if (shown || hover) {
            c.d->AddRectFilled(ImVec2(tx, 0.0f), ImVec2(tx + tw[static_cast<std::size_t>(i)], hh - 3.0f * s),
                               shown ? kTabInset : Alpha(kTabInset, 0.35f));
        }
        TextMid(c, c.f.heading, 22.0f * s, tx + 26.0f * s, ty, shown ? kInkBright : kBannerInk,
                TabTitles()[static_cast<std::size_t>(i)]);
        hits_.push_back(
            {tx, 0.0f, tx + tw[static_cast<std::size_t>(i)], hh, HitKind::Tab, i, 0, 0, 0, 0});
        tx += tw[static_cast<std::size_t>(i)];
    }

    // Right of the banner: switch to the compact bar (E47.T4)
    {
        const char* lab = "Compacto";
        const float pw = PillWidth(c, lab);
        const ImVec2 a(c.w - c.safe - pw, ty - 14.0f * s);
        const ImVec2 b(c.w - c.safe, ty + 14.0f * s);
        if (a.x > tx + 60.0f * s) { // tx is past the last tab here
            PillButton(c, a, b, lab, hover_compact_);
            hits_.push_back({a.x, a.y, b.x, b.y, HitKind::BackToCompact, -1, 0, 0, 0, 0});
        }
    }

    // Left badge inside banner: the running game's icon, name and ID (E47.T12), read from its
    // NACP so the menu fits any game of the best build; "Lichii" until the core reports one.
    {
        float room = (c.w - total) * 0.5f - 54.0f * s - c.safe;
        float x = c.safe;
        if (c.has_game_icon && room > hh) {
            const float side = hh - 14.0f * s;
            const ImVec2 p0{x, (hh - 3.0f * s - side) * 0.5f};
            c.d->AddImageRounded(c.game_icon, p0, ImVec2(p0.x + side, p0.y + side), ImVec2(0, 0),
                                 ImVec2(1, 1), IM_COL32_WHITE, 6.0f * s);
            c.d->AddRect(p0, ImVec2(p0.x + side, p0.y + side), kBannerEdge, 6.0f * s, 0, 1.0f * s);
            x += side + 10.0f * s;
            room -= side + 10.0f * s;
        }
        const bool has_title = c.game_title && c.game_title[0] != '\0';
        const char* title = has_title ? c.game_title : "Lichii";
        const float title_px = FitPx(17.0f * s, 10.0f * s, 0.5f * s, room, [&](float px) {
            return SizeOfRaw(c.f.strong, px, title).x;
        });
        if (title_px > 0.0f && room > 0.0f) {
            const bool two_lines = has_title && c.game_id && c.game_id[0] != '\0';
            TextMidRaw(c, c.f.strong, title_px, x, two_lines ? ty - 8.0f * s : ty, kBannerInk,
                       title);
            if (two_lines) {
                const float id_px = FitPx(12.0f * s, 9.0f * s, 0.5f * s, room, [&](float px) {
                    return SizeOfRaw(c.f.body, px, c.game_id).x;
                });
                if (id_px > 0.0f) {
                    TextMidRaw(c, c.f.body, id_px, x, ty + 10.0f * s, kBannerEdge, c.game_id);
                }
            }
        }
    }

    float cx0 = 0.0f, cx1 = 0.0f, hx0 = 0.0f, hx1 = 0.0f;
    bool narrow = false;
    Columns(c, &cx0, &cx1, &hx0, &hx1, &narrow);
    DrawColumn(c, cx0, cx1, hh);

    const auto tab_rows = RowsForTab(cur_tab_);
    ClampSelection(tab_rows);
    const std::size_t tid = static_cast<std::size_t>(cur_tab_);
    const int sel = tab_sel_[tid];
    const int n = static_cast<int>(tab_rows.size());

    const float list_top = hh + 14.0f * s;
    const float list_bottom =
        (cur_tab_ == Tab::Controls) ? (hh + 280.0f * s) : (PromptBarTop(c) - 10.0f * s);
    const float pitch = 44.0f * s;
    const int fit = std::max(1, static_cast<int>((list_bottom - list_top) / pitch));

    float& at = scroll_at_[tid];
    float& want = scroll_want_[tid];
    if (n <= fit) {
        want = 0.0f;
    } else {
        if (sel != last_scrolled_[tid]) {
            const int margin = fit > 4 ? 1 : 0;
            if (static_cast<float>(sel - margin) < want) {
                want = static_cast<float>(std::max(0, sel - margin));
            }
            if (static_cast<float>(sel + margin + 1) > want + static_cast<float>(fit)) {
                want = static_cast<float>(sel + margin + 1 - fit);
            }
        }
        want = std::clamp(want, 0.0f, static_cast<float>(n - fit));
    }
    last_scrolled_[tid] = sel;
    at += (want - at) * 0.25f;
    if (std::fabs(want - at) < 0.02f) {
        at = want;
    }

    const float rx0 = cx0 + 18.0f * s;
    const float rx1 = cx1 - 24.0f * s;
    const float split = rx0 + (rx1 - rx0) * 0.50f;
    const float row_h = 37.0f * s;

    bool dd_anchor_found = false;
    ImVec2 dd_a{};
    ImVec2 dd_b{};
    c.d->PushClipRect(ImVec2(cx0, hh + 2.0f * s), ImVec2(cx1, list_bottom));
    for (int i = 0; i < n; ++i) {
        const float ry0 = list_top + (static_cast<float>(i) - at) * pitch;
        const float ry1 = ry0 + row_h;
        if (ry1 < hh || ry0 > list_bottom) {
            continue;
        }
        const auto& r = *tab_rows[static_cast<std::size_t>(i)];
        if (r.kind == RowKind::Heading) {
            TextMid(c, c.f.heading, 15.0f * s, rx0 + 4.0f * s, (ry0 + ry1) * 0.5f + 2.0f * s,
                    kAccent, r.label);
            c.d->AddLine(ImVec2(rx0, ry1 - 4.0f * s), ImVec2(rx1, ry1 - 4.0f * s), Alpha(kEdge, 0.7f),
                         std::max(1.0f, s));
            continue;
        }

        const bool is_sel = (i == sel);
        const bool is_hov = (i == hover_row_);
        const int val = ReadRowValue(r.id, opts);
        const bool pending = (r.requires_restart || std::strcmp(r.id, "upscaler") == 0) &&
                             IsRowPendingRestart(r.id, opts);
        const bool disabled = IsRowDisabled(r.id, tel);

        if (r.kind == RowKind::Action) {
            const ImU32 label_col = is_sel ? kGoldInk : (disabled ? kDim : kInk);
            if (is_sel && !disabled) {
                GoldFill(c, ImVec2(rx0, ry0), ImVec2(rx1, ry1));
            } else {
                c.d->AddRectFilled(ImVec2(rx0, ry0), ImVec2(rx1, ry1),
                                   is_hov ? kCellNameSel : kCellName);
            }
            c.d->AddRect(ImVec2(rx0, ry0), ImVec2(rx1, ry1), kEdge, 0.0f, std::max(1.0f, s));
            const std::string action_text =
                confirm_action_id_ == r.id ? Tr("Confirmar novamente") : Tr(r.label);
            TextMidCentreRaw(c, is_sel ? c.f.strong : c.f.body, 18.0f * s, (rx0 + rx1) * 0.5f,
                             (ry0 + ry1) * 0.5f, label_col, action_text.c_str());
            if (is_sel) {
                Glow(c, ImVec2(rx0, ry0), ImVec2(rx1, ry1));
            }
            hits_.push_back({rx0, ry0, rx1, ry1, HitKind::Row, i, rx0, rx1, 0, 0});
            continue;
        }

        DrawNameCell(c, ImVec2(rx0, ry0), ImVec2(split, ry1), r.label, is_sel || is_hov, pending,
                     !r.requires_restart);
        const ImVec2 va(split, ry0);
        const ImVec2 vb(rx1, ry1);
        float tx0 = 0.0f;
        float tx1 = 0.0f;
        const std::string val_str = FormatRowValue(r, val, tel);
        if (r.kind == RowKind::Slider) {
            DrawSliderCell(c, va, vb, val_str,
                           static_cast<int>((val - r.lo) * 1000LL / std::max(1, r.hi - r.lo)),
                           is_sel, &tx0, &tx1);
        } else if (r.kind == RowKind::Toggle) {
            const char* lo_l =
                std::strcmp(r.id, "gpu_acc") == 0
                    ? "Baixa (Normal)"
                    : (std::strcmp(r.id, "vram") == 0
                           ? "Conservador"
                           : (std::strcmp(r.id, "docked") == 0
                                  ? "Portátil (720p)"
                                  : (std::strcmp(r.id, "oot_aa") == 0 ? "Nitidez Original"
                                                                      : "Desligado")));
            const char* hi_l =
                std::strcmp(r.id, "gpu_acc") == 0
                    ? "Alta"
                    : (std::strcmp(r.id, "vram") == 0
                           ? "Agressivo"
                           : (std::strcmp(r.id, "docked") == 0
                                  ? "Dock (Alta Res.)"
                                  : (std::strcmp(r.id, "oot_aa") == 0 ? "Sem Nitidez"
                                                                      : "Ligado")));
            DrawHalvesCell(c, va, vb, lo_l, hi_l, val != 0, is_sel, disabled);
        } else {
            DrawChoiceCell(c, va, vb, val_str, is_sel, !disabled, !disabled, disabled);
        }
        if (dd_row_ == &r) {
            dd_anchor_found = true;
            dd_a = va;
            dd_b = vb;
        }
        hits_.push_back({rx0, ry0, rx1, ry1,
                         r.kind == RowKind::Slider ? HitKind::SliderTrack : HitKind::Row, i, va.x,
                         vb.x, tx0, tx1});
    }
    c.d->PopClipRect();

    if (n > fit) {
        const float sx = cx1 - 9.0f * s;
        const float span = list_bottom - list_top;
        const float sb0 = list_top + span * at / static_cast<float>(n);
        const float sb1 = list_top + span * (at + static_cast<float>(fit)) / static_cast<float>(n);
        c.d->AddRectFilled(ImVec2(sx - 2.0f * s, list_top), ImVec2(sx + 2.0f * s, list_bottom),
                           Rgba(0, 0, 0, 140), 2.0f * s);
        c.d->AddRectFilled(ImVec2(sx - 2.0f * s, sb0), ImVec2(sx + 2.0f * s, sb1), kAccent, 2.0f * s);
    }

    if (cur_tab_ == Tab::Controls) {
        DrawControlsBindings(c, cx0 + 12.0f * s, cx1 - 12.0f * s, list_bottom + 4.0f * s,
                             PromptBarTop(c) - 10.0f * s, bindings);
    }

    // Right Help & Live Telemetry Card
    const auto& sel_row = *tab_rows[static_cast<std::size_t>(sel)];
    const auto pending_list = CollectPendingRowNames(opts);
    const bool any_pending = !pending_list.empty() || tel.needs_restart;

    if (!narrow) {
        const float hy0 = hh + 24.0f * s;
        const float hy1 = PromptBarTop(c) - 18.0f * s;
        Box(c, ImVec2(hx0, hy0), ImVec2(hx1, hy1));
        c.d->PushClipRect(ImVec2(hx0, hy0), ImVec2(hx1, hy1));

        float y = hy0 + 22.0f * s;
        const float w_txt = hx1 - hx0 - 48.0f * s;
        Text(c, c.f.heading, 21.0f * s, ImVec2(hx0 + 24.0f * s, y), kInk, sel_row.label);
        y += 32.0f * s;

        if (sel_row.requires_restart) {
            Diamond(c, ImVec2(hx0 + 29.0f * s, y + 9.0f * s), 4.0f * s, kGoldHi);
            TextMid(c, c.f.strong, 15.5f * s, hx0 + 42.0f * s, y + 9.0f * s, kGoldHi,
                    "Requer reinício");
            y += 28.0f * s;
        } else {
            c.d->AddCircleFilled(ImVec2(hx0 + 29.0f * s, y + 9.0f * s), 3.8f * s, kLiveGreen);
            TextMid(c, c.f.strong, 15.5f * s, hx0 + 42.0f * s, y + 9.0f * s, kLiveGreen, "Ao Vivo");
            y += 28.0f * s;
        }

        if (sel_row.help && *sel_row.help) {
            y += Para(c, c.f.body, 17.5f * s, ImVec2(hx0 + 24.0f * s, y), kInk, sel_row.help,
                      w_txt);
            y += 16.0f * s;
        }

        // Live telemetry readout inside help panel
        char tel_buf[256];
        std::snprintf(tel_buf, sizeof(tel_buf), "%s: %ux%u -> %ux%u | %s\nFPS: %.1f (%.2f ms)",
                      Tr("Upscaler (OoT)"), tel.render_width, tel.render_height, tel.output_width,
                      tel.output_height, tel.upscaler_name.c_str(), tel.fps, tel.frametime_ms);
        c.d->AddRectFilled(ImVec2(hx0 + 24.0f * s, y + 2.0f * s),
                           ImVec2(hx0 + 26.5f * s, y + 42.0f * s), kAccent, s);
        TextRaw(c, c.f.body, 15.0f * s, ImVec2(hx0 + 38.0f * s, y), kSoft, tel_buf, w_txt - 16.0f * s);
        y += 52.0f * s;

        std::string ctrl_line;
        if (bindings.pad_connected && !bindings.active_pad_name.empty()) {
            ctrl_line = std::string(Tr("Layout do Controle")) + ": " + bindings.active_pad_name;
        } else {
            ctrl_line = std::string(Tr("Teclado")) +
                        (tel.has_prompts ? " (Prompts Dinâmicos ON)" : "");
        }
        TextRaw(c, c.f.body, 14.5f * s, ImVec2(hx0 + 24.0f * s, y), kDim, ctrl_line.c_str(),
                w_txt);

        if (any_pending && hy1 - 86.0f * s > y + 28.0f * s) {
            const float sep_y = hy1 - 92.0f * s;
            c.d->AddLine(ImVec2(hx0 + 24.0f * s, sep_y), ImVec2(hx1 - 24.0f * s, sep_y),
                         Alpha(kEdge, 0.8f), std::max(1.0f, s));
            // Longshot parity (E47.T13): name what waits for the restart, not just how many.
            std::string msg = std::string(Tr("Requer reinício")) + ":";
            for (std::size_t i = 0; i < pending_list.size(); ++i) {
                msg += (i == 0 ? " " : ", ");
                msg += Tr(pending_list[i]);
            }
            if (pending_list.empty()) {
                msg = Tr("Após reiniciar");
            }
            const float room_w = hx1 - hx0 - 48.0f * s;
            float px = FitPx(14.5f * s, 11.0f * s, 0.5f * s, room_w, [&](float v) {
                return SizeOfRaw(c.f.strong, v, msg.c_str()).x;
            });
            if (px <= 0.0f || SizeOfRaw(c.f.strong, 11.0f * s, msg.c_str()).x > room_w) {
                msg = std::string(Tr("Requer reinício")) + ": " +
                      std::to_string(pending_list.size()) + " " + Tr("opções");
                px = 14.5f * s;
            }
            TextMidRaw(c, c.f.strong, px, hx0 + 24.0f * s, sep_y + 18.0f * s, kGoldHi,
                       msg.c_str());
            const ImVec2 b0(hx0 + 24.0f * s, hy1 - 54.0f * s);
            const ImVec2 b1(hx1 - 24.0f * s, hy1 - 14.0f * s);
            Button(c, b0, b1, "Aplicar e Reiniciar", hover_apply_);
            hits_.push_back({b0.x, b0.y, b1.x, b1.y, HitKind::ApplyRestart, -1, 0, 0, 0, 0});
        }
        c.d->PopClipRect();
    }

    const char* compact_k =
        opts.compact_key == 1 ? "RCtrl" : (opts.compact_key == 2 ? "F4" : "RShift");
    if (DropdownOpen()) {
        const std::array<Prompt, 3> dd_prompts{{
            {Glyph::UpDown, "Escolher"},
            {Glyph::Confirm, "Aplicar"},
            {Glyph::Back, "Fechar"},
        }};
        PromptBar(c, dd_prompts.data(), static_cast<int>(dd_prompts.size()));
        if (dd_anchor_found) {
            DrawDropdown(c, dd_a, dd_b, false, tel, opts);
        } else {
            CloseDropdown();
        }
        return;
    }
    const bool sel_choice = sel_row.kind == RowKind::Choice;
    const std::array<Prompt, 6> prompt_list{{
        {Glyph::UpDown, "Escolher"},
        {sel_row.kind == RowKind::Action ? Glyph::Confirm : Glyph::LeftRight, "Alterar"},
        {Glyph::Tabs, "Abas"},
        {Glyph::Compact, "Compacto", compact_k},
        {Glyph::Confirm, sel_choice ? "Lista" : "Confirmar"},
        {Glyph::Back, "Fechar"},
    }};
    PromptBar(c, prompt_list.data(), static_cast<int>(prompt_list.size()));
}

void OotMenuController::DrawFpsHud(Ctx& c, const Bridge& bridge, bool menu_open, bool compact) {
    // Levels after Longshot's counter (gpu/vk/fps_stats.h): Off, FPS, Frame time, Graph,
    // Detailed (graph + where the frame goes) and Dev (the renderer's own numbers).
    const auto opts = bridge.GetOptions();
    int level = std::clamp(opts.fps_hud, 0, 5);
    if (level <= 0) {
        return;
    }
    if (menu_open && !compact) {
        level = std::min(level, 2); // the panel would cover the help card, which has the numbers
    }
    const auto tel = bridge.GetTelemetry();
    const float s = c.s;
    const float big = 27.0f * s;
    const float small = 13.0f * s;
    const float line = 16.0f * s;
    const int corner = std::clamp(opts.fps_corner, 0, 3);
    const bool left = (corner & 1) != 0;
    const bool bottom = corner >= 2;
    const ImU32 green = Rgba(72, 255, 110);
    const ImU32 amber = Rgba(255, 200, 60);
    const ImU32 red = Rgba(255, 95, 80);
    const auto fps_col = [&](double fps) { return fps >= 55.0 ? green : (fps >= 38.0 ? amber : red); };

    // Lines of text below the number: built first so the panel can be sized and anchored
    struct HudLine {
        std::string label;
        std::string value;
        ImU32 col;
    };
    std::vector<HudLine> lines;
    char b[128];
    if (tel.output_height > 0) {
        std::snprintf(b, sizeof(b), "%up %s", tel.output_height, tel.upscaler_name.c_str());
    } else {
        std::snprintf(b, sizeof(b), "%s", tel.upscaler_name.c_str());
    }
    lines.push_back({"", b, kInkBright});
    if (level >= 2) {
        const double low = tel.frametime_p99_ms > 0.0 ? 1000.0 / tel.frametime_p99_ms : 0.0;
        std::snprintf(b, sizeof(b), "%.1f ms", tel.frametime_ms);
        lines.push_back({"Frame time", b, kSoft});
        std::snprintf(b, sizeof(b), "%.0f FPS (%.1f ms)", low, tel.frametime_p99_ms);
        lines.push_back({"1% low", b, fps_col(low)});
    }
    if (level >= 4) {
        std::snprintf(b, sizeof(b), "%ux%u > %ux%u", tel.render_width, tel.render_height,
                      tel.output_width, tel.output_height);
        lines.push_back({"Render > saída", b, kSoft});
        if (tel.base_fps > 0.0 && tel.fps > tel.base_fps * 1.15) {
            std::snprintf(b, sizeof(b), "%.0f base > %.0f exibidos", tel.base_fps, tel.fps);
            lines.push_back({"Frame gen", b, kSoft});
        }
        if (tel.host_stats_valid) {
            std::snprintf(b, sizeof(b), "%.0f%%  ·  jogo %.0f FPS", tel.emulation_speed * 100.0,
                          tel.game_fps);
            lines.push_back({"Velocidade", b, kSoft});
            if (tel.shaders_building > 0) {
                std::snprintf(b, sizeof(b), "%d", tel.shaders_building);
                lines.push_back({"Compilando shaders", b, amber});
            }
        }
        if (tel.vram_budget > 0) {
            const double gb = 1024.0 * 1024.0 * 1024.0;
            std::snprintf(b, sizeof(b), "%.1f de %.1f GB", static_cast<double>(tel.vram_used) / gb,
                          static_cast<double>(tel.vram_budget) / gb);
            lines.push_back({"VRAM", b,
                             tel.vram_used > tel.vram_budget * 9 / 10 ? amber : kSoft});
        }
    }
    if (level >= 5) {
        std::snprintf(b, sizeof(b), "%.1f / %.1f / %.1f ms", tel.frametime_min_ms,
                      tel.frametime_ms, tel.frametime_max_ms);
        lines.push_back({"Mín / média / máx", b, kSoft});
        std::snprintf(b, sizeof(b), "%llu de %llu quadros",
                      static_cast<unsigned long long>(tel.stutters),
                      static_cast<unsigned long long>(tel.total_frames));
        lines.push_back({"Travadas > 50 ms", b, tel.stutters > 0 ? amber : kSoft});
        static constexpr std::array<const char*, 4> kVsync{"Imediato", "Mailbox", "FIFO",
                                                          "FIFO relaxado"};
        std::snprintf(
            b, sizeof(b), "GPU %s · %s · async %s",
            Settings::values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::High ? "alta"
                                                                                    : "normal",
            kVsync[static_cast<std::size_t>(
                std::clamp(static_cast<int>(Settings::values.vsync_mode.GetValue()), 0, 3))],
            Settings::values.use_asynchronous_shaders.GetValue() ? "sim" : "não");
        lines.push_back({"Dev", b, kDim});
    }

    // Size: number row, text lines, graph
    const float label_w = level >= 2 ? 118.0f * s : 0.0f;
    float text_w = 0.0f;
    for (const auto& l : lines) {
        text_w = std::max(text_w, (l.label.empty() ? 0.0f : label_w) +
                                      SizeOfRaw(c.f.body, small, l.value.c_str()).x);
    }
    const bool graph = level >= 3;
    const float gw = graph ? std::max(220.0f * s, text_w) : 0.0f;
    const float gh = 48.0f * s;
    const float pad = level >= 3 ? 8.0f * s : 0.0f;
    const float w = std::max({gw, text_w, 90.0f * s}) + 2.0f * pad;
    const float h = 30.0f * s + static_cast<float>(lines.size()) * line + (graph ? gh + 8.0f * s : 0.0f) +
                    2.0f * pad;
    const float x0 = left ? c.safe : (c.w - c.safe - w);
    const float y0 = bottom ? (c.h - c.safe - h)
                            : ((menu_open && !compact && corner < 2) ? (BannerHeight(c) + 10.0f * s)
                                                                     : c.safe);
    if (level >= 3) {
        c.d->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + w, y0 + h), Rgba(10, 8, 5, 168), 5.0f * s);
        c.d->AddRect(ImVec2(x0, y0), ImVec2(x0 + w, y0 + h), Alpha(kEdge, 0.7f), 5.0f * s, 1.0f);
    }
    const float ix0 = x0 + pad;
    const float ix1 = x0 + w - pad;
    float y = y0 + pad;

    char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "%d", static_cast<int>(std::lround(tel.fps)));
    const float nw = SizeOfRaw(c.f.strong, big, fps_buf).x;
    const float lw = SizeOfRaw(c.f.strong, small, "FPS").x;
    const float x_num = left ? ix0 : (ix1 - nw - 4.0f * s - lw);
    OutlinedText(c, c.f.strong, big, x_num, y, fps_col(tel.fps), fps_buf);
    OutlinedText(c, c.f.strong, small, x_num + nw + 4.0f * s, y + 12.0f * s, fps_col(tel.fps), "FPS");
    y += 30.0f * s;

    for (const auto& l : lines) {
        if (l.label.empty()) {
            const float vw = SizeOfRaw(c.f.strong, small, l.value.c_str()).x;
            OutlinedText(c, c.f.strong, small, left ? ix0 : (ix1 - vw), y, l.col, l.value.c_str());
        } else {
            const float vw = SizeOfRaw(c.f.body, small, l.value.c_str()).x;
            const float lx = left ? ix0 : (ix1 - label_w - vw);
            OutlinedText(c, c.f.body, small, lx, y, kDim, Tr(l.label.c_str()));
            OutlinedText(c, c.f.body, small, lx + label_w, y, l.col, l.value.c_str());
        }
        y += line;
    }

    if (graph) {
        y += 4.0f * s;
        bridge.CopyRecentFrames(graph_frames_);
        const float gx0 = left ? ix0 : (ix1 - gw);
        const float gx1 = gx0 + gw;
        c.d->AddRectFilled(ImVec2(gx0, y), ImVec2(gx1, y + gh), Rgba(0, 0, 0, 120), 3.0f * s);
        // Column = the slowest frame among the frames it covers (a spike is never averaged away)
        const int cols = 80;
        const int n = static_cast<int>(graph_frames_.size());
        float worst = 0.0f;
        for (const float f : graph_frames_) {
            worst = std::max(worst, f);
        }
        const float scale = std::clamp(worst * 1.1f, 33.4f, 100.0f);
        for (const float guide : {16.7f, 33.3f}) {
            const float gy = y + gh - (guide / scale) * gh;
            c.d->AddLine(ImVec2(gx0, gy), ImVec2(gx1, gy), Rgba(255, 255, 255, 46), 1.0f);
            char gb[16];
            std::snprintf(gb, sizeof(gb), "%.0f", guide);
            c.d->AddText(c.f.body, 10.0f * s, ImVec2(gx0 + 2.0f * s, gy - 11.0f * s),
                         Rgba(255, 255, 255, 110), gb);
        }
        if (n > 0) {
            const float cw = gw / static_cast<float>(cols);
            // Many frames: each column is the slowest of the frames it covers.
            // Few frames: one frame per column, right-aligned.
            for (int col = 0; col < cols; ++col) {
                int i0 = 0;
                int i1 = 0;
                if (n >= cols) {
                    i0 = (n * col) / cols;
                    i1 = std::max(i0 + 1, (n * (col + 1)) / cols);
                } else {
                    i0 = n - cols + col;
                    i1 = i0 + 1;
                    if (i0 < 0) {
                        continue;
                    }
                }
                float f = 0.0f;
                for (int i = i0; i < i1 && i < n; ++i) {
                    f = std::max(f, graph_frames_[static_cast<std::size_t>(i)]);
                }
                const float bh = std::min(1.0f, f / scale) * gh;
                const ImU32 bc = f > 33.4f ? red : (f > 20.0f ? amber : green);
                c.d->AddRectFilled(ImVec2(gx0 + static_cast<float>(col) * cw, y + gh - bh),
                                   ImVec2(gx0 + static_cast<float>(col + 1) * cw - 1.0f, y + gh),
                                   bc);
            }
        }
        char sb[32];
        std::snprintf(sb, sizeof(sb), "%.0f ms", scale);
        const float sw = SizeOfRaw(c.f.body, 10.0f * s, sb).x;
        c.d->AddText(c.f.body, 10.0f * s, ImVec2(gx1 - sw - 2.0f * s, y + 1.0f * s),
                     Rgba(255, 255, 255, 110), sb);
    }
}

void OotMenuController::RunValueSelfTest(Bridge& bridge) {
    int pass = 0;
    int fail = 0;
    for (const auto& r : AllRows()) {
        if (r.kind == RowKind::Heading || r.kind == RowKind::Action ||
            std::strcmp(r.id, "rebind") == 0 || std::strcmp(r.id, "gamemode") == 0) {
            continue; // no value, or a side effect outside the menu (controller capture, gamemode)
        }
        auto opts = bridge.GetOptions();
        const int original = ReadRowValue(r.id, opts);
        std::vector<int> values;
        if (r.kind == RowKind::Slider) {
            const int mid = r.lo + ((r.hi - r.lo) / 2 / r.step) * r.step;
            values = {r.lo, mid, r.hi};
        } else {
            for (int v = r.lo; v <= r.hi; ++v) {
                values.push_back(v);
            }
        }
        std::string bad;
        for (const int v : values) {
            opts = bridge.GetOptions();
            WriteRowValue(r.id, v, opts, bridge);
            const int got = ReadRowValue(r.id, bridge.GetOptions());
            if (got == v) {
                ++pass;
            } else {
                ++fail;
                bad += " " + std::to_string(v) + "->" + std::to_string(got);
            }
        }
        opts = bridge.GetOptions();
        WriteRowValue(r.id, original, opts, bridge);
        const int restored = ReadRowValue(r.id, bridge.GetOptions());
        if (restored != original) {
            ++fail;
            bad += " restore " + std::to_string(original) + "->" + std::to_string(restored);
        }
        if (bad.empty()) {
            LOG_INFO(Render_Vulkan, "OoT menu selftest: PASS {} ({} values)", r.id, values.size());
        } else {
            LOG_WARNING(Render_Vulkan, "OoT menu selftest: FAIL {}:{}", r.id, bad);
        }
    }
    LOG_INFO(Render_Vulkan, "OoT menu selftest: values done, {} pass, {} fail", pass, fail);
}

void OotMenuController::DriveSelfTest(Ctx& c, Bridge& bridge) {
    static const bool enabled = [] {
        const char* v = std::getenv("EDEN_NX2_OOT_MENU_SELFTEST");
        return v != nullptr && v[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (selftest_start_ == std::chrono::steady_clock::time_point{}) {
        selftest_start_ = now;
    }
    const float t = std::chrono::duration<float>(now - selftest_start_).count();
    if (!selftest_values_done_ && t >= 3.0f) {
        selftest_values_done_ = true;
        RunValueSelfTest(bridge);
    }
    // E47.T9: switch the upscaler live once the scene runs (the value test runs before the game
    // loads); the rasterizer must log "rebuilding it" and then evaluate the new upscaler.
    if (selftest_live_step_ == 0 && t >= 95.0f) {
        selftest_live_step_ = 1;
        LOG_INFO(Render_Vulkan, "OoT menu selftest: live upscaler -> FSR 3.1");
        auto opts = bridge.GetOptions();
        WriteRowValue("upscaler", 1, opts, bridge);
    } else if (selftest_live_step_ == 1 && t >= 105.0f) {
        selftest_live_step_ = 2;
        LOG_INFO(Render_Vulkan, "OoT menu selftest: live upscaler -> FSR 4.1");
        auto opts = bridge.GetOptions();
        WriteRowValue("upscaler", 2, opts, bridge);
    }
    static constexpr std::array<const char*, 7> kPhases{
        "Gráficos", "Desempenho", "Controles", "Áudio & UI", "Lista aberta", "Barra compacta",
        "HUD Dev"};
    const int phase = t < 5.0f ? -1 : (static_cast<int>((t - 5.0f) / 8.0f) % 7);
    if (phase != selftest_phase_) {
        auto opts = bridge.GetOptions();
        if (selftest_phase_ == 6 && selftest_saved_hud_ >= 0) {
            opts.fps_hud = selftest_saved_hud_;
            bridge.SetOptions(opts);
        }
        CloseDropdown();
        selftest_phase_ = phase;
        if (phase >= 0 && phase <= 3) {
            bridge.SetOpen(true, false);
            cur_tab_ = static_cast<Tab>(phase);
        } else if (phase == 4) {
            bridge.SetOpen(true, false);
            cur_tab_ = Tab::Graphics;
            const auto rows = RowsForTab(cur_tab_);
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                if (std::strcmp(rows[static_cast<std::size_t>(i)]->id, "preset") == 0) {
                    tab_sel_[0] = i;
                    OpenDropdown(*rows[static_cast<std::size_t>(i)], opts);
                }
            }
        } else if (phase == 5) {
            bridge.SetOpen(true, true);
        } else if (phase == 6) {
            bridge.SetOpen(false);
            selftest_saved_hud_ = opts.fps_hud;
            opts.fps_hud = static_cast<int>(Common::OotMenu::FpsHudLevel::Dev);
            opts.fps_corner = 0;
            bridge.SetOptions(opts);
        }
        if (phase >= 0) {
            LOG_INFO(Render_Vulkan, "OoT menu selftest: showing {} at {:.1f} s", kPhases[phase], t);
        }
    }
    if (phase >= 0) {
        char label[64];
        std::snprintf(label, sizeof(label), "SELFTEST %d/7: %s", phase + 1, kPhases[phase]);
        OutlinedText(c, c.f.strong, 16.0f * c.s, c.safe, c.h - c.safe - 20.0f * c.s, kGoldHi, label);
    }
}

void OotMenuController::TickAndDraw(Ctx& c) {
    auto& bridge = Bridge::Instance();
    if (!bridge.IsOcarina()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    float dt_sec = 1.0f / 60.0f;
    if (last_tick_ != std::chrono::steady_clock::time_point{}) {
        dt_sec = std::clamp(std::chrono::duration<float>(now - last_tick_).count(), 0.001f, 0.25f);
    }
    last_tick_ = now;

    const auto opts = bridge.GetOptions();
    CaptureBaselineOnce(opts);
    ProcessInput(bridge, dt_sec);

    const bool open = bridge.IsOpen();
    const bool compact = bridge.IsCompact();

    if (open) {
        if (!compact) {
            c.d->AddRectFilled(ImVec2(0.0f, 0.0f), ImVec2(c.w, c.h), Rgba(8, 5, 2, kDimAlpha));
        }
        Ctx k = c;
        k.w = std::min(c.w, std::floor(c.h * 16.0f / 9.0f + 0.5f));
        menu_ox_ = std::floor((c.w - k.w) * 0.5f);
        const int vtx0 = c.d->VtxBuffer.Size;
        const int cmd0 = c.d->CmdBuffer.Size;
        hits_.clear();
        if (compact) {
            DrawCompactBar(k, bridge);
        } else {
            DrawFullMenu(k, bridge);
        }
        if (menu_ox_ > 0.0f) {
            for (int i = vtx0; i < c.d->VtxBuffer.Size; ++i) {
                c.d->VtxBuffer[i].pos.x += menu_ox_;
            }
            for (int i = std::max(0, cmd0 - 1); i < c.d->CmdBuffer.Size; ++i) {
                ImVec4& rect = c.d->CmdBuffer[i].ClipRect;
                if (rect.x <= 0.0f && rect.z >= c.w) {
                    continue;
                }
                rect.x += menu_ox_;
                rect.z += menu_ox_;
            }
        }
    } else {
        menu_ox_ = 0.0f;
        hits_.clear();
    }

    DrawFpsHud(c, bridge, open, compact);
    DriveSelfTest(c, bridge);
}

} // namespace Vulkan::OotUi
