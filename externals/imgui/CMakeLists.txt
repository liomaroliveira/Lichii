# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(OOT_UI_FONTS_CPP "${CMAKE_CURRENT_BINARY_DIR}/oot_ui_fonts.cpp")
add_custom_command(
    OUTPUT "${OOT_UI_FONTS_CPP}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../fonts/ui_fonts.py" "${OOT_UI_FONTS_CPP}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/../fonts/ui_fonts.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../fonts/libertinus/LibertinusSans-Regular.ttf"
        "${CMAKE_CURRENT_SOURCE_DIR}/../fonts/libertinus/LibertinusSans-Bold.ttf"
        "${CMAKE_CURRENT_SOURCE_DIR}/../fonts/libertinus/LibertinusSerif-Semibold.ttf"
    COMMENT "Generating Ocarina UI Libertinus fallback fonts source"
    VERBATIM
)

add_library(imgui STATIC
    imgui.cpp
    imgui.h
    imconfig.h
    imgui_draw.cpp
    imgui_internal.h
    imgui_tables.cpp
    imgui_widgets.cpp
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h
    backends/imgui_impl_vulkan.cpp
    backends/imgui_impl_vulkan.h
    "${OOT_UI_FONTS_CPP}"
)

target_include_directories(imgui PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends"
)

target_compile_definitions(imgui PUBLIC
    VK_NO_PROTOTYPES
    IMGUI_IMPL_VULKAN_NO_PROTOTYPES
)

if (TARGET Vulkan::Headers)
    target_link_libraries(imgui PUBLIC Vulkan::Headers)
endif()

if (NOT MSVC)
    target_compile_options(imgui PRIVATE -w)
else()
    target_compile_options(imgui PRIVATE /W0)
endif()

add_library(imgui::imgui ALIAS imgui)
