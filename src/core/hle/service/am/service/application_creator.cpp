// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/cpu_features.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/nca_metadata.h"
#include "core/hle/kernel/k_system_control.h"
#include "core/hle/service/acc/profile_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/hle/service/am/am_types.h"
#include "core/hle/service/am/applet.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/service/am/process_creation.h"
#include "core/hle/service/am/service/application_accessor.h"
#include "core/hle/service/am/service/application_creator.h"
#include "core/hle/service/am/window_system.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/loader/loader.h"
#include "core/launch_timestamp_cache.h"
#include "core/hle/api_version.h"

namespace Service::AM {

namespace {

Result CreateGuestApplication(SharedPointer<IApplicationAccessor>* out_application_accessor, Core::System& system, WindowSystem& window_system, u64 program_id) {
    FileSys::VirtualFile nca_raw{};

    // Get the program NCA from storage.
    auto& storage = system.GetContentProviderUnion();
    nca_raw = storage.GetEntryRaw(program_id, FileSys::ContentRecordType::Program);

    // Ensure we retrieved a program NCA.
    R_UNLESS(nca_raw != nullptr, ResultUnknown);

    // A game started by the home menu skips System::Load, where the Switch 2 globals are set
    // from the booted file. Set them here for a Switch 2 program: the current-program flag, the
    // application program id the kernel reports, and the 31.25 MHz guest counter its SDK
    // assumes. The counter only moves forward when raised, so it is left raised on exit.
    // The 12 GiB memory layout has to be chosen at boot (System::Load does it for qlaunch).
    if (const auto switch2_id = Core::FindSwitch2Program(nca_raw, program_id, 0)) {
        LOG_INFO(Service_AM, "home menu start of Switch 2 program {:016X}", *switch2_id);
        Settings::SetCurrentProgramIsSwitch2(true);
        Kernel::KSystemControl::SetApplicationProgramId(*switch2_id);
        if (std::getenv("EDEN_NX2_CNTFRQ") == nullptr) {
            Common::g_wall_clock.SetGuestCNTFRQ(Common::WallClock::Switch2CNTFRQ);
        }
        if (Settings::values.memory_layout_mode.GetValue() != Settings::MemoryLayout::Memory_12Gb) {
            LOG_WARNING(Service_AM, "Switch 2 program started without the 12 GiB memory layout");
        }
    }

    std::vector<u8> control;
    std::unique_ptr<Loader::AppLoader> loader;
    Loader::ResultStatus result;
    auto process = CreateApplicationProcess(control, loader, result, system, nca_raw, program_id, 0);
    R_UNLESS(process != nullptr, ResultUnknown);

    const auto applet = std::make_shared<Applet>(system, std::move(process), true);
    applet->program_id = program_id;
    applet->applet_id = AppletId::Application;
    applet->type = AppletType::Application;
    applet->library_applet_mode = LibraryAppletMode::AllForeground;

    // The home menu pushes a preselected user only when the game's NACP asks for an account at
    // startup; the NACP a NUP builds does not, and without one the game panics reading the
    // profile of user 0. Give it the current user, as a frontend boot does. A user the menu
    // pushes later is popped first (the channel is read from the back).
    {
        LaunchParameterAccountPreselectedUser lp{};
        lp.magic = LaunchParameterAccountPreselectedUserMagic;
        lp.is_account_selected = 1;
        Account::ProfileManager profile_manager{};
        if (const auto uuid =
                profile_manager.GetUser(static_cast<s32>(Settings::values.current_user));
            uuid.has_value() && uuid->IsValid()) {
            lp.current_user = *uuid;
            std::vector<u8> buffer(sizeof(LaunchParameterAccountPreselectedUser));
            std::memcpy(buffer.data(), &lp, buffer.size());
            applet->preselected_user_launch_parameter.push_back(std::move(buffer));
        }
    }

    window_system.TrackApplet(applet, true);

    *out_application_accessor = std::make_shared<IApplicationAccessor>(system, applet, window_system);
    R_SUCCEED();
}

} // namespace

IApplicationCreator::IApplicationCreator(Core::System& system_, WindowSystem& window_system)
    : ServiceFramework{system_, "IApplicationCreator"}, m_window_system{window_system} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IApplicationCreator::CreateApplication>, "CreateApplication"},
        {1, nullptr, "PopLaunchRequestedApplication"},
        {10, D<&IApplicationCreator::CreateSystemApplication>, "CreateSystemApplication"},
        {100, nullptr, "PopFloatingApplicationForDevelopment"},
    };
    // clang-format on

    RegisterHandlers(functions);
}

IApplicationCreator::~IApplicationCreator() = default;

Result IApplicationCreator::CreateApplication(
    Out<SharedPointer<IApplicationAccessor>> out_application_accessor, u64 application_id) {
    LOG_INFO(Service_NS, "called, application_id={:016X}", application_id);
    Core::LaunchTimestampCache::SaveLaunchTimestamp(application_id);
    R_RETURN(
        CreateGuestApplication(out_application_accessor, system, m_window_system, application_id));
}

Result IApplicationCreator::CreateSystemApplication(
    Out<SharedPointer<IApplicationAccessor>> out_application_accessor, u64 application_id) {

    FileSys::VirtualFile nca_raw{};

    auto& storage = system.GetContentProviderUnion();
    nca_raw = storage.GetEntryRaw(application_id, FileSys::ContentRecordType::Program);

    R_UNLESS(nca_raw != nullptr, ResultUnknown);

    std::vector<u8> control;
    std::unique_ptr<Loader::AppLoader> loader;
    auto process = CreateProcess(system, application_id, 1, HLE::ApiVersion::HOS_VERSION_MAJOR);
    R_UNLESS(process != nullptr, ResultUnknown);

    const auto applet = std::make_shared<Applet>(system, std::move(process), true);
    applet->program_id = application_id;
    applet->applet_id = AppletId::Starter;
    applet->type = AppletType::LibraryApplet;
    applet->library_applet_mode = LibraryAppletMode::AllForeground;

    m_window_system.TrackApplet(applet, true);

    *out_application_accessor = std::make_shared<IApplicationAccessor>(system, applet, m_window_system);
    Core::LaunchTimestampCache::SaveLaunchTimestamp(application_id);
    R_SUCCEED();
}

} // namespace Service::AM
