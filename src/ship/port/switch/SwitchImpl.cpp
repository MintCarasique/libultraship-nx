#ifdef __SWITCH__
#include "ship/port/switch/SwitchImpl.h"

#include <atomic>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <switch.h>
#include <SDL2/SDL.h>
#include <spdlog/spdlog.h>
#include <imgui.h>
#include <imgui_internal.h>

#include "ship/Context.h"
#include "ship/audio/Audio.h"
#include "libultraship/bridge/consolevariablebridge.h"

#define DOCKED_MODE 1
#define HANDHELD_MODE 0

static const uint32_t SWITCH_CPU_SPEEDS_VALUES[] = { 1785000000, 1581000000, 1224000000, 1020000000,
                                                     918000000,  816000000,  714000000 };

static AppletHookCookie sAppletHookCookie;
static bool sIsRunning = true;
static bool sHasFocus = true;
static bool sIsShowingVirtualKeyboard = true;
static bool sClkrstInitialized = false;

static void OnAppletHook(AppletHookType hook, void* param);
static void DetectAppletMode();
static void StartNetLog();

static uint32_t sTargetCpuHz = 0;

static Result SetCpuClock(uint32_t hz) {
    Result rc = MAKERESULT(Module_Libnx, LibnxError_NotInitialized);
    sTargetCpuHz = hz;
    if (hosversionBefore(8, 0, 0)) {
        rc = pcvSetClockRate(PcvModule_CpuBus, hz);
    } else if (sClkrstInitialized) {
        ClkrstSession session = { 0 };
        rc = clkrstOpenSession(&session, PcvModuleId_CpuBus, 3);
        if (R_SUCCEEDED(rc)) {
            rc = clkrstSetClockRate(&session, hz);
            clkrstCloseSession(&session);
        }
    }
    return rc;
}

void Ship::Switch::Init(SwitchPhase phase) {
    switch (phase) {
        case PreInitPhase:
            // Must run before anything touches stdout (spdlog's console sink, printf)
            StartNetLog();
            DetectAppletMode();
            break;
        case PostInitPhase:
            appletInitializeGamePlayRecording();
            appletSetGamePlayRecordingState(true);
            appletHook(&sAppletHookCookie, OnAppletHook, NULL);
            appletSetFocusHandlingMode(AppletFocusHandlingMode_NoSuspend);
            if (hosversionBefore(8, 0, 0)) {
                pcvInitialize();
            } else {
                sClkrstInitialized = R_SUCCEEDED(clkrstInitialize());
            }
            ApplyOverclock();
            break;
    }
}

void Ship::Switch::Exit() {
    // Hand the CPU back at stock clocks, but only if we changed them
    if (sTargetCpuHz != 0) {
        SetCpuClock(SWITCH_CPU_SPEEDS_VALUES[STOCK]);
    }
    if (hosversionBefore(8, 0, 0)) {
        pcvExit();
    } else if (sClkrstInitialized) {
        clkrstExit();
        sClkrstInitialized = false;
    }
    appletUnhook(&sAppletHookCookie);
    appletSetGamePlayRecordingState(false);
}

void Ship::Switch::ImGuiProcessEvent(bool wantsTextInput) {
    ImGuiInputTextState* state = ImGui::GetInputTextState(ImGui::GetActiveID());

    if (wantsTextInput) {
        if (!sIsShowingVirtualKeyboard) {
            if (state != nullptr) {
                state->ClearText();
            }

            sIsShowingVirtualKeyboard = true;
            SDL_StartTextInput();
        }
    } else {
        if (sIsShowingVirtualKeyboard) {
            sIsShowingVirtualKeyboard = false;
            SDL_StopTextInput();
        }
    }
}

bool Ship::Switch::IsRunning() {
    return sIsRunning;
}

void Ship::Switch::GetDisplaySize(int* width, int* height) {
    switch (appletGetOperationMode()) {
        case DOCKED_MODE:
            *width = 1920;
            *height = 1080;
            break;
        case HANDHELD_MODE:
        default:
            *width = 1280;
            *height = 720;
            break;
    }
}

void Ship::Switch::ApplyOverclock() {
    int32_t perfMode = CVarGetInteger(CVAR_SWITCH_PERF_MODE, (int32_t)SYSTEM_DEFAULT);

    if (perfMode == SYSTEM_DEFAULT) {
        // Leave clock management to the system / sys-clk. If we pinned a clock earlier this
        // session, hand the CPU back at stock once.
        if (sTargetCpuHz != 0) {
            SetCpuClock(SWITCH_CPU_SPEEDS_VALUES[STOCK]);
            sTargetCpuHz = 0;
        }
        SPDLOG_INFO("Switch perf mode: system default (clocks untouched)");
        return;
    }

    if (perfMode >= 0 && perfMode <= POWERSAVINGM3) {
        Result rc = SetCpuClock(SWITCH_CPU_SPEEDS_VALUES[perfMode]);
        SPDLOG_INFO("Switch perf mode {} -> {} MHz (rc 0x{:x}, clkrst {})", perfMode,
                    SWITCH_CPU_SPEEDS_VALUES[perfMode] / 1000000, rc, sClkrstInitialized);
    }
}

void Ship::Switch::PrintErrorMessageToScreen(const char* str, ...) {
    consoleInit(NULL);

    va_list args;
    va_start(args, str);
    vprintf(str, args);
    va_end(args);

    PadState pad;
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    padInitializeDefault(&pad);

    while (appletMainLoop()) {
        padUpdate(&pad);
        if (padGetButtonsDown(&pad) & HidNpadButton_Plus) {
            break;
        }
        consoleUpdate(NULL);
    }

    consoleExit(NULL);
    exit(1);
}

static void OnAppletHook(AppletHookType hook, void* param) {
    switch (hook) {
        case AppletHookType_OnExitRequest:
            sIsRunning = false;
            break;

        case AppletHookType_OnFocusState:
            sHasFocus = appletGetFocusState() == AppletFocusState_InFocus;

            if (!sHasFocus) {
                // Drop our pinned clock while in the background (only if we pinned one)
                if (sTargetCpuHz != 0) {
                    const uint32_t target = sTargetCpuHz;
                    SetCpuClock(SWITCH_CPU_SPEEDS_VALUES[Ship::STOCK]);
                    sTargetCpuHz = target;
                }
            } else {
                Ship::Switch::ApplyOverclock();
                // Reinitialize the audio subsystem to fix audio problems after resuming from sleep
                // see https://github.com/HarbourMasters/Shipwright/issues/3317
                auto context = Ship::Context::GetRawInstance();
                if (context != nullptr && context->GetAudio() != nullptr) {
                    SPDLOG_INFO("restarting SDL audio system to work around audio problems on resume");
                    context->GetAudio()->SetCurrentAudioBackend(Ship::AudioBackend::SDL);
                }
            }
            break;

        case AppletHookType_OnPerformanceMode:
            Ship::Switch::ApplyOverclock();
            break;

        default:
            break;
    }
}

static void DetectAppletMode() {
    AppletType at = appletGetAppletType();
    if (at == AppletType_Application || at == AppletType_SystemApplication) {
        return;
    }

    Ship::Switch::PrintErrorMessageToScreen("\x1b[2;2HYou've launched the Ship while in Applet mode."
                                            "\x1b[4;2HPlease relaunch while in full-memory mode."
                                            "\x1b[5;2HHold R when opening any game to enter HBMenu."
                                            "\x1b[7;2HPress + to exit.");
}

void Ship::Switch::PinCurrentThreadToCore(int core, bool raisePriority) {
    Result rc = svcSetThreadCoreMask(CUR_THREAD_HANDLE, core, 1u << core);
    if (R_FAILED(rc)) {
        SPDLOG_WARN("Failed to move thread to core {} (0x{:x})", core, rc);
    }
    if (raisePriority) {
        s32 priority = 0;
        if (R_SUCCEEDED(svcGetThreadPriority(&priority, CUR_THREAD_HANDLE)) && priority > 0x20) {
            // Lower value = higher priority
            svcSetThreadPriority(CUR_THREAD_HANDLE, priority - 1);
        }
    }
}

bool Ship::Switch::IsModuleAddress(uintptr_t addr) {
    MemoryInfo info = {};
    u32 pageInfo = 0;
    if (R_FAILED(svcQueryMemory(&info, &pageInfo, addr))) {
        return false;
    }
    switch (info.type & MemState_Type) {
        case MemType_CodeStatic:
        case MemType_CodeMutable:
        case MemType_ModuleCodeStatic:
        case MemType_ModuleCodeMutable:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------------------------
// Network log + profiler
// ---------------------------------------------------------------------------------------------
static std::atomic<bool> sProfEnabled = false;
static std::atomic<uint64_t> sProfAccum[Ship::PROF_COUNT];
static std::atomic<uint32_t> sProfCalls[Ship::PROF_COUNT];
static std::atomic<int> sAudioBuffered = 0;
static int sAudioBufferedMin = INT32_MAX;
static std::atomic<int> sNetLogClient = -1;

// Raw write() to the nxlink-redirected stdout fd: bypasses stdio buffering so it is safe from
// any thread, including the exception handler.
static void NetLogWrite(const char* fmt, ...) {
    int sock = sNetLogClient.load();
    if (sock < 0) {
        return;
    }
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (len <= 0) {
        return;
    }
    if (len >= (int)sizeof(buf)) {
        len = sizeof(buf) - 1;
    }
    if (write(sock, buf, len) < 0) {
        sNetLogClient = -1;
        sProfEnabled = false;
    }
}

// libnx calls this on CPU exceptions (data abort etc.) instead of letting the process die
// silently. Report the faulting context over the network log with module-relative offsets
// so it can be symbolized against soh.elf (aarch64-none-elf-addr2line -e soh.elf <offset>).
extern "C" {
alignas(16) u8 __nx_exception_stack[0x8000];
u64 __nx_exception_stack_size = sizeof(__nx_exception_stack);

static uintptr_t GetModuleBase() {
    MemoryInfo info = {};
    u32 pageInfo = 0;
    svcQueryMemory(&info, &pageInfo, (uintptr_t)&GetModuleBase);
    return info.addr;
}

void __libnx_exception_handler(ThreadExceptionDump* ctx) {
    const uintptr_t base = GetModuleBase();
    auto rel = [base](uint64_t a) { return a >= base ? a - base : a; };
    NetLogWrite("[crash] exception 0x%x  pc +0x%lx  lr +0x%lx  far 0x%lx  sp 0x%lx  base 0x%lx\n",
                ctx->error_desc, rel(ctx->pc.x), rel(ctx->lr.x), ctx->far.x, ctx->sp.x, base);
    // Walk the frame-pointer chain
    uint64_t fp = ctx->fp.x;
    for (int i = 0; i < 24 && fp != 0 && (fp & 0xF) == 0; i++) {
        MemoryInfo info = {};
        u32 pageInfo = 0;
        if (R_FAILED(svcQueryMemory(&info, &pageInfo, fp)) || (info.perm & Perm_R) == 0) {
            break;
        }
        uint64_t* frame = (uint64_t*)fp;
        NetLogWrite("[crash]   #%d +0x%lx\n", i, rel(frame[1]));
        fp = frame[0];
    }
    svcSleepThread(2000000000ULL);
}
}

uintptr_t Ship::Switch::RelativeAddress(uintptr_t addr) {
    const uintptr_t base = GetModuleBase();
    return addr >= base ? addr - base : addr;
}

// When launched through nxlink with -s (host listening on 28771), stream stdout/stderr to the
// host via the standard nxlinkStdio(). This runs at the very start, before any stdio use.
static void StartNetLog() {
    static bool started = false;
    if (started || __nxlink_host.s_addr == 0) {
        return;
    }
    started = true;
    if (R_FAILED(socketInitializeDefault())) {
        return;
    }
    int sock = nxlinkStdio();
    if (sock < 0) {
        return;
    }
    sNetLogClient = STDOUT_FILENO;
    NetLogWrite("[netlog] nxlink connected, profiler on (mode: %s)\n",
                appletGetOperationMode() == DOCKED_MODE ? "docked" : "handheld");
    sProfEnabled = true;
}

bool Ship::SwitchProf::Enabled() {
    return sProfEnabled.load(std::memory_order_relaxed);
}

uint64_t Ship::SwitchProf::Now() {
    return armGetSystemTick();
}

void Ship::SwitchProf::Add(SwitchProfSlot slot, uint64_t startTick) {
    sProfAccum[slot].fetch_add(armGetSystemTick() - startTick, std::memory_order_relaxed);
    sProfCalls[slot].fetch_add(1, std::memory_order_relaxed);
}

void Ship::SwitchProf::NetLogWriteRaw(const char* data, size_t len) {
    int sock = sNetLogClient.load();
    while (sock >= 0 && len > 0) {
        ssize_t n = write(sock, data, len);
        if (n <= 0) {
            sNetLogClient = -1;
            sProfEnabled = false;
            return;
        }
        data += n;
        len -= n;
    }
}

void Ship::SwitchProf::SetAudioBuffered(int samples) {
    sAudioBuffered = samples;
    if (samples < sAudioBufferedMin) {
        sAudioBufferedMin = samples;
    }
}

static uint32_t GetClockMHz(PcvModuleId module) {
    if (!sClkrstInitialized) {
        return 0;
    }
    ClkrstSession session = {};
    u32 hz = 0;
    if (R_SUCCEEDED(clkrstOpenSession(&session, module, 3))) {
        clkrstGetClockRate(&session, &hz);
        clkrstCloseSession(&session);
    }
    return hz / 1000000;
}

void Ship::SwitchProf::FrameEnd() {
    static uint64_t windowStart = 0;
    static uint64_t lastFrame = 0;
    static uint32_t frames = 0;
    static uint64_t maxFrame = 0;
    static uint32_t slowFrames = 0;

    if (!Enabled()) {
        return;
    }
    const uint64_t now = armGetSystemTick();
    if (windowStart == 0) {
        windowStart = lastFrame = now;
        for (int i = 0; i < PROF_COUNT; i++) {
            sProfAccum[i] = 0;
            sProfCalls[i] = 0;
        }
        return;
    }
    const uint64_t dt = now - lastFrame;
    lastFrame = now;
    frames++;
    if (dt > maxFrame) {
        maxFrame = dt;
    }
    if (armTicksToNs(dt) > 20000000) { // > 20 ms
        slowFrames++;
    }

    const uint64_t window = now - windowStart;
    if (armTicksToNs(window) < 1000000000) {
        return;
    }

    auto ms = [](uint64_t ticks) { return armTicksToNs(ticks) / 1000000.0; };
    const double windowMs = ms(window);
    double slot[PROF_COUNT];
    uint32_t calls[PROF_COUNT];
    for (int i = 0; i < PROF_COUNT; i++) {
        slot[i] = ms(sProfAccum[i].exchange(0)) * 1000.0 / windowMs; // ms per second
        calls[i] = sProfCalls[i].exchange(0);
    }
    static bool firstWindow = true;
    if (firstWindow) {
        NetLogWrite("[prof] first window, reading clocks (clkrst %d)\n", sClkrstInitialized ? 1 : 0);
    }
    const uint32_t cpuMHz = GetClockMHz(PcvModuleId_CpuBus);
    const uint32_t gpuMHz = GetClockMHz(PcvModuleId_GPU);
    const uint32_t emcMHz = GetClockMHz(PcvModuleId_EMC);
    if (firstWindow) {
        NetLogWrite("[prof] clocks ok\n");
        firstWindow = false;
    }
    NetLogWrite("[prof] fps %.1f  max %.1fms  slow %u | ms/s: logic %.0f render %.0f audwait %.0f swap %.0f "
           "mix %.0f(%u) shader %.0f(%u) | audbuf %d min %d | cpu %u gpu %u emc %u\n",
           frames * 1000.0 / windowMs, ms(maxFrame), slowFrames, slot[PROF_LOGIC], slot[PROF_RENDER],
           slot[PROF_AUDIO_WAIT], slot[PROF_SWAP], slot[PROF_AUDIO_MIX], calls[PROF_AUDIO_MIX], slot[PROF_SHADER],
           calls[PROF_SHADER], sAudioBuffered.load(), sAudioBufferedMin == INT32_MAX ? -1 : sAudioBufferedMin,
           cpuMHz, gpuMHz, emcMHz);
    // Render breakdown. Everything below runs inside "render"; whatever is left over is the
    // Fast3D interpreter itself (display list walk, CPU vertex transform/lighting, batching).
    const double glTotal = slot[PROF_GL_DRAW] + slot[PROF_GL_TEX] + slot[PROF_GL_READBACK] + slot[PROF_GL_OTHER];
    const double interp =
        slot[PROF_RENDER] - slot[PROF_SWAP] - glTotal - slot[PROF_IMGUI] - slot[PROF_SHADER];
    NetLogWrite("[gl]   ms/s: interp %.0f | gl draw %.0f (%.0f draws/frame) tex %.0f(%u) readback %.0f(%u) "
                "other %.0f | imgui %.0f\n",
                interp, slot[PROF_GL_DRAW], frames ? (double)calls[PROF_GL_DRAW] / frames : 0.0, slot[PROF_GL_TEX],
                calls[PROF_GL_TEX], slot[PROF_GL_READBACK], calls[PROF_GL_READBACK], slot[PROF_GL_OTHER],
                slot[PROF_IMGUI]);

    windowStart = now;
    frames = 0;
    maxFrame = 0;
    slowFrames = 0;
    sAudioBufferedMin = INT32_MAX;
}

void Ship::Switch::ThrowMissingOTR(const std::string& otrPath) {
    Ship::Switch::PrintErrorMessageToScreen("\x1b[2;2HYou've launched the Ship without the OTR file."
                                            "\x1b[4;2HPlease relaunch making sure %s exists."
                                            "\x1b[6;2HPress + to exit.",
                                            otrPath.c_str());
}
#endif
