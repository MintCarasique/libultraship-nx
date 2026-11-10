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

// ---------------------------------------------------------------------------------------------
// State machines of the platform layer. Each is one enum; transitions that matter for debugging
// are logged as "[state] <machine>: <from> -> <to>" (SD card log and nxlink), so a log shows what
// the port believed was going on. Overview: tools/switch/STATE_MACHINES.md in Shipwright-NX.
// ---------------------------------------------------------------------------------------------

// Applet integration (hooks, gameplay recording, clock services).
//   Boot -> PreInit -> Running -> Exited (-> Running again if the window is re-created)
enum class Lifecycle { Boot, PreInit, Running, Exited };

// Who owns the CPU clock.
//   Untouched: never written, or handed back. HOS / sys-clk decide. The default.
//   Pinned:    we set sPinnedHz and the app is in focus.
//   Yielded:   a pin is wanted (sPinnedHz) but the app is in the background, so the clock was
//              put back to stock; it is pinned again when focus returns.
enum class ClockState { Untouched, Pinned, Yielded };

// Software keyboard. Unknown until the first ImGui event, which brings SDL in line.
enum class KeyboardState { Unknown, Hidden, Shown };

// nxlink log / profiler link. Offline and Lost are final; only Connected sends anything.
enum class NetLogState { NotStarted, Offline, Connected, Lost };

static const char* Name(Lifecycle s) {
    static const char* const names[] = { "Boot", "PreInit", "Running", "Exited" };
    return names[(int)s];
}

static const char* Name(ClockState s) {
    static const char* const names[] = { "Untouched", "Pinned", "Yielded" };
    return names[(int)s];
}

static AppletHookCookie sAppletHookCookie;
static Lifecycle sLifecycle = Lifecycle::Boot;
static bool sExitRequested = false; // latched by the applet hook, read by the main loop
static bool sHasFocus = true;
static KeyboardState sKeyboard = KeyboardState::Unknown;
static bool sClkrstInitialized = false;
static ClockState sClockState = ClockState::Untouched;
static uint32_t sPinnedHz = 0; // meaningful in Pinned and Yielded
static std::atomic<NetLogState> sNetLog = NetLogState::NotStarted;

static void OnAppletHook(AppletHookType hook, void* param);
static void DetectAppletMode();
static void StartNetLog();

static void SetLifecycle(Lifecycle next) {
    if (next != sLifecycle) {
        SPDLOG_INFO("[state] lifecycle: {} -> {}", Name(sLifecycle), Name(next));
        sLifecycle = next;
    }
}

static void SetClockState(ClockState next) {
    if (next != sClockState) {
        SPDLOG_INFO("[state] cpu-clock: {} -> {} ({} MHz)", Name(sClockState), Name(next), sPinnedHz / 1000000);
        sClockState = next;
    }
}

// Raw clock write; ownership is tracked by the callers through sClockState
static Result WriteCpuClock(uint32_t hz) {
    Result rc = MAKERESULT(Module_Libnx, LibnxError_NotInitialized);
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

// Hands the CPU back at stock clocks, but only if we are the ones who changed them
static void ReleaseCpuClock() {
    if (sClockState == ClockState::Pinned) {
        WriteCpuClock(SWITCH_CPU_SPEEDS_VALUES[Ship::STOCK]);
    }
    SetClockState(ClockState::Untouched);
    sPinnedHz = 0;
}

void Ship::Switch::Init(SwitchPhase phase) {
    switch (phase) {
        case PreInitPhase:
            if (sLifecycle != Lifecycle::Boot) {
                break;
            }
            // Must run before anything touches stdout (spdlog's console sink, printf)
            StartNetLog();
            DetectAppletMode();
            SetLifecycle(Lifecycle::PreInit);
            break;
        case PostInitPhase:
            if (sLifecycle == Lifecycle::Running) {
                break; // already hooked
            }
            appletInitializeGamePlayRecording();
            appletSetGamePlayRecordingState(true);
            appletHook(&sAppletHookCookie, OnAppletHook, NULL);
            appletSetFocusHandlingMode(AppletFocusHandlingMode_NoSuspend);
            if (hosversionBefore(8, 0, 0)) {
                pcvInitialize();
            } else {
                sClkrstInitialized = R_SUCCEEDED(clkrstInitialize());
            }
            SetLifecycle(Lifecycle::Running);
            ApplyOverclock();
            break;
    }
}

void Ship::Switch::Exit() {
    if (sLifecycle != Lifecycle::Running) {
        return; // nothing was hooked or initialized (or it is already undone)
    }
    ReleaseCpuClock();
    if (hosversionBefore(8, 0, 0)) {
        pcvExit();
    } else if (sClkrstInitialized) {
        clkrstExit();
        sClkrstInitialized = false;
    }
    appletUnhook(&sAppletHookCookie);
    appletSetGamePlayRecordingState(false);
    SetLifecycle(Lifecycle::Exited);
}

void Ship::Switch::ImGuiProcessEvent(bool wantsTextInput) {
    if (wantsTextInput) {
        if (sKeyboard != KeyboardState::Shown) {
            ImGuiInputTextState* state = ImGui::GetInputTextState(ImGui::GetActiveID());
            if (state != nullptr) {
                state->ClearText();
            }

            sKeyboard = KeyboardState::Shown;
            SDL_StartTextInput();
        }
    } else {
        if (sKeyboard != KeyboardState::Hidden) {
            sKeyboard = KeyboardState::Hidden;
            SDL_StopTextInput();
        }
    }
}

bool Ship::Switch::IsRunning() {
    return !sExitRequested;
}

uint32_t Ship::Switch::GetConnectedPadSlots() {
    uint32_t slots = 0;
    for (uint32_t i = 0; i < 8; i++) {
        u32 styles = hidGetNpadStyleSet((HidNpadIdType)(HidNpadIdType_No1 + i));
        if (i == 0) {
            styles |= hidGetNpadStyleSet(HidNpadIdType_Handheld);
        }
        if (styles != 0) {
            slots |= 1u << i;
        }
    }
    return slots;
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
    const int32_t perfMode = CVarGetInteger(CVAR_SWITCH_PERF_MODE, (int32_t)SYSTEM_DEFAULT);

    if (perfMode < 0 || perfMode > POWERSAVINGM3) {
        // System Default (or a value we don't know): leave clock management to the system /
        // sys-clk. If we pinned a clock earlier this session, hand the CPU back at stock once.
        ReleaseCpuClock();
        SPDLOG_INFO("Switch perf mode: system default (clocks untouched)");
        return;
    }

    const uint32_t hz = SWITCH_CPU_SPEEDS_VALUES[perfMode];
    if (!sHasFocus) {
        // In the background the clock stays at stock; the pin is applied when focus returns
        if (sClockState == ClockState::Pinned) {
            WriteCpuClock(SWITCH_CPU_SPEEDS_VALUES[STOCK]);
        }
        sPinnedHz = hz;
        SetClockState(ClockState::Yielded);
        return;
    }

    const Result rc = WriteCpuClock(hz);
    SPDLOG_INFO("Switch perf mode {} -> {} MHz (rc 0x{:x}, clkrst {})", perfMode, hz / 1000000, rc,
                sClkrstInitialized);
    if (R_SUCCEEDED(rc)) {
        sPinnedHz = hz;
        SetClockState(ClockState::Pinned);
    } else if (sClockState == ClockState::Yielded) {
        // The clock is at stock, where we left it when focus was lost: nothing of ours remains
        ReleaseCpuClock();
    }
    // A failed write from Untouched or Pinned changed nothing, so the state stands
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
            SPDLOG_INFO("[state] lifecycle: exit requested");
            sExitRequested = true;
            break;

        case AppletHookType_OnFocusState:
            sHasFocus = appletGetFocusState() == AppletFocusState_InFocus;
            SPDLOG_INFO("[state] focus: {}", sHasFocus ? "in focus" : "background");

            if (!sHasFocus) {
                // Drop our pinned clock while in the background (only if we pinned one)
                if (sClockState == ClockState::Pinned) {
                    WriteCpuClock(SWITCH_CPU_SPEEDS_VALUES[Ship::STOCK]);
                    SetClockState(ClockState::Yielded);
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

        case AppletHookType_OnOperationMode:
            // The window and the swapchain pick the new size up on their own (GetDisplaySize)
            SPDLOG_INFO("[state] display: {}", appletGetOperationMode() == DOCKED_MODE ? "docked" : "handheld");
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
static std::atomic<uint64_t> sProfAccum[Ship::PROF_COUNT];
static std::atomic<uint32_t> sProfCalls[Ship::PROF_COUNT];
static std::atomic<int> sAudioBuffered = 0;
static int sAudioBufferedMin = INT32_MAX;

// Raw write() to the nxlink-redirected stdout fd: bypasses stdio buffering so it is safe from
// any thread, including the exception handler.
static void NetLogWrite(const char* fmt, ...) {
    if (sNetLog.load() != NetLogState::Connected) {
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
    if (write(STDOUT_FILENO, buf, len) < 0) {
        sNetLog = NetLogState::Lost; // the host went away: stop sending and profiling
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
    if (sNetLog.load() != NetLogState::NotStarted) {
        return;
    }
    sNetLog = NetLogState::Offline;
    if (__nxlink_host.s_addr == 0 || R_FAILED(socketInitializeDefault())) {
        return;
    }
    // nxlinkStdio() redirects stdout/stderr to the host: from here the link is STDOUT_FILENO
    if (nxlinkStdio() < 0) {
        return;
    }
    sNetLog = NetLogState::Connected;
    NetLogWrite("[netlog] nxlink connected, profiler on (mode: %s)\n",
                appletGetOperationMode() == DOCKED_MODE ? "docked" : "handheld");
}

bool Ship::SwitchProf::Enabled() {
    return sNetLog.load(std::memory_order_relaxed) == NetLogState::Connected;
}

uint64_t Ship::SwitchProf::Now() {
    return armGetSystemTick();
}

void Ship::SwitchProf::Add(SwitchProfSlot slot, uint64_t startTick) {
    sProfAccum[slot].fetch_add(armGetSystemTick() - startTick, std::memory_order_relaxed);
    sProfCalls[slot].fetch_add(1, std::memory_order_relaxed);
}

void Ship::SwitchProf::NetLogWriteRaw(const char* data, size_t len) {
    while (len > 0 && sNetLog.load() == NetLogState::Connected) {
        ssize_t n = write(STDOUT_FILENO, data, len);
        if (n <= 0) {
            sNetLog = NetLogState::Lost;
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
