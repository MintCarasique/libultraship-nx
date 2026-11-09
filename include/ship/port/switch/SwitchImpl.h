#pragma once

#ifdef __SWITCH__
#include <cstdint>
#include <string>

namespace Ship {
// SYSTEM_DEFAULT leaves the clocks alone (HOS / sys-clk decide); the others pin the CPU clock.
enum SwitchProfiles { MAXIMUM, HIGH, BOOST, STOCK, POWERSAVINGM1, POWERSAVINGM2, POWERSAVINGM3, SYSTEM_DEFAULT };

enum SwitchPhase { PreInitPhase, PostInitPhase };

static const char* const SWITCH_CPU_PROFILES[] = {
    "Maximum Performance", "High Performance",   "Boost Performance",  "Stock Performance",
    "Powersaving Mode 1",  "Powersaving Mode 2", "Powersaving Mode 3", "System Default (don't change)",
};

// CVar holding the selected SwitchProfiles value
#define CVAR_SWITCH_PERF_MODE "gEnhancements.SwitchPerfMode"

class Switch {
  public:
    static void Init(SwitchPhase phase);
    static void Exit();
    static void ImGuiProcessEvent(bool wantsTextInput);
    static bool IsRunning();
    static void GetDisplaySize(int* width, int* height);
    // Bit i set: player slot i has a controller (slot 0 includes the handheld Joy-Cons)
    static uint32_t GetConnectedPadSlots();
    static void ApplyOverclock();
    static void ThrowMissingOTR(const std::string& otrPath);
    // Moves the calling thread off the main core (all threads start on core 0 under libnx)
    // and optionally raises its priority. Used for the audio thread.
    static void PinCurrentThreadToCore(int core, bool raisePriority);
    // addr minus the module base, for symbolizing with addr2line against soh.elf
    static uintptr_t RelativeAddress(uintptr_t addr);
    // True if addr is inside a mapped executable image (stand-in for dladdr)
    static bool IsModuleAddress(uintptr_t addr);
    static void PrintErrorMessageToScreen(const char* str, ...);
};
// Lightweight frame profiler. Only active while a client is connected to the network log
// (port SWITCH_NETLOG_PORT, e.g. `nc <switch-ip> 28771`); prints one summary line per second.
#define SWITCH_NETLOG_PORT 28771
enum SwitchProfSlot {
    PROF_LOGIC,       // game update between two Graph_ProcessGfxCommands calls
    PROF_RENDER,      // display list interpretation + GL submission
    PROF_AUDIO_WAIT,  // main thread blocked on the audio thread
    PROF_SWAP,        // frame pacing sleep + SDL_GL_SwapWindow
    PROF_AUDIO_MIX,   // audio thread synthesis
    PROF_SHADER,      // shader compile + link
    PROF_GL_DRAW,     // GfxRenderingAPIOGL::DrawTriangles (VBO upload + draw call + state)
    PROF_GL_TEX,      // texture uploads
    PROF_GL_READBACK, // GPU->CPU reads (GetPixelDepth / ReadFramebufferToCPU): full GPU syncs
    PROF_GL_OTHER,    // remaining GL backend calls (shader/texture binds, framebuffers, clears)
    PROF_IMGUI,       // ImGui draw data submission
    PROF_COUNT
};
class SwitchProf {
  public:
    static bool Enabled();
    static uint64_t Now();
    static void Add(SwitchProfSlot slot, uint64_t startTick);
    static void SetAudioBuffered(int samples);
    // Raw bytes to the nxlink host (no-op when not connected)
    static void NetLogWriteRaw(const char* data, size_t len);
    static void FrameEnd(); // once per presented frame
};
struct SwitchProfScope {
    SwitchProfSlot slot;
    uint64_t start;
    explicit SwitchProfScope(SwitchProfSlot s) : slot(s), start(SwitchProf::Enabled() ? SwitchProf::Now() : 0) {
    }
    ~SwitchProfScope() {
        if (start != 0) {
            SwitchProf::Add(slot, start);
        }
    }
};
}; // namespace Ship
#endif
