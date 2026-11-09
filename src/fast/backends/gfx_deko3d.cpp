// Fast3D rendering backend on deko3d. See include/fast/backends/gfx_deko3d.h for the overview.
//
// Layout of this file:
//   1. small utilities (hash, debug callback, image tile-size fix)
//   2. GPU memory: linear rings (commands, uploads), an image heap with deferred frees, code heap
//   3. the shader compiler service (uam on its own thread with a big stack, DKSH cache on SD)
//   4. GfxRenderingAPIDeko3d::Impl: device, frames, textures, framebuffers, shaders, draws
//   5. the ImGui renderer
//   6. the GfxRenderingAPI entry points, forwarding to Impl
#ifdef __SWITCH__
#include "fast/backends/gfx_deko3d.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <variant>

#include <switch.h>
#include <imgui.h>
#include <prism/processor.h>
#include <spdlog/spdlog.h>
#include <uam_api.h>

#include "fast/interpreter.h"
#include "fast/lus_gbi.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include "ship/resource/ResourceManager.h"
#include "ship/resource/factory/ShaderFactory.h"
#include "ship/resource/type/Shader.h"
#include "ship/port/switch/SwitchImpl.h"

namespace Fast {
// Shared with the OpenGL backend (gfx_opengl.cpp): builds the colour-combiner expressions. The GLSL
// names it emits (vInputN, texValN, texel, noise helpers) are the same in the deko3d template.
prism::ContextTypes* append_formula(prism::ContextTypes* _, prism::ContextTypes* a_arg, prism::ContextTypes* a_single,
                                    prism::ContextTypes* a_mult, prism::ContextTypes* a_mix,
                                    prism::ContextTypes* a_with_alpha, prism::ContextTypes* a_only_alpha,
                                    prism::ContextTypes* a_alpha, prism::ContextTypes* a_first_cycle);
const char* gfx_get_shader(int16_t id);
} // namespace Fast

#define SWITCH_PROF_DK(slot) Ship::SwitchProfScope switchProfScope(Ship::slot)
// Bring-up trace: straight to stderr (nxlink), only for the first frames
static int gDkTraceFrames = 0; // set > 0 to trace the first frames
#define DKTRACE(...)                                  \
    do {                                              \
        if (gDkTraceFrames > 0) {                     \
            fprintf(stderr, "[dktrace] " __VA_ARGS__); \
            fputc('\n', stderr);                     \
        }                                             \
    } while (0)

namespace Fast {
namespace {

// ---------------------------------------------------------------------------------------------
// 1. Utilities
// ---------------------------------------------------------------------------------------------

constexpr uint32_t kFramesInFlight = 2;
constexpr uint32_t kCmdMemSize = 4 * 1024 * 1024;
constexpr uint32_t kCmdMemExtraChunk = 1 * 1024 * 1024;
constexpr uint32_t kUploadRingSize = 48 * 1024 * 1024;
constexpr uint32_t kImageHeapBlockSize = 64 * 1024 * 1024;
constexpr uint32_t kCodeHeapSize = 32 * 1024 * 1024;
constexpr uint32_t kReadbackSize = 8 * 1024 * 1024;
constexpr uint32_t kMaxImageDescriptors = 8192;
constexpr uint32_t kNumSamplers = 33; // 2 filters x 4 wrap S x 4 wrap T, + the ImGui sampler
constexpr uint32_t kImGuiSampler = 32;
constexpr uint32_t kSwapchainImages = 3;
// DKSH cache key: bump when the template conventions, uam or its patches change
constexpr const char* kShaderCacheTag = "soh-dk1/uam1.1.0-p7";
// Temporary: shader debug visualisation for the texture investigation
constexpr int kDkDebugMode = 0;

uint64_t Fnv1a64(const void* data, size_t len, uint64_t h = 0xcbf29ce484222325ULL) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint32_t AlignUp(uint32_t v, uint32_t a) {
    return (v + a - 1) & ~(a - 1);
}

struct HashPairU64 {
    size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
        return std::hash<uint64_t>{}(p.first * 0x9E3779B97F4A7C15ULL ^ p.second);
    }
};

void DkDebugCallback(void* userData, const char* context, DkResult result, const char* message) {
    // Release libdeko3d aborts on errors before calling us; the debug library calls us first.
    // stderr goes straight to the nxlink host (unbuffered), unlike the async logger
    fprintf(stderr, "[deko3d] %s (result %d): %s\n", context ? context : "?", (int)result, message ? message : "");
    SPDLOG_CRITICAL("deko3d: {} (result {}): {}", context ? context : "?", (int)result, message ? message : "");
    spdlog::default_logger()->flush();
    if (result != DkResult_Success) {
        svcSleepThread(1000000000ULL); // let the log reach the SD card / host before deko3d aborts
    }
}

// deko3d 0.5.0 bug: for an image whose level 0 has 6-8 rows of 8-row GOBs, the layout and the
// descriptor disagree about the tile height. Force the shrunk tile size the hardware will use.
// (Workaround documented by the SwitchWakerHD port.)
void ImageTileSizeFix(DkImageLayoutMaker& m, uint32_t rows) {
    if (m.flags & (DkImageFlags_PitchLinear | DkImageFlags_CustomTileSize | DkImageFlags_UsageVideo)) {
        return;
    }
    if (m.type != DkImageType_2D) {
        return;
    }
    const uint32_t gobs = (rows + rows / 2 + 7) / 8;
    uint32_t tile = gobs >= 16 ? 4 : gobs >= 8 ? 3 : gobs >= 4 ? 2 : gobs >= 2 ? 1 : 0;
    uint32_t shrunk = tile;
    while (shrunk && (8u << (shrunk - 1)) >= rows) {
        shrunk--;
    }
    if (shrunk == tile) {
        return;
    }
    m.flags |= DkImageFlags_CustomTileSize;
    m.tileSize = (DkTileSize)shrunk;
}

// ---------------------------------------------------------------------------------------------
// 2. GPU memory
// ---------------------------------------------------------------------------------------------

struct MemRing {
    DkMemBlock block = nullptr;
    uint8_t* cpu = nullptr;
    DkGpuAddr gpu = 0;
    uint32_t size = 0;
    uint32_t offset = 0;

    void Create(DkDevice dev, uint32_t sz, uint32_t flags) {
        DkMemBlockMaker m;
        dkMemBlockMakerDefaults(&m, dev, AlignUp(sz, DK_MEMBLOCK_ALIGNMENT));
        m.flags = flags;
        block = dkMemBlockCreate(&m);
        cpu = static_cast<uint8_t*>(dkMemBlockGetCpuAddr(block));
        gpu = dkMemBlockGetGpuAddr(block);
        size = sz;
        offset = 0;
    }
    void Destroy() {
        if (block) {
            dkMemBlockDestroy(block);
            block = nullptr;
        }
    }
    // Returns the offset, or UINT32_MAX when full
    uint32_t Alloc(uint32_t sz, uint32_t align) {
        uint32_t off = AlignUp(offset, align);
        if (off + sz > size) {
            return UINT32_MAX;
        }
        offset = off + sz;
        return off;
    }
};

struct ImageAlloc {
    int32_t block = -1;
    uint32_t offset = 0;
    uint32_t size = 0;
    bool Valid() const {
        return block >= 0;
    }
};

// First-fit sub-allocator over big image memory blocks.
struct ImageHeap {
    struct Block {
        DkMemBlock mem;
        std::vector<std::pair<uint32_t, uint32_t>> freeList; // offset, size (sorted by offset)
    };
    DkDevice dev = nullptr;
    std::vector<Block> blocks;

    ImageAlloc Alloc(uint32_t size, uint32_t align) {
        size = AlignUp(size, align);
        for (int32_t bi = 0; bi < (int32_t)blocks.size(); bi++) {
            auto& fl = blocks[bi].freeList;
            for (size_t i = 0; i < fl.size(); i++) {
                uint32_t start = AlignUp(fl[i].first, align);
                uint32_t pad = start - fl[i].first;
                if (fl[i].second >= pad + size) {
                    uint32_t end = start + size;
                    uint32_t freeEnd = fl[i].first + fl[i].second;
                    std::vector<std::pair<uint32_t, uint32_t>> repl;
                    if (pad) {
                        repl.push_back({ fl[i].first, pad });
                    }
                    if (freeEnd > end) {
                        repl.push_back({ end, freeEnd - end });
                    }
                    fl.erase(fl.begin() + i);
                    fl.insert(fl.begin() + i, repl.begin(), repl.end());
                    return { bi, start, size };
                }
            }
        }
        uint32_t blockSize = std::max(kImageHeapBlockSize, AlignUp(size, DK_MEMBLOCK_ALIGNMENT));
        DkMemBlockMaker m;
        dkMemBlockMakerDefaults(&m, dev, blockSize);
        m.flags = DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image;
        Block b;
        b.mem = dkMemBlockCreate(&m);
        b.freeList.push_back({ size, blockSize - size });
        blocks.push_back(std::move(b));
        SPDLOG_INFO("deko3d: image heap grew to {} blocks", blocks.size());
        return { (int32_t)blocks.size() - 1, 0, size };
    }

    void Free(const ImageAlloc& a) {
        if (!a.Valid()) {
            return;
        }
        auto& fl = blocks[a.block].freeList;
        auto it = std::lower_bound(fl.begin(), fl.end(), std::make_pair(a.offset, 0u));
        it = fl.insert(it, { a.offset, a.size });
        // merge with next
        auto next = it + 1;
        if (next != fl.end() && it->first + it->second == next->first) {
            it->second += next->second;
            fl.erase(next);
        }
        // merge with previous
        if (it != fl.begin()) {
            auto prev = it - 1;
            if (prev->first + prev->second == it->first) {
                prev->second += it->second;
                fl.erase(it);
            }
        }
    }

    DkMemBlock Mem(const ImageAlloc& a) const {
        return blocks[a.block].mem;
    }
};

// ---------------------------------------------------------------------------------------------
// 3. Shader compiler service
// ---------------------------------------------------------------------------------------------

// uam is not reentrant and recurses deeply (Mesa's GLSL front end, nv50_ir): it runs on one
// thread with an 8 MB stack. Callers block until their shader is ready; compiled shaders are
// cached on the SD card so each combiner is compiled once per install.
class ShaderCompiler {
  public:
    void Start() {
        mCacheDir = Ship::Context::GetPathRelativeToAppDirectory("cache/dksh");
        std::error_code ec;
        std::filesystem::create_directories(mCacheDir, ec);
        Result rc = threadCreate(&mThread, &ShaderCompiler::ThreadEntry, this, nullptr, 8 * 1024 * 1024, 0x3B, -2);
        if (R_FAILED(rc)) {
            SPDLOG_CRITICAL("deko3d: cannot create the shader compiler thread (0x{:x})", rc);
            return;
        }
        threadStart(&mThread);
        mStarted = true;
    }

    // Returns the DKSH bytes for this stage and source, or an empty vector on failure.
    std::vector<uint8_t> Get(uam::Stage stage, const std::string& glsl) {
        const char stageTag = stage == uam::Stage::Vertex ? 'v' : 'f';
        uint64_t key = Fnv1a64(kShaderCacheTag, strlen(kShaderCacheTag));
        key = Fnv1a64(&stageTag, 1, key);
        key = Fnv1a64(glsl.data(), glsl.size(), key);
        char name[32];
        snprintf(name, sizeof(name), "%016llx.dksh", (unsigned long long)key);
        const std::string path = mCacheDir + "/" + name;

        {
            std::ifstream f(path, std::ios::binary);
            if (f) {
                std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                if (data.size() > 24 && memcmp(data.data(), "DKSH", 4) == 0) {
                    return data;
                }
            }
        }

        if (!mStarted) {
            return {};
        }
        std::unique_lock<std::mutex> lock(mMutex);
        mJobStage = stage;
        mJobSource = &glsl;
        mJobDone = false;
        mHasJob = true;
        mCv.notify_all();
        mCv.wait(lock, [this] { return mJobDone; });
        std::vector<uint8_t> result = std::move(mJobResult);
        lock.unlock();

        if (!result.empty()) {
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(result.data()), result.size());
        }
        return result;
    }

  private:
    static void ThreadEntry(void* arg) {
        static_cast<ShaderCompiler*>(arg)->Run();
    }

    void Run() {
        uam::init(true);
        SPDLOG_INFO("deko3d: uam ready");
        while (true) {
            std::unique_lock<std::mutex> lock(mMutex);
            mCv.wait(lock, [this] { return mHasJob; });
            mHasJob = false;
            const uam::Stage stage = mJobStage;
            const std::string* src = mJobSource;
            lock.unlock();

            const uint64_t t0 = armGetSystemTick();
            uam::Result r = uam::compile(stage, src->c_str());
            const double ms = armTicksToNs(armGetSystemTick() - t0) / 1e6;
            if (!r.ok) {
                SPDLOG_ERROR("deko3d: shader compile failed ({} ms):\n{}\n--- source ---\n{}", ms, r.log, *src);
            } else {
                SPDLOG_INFO("deko3d: compiled {} shader in {:.1f} ms ({} bytes)",
                            stage == uam::Stage::Vertex ? "vertex" : "fragment", ms, r.dksh.size());
                if (!r.log.empty()) {
                    SPDLOG_WARN("deko3d: uam: {}", r.log);
                }
            }

            lock.lock();
            mJobResult = r.ok ? std::move(r.dksh) : std::vector<uint8_t>();
            mJobDone = true;
            mCv.notify_all();
        }
    }

    Thread mThread{};
    bool mStarted = false;
    std::string mCacheDir;
    std::mutex mMutex;
    std::condition_variable mCv;
    bool mHasJob = false;
    bool mJobDone = false;
    uam::Stage mJobStage = uam::Stage::Vertex;
    const std::string* mJobSource = nullptr;
    std::vector<uint8_t> mJobResult;
};

struct DkshHeader {
    uint32_t magic; // "DKSH"
    uint32_t header_sz;
    uint32_t control_sz;
    uint32_t code_sz;
    uint32_t programs_off;
    uint32_t num_programs;
};

// ---------------------------------------------------------------------------------------------
// Resource records
// ---------------------------------------------------------------------------------------------

struct ShaderProgramDk {
    uint64_t id0 = 0, id1 = 0;
    uint8_t numInputs = 0;
    bool usedTextures[SHADER_MAX_TEXTURES] = {};
    DkShader vs{}, fs{};
    bool valid = false;
    DkVtxAttribState attribs[16] = {};
    uint32_t numAttribs = 0;
    uint32_t stride = 0;
    bool layoutChecked = true; // false: compare the interpreter's vertex size with the layout
    std::string vsSrc, fsSrc;
};

struct TextureDk {
    bool allocated = false;
    bool hasImage = false;
    DkImage image{};
    ImageAlloc mem;
    uint32_t width = 0, height = 0;
    uint8_t sampler = 0;
    int filtering = FILTER_NONE;
    bool isFramebuffer = false;
    uint64_t lastUseSerial = 0; // barrier epoch in which a draw last sampled it
};

struct FramebufferDk {
    uint32_t width = 0, height = 0;
    bool invertY = false;
    bool hasDepth = false;
    DkImage color{};
    ImageAlloc colorMem;
    bool hasColor = false;
    DkImage depth{};
    ImageAlloc depthMem;
    bool hasDepthImage = false;
    uint32_t texId = 0; // texture slot that samples this framebuffer's colour
};

struct FsParams {
    int32_t frameCount;
    float noiseScale;
    float primDepth;
    float pad0;
    int32_t texSize[4];
    int32_t texFiltering[4];
};
static_assert(sizeof(FsParams) == 48);

struct DeferredFree {
    ImageAlloc mem;
    uint32_t texId = 0; // texture slot to return to the free list (0 = none)
};

struct FrameSlot {
    DkMemBlock cmdMem = nullptr;
    std::vector<DkMemBlock> extraCmdMem;
    uint32_t extraCmdUsed = 0;
    MemRing upload;
    DkFence fence{};
    bool submitted = false;
    std::vector<DeferredFree> deferred;
};

constexpr const char* kImGuiVs = R"(#version 460
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
layout(location = 2) in vec4 aColor;
layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 vColor;
layout(std140, binding = 0) uniform ImGuiParams { vec4 uScaleTranslate; };
void main() {
    vUv = aUv;
    vColor = aColor;
    gl_Position = vec4(aPos * uScaleTranslate.xy + uScaleTranslate.zw, 0.0, 1.0);
}
)";

constexpr const char* kImGuiFs = R"(#version 460
layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vColor;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D uTex;
void main() {
    outColor = vColor * texture(uTex, vUv);
}
)";

} // namespace

// ---------------------------------------------------------------------------------------------
// 4. Impl
// ---------------------------------------------------------------------------------------------

struct GfxRenderingAPIDeko3d::Impl {
    DkDevice device = nullptr;
    DkQueue queue = nullptr;
    DkCmdBuf cmdbuf = nullptr;
    FrameSlot slots[kFramesInFlight];
    uint32_t frameIndex = 0; // monotonic
    FrameSlot* cur = nullptr;
    bool frameOpen = false;

    ImageHeap images;
    MemRing codeHeap;
    MemRing descMem; // image descriptors, then sampler descriptors
    DkGpuAddr imageDescAddr = 0;
    DkGpuAddr samplerDescAddr = 0;
    MemRing readback;
    DkMemBlock immCmdMem = nullptr; // command memory for read-backs outside a frame

    ShaderCompiler compiler;

    // swapchain / framebuffer 0
    DkSwapchain swapchain = nullptr;
    DkImage swapImages[kSwapchainImages]{};
    ImageAlloc swapMem[kSwapchainImages];
    int swapSlot = -1;
    uint32_t swapWidth = 0, swapHeight = 0;
    bool vsync = true;

    std::vector<FramebufferDk> fbs;
    int curFb = 0;

    std::vector<TextureDk> textures; // index = texture id = image descriptor slot
    std::vector<uint32_t> freeTexIds;
    int curTile = 0;
    uint32_t curTexIds[SHADER_MAX_TEXTURES] = {};

    std::unordered_map<std::pair<uint64_t, uint64_t>, ShaderProgramDk, HashPairU64> programs;
    ShaderProgramDk* curProg = nullptr;
    ShaderProgramDk* boundProg = nullptr;

    // per-frame uniform buffers (bound once, updated with push constants)
    DkGpuAddr fsUbo = 0, vsUbo = 0;
    uint8_t* fsUboCpu = nullptr;          // CPU view of fsUbo (diagnostics)
    uint8_t* slotFsUboCpu[kFramesInFlight] = {};
    int uboDiagFrames = 0;           // > 0: log the fragment UBO after the first frames
    bool dumpedTexturedShader = true; // false: log the first textured shader's source
    FsParams fsParams{};
    bool fsParamsDirty = true;
    float vsYScale = 1.0f, boundVsYScale = 0.0f;
    bool vsUboBound = false;

    // pipeline-ish state
    bool depthTest = false, depthMask = false, decal = false, useAlpha = false;
    int8_t boundDepthState = -1, boundDecal = -1, boundBlend = -1;
    float primDepth = 0.0f;
    int vpX = 0, vpY = 0, vpW = 0, vpH = 0;
    int scX = 0, scY = 0, scW = 0, scH = 0;
    bool viewportDirty = true, scissorDirty = true;
    bool shadersDirty = true, texturesDirty = true;
    DkResHandle boundHandles[SHADER_MAX_TEXTURES] = {};
    bool shaderCodeDirty = false;
    bool barrierPending = false;
    // Copies (texture uploads) run on the copy engine and blits on the 2D engine: a Fragments
    // barrier only waits for 3D work, so their results need a Full barrier before sampling.
    bool barrierPendingFull = false;
    uint64_t barrierEpoch = 1;

    FilteringMode filterMode = FILTER_THREE_POINT;
    bool srgbMode = false;
    int frameCount = 0;
    float noiseScale = 1.0f;

    // ImGui
    bool imguiReady = false;
    ShaderProgramDk imguiProg;
    uint32_t imguiFontTex = 0;

    // ------------------------------------------------------------------ setup

    void Init() {
        DkDeviceMaker dm;
        dkDeviceMakerDefaults(&dm);
        dm.cbDebug = DkDebugCallback;
        // OpenGL conventions for depth (clip z in -1..1). Window origin upper-left: see the header.
        dm.flags = DkDeviceFlags_DepthMinusOneToOne | DkDeviceFlags_OriginUpperLeft;
        device = dkDeviceCreate(&dm);

        DkQueueMaker qm;
        dkQueueMakerDefaults(&qm, device);
        qm.flags = DkQueueFlags_Graphics | DkQueueFlags_DisableZcull;
        queue = dkQueueCreate(&qm);

        DkCmdBufMaker cm;
        dkCmdBufMakerDefaults(&cm, device);
        cm.userData = this;
        cm.cbAddMem = &Impl::AddCmdMem;
        cmdbuf = dkCmdBufCreate(&cm);

        for (auto& s : slots) {
            DkMemBlockMaker m;
            dkMemBlockMakerDefaults(&m, device, kCmdMemSize);
            m.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
            s.cmdMem = dkMemBlockCreate(&m);
            s.upload.Create(device, kUploadRingSize, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached);
        }

        images.dev = device;
        codeHeap.Create(device, kCodeHeapSize,
                        DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached | DkMemBlockFlags_Code);
        const uint32_t imgDescBytes = kMaxImageDescriptors * sizeof(DkImageDescriptor);
        const uint32_t smpDescBytes = kNumSamplers * sizeof(DkSamplerDescriptor);
        descMem.Create(device, imgDescBytes + smpDescBytes, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached);
        imageDescAddr = descMem.gpu;
        samplerDescAddr = descMem.gpu + imgDescBytes;
        memset(descMem.cpu, 0, imgDescBytes + smpDescBytes);
        readback.Create(device, kReadbackSize, DkMemBlockFlags_CpuCached | DkMemBlockFlags_GpuUncached);
        {
            DkMemBlockMaker m;
            dkMemBlockMakerDefaults(&m, device, 256 * 1024);
            m.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
            immCmdMem = dkMemBlockCreate(&m);
        }

        // All sampler states Fast3D can ask for, written once
        auto* smp = reinterpret_cast<DkSamplerDescriptor*>(descMem.cpu + imgDescBytes);
        static const DkWrapMode wraps[4] = { DkWrapMode_Repeat, DkWrapMode_MirroredRepeat, DkWrapMode_ClampToEdge,
                                             DkWrapMode_MirrorClampToEdge };
        for (uint32_t i = 0; i < 32; i++) {
            DkSampler s;
            dkSamplerDefaults(&s);
            const bool linear = (i >> 4) & 1;
            s.minFilter = s.magFilter = linear ? DkFilter_Linear : DkFilter_Nearest;
            s.mipFilter = DkMipFilter_None;
            s.wrapMode[0] = wraps[(i >> 2) & 3];
            s.wrapMode[1] = wraps[i & 3];
            s.wrapMode[2] = DkWrapMode_ClampToEdge;
            dkSamplerDescriptorInitialize(&smp[i], &s);
        }
        {
            DkSampler s;
            dkSamplerDefaults(&s);
            s.minFilter = s.magFilter = DkFilter_Linear;
            s.mipFilter = DkMipFilter_None;
            s.wrapMode[0] = s.wrapMode[1] = s.wrapMode[2] = DkWrapMode_ClampToEdge;
            dkSamplerDescriptorInitialize(&smp[kImGuiSampler], &s);
        }

        // id 0 is "no texture": a real (1x1) image so that a stray binding never reads a garbage
        // descriptor
        textures.resize(1);
        {
            DkImageLayoutMaker lm;
            dkImageLayoutMakerDefaults(&lm, device);
            lm.format = DkImageFormat_RGBA8_Unorm;
            lm.dimensions[0] = 1;
            lm.dimensions[1] = 1;
            DkImageLayout layout;
            dkImageLayoutInitialize(&layout, &lm);
            TextureDk& t0 = textures[0];
            t0.mem = images.Alloc((uint32_t)dkImageLayoutGetSize(&layout), dkImageLayoutGetAlignment(&layout));
            dkImageInitialize(&t0.image, &layout, images.Mem(t0.mem), t0.mem.offset);
            t0.width = t0.height = 1;
            WriteImageDescriptor(0, t0.image);
            barrierPending = false;
        }
        fbs.resize(1); // framebuffer 0 = swapchain
        compiler.Start();
        SPDLOG_INFO("deko3d: renderer initialized");
    }

    static void AddCmdMem(void* userData, DkCmdBuf cb, size_t minReqSize) {
        auto* self = static_cast<Impl*>(userData);
        FrameSlot* s = self->cur;
        const uint32_t need = AlignUp(std::max<uint32_t>(kCmdMemExtraChunk, (uint32_t)minReqSize), DK_MEMBLOCK_ALIGNMENT);
        if (s->extraCmdUsed >= s->extraCmdMem.size()) {
            DkMemBlockMaker m;
            dkMemBlockMakerDefaults(&m, self->device, need);
            m.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
            s->extraCmdMem.push_back(dkMemBlockCreate(&m));
            SPDLOG_WARN("deko3d: command memory grew ({} extra chunks)", s->extraCmdMem.size());
        }
        DkMemBlock blk = s->extraCmdMem[s->extraCmdUsed++];
        dkCmdBufAddMemory(cb, blk, 0, kCmdMemExtraChunk);
    }

    void CheckQueue(const char* where) {
        if (dkQueueIsInErrorState(queue)) {
            SPDLOG_CRITICAL("deko3d: queue in error state before {}", where);
            spdlog::default_logger()->flush();
            Ship::Switch::PrintErrorMessageToScreen("\x1b[2;2Hdeko3d: GPU fault (queue in error state)\x1b[4;2Hbefore %s",
                                                    where);
        }
    }

    // ------------------------------------------------------------------ swapchain

    void DestroySwapchain() {
        if (!swapchain) {
            return;
        }
        dkQueueWaitIdle(queue);
        dkSwapchainDestroy(swapchain);
        swapchain = nullptr;
        for (uint32_t i = 0; i < kSwapchainImages; i++) {
            images.Free(swapMem[i]);
            swapMem[i] = {};
        }
    }

    void CreateSwapchain(uint32_t w, uint32_t h) {
        DKTRACE("CreateSwapchain %ux%u", w, h);
        DestroySwapchain();
        NWindow* win = nwindowGetDefault();
        // SDL's Switch video driver attaches an EGL surface to the default NWindow when it creates
        // the window, which configures its buffers; the swapchain can only connect once those are
        // released. (The EGL surface is never used with deko3d.)
        Result rel = nwindowReleaseBuffers(win);
        DKTRACE("nwindowReleaseBuffers 0x%x", rel);
        nwindowSetDimensions(win, w, h);

        DkImageLayoutMaker lm;
        dkImageLayoutMakerDefaults(&lm, device);
        // No HwCompression: framebuffer 0 can be a blit source (CopyFramebuffer) for the 2D engine
        lm.flags = DkImageFlags_UsageRender | DkImageFlags_UsagePresent | DkImageFlags_Usage2DEngine;
        lm.format = DkImageFormat_RGBA8_Unorm;
        lm.dimensions[0] = w;
        lm.dimensions[1] = h;
        DkImageLayout layout;
        dkImageLayoutInitialize(&layout, &lm);
        const uint32_t size = (uint32_t)dkImageLayoutGetSize(&layout);
        const uint32_t align = dkImageLayoutGetAlignment(&layout);
        DkImage const* ptrs[kSwapchainImages];
        for (uint32_t i = 0; i < kSwapchainImages; i++) {
            swapMem[i] = images.Alloc(size, align);
            dkImageInitialize(&swapImages[i], &layout, images.Mem(swapMem[i]), swapMem[i].offset);
            ptrs[i] = &swapImages[i];
        }
        DkSwapchainMaker sm;
        dkSwapchainMakerDefaults(&sm, device, win, ptrs, kSwapchainImages);
        DKTRACE("dkSwapchainCreate");
        swapchain = dkSwapchainCreate(&sm);
        DKTRACE("swapchain created");
        dkSwapchainSetSwapInterval(swapchain, vsync ? 1 : 0);
        swapWidth = w;
        swapHeight = h;

        FramebufferDk& fb0 = fbs[0];
        fb0.width = w;
        fb0.height = h;
        fb0.invertY = false;
        if (fb0.hasDepthImage) {
            DeferFree({ fb0.depthMem, 0 });
            fb0.hasDepthImage = false;
        }
        CreateDepthImage(fb0, w, h);
        SPDLOG_INFO("deko3d: swapchain {}x{}", w, h);
    }

    // ------------------------------------------------------------------ frames

    void BeginCommands() {
        FrameSlot& s = *cur;
        dkCmdBufClear(cmdbuf);
        s.extraCmdUsed = 0;
        dkCmdBufAddMemory(cmdbuf, s.cmdMem, 0, kCmdMemSize);

        dkCmdBufBindImageDescriptorSet(cmdbuf, imageDescAddr, kMaxImageDescriptors);
        dkCmdBufBindSamplerDescriptorSet(cmdbuf, samplerDescAddr, kNumSamplers);

        // Uniform buffers for this command list: written whole now, then updated in place with
        // push constants (which keep earlier draws' values intact).
        uint32_t off = s.upload.Alloc(DK_UNIFORM_BUF_ALIGNMENT * 2, DK_UNIFORM_BUF_ALIGNMENT);
        fsUbo = s.upload.gpu + off;
        fsUboCpu = s.upload.cpu + off;
        slotFsUboCpu[cur - slots] = fsUboCpu;
        vsUbo = fsUbo + DK_UNIFORM_BUF_ALIGNMENT;
        memcpy(s.upload.cpu + off, &fsParams, sizeof(fsParams));
        float vsInit[4] = { 1.0f, vsYScale, 1.0f, 1.0f };
        memcpy(s.upload.cpu + off + DK_UNIFORM_BUF_ALIGNMENT, vsInit, sizeof(vsInit));
        boundVsYScale = vsYScale;
        dkCmdBufBindUniformBuffer(cmdbuf, DkStage_Fragment, 0, fsUbo, DK_UNIFORM_BUF_ALIGNMENT);
        dkCmdBufBindUniformBuffer(cmdbuf, DkStage_Vertex, 0, vsUbo, DK_UNIFORM_BUF_ALIGNMENT);
        vsUboBound = true;

        DkRasterizerState rs;
        dkRasterizerStateDefaults(&rs);
        rs.depthClampEnable = 1;
        rs.cullMode = DkFace_None;
        dkCmdBufBindRasterizerState(cmdbuf, &rs);
        DkColorWriteState cws;
        dkColorWriteStateDefaults(&cws);
        dkCmdBufBindColorWriteState(cmdbuf, &cws);
        DkBlendState bs;
        dkBlendStateDefaults(&bs);
        dkBlendStateSetFactors(&bs, DkBlendFactor_SrcAlpha, DkBlendFactor_InvSrcAlpha, DkBlendFactor_SrcAlpha,
                               DkBlendFactor_InvSrcAlpha);
        dkCmdBufBindBlendState(cmdbuf, 0, &bs);
        dkCmdBufTiledCacheOp(cmdbuf, DkTiledCacheOp_Disable);

        MarkAllDirty();
        BindFramebufferTargets(curFb);
    }

    void MarkAllDirty() {
        shadersDirty = texturesDirty = viewportDirty = scissorDirty = true;
        fsParamsDirty = true;
        boundProg = nullptr;
        boundDepthState = boundDecal = boundBlend = -1;
        memset(boundHandles, 0xFF, sizeof(boundHandles));
    }

    void StartFrame() {
        const uint32_t slotIdx = frameIndex % kFramesInFlight;
        cur = &slots[slotIdx];
        if (cur->submitted) {
            dkFenceWait(&cur->fence, -1);
            cur->submitted = false;
        }
        ProcessDeferred(*cur);
        if (uboDiagFrames > 0 && slotFsUboCpu[slotIdx] != nullptr) {
            // What the GPU left in the fragment UBO after the frame two ago (push constants included)
            const int32_t* u = reinterpret_cast<const int32_t*>(slotFsUboCpu[slotIdx]);
            fprintf(stderr, "[dkdiag] fsUbo after frame: frame %d noise %f prim %f size %d %d %d %d filt %d %d | cpu-side size %d %d\n",
                    u[0], *(const float*)&u[1], *(const float*)&u[2], u[4], u[5], u[6], u[7], u[8], u[9],
                    fsParams.texSize[0], fsParams.texSize[1]);
            uboDiagFrames--;
        }
        cur->upload.offset = 0;

        const bool wantVsync =
            Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(CVAR_VSYNC_ENABLED, 1) != 0;
        if (wantVsync != vsync && swapchain) {
            vsync = wantVsync;
            dkSwapchainSetSwapInterval(swapchain, vsync ? 1 : 0);
        }

        CheckQueue("acquire");
        DKTRACE("acquire (frame %u)", frameIndex);
        swapSlot = dkQueueAcquireImage(queue, swapchain);
        DKTRACE("acquired slot %d", swapSlot);
        fbs[0].color = swapImages[swapSlot];
        fbs[0].hasColor = true;

        frameCount++;
        fsParams.frameCount = frameCount;
        frameOpen = true;
        curFb = 0;
        vsYScale = 1.0f;
        BeginCommands();
    }

    void EndFrame() {
        if (!frameOpen) {
            return;
        }
        // Debug screenshots over nxlink: every 15 s while the host log is connected
        static uint64_t lastShot = 0;
        const uint64_t now = armGetSystemTick();
        const bool shoot = Ship::SwitchProf::Enabled() && armTicksToNs(now - lastShot) > 15000000000ULL &&
                           (uint64_t)swapWidth * swapHeight * 4 <= kReadbackSize;
        if (shoot) {
            lastShot = now;
            dkCmdBufBarrier(cmdbuf, DkBarrier_Full, DkInvalidateFlags_L2Cache);
            DkImageView v;
            dkImageViewDefaults(&v, &fbs[0].color);
            DkImageRect r = { 0, 0, 0, swapWidth, swapHeight, 1 };
            DkCopyBuf dst = { readback.gpu, 0, 0 };
            dkMemBlockFlushCpuCache(readback.block, 0, swapWidth * swapHeight * 4);
            dkCmdBufCopyImageToBuffer(cmdbuf, &v, &r, &dst, 0);
        }

        CheckQueue("submit");
        DKTRACE("submit + present");
        DkCmdList list = dkCmdBufFinishList(cmdbuf);
        dkQueueSubmitCommands(queue, list);
        dkQueuePresentImage(queue, swapchain, swapSlot);
        dkQueueSignalFence(queue, &cur->fence, true);
        dkQueueFlush(queue);
        if (shoot) {
            dkQueueWaitIdle(queue);
            dkMemBlockFlushCpuCache(readback.block, 0, swapWidth * swapHeight * 4);
            SendScreenshot(readback.cpu, swapWidth, swapHeight);
        }
        cur->submitted = true;
        frameOpen = false;
        DKTRACE("frame %u done", frameIndex);
        if (gDkTraceFrames > 0) {
            gDkTraceFrames--;
        }
        frameIndex++;
    }

    // Half-size RGB, base64, framed by marker lines; nxlink_screens.py on the PC turns it into PNGs
    static void SendScreenshot(const uint8_t* rgba, uint32_t w, uint32_t h) {
        // Full resolution in handheld (1280x720); halved when docked (1920x1080)
        const uint32_t div = w > 1280 ? 2 : 1;
        const uint32_t ow = w / div, oh = h / div;
        std::vector<uint8_t> rgb(ow * oh * 3);
        for (uint32_t y = 0; y < oh; y++) {
            for (uint32_t x = 0; x < ow; x++) {
                const uint8_t* a = rgba + ((y * div) * w + x * div) * 4;
                for (int c = 0; c < 3; c++) {
                    rgb[(y * ow + x) * 3 + c] = a[c];
                }
            }
        }
        static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(rgb.size() * 4 / 3 + 128);
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "\n[screenshot %u %u]\n", ow, oh);
        out += hdr;
        for (size_t i = 0; i + 2 < rgb.size(); i += 3) {
            uint32_t v = (rgb[i] << 16) | (rgb[i + 1] << 8) | rgb[i + 2];
            out += tbl[(v >> 18) & 63];
            out += tbl[(v >> 12) & 63];
            out += tbl[(v >> 6) & 63];
            out += tbl[v & 63];
        }
        out += "\n[/screenshot]\n";
        Ship::SwitchProf::NetLogWriteRaw(out.data(), out.size());
    }

    // Submits what was recorded, waits for the GPU and starts a new command list on the same
    // slot (used for read-backs and when a ring runs out of space).
    void FlushAndWait() {
        if (!frameOpen) {
            return;
        }
        CheckQueue("flush");
        DkCmdList list = dkCmdBufFinishList(cmdbuf);
        dkQueueSubmitCommands(queue, list);
        dkQueueFlush(queue);
        dkQueueWaitIdle(queue);
        cur->upload.offset = 0;
        BeginCommands();
    }

    // Read-backs: inside a frame they are recorded into the frame's commands (then flushed and
    // waited for); between frames (the game asks for pixel depth from its update) they get a
    // small command list of their own, after everything already submitted.
    void BeginReadback() {
        if (frameOpen) {
            RequestBarrier();
            EmitPendingBarrier();
            return;
        }
        FrameSlot* s = cur ? cur : &slots[frameIndex % kFramesInFlight];
        cur = s;
        dkCmdBufClear(cmdbuf);
        s->extraCmdUsed = 0;
        dkCmdBufAddMemory(cmdbuf, immCmdMem, 0, 256 * 1024);
        dkCmdBufBarrier(cmdbuf, DkBarrier_Full, DkInvalidateFlags_Image | DkInvalidateFlags_L2Cache);
    }

    void EndReadback() {
        if (frameOpen) {
            FlushAndWait();
            return;
        }
        CheckQueue("readback");
        DkCmdList list = dkCmdBufFinishList(cmdbuf);
        dkQueueSubmitCommands(queue, list);
        dkQueueFlush(queue);
        dkQueueWaitIdle(queue);
    }

    uint32_t UploadAlloc(uint32_t size, uint32_t align, uint8_t** cpuOut, DkGpuAddr* gpuOut) {
        uint32_t off = cur->upload.Alloc(size, align);
        if (off == UINT32_MAX) {
            SPDLOG_WARN("deko3d: upload ring full ({} MB), flushing", kUploadRingSize >> 20);
            FlushAndWait();
            off = cur->upload.Alloc(size, align);
            if (off == UINT32_MAX) {
                SPDLOG_CRITICAL("deko3d: upload of {} bytes does not fit the ring", size);
                return UINT32_MAX;
            }
        }
        *cpuOut = cur->upload.cpu + off;
        *gpuOut = cur->upload.gpu + off;
        return off;
    }

    void DeferFree(const DeferredFree& d) {
        FrameSlot& s = cur ? *cur : slots[frameIndex % kFramesInFlight];
        s.deferred.push_back(d);
    }

    void ProcessDeferred(FrameSlot& s) {
        for (auto& d : s.deferred) {
            images.Free(d.mem);
            if (d.texId) {
                freeTexIds.push_back(d.texId);
            }
        }
        s.deferred.clear();
    }

    // A barrier before the next draw: makes uploads/copies/descriptor writes and render-target
    // writes visible to texture sampling.
    void RequestBarrier() {
        barrierPending = true;
    }

    void RequestFullBarrier() {
        barrierPending = true;
        barrierPendingFull = true;
    }

    void EmitPendingBarrier() {
        if (barrierPending) {
            dkCmdBufBarrier(cmdbuf, barrierPendingFull ? DkBarrier_Full : DkBarrier_Fragments,
                            DkInvalidateFlags_Image | DkInvalidateFlags_Descriptors | DkInvalidateFlags_L2Cache);
            barrierPending = false;
            barrierPendingFull = false;
            barrierEpoch++;
        }
    }

    // ------------------------------------------------------------------ textures

    uint32_t AllocTexId() {
        uint32_t id;
        if (!freeTexIds.empty()) {
            id = freeTexIds.back();
            freeTexIds.pop_back();
        } else {
            id = (uint32_t)textures.size();
            if (id >= kMaxImageDescriptors) {
                SPDLOG_CRITICAL("deko3d: out of texture slots ({})", kMaxImageDescriptors);
                return 0;
            }
            textures.emplace_back();
        }
        textures[id] = TextureDk{};
        textures[id].allocated = true;
        return id;
    }

    void WriteImageDescriptor(uint32_t id, const DkImage& image) {
        DkImageView view;
        dkImageViewDefaults(&view, &image);
        DkImageDescriptor desc;
        dkImageDescriptorInitialize(&desc, &view, false, false);
        if (frameOpen) {
            dkCmdBufPushData(cmdbuf, imageDescAddr + id * sizeof(DkImageDescriptor), &desc, sizeof(desc));
        } else {
            memcpy(descMem.cpu + id * sizeof(DkImageDescriptor), &desc, sizeof(desc));
        }
        RequestBarrier();
    }

    // Before overwriting a texture slot's descriptor or image: if a draw since the last barrier
    // sampled it, those draws must finish first.
    void GuardTextureWrite(TextureDk& t) {
        if (t.lastUseSerial == barrierEpoch) {
            barrierPending = true;
            EmitPendingBarrier();
        }
    }

    void UploadTexture(const uint8_t* rgba, uint32_t w, uint32_t h) {
        SWITCH_PROF_DK(PROF_GL_TEX);
        if (w == 0 || h == 0 || !frameOpen) {
            return;
        }
        const uint32_t id = curTexIds[curTile];
        if (id == 0 || id >= textures.size()) {
            return;
        }
        TextureDk& t = textures[id];
        GuardTextureWrite(t);

        // A new image every time: the old one may still be read by queued GPU work
        if (t.hasImage) {
            DeferFree({ t.mem, 0 });
            t.hasImage = false;
        }
        DkImageLayoutMaker lm;
        dkImageLayoutMakerDefaults(&lm, device);
        lm.format = DkImageFormat_RGBA8_Unorm;
        lm.dimensions[0] = w;
        lm.dimensions[1] = h;
        ImageTileSizeFix(lm, h);
        DkImageLayout layout;
        dkImageLayoutInitialize(&layout, &lm);
        t.mem = images.Alloc((uint32_t)dkImageLayoutGetSize(&layout), dkImageLayoutGetAlignment(&layout));
        dkImageInitialize(&t.image, &layout, images.Mem(t.mem), t.mem.offset);
        t.hasImage = true;
        t.width = w;
        t.height = h;

        uint8_t* cpu;
        DkGpuAddr gpu;
        if (UploadAlloc(w * h * 4, 256, &cpu, &gpu) == UINT32_MAX) {
            return;
        }
        memcpy(cpu, rgba, w * h * 4);
        DkImageView view;
        dkImageViewDefaults(&view, &t.image);
        DkCopyBuf src = { gpu, 0, 0 }; // tightly packed rows
        DkImageRect rect = { 0, 0, 0, w, h, 1 };
        dkCmdBufCopyBufferToImage(cmdbuf, &src, &view, &rect, 0);
        RequestFullBarrier();
        WriteImageDescriptor(id, t.image);
        texturesDirty = true;
    }

    static uint8_t WrapIndex(uint32_t cm) {
        switch (cm & 3) {
            case G_TX_NOMIRROR | G_TX_WRAP:
                return 0;
            case G_TX_MIRROR | G_TX_WRAP:
                return 1;
            case G_TX_NOMIRROR | G_TX_CLAMP:
                return 2;
            default:
                return 3; // mirror + clamp
        }
    }

    void SetSamplerParameters(int tile, bool linear, uint32_t cms, uint32_t cmt) {
        const uint32_t id = curTexIds[tile];
        if (id == 0 || id >= textures.size()) {
            return;
        }
        TextureDk& t = textures[id];
        const bool useLinear = linear && filterMode == FILTER_LINEAR;
        t.sampler = (uint8_t)((useLinear ? 16 : 0) | (WrapIndex(cms) << 2) | WrapIndex(cmt));
        t.filtering = !linear ? FILTER_LINEAR : FILTER_THREE_POINT;
        texturesDirty = true;
        fsParamsDirty = true;
    }

    // ------------------------------------------------------------------ framebuffers

    void CreateDepthImage(FramebufferDk& fb, uint32_t w, uint32_t h) {
        DkImageLayoutMaker lm;
        dkImageLayoutMakerDefaults(&lm, device);
        lm.flags = DkImageFlags_UsageRender;
        lm.format = DkImageFormat_Z24S8;
        lm.dimensions[0] = w;
        lm.dimensions[1] = h;
        DkImageLayout layout;
        dkImageLayoutInitialize(&layout, &lm);
        fb.depthMem = images.Alloc((uint32_t)dkImageLayoutGetSize(&layout), dkImageLayoutGetAlignment(&layout));
        dkImageInitialize(&fb.depth, &layout, images.Mem(fb.depthMem), fb.depthMem.offset);
        fb.hasDepthImage = true;
    }

    int CreateFramebuffer() {
        DKTRACE("CreateFramebuffer");
        FramebufferDk fb;
        fb.texId = AllocTexId();
        textures[fb.texId].isFramebuffer = true;
        fbs.push_back(fb);
        return (int)fbs.size() - 1;
    }

    void UpdateFramebufferParameters(int id, uint32_t w, uint32_t h, bool invertY, bool hasDepth) {
        SWITCH_PROF_DK(PROF_GL_OTHER);
        w = std::max(w, 1u);
        h = std::max(h, 1u);
        if (id == 0) {
            // The screen size follows the operation mode, whatever size the caller remembers
            int dw, dh;
            Ship::Switch::GetDisplaySize(&dw, &dh);
            w = dw;
            h = dh;
            if (!swapchain || w != swapWidth || h != swapHeight) {
                if (frameOpen) {
                    SPDLOG_WARN("deko3d: swapchain resize requested mid-frame; deferred");
                } else {
                    CreateSwapchain(w, h);
                }
            }
            fbs[0].invertY = invertY;
            return;
        }
        FramebufferDk& fb = fbs[id];
        const bool sizeChanged = fb.width != w || fb.height != h;
        if (sizeChanged || !fb.hasColor) {
            TextureDk& t = textures[fb.texId];
            GuardTextureWrite(t);
            if (fb.hasColor) {
                DeferFree({ fb.colorMem, 0 });
            }
            DkImageLayoutMaker lm;
            dkImageLayoutMakerDefaults(&lm, device);
            lm.flags = DkImageFlags_UsageRender | DkImageFlags_Usage2DEngine;
            lm.format = DkImageFormat_RGBX8_Unorm; // like OpenGL's GL_RGB8: alpha reads as 1
            lm.dimensions[0] = w;
            lm.dimensions[1] = h;
            DkImageLayout layout;
            dkImageLayoutInitialize(&layout, &lm);
            fb.colorMem = images.Alloc((uint32_t)dkImageLayoutGetSize(&layout), dkImageLayoutGetAlignment(&layout));
            dkImageInitialize(&fb.color, &layout, images.Mem(fb.colorMem), fb.colorMem.offset);
            fb.hasColor = true;
            t.image = fb.color;
            t.hasImage = true;
            t.width = w;
            t.height = h;
            t.sampler = 16 | (2 << 2) | 2; // linear, clamp
            WriteImageDescriptor(fb.texId, fb.color);
        }
        if (hasDepth && (sizeChanged || !fb.hasDepthImage)) {
            if (fb.hasDepthImage) {
                DeferFree({ fb.depthMem, 0 });
                fb.hasDepthImage = false;
            }
            CreateDepthImage(fb, w, h);
        }
        fb.width = w;
        fb.height = h;
        fb.invertY = invertY;
        fb.hasDepth = hasDepth;
        if (id == curFb && frameOpen) {
            BindFramebufferTargets(id);
        }
    }

    void BindFramebufferTargets(int id) {
        FramebufferDk& fb = fbs[id];
        if (!fb.hasColor) {
            return;
        }
        DkImageView colorView, depthView;
        dkImageViewDefaults(&colorView, &fb.color);
        const bool depth = fb.hasDepthImage && (id == 0 || fb.hasDepth);
        if (depth) {
            dkImageViewDefaults(&depthView, &fb.depth);
        }
        dkCmdBufBindRenderTarget(cmdbuf, &colorView, depth ? &depthView : nullptr);
        viewportDirty = scissorDirty = true;
    }

    void StartDrawToFramebuffer(int id, float noise) {
        SWITCH_PROF_DK(PROF_GL_OTHER);
        if (noise != 0.0f) {
            noiseScale = 1.0f / noise;
            if (fsParams.noiseScale != noiseScale) {
                fsParams.noiseScale = noiseScale;
                fsParamsDirty = true;
            }
        }
        if (id != curFb) {
            // Render-target writes must land before anything samples or copies them
            RequestBarrier();
            EmitPendingBarrier();
        }
        curFb = id;
        // Offscreen targets keep OpenGL's row order (y negated); framebuffer 0 is presented upright
        vsYScale = id == 0 ? 1.0f : -1.0f;
        if (frameOpen) {
            BindFramebufferTargets(id);
        }
    }

    // OpenGL rectangle (origin bottom-left) -> deko3d rectangle (origin top-left) for the current
    // target. Offscreen targets keep OpenGL rows, so only framebuffer 0 needs flipping.
    void ToTargetRect(int fbId, int& x, int& y, int w, int h) const {
        if (fbId == 0) {
            y = (int)fbs[0].height - y - h;
        }
        (void)x;
        (void)w;
    }

    void ClearFramebuffer(bool color, bool depth) {
        SWITCH_PROF_DK(PROF_GL_OTHER);
        const FramebufferDk& fb = fbs[curFb];
        DkScissor full = { 0, 0, fb.width, fb.height };
        dkCmdBufSetScissors(cmdbuf, 0, &full, 1);
        if (color) {
            dkCmdBufClearColorFloat(cmdbuf, 0, DkColorMask_RGBA, 0.0f, 0.0f, 0.0f, 1.0f);
        }
        if (depth && fb.hasDepthImage) {
            dkCmdBufClearDepthStencil(cmdbuf, true, 1.0f, 0, 0);
        }
        scissorDirty = true;
    }

    void ClearDepthRegion(int x, int y, int w, int h) {
        SWITCH_PROF_DK(PROF_GL_OTHER);
        const FramebufferDk& fb = fbs[curFb];
        if (!fb.hasDepthImage) {
            return;
        }
        ToTargetRect(curFb, x, y, w, h);
        DkScissor sc = ClampScissor(fb, x, y, w, h);
        dkCmdBufSetScissors(cmdbuf, 0, &sc, 1);
        dkCmdBufClearDepthStencil(cmdbuf, true, 1.0f, 0, 0);
        scissorDirty = true;
    }

    static DkScissor ClampScissor(const FramebufferDk& fb, int x, int y, int w, int h) {
        int x0 = std::clamp(x, 0, (int)fb.width), y0 = std::clamp(y, 0, (int)fb.height);
        int x1 = std::clamp(x + w, 0, (int)fb.width), y1 = std::clamp(y + h, 0, (int)fb.height);
        return { (uint32_t)x0, (uint32_t)y0, (uint32_t)std::max(0, x1 - x0), (uint32_t)std::max(0, y1 - y0) };
    }

    // OpenGL window y (bottom-up) -> image row, for a framebuffer
    int GlYToRow(int fbId, int y) const {
        return fbId == 0 ? (int)fbs[0].height - y : y;
    }

    void CopyFramebuffer(int dstId, int srcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) {
        SWITCH_PROF_DK(PROF_GL_OTHER);
        if (dstId >= (int)fbs.size() || srcId >= (int)fbs.size()) {
            return;
        }
        const FramebufferDk& src = fbs[srcId];
        const FramebufferDk& dst = fbs[dstId];
        if (!src.hasColor || !dst.hasColor) {
            return;
        }
        // Same coordinate adjustments as the OpenGL backend, which works in OpenGL window space
        if (!src.invertY) {
            int temp = srcY1 - srcY0;
            srcY1 = src.height - srcY0;
            srcY0 = srcY1 - temp;
        }
        if (src.invertY != dst.invertY) {
            std::swap(srcY0, srcY1);
        }
        // OpenGL window space -> image rows
        srcY0 = GlYToRow(srcId, srcY0);
        srcY1 = GlYToRow(srcId, srcY1);
        dstY0 = GlYToRow(dstId, dstY0);
        dstY1 = GlYToRow(dstId, dstY1);

        uint32_t flags = DkBlitFlag_FilterNearest;
        const bool srcFlipX = srcX0 > srcX1, srcFlipY = srcY0 > srcY1;
        const bool dstFlipX = dstX0 > dstX1, dstFlipY = dstY0 > dstY1;
        if (srcFlipX != dstFlipX) {
            flags |= DkBlitFlag_FlipX;
        }
        if (srcFlipY != dstFlipY) {
            flags |= DkBlitFlag_FlipY;
        }
        auto norm = [](int a, int b, int limit, uint32_t& start, uint32_t& len) {
            int lo = std::clamp(std::min(a, b), 0, limit), hi = std::clamp(std::max(a, b), 0, limit);
            start = lo;
            len = hi - lo;
        };
        DkImageRect sr = { 0, 0, 0, 0, 0, 1 }, dr = { 0, 0, 0, 0, 0, 1 };
        norm(srcX0, srcX1, src.width, sr.x, sr.width);
        norm(srcY0, srcY1, src.height, sr.y, sr.height);
        norm(dstX0, dstX1, dst.width, dr.x, dr.width);
        norm(dstY0, dstY1, dst.height, dr.y, dr.height);
        if (!sr.width || !sr.height || !dr.width || !dr.height) {
            return;
        }
        RequestBarrier();
        EmitPendingBarrier();
        DkImageView sv, dv;
        dkImageViewDefaults(&sv, &src.color);
        dkImageViewDefaults(&dv, &dst.color);
        dkCmdBufBlitImage(cmdbuf, &sv, &sr, &dv, &dr, flags, 0);
        RequestFullBarrier();
    }

    void ReadFramebufferToCPU(int id, uint32_t w, uint32_t h, uint16_t* out) {
        SWITCH_PROF_DK(PROF_GL_READBACK);
        if (id >= (int)fbs.size() || !fbs[id].hasColor) {
            return;
        }
        const FramebufferDk& fb = fbs[id];
        w = std::min(w, fb.width);
        h = std::min(h, fb.height);
        if (w * h * 4 > kReadbackSize) {
            return;
        }
        BeginReadback();
        DkImageView v;
        dkImageViewDefaults(&v, &fb.color);
        // OpenGL rows start at the bottom; offscreen images already use that order
        DkImageRect r = { 0, id == 0 ? fb.height - h : 0, 0, w, h, 1 };
        DkCopyBuf dst = { readback.gpu, 0, 0 };
        dkMemBlockFlushCpuCache(readback.block, 0, w * h * 4);
        dkCmdBufCopyImageToBuffer(cmdbuf, &v, &r, &dst, id == 0 ? DkBlitFlag_FlipY : 0);
        EndReadback();
        dkMemBlockFlushCpuCache(readback.block, 0, w * h * 4);
        const uint8_t* px = readback.cpu;
        for (uint32_t i = 0; i < w * h; i++) {
            uint8_t r8 = (px[i * 4 + 0] >> 3) & 0x1F;
            uint8_t g8 = (px[i * 4 + 1] >> 3) & 0x1F;
            uint8_t b8 = (px[i * 4 + 2] >> 3) & 0x1F;
            uint8_t a = px[i * 4 + 3] ? 1 : 0;
            out[i] = (r8 << 11) | (g8 << 6) | (b8 << 1) | a;
        }
    }

    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int id, const std::set<std::pair<float, float>>& coords) {
        SWITCH_PROF_DK(PROF_GL_READBACK);
        std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
        if (id >= (int)fbs.size() || coords.empty()) {
            return res;
        }
        const FramebufferDk& fb = fbs[id];
        if (!fb.hasDepthImage) {
            // Nothing to read: answer "far" for every point, never an empty map (callers index it)
            for (const auto& c : coords) {
                res.emplace(c, (uint16_t)0xFFFC);
            }
            return res;
        }
        BeginReadback();
        DkImageView v;
        dkImageViewDefaults(&v, &fb.depth);
        size_t i = 0;
        for (const auto& c : coords) {
            if ((i + 1) * 4 > kReadbackSize) {
                break;
            }
            int x = (int)c.first;
            int y = (int)c.second;
            int glY = fb.invertY ? (int)fb.height - y : y; // as the OpenGL backend reads it
            int row = id == 0 ? (int)fb.height - 1 - glY : glY;
            x = std::clamp(x, 0, (int)fb.width - 1);
            row = std::clamp(row, 0, (int)fb.height - 1);
            DkImageRect r = { (uint32_t)x, (uint32_t)row, 0, 1, 1, 1 };
            DkCopyBuf dst = { readback.gpu + i * 4, 0, 0 };
            dkCmdBufCopyImageToBuffer(cmdbuf, &v, &r, &dst, 0);
            i++;
        }
        EndReadback();
        dkMemBlockFlushCpuCache(readback.block, 0, (uint32_t)(i * 4));
        const uint32_t* vals = reinterpret_cast<const uint32_t*>(readback.cpu);
        i = 0;
        for (const auto& c : coords) {
            if ((i + 1) * 4 > kReadbackSize) {
                break;
            }
            // Z24S8 is stored as depth << 8 | stencil, like GL_UNSIGNED_INT_24_8
            res.emplace(c, (uint16_t)((vals[i] >> 18) << 2));
            i++;
        }
        return res;
    }

    // ------------------------------------------------------------------ shaders

    static std::optional<std::string> IncludeLoader(const std::string& path) {
        auto init = std::make_shared<Ship::ResourceInitData>();
        init->Type = (uint32_t)Ship::ResourceType::Shader;
        init->ByteOrder = Ship::Endianness::Native;
        init->Format = RESOURCE_FORMAT_BINARY;
        auto res = std::static_pointer_cast<Ship::Shader>(
            Ship::Context::GetRawInstance()->GetResourceManager()->LoadResource(path, true, init));
        if (res == nullptr) {
            return std::nullopt;
        }
        return *static_cast<std::string*>(res->GetRawPointer());
    }

    std::string LoadTemplate(const CCFeatures& cc) {
        std::string path = "shaders/deko3d/default.shader.glsl";
        if (const char* custom = gfx_get_shader(cc.shader_id)) {
            // Custom shaders are written for OpenGL; a deko3d variant may ship next to them
            std::string candidate = std::string(custom) + ".deko3d.glsl";
            if (IncludeLoader(candidate)) {
                path = candidate;
            }
        }
        auto src = IncludeLoader(path);
        if (!src) {
            // soh.o2r from before the deko3d renderer: use the copy built into the executable
            static const char* kBuiltin =
#include "fast/deko3d_default_shader.inc"
                ;
            static bool warned = false;
            if (!warned) {
                SPDLOG_WARN("deko3d: {} not in the archives, using the built-in template", path);
                warned = true;
            }
            return kBuiltin;
        }
        return *src;
    }

    std::string BuildShader(const CCFeatures& cc, bool vertex) {
        prism::Processor processor;
        prism::ContextItems ctx = {
            { "VERTEX_SHADER", vertex },
            { "o_c", M_ARRAY(cc.c, int, 2, 2, 4) },
            { "o_alpha", cc.opt_alpha },
            { "o_fog", cc.opt_fog },
            { "o_texture_edge", cc.opt_texture_edge },
            { "o_noise", cc.opt_noise },
            { "o_2cyc", cc.opt_2cyc },
            { "o_alpha_threshold", cc.opt_alpha_threshold },
            { "o_invisible", cc.opt_invisible },
            { "o_grayscale", cc.opt_grayscale },
            { "o_prim_depth", cc.opt_prim_depth },
            { "o_textures", M_ARRAY(cc.usedTextures, bool, 2) },
            { "o_masks", M_ARRAY(cc.used_masks, bool, 2) },
            { "o_blend", M_ARRAY(cc.used_blend, bool, 2) },
            { "o_clamp", M_ARRAY(cc.clamp, bool, 2, 2) },
            { "o_inputs", cc.numInputs },
            { "o_do_mix", M_ARRAY(cc.do_mix, bool, 2, 2) },
            { "o_do_single", M_ARRAY(cc.do_single, bool, 2, 2) },
            { "o_do_multiply", M_ARRAY(cc.do_multiply, bool, 2, 2) },
            { "o_color_alpha_same", M_ARRAY(cc.color_alpha_same, bool, 2) },
            { "FILTER_THREE_POINT", FILTER_THREE_POINT },
            { "FILTER_LINEAR", FILTER_LINEAR },
            { "FILTER_NONE", FILTER_NONE },
            { "srgb_mode", srgbMode },
            { "SHADER_0", SHADER_0 },
            { "SHADER_INPUT_1", SHADER_INPUT_1 },
            { "SHADER_INPUT_2", SHADER_INPUT_2 },
            { "SHADER_INPUT_3", SHADER_INPUT_3 },
            { "SHADER_INPUT_4", SHADER_INPUT_4 },
            { "SHADER_INPUT_5", SHADER_INPUT_5 },
            { "SHADER_INPUT_6", SHADER_INPUT_6 },
            { "SHADER_INPUT_7", SHADER_INPUT_7 },
            { "SHADER_TEXEL0", SHADER_TEXEL0 },
            { "SHADER_TEXEL0A", SHADER_TEXEL0A },
            { "SHADER_TEXEL1", SHADER_TEXEL1 },
            { "SHADER_TEXEL1A", SHADER_TEXEL1A },
            { "SHADER_1", SHADER_1 },
            { "SHADER_COMBINED", SHADER_COMBINED },
            { "SHADER_NOISE", SHADER_NOISE },
            { "o_three_point_filtering", filterMode == FILTER_THREE_POINT },
            { "append_formula", (InvokeFunc)append_formula },
            // Debug visualisation (0 = off, 1 = texcoord0 as colour, 2 = texture size/filter)
            { "dk_debug_mode", kDkDebugMode },
        };
        processor.populate(ctx);
        processor.load(LoadTemplate(cc));
        processor.bind_include_loader(IncludeLoader);
        return processor.process();
    }

    bool LoadDksh(const std::vector<uint8_t>& dksh, DkShader& out) {
        if (dksh.size() < sizeof(DkshHeader)) {
            return false;
        }
        DkshHeader hdr;
        memcpy(&hdr, dksh.data(), sizeof(hdr));
        if (hdr.magic != 0x48534B44 || hdr.control_sz + hdr.code_sz > dksh.size()) {
            SPDLOG_ERROR("deko3d: bad DKSH");
            return false;
        }
        uint32_t off = codeHeap.Alloc(hdr.code_sz, DK_SHADER_CODE_ALIGNMENT);
        if (off == UINT32_MAX || off + hdr.code_sz + DK_SHADER_CODE_UNUSABLE_SIZE > codeHeap.size) {
            SPDLOG_CRITICAL("deko3d: shader code heap full");
            return false;
        }
        memcpy(codeHeap.cpu + off, dksh.data() + hdr.control_sz, hdr.code_sz);
        DkShaderMaker sm;
        dkShaderMakerDefaults(&sm, codeHeap.block, off);
        sm.control = dksh.data();
        dkShaderInitialize(&out, &sm);
        shaderCodeDirty = true;
        return true;
    }

    ShaderProgramDk* CreateProgram(uint64_t id0, uint64_t id1) {
        SWITCH_PROF_DK(PROF_SHADER);
        CCFeatures cc;
        gfx_cc_get_features(id0, id1, &cc);
        DKTRACE("CreateProgram %016llx/%016llx", (unsigned long long)id0, (unsigned long long)id1);
        const std::string vsSrc = BuildShader(cc, true);
        const std::string fsSrc = BuildShader(cc, false);
        if (!dumpedTexturedShader && cc.usedTextures[0]) {
            dumpedTexturedShader = true;
            fprintf(stderr, "[dkdiag] textured VS:\n%s\n[dkdiag] textured FS:\n%s\n", vsSrc.c_str(), fsSrc.c_str());
        }

        ShaderProgramDk& prg = programs[std::make_pair(id0, id1)];
        prg.id0 = id0;
        prg.id1 = id1;
        prg.numInputs = cc.numInputs;
        prg.usedTextures[0] = cc.usedTextures[0];
        prg.usedTextures[1] = cc.usedTextures[1];
        prg.usedTextures[2] = cc.used_masks[0];
        prg.usedTextures[3] = cc.used_masks[1];
        prg.usedTextures[4] = cc.used_blend[0];
        prg.usedTextures[5] = cc.used_blend[1];

        // Vertex layout: same order as the interpreter writes it (and as the OpenGL backend reads)
        struct Attr {
            int location;
            int floats;
        };
        std::vector<Attr> attrs;
        attrs.push_back({ 0, 4 });
        for (int i = 0; i < 2; i++) {
            if (cc.usedTextures[i]) {
                attrs.push_back({ 1 + i * 3, 2 });
                for (int j = 0; j < 2; j++) {
                    if (cc.clamp[i][j]) {
                        attrs.push_back({ 2 + i * 3 + j, 1 });
                    }
                }
            }
        }
        if (cc.opt_fog) {
            attrs.push_back({ 7, 4 });
        }
        if (cc.opt_grayscale) {
            attrs.push_back({ 8, 4 });
        }
        for (int i = 0; i < cc.numInputs; i++) {
            attrs.push_back({ 9 + i, cc.opt_alpha ? 4 : 3 });
        }
        static const DkVtxAttribSize sizes[5] = { DkVtxAttribSize_1x32, DkVtxAttribSize_1x32, DkVtxAttribSize_2x32,
                                                  DkVtxAttribSize_3x32, DkVtxAttribSize_4x32 };
        int maxLoc = 0;
        uint32_t offset = 0;
        for (auto& a : attrs) {
            maxLoc = std::max(maxLoc, a.location);
        }
        prg.numAttribs = maxLoc + 1;
        for (uint32_t l = 0; l < prg.numAttribs; l++) {
            prg.attribs[l] = DkVtxAttribState{};
            prg.attribs[l].isFixed = 1;
            prg.attribs[l].size = DkVtxAttribSize_1x32;
            prg.attribs[l].type = DkVtxAttribType_Float;
        }
        for (auto& a : attrs) {
            DkVtxAttribState& s = prg.attribs[a.location];
            s.bufferId = 0;
            s.isFixed = 0;
            s.offset = offset;
            s.size = sizes[a.floats];
            s.type = DkVtxAttribType_Float;
            offset += a.floats * 4;
        }
        prg.stride = offset;
        prg.vsSrc = vsSrc;
        prg.fsSrc = fsSrc;

        const auto vsBin = compiler.Get(uam::Stage::Vertex, vsSrc);
        const auto fsBin = compiler.Get(uam::Stage::Fragment, fsSrc);
        DKTRACE("dksh vs %zu fs %zu", vsBin.size(), fsBin.size());
        prg.valid = !vsBin.empty() && !fsBin.empty() && LoadDksh(vsBin, prg.vs) && LoadDksh(fsBin, prg.fs);
        DKTRACE("program valid %d", prg.valid ? 1 : 0);
        if (!prg.valid) {
            SPDLOG_ERROR("deko3d: shader {:016x}/{:016x} unavailable; its draws are skipped", id0, id1);
        }
        return &prg;
    }

    // ------------------------------------------------------------------ draws

    void FlushState() {
        if (shaderCodeDirty) {
            dkCmdBufBarrier(cmdbuf, DkBarrier_None, DkInvalidateFlags_Shader);
            shaderCodeDirty = false;
        }
        EmitPendingBarrier();

        if (shadersDirty || boundProg != curProg) {
            DkShader const* shaders[2] = { &curProg->vs, &curProg->fs };
            dkCmdBufBindShaders(cmdbuf, DkStageFlag_Vertex | DkStageFlag_Fragment, shaders, 2);
            dkCmdBufBindVtxAttribState(cmdbuf, curProg->attribs, curProg->numAttribs);
            DkVtxBufferState vbs = { curProg->stride, 0 };
            dkCmdBufBindVtxBufferState(cmdbuf, &vbs, 1);
            boundProg = curProg;
            shadersDirty = false;
            texturesDirty = true;
        }

        if (!vsUboBound) {
            dkCmdBufBindUniformBuffer(cmdbuf, DkStage_Vertex, 0, vsUbo, DK_UNIFORM_BUF_ALIGNMENT);
            vsUboBound = true;
            boundVsYScale = 0.0f;
        }
        if (boundVsYScale != vsYScale) {
            float v[4] = { 1.0f, vsYScale, 1.0f, 1.0f };
            dkCmdBufPushConstants(cmdbuf, vsUbo, DK_UNIFORM_BUF_ALIGNMENT, 0, sizeof(v), v);
            boundVsYScale = vsYScale;
        }

        // Textures and their per-draw uniforms
        FsParams p = fsParams;
        p.primDepth = primDepth;
        DkResHandle handles[SHADER_MAX_TEXTURES];
        for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
            uint32_t id = curTexIds[i];
            if (id >= textures.size() || !textures[id].hasImage) {
                id = 0;
            }
            const TextureDk& t = textures[id];
            handles[i] = dkMakeTextureHandle(id, t.sampler);
            if (curProg->usedTextures[i] && id) {
                textures[id].lastUseSerial = barrierEpoch;
            }
        }
        const TextureDk& t0 = textures[curTexIds[0] < textures.size() ? curTexIds[0] : 0];
        const TextureDk& t1 = textures[curTexIds[1] < textures.size() ? curTexIds[1] : 0];
        p.texSize[0] = t0.width;
        p.texSize[1] = t0.height;
        p.texSize[2] = t1.width;
        p.texSize[3] = t1.height;
        p.texFiltering[0] = t0.filtering;
        p.texFiltering[1] = t1.filtering;
        if (texturesDirty || memcmp(handles, boundHandles, sizeof(handles)) != 0) {
            dkCmdBufBindTextures(cmdbuf, DkStage_Fragment, 0, handles, SHADER_MAX_TEXTURES);
            memcpy(boundHandles, handles, sizeof(handles));
            texturesDirty = false;
        }
        if (fsParamsDirty || memcmp(&p, &fsParams, sizeof(p)) != 0) {
            fsParams = p;
            dkCmdBufPushConstants(cmdbuf, fsUbo, DK_UNIFORM_BUF_ALIGNMENT, 0, sizeof(fsParams), &fsParams);
            fsParamsDirty = false;
        }

        const FramebufferDk& fb = fbs[curFb];
        if (viewportDirty) {
            int x = vpX, y = vpY;
            ToTargetRect(curFb, x, y, vpW, vpH);
            DkViewport vp = { (float)x, (float)y, (float)vpW, (float)vpH, 0.0f, 1.0f };
            dkCmdBufSetViewports(cmdbuf, 0, &vp, 1);
            viewportDirty = false;
        }
        if (scissorDirty) {
            int x = scX, y = scY;
            ToTargetRect(curFb, x, y, scW, scH);
            DkScissor sc = ClampScissor(fb, x, y, scW, scH);
            dkCmdBufSetScissors(cmdbuf, 0, &sc, 1);
            scissorDirty = false;
        }

        const int8_t depthState = (int8_t)((depthTest ? 1 : 0) | (depthMask ? 2 : 0) | (decal ? 4 : 0));
        if (depthState != boundDepthState) {
            DkDepthStencilState ds;
            dkDepthStencilStateDefaults(&ds);
            if (depthTest || depthMask) {
                ds.depthTestEnable = 1;
                ds.depthWriteEnable = depthMask ? 1 : 0;
                ds.depthCompareOp =
                    depthTest ? (decal ? DkCompareOp_Lequal : DkCompareOp_Less) : DkCompareOp_Always;
            } else {
                ds.depthTestEnable = 0;
                ds.depthWriteEnable = 0;
            }
            dkCmdBufBindDepthStencilState(cmdbuf, &ds);
            boundDepthState = depthState;
        }
        if ((int8_t)decal != boundDecal) {
            DkRasterizerState rs;
            dkRasterizerStateDefaults(&rs);
            rs.depthClampEnable = 1;
            rs.cullMode = DkFace_None;
            if (decal) {
                float ssdb = -2.0f;
                switch (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(CVAR_Z_FIGHTING_MODE, 0)) {
                    case 1:
                        ssdb = -1.0f * (float)fb.height / 120.0f;
                        break;
                    case 2:
                        ssdb = -1.0f * (float)fb.height / 100.0f;
                        break;
                    default:
                        break;
                }
                rs.depthBiasEnableMask = DkPolygonFlag_Fill;
                // glPolygonOffset(ssdb, -2); Mesa's nvc0 doubles the units for the hardware
                dkCmdBufSetDepthBias(cmdbuf, -2.0f * 2.0f, 0.0f, ssdb);
            }
            dkCmdBufBindRasterizerState(cmdbuf, &rs);
            boundDecal = decal;
        }
        if ((int8_t)useAlpha != boundBlend) {
            DkColorState cs;
            dkColorStateDefaults(&cs);
            dkColorStateSetBlendEnable(&cs, 0, useAlpha);
            dkCmdBufBindColorState(cmdbuf, &cs);
            boundBlend = useAlpha;
        }
    }

    void DrawTriangles(float* vbo, size_t lenFloats, size_t numTris) {
        SWITCH_PROF_DK(PROF_GL_DRAW);
        if (!frameOpen || !curProg || !curProg->valid || numTris == 0) {
            return;
        }
        uint8_t* cpu;
        DkGpuAddr gpu;
        const uint32_t bytes = (uint32_t)(lenFloats * sizeof(float));
        if (UploadAlloc(bytes, 16, &cpu, &gpu) == UINT32_MAX) {
            return;
        }
        memcpy(cpu, vbo, bytes);
        if (!curProg->layoutChecked) {
            curProg->layoutChecked = true;
            const uint32_t vboFloats = (uint32_t)(lenFloats / (numTris * 3));
            const uint32_t progFloats = curProg->stride / 4;
            static int reported = 0;
            if (vboFloats != progFloats && reported < 6) {
                reported++;
                std::string attrs;
                for (uint32_t l = 0; l < curProg->numAttribs; l++) {
                    const auto& a = curProg->attribs[l];
                    char b[64];
                    snprintf(b, sizeof(b), " L%u:%s@%u/0x%x", l, a.isFixed ? "fixed" : "buf", (unsigned)a.offset,
                             (unsigned)a.size);
                    attrs += b;
                }
                fprintf(stderr,
                        "[dkdiag] VBO MISMATCH prog %016llx/%016llx: vbo %u floats/vertex, program %u |%s\n"
                        "[dkdiag] VS:\n%s\n[dkdiag] FS:\n%s\n",
                        (unsigned long long)curProg->id0, (unsigned long long)curProg->id1, vboFloats, progFloats,
                        attrs.c_str(), curProg->vsSrc.c_str(), curProg->fsSrc.c_str());
            } else if (reported < 6) {
                static int okReported = 0;
                if (okReported++ < 8) {
                    fprintf(stderr, "[dkdiag] layout ok prog %016llx/%016llx: %u floats/vertex\n",
                            (unsigned long long)curProg->id0, (unsigned long long)curProg->id1, vboFloats);
                }
            }
        }
        FlushState();
        dkCmdBufBindVtxBuffer(cmdbuf, 0, gpu, bytes);
        dkCmdBufDraw(cmdbuf, DkPrimitive_Triangles, (uint32_t)numTris * 3, 1, 0, 0);
    }

    // ------------------------------------------------------------------ ImGui

    void ImGuiSetup() {
        DKTRACE("ImGuiSetup");
        auto vsBin = compiler.Get(uam::Stage::Vertex, kImGuiVs);
        auto fsBin = compiler.Get(uam::Stage::Fragment, kImGuiFs);
        imguiProg.valid =
            !vsBin.empty() && !fsBin.empty() && LoadDksh(vsBin, imguiProg.vs) && LoadDksh(fsBin, imguiProg.fs);
        imguiProg.numAttribs = 3;
        imguiProg.attribs[0] = DkVtxAttribState{ 0, 0, offsetof(ImDrawVert, pos), DkVtxAttribSize_2x32,
                                                 DkVtxAttribType_Float, 0 };
        imguiProg.attribs[1] =
            DkVtxAttribState{ 0, 0, offsetof(ImDrawVert, uv), DkVtxAttribSize_2x32, DkVtxAttribType_Float, 0 };
        imguiProg.attribs[2] =
            DkVtxAttribState{ 0, 0, offsetof(ImDrawVert, col), DkVtxAttribSize_4x8, DkVtxAttribType_Unorm, 0 };
        imguiProg.stride = sizeof(ImDrawVert);

        // Font atlas
        ImGuiIO& io = ImGui::GetIO();
        unsigned char* pixels;
        int w, h;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
        imguiFontTex = AllocTexId();
        uint32_t savedTile = curTile, savedTex = curTexIds[0];
        curTile = 0;
        curTexIds[0] = imguiFontTex;
        UploadTexture(pixels, w, h);
        curTile = savedTile;
        curTexIds[0] = savedTex;
        io.Fonts->SetTexID(reinterpret_cast<ImTextureID>((uintptr_t)imguiFontTex));
        imguiReady = imguiProg.valid;
        if (!imguiReady) {
            SPDLOG_CRITICAL("deko3d: ImGui shaders failed to compile");
        }
    }

    void ImGuiRender(ImDrawData* dd) {
        SWITCH_PROF_DK(PROF_IMGUI);
        if (!frameOpen) {
            return;
        }
        if (!imguiReady) {
            ImGuiSetup();
            if (!imguiReady) {
                return;
            }
        } else if (ImGui::GetIO().Fonts->TexID == (ImTextureID)0) {
            // The font atlas was rebuilt (e.g. a font size change): upload it again
            ImGuiIO& io = ImGui::GetIO();
            unsigned char* pixels;
            int w, h;
            io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
            uint32_t savedTile = curTile, savedTex = curTexIds[0];
            curTile = 0;
            curTexIds[0] = imguiFontTex;
            UploadTexture(pixels, w, h);
            curTile = savedTile;
            curTexIds[0] = savedTex;
            io.Fonts->SetTexID(reinterpret_cast<ImTextureID>((uintptr_t)imguiFontTex));
        }
        const int fbW = (int)(dd->DisplaySize.x * dd->FramebufferScale.x);
        const int fbH = (int)(dd->DisplaySize.y * dd->FramebufferScale.y);
        if (fbW <= 0 || fbH <= 0 || dd->CmdListsCount == 0) {
            return;
        }
        if (curFb != 0) {
            StartDrawToFramebuffer(0, 0.0f);
        }
        EmitPendingBarrier();
        if (shaderCodeDirty) {
            dkCmdBufBarrier(cmdbuf, DkBarrier_None, DkInvalidateFlags_Shader);
            shaderCodeDirty = false;
        }

        // Upload all vertices and indices
        const uint32_t vtxBytes = dd->TotalVtxCount * sizeof(ImDrawVert);
        const uint32_t idxBytes = dd->TotalIdxCount * sizeof(ImDrawIdx);
        uint8_t *vcpu, *icpu;
        DkGpuAddr vgpu, igpu;
        if (UploadAlloc(vtxBytes, 16, &vcpu, &vgpu) == UINT32_MAX ||
            UploadAlloc(idxBytes, 16, &icpu, &igpu) == UINT32_MAX) {
            return;
        }
        {
            uint8_t* v = vcpu;
            uint8_t* i = icpu;
            for (int n = 0; n < dd->CmdListsCount; n++) {
                const ImDrawList* cl = dd->CmdLists[n];
                memcpy(v, cl->VtxBuffer.Data, cl->VtxBuffer.Size * sizeof(ImDrawVert));
                memcpy(i, cl->IdxBuffer.Data, cl->IdxBuffer.Size * sizeof(ImDrawIdx));
                v += cl->VtxBuffer.Size * sizeof(ImDrawVert);
                i += cl->IdxBuffer.Size * sizeof(ImDrawIdx);
            }
        }

        // ImGui state
        uint8_t* ucpu;
        DkGpuAddr ugpu;
        if (UploadAlloc(DK_UNIFORM_BUF_ALIGNMENT, DK_UNIFORM_BUF_ALIGNMENT, &ucpu, &ugpu) == UINT32_MAX) {
            return;
        }
        // ImGui pixels (y down) -> clip space (y up on deko3d)
        const float L = dd->DisplayPos.x, T = dd->DisplayPos.y;
        const float R = L + dd->DisplaySize.x, B = T + dd->DisplaySize.y;
        float st[4] = { 2.0f / (R - L), -2.0f / (B - T), -(R + L) / (R - L), (B + T) / (B - T) };
        memcpy(ucpu, st, sizeof(st));
        dkCmdBufBindUniformBuffer(cmdbuf, DkStage_Vertex, 0, ugpu, DK_UNIFORM_BUF_ALIGNMENT);
        vsUboBound = false;

        DkShader const* shaders[2] = { &imguiProg.vs, &imguiProg.fs };
        dkCmdBufBindShaders(cmdbuf, DkStageFlag_Vertex | DkStageFlag_Fragment, shaders, 2);
        dkCmdBufBindVtxAttribState(cmdbuf, imguiProg.attribs, imguiProg.numAttribs);
        DkVtxBufferState vbs = { imguiProg.stride, 0 };
        dkCmdBufBindVtxBufferState(cmdbuf, &vbs, 1);
        dkCmdBufBindVtxBuffer(cmdbuf, 0, vgpu, vtxBytes);
        dkCmdBufBindIdxBuffer(cmdbuf, sizeof(ImDrawIdx) == 2 ? DkIdxFormat_Uint16 : DkIdxFormat_Uint32, igpu);

        DkRasterizerState rs;
        dkRasterizerStateDefaults(&rs);
        rs.cullMode = DkFace_None;
        dkCmdBufBindRasterizerState(cmdbuf, &rs);
        DkDepthStencilState ds;
        dkDepthStencilStateDefaults(&ds);
        ds.depthTestEnable = 0;
        ds.depthWriteEnable = 0;
        dkCmdBufBindDepthStencilState(cmdbuf, &ds);
        DkColorState cs;
        dkColorStateDefaults(&cs);
        dkColorStateSetBlendEnable(&cs, 0, true);
        dkCmdBufBindColorState(cmdbuf, &cs);
        DkViewport vp = { 0.0f, 0.0f, (float)fbs[0].width, (float)fbs[0].height, 0.0f, 1.0f };
        dkCmdBufSetViewports(cmdbuf, 0, &vp, 1);

        const ImVec2 clipOff = dd->DisplayPos;
        const ImVec2 clipScale = dd->FramebufferScale;
        uint32_t vtxBase = 0, idxBase = 0;
        DkResHandle lastHandle = 0xFFFFFFFF;
        for (int n = 0; n < dd->CmdListsCount; n++) {
            const ImDrawList* cl = dd->CmdLists[n];
            for (int c = 0; c < cl->CmdBuffer.Size; c++) {
                const ImDrawCmd* cmd = &cl->CmdBuffer[c];
                if (cmd->UserCallback != nullptr) {
                    if (cmd->UserCallback != ImDrawCallback_ResetRenderState) {
                        cmd->UserCallback(cl, cmd);
                    }
                    continue;
                }
                ImVec2 cmin((cmd->ClipRect.x - clipOff.x) * clipScale.x, (cmd->ClipRect.y - clipOff.y) * clipScale.y);
                ImVec2 cmax((cmd->ClipRect.z - clipOff.x) * clipScale.x, (cmd->ClipRect.w - clipOff.y) * clipScale.y);
                cmin.x = std::max(cmin.x, 0.0f);
                cmin.y = std::max(cmin.y, 0.0f);
                cmax.x = std::min(cmax.x, (float)fbs[0].width);
                cmax.y = std::min(cmax.y, (float)fbs[0].height);
                if (cmax.x <= cmin.x || cmax.y <= cmin.y) {
                    continue;
                }
                DkScissor sc = { (uint32_t)cmin.x, (uint32_t)cmin.y, (uint32_t)(cmax.x - cmin.x),
                                 (uint32_t)(cmax.y - cmin.y) };
                dkCmdBufSetScissors(cmdbuf, 0, &sc, 1);

                uint32_t texId = (uint32_t)(uintptr_t)cmd->GetTexID();
                if (texId >= textures.size() || !textures[texId].hasImage) {
                    texId = imguiFontTex;
                }
                const DkResHandle h = dkMakeTextureHandle(texId, kImGuiSampler);
                if (h != lastHandle) {
                    dkCmdBufBindTexture(cmdbuf, DkStage_Fragment, 0, h);
                    lastHandle = h;
                }
                textures[texId].lastUseSerial = barrierEpoch;
                dkCmdBufDrawIndexed(cmdbuf, DkPrimitive_Triangles, cmd->ElemCount, 1, idxBase + cmd->IdxOffset,
                                    (int32_t)(vtxBase + cmd->VtxOffset), 0);
            }
            vtxBase += cl->VtxBuffer.Size;
            idxBase += cl->IdxBuffer.Size;
        }
        // Fast3D draws after this rebind everything
        MarkAllDirty();
    }
};

// ---------------------------------------------------------------------------------------------
// 6. Entry points
// ---------------------------------------------------------------------------------------------

GfxRenderingAPIDeko3d::GfxRenderingAPIDeko3d() : mImpl(std::make_unique<Impl>()) {
}

GfxRenderingAPIDeko3d::~GfxRenderingAPIDeko3d() = default;

const char* GfxRenderingAPIDeko3d::GetName() {
    return "deko3d";
}

int GfxRenderingAPIDeko3d::GetMaxTextureSize() {
    return 16384;
}

GfxClipParameters GfxRenderingAPIDeko3d::GetClipParameters() {
    return { false, mImpl->fbs[mImpl->curFb].invertY };
}

void GfxRenderingAPIDeko3d::UnloadShader(ShaderProgram* oldPrg) {
}

void GfxRenderingAPIDeko3d::LoadShader(ShaderProgram* newPrg) {
    SWITCH_PROF_DK(PROF_GL_OTHER);
    mImpl->curProg = reinterpret_cast<ShaderProgramDk*>(newPrg);
}

void GfxRenderingAPIDeko3d::ClearShaderCache() {
    // Programs stay valid (their code is in the code heap); the interpreter re-requests shaders
    // after this, e.g. when the filtering mode changes, and gets the same objects back.
}

ShaderProgram* GfxRenderingAPIDeko3d::CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto* prg = mImpl->CreateProgram(shaderId0, shaderId1);
    mImpl->curProg = prg;
    return reinterpret_cast<ShaderProgram*>(prg);
}

ShaderProgram* GfxRenderingAPIDeko3d::LookupShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto it = mImpl->programs.find(std::make_pair(shaderId0, shaderId1));
    return it == mImpl->programs.end() ? nullptr : reinterpret_cast<ShaderProgram*>(&it->second);
}

void GfxRenderingAPIDeko3d::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    auto* p = reinterpret_cast<ShaderProgramDk*>(prg);
    *numInputs = p->numInputs;
    usedTextures[0] = p->usedTextures[0];
    usedTextures[1] = p->usedTextures[1];
}

uint32_t GfxRenderingAPIDeko3d::NewTexture() {
    SWITCH_PROF_DK(PROF_GL_OTHER);
    return mImpl->AllocTexId();
}

void GfxRenderingAPIDeko3d::SelectTexture(int tile, uint32_t textureId) {
    mImpl->curTexIds[tile] = textureId;
    mImpl->curTile = tile;
}

void GfxRenderingAPIDeko3d::UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) {
    mImpl->UploadTexture(rgba32Buf, width, height);
}

void GfxRenderingAPIDeko3d::SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) {
    mImpl->SetSamplerParameters(sampler, linear_filter, cms, cmt);
}

void GfxRenderingAPIDeko3d::SetDepthTestAndMask(bool depth_test, bool z_upd) {
    mImpl->depthTest = depth_test;
    mImpl->depthMask = z_upd;
}

void GfxRenderingAPIDeko3d::SetZmodeDecal(bool decal) {
    mImpl->decal = decal;
}

void GfxRenderingAPIDeko3d::SetViewport(int x, int y, int width, int height) {
    auto& i = *mImpl;
    if (i.vpX != x || i.vpY != y || i.vpW != width || i.vpH != height) {
        i.vpX = x;
        i.vpY = y;
        i.vpW = width;
        i.vpH = height;
        i.viewportDirty = true;
    }
}

void GfxRenderingAPIDeko3d::SetScissor(int x, int y, int width, int height) {
    auto& i = *mImpl;
    if (i.scX != x || i.scY != y || i.scW != width || i.scH != height) {
        i.scX = x;
        i.scY = y;
        i.scW = width;
        i.scH = height;
        i.scissorDirty = true;
    }
}

void GfxRenderingAPIDeko3d::SetUseAlpha(bool useAlpha) {
    mImpl->useAlpha = useAlpha;
}

void GfxRenderingAPIDeko3d::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    mImpl->DrawTriangles(buf_vbo, buf_vbo_len, buf_vbo_num_tris);
}

void GfxRenderingAPIDeko3d::Init() {
    mImpl->Init();
}

void GfxRenderingAPIDeko3d::OnResize() {
}

void GfxRenderingAPIDeko3d::StartFrame() {
    SWITCH_PROF_DK(PROF_GL_OTHER);
    mImpl->StartFrame();
}

void GfxRenderingAPIDeko3d::EndFrame() {
    SWITCH_PROF_DK(PROF_GL_OTHER);
    mImpl->EndFrame();
}

void GfxRenderingAPIDeko3d::FinishRender() {
}

int GfxRenderingAPIDeko3d::CreateFramebuffer() {
    return mImpl->CreateFramebuffer();
}

void GfxRenderingAPIDeko3d::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height,
                                                        uint32_t msaa_level, bool opengl_invertY, bool render_target,
                                                        bool has_depth_buffer, bool can_extract_depth) {
    // MSAA is not implemented: framebuffers are single-sampled and "resolves" are copies
    mImpl->UpdateFramebufferParameters(fb_id, width, height, opengl_invertY, has_depth_buffer);
}

void GfxRenderingAPIDeko3d::StartDrawToFramebuffer(int fbId, float noiseScale) {
    mImpl->StartDrawToFramebuffer(fbId, noiseScale);
}

void GfxRenderingAPIDeko3d::CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1,
                                            int dstX0, int dstY0, int dstX1, int dstY1) {
    mImpl->CopyFramebuffer(fbDstId, fbSrcId, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1);
}

void GfxRenderingAPIDeko3d::ClearFramebuffer(bool color, bool depth) {
    mImpl->ClearFramebuffer(color, depth);
}

void GfxRenderingAPIDeko3d::ClearDepthRegion(int x, int y, int w, int h) {
    mImpl->ClearDepthRegion(x, y, w, h);
}

void GfxRenderingAPIDeko3d::ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) {
    mImpl->ReadFramebufferToCPU(fbId, width, height, rgba16Buf);
}

void GfxRenderingAPIDeko3d::ResolveMSAAColorBuffer(int fbIdTarget, int fbIdSrc) {
    auto& src = mImpl->fbs[fbIdSrc];
    auto& dst = mImpl->fbs[fbIdTarget];
    mImpl->CopyFramebuffer(fbIdTarget, fbIdSrc, 0, 0, src.width, src.height, 0, 0, dst.width, dst.height);
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIDeko3d::GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) {
    return mImpl->GetPixelDepth(fb_id, coordinates);
}

void* GfxRenderingAPIDeko3d::GetFramebufferTextureId(int fbId) {
    return (void*)(uintptr_t)mImpl->fbs[fbId].texId;
}

void GfxRenderingAPIDeko3d::SelectTextureFb(int fbId) {
    SelectTexture(0, mImpl->fbs[fbId].texId);
}

void GfxRenderingAPIDeko3d::DeleteTexture(uint32_t texId) {
    auto& i = *mImpl;
    if (texId == 0 || texId >= i.textures.size() || !i.textures[texId].allocated || i.textures[texId].isFramebuffer) {
        return;
    }
    TextureDk& t = i.textures[texId];
    DeferredFree d;
    if (t.hasImage) {
        d.mem = t.mem;
    }
    d.texId = texId;
    i.DeferFree(d);
    t = TextureDk{};
}

void GfxRenderingAPIDeko3d::SetTextureFilter(FilteringMode mode) {
    gfx_texture_cache_clear();
    mImpl->filterMode = mode;
}

FilteringMode GfxRenderingAPIDeko3d::GetTextureFilter() {
    return mImpl->filterMode;
}

void GfxRenderingAPIDeko3d::SetSrgbMode() {
    mImpl->srgbMode = true;
}

ImTextureID GfxRenderingAPIDeko3d::GetTextureById(int id) {
    return reinterpret_cast<ImTextureID>((uintptr_t)id);
}

void GfxRenderingAPIDeko3d::SetCurrentPrimDepth(float depth) {
    mImpl->primDepth = depth;
}

void GfxRenderingAPIDeko3d::ImGuiInit() {
    // Resources are created on the first ImGui draw: the window and GUI start before Init()
}

void GfxRenderingAPIDeko3d::ImGuiShutdown() {
}

void GfxRenderingAPIDeko3d::ImGuiNewFrame() {
    // ImGui::NewFrame needs a built font atlas; the texture itself is uploaded on the first draw
    ImGuiIO& io = ImGui::GetIO();
    if (!io.Fonts->IsBuilt()) {
        io.Fonts->Build();
    }
}

void GfxRenderingAPIDeko3d::ImGuiRenderDrawData(ImDrawData* drawData) {
    mImpl->ImGuiRender(drawData);
}

void GfxRenderingAPIDeko3d::SetVsync(bool enabled) {
    mImpl->vsync = enabled;
}

} // namespace Fast
#endif
