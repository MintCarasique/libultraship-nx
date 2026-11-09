#pragma once
#ifdef __SWITCH__

// Fast3D rendering backend on deko3d (devkitPro's low-level API for the Switch GPU).
//
// Replaces OpenGL on Mesa/nouveau: there is no driver between the interpreter and the GPU, so
// each draw costs only the commands written here. Shaders are the same prism templates as the
// OpenGL backend (shaders/deko3d/default.shader.glsl), compiled at run time by uam and cached on
// the SD card as DKSH.
//
// Coordinate conventions (see gfx_deko3d.cpp): offscreen framebuffers keep OpenGL's row order in
// memory (the vertex shader negates clip-space y), so the interpreter's viewports, scissors,
// copies and read-backs work unchanged. Framebuffer 0 is the swapchain image and is drawn upright
// for the display; its rectangles are converted instead.

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <deko3d.h>

#include "gfx_rendering_api.h"
#include "../interpreter.h"

struct ImDrawData;

namespace Fast {

class GfxRenderingAPIDeko3d final : public GfxRenderingAPI {
  public:
    GfxRenderingAPIDeko3d();
    ~GfxRenderingAPIDeko3d() override;

    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    void ClearShaderCache() override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depth_test, bool z_upd) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                     bool can_extract_depth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ClearDepthRegion(int x, int y, int w, int h) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarget, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;
    void SetCurrentPrimDepth(float depth) override;

    // ImGui renderer (called by Fast3dGui for FAST3D_SDL_DEKO3D)
    void ImGuiInit();
    void ImGuiShutdown();
    void ImGuiNewFrame();
    void ImGuiRenderDrawData(ImDrawData* drawData);

    // Swap interval for the next present (0 = no vsync)
    void SetVsync(bool enabled);

    struct Impl;

  private:
    std::unique_ptr<Impl> mImpl;
};

} // namespace Fast
#endif
