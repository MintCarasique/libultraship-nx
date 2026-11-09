@prism(type='fragment', name='Fast3D Fragment Shader (deko3d)', version='1.0.0', description='GLSL 4.60 variant of the Fast3D shader for uam/deko3d', author='Emill & Prism Team')

#version 460

// uam compiles each stage on its own (no linking), so every interface variable, uniform block and
// sampler needs an explicit location/binding. Fixed slots, shared with gfx_deko3d.cpp:
//   vertex attributes: 0 pos, 1/2/3 texcoord0 + clamp s/t, 4/5/6 texcoord1 + clamp s/t,
//                      7 fog, 8 grayscale, 9.. inputs
//   varyings:          attribute location - 1
//   samplers:          0/1 tex, 2/3 mask, 4/5 blend
// Note: prism's expression parser gives '+' higher precedence than '*' ('i * 3 + 1' is
// i * 4), so the location arithmetic below uses additions only.
@if(VERTEX_SHADER)
    layout(location = 0) in vec4 aVtxPos;

    layout(std140, binding = 0) uniform VsParams {
        // x: unused, y: clip-space y multiplier (-1 flips offscreen targets to OpenGL's row order)
        vec4 uPosScale;
    };

    @for(i in 0..2)
        @if(o_textures[i])
            layout(location = @{i + i + i + 1}) in vec2 aTexCoord@{i};
            layout(location = @{i + i + i}) out vec2 vTexCoord@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        layout(location = @{i + i + i + 2}) in float aTexClampS@{i};
                        layout(location = @{i + i + i + 1}) out float vTexClampS@{i};
                    @else
                        layout(location = @{i + i + i + 3}) in float aTexClampT@{i};
                        layout(location = @{i + i + i + 2}) out float vTexClampT@{i};
                    @end
                @end
            @end
        @end
    @end

    @if(o_fog)
        layout(location = 7) in vec4 aFog;
        layout(location = 6) out vec4 vFog;
    @end

    @if(o_grayscale)
        layout(location = 8) in vec4 aGrayscaleColor;
        layout(location = 7) out vec4 vGrayscaleColor;
    @end

    @for(i in 0..o_inputs)
        @if(o_alpha)
            layout(location = @{9 + i}) in vec4 aInput@{i + 1};
            layout(location = @{8 + i}) out vec4 vInput@{i + 1};
        @else
            layout(location = @{9 + i}) in vec3 aInput@{i + 1};
            layout(location = @{8 + i}) out vec3 vInput@{i + 1};
        @end
    @end

    void main() {
        @for(i in 0..2)
            @if(o_textures[i])
                vTexCoord@{i} = aTexCoord@{i};
                @for(j in 0..2)
                    @if(o_clamp[i][j])
                        @if(j == 0)
                            vTexClampS@{i} = aTexClampS@{i};
                        @else
                            vTexClampT@{i} = aTexClampT@{i};
                        @end
                    @end
                @end
            @end
        @end
        @if(o_fog)
            vFog = aFog;
        @end
        @if(o_grayscale)
            vGrayscaleColor = aGrayscaleColor;
        @end
        @for(i in 0..o_inputs)
            vInput@{i + 1} = aInput@{i + 1};
        @end
        gl_Position = vec4(aVtxPos.x, aVtxPos.y * uPosScale.y, aVtxPos.z, aVtxPos.w);
    }
@else
    layout(location = 0) out vec4 vOutColor;

    @for(i in 0..2)
        @if(o_textures[i])
            layout(location = @{i + i + i}) in vec2 vTexCoord@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        layout(location = @{i + i + i + 1}) in float vTexClampS@{i};
                    @else
                        layout(location = @{i + i + i + 2}) in float vTexClampT@{i};
                    @end
                @end
            @end
        @end
    @end

    @if(o_fog) layout(location = 6) in vec4 vFog;
    @if(o_grayscale) layout(location = 7) in vec4 vGrayscaleColor;

    @for(i in 0..o_inputs)
        @if(o_alpha)
            layout(location = @{8 + i}) in vec4 vInput@{i + 1};
        @else
            layout(location = @{8 + i}) in vec3 vInput@{i + 1};
        @end
    @end

    @if(o_textures[0]) layout(binding = 0) uniform sampler2D uTex0;
    @if(o_textures[1]) layout(binding = 1) uniform sampler2D uTex1;

    @if(o_masks[0]) layout(binding = 2) uniform sampler2D uTexMask0;
    @if(o_masks[1]) layout(binding = 3) uniform sampler2D uTexMask1;

    @if(o_blend[0]) layout(binding = 4) uniform sampler2D uTexBlend0;
    @if(o_blend[1]) layout(binding = 5) uniform sampler2D uTexBlend1;

    layout(std140, binding = 0) uniform FsParams {
        int frame_count;
        float noise_scale;
        float prim_depth;
        float fs_pad0;
        ivec4 tex_size;      // w0, h0, w1, h1
        ivec4 tex_filtering; // filtering mode of tile 0, tile 1
    };

    #define TEX_OFFSET(off) texture(tex, texCoord - off / texSize)
    #define WRAP(x, low, high) mod((x)-(low), (high)-(low)) + (low)

    float random(in vec3 value) {
        float random = dot(sin(value), vec3(12.9898, 78.233, 37.719));
        return fract(sin(random) * 143758.5453);
    }

    vec4 fromLinear(vec4 linearRGB){
        bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
        vec3 higher = vec3(1.055)*pow(linearRGB.rgb, vec3(1.0/2.4)) - vec3(0.055);
        vec3 lower = linearRGB.rgb * vec3(12.92);
        return vec4(mix(higher, lower, cutoff), linearRGB.a);
    }

    vec4 filter3point(in sampler2D tex, in vec2 texCoord, in vec2 texSize) {
        vec2 offset = fract(texCoord*texSize - vec2(0.5));
        offset -= step(1.0, offset.x + offset.y);
        vec4 c0 = TEX_OFFSET(offset);
        vec4 c1 = TEX_OFFSET(vec2(offset.x - sign(offset.x), offset.y));
        vec4 c2 = TEX_OFFSET(vec2(offset.x, offset.y - sign(offset.y)));
        return c0 + abs(offset.x)*(c1-c0) + abs(offset.y)*(c2-c0);
    }

    // The filtering mode is passed by value: indexing a UBO vector with a function argument is a
    // dynamic component index, which uam's front end handles poorly.
    vec4 hookTexture2D(in int filtering, sampler2D tex, in vec2 uv, in vec2 texSize) {
    @if(o_three_point_filtering)
        if(filtering == @{FILTER_THREE_POINT}) {
            return filter3point(tex, uv, texSize);
        }
    @end
        return texture(tex, uv);
    }

    void main() {
        @for(i in 0..2)
            @if(o_textures[i])
                @{s = o_clamp[i][0]}
                @{t = o_clamp[i][1]}

                @if(i == 0)
                    vec2 texSize0 = vec2(textureSize(uTex0, 0));
                    int texFilter0 = tex_filtering.x;
                @else
                    vec2 texSize1 = vec2(textureSize(uTex1, 0));
                    int texFilter1 = tex_filtering.y;
                @end

                @if(!s && !t)
                    vec2 vTexCoordAdj@{i} = vTexCoord@{i};
                @else
                    @if(s && t)
                        vec2 vTexCoordAdj@{i} = clamp(vTexCoord@{i}, 0.5 / texSize@{i}, vec2(vTexClampS@{i}, vTexClampT@{i}));
                    @elseif(s)
                        vec2 vTexCoordAdj@{i} = vec2(clamp(vTexCoord@{i}.s, 0.5 / texSize@{i}.s, vTexClampS@{i}), vTexCoord@{i}.t);
                    @else
                        vec2 vTexCoordAdj@{i} = vec2(vTexCoord@{i}.s, clamp(vTexCoord@{i}.t, 0.5 / texSize@{i}.t, vTexClampT@{i}));
                    @end
                @end

                vec4 texVal@{i} = hookTexture2D(texFilter@{i}, uTex@{i}, vTexCoordAdj@{i}, texSize@{i});

                @if(o_masks[i])
                    vec2 maskSize@{i} = vec2(textureSize(uTexMask@{i}, 0));

                    vec4 maskVal@{i} = hookTexture2D(texFilter@{i}, uTexMask@{i}, vTexCoordAdj@{i}, maskSize@{i});

                    @if(o_blend[i])
                        vec4 blendVal@{i} = hookTexture2D(texFilter@{i}, uTexBlend@{i}, vTexCoordAdj@{i}, texSize@{i});
                    @else
                        vec4 blendVal@{i} = vec4(0, 0, 0, 0);
                    @end

                    texVal@{i} = mix(texVal@{i}, blendVal@{i}, maskVal@{i}.a);
                @end
            @end
        @end

        @if(o_alpha)
            vec4 texel;
        @else
            vec3 texel;
        @end

        @if(o_2cyc)
            @{f_range = 2}
        @else
            @{f_range = 1}
        @end

        @for(c in 0..f_range)
            @if(c == 1)
                @if(o_alpha)
                    @if(o_c[c][1][2] == SHADER_COMBINED)
                        texel.a = WRAP(texel.a, -1.01, 1.01);
                    @else
                        texel.a = WRAP(texel.a, -0.51, 1.51);
                    @end
                @end

                @if(o_c[c][0][2] == SHADER_COMBINED)
                    texel.rgb = WRAP(texel.rgb, -1.01, 1.01);
                @else
                    texel.rgb = WRAP(texel.rgb, -0.51, 1.51);
                @end
            @end

            @if(!o_color_alpha_same[c] && o_alpha)
                texel = vec4(@{
                append_formula(o_c[c], o_do_single[c][0],
                            o_do_multiply[c][0], o_do_mix[c][0], false, false, true, c == 0)
                }, @{append_formula(o_c[c], o_do_single[c][1],
                            o_do_multiply[c][1], o_do_mix[c][1], true, true, true, c == 0)
                });
            @else
                texel = @{append_formula(o_c[c], o_do_single[c][0],
                            o_do_multiply[c][0], o_do_mix[c][0], o_alpha, false,
                            o_alpha, c == 0)};
            @end
        @end

        texel = WRAP(texel, -0.51, 1.51);
        texel = clamp(texel, 0.0, 1.0);
        @if(o_fog)
            @if(o_alpha)
                texel = vec4(mix(texel.rgb, vFog.rgb, vFog.a), texel.a);
            @else
                texel = mix(texel, vFog.rgb, vFog.a);
            @end
        @end

        @if(o_texture_edge && o_alpha)
            if (texel.a > 0.19) texel.a = 1.0; else discard;
        @end

        @if(o_alpha && o_noise)
            texel.a *= floor(clamp(random(vec3(floor(gl_FragCoord.xy * noise_scale), float(frame_count))) + texel.a, 0.0, 1.0));
        @end

        @if(o_grayscale)
            float intensity = (texel.r + texel.g + texel.b) / 3.0;
            vec3 new_texel = vGrayscaleColor.rgb * intensity;
            texel.rgb = mix(texel.rgb, new_texel, vGrayscaleColor.a);
        @end

        @if(o_alpha)
            @if(o_alpha_threshold)
                if (texel.a < 8.0 / 256.0) discard;
            @end
            @if(o_invisible)
                texel.a = 0.0;
            @end
            vOutColor = texel;
        @else
            vOutColor = vec4(texel, 1.0);
        @end

        @if(srgb_mode)
            vOutColor = fromLinear(vOutColor);
        @end

        @if(o_textures[0])
            @if(dk_debug_mode == 1)
                vOutColor = vec4(fract(vTexCoord0), min(abs(vTexCoord0.x), 16.0) / 16.0, 1.0);
            @end
            @if(dk_debug_mode == 2)
                vOutColor = vec4(texSize0 / 256.0, float(texFilter0) / 2.0, 1.0);
            @end
        @end

        @if(o_prim_depth)
            gl_FragDepth = prim_depth;
        @end
    }
@end
