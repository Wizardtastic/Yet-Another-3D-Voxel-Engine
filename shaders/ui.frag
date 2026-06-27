#version 450

// UI fragment shader. Samples one of two atlases based on tex_id:
//   0 → block atlas (hotbar icons)
//   1 → font atlas  (text)
// Multiplies the sampled colour by the per-vertex colour tint.

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in float frag_tex_id;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D block_atlas;
layout(set = 0, binding = 1) uniform sampler2D font_atlas;

void main() {
    vec4 tex;
    if (frag_tex_id < 0.5) {
        tex = texture(block_atlas, frag_uv);
    } else {
        tex = texture(font_atlas, frag_uv);
    }
    out_color = tex * frag_color;
}
