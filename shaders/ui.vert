#version 450

// UI vertex shader. Draws 2D quads in screen-space pixel coordinates with
// (0, 0) at the top-left. The push constant supplies the drawable size.

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;
layout(location = 3) in float in_tex_id;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out float frag_tex_id;

layout(push_constant) uniform Push {
    vec2 screen_size;
} push;

void main() {
    vec2 ndc = vec2(
        (in_pos.x / push.screen_size.x) * 2.0 - 1.0,
        (in_pos.y / push.screen_size.y) * 2.0 - 1.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    frag_uv = in_uv;
    frag_color = in_color;
    frag_tex_id = in_tex_id;
}
