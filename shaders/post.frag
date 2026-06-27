#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_color;

layout(push_constant) uniform Push {
    vec4 params; // x = exposure, y = vignette_strength, z = time, w = unused
} push;

// ACES filmic tone mapping.
vec3 aces_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

void main() {
    vec3 color = texture(scene_color, frag_uv).rgb;
    color *= push.params.x;
    color = aces_tonemap(color);

    vec2 uv = frag_uv * 2.0 - 1.0;
    float vignette = 1.0 - dot(uv, uv) * push.params.y;
    vignette = clamp(vignette, 0.0, 1.0);
    color *= vignette;

    out_color = vec4(color, 1.0);
}
