#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in float in_light;

layout(push_constant) uniform Push {
    mat4 light_view_proj;   // 64 bytes
    vec4 chunk_origin;      // 16 bytes (xyz = origin, w = cascade index)
} push;

void main() {
    vec3 world = push.chunk_origin.xyz + in_pos;
    gl_Position = push.light_view_proj * vec4(world, 1.0);
}
