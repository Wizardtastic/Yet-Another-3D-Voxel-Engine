#version 450

// Sky vertex shader. Generates a full-screen triangle from vertex ID — no
// vertex buffer needed. Outputs clip position and a view direction for the
// fragment shader to reconstruct the world-space ray.

layout(location = 0) out vec3 frag_dir;

layout(push_constant) uniform Push {
    mat4 inverse_view_proj;
    vec4 camera_pos;
} push;

void main() {
    // Full-screen triangle: 3 vertices covering the screen.
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.9999, 1.0);

    // Reconstruct world position from clip-space, then compute view direction.
    vec4 world = push.inverse_view_proj * vec4(pos * 2.0 - 1.0, 1.0, 1.0);
    frag_dir = normalize(world.xyz / world.w - push.camera_pos.xyz);
}
