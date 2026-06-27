#version 450

// Chunk vertex shader. Vertex layout (24 bytes):
//   location 0: vec3 pos   — local block corner in [0, 16]
//   location 1: vec2 uv    — atlas UV in [0, 1)
//   location 2: float light — baked face shading in [0, 1]
//                            — values > 1.0 signal water: 1.0 + (level/8)*0.5
//
// Push constants (offset 0): vec4 chunk_origin_and_pad = (originX, originY, originZ, _)
// Push constants (offset 16): mat4 view_proj
// Push constants (offset 80): vec4 time_and_pad = (game_time, _, _, _)
//
// World position = chunk_origin + local_pos. Clip = view_proj * world_pos.

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in float in_light;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_light;
layout(location = 2) out float frag_fog;
layout(location = 3) out vec3 frag_world_pos;

layout(push_constant) uniform Push {
    vec4 origin_pad;   // xyz = chunk world origin, w unused
    mat4 view_proj;
    vec4 time_and_pad; // x = game_time (seconds)
} push;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist; // xyz = camera pos, w = fog max distance
} cam;

void main() {
    vec3 local = in_pos;

    // Water animation: detect water via light > 1.0.
    if (in_light > 1.0) {
        // Extract water level from light encoding.
        float water_level = (in_light - 1.0) / 0.5 * 8.0; // 1.0..8.0
        float height_frac = water_level / 8.0;

        // Apply sine-wave to the top face vertices (Y component).
        // Use world XZ position for wave pattern.
        vec3 world_no_anim = push.origin_pad.xyz + local;
        float wave = sin(world_no_anim.x * 1.5 + push.time_and_pad.x * 1.8)
                   * cos(world_no_anim.z * 1.2 + push.time_and_pad.x * 1.4) * 0.04;
        // Only animate vertices at the water surface (those at the top of the
        // water block). Side faces have corners at y=0 and y=height_frac;
        // top faces have all corners at y=height_frac. Animate only the
        // upper corners (y > 0.0 relative to the block base) — this catches
        // the top face and the upper edge of side faces.
        if (abs(local.y - height_frac) < 0.01) {
            local.y += wave * height_frac;
        }
    }

    vec3 world = push.origin_pad.xyz + local;
    gl_Position = push.view_proj * vec4(world, 1.0);
    frag_uv = in_uv;
    frag_light = in_light;
    frag_world_pos = world;

    float dist = length(world - cam.cam_pos_and_maxdist.xyz);
    frag_fog = clamp(dist / cam.cam_pos_and_maxdist.w, 0.0, 1.0);
}
