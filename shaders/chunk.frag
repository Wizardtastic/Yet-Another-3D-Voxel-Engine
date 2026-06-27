#version 450

#extension GL_EXT_nonuniform_qualifier : enable

// Chunk fragment shader. Samples the texture atlas at frag_uv, applies baked
// face light, and blends toward a sky fog colour based on frag_fog.

layout(constant_id = 0) const bool SHADOW_ENABLED = false;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in float frag_light;
layout(location = 2) in float frag_fog;
layout(location = 3) in vec3 frag_world_pos;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist; // xyz = camera pos, w = fog max distance
} cam;

layout(set = 0, binding = 1) uniform sampler2D atlas;

layout(set = 0, binding = 2) uniform Fog {
    vec4 color_and_density;   // rgb = fog colour, a unused
    vec4 ambient_and_sun;     // x = ambient brightness, yzw = sun direction
} fog;

layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadow_map;

layout(set = 0, binding = 4) uniform ShadowData {
    mat4 cascade_vps[4];      // 4 light-space view-projection matrices (256 bytes)
    vec4 cascade_splits;       // far-plane distance per cascade
    vec4 light_dir_and_bias;   // xyz = light direction, w = shadow bias
} shadow;

// 3x3 percentage-closer filtering against the selected cascade. Returns 1.0
// for fully lit fragments and 0.0 for fully occluded ones. Fragments outside
// the shadow map's valid range are treated as lit.
float compute_shadow_factor(vec3 world_pos, float view_depth) {
    int cascade_idx = 0;
    if (view_depth < shadow.cascade_splits.x) cascade_idx = 0;
    else if (view_depth < shadow.cascade_splits.y) cascade_idx = 1;
    else if (view_depth < shadow.cascade_splits.z) cascade_idx = 2;
    else cascade_idx = 3;

    vec4 light_pos = shadow.cascade_vps[cascade_idx] * vec4(world_pos, 1.0);
    vec3 proj_coords = light_pos.xyz / light_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;

    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        proj_coords.z > 1.0) {
        return 1.0;
    }

    float bias = shadow.light_dir_and_bias.w;
    float current_depth = proj_coords.z;

    vec2 texel_size = 1.0 / vec2(2048.0);
    float shadow_accum = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(x, y) * texel_size;
            shadow_accum += texture(shadow_map, vec4(proj_coords.xy + offset, float(cascade_idx), current_depth - bias));
        }
    }
    shadow_accum /= 9.0;

    return shadow_accum;
}

void main() {
    vec4 tex = texture(atlas, frag_uv);
    // Discard near-zero-alpha fragments so leaves/glass cutouts look right.
    if (tex.a < 0.1) {
        discard;
    }
    // frag_light > 1.0 signals water (encoded level). Clamp for lighting.
    float light = min(frag_light, 1.0);
    // Apply baked per-vertex light * dynamic ambient (day/night dimming).
    float ambient = fog.ambient_and_sun.x;

    float shadow_factor = 1.0;
    if (SHADOW_ENABLED) {
        float view_depth = length(cam.cam_pos_and_maxdist.xyz - frag_world_pos);
        shadow_factor = compute_shadow_factor(frag_world_pos, view_depth);
    }
    vec3 lit = tex.rgb * light * ambient * shadow_factor;
    vec3 final = mix(lit, fog.color_and_density.rgb, frag_fog);
    out_color = vec4(final, tex.a);
}