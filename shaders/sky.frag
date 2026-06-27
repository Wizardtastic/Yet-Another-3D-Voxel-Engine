#version 450

// Sky fragment shader. Renders a horizon-to-zenith gradient + sun glow.
// The sky pass runs first (depth=1.0, no depth write), so chunks overwrite
// it where geometry exists.

layout(location = 0) in vec3 frag_dir;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Sky {
    vec4 horizon_color;   // rgb = horizon colour, a unused
    vec4 zenith_color;    // rgb = zenith colour, a unused
    vec4 sun_dir_and_color; // xyz = sun direction (normalized), w = unused
} sky;

// Hash constants for the star noise function. Standard sin/fract hash.
const float STAR_FREQ = 800.0;
const vec2 STAR_HASH_A = vec2(12.9898, 78.233);
const float STAR_HASH_B = 43758.5453;
const float STAR_THRESHOLD = 0.998;
const float STAR_BRIGHTNESS = 500.0;

void main() {
    vec3 dir = normalize(frag_dir);

    // Horizon-to-zenith gradient based on the up component of the direction.
    float up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky_color = mix(sky.horizon_color.rgb, sky.zenith_color.rgb, up);

    // Sun glow: bright disc where the direction aligns with the sun.
    vec3 sun_dir = normalize(sky.sun_dir_and_color.xyz);
    float sun_dot = max(dot(dir, sun_dir), 0.0);
    float sun_glow = pow(sun_dot, 256.0);  // tight disc
    float sun_halo = pow(sun_dot, 8.0) * 0.3;  // soft halo

    // Only show sun when it's above the horizon.
    if (sun_dir.y > 0.0) {
        sky_color += vec3(1.0, 0.95, 0.8) * (sun_glow + sun_halo);
    }

    // Stars at night: simple hash-based noise, visible when the sky is dark.
    float star_noise = fract(sin(dot(dir.xz * STAR_FREQ, STAR_HASH_A)) * STAR_HASH_B);
    float night = clamp(-sky.sun_dir_and_color.y * 2.0, 0.0, 1.0);
    if (star_noise > STAR_THRESHOLD && dir.y > 0.0 && night > 0.3) {
        sky_color += vec3(0.8, 0.8, 1.0) * night * (star_noise - STAR_THRESHOLD) * STAR_BRIGHTNESS;
    }

    out_color = vec4(sky_color, 1.0);
}
