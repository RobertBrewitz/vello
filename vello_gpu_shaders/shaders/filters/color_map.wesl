// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

import package::helpers::linear_color::srgb_to_linear;

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055,
                  color * 12.92, color <= vec3<f32>(0.0031308));
}

fn filter_color(color: vec4<f32>, linear: bool) -> vec4<f32> {
    var rgb = color.rgb;
    if linear { rgb = srgb_to_linear(rgb); }
    return vec4<f32>(rgb * color.a, color.a);
}

fn tint_color(source: vec4<f32>, black: vec3<f32>, white: vec3<f32>, amount: f32, linear: bool) -> vec4<f32> {
    if source.a <= 0.0 { return vec4<f32>(0.0); }
    var rgb = source.rgb / source.a;
    var peak = 1.0;
    if linear {
        // Our tint policy maps normalized sRGB and carries HDR radiance outside that mapping.
        peak = max(1.0, max(rgb.r, max(rgb.g, rgb.b)));
        rgb = linear_to_srgb(rgb / peak);
    }
    let luminance = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    var mapped = mix(rgb, mix(black, white, luminance), amount);
    if linear { mapped = srgb_to_linear(mapped) * peak; }
    return vec4<f32>(mapped * source.a, source.a);
}
