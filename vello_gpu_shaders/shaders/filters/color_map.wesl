// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

fn filter_color(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

fn tint_color(source: vec4<f32>, black: vec3<f32>, white: vec3<f32>, amount: f32) -> vec4<f32> {
    if source.a <= 0.0 { return vec4<f32>(0.0); }
    let rgb = source.rgb / source.a;
    let luminance = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mapped = mix(rgb, mix(black, white, luminance), amount);
    return vec4<f32>(mapped * source.a, source.a);
}
