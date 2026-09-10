// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let rgb = max(c, vec3(0.0));
    return select(pow((rgb + vec3(0.055)) / 1.055, vec3(2.4)), rgb / 12.92, rgb <= vec3(0.04045));
}

fn linearize_premul(c: vec4<f32>) -> vec4<f32> {
    if c.a == 0.0 {
        return vec4(0.0);
    }
    return vec4(srgb_to_linear(c.rgb / c.a) * c.a, c.a);
}
