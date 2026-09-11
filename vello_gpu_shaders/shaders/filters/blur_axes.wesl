// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

fn edge_index(index: i32, size: i32, mode: u32) -> i32 {
    if mode == 0u { return clamp(index, 0, size - 1); }
    if mode == 1u { return ((index % size) + size) % size; }
    if mode == 2u {
        let period = size * 2;
        let mapped = ((index % period) + period) % period;
        return min(mapped, period - mapped - 1);
    }
    return index;
}

fn edge_pixel(
    source: texture_2d<f32>, origin: vec2<u32>, size: vec2<u32>,
    point: vec2<i32>, mode: u32,
) -> vec4<f32> {
    let p = vec2<i32>(edge_index(point.x, i32(size.x), mode), edge_index(point.y, i32(size.y), mode));
    if any(p < vec2<i32>(0)) || any(p >= vec2<i32>(size)) { return vec4<f32>(0.0); }
    return textureLoad(source, vec2<i32>(origin) + p, 0);
}

fn edge_sample(
    source: texture_2d<f32>, origin: vec2<u32>, size: vec2<u32>,
    point: vec2<f32>, mode: u32,
) -> vec4<f32> {
    let base = vec2<i32>(floor(point));
    let f = fract(point);
    let top = mix(edge_pixel(source, origin, size, base, mode),
                  edge_pixel(source, origin, size, base + vec2<i32>(1, 0), mode), f.x);
    let bottom = mix(edge_pixel(source, origin, size, base + vec2<i32>(0, 1), mode),
                     edge_pixel(source, origin, size, base + vec2<i32>(1, 1), mode), f.x);
    return mix(top, bottom, f.y);
}

fn blur_axis(
    source: texture_2d<f32>, origin: vec2<u32>, size: vec2<u32>,
    point: vec2<f32>, axis: vec2<f32>, mode: u32,
) -> vec4<f32> {
    let sigma = length(axis);
    if sigma <= 0.0 { return edge_sample(source, origin, size, point, mode); }
    let direction = axis / sigma;
    let radius = i32(ceil(3.0 * sigma));
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = -radius; i <= radius; i += 1) {
        let offset = f32(i);
        let weight = exp(-0.5 * (offset / sigma) * (offset / sigma));
        sum += edge_sample(source, origin, size, point + direction * offset, mode) * weight;
        weight_sum += weight;
    }
    return sum / weight_sum;
}
