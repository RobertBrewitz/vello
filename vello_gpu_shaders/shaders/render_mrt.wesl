// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

import package::render::shade_fragment;

struct Radiance {
    @location(0) lit: vec4<f32>,
    @location(1) unlit: vec4<f32>,
}

@fragment
fn fs_lit(
    @location(0) @interpolate(flat) paint_and_rect_flag: u32,
    @location(1) tex_coord: vec2<f32>,
    @location(2) sample_xy: vec2<f32>,
    @location(3) @interpolate(flat) dense_end_or_rect_size: u32,
    @location(4) @interpolate(flat) payload: u32,
    @location(5) @interpolate(flat) rect_frac: u32,
    @builtin(position) position: vec4<f32>,
) -> Radiance {
    let color = shade_fragment(paint_and_rect_flag, tex_coord, sample_xy,
        dense_end_or_rect_size, payload, rect_frac, position);
    // Both destinations must be attenuated by coverage, regardless of the RGB route.
    return Radiance(color, vec4(vec3(0.0), color.a));
}

@fragment
fn fs_unlit(
    @location(0) @interpolate(flat) paint_and_rect_flag: u32,
    @location(1) tex_coord: vec2<f32>,
    @location(2) sample_xy: vec2<f32>,
    @location(3) @interpolate(flat) dense_end_or_rect_size: u32,
    @location(4) @interpolate(flat) payload: u32,
    @location(5) @interpolate(flat) rect_frac: u32,
    @builtin(position) position: vec4<f32>,
) -> Radiance {
    let color = shade_fragment(paint_and_rect_flag, tex_coord, sample_xy,
        dense_end_or_rect_size, payload, rect_frac, position);
    return Radiance(vec4(vec3(0.0), color.a), color);
}
