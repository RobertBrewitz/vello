// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::{filter::PreparedFilter, filter_effects::EdgeMode, kurbo::Vec2, pixmap::Pixmap};

pub(super) fn apply(filter: &PreparedFilter, pixmap: &mut Pixmap) {
    let mut pixels: Vec<[f32; 4]> = pixmap
        .data_as_u8_slice()
        .chunks_exact(4)
        .map(|p| core::array::from_fn(|i| f32::from(p[i]) / 255.0))
        .collect();
    match filter {
        PreparedFilter::Fill { color } => {
            let c = color.components;
            for p in &mut pixels {
                let alpha = p[3] * c[3];
                *p = [c[0] * alpha, c[1] * alpha, c[2] * alpha, alpha];
            }
        }
        PreparedFilter::Tint {
            black,
            white,
            amount,
        } => {
            for p in &mut pixels {
                if p[3] <= 0.0 {
                    continue;
                }
                let gray = (p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722) / p[3];
                for i in 0..3 {
                    let mapped =
                        black.components[i] + (white.components[i] - black.components[i]) * gray;
                    p[i] += (mapped * p[3] - p[i]) * amount;
                }
            }
        }
        PreparedFilter::GaussianBlurAxes { axes, edge_mode } => {
            let width = usize::from(pixmap.width());
            let height = usize::from(pixmap.height());
            if width == 0 || height == 0 {
                return;
            }
            let mut bounds = (width, height, 0, 0);
            for (i, _) in pixels.iter().enumerate().filter(|(_, p)| p[3] > 0.0) {
                let (x, y) = (i % width, i / width);
                bounds.0 = bounds.0.min(x);
                bounds.1 = bounds.1.min(y);
                bounds.2 = bounds.2.max(x + 1);
                bounds.3 = bounds.3.max(y + 1);
            }
            if bounds.2 == 0 || bounds.3 == 0 {
                return;
            }
            for (pass, &axis) in axes.iter().enumerate() {
                let sigma = axis.hypot();
                if !sigma.is_finite() {
                    continue;
                }
                let direction = if sigma > 0.0 {
                    axis / sigma
                } else {
                    Vec2::ZERO
                };
                let radius = (sigma * 3.0).ceil() as i32;
                let weights: Vec<f32> = (-radius..=radius)
                    .map(|i| {
                        if sigma == 0.0 {
                            return 1.0;
                        }
                        let offset = f64::from(i) / sigma;
                        (-0.5 * offset * offset).exp() as f32
                    })
                    .collect();
                let total: f32 = weights.iter().sum();
                let source = pixels.clone();
                let sample_bounds = if pass == 0 || *edge_mode == EdgeMode::Wrap {
                    bounds
                } else {
                    (0, 0, width, height)
                };
                for y in 0..height {
                    for x in 0..width {
                        let mut sum = [0.0; 4];
                        for (i, &weight) in (-radius..=radius).zip(&weights) {
                            let point = Vec2::new(x as f64, y as f64) + direction * f64::from(i);
                            let sample = bilinear(&source, width, sample_bounds, point, *edge_mode);
                            for c in 0..4 {
                                sum[c] += sample[c] * weight;
                            }
                        }
                        pixels[y * width + x] = sum.map(|v| v / total);
                    }
                }
            }
        }
        _ => unreachable!("not a layer color or directional blur filter"),
    }
    for (dst, p) in pixmap
        .data_as_u8_slice_mut()
        .chunks_exact_mut(4)
        .zip(pixels)
    {
        for i in 0..4 {
            dst[i] = (p[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

fn edge_index(index: i32, size: i32, mode: EdgeMode) -> Option<usize> {
    let index = match mode {
        EdgeMode::Duplicate => index.clamp(0, size - 1),
        EdgeMode::Wrap => index.rem_euclid(size),
        EdgeMode::Mirror => {
            let p = index.rem_euclid(size * 2);
            p.min(size * 2 - p - 1)
        }
        EdgeMode::None => index,
    };
    (index >= 0 && index < size).then_some(index as usize)
}

fn bilinear(
    pixels: &[[f32; 4]],
    width: usize,
    bounds: (usize, usize, usize, usize),
    point: Vec2,
    mode: EdgeMode,
) -> [f32; 4] {
    let (x0, y0, x1, y1) = bounds;
    let x = point.x.floor() as i32 - x0 as i32;
    let y = point.y.floor() as i32 - y0 as i32;
    let fx = (point.x - point.x.floor()) as f32;
    let fy = (point.y - point.y.floor()) as f32;
    let mut result = [0.0; 4];
    for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
        for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
            if let (Some(x), Some(y)) = (
                edge_index(x + dx, (x1 - x0) as i32, mode),
                edge_index(y + dy, (y1 - y0) as i32, mode),
            ) {
                let p = pixels[(y + y0) * width + x + x0];
                for c in 0..4 {
                    result[c] += p[c] * wx * wy;
                }
            }
        }
    }
    result
}
