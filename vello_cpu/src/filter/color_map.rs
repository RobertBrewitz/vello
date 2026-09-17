// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::color::{AlphaColor, Srgb};
#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

pub(super) fn fill(pixmap: &mut Pixmap, color: AlphaColor<Srgb>) {
    let c = color.components;
    for pixel in pixmap.data_as_u8_slice_mut().chunks_exact_mut(4) {
        let alpha = f32::from(pixel[3]) / 255.0 * c[3];
        let mapped = [c[0] * alpha, c[1] * alpha, c[2] * alpha, alpha];
        for (dst, value) in pixel.iter_mut().zip(mapped) {
            *dst = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

pub(super) fn tint(
    pixmap: &mut Pixmap,
    black: AlphaColor<Srgb>,
    white: AlphaColor<Srgb>,
    amount: f32,
) {
    if amount == 0.0 {
        return;
    }
    for pixel in pixmap.data_as_u8_slice_mut().chunks_exact_mut(4) {
        if pixel[3] == 0 {
            continue;
        }
        let source: [f32; 4] = core::array::from_fn(|i| f32::from(pixel[i]) / 255.0);
        let gray = (source[0] * 0.2126 + source[1] * 0.7152 + source[2] * 0.0722) / source[3];
        for i in 0..3 {
            let mapped = black.components[i] + (white.components[i] - black.components[i]) * gray;
            let value = source[i] + (mapped * source[3] - source[i]) * amount;
            pixel[i] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}
