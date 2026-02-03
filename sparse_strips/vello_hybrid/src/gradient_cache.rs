// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient ramp cache for `vello_hybrid` renderer.
//!
//! Gradient LUTs persist across frames.  When a gradient is requested that
//! already exists in the cache the existing offset is returned and no GPU
//! re-upload is needed.  At the end of each frame, gradients not used that
//! frame are evicted and the buffer is compacted.

use alloc::vec::Vec;
use hashbrown::HashMap;
use vello_common::encode::{EncodedGradient, GradientCacheKey};
use vello_common::fearless_simd::{Level, Simd, dispatch};
use vello_common::peniko::color::cache_key::CacheKey;

/// Number of bytes per texel in the gradient texture.
/// Gradient textures use `Rgba8Unorm` format (4 bytes per texel).
/// This constant is used to convert between byte offsets and texel indices.
const BYTES_PER_TEXEL: u32 = 4;

/// Packed gradient look-up tables that persist across frames.
#[derive(Debug)]
pub(crate) struct GradientRampCache {
    /// Cache mapping gradient key to its ramp location and last-used epoch.
    cache: HashMap<CacheKey<GradientCacheKey>, CachedRamp>,
    /// Packed gradient luts.
    luts: Vec<u8>,
    /// Whether the packed luts needs to be re-uploaded.
    has_changed: bool,
    /// Current frame epoch, incremented each frame in `maintain()`.
    epoch: u64,
    /// SIMD level used for gradient LUT generation.
    level: Level,
}

impl GradientRampCache {
    /// Create a new gradient ramp cache.
    pub(crate) fn new(level: Level) -> Self {
        Self {
            cache: HashMap::new(),
            luts: Vec::new(),
            has_changed: false,
            epoch: 0,
            level,
        }
    }

    /// Get or generate a gradient ramp, returning its (`lut_start`, `width`) in the packed luts.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Conversion from usize to u32 is safe, used for texture coordinates"
    )]
    pub(crate) fn get_or_create_ramp(&mut self, gradient: &EncodedGradient) -> (u32, u32) {
        if let Some(ramp) = self.cache.get_mut(&gradient.cache_key) {
            ramp.last_used = self.epoch;
            return (ramp.lut_start, ramp.width);
        }

        // Generate new gradient LUT.
        let lut_start = self.luts.len() as u32 / BYTES_PER_TEXEL;
        let width = dispatch!(self.level, simd => generate_gradient_lut_impl(simd, gradient, &mut self.luts))
            as u32;
        self.cache.insert(
            gradient.cache_key.clone(),
            CachedRamp {
                lut_start,
                width,
                last_used: self.epoch,
            },
        );
        self.has_changed = true;
        (lut_start, width)
    }

    /// End-of-frame maintenance: evict unused entries and compact the buffer.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Conversion from usize to u32 is safe, used for texture coordinates"
    )]
    pub(crate) fn maintain(&mut self) {
        let len_before = self.cache.len();
        self.cache.retain(|_, r| r.last_used >= self.epoch);
        if self.cache.len() < len_before {
            // Rebuild the LUT buffer compactly from surviving entries.
            let mut new_luts = Vec::with_capacity(self.luts.len());
            for (_, ramp) in self.cache.iter_mut() {
                let src_start = (ramp.lut_start * BYTES_PER_TEXEL) as usize;
                let src_end = src_start + (ramp.width * BYTES_PER_TEXEL) as usize;
                ramp.lut_start = new_luts.len() as u32 / BYTES_PER_TEXEL;
                new_luts.extend_from_slice(&self.luts[src_start..src_end]);
            }
            self.luts = new_luts;
            self.has_changed = true;
        }
        self.epoch += 1;
    }

    /// Get the size of the packed luts.
    pub(crate) fn luts_size(&self) -> usize {
        self.luts.len()
    }

    /// Check if the packed luts is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.luts.is_empty()
    }

    /// Check if the luts data has changed.
    pub(crate) fn has_changed(&self) -> bool {
        self.has_changed
    }

    /// Mark the luts as synced.
    pub(crate) fn mark_synced(&mut self) {
        self.has_changed = false;
    }

    /// Take ownership of the luts, leaving an empty vector in its place.
    pub(crate) fn take_luts(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.luts)
    }

    /// Restore the luts. The restored luts should have the same logical content as the original.
    pub(crate) fn restore_luts(&mut self, luts: Vec<u8>) {
        self.luts = luts;
    }
}

/// Cached gradient ramp location in the packed LUT buffer.
#[derive(Debug, Clone)]
struct CachedRamp {
    /// Width of this gradient's LUT in texels.
    width: u32,
    /// Offset in the packed LUT buffer where this ramp starts (in texels).
    lut_start: u32,
    /// Epoch when this ramp was last used.
    last_used: u64,
}

/// Generate the gradient LUT.
// TODO: Consider adding a method that generates LUT data directly into output buffer
// to avoid duplicate allocation when lut() is only used once (e.g., in gradient cache).
// The current approach allocates LUT in OnceCell and then copies to output, keeping
// both allocations alive.
#[inline(always)]
fn generate_gradient_lut_impl<S: Simd>(
    simd: S,
    gradient: &EncodedGradient,
    output: &mut Vec<u8>,
) -> usize {
    let lut = gradient.u8_lut(simd);
    let bytes: &[u8] = bytemuck::cast_slice(lut.lut());
    output.reserve(bytes.len());
    output.extend_from_slice(bytes);
    lut.width()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use vello_common::color::{ColorSpaceTag, DynamicColor, HueDirection};
    use vello_common::encode::{EncodeExt, EncodedPaint};
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko::{Color, ColorStop, ColorStops, Gradient, LinearGradientPosition};

    fn insert_entry(cache: &mut GradientRampCache, gradient: Gradient) -> (u32, u32) {
        let encoded_gradient = create_encoded_gradient(gradient);
        cache.get_or_create_ramp(&encoded_gradient)
    }

    fn create_encoded_gradient(gradient: Gradient) -> EncodedGradient {
        let mut encoded_paints = vec![];
        gradient.encode_into(&mut encoded_paints, Affine::IDENTITY, None);
        match encoded_paints.into_iter().last().unwrap() {
            EncodedPaint::Gradient(encoded_gradient) => encoded_gradient,
            _ => panic!("Expected a gradient paint"),
        }
    }

    fn create_gradient(offset: f32) -> Gradient {
        Gradient {
            kind: LinearGradientPosition {
                start: Point::new(0.0, 0.0),
                end: Point::new(100.0, 0.0),
            }
            .into(),
            stops: ColorStops(
                vec![
                    ColorStop {
                        offset: 0.0,
                        color: DynamicColor::from_alpha_color(Color::from_rgb8(255, 0, 0)),
                    },
                    ColorStop {
                        offset,
                        color: DynamicColor::from_alpha_color(Color::from_rgb8(0, 255, 0)),
                    },
                    ColorStop {
                        offset: 1.0,
                        color: DynamicColor::from_alpha_color(Color::from_rgb8(0, 0, 255)),
                    },
                ]
                .into(),
            ),
            interpolation_cs: ColorSpaceTag::Srgb,
            hue_direction: HueDirection::Shorter,
            ..Default::default()
        }
    }

    #[test]
    fn test_empty() {
        let cache = GradientRampCache::new(Level::baseline());
        assert!(cache.is_empty());
        assert!(!cache.has_changed());
    }

    #[test]
    fn test_insert_creates_lut_data() {
        let mut cache = GradientRampCache::new(Level::baseline());
        let (start, width) = insert_entry(&mut cache, create_gradient(0.5));

        assert_eq!(start, 0);
        assert!(width > 0);
        assert!(!cache.is_empty());
        assert!(cache.has_changed());
        assert_eq!(cache.luts_size(), (width * BYTES_PER_TEXEL) as usize);
    }

    #[test]
    fn test_cache_hit_no_buffer_growth() {
        let mut cache = GradientRampCache::new(Level::baseline());
        let (start1, width1) = insert_entry(&mut cache, create_gradient(0.5));
        let size_after_first = cache.luts_size();

        cache.mark_synced();
        cache.maintain();
        let (start2, width2) = insert_entry(&mut cache, create_gradient(0.5));

        assert_eq!(start1, start2);
        assert_eq!(width1, width2);
        assert_eq!(cache.luts_size(), size_after_first);
        assert!(!cache.has_changed());
    }

    #[test]
    fn test_multiple_inserts_are_contiguous() {
        let mut cache = GradientRampCache::new(Level::baseline());

        let (start1, width1) = insert_entry(&mut cache, create_gradient(0.1));
        let (start2, width2) = insert_entry(&mut cache, create_gradient(0.2));
        let (start3, _width3) = insert_entry(&mut cache, create_gradient(0.3));

        assert_eq!(start1, 0);
        assert_eq!(start2, start1 + width1);
        assert_eq!(start3, start2 + width2);
    }

    #[test]
    fn test_static_scene_no_reupload() {
        let mut cache = GradientRampCache::new(Level::baseline());

        // Frame 1: insert gradients, then maintain.
        insert_entry(&mut cache, create_gradient(0.1));
        insert_entry(&mut cache, create_gradient(0.2));
        assert!(cache.has_changed());
        cache.mark_synced();
        cache.maintain();

        // Frame 2: same gradients — all cache hits.
        insert_entry(&mut cache, create_gradient(0.1));
        insert_entry(&mut cache, create_gradient(0.2));
        cache.maintain();
        assert!(
            !cache.has_changed(),
            "Static scene should not trigger re-upload"
        );
    }

    #[test]
    fn test_stale_entries_evicted() {
        let mut cache = GradientRampCache::new(Level::baseline());

        // Frame 1: insert A and B.
        insert_entry(&mut cache, create_gradient(0.1));
        insert_entry(&mut cache, create_gradient(0.2));
        let size_with_two = cache.luts_size();
        cache.mark_synced();
        cache.maintain();

        // Frame 2: only use A. B becomes stale.
        insert_entry(&mut cache, create_gradient(0.1));
        cache.maintain();

        assert!(cache.has_changed(), "Eviction should trigger re-upload");
        assert!(
            cache.luts_size() < size_with_two,
            "Buffer should shrink after evicting stale entry"
        );
    }

    #[test]
    fn test_dynamic_gradients_dont_accumulate() {
        let mut cache = GradientRampCache::new(Level::baseline());

        // Frame 1: gradient A.
        let (_, width_a) = insert_entry(&mut cache, create_gradient(0.1));
        let size_a = (width_a * BYTES_PER_TEXEL) as usize;
        cache.mark_synced();
        cache.maintain();

        // Frame 2: completely different gradient B. A should be evicted.
        let (_, width_b) = insert_entry(&mut cache, create_gradient(0.2));
        let size_b = (width_b * BYTES_PER_TEXEL) as usize;
        cache.maintain();

        // Buffer should contain only B, not A+B.
        assert_eq!(cache.luts_size(), size_b);
        assert!(
            cache.luts_size() <= size_a + size_b,
            "Buffer should not contain both A and B"
        );
    }

    #[test]
    fn test_compaction_offset_correctness() {
        let mut cache = GradientRampCache::new(Level::baseline());

        // Frame 1: insert A, B, C contiguously.
        let (start_a, width_a) = insert_entry(&mut cache, create_gradient(0.1));
        let (start_b, _width_b) = insert_entry(&mut cache, create_gradient(0.2));
        let (_start_c, width_c) = insert_entry(&mut cache, create_gradient(0.3));

        assert_eq!(start_a, 0);
        assert!(start_b > start_a);
        cache.mark_synced();
        cache.maintain();

        // Frame 2: use A and C but not B. B should be evicted.
        let encoded_a = create_encoded_gradient(create_gradient(0.1));
        let encoded_c = create_encoded_gradient(create_gradient(0.3));
        cache.get_or_create_ramp(&encoded_a);
        cache.get_or_create_ramp(&encoded_c);
        cache.maintain();

        // Re-read offsets after compaction (maintain updated lut_start in-place).
        let (new_start_a, new_width_a) = cache.get_or_create_ramp(&encoded_a);
        let (new_start_c, new_width_c) = cache.get_or_create_ramp(&encoded_c);

        // Widths should be unchanged.
        assert_eq!(new_width_a, width_a);
        assert_eq!(new_width_c, width_c);

        // After compaction, entries should be contiguous starting from 0.
        let mut offsets = [(new_start_a, new_width_a), (new_start_c, new_width_c)];
        offsets.sort_by_key(|(start, _)| *start);

        assert_eq!(offsets[0].0, 0, "First entry should start at 0 after compaction");
        assert_eq!(
            offsets[1].0,
            offsets[0].0 + offsets[0].1,
            "Entries should be contiguous after compaction"
        );

        // Total buffer size should equal sum of surviving widths.
        let total_width = new_width_a + new_width_c;
        assert_eq!(cache.luts_size(), (total_width * BYTES_PER_TEXEL) as usize);
    }

    #[test]
    fn test_take_and_restore_luts() {
        let mut cache = GradientRampCache::new(Level::baseline());

        insert_entry(&mut cache, create_gradient(0.1));
        insert_entry(&mut cache, create_gradient(0.2));
        let original_size = cache.luts_size();

        let luts = cache.take_luts();
        assert_eq!(luts.len(), original_size);
        assert!(cache.is_empty());

        cache.restore_luts(luts);
        assert_eq!(cache.luts_size(), original_size);
        assert!(!cache.is_empty());
    }
}
