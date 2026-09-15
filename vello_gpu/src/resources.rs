// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Persistent renderer resources shared across frames.

#[cfg(feature = "text")]
use crate::text::GlyphAtlasResources;
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;

/// Persistent resources required by Vello GPU for rendering.
///
/// A set of resources must only be used with the renderer instance associated with it.
#[derive(Debug)]
pub struct Resources {
    pub(crate) frame_active: bool,
    #[cfg(feature = "text")]
    pub(crate) cache_diagnostics: CacheDiagnostics,
    pub(crate) image_cache: ImageCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

/// Cache gauges and cumulative work counters for one renderer's resources.
/// Atlas space includes images as well as glyphs. Counters also include diagnostic text.
#[cfg(feature = "text")]
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheDiagnostics {
    /// Rasterized glyph cache lookups that found an entry.
    pub glyph_hits: u64,
    /// Rasterized glyph cache lookups that found no entry; not necessarily evictions.
    pub glyph_misses: u64,
    /// Rasterized glyph entries currently retained.
    pub glyph_entries: usize,
    /// Outline paths currently retained, excluding hinting instances.
    pub outline_entries: usize,
    /// Rasterized glyph entries removed by age-based maintenance.
    pub glyph_evictions: u64,
    /// Outline paths removed by maintenance.
    pub outline_evictions: u64,
    /// Maintenance calls, normally one per explicit resource frame.
    pub maintenance_calls: u64,
    /// Scene preparations using glyph resources, excluding atlas raster passes.
    pub render_calls: u64,
    /// Passes that rasterized new glyphs into an atlas, not glyph count.
    pub atlas_raster_passes: u64,
    /// Bitmap glyph uploads to atlas textures.
    pub bitmap_uploads: u64,
    /// Evicted glyph regions cleared before reuse; not whole-page drops.
    pub atlas_clear_rects: u64,
    /// Current shared image/glyph atlas page count.
    pub atlas_pages: usize,
    /// Allocated shared atlas area in pixels, including padding.
    pub atlas_used_pixels: u64,
    /// Total shared atlas capacity in pixels.
    pub atlas_capacity_pixels: u64,
    /// Failed image or glyph allocations, including oversized requests.
    pub atlas_allocation_failures: u64,
}

impl Resources {
    /// Snapshot without resetting counters or changing cache behavior.
    #[cfg(feature = "text")]
    pub fn cache_diagnostics(&self) -> CacheDiagnostics {
        let mut stats = self.cache_diagnostics;
        if let Some(resources) = &self.glyph_resources {
            stats.glyph_hits = resources.glyph_atlas.cache_hits();
            stats.glyph_misses = resources.glyph_atlas.cache_misses();
            stats.glyph_entries = resources.glyph_atlas.len();
        }
        stats.outline_entries = self.glyph_prep_cache.outline_count();
        stats.atlas_pages = self.image_cache.atlas_count();
        stats.atlas_allocation_failures = self.image_cache.allocation_failures();
        for (_, atlas) in self.image_cache.atlas_manager().atlas_stats() {
            stats.atlas_used_pixels += u64::from(atlas.allocated_area);
            stats.atlas_capacity_pixels += u64::from(atlas.total_area);
        }
        stats
    }

    /// Defers glyph eviction across multiple renders sharing already encoded scenes.
    /// Finish with the renderer's `end_frame` after submitting every draw using these resources.
    /// Without an explicit frame, each render retains the single-render eviction behavior.
    ///
    /// # Panics
    ///
    /// Panics if a frame is already active.
    pub fn begin_frame(&mut self) {
        assert!(!self.frame_active, "A resource frame is already active");
        self.frame_active = true;
    }

    pub(crate) fn new(image_atlas_config: AtlasConfig) -> Self {
        Self {
            frame_active: false,
            #[cfg(feature = "text")]
            cache_diagnostics: CacheDiagnostics::default(),
            image_cache: ImageCache::new_with_config(image_atlas_config),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(feature = "text")]
            glyph_resources: None,
        }
    }
}
