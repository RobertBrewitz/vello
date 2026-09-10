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
    pub(crate) image_cache: ImageCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
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
            image_cache: ImageCache::new_with_config(image_atlas_config),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(feature = "text")]
            glyph_resources: None,
        }
    }
}
