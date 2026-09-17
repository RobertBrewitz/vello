// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{RecordedDraw, Scene};
use alloc::vec::Vec;
use core::mem;
use vello_common::paint::{IndexedPaint, Paint};
use vello_common::record::Node;

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Scene>();
};

impl Scene {
    /// Takes the completed recording while retaining rasterizer scratch for reuse.
    ///
    /// The returned scene owns the recorded data and starts with default drawing state.
    /// This scene is reset on success. Returns `None` without changing anything while layers
    /// or clip paths remain open. No GPU resources are accessed.
    pub fn take_recording(&mut self) -> Option<Self> {
        if !self.recording_is_closed() {
            return None;
        }
        let mut recording = Self::new(self.width, self.height);
        mem::swap(&mut self.recorder, &mut recording.recorder);
        recording.encoded_paints = mem::take(&mut self.encoded_paints);
        mem::swap(
            self.strip_storage.get_mut(),
            recording.strip_storage.get_mut(),
        );
        self.reset();
        Some(recording)
    }

    fn recording_is_closed(&self) -> bool {
        !self.recorder.has_layers() && self.viewport_state.clip().is_none()
    }

    /// Appends already rasterized scene data in painter order, then resets `other`.
    ///
    /// No additional isolation layer is introduced, and paths are not rasterized again.
    /// Coordinates, clips, filters and paints retain their recorded viewport-space meaning;
    /// the destination's drawing state is neither applied to the appended data nor changed.
    /// Build both scenes for the same viewport and final transforms. Resource handles in both
    /// scenes must resolve against the same rendering resources and texture bindings.
    ///
    /// Returns `false`, leaving both scenes unchanged, if dimensions differ, either scene has
    /// open layers or clip paths, or the combined recording would exceed its index limits.
    /// The caller can then use its ordinary drawing path instead.
    ///
    /// This moves owned payloads and copies/remaps buffer contents in linear time. It does not
    /// upload GPU resources or make shared mutable scene access safe.
    ///
    /// # Usage
    ///
    /// Prepare fragments on worker-local scenes, then collect them in painter order rather than
    /// worker completion order. Drawing callbacks must use final viewport-space transforms and
    /// close all layers and clip paths; workers must not access shared mutable rendering resources.
    ///
    /// ```
    /// use vello_gpu::Scene;
    ///
    /// fn prepare_entity(
    ///     worker_scene: &mut Scene,
    ///     viewport: (u16, u16),
    ///     draw: impl FnOnce(&mut Scene),
    /// ) -> Option<Scene> {
    ///     worker_scene.reset_and_resize(viewport.0, viewport.1);
    ///     draw(worker_scene);
    ///     worker_scene.take_recording()
    /// }
    ///
    /// fn compose_entities(
    ///     scene: &mut Scene,
    ///     fragments_in_painter_order: &mut [Option<Scene>],
    ///     mut draw_entity: impl FnMut(usize, &mut Scene),
    /// ) {
    ///     for (entity, fragment) in fragments_in_painter_order.iter_mut().enumerate() {
    ///         let appended = fragment
    ///             .as_mut()
    ///             .is_some_and(|fragment| scene.try_append(fragment));
    ///         if !appended {
    ///             draw_entity(entity, scene);
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// A successful append consumes the fragment's recording; it is not a reusable cache.
    #[must_use]
    pub fn try_append(&mut self, other: &mut Self) -> bool {
        if self.width != other.width
            || self.height != other.height
            || !self.recording_is_closed()
            || !other.recording_is_closed()
        {
            return false;
        }

        let storage = self.strip_storage.get_mut();
        let incoming = other.strip_storage.get_mut();
        let fits =
            |a: usize, b: usize, max: usize| a.checked_add(b).is_some_and(|total| total <= max);
        // The high bit of a strip's alpha index stores its fill-gap flag.
        if !fits(
            storage.alphas.len(),
            incoming.alphas.len(),
            (u32::MAX >> 1) as usize,
        ) || !fits(storage.strips.len(), incoming.strips.len(), usize::MAX)
            || !fits(
                self.recorder.draws.len(),
                other.recorder.draws.len(),
                u32::MAX as usize,
            )
            || !fits(
                self.recorder.layers.len(),
                other.recorder.layers.len(),
                u32::MAX as usize,
            )
            || !fits(
                self.encoded_paints.len(),
                other.encoded_paints.len(),
                u32::MAX as usize,
            )
        {
            return false;
        }

        let strip_base = storage.strips.len();
        let alpha_base = u32::try_from(storage.alphas.len()).unwrap();
        let paint_base = self.encoded_paints.len();
        let draw_base = u32::try_from(self.recorder.draws.len()).unwrap();
        let layer_base = u32::try_from(self.recorder.layers.len()).unwrap();

        for strip in &mut incoming.strips {
            strip.set_alpha_idx(strip.alpha_idx() + alpha_base);
        }
        for draw in &mut other.recorder.draws {
            let paint = match draw {
                RecordedDraw::Path(path) => {
                    path.strips.start += strip_base;
                    path.strips.end += strip_base;
                    &mut path.paint
                }
                RecordedDraw::Rect(rect) => &mut rect.paint,
            };
            if let Paint::Indexed(index) = paint {
                *index = IndexedPaint::new(index.index() + paint_base);
            }
        }
        let remap_node = |node: &mut Node| {
            node.draws.start += draw_base;
            node.draws.end += draw_base;
            if let Some(layer) = &mut node.layer {
                *layer += layer_base;
            }
        };
        for node in &mut other.recorder.nodes {
            remap_node(node);
        }
        for layer in &mut other.recorder.layers {
            for node in &mut layer.nodes {
                remap_node(node);
            }
            if let Some(clip) = &mut layer.props.clip_path {
                clip.strip_range.start += strip_base;
                clip.strip_range.end += strip_base;
            }
        }
        for layer in &mut other.recorder.filter_layers {
            *layer += layer_base;
        }

        append_vec(&mut storage.strips, &mut incoming.strips);
        append_vec(&mut storage.alphas, &mut incoming.alphas);
        append_vec(&mut self.encoded_paints, &mut other.encoded_paints);

        let target = &mut self.recorder;
        let source = &mut other.recorder;
        if let (Some(last), Some(first)) = (target.nodes.last(), source.nodes.first_mut())
            && last.layer.is_none()
            && last.draws.end == first.draws.start
        {
            first.draws.start = last.draws.start;
            target.nodes.pop();
        }
        append_vec(&mut target.nodes, &mut source.nodes);
        append_vec(&mut target.draws, &mut source.draws);
        append_vec(&mut target.layers, &mut source.layers);
        append_vec(&mut target.filter_layers, &mut source.filter_layers);
        target.root_is_blend_target |= source.root_is_blend_target;
        target.has_non_default_blend |= source.has_non_default_blend;
        target.max_layer_depth = target.max_layer_depth.max(source.max_layer_depth);
        if let Some(size) = source.largest_layer_size {
            target.largest_layer_size =
                Some(target.largest_layer_size.map_or(size, |s| s.max(size)));
        }
        if let Some(size) = source.largest_filter_layer_size {
            target.largest_filter_layer_size = Some(
                target
                    .largest_filter_layer_size
                    .map_or(size, |s| s.max(size)),
            );
        }
        other.reset();
        true
    }
}

fn append_vec<T>(target: &mut Vec<T>, source: &mut Vec<T>) {
    // Keep reusable destination capacity instead of replacing it with a smaller fragment.
    if target.is_empty() && target.capacity() < source.len() {
        mem::swap(target, source);
    } else {
        target.append(source);
    }
}
