// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use crate::draw::{Draw, DrawBuffers, DrawBuilder, DrawState};
use crate::scene::RecordedDraw;
use crate::util::VecExt;
use vello_common::util::Clear;

/// Maximum number of scenes sharing one set of immutable GPU uploads.
pub const MAX_FLAT_GROUP_SCENES: usize = 8;
const MAX_GROUP_BYTES: usize = 4 * 1024 * 1024;
const MAX_GROUP_DRAWS: usize = 4096;
const MAX_GROUP_STRIPS: usize = 65536;

#[derive(Debug, Default)]
pub(super) struct FlatGroupStorage {
    paints: Vec<GpuEncodedPaint>,
    alphas: Vec<u8>,
    buffers: DrawBuffers,
    draws: Vec<Draw>,
    ranges: Vec<Range<u64>>,
    bindings: HashMap<ExternalTextureBindings, BindGroup>,
}

impl FlatGroupStorage {
    fn clear(&mut self) {
        self.paints.clear();
        self.alphas.clear();
        self.buffers.clear();
        for draw in &mut self.draws {
            draw.clear();
        }
        self.ranges.clear();
        self.bindings.clear();
    }
}

/// Prepared SDR scenes whose uploads remain immutable while drawing the group.
/// Drop this handle and submit its encoder before preparing any further renders.
#[derive(Debug)]
pub struct PreparedFlatGroup<'a> {
    renderer: &'a mut Renderer,
    storage: FlatGroupStorage,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
    texture_bindings: &'a TextureBindings,
}

impl PreparedFlatGroup<'_> {
    /// Number of leading input scenes accepted by this group.
    pub fn scene_count(&self) -> usize {
        self.storage.ranges.len()
    }

    /// Clear the shared target and draw one scene. Consume its output before drawing another.
    pub fn render_scene(&mut self, index: usize) {
        let range = self.storage.ranges[index].clone();
        let draw = &self.storage.draws[index];
        let strips = self.storage.buffers.strips.ranged(&draw.strip_ranges);
        let mut ctx = RendererContext {
            programs: &mut self.renderer.programs,
            device: self.device,
            queue: self.queue,
            encoder: self.encoder,
            view: self.view,
            depth_view: None,
            texture_bindings: self.texture_bindings,
            external_texture_bind_groups: core::mem::take(&mut self.storage.bindings),
            scratch_buffers: &mut self.renderer.scratch_buffers,
            root_load_op: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            uploaded_strips: Some(range),
        };
        ctx.strip_pass_inner(
            &[],
            strips,
            &draw.external_texture_runs,
            DrawPassTarget::Root(RootTarget::UserSurface),
            None,
        );
        ctx.finish_root_clear();
        self.storage.bindings = ctx.external_texture_bind_groups;
    }

    /// Access the encoder to compose each scene's output in painter order.
    pub fn encoder(&mut self) -> &mut CommandEncoder {
        self.encoder
    }
}

impl Drop for PreparedFlatGroup<'_> {
    fn drop(&mut self) {
        self.renderer.gradient_cache.maintain();
        self.storage.clear();
        self.renderer.programs.flat_group = core::mem::take(&mut self.storage);
    }
}

impl Renderer {
    /// Prepare a bounded prefix of flat, closed, source-over scenes without depth.
    /// Returns `None` if the first scene is unsupported or exceeds the upload budget.
    /// Use ordinary rendering for that scene, then resume grouping subsequent scenes.
    /// Returns `None` outside an explicit resource frame. Submit the encoder after dropping the group
    /// and before another renderer operation can overwrite its shared uploads.
    pub fn prepare_flat_group<'a>(
        &'a mut self,
        scenes: &[&Scene],
        resources: &mut Resources,
        device: &'a Device,
        queue: &'a Queue,
        encoder: &'a mut CommandEncoder,
        render_size: &RenderSize,
        view: &'a TextureView,
        texture_bindings: &'a TextureBindings,
    ) -> Result<Option<PreparedFlatGroup<'a>>, RenderError> {
        if !resources.frame_active || self.flat_group_prefix(scenes, device) == 0 {
            return Ok(None);
        }
        #[cfg(feature = "text")]
        self.prepare_glyphs(resources, device, queue, encoder, texture_bindings)?;
        let count = self.flat_group_prefix(scenes, device);
        if count == 0 {
            return Ok(None);
        }
        let mut storage = core::mem::take(&mut self.programs.flat_group);
        storage.clear();
        if let Err(error) = self.build_flat_group(
            &mut storage,
            &scenes[..count],
            resources,
            device,
            queue,
            render_size,
            view,
            texture_bindings,
        ) {
            storage.clear();
            self.programs.flat_group = storage;
            self.gradient_cache.maintain();
            return Err(error);
        }
        Ok(Some(PreparedFlatGroup {
            renderer: self,
            storage,
            device,
            queue,
            encoder,
            view,
            texture_bindings,
        }))
    }

    fn flat_group_prefix(&self, scenes: &[&Scene], device: &Device) -> usize {
        let dimension = self.programs.resources.resource_texture_dimension_2d as usize;
        let byte_limit = MAX_GROUP_BYTES
            .min(usize::try_from(device.limits().max_buffer_size).unwrap_or(usize::MAX))
            .min(dimension * dimension * 4);
        let mut bytes = self.gradient_cache.luts_size();
        let mut draws = 0_usize;
        let mut strips = 0_usize;
        let mut count = 0;
        for scene in scenes.iter().take(MAX_FLAT_GROUP_SCENES) {
            if scene.has_open_layers()
                || !scene.recorder.layers.is_empty()
                || scene.recorder.root_is_blend_target
                || scene.recorder.has_non_default_blend
            {
                break;
            }
            draws = draws.saturating_add(scene.recorder.draws.len());
            if draws > MAX_GROUP_DRAWS || scene.encoded_paints.len() > MAX_GROUP_DRAWS {
                break;
            }
            let scene_strips = scene.recorder.draws.iter().fold(0_usize, |total, draw| {
                total.saturating_add(match draw {
                    RecordedDraw::Path(path) => path.strips.len().saturating_mul(2),
                    RecordedDraw::Rect(_) => 5,
                })
            });
            strips = strips.saturating_add(scene_strips);
            if strips > MAX_GROUP_STRIPS {
                break;
            }
            let gradients = scene
                .encoded_paints
                .iter()
                .filter(|paint| matches!(paint, EncodedPaint::Gradient(_)))
                .count();
            bytes = bytes
                .saturating_add(
                    scene
                        .strip_storage
                        .borrow()
                        .alphas
                        .len()
                        .next_multiple_of(16),
                )
                .saturating_add(scene.encoded_paints.len() * size_of::<GpuEncodedPaint>())
                .saturating_add(scene_strips * size_of::<GpuStrip>())
                .saturating_add(gradients * MAX_GRADIENT_LUT_SIZE * 4);
            if bytes > byte_limit {
                break;
            }
            count += 1;
        }
        count
    }

    fn build_flat_group(
        &mut self,
        storage: &mut FlatGroupStorage,
        scenes: &[&Scene],
        resources: &Resources,
        device: &Device,
        queue: &Queue,
        render_size: &RenderSize,
        view: &TextureView,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        storage
            .draws
            .resize_with(storage.draws.len().max(scenes.len()), Draw::default);
        let mut paint_texels = 0;
        for (index, scene) in scenes.iter().enumerate() {
            self.prepare_gpu_encoded_paints(
                &scene.encoded_paints,
                &resources.image_cache,
                texture_bindings,
                view.texture(),
            )?;
            for offset in &mut self.paint_idxs {
                *offset += paint_texels;
            }
            paint_texels = *self.paint_idxs.last().unwrap();
            storage.paints.append(&mut self.encoded_paints);
            let strips = scene.strip_storage.borrow();
            let alpha_offset =
                u32::try_from(storage.alphas.len() / usize::from(Tile::HEIGHT)).unwrap();
            storage.alphas.extend_from_slice(&strips.alphas);
            storage
                .alphas
                .resize(storage.alphas.len().next_multiple_of(16), 0);
            let first = storage.buffers.strips.len();
            let mut state = DrawState::new(
                RootTarget::UserSurface,
                RectU16::new(
                    0,
                    0,
                    scene.recorder.scene_size.width(),
                    scene.recorder.scene_size.height(),
                ),
                false,
            );
            let resolver = PaintResolver::new(&scene.encoded_paints, &self.paint_idxs)
                .with_image_cache(&resources.image_cache);
            let mut builder =
                DrawBuilder::new(&mut storage.draws[index], &mut storage.buffers, &mut state);
            for node in &scene.recorder.nodes {
                for draw in node.draws_in(&scene.recorder.draws) {
                    builder.push_draw(draw, &strips, resolver);
                }
            }
            for strip in &mut storage.buffers.strips[first..] {
                strip.offset_alpha_columns(alpha_offset);
            }
            storage.ranges.push(
                (first * size_of::<GpuStrip>()) as u64
                    ..(storage.buffers.strips.len() * size_of::<GpuStrip>()) as u64,
            );
        }
        self.programs.prepare(
            device,
            queue,
            &mut self.gradient_cache,
            &storage.paints,
            &mut storage.alphas,
            render_size,
            &[paint_texels],
            &FilterContext::default(),
        );
        let bytes = bytemuck::cast_slice(&storage.buffers.strips);
        self.programs
            .strips_arena
            .begin_frame(device, bytes.len() as u64);
        if !bytes.is_empty() {
            queue.write_buffer(&self.programs.strips_arena.buffer, 0, bytes);
        }
        Ok(())
    }
}
