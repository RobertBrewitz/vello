// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use crate::draw::{Draw, DrawBuffers, DrawBuilder, DrawState};
use crate::scene::RecordedDraw;
use crate::util::VecExt;
use core::ops::Range;
use vello_common::util::Clear;

/// Maximum number of scenes prepared in one bounded group.
pub const MAX_FLAT_GROUP_SCENES: usize = 8;
const MAX_GROUP_BYTES: usize = 4 * 1024 * 1024;
const MAX_GROUP_DRAWS: usize = 4096;
const MAX_GROUP_STRIPS: usize = 65536;

/// RGB destination for a source-over scene; coverage attenuates both destinations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadianceRoute {
    /// Contributes RGB to the lit attachment.
    Lit,
    /// Contributes RGB to the unlit attachment.
    Unlit,
}

#[derive(Debug, Default)]
pub(super) struct FlatGroupStorage {
    paints: Vec<GpuEncodedPaint>,
    buffers: DrawBuffers,
    draws: Vec<Draw>,
    ranges: Vec<Range<u64>>,
    bindings: HashMap<ExternalTextureBindings, BindGroup>,
    strip_buffer: Option<Buffer>,
}

impl FlatGroupStorage {
    fn clear(&mut self) {
        self.paints.clear();
        self.buffers.clear();
        for draw in &mut self.draws {
            draw.clear();
        }
        self.ranges.clear();
        self.bindings.clear();
    }
}

/// A bounded group sharing immutable uploads until its command encoder is submitted.
/// Drop the group before submitting, and submit before using the renderer again.
#[derive(Debug)]
pub struct PreparedFlatGroup<'a> {
    renderer: &'a mut Renderer,
    storage: FlatGroupStorage,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
    radiance_view: Option<&'a TextureView>,
    texture_bindings: &'a TextureBindings,
}

impl PreparedFlatGroup<'_> {
    /// Number of leading input scenes accepted by this group.
    pub fn scene_count(&self) -> usize {
        self.storage.ranges.len()
    }

    /// Clear the target and draw one prepared scene, without updating shared GPU data.
    /// Compose its result before drawing another scene into the same target.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not less than [`Self::scene_count`].
    pub fn render_scene(&mut self, index: usize) {
        let started = self.renderer.programs.diagnostics.start_timer();
        let range = self.storage.ranges[index].clone();
        let draw = &self.storage.draws[index];
        let strips = self.storage.buffers.strips.ranged(&draw.strip_ranges);
        self.renderer
            .programs
            .diagnostics
            .update(|report| report.render_calls += 1);
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
            pending_root_clear: true,
            uploaded_strips: if range.is_empty() {
                None
            } else {
                Some(self.storage.strip_buffer.as_ref().unwrap().slice(range))
            },
        };
        ctx.strip_pass_inner(
            &[],
            strips,
            &draw.external_texture_runs,
            DrawPassTarget::Root(RootTarget::UserSurface),
            None,
        );
        if ctx.pending_root_clear {
            Renderer::clear_view(ctx.encoder, ctx.view, &mut ctx.programs.diagnostics);
        }
        self.storage.bindings = ctx.external_texture_bind_groups;
        self.renderer
            .programs
            .diagnostics
            .end_timer(started, |cpu| &mut cpu.pass_recording);
        self.renderer
            .programs
            .diagnostics
            .end_timer(started, |cpu| &mut cpu.render);
    }

    /// Draw all prepared scenes into the two radiance attachments in one pass.
    /// `clear` initializes both attachments to transparent; otherwise both are loaded.
    ///
    /// # Panics
    ///
    /// Panics unless prepared with [`Renderer::prepare_radiance_group`], or if the route
    /// count differs from [`Self::scene_count`].
    pub fn render_radiance(&mut self, routes: &[RadianceRoute], clear: bool) {
        let unlit_view = self
            .radiance_view
            .expect("Group was not prepared for radiance output");
        assert_eq!(routes.len(), self.scene_count());
        let started = self.renderer.programs.diagnostics.start_timer();
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
            pending_root_clear: false,
            uploaded_strips: None,
        };
        for draw in &self.storage.draws[..routes.len()] {
            for run in &draw.external_texture_runs {
                ctx.external_texture_bind_group_for_textures(run.bindings);
            }
        }
        ctx.programs
            .ensure_strip_bind_group(self.device, (StripTargetKind::Root, None));
        self.storage.bindings = ctx.external_texture_bind_groups;
        let programs = &mut self.renderer.programs;
        programs
            .diagnostics
            .update(|report| report.render_calls += routes.len() as u64);
        let attachments = [self.view, unlit_view].map(|view| {
            Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })
        });
        let mut pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Flat Group Radiance"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: programs.diagnostics.pass(PassKind::RootRadiance),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(
            0,
            &programs.strip_layer_bind_groups[&(StripTargetKind::Root, None)],
            &[],
        );
        pass.set_bind_group(2, &programs.resources.encoded_paints_bind_group, &[]);
        pass.set_bind_group(3, &programs.resources.gradient_bind_group, &[]);
        if let Some(buffer) = &self.storage.strip_buffer {
            pass.set_vertex_buffer(0, buffer.slice(..));
        }
        let pipelines = programs.radiance_pipelines.as_ref().unwrap();
        for (index, route) in routes.iter().enumerate() {
            let range = &self.storage.ranges[index];
            if range.is_empty() {
                continue;
            }
            pass.set_pipeline(
                &pipelines[match route {
                    RadianceRoute::Lit => 0,
                    RadianceRoute::Unlit => 1,
                }],
            );
            let first = (range.start / size_of::<GpuStrip>() as u64) as u32;
            let count = ((range.end - range.start) / size_of::<GpuStrip>() as u64) as u32;
            let runs = &self.storage.draws[index].external_texture_runs;
            if runs.is_empty() {
                pass.set_bind_group(
                    1,
                    &programs.resources.empty_external_texture_bind_group,
                    &[],
                );
                pass.draw(0..4, first..first + count);
            } else {
                for (run_index, run) in runs.iter().enumerate() {
                    pass.set_bind_group(1, &self.storage.bindings[&run.bindings], &[]);
                    let start = run.strips_start as u32;
                    let end = runs
                        .get(run_index + 1)
                        .map_or(count, |next| next.strips_start as u32);
                    pass.draw(0..4, first + start..first + end);
                }
            }
        }
        drop(pass);
        programs
            .diagnostics
            .end_timer(started, |cpu| &mut cpu.pass_recording);
        programs
            .diagnostics
            .end_timer(started, |cpu| &mut cpu.render);
    }

    /// Access the group's encoder to compose or copy each scene's result in order.
    pub fn encoder(&mut self) -> &mut CommandEncoder {
        self.encoder
    }
}

impl Drop for PreparedFlatGroup<'_> {
    fn drop(&mut self) {
        let started = self.renderer.programs.diagnostics.start_timer();
        self.renderer.gradient_cache.maintain();
        self.storage.clear();
        self.renderer.programs.flat_group = core::mem::take(&mut self.storage);
        self.renderer
            .programs
            .diagnostics
            .end_timer(started, |cpu| &mut cpu.render);
    }
}

impl Renderer {
    /// Prepare a bounded prefix of flat, closed, source-over scenes without depth.
    /// Returns `None` when the first scene is unsupported or exceeds the group budget;
    /// render it with [`Self::render`] instead. Groups contain at most eight scenes,
    /// 4096 recorded draws, 65536 estimated strips and 4 MiB of estimated stream/LUT data.
    ///
    /// All scenes use the same target and dimensions. Preparation uploads shared data once;
    /// use the returned handle to draw and compose each scene in painter's order.
    /// Submit `encoder` after dropping the handle and before any subsequent renderer call
    /// that updates resources, including another group or an ordinary render.
    /// An explicit [`Resources::begin_frame`] is required; call [`Self::end_frame`] only
    /// after all groups and fallback draws have been submitted.
    ///
    /// # Panics
    ///
    /// Panics if no explicit resource frame is active.
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
        self.prepare_group(
            scenes,
            resources,
            device,
            queue,
            encoder,
            render_size,
            view,
            None,
            texture_bindings,
        )
    }

    /// Prepare HDR scenes for direct dual-radiance rendering in painter order.
    /// The two targets must be distinct full-size, single-sample RGBA16F 2D views.
    /// Returns `None` for unsupported devices/renderers or an ineligible first scene.
    /// The frame, group bounds and submission contract match [`Self::prepare_flat_group`].
    pub fn prepare_radiance_group<'a>(
        &'a mut self,
        scenes: &[&Scene],
        resources: &mut Resources,
        device: &'a Device,
        queue: &'a Queue,
        encoder: &'a mut CommandEncoder,
        render_size: &RenderSize,
        views: [&'a TextureView; 2],
        texture_bindings: &'a TextureBindings,
    ) -> Result<Option<PreparedFlatGroup<'a>>, RenderError> {
        if !self.hdr
            || device.limits().max_color_attachments < 2
            || device.limits().max_color_attachment_bytes_per_sample < 16
        {
            return Ok(None);
        }
        if views[0].texture() == views[1].texture()
            || views.iter().any(|view| {
                let texture = view.texture();
                texture.format() != wgpu::TextureFormat::Rgba16Float
                    || texture.sample_count() != 1
                    || texture.dimension() != wgpu::TextureDimension::D2
                    || texture.depth_or_array_layers() != 1
                    || texture.width() != render_size.width
                    || texture.height() != render_size.height
                    || !texture
                        .usage()
                        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
            })
        {
            return Err(RenderError::InvalidRadianceTarget);
        }
        if self.flat_group_prefix(scenes, device) == 0 {
            return Ok(None);
        }
        self.programs.ensure_radiance_pipelines(device);
        self.prepare_group(
            scenes,
            resources,
            device,
            queue,
            encoder,
            render_size,
            views[0],
            Some(views[1]),
            texture_bindings,
        )
    }

    fn prepare_group<'a>(
        &'a mut self,
        scenes: &[&Scene],
        resources: &mut Resources,
        device: &'a Device,
        queue: &'a Queue,
        encoder: &'a mut CommandEncoder,
        render_size: &RenderSize,
        view: &'a TextureView,
        radiance_view: Option<&'a TextureView>,
        texture_bindings: &'a TextureBindings,
    ) -> Result<Option<PreparedFlatGroup<'a>>, RenderError> {
        assert!(
            resources.frame_active,
            "Flat groups require an explicit resource frame"
        );
        if self.flat_group_prefix(scenes, device) == 0 {
            return Ok(None);
        }
        if self.hdr
            && (render_size.width == 0
                || render_size.height == 0
                || view.texture().format() != wgpu::TextureFormat::Rgba16Float
                || view.texture().sample_count() != 1)
        {
            return Err(RenderError::InvalidHdrTarget);
        }
        let started = self.programs.diagnostics.start_timer();
        #[cfg(feature = "text")]
        self.prepare_glyphs(resources, device, queue, encoder, texture_bindings);
        // Glyph preparation can change the shared LUT cache, so recompute the budget.
        let count = self.flat_group_prefix(scenes, device);
        if count == 0 {
            self.programs
                .diagnostics
                .end_timer(started, |cpu| &mut cpu.render);
            return Ok(None);
        }
        #[cfg(feature = "text")]
        {
            resources.cache_diagnostics.render_calls += count as u64 - 1;
        }
        let mut storage = core::mem::take(&mut self.programs.flat_group);
        storage.clear();
        let result = self.build_flat_group(
            &mut storage,
            &scenes[..count],
            resources,
            device,
            queue,
            render_size,
            view,
            radiance_view,
            texture_bindings,
        );
        self.programs
            .diagnostics
            .end_timer(started, |cpu| &mut cpu.render);
        if let Err(error) = result {
            storage.clear();
            self.programs.flat_group = storage;
            self.gradient_cache.maintain();
            self.programs
                .diagnostics
                .update(|report| report.render_errors += 1);
            return Err(error);
        }
        Ok(Some(PreparedFlatGroup {
            renderer: self,
            storage,
            device,
            queue,
            encoder,
            view,
            radiance_view,
            texture_bindings,
        }))
    }

    fn flat_group_prefix(&self, scenes: &[&Scene], device: &Device) -> usize {
        let dimension = self.programs.resources.resource_texture_dimension_2d as usize;
        let byte_limit = MAX_GROUP_BYTES
            .min(usize::try_from(device.limits().max_buffer_size).unwrap_or(usize::MAX))
            .min(dimension * dimension * self.gradient_cache.bytes_per_texel() as usize);
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
            let alpha_bytes = AlphaData::segment_len(&scene.strip_storage.borrow().alphas);
            let gradient_count = scene
                .encoded_paints
                .iter()
                .filter(|paint| matches!(paint, EncodedPaint::Gradient(_)))
                .count();
            bytes = bytes
                .saturating_add(alpha_bytes)
                .saturating_add(scene.encoded_paints.len() * size_of::<GpuEncodedPaint>())
                .saturating_add(scene_strips * size_of::<GpuStrip>())
                .saturating_add(
                    gradient_count
                        * MAX_GRADIENT_LUT_SIZE
                        * self.gradient_cache.bytes_per_texel() as usize,
                );
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
        radiance_view: Option<&TextureView>,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        storage
            .draws
            .resize_with(storage.draws.len().max(scenes.len()), Draw::default);
        assert!(scenes.len() <= MAX_FLAT_GROUP_SCENES);
        let alpha_storage: [_; MAX_FLAT_GROUP_SCENES] = core::array::from_fn(|index| {
            scenes.get(index).map(|scene| scene.strip_storage.borrow())
        });
        let alpha_segments = alpha_storage.each_ref().map(|storage| {
            storage
                .as_ref()
                .map_or(&[][..], |storage| storage.alphas.as_slice())
        });
        let mut alpha_bytes = 0_usize;
        let mut paint_texels = 0;
        for (index, scene) in scenes.iter().enumerate() {
            if let Some(radiance_view) = radiance_view {
                for paint in &scene.encoded_paints {
                    if let EncodedPaint::Image(image) = paint
                        && let ImageSource::ExternalTexture { id, .. } = &image.source
                        && texture_bindings
                            .get(*id)
                            .is_some_and(|view| view.texture() == radiance_view.texture())
                    {
                        return Err(RenderError::TextureFeedbackLoop(*id));
                    }
                }
            }
            let started = self.programs.diagnostics.start_timer();
            let result = self.prepare_gpu_encoded_paints(
                &scene.encoded_paints,
                &resources.image_cache,
                texture_bindings,
                view.texture(),
            );
            self.programs
                .diagnostics
                .end_timer(started, |cpu| &mut cpu.paint_preparation);
            result?;
            for offset in &mut self.paint_idxs {
                *offset += paint_texels;
            }
            paint_texels = *self.paint_idxs.last().unwrap();
            storage.paints.append(&mut self.encoded_paints);
            let started = self.programs.diagnostics.start_timer();
            let strip_storage = scene.strip_storage.borrow();
            let alpha_offset = u32::try_from(alpha_bytes / usize::from(Tile::HEIGHT)).unwrap();
            alpha_bytes = alpha_bytes
                .checked_add(AlphaData::segment_len(alpha_segments[index]))
                .expect("group alpha size overflow");
            self.programs.diagnostics.update(|report| {
                let stats = &mut report.scene_alphas;
                let capacity = strip_storage.alphas.capacity() as u64;
                stats.observations += 1;
                stats.active_bytes += strip_storage.alphas.len() as u64;
                stats.capacity_before_bytes += capacity;
                stats.capacity_after_bytes += capacity;
                stats.max_capacity_after_bytes = stats.max_capacity_after_bytes.max(capacity);
            });
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
                .with_image_cache(&resources.image_cache)
                .with_linear_color(self.hdr);
            let mut builder =
                DrawBuilder::new(&mut storage.draws[index], &mut storage.buffers, &mut state);
            for node in &scene.recorder.nodes {
                for draw in node.draws_in(&scene.recorder.draws) {
                    builder.push_draw(draw, &strip_storage, resolver);
                }
            }
            for strip in &mut storage.buffers.strips[first..] {
                strip.offset_alpha_columns(alpha_offset);
            }
            storage.ranges.push(
                (first * size_of::<GpuStrip>()) as u64
                    ..(storage.buffers.strips.len() * size_of::<GpuStrip>()) as u64,
            );
            self.programs
                .diagnostics
                .end_timer(started, |cpu| &mut cpu.scheduling);
        }
        let started = self.programs.diagnostics.start_timer();
        self.programs.prepare(
            device,
            queue,
            &mut self.gradient_cache,
            &storage.paints,
            AlphaData::Segmented(&alpha_segments[..scenes.len()]),
            render_size,
            &[paint_texels],
            &FilterContext::default(),
        );
        self.programs
            .update_config_color(queue, u32::from(self.hdr));
        let bytes = bytemuck::cast_slice(&storage.buffers.strips);
        if !bytes.is_empty() {
            let size = bytes.len() as u64;
            if storage
                .strip_buffer
                .as_ref()
                .is_none_or(|buffer| buffer.size() < size)
            {
                let capacity = size
                    .next_power_of_two()
                    .min(device.limits().max_buffer_size);
                let buffer = Programs::create_strips_buffer(device, capacity);
                self.programs
                    .diagnostics
                    .buffer(BufferKind::Strip, &buffer, 0);
                storage.strip_buffer = Some(buffer);
            }
            queue.write_buffer(storage.strip_buffer.as_ref().unwrap(), 0, bytes);
            self.programs
                .diagnostics
                .buffer_write(BufferKind::Strip, size);
        }
        self.programs
            .diagnostics
            .end_timer(started, |cpu| &mut cpu.resource_preparation);
        Ok(())
    }
}

impl Programs {
    fn ensure_radiance_pipelines(&mut self, device: &Device) {
        if self.radiance_pipelines.is_some() {
            return;
        }
        let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Radiance Strip Vertex"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::RENDER.into()),
        });
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Radiance Strip Fragment"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::RENDER_MRT.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Radiance Strip Layout"),
            bind_group_layouts: &[
                Some(&self.strip_bind_group_layout),
                Some(&self.external_texture_bind_group_layout),
                Some(&self.encoded_paints_bind_group_layout),
                Some(&self.gradient_bind_group_layout),
            ],
            immediate_size: 0,
        });
        self.radiance_pipelines = Some(["fs_lit", "fs_unlit"].map(|entry| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("vs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: size_of::<GpuStrip>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &GpuStrip::vertex_attributes(),
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some(entry),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &core::array::from_fn::<_, 2, _>(|_| {
                        Some(ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: ColorWrites::ALL,
                        })
                    }),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        }));
    }
}
