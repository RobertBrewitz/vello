// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{sync::Arc, vec::Vec};
use core::{
    sync::atomic::{AtomicU8, Ordering},
    time::Duration,
};
use web_time::Instant;
use wgpu::{Buffer, CommandEncoder, Device, QuerySet, Queue, RenderPassTimestampWrites, Texture};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum RenderPassKind {
    RootClear,
    RootStrip,
    AtlasStrip,
    LayerStrip,
    Blend,
    BlendCopy,
    FilterCopy,
    Filter,
    LayerClear,
    AtlasClear,
    RootRadiance,
}

impl RenderPassKind {
    pub const COUNT: usize = 11;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PassDiagnostics {
    pub count: u64,
    pub timed_count: u64,
    pub gpu_nanoseconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum RenderBufferKind {
    Strip,
    Blend,
    Copy,
    Filter,
    Clear,
    Config,
}

impl RenderBufferKind {
    pub const COUNT: usize = 6;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BufferDiagnostics {
    pub creations: u64,
    pub created_bytes: u64,
    pub largest_creation_bytes: u64,
    pub initialized_bytes: u64,
    pub queue_writes: u64,
    pub queue_write_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum RenderBindGroupKind {
    Image,
    Strip,
    Blend,
    Filter,
    Copy,
    Paint,
    Gradient,
}

impl RenderBindGroupKind {
    pub const COUNT: usize = 7;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum RenderUploadKind {
    Alpha,
    Paint,
    Filter,
    Gradient,
}

impl RenderUploadKind {
    pub const COUNT: usize = 4;
}

/// Byte sums are observations per scene/group preparation, not unique resident allocations.
/// Upload bytes are requested texel bytes, excluding opaque wgpu/driver staging padding.
#[derive(Clone, Copy, Debug, Default)]
pub struct UploadDiagnostics {
    pub observations: u64,
    pub active_bytes: u64,
    pub capacity_bytes: u64,
    pub max_capacity_bytes: u64,
    pub uploaded_bytes: u64,
    pub queue_writes: u64,
}

/// Capacity sums count repeated renders of the same scene repeatedly. Atlas scenes are separate.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlphaDiagnostics {
    pub observations: u64,
    pub active_bytes: u64,
    pub capacity_before_bytes: u64,
    pub capacity_after_bytes: u64,
    pub capacity_growth_bytes: u64,
    pub max_capacity_after_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderCpuTimings {
    /// Renderer CPU time, including grouped preparation/draws and nested atlas work.
    /// Caller composition between grouped draws is excluded.
    pub render: Duration,
    pub paint_preparation: Duration,
    pub scheduling: Duration,
    pub resource_preparation: Duration,
    pub pass_recording: Duration,
    /// Only internally owned atlas encoders/submissions; caller finish/submit is not observable.
    pub atlas_finish: Duration,
    pub atlas_submit: Duration,
}

/// One explicitly bounded capture of this renderer's wgpu calls, excluding constructor and
/// diagnostic resources. Native allocations, caller passes/submissions and custom AtlasWriter
/// internals are not observable. No counts represent live GPU allocations or measured bandwidth.
#[derive(Clone, Debug, Default)]
pub struct RenderDiagnostics {
    pub render_calls: u64,
    pub render_errors: u64,
    pub atlas_render_calls: u64,
    pub scheduler_rounds: u64,
    pub internal_encoders: u64,
    pub internal_submissions: u64,
    pub passes: [PassDiagnostics; RenderPassKind::COUNT],
    pub buffers: [BufferDiagnostics; RenderBufferKind::COUNT],
    pub bind_group_creations: [u64; RenderBindGroupKind::COUNT],
    pub uploads: [UploadDiagnostics; RenderUploadKind::COUNT],
    pub scene_alphas: AlphaDiagnostics,
    pub atlas_alphas: AlphaDiagnostics,
    pub atlas_clear_writes: u64,
    pub atlas_clear_bytes: u64,
    /// Image writer invocations and destination area; not necessarily CPU-to-GPU uploads.
    pub atlas_image_writes: u64,
    pub atlas_image_pixels: u64,
    pub max_intermediate_textures: u64,
    pub max_intermediate_capacity_bytes: u64,
    pub cpu: RenderCpuTimings,
    pub gpu_timestamps_enabled: bool,
    /// Includes disabled/unsupported timestamps and passes beyond the query budget.
    pub gpu_untimed_passes: u64,
    pub gpu_readback_failed: bool,
}

#[derive(Debug, Default)]
pub(super) struct Diagnostics {
    active: bool,
    report: Option<RenderDiagnostics>,
    gpu: Option<GpuCapture>,
}

#[derive(Debug)]
struct GpuCapture {
    queries: QuerySet,
    resolve: Buffer,
    readback: Buffer,
    kinds: Vec<RenderPassKind>,
    max_passes: u32,
    period: f64,
    ready: Arc<AtomicU8>,
}

impl Diagnostics {
    pub fn begin(&mut self, device: &Device, queue: &Queue, max_gpu_passes: u32) -> bool {
        if self.report.is_some() {
            return false;
        }
        let max_passes = max_gpu_passes.min(wgpu::QUERY_SET_MAX_QUERIES / 2);
        let enabled = max_passes > 0 && device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        self.gpu = enabled.then(|| {
            let size = u64::from(max_passes) * 2 * wgpu::QUERY_SIZE as u64;
            GpuCapture {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("Vello diagnostics timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: max_passes * 2,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Vello diagnostics resolve"),
                    size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Vello diagnostics readback"),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                kinds: Vec::with_capacity(max_passes as usize),
                max_passes,
                period: f64::from(queue.get_timestamp_period()),
                ready: Arc::new(AtomicU8::new(0)),
            }
        });
        self.report = Some(RenderDiagnostics {
            gpu_timestamps_enabled: enabled,
            ..RenderDiagnostics::default()
        });
        self.active = true;
        true
    }

    pub fn finish(&mut self, encoder: &mut CommandEncoder) {
        if !self.active {
            return;
        }
        self.active = false;
        if let Some(gpu) = &self.gpu {
            let count = gpu.kinds.len() as u32 * 2;
            if count == 0 {
                gpu.ready.store(1, Ordering::Release);
                return;
            }
            let size = u64::from(count) * wgpu::QUERY_SIZE as u64;
            encoder.resolve_query_set(&gpu.queries, 0..count, &gpu.resolve, 0);
            encoder.copy_buffer_to_buffer(&gpu.resolve, 0, &gpu.readback, 0, size);
            let ready = gpu.ready.clone();
            encoder.map_buffer_on_submit(
                &gpu.readback,
                wgpu::MapMode::Read,
                0..size,
                move |result| {
                    ready.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release);
                },
            );
        }
    }

    pub fn take(&mut self) -> Option<RenderDiagnostics> {
        if self.active || self.report.is_none() {
            return None;
        }
        if let Some(gpu) = &self.gpu {
            let status = gpu.ready.load(Ordering::Acquire);
            if status == 0 {
                return None;
            }
            let report = self.report.as_mut().unwrap();
            report.gpu_readback_failed = status == 2;
            if status == 1 && !gpu.kinds.is_empty() {
                let size = gpu.kinds.len() as u64 * 2 * wgpu::QUERY_SIZE as u64;
                let data = gpu.readback.slice(0..size).get_mapped_range();
                for (kind, bytes) in gpu.kinds.iter().zip(data.chunks_exact(16)) {
                    let start = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
                    let end = u64::from_ne_bytes(bytes[8..].try_into().unwrap());
                    let pass = &mut report.passes[*kind as usize];
                    pass.timed_count += 1;
                    pass.gpu_nanoseconds += end.wrapping_sub(start) as f64 * gpu.period;
                }
                drop(data);
                gpu.readback.unmap();
            }
        }
        self.gpu = None;
        self.report.take()
    }

    pub fn update(&mut self, update: impl FnOnce(&mut RenderDiagnostics)) {
        if self.active {
            update(self.report.as_mut().unwrap());
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn start_timer(&self) -> Option<Instant> {
        self.active.then(Instant::now)
    }

    pub fn end_timer(
        &mut self,
        start: Option<Instant>,
        field: impl FnOnce(&mut RenderCpuTimings) -> &mut Duration,
    ) {
        if let Some(start) = start {
            self.update(|report| *field(&mut report.cpu) += start.elapsed());
        }
    }

    pub fn pass(&mut self, kind: RenderPassKind) -> Option<RenderPassTimestampWrites<'_>> {
        if !self.active {
            return None;
        }
        let report = self.report.as_mut().unwrap();
        report.passes[kind as usize].count += 1;
        let Some(gpu) = &mut self.gpu else {
            report.gpu_untimed_passes += 1;
            return None;
        };
        if gpu.kinds.len() == gpu.max_passes as usize {
            report.gpu_untimed_passes += 1;
            return None;
        }
        let index = gpu.kinds.len() as u32 * 2;
        gpu.kinds.push(kind);
        Some(RenderPassTimestampWrites {
            query_set: &gpu.queries,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        })
    }

    pub fn buffer(&mut self, kind: RenderBufferKind, buffer: &Buffer, initialized_bytes: u64) {
        self.update(|report| {
            let stats = &mut report.buffers[kind as usize];
            stats.creations += 1;
            stats.created_bytes += buffer.size();
            stats.largest_creation_bytes = stats.largest_creation_bytes.max(buffer.size());
            stats.initialized_bytes += initialized_bytes;
        });
    }

    pub fn buffer_write(&mut self, kind: RenderBufferKind, bytes: u64) {
        self.update(|report| {
            let stats = &mut report.buffers[kind as usize];
            stats.queue_writes += 1;
            stats.queue_write_bytes += bytes;
        });
    }

    pub fn bind_groups(&mut self, kind: RenderBindGroupKind, count: u64) {
        self.update(|report| report.bind_group_creations[kind as usize] += count);
    }

    pub fn data(&mut self, kind: RenderUploadKind, active: u64, texture: &Texture) {
        self.update(|report| {
            let capacity = u64::from(texture.width())
                * u64::from(texture.height())
                * u64::from(texture.format().block_copy_size(None).unwrap());
            let stats = &mut report.uploads[kind as usize];
            stats.observations += 1;
            stats.active_bytes += active;
            stats.capacity_bytes += capacity;
            stats.max_capacity_bytes = stats.max_capacity_bytes.max(capacity);
        });
    }

    pub fn upload(&mut self, kind: RenderUploadKind, bytes: u64) {
        self.update(|report| {
            let stats = &mut report.uploads[kind as usize];
            stats.uploaded_bytes += bytes;
            stats.queue_writes += 1;
        });
    }
}
