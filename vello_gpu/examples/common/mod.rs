// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::{fs::File, io::BufWriter, path::Path};
use vello_common::{
    peniko::ImageAlphaType,
    pixmap::{PixelMetadata, Pixmap},
};
use vello_gpu::{RenderSize, RenderTargetConfig, Renderer, Scene};

pub(super) async fn render_to_png(scene: &Scene, output: impl AsRef<Path>) {
    let width = u32::from(scene.width());
    let height = u32::from(scene.height());
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find a GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("Failed to create device");
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("PNG Output"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let config = RenderTargetConfig {
        format: texture.format(),
        width,
        height,
    };
    let (mut renderer, mut resources) = Renderer::new(&device, &config);
    let render_size = RenderSize { width, height };
    let depth = Renderer::create_depth_texture_view(&device, &render_size);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    renderer
        .render(
            scene,
            &mut resources,
            &device,
            &queue,
            &mut encoder,
            &render_size,
            &view,
            Some(&depth),
            &vello_gpu::TextureBindings::new(),
        )
        .expect("Failed to render scene");

    let bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PNG Readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("Failed to map PNG readback");
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut pixels =
        Vec::with_capacity(usize::from(scene.width()) * usize::from(scene.height()) * 4);
    {
        let mapped = readback.slice(..).get_mapped_range();
        for row in mapped.chunks_exact(bytes_per_row as usize) {
            pixels.extend_from_slice(&row[..width as usize * 4]);
        }
    }
    readback.unmap();
    let mut png = png::Encoder::new(BufWriter::new(File::create(output).unwrap()), width, height);
    png.set_color(png::ColorType::Rgba);
    png.set_depth(png::BitDepth::Eight);
    let pixmap = Pixmap::from_parts(
        pixels,
        scene.width(),
        scene.height(),
        PixelMetadata::default(),
    );
    png.write_header()
        .unwrap()
        .write_image_data(&pixmap.take_rgba8(ImageAlphaType::Alpha))
        .unwrap();
}
