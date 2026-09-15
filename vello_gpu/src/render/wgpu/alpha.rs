// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{Diagnostics, UploadKind};
use wgpu::{Extent3d, Origin3d, Queue, Texture};

const BYTES_PER_TEXEL: usize = 16;

#[derive(Clone, Copy)]
pub(super) enum AlphaData<'a> {
    Contiguous(&'a [u8]),
    Segmented(&'a [&'a [u8]]),
}

impl AlphaData<'_> {
    pub(super) fn len(self) -> usize {
        match self {
            Self::Contiguous(data) => data.len(),
            Self::Segmented(segments) => segments.iter().fold(0_usize, |len, segment| {
                len.checked_add(Self::segment_len(segment))
                    .expect("alpha data size overflow")
            }),
        }
    }

    pub(super) fn segment_len(segment: &[u8]) -> usize {
        segment
            .len()
            .div_ceil(BYTES_PER_TEXEL)
            .checked_mul(BYTES_PER_TEXEL)
            .expect("alpha segment size overflow")
    }
}

pub(super) fn upload_segments(
    queue: &Queue,
    texture: &Texture,
    segments: &[&[u8]],
    diagnostics: &mut Diagnostics,
) {
    assert_eq!(texture.format().block_copy_size(None), Some(16));
    let mut texel_offset = 0_u32;
    for segment in segments {
        let full_bytes = segment.len() / BYTES_PER_TEXEL * BYTES_PER_TEXEL;
        upload_texels(
            queue,
            texture,
            &segment[..full_bytes],
            &mut texel_offset,
            diagnostics,
        );
        if full_bytes != segment.len() {
            // Segment bases must be texel-aligned even when coverage ends inside a texel.
            let mut tail = [0; BYTES_PER_TEXEL];
            tail[..segment.len() - full_bytes].copy_from_slice(&segment[full_bytes..]);
            upload_texels(queue, texture, &tail, &mut texel_offset, diagnostics);
        }
    }
}

fn upload_texels(
    queue: &Queue,
    texture: &Texture,
    mut data: &[u8],
    texel_offset: &mut u32,
    diagnostics: &mut Diagnostics,
) {
    let stride = texture.width();
    while !data.is_empty() {
        let remaining = u32::try_from(data.len() / BYTES_PER_TEXEL).unwrap();
        let x = *texel_offset % stride;
        let y = *texel_offset / stride;
        let (width, height) = if x == 0 && remaining >= stride {
            (stride, remaining / stride)
        } else {
            ((stride - x).min(remaining), 1)
        };
        assert!(
            y.checked_add(height)
                .is_some_and(|end| end <= texture.height())
        );
        let texels = width.checked_mul(height).unwrap();
        let bytes = texels as usize * BYTES_PER_TEXEL;
        // Queue writes copy before returning; no scene or worker buffer is retained by wgpu.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &data[..bytes],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * BYTES_PER_TEXEL as u32),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        diagnostics.upload(UploadKind::Alpha, bytes as u64);
        *texel_offset = texel_offset.checked_add(texels).unwrap();
        data = &data[bytes..];
    }
}
