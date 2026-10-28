# TODO: linear HDR rendering in vello_gpu

Goal: let the engine render world, lighting, glow and UI as one Vello scene, so its custom
pipeline (`world_batch.wgsl` per-batch decode + `composition.wesl`) shrinks to a single
exposure/tone-map/dither/sRGB present pass.

An experiment (engine commit history, Oct 2026) did this on top of `Scene::append` and
showed why it isn't enough today: values clamp at 1 and blending happens in sRGB, so lights
saturate when added, can't brighten past the base colour, and a second ColorDodge light pass
was needed to fake illumination above 1. Layer/filter intermediates are already `Rgba16Float`
(029b88d0), so the remaining limits are below.

## 1. Float colour payloads

Colours are quantized to premultiplied RGBA8 at record time and unpacked with
`unpack4x8unorm`, so nothing above 1 can be recorded.

- CPU encode: `vello_common` `Paint::Solid(PremulColor)` / `paint.rs` (`as_premul_rgba8`),
  `vello_gpu/src/render/common.rs` (`to_rgba8().to_u32()`), `filter.rs` (flood and drop
  shadow colours), `render/wgpu/mod.rs` (rect fast path and clear colour).
- Shaders: `render.wesl` (solid payload), `clear.wesl`, `filter.wesl` (flood),
  `filters/drop_shadow.wesl`, `helpers/blurred_rounded_rect.wesl`,
  `helpers/external_texture.wesl` (tint).
- Pack as two `u32` of f16 pairs (`pack2x16float` / `unpack2x16float`). This doubles the
  colour field in strip/draw data; keep it behind a renderer setting so SDR users keep the
  smaller layout.

## 2. Unclamped blending

`helpers/blend.wesl` clamps:

- `COMPOSE_PLUS`: `min(vec4(1.0), ...)` on colour.
- Final composite: `min(as_fa + ab_fb, 1.0)` on alpha (keep this, alpha is coverage).
- Colour dodge / burn: `min(1.0, ...)`.

In HDR mode clamp alpha only. Multiply, Screen and the separable mixes need checking for
assumptions of `cs, cb <= 1` (Screen in particular is only meaningful in 0..1; document it).

## 3. Linear working space

Vello premultiplies, blends and interpolates gradients on sRGB-encoded values.

- Convert paint colours to linear at encode time (CPU side, before packing).
- Gradient LUTs (`gradient_cache.rs`, already `Rgba32Float`): build ramps in linear, and
  respect the gradient's interpolation colour space instead of the fixed `Srgb` tag.
- Image atlas and glyph atlas stay `Rgba8Unorm` for memory; decode sRGB to linear on sample
  (an `Rgba8UnormSrgb` view where possible, otherwise in `external_texture.wesl` and the image
  sampling path).
- Output: the target receives linear premultiplied values; the caller tone-maps and encodes.
- Mirror in `render/webgl/mod.rs` (falls back to RGBA8 intermediates on some WebGL2
  devices; HDR there may need to stay unsupported).

## 4. Performance in the engine's single-scene frame

With 1–3 done the frame becomes: background, world, a Multiply layer of `ambient + lights`
(Plus), a Plus layer of glow, UI, then one tone-map pass. Points to keep it fast:

- Lights and glow currently render at reduced resolution. `Scene::append` requires equal
  sizes, so in one scene they'd go full resolution. Instead render the light scene and the
  glow scene once into small `Rgba16Float` textures and draw them into the frame scene as
  external-texture image paints (`TextureBindings`) inside the Multiply / Plus layers. One
  light rasterization, upscaled by the sampler, no second light pass.
- Unlit content: lit and unlit runs interleave in painter's order. With the light map as an
  image paint, each lit run can sit in its own Multiply-by-light-map layer cheaply; measure
  the per-layer cost before choosing this over the current emission target.
- HDR intensity per batch (`scene_with_intensity`) becomes a plain colour scale at encode
  time once colours can exceed 1, removing the batch split for intensity.
- Check `prepare_flat_group` still groups scenes after the payload change
  (`MAX_GROUP_DRAWS`, encoded paint size) and that the f16 payload doesn't regress the perf
  app's baseline.

## 5. What stays outside Vello

- The present pass is irreducible: exposure, tone mapping, dither and sRGB encode into the
  8-bit swapchain have no blend-mode equivalent. Moving it into `vello_gpu` as an output
  transform only relocates the shader.
- Colour grading (contrast, saturation; `color_grading.wesl`) needs either that pass or a
  filter layer. `vello_common` defines `ColorMatrix` / `Contrast`; check `vello_gpu`
  implements them before relying on it. A filter layer is still a full-screen pass.

## Done

- `Scene::append` (5514eb1d): merge separately recorded scenes into one render.

# TODO: textured strokes

Paint a stroke with an image mapped along the path: u is the arc length from the path's
start, v runs across the stroke's width. Art tools call it a pattern or art brush. Wanted
for a creature's tongue in the engine's twin_stick_shooter, and for trails, rope and cables.

- `PaintType::Image` today sits in paint space; a stroke painted with it shows the image
  through the stroke's shape, not along it.
- u starts at the path's start and is never normalized to its length, so a growing path
  uncovers more texture at its end and nothing already drawn moves. Repeat or clamp along
  u, per the image's extend.
- Joins and caps: u carries on round joins; v is clamped across the width.
- A possible route, unverified: arc length is known per segment at flattening; carry it per strip into the fine stage and
  sample the image there instead of in paint space.
- Until then the engine draws one image-painted quad per segment, transformed to the
  segment.

# TODO: upstream one submit per frame

ab30af84 makes renders inside `begin_frame` / `end_frame` append to the shared uploads,
shifting alpha columns, paint indices, filter offsets and strip ranges, so the engine
submits once per frame instead of ~19 times. Upstream can't take it alone: frames
(2931aea7) and flat groups (6aef7894) are fork-only.

- Measure in the engine's `apps/perf` (gpu_timing off), before and after ab30af84 + engine
  f5e1663f: frame time and `cpu_world_ms`.
- If it's a clear win, open an issue on linebender/vello with the numbers: many small renders
  per frame, each forced into its own submit by the shared uploads; propose explicit
  frames plus appended uploads.
- If maintainers are interested, follow with a PR against upstream `vello_gpu` as it stands.
- Until then keep it in the fork.
