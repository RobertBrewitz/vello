// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render an SVG to a PNG without a window or display.
//!
//! Arguments: input SVG and output PNG.

mod common;

use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_gpu::{DimensionConstraints, Scene};

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let mut args = std::env::args().skip(1);
    let svg_filename = args.next().expect("svg filename is first arg");
    let output_filename = args.next().expect("output filename is second arg");
    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let render_scale = 5.0;
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let constraints = DimensionConstraints::default();
    let svg_width = parsed.size.width * render_scale;
    let svg_height = parsed.size.height * render_scale;
    let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);
    let width = DimensionConstraints::convert_dimension(width);
    let height = DimensionConstraints::convert_dimension(height);

    let mut scene = Scene::new(width, height);
    render_svg(&mut scene, &parsed.items, Affine::scale(render_scale));
    common::render_to_png(&scene, output_filename).await;
}

fn render_svg(ctx: &mut Scene, items: &[Item], transform: Affine) {
    ctx.set_transform(transform);
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color);
                ctx.fill_path(&fill_item.path);
            }
            Item::Stroke(stroke_item) => {
                ctx.set_stroke(Stroke::new(stroke_item.width));
                ctx.set_paint(stroke_item.color);
                ctx.stroke_path(&stroke_item.path);
            }
            Item::Group(group_item) => {
                render_svg(ctx, &group_item.children, transform * group_item.affine);
                ctx.set_transform(transform);
            }
        }
    }
}
