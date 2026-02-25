// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmarks for the gradient ramp cache.

#![allow(missing_docs, reason = "criterion macros generate undocumented items")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vello_common::color::{ColorSpaceTag, DynamicColor, HueDirection};
use vello_common::encode::{EncodeExt, EncodedGradient, EncodedPaint};
use vello_common::kurbo::{Affine, Point};
use vello_common::peniko::{Color, ColorStop, ColorStops, Gradient, LinearGradientPosition};
use vello_hybrid::gradient_cache::GradientRampCache;

use vello_common::fearless_simd::Level;

fn create_gradient(offset: f32) -> Gradient {
    Gradient {
        kind: LinearGradientPosition {
            start: Point::new(0.0, 0.0),
            end: Point::new(100.0, 0.0),
        }
        .into(),
        stops: ColorStops(
            vec![
                ColorStop {
                    offset: 0.0,
                    color: DynamicColor::from_alpha_color(Color::from_rgb8(255, 0, 0)),
                },
                ColorStop {
                    offset,
                    color: DynamicColor::from_alpha_color(Color::from_rgb8(0, 255, 0)),
                },
                ColorStop {
                    offset: 1.0,
                    color: DynamicColor::from_alpha_color(Color::from_rgb8(0, 0, 255)),
                },
            ]
            .into(),
        ),
        interpolation_cs: ColorSpaceTag::Srgb,
        hue_direction: HueDirection::Shorter,
        ..Default::default()
    }
}

fn create_encoded_gradient(gradient: Gradient) -> EncodedGradient {
    let mut encoded_paints = vec![];
    gradient.encode_into(&mut encoded_paints, Affine::IDENTITY, None);
    match encoded_paints.into_iter().last().unwrap() {
        EncodedPaint::Gradient(encoded_gradient) => encoded_gradient,
        _ => panic!("Expected a gradient paint"),
    }
}

/// Pre-generate a set of unique encoded gradients.
fn make_gradients(count: usize) -> Vec<EncodedGradient> {
    (0..count)
        .map(|i| {
            let offset = (i as f32 + 1.0) / (count as f32 + 1.0);
            create_encoded_gradient(create_gradient(offset))
        })
        .collect()
}

fn bench_cache_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hit");
    for &n in &[10, 100, 1000] {
        let gradients = make_gradients(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &gradients, |b, grads| {
            let mut cache = GradientRampCache::new(Level::baseline());
            // Populate the cache first.
            for g in grads {
                cache.get_or_create_ramp(g);
            }
            cache.maintain();
            b.iter(|| {
                for g in grads {
                    cache.get_or_create_ramp(g);
                }
            });
        });
    }
    group.finish();
}

fn bench_cache_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_miss");
    for &n in &[10, 100, 1000] {
        let gradients = make_gradients(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &gradients, |b, grads| {
            b.iter(|| {
                let mut cache = GradientRampCache::new(Level::baseline());
                for g in grads {
                    cache.get_or_create_ramp(g);
                }
            });
        });
    }
    group.finish();
}

fn bench_maintain_no_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("maintain_no_eviction");
    for &n in &[10, 100, 1000] {
        let gradients = make_gradients(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &gradients, |b, grads| {
            let mut cache = GradientRampCache::new(Level::baseline());
            for g in grads {
                cache.get_or_create_ramp(g);
            }
            b.iter(|| {
                // Touch all entries so nothing is stale.
                for g in grads {
                    cache.get_or_create_ramp(g);
                }
                cache.maintain();
            });
        });
    }
    group.finish();
}

fn bench_maintain_full_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("maintain_full_eviction");
    for &n in &[10, 100, 1000] {
        let gradients = make_gradients(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &gradients, |b, grads| {
            b.iter(|| {
                let mut cache = GradientRampCache::new(Level::baseline());
                for g in grads {
                    cache.get_or_create_ramp(g);
                }
                cache.maintain();
                // All entries are now stale (no lookups this frame).
                cache.maintain();
            });
        });
    }
    group.finish();
}

fn bench_frame_static(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_static");
    for &n in &[10, 100, 1000] {
        let gradients = make_gradients(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &gradients, |b, grads| {
            let mut cache = GradientRampCache::new(Level::baseline());
            // Warm up: first frame populates the cache.
            for g in grads {
                cache.get_or_create_ramp(g);
            }
            cache.maintain();
            b.iter(|| {
                for g in grads {
                    cache.get_or_create_ramp(g);
                }
                cache.maintain();
            });
        });
    }
    group.finish();
}

fn bench_frame_dynamic(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_dynamic");
    for &n in &[10, 100, 1000] {
        let gradients = make_gradients(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &gradients, |b, grads| {
            b.iter(|| {
                let mut cache = GradientRampCache::new(Level::baseline());
                for g in grads {
                    cache.get_or_create_ramp(g);
                }
                cache.maintain();
                // All stale — full eviction.
                cache.maintain();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cache_hit,
    bench_cache_miss,
    bench_maintain_no_eviction,
    bench_maintain_full_eviction,
    bench_frame_static,
    bench_frame_dynamic,
);
criterion_main!(benches);
