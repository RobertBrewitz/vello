// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use super::Tile;

pub(super) struct InputOrder {
    runs: u64,
    row_runs: u64,
    location_runs: u64,
}

pub(super) fn input_order(tiles: &[Tile]) -> InputOrder {
    let initial = u64::from(!tiles.is_empty());
    let mut order = InputOrder {
        runs: initial,
        row_runs: initial,
        location_runs: initial,
    };
    for pair in tiles.windows(2) {
        order.runs += u64::from(pair[0] > pair[1]);
        order.row_runs += u64::from(!pair[0].same_row(&pair[1]));
        order.location_runs += u64::from(!pair[0].same_loc(&pair[1]));
    }
    order
}

struct Sample {
    started: Instant,
    calls: u64,
    counts: [u64; 18],
    tiles: [u64; 18],
    sort_ns: [u128; 18],
    max_tiles: usize,
    runs: [u64; 18],
    row_runs: [u64; 18],
    location_runs: [u64; 18],
    unique_rows: [u64; 18],
    unique_locations: [u64; 18],
    sorted_paths: [u64; 18],
}

impl Sample {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            calls: 0,
            counts: [0; 18],
            tiles: [0; 18],
            sort_ns: [0; 18],
            max_tiles: 0,
            runs: [0; 18],
            row_runs: [0; 18],
            location_runs: [0; 18],
            unique_rows: [0; 18],
            unique_locations: [0; 18],
            sorted_paths: [0; 18],
        }
    }
}

std::thread_local! {
    static SAMPLE: RefCell<Sample> = RefCell::new(Sample::new());
}

pub(super) fn record(sorted: &[Tile], order: InputOrder, elapsed: Duration) {
    let tiles = sorted.len();
    let output = input_order(sorted);
    SAMPLE.with_borrow_mut(|sample| {
        let bucket = if tiles == 0 {
            0
        } else {
            (tiles.ilog2() as usize + 1).min(17)
        };
        sample.calls += 1;
        sample.counts[bucket] += 1;
        sample.tiles[bucket] += tiles as u64;
        sample.sort_ns[bucket] += elapsed.as_nanos();
        sample.max_tiles = sample.max_tiles.max(tiles);
        sample.runs[bucket] += order.runs;
        sample.row_runs[bucket] += order.row_runs;
        sample.location_runs[bucket] += order.location_runs;
        sample.unique_rows[bucket] += output.row_runs;
        sample.unique_locations[bucket] += output.location_runs;
        sample.sorted_paths[bucket] += u64::from(order.runs <= 1);
        // Avoid an additional clock query for every small path.
        if sample.calls % 4096 == 0 && sample.started.elapsed() >= Duration::from_secs(1) {
            log::debug!(target: "vello_tiles",
                "Tile sort histogram (per-thread; diagnostic timing overhead; buckets: 0, 1, 2-3, 4-7, ..., 32768-65535, >=65536); calls={} elapsed_ms={} counts={:?} tiles={:?} sort_ns={:?} max_tiles={} runs={:?} row_runs={:?} location_runs={:?} unique_rows={:?} unique_locations={:?} sorted_paths={:?}",
                sample.calls, sample.started.elapsed().as_millis(), sample.counts,
                sample.tiles, sample.sort_ns, sample.max_tiles, sample.runs, sample.row_runs,
                sample.location_runs, sample.unique_rows, sample.unique_locations, sample.sorted_paths);
            *sample = Sample::new();
        }
    });
}
