// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Benchmarks for tensor operations.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_matmul(_c: &mut Criterion) {
    todo!("Implement matmul benchmarks")
}

fn bench_softmax(_c: &mut Criterion) {
    todo!("Implement softmax benchmarks")
}

criterion_group!(benches, bench_matmul, bench_softmax);
criterion_main!(benches);
