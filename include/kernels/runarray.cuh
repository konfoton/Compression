#pragma once

#include <cuda_runtime.h>

__global__ void running_scan_blocks(const unsigned char* __restrict__ input,
									int size,
									int* __restrict__ d_running,
									int* __restrict__ d_blocks);

// Scan an array of size n that contains up to 1024 elements (one block) in-place.
__global__ void scan_block_sums_inplace(int* d_blocks, int n);

__global__ void add_block_offsets(int* __restrict__ d_running,
								  int size,
								  const int* __restrict__ d_blocks);

__global__ void final_run(unsigned char* input, int* h_index_array, unsigned char* symbols, int* counts, int n);

__global__ void differce_scan(int* __restrict__ scan, int* __restrict__ index, int size_whole, int n);

// Generic per-block inclusive scan over `scan` (in-place) producing per-block totals in `blocks`
__global__ void dec_scan_block_level(int* __restrict__ scan, int size, int* __restrict__ blocks);

// Make exclusive offsets from inclusive scan and original values: exclusive[i] = inclusive[i] - values[i]
__global__ void inclusive_to_exclusive(const int* __restrict__ inclusive,
									   const int* __restrict__ values,
									   int n,
									   int* __restrict__ exclusive);

// Mark run start flags: flags[offsets[i]] = 1 for i in [0, nRuns)
__global__ void mark_run_starts(const int* __restrict__ offsets,
								int nRuns,
								int total_len,
								int* __restrict__ flags);

// Map each position to symbol by run index (+1) array obtained from scanning flags
__global__ void map_symbols_from_runs(const int* __restrict__ runIndexPlus1,
									  const unsigned char* __restrict__ symbols,
									  int total_len,
									  unsigned char* __restrict__ out);