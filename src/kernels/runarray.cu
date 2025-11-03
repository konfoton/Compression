// Implementation of block-wise running scan and block offset addition
#include "kernels/runarray.cuh"

// Warp inclusive scan using shuffle ops
__device__ __forceinline__ int warp_inclusive_scan(int val) {
    unsigned mask = 0xffffffffu;
    int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

// Block inclusive scan for arbitrary blockDim.x (multiple warps)
template <int BLOCK_SIZE>
__device__ __forceinline__ int block_inclusive_scan(int val, int* shared) {
    // shared must be at least BLOCK_SIZE/32 ints for warp sums
    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x >> 5; // warp id

    // Intra-warp scan
    int scan = warp_inclusive_scan(val);

    // Write warp sum to shared by last lane
    if (lane == 31) shared[wid] = scan;
    __syncthreads();

    // Let warp 0 scan the warp sums
    int warp_offset = 0;
    if (wid == 0) {
        const int nWarps = (BLOCK_SIZE + 31) / 32;
        int x = (lane < nWarps) ? shared[lane] : 0;
        x = warp_inclusive_scan(x);
        if (lane < nWarps) shared[lane] = x;
    }
    __syncthreads();

    if (wid > 0) warp_offset = shared[wid - 1];

    return scan + warp_offset;
}

__global__ void running_scan_blocks(const unsigned char* __restrict__ input,
                                    int size,
                                    int* __restrict__ d_running,
                                    int* __restrict__ d_blocks) {
    extern __shared__ int s_mem[]; // at least numWarps ints
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute predicate: 1 if new run starts at idx, else 0.
    int pred = 0;
    if (idx < size) {
        if (idx == 0) pred = 1;
        else pred = (input[idx] != input[idx - 1]) ? 1 : 0;
    }


    int inclusive;
    inclusive = block_inclusive_scan<1024>(pred, s_mem);

    if (idx < size) d_running[idx] = inclusive;

    if (threadIdx.x == blockDim.x - 1 || idx == size - 1) {
        d_blocks[blockIdx.x] = inclusive;
    }
}
// d_blocks length n <= 1024 (one block)
__global__ void scan_block_sums_inplace(int* d_blocks, int n) {
    __shared__ int s_mem[32];
    int idx = threadIdx.x;
    int val = (idx < n) ? d_blocks[idx] : 0;
    int inclusive = block_inclusive_scan<1024>(val, s_mem);
    if (idx < n) d_blocks[idx] = inclusive;
}

__global__ void add_block_offsets(int* __restrict__ d_running,
                                  int size,
                                  const int* __restrict__ d_blocks) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    int offset = (blockIdx.x == 0) ? 0 : d_blocks[blockIdx.x - 1];
    d_running[idx] += offset;
}

__global__ void differce_scan(int* __restrict__ scan, int* __restrict__ index, int size_whole, int n){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size_whole){
        if(idx == 0){
            index[0] = 0;
        } else {
            if(scan[idx] != scan[idx - 1]){
                index[scan[idx] - 1] = idx;
            }
            if(idx == size_whole - 1){
                index[n] = size_whole;
            }
        }
    }
}
__global__ void final_run(unsigned char* input, int* h_index_array, unsigned char* symbols, int* counts, int n){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int a = h_index_array[idx];
        int b = h_index_array[idx + 1];
        counts[idx] = b - a;
        symbols[idx] = input[a];
    }
}

__global__ void dec_scan_block_level(int* __restrict__ scan, int size, int* __restrict__ blocks){
    extern __shared__ int s_mem[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < size) ? scan[idx] : 0;
    int inclusive = block_inclusive_scan<1024>(val, s_mem);
    if (idx < size) scan[idx] = inclusive;
    if (threadIdx.x == blockDim.x - 1 || idx == size - 1) {
        blocks[blockIdx.x] = inclusive;
    }
}

__global__ void inclusive_to_exclusive(const int* __restrict__ inclusive,
                                       const int* __restrict__ values,
                                       int n,
                                       int* __restrict__ exclusive){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        exclusive[idx] = inclusive[idx] - values[idx];
    }
}

__global__ void mark_run_starts(const int* __restrict__ offsets,
                                int nRuns,
                                int total_len,
                                int* __restrict__ flags){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_len) {
        // zero init
        flags[idx] = 0;
    }
    if (idx < nRuns){
        int pos = offsets[idx];
        if (pos < total_len) flags[pos] = 1;
    }
}

__global__ void map_symbols_from_runs(const int* __restrict__ runIndexPlus1,
                                      const unsigned char* __restrict__ symbols,
                                      int total_len,
                                      unsigned char* __restrict__ out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_len){
        int runIdx = runIndexPlus1[idx] - 1; // convert to 0-based
        out[idx] = symbols[(runIdx >= 0) ? runIdx : 0];
    }
}