#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernels/runarray.cuh"
#include <iostream>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _cuda_check_err = (call);                                            \
        if (_cuda_check_err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_cuda_check_err));     \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)
#endif

#ifndef CUDA_CHECK_LAST_KERNEL
#define CUDA_CHECK_LAST_KERNEL(msg)                                                      \
    do {                                                                                 \
        cudaError_t _e1 = cudaGetLastError();                                            \
        if (_e1 != cudaSuccess) {                                                        \
            fprintf(stderr, "Kernel error after %s at %s:%d -> %s\n",                  \
                    msg, __FILE__, __LINE__, cudaGetErrorString(_e1));                   \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                             \
    } while (0)
#endif

/*RLE assuming image number of input is smaller than < 1024 * 1024 = 1 048 576*/
int main() {
    // Example input; replace with your data
    const int size = 1 << 16; // 65536
    std::vector<unsigned char> h_input(size);
    for (int i = 0; i < size; ++i) h_input[i] = static_cast<unsigned char>((i / 100) & 0xFF);

    unsigned char *d_input = nullptr;
    int *d_running = nullptr;
    int *d_blocks = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_running, size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    const int maxium_number_of_blocks = 1024;
    const int block_size = 1024;
    const int number_of_blocks = (size + block_size - 1) / block_size;
    CUDA_CHECK(cudaMalloc(&d_blocks, number_of_blocks * sizeof(int)));


    /*Compression*/
    // Pass 1: block scans
    size_t numWarps = (block_size + 31) / 32;
    running_scan_blocks<<<number_of_blocks, block_size, numWarps * sizeof(int)>>>(d_input, size, d_running, d_blocks);
    CUDA_CHECK_LAST_KERNEL("running_scan_blocks");



    // Pass 2 block wide scan
    scan_block_sums_inplace<<<1, maxium_number_of_blocks>>>(d_blocks, number_of_blocks);
    CUDA_CHECK_LAST_KERNEL("scan_block_sums_inplace");

    // Pass 3: add offsets
    add_block_offsets<<<number_of_blocks, block_size>>>(d_running, size, d_blocks);
    CUDA_CHECK_LAST_KERNEL("add_block_offsets");


    // Pass 4 getting index array
    int number_of_distinct;
    CUDA_CHECK(cudaMemcpy(&number_of_distinct, d_blocks + (number_of_blocks-1), sizeof(int), cudaMemcpyDeviceToHost));
    int* d_index_array;
    CUDA_CHECK(cudaMalloc(&d_index_array, (number_of_distinct + 1) * sizeof(int)));
    differce_scan<<<number_of_blocks, block_size>>>(d_running, d_index_array, size, number_of_distinct);
    CUDA_CHECK_LAST_KERNEL("difference scan");

    // Pass 5 final
    unsigned char* d_symbols;
    int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_symbols, number_of_distinct * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_counts, number_of_distinct * sizeof(int)));
    int new_block_size = 1024;
    int new_number_of_blocks = (number_of_distinct + new_block_size - 1) / new_block_size;
    final_run<<<new_number_of_blocks, new_block_size>>>(d_input, d_index_array, d_symbols, d_counts, number_of_distinct);
    CUDA_CHECK_LAST_KERNEL("final runs");


    /*
    Decompression
    */
    // Decompression:
    // Given (d_symbols, d_counts) of length number_of_distinct, reconstruct d_output of length `size`.
    int dec_n = number_of_distinct;
    int dec_block_size = 1024;
    int dec_grid = (dec_n + dec_block_size - 1) / dec_block_size;
    int* d_counts_scan = nullptr;
    int* d_dec_blocks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts_scan, dec_n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_counts_scan, d_counts, dec_n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_counts_scan, d_counts, dec_n * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMalloc(&d_dec_blocks, dec_grid * sizeof(int)));


    // Block-level scan of counts -> d_counts_scan with per-block totals in d_dec_blocks
    dec_scan_block_level<<<dec_grid, dec_block_size, ((dec_block_size + 31)/32)*sizeof(int)>>>(d_counts_scan, dec_n, d_dec_blocks);
    CUDA_CHECK_LAST_KERNEL("dec_scan_block_level counts");



    // Scan block totals and add offsets to make a full inclusive scan over counts
    scan_block_sums_inplace<<<1, 1024>>>(d_dec_blocks, dec_grid);
    CUDA_CHECK_LAST_KERNEL("scan dec block sums");



    add_block_offsets<<<dec_grid, dec_block_size>>>(d_counts_scan, dec_n, d_dec_blocks);
    CUDA_CHECK_LAST_KERNEL("add dec offsets");



    // Convert inclusive scan of counts to exclusive offsets
    inclusive_to_exclusive<<<dec_grid, dec_block_size>>>(d_counts_scan, d_counts, dec_n, d_counts);
    CUDA_CHECK_LAST_KERNEL("inclusive_to_exclusive");



    // Build flags of length `size` with 1 at each run start offset
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, size * sizeof(int)));
    int gridFlags = (std::max(size, dec_n) + dec_block_size - 1) / dec_block_size;
    mark_run_starts<<<gridFlags, dec_block_size>>>(d_counts, dec_n, size, d_flags);
    CUDA_CHECK_LAST_KERNEL("mark_run_starts");


    // Scan flags to get run index+1 for each position
    int gridFull = (size + dec_block_size - 1) / dec_block_size;
    int* d_block_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_tmp, gridFull * sizeof(int)));
    dec_scan_block_level<<<gridFull, dec_block_size, ((dec_block_size + 31)/32)*sizeof(int)>>>(d_flags, size, d_block_tmp);
    CUDA_CHECK_LAST_KERNEL("scan flags block-level");


    scan_block_sums_inplace<<<1, 1024>>>(d_block_tmp, gridFull);
    CUDA_CHECK_LAST_KERNEL("scan flags block sums");


    add_block_offsets<<<gridFull, dec_block_size>>>(d_flags, size, d_block_tmp);
    CUDA_CHECK_LAST_KERNEL("add flags offsets");


    // Map each position to symbol by run index (+1)
    unsigned char* d_symbols_dec;
    CUDA_CHECK(cudaMalloc(&d_symbols_dec, size * sizeof(unsigned char)));
    int gridOut = gridFull;
    map_symbols_from_runs<<<gridOut, dec_block_size>>>(d_flags, d_symbols, size, d_symbols_dec);
    CUDA_CHECK_LAST_KERNEL("map_symbols_from_runs");

    
    

    

    // Inspect a few values
    std::vector<unsigned char> h_out(size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_symbols_dec, h_out.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    int counter = 0;
    for(int i = 0; i < size; i++){
        if(h_out[i] != h_input[i]){
            counter++;
        }
    }
    std::cout << counter;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_running));
    CUDA_CHECK(cudaFree(d_blocks));
    CUDA_CHECK(cudaFree(d_index_array));
    CUDA_CHECK(cudaFree(d_symbols));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_counts_scan));
    CUDA_CHECK(cudaFree(d_dec_blocks));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_block_tmp));
    CUDA_CHECK(cudaFree(d_symbols_dec));
    return 0;
}