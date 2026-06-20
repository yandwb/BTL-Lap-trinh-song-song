/**
 * TỔNG HỢP BENCHMARK: SEQUENTIAL vs OPENMP vs CUDA
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d - %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int CUDA_BLOCK_SIZE = 256; 
static const int OMP_BLOCK_SIZE = 64; 

// ==============================================================================
// 1. PHIÊN BẢN TUẦN TỰ (CPU SINGLE-THREAD)
// ==============================================================================
float dtw_sequential(const vector<float>& A, const vector<float>& B) {
    int n = A.size(), m = B.size();
    int cols = m + 1;
    vector<float> dtw((size_t)(n + 1) * cols, INFINITY);
    dtw[0 * cols + 0] = 0.0f;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = std::abs(A[i - 1] - B[j - 1]);
            float ins  = dtw[(i - 1) * cols + j];
            float del  = dtw[i * cols + (j - 1)];
            float mat  = dtw[(i - 1) * cols + (j - 1)];
            dtw[i * cols + j] = cost + fminf(ins, fminf(del, mat));
        }
    }
    return dtw[n * cols + m];
}

// ==============================================================================
// 2. PHIÊN BẢN OPENMP (CPU MULTI-THREADING TASK TILING)
// ==============================================================================
float dtw_openmp(const vector<float>& A, const vector<float>& B, int block_size) {
    int n = A.size(), m = B.size();
    int cols = m + 1;
    vector<float> dtw((size_t)(n + 1) * cols, INFINITY);
    dtw[0 * cols + 0] = 0.0f;

    int num_blocks_x = (n + block_size - 1) / block_size;
    int num_blocks_y = (m + block_size - 1) / block_size;
    int sync_cols = num_blocks_y + 1;
    vector<int> sync_vec((num_blocks_x + 1) * sync_cols, 0);
    int* sync = sync_vec.data(); 

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bx = 0; bx < num_blocks_x; ++bx) {
                for (int by = 0; by < num_blocks_y; ++by) {
                    
                    #pragma omp task depend(in: sync[bx * sync_cols + (by + 1)], \
                                                sync[(bx + 1) * sync_cols + by], \
                                                sync[bx * sync_cols + by]) \
                                     depend(out: sync[(bx + 1) * sync_cols + (by + 1)]) \
                                     shared(dtw, A, B, sync) \
                                     firstprivate(bx, by, n, m, cols, block_size, sync_cols)
                    {
                        int i_start = bx * block_size + 1;
                        int i_end = min(n, (bx + 1) * block_size);
                        int j_start = by * block_size + 1;
                        int j_end = min(m, (by + 1) * block_size);

                        for (int i = i_start; i <= i_end; ++i) {
                            for (int j = j_start; j <= j_end; ++j) {
                                float cost = std::abs(A[i - 1] - B[j - 1]);
                                float ins  = dtw[(i - 1) * cols + j];     
                                float del  = dtw[i * cols + (j - 1)];     
                                float mat  = dtw[(i - 1) * cols + (j - 1)]; 
                                dtw[i * cols + j] = cost + fminf(ins, fminf(del, mat));
                            }
                        }
                    } 
                }
            }
        } 
    }
    return dtw[n * cols + m];
}

// ==============================================================================
// 3. PHIÊN BẢN CUDA (GPU WAVEFRONT)
// ==============================================================================
__global__ void dtw_wavefront_kernel(const float* __restrict__ A_dev, const float* __restrict__ B_dev, float* dtw_dev, int n, int m, int d) {
    int cols = m + 1;
    int i_min = max(1, d + 2 - m);
    int i_max = min(n, d + 1);
    int diag_len = i_max - i_min + 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= diag_len) return;
    int i = i_min + tid;
    int j = d + 2 - i;      
    if (j < 1 || j > m) return;

    float cost = fabsf(A_dev[i-1] - B_dev[j-1]);
    float ins  = dtw_dev[(i-1)*cols + j];
    float del  = dtw_dev[i*cols + (j-1)];
    float mat  = dtw_dev[(i-1)*cols + (j-1)];
    dtw_dev[i*cols + j] = cost + fminf(ins, fminf(del, mat));
}

__global__ void init_boundary_kernel(float* dtw_dev, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) dtw_dev[idx] = (idx == 0) ? 0.0f : INFINITY; 
}

float dtw_cuda(const vector<float>& A, const vector<float>& B) {
    int n = A.size(), m = B.size();
    int rows = n + 1, cols = m + 1;
    size_t mat_bytes = (size_t)rows * cols * sizeof(float);
    size_t seq_bytes = n * sizeof(float);

    float *A_dev, *B_dev, *dtw_dev;
    CUDA_CHECK(cudaMalloc(&A_dev, seq_bytes));
    CUDA_CHECK(cudaMalloc(&B_dev, seq_bytes));
    CUDA_CHECK(cudaMalloc(&dtw_dev, mat_bytes));

    CUDA_CHECK(cudaMemcpy(A_dev, A.data(), seq_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_dev, B.data(), seq_bytes, cudaMemcpyHostToDevice));

    int blocks = (rows * cols + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    init_boundary_kernel<<<blocks, CUDA_BLOCK_SIZE>>>(dtw_dev, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_diags = n + m - 1;
    for (int d = 0; d < num_diags; d++) {
        int diag_len = min(n, d + 1) - max(1, d + 2 - m) + 1;
        int num_blocks = (diag_len + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        dtw_wavefront_kernel<<<num_blocks, CUDA_BLOCK_SIZE>>>(A_dev, B_dev, dtw_dev, n, m, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float result;
    CUDA_CHECK(cudaMemcpy(&result, dtw_dev + (size_t)n*cols + m, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(A_dev)); CUDA_CHECK(cudaFree(B_dev)); CUDA_CHECK(cudaFree(dtw_dev));
    return result;
}

// ==============================================================================
// 4. CHƯƠNG TRÌNH SO SÁNH (BENCHMARK)
// ==============================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    omp_set_num_threads(omp_get_max_threads()); // Ép OpenMP dùng tối đa luồng CPU
    
    cout << "======================================================================================\n";
    cout << " TONG HOP BENCHMARK: SEQUENTIAL vs OPENMP vs CUDA\n";
    cout << " CPU        : Intel Core i7 (Thuc thi qua " << omp_get_max_threads() << " luong OpenMP)\n";
    cout << " GPU        : " << prop.name << "\n";
    cout << "======================================================================================\n";

    // Warm-up CUDA (Bỏ qua độ trễ khởi tạo ban đầu)
    vector<float> wA(100, 0.0f), wB(100, 0.0f);
    dtw_cuda(wA, wB);

    // Bảng kết quả
    cout << left 
         << setw(8)  << "N" 
         << setw(15) << "Seq (ms)" 
         << setw(15) << "OpenMP (ms)" 
         << setw(15) << "CUDA (ms)" 
         << setw(15) << "Speedup OMP" 
         << setw(15) << "Speedup CUDA" << "\n";
    cout << string(84, '-') << "\n";

    // Chạy các mốc (10k -> 30k)
    vector<int> sizes = {10000, 15000, 20000, 25000, 30000};

    for (int n : sizes) {
        vector<float> A(n), B(n);
        for (int i = 0; i < n; i++) {
            A[i] = sin(i * 0.05f) * 5.0f;
            B[i] = cos(i * 0.05f) * 5.0f;
        }

        // Đo Sequential
        auto t0 = chrono::high_resolution_clock::now();
        dtw_sequential(A, B);
        auto t1 = chrono::high_resolution_clock::now();
        double ms_seq = chrono::duration<double, milli>(t1 - t0).count();

        // Đo OpenMP
        t0 = chrono::high_resolution_clock::now();
        dtw_openmp(A, B, OMP_BLOCK_SIZE);
        t1 = chrono::high_resolution_clock::now();
        double ms_omp = chrono::duration<double, milli>(t1 - t0).count();

        // Đo CUDA
        t0 = chrono::high_resolution_clock::now();
        dtw_cuda(A, B);
        t1 = chrono::high_resolution_clock::now();
        double ms_cuda = chrono::duration<double, milli>(t1 - t0).count();

        // In kết quả
        cout << left << setw(8)  << n
             << fixed << setprecision(2)
             << setw(15) << ms_seq
             << setw(15) << ms_omp
             << setw(15) << ms_cuda
             << setw(15) << (ms_seq / ms_omp)
             << setw(15) << (ms_seq / ms_cuda) << "\n";
    }
    
    cout << string(84, '-') << "\n";
    cout << "=> Done! Hoan thanh qua trinh danh gia hieu nang Tong the.\n";
    return 0;
}