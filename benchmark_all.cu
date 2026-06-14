#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <iomanip>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ======================================================
// MACRO BẮT LỖI CUDA
// ======================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ======================================================
// HÀM ĐỌC FILE
// ======================================================
vector<float> load_from_txt(const string& filename) {
    vector<float> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "❌ Khong the mo file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    float v;
    while (file >> v) data.push_back(v);
    return data;
}

// ======================================================
// 1. THUẬT TOÁN TUẦN TỰ (CPU TRUYỀN THỐNG)
// ======================================================
float dtw_sequential(const vector<float>& A, const vector<float>& B) {
    int n = A.size();
    int m = B.size();
    int cols = m + 1;
    vector<float> dtw((n + 1) * cols, INFINITY);
    dtw[0 * cols + 0] = 0.0f;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = std::abs(A[i - 1] - B[j - 1]);
            float insert_cost = dtw[(i - 1) * cols + j];
            float delete_cost = dtw[i * cols + (j - 1)];
            float match_cost  = dtw[(i - 1) * cols + (j - 1)];
            dtw[i * cols + j] = cost + fminf(insert_cost, fminf(delete_cost, match_cost));
        }
    }
    return dtw[n * cols + m];
}

// ======================================================
// 2. THUẬT TOÁN SONG SONG CPU (OPENMP TASK TILING)
// ======================================================
float dtw_openmp_task(const vector<float>& A, const vector<float>& B, int block_size) {
    int n = A.size();
    int m = B.size();
    int cols = m + 1;
    vector<float> dtw((n + 1) * cols, INFINITY);
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
                                float insert_cost = dtw[(i - 1) * cols + j];     
                                float delete_cost = dtw[i * cols + (j - 1)];     
                                float match_cost  = dtw[(i - 1) * cols + (j - 1)]; 
                                dtw[i * cols + j] = cost + fminf(insert_cost, fminf(delete_cost, match_cost));
                            }
                        }
                    } 
                }
            }
        } 
    }
    return dtw[n * cols + m];
}

// ======================================================
// 3. THUẬT TOÁN GPU (CUDA WAVEFRONT)
// ======================================================
__global__ void dtw_wavefront_kernel(float* dtw, const float* A, const float* B, int n, int m, int diag) {
    int cols = m + 1;
    int start_i = max(1, diag - m);
    int end_i   = min(n, diag - 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (start_i + idx <= end_i) {
        int i = start_i + idx;
        int j = diag - i;
        float cost = fabsf(A[i - 1] - B[j - 1]);
        int base = i * cols;
        float left  = dtw[base + (j - 1)];
        float up    = dtw[(i - 1) * cols + j];
        float diagv = dtw[(i - 1) * cols + (j - 1)];
        dtw[base + j] = cost + fminf(left, fminf(up, diagv));
    }
}

float dtw_cuda_wavefront(const vector<float>& A, const vector<float>& B) {
    int n = A.size();
    int m = B.size();
    int cols = m + 1;
    long long total = (long long)(n + 1) * cols;

    vector<float> h_dtw(total, INFINITY);
    h_dtw[0] = 0.0f;
    float *d_A, *d_B, *d_dtw;

    CUDA_CHECK(cudaMalloc(&d_A, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dtw, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dtw, h_dtw.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;

    // Vòng lặp bắn Kernel vào Default Stream (Không có DeviceSync cản đường)
    for (int diag = 2; diag <= n + m; ++diag) {
        int start_i = max(1, diag - m);
        int end_i   = min(n, diag - 1);
        int num = end_i - start_i + 1;

        if (num > 0) {
            int blocks = (num + threads - 1) / threads;
            dtw_wavefront_kernel<<<blocks, threads>>>(d_dtw, d_A, d_B, n, m, diag);
        }
    }
    
    // Đã đưa Sync ra ngoài vòng lặp -> Tốc độ tăng vọt!
    CUDA_CHECK(cudaDeviceSynchronize());

    float result;
    CUDA_CHECK(cudaMemcpy(&result, &d_dtw[n * cols + m], sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_dtw));

    return result;
}

// ======================================================
// MAIN BENCHMARK REPORT
// ======================================================
int main() {
    omp_set_num_threads(8); // Tối ưu CPU đa luồng
    int omp_block_size = 512;

    cout << "🔥 Khoi dong GPU (Warm-up)..." << endl;
    vector<float> warm = load_from_txt("data/100_1k.txt");
    dtw_cuda_wavefront(warm, warm);

    vector<string> sizes = {"1k", "2k", "3k", "4k", "5k"};
    
    // In Header Bảng
    cout << "\n========================================================================================================\n";
    cout << left << setw(10) << "Size" 
         << setw(15) << "Seq (ms)" 
         << setw(15) << "OpenMP (ms)" 
         << setw(15) << "CUDA (ms)" 
         << setw(20) << "Speedup (OMP)" 
         << setw(20) << "Speedup (CUDA)" << endl;
    cout << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& s : sizes) {
        string file = "data/100_" + s + ".txt";
        vector<float> seq = load_from_txt(file);

        // 1. Đo Tuần tự
        auto start_seq = high_resolution_clock::now();
        dtw_sequential(seq, seq);
        auto end_seq = high_resolution_clock::now();
        float time_seq = duration_cast<milliseconds>(end_seq - start_seq).count();

        // 2. Đo OpenMP
        auto start_omp = high_resolution_clock::now();
        dtw_openmp_task(seq, seq, omp_block_size);
        auto end_omp = high_resolution_clock::now();
        float time_omp = duration_cast<milliseconds>(end_omp - start_omp).count();

        // 3. Đo CUDA
        auto start_cuda = high_resolution_clock::now();
        dtw_cuda_wavefront(seq, seq);
        auto end_cuda = high_resolution_clock::now();
        float time_cuda = duration_cast<milliseconds>(end_cuda - start_cuda).count();

        // 4. Tính toán Speedup
        float speedup_omp = (time_omp > 0) ? (time_seq / time_omp) : 0;
        float speedup_cuda = (time_cuda > 0) ? (time_seq / time_cuda) : 0;

        // 5. In kết quả từng hàng
        cout << left << setw(10) << (s + "x" + s)
             << setw(15) << time_seq
             << setw(15) << time_omp
             << setw(15) << time_cuda
             << fixed << setprecision(2)
             << setw(20) << (to_string(speedup_omp) + "x")
             << setw(20) << (to_string(speedup_cuda) + "x") << endl;
    }
    cout << "========================================================================================================\n";

    return 0;
}