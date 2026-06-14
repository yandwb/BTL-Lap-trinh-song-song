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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

vector<float> load_from_txt(const string& filename) {
    vector<float> data;
    ifstream file(filename);
    float v;
    while (file >> v) data.push_back(v);
    return data;
}

// ======================================================
// 1. LÕI TUẦN TỰ (CHỈ TÍNH TOÁN)
// ======================================================
void dtw_sequential_core(const vector<float>& A, const vector<float>& B, vector<float>& dtw) {
    int n = A.size(), m = B.size(), cols = m + 1;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = std::abs(A[i - 1] - B[j - 1]);
            dtw[i * cols + j] = cost + fminf(dtw[(i - 1) * cols + j], fminf(dtw[i * cols + (j - 1)], dtw[(i - 1) * cols + (j - 1)]));
        }
    }
}

// ======================================================
// 2. LÕI OPENMP (CHỈ TÍNH TOÁN)
// ======================================================
void dtw_openmp_core(const vector<float>& A, const vector<float>& B, int block_size, vector<float>& dtw) {
    int n = A.size(), m = B.size(), cols = m + 1;
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
                    #pragma omp task depend(in: sync[bx * sync_cols + (by + 1)], sync[(bx + 1) * sync_cols + by], sync[bx * sync_cols + by]) \
                                     depend(out: sync[(bx + 1) * sync_cols + (by + 1)]) shared(dtw, A, B, sync) firstprivate(bx, by, n, m, cols, block_size, sync_cols)
                    {
                        int i_start = bx * block_size + 1, i_end = min(n, (bx + 1) * block_size);
                        int j_start = by * block_size + 1, j_end = min(m, (by + 1) * block_size);
                        for (int i = i_start; i <= i_end; ++i) {
                            for (int j = j_start; j <= j_end; ++j) {
                                dtw[i * cols + j] = std::abs(A[i - 1] - B[j - 1]) + fminf(dtw[(i - 1) * cols + j], fminf(dtw[i * cols + (j - 1)], dtw[(i - 1) * cols + (j - 1)]));
                            }
                        }
                    } 
                }
            }
        } 
    }
}

// ======================================================
// 3. KERNEL CUDA TỐI ƯU TOÁN HỌC TUYỆT ĐỐI
// ======================================================
__global__ void init_dtw_kernel(float* __restrict__ dtw, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dtw[idx] = INFINITY;
        if (idx == 0) dtw[0] = 0.0f;
    }
}

__global__ void dtw_wavefront_kernel(float* __restrict__ dtw, const float* __restrict__ A, const float* __restrict__ B, int n, int m, int diag) {
    int cols = m + 1;
    int start_i = max(1, diag - m);
    int end_i   = min(n, diag - 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (start_i + idx <= end_i) {
        int i = start_i + idx;
        int j = diag - i;
        
        // Tối ưu triệt để phép nhân ma trận thành cộng trừ chỉ mục
        int base = i * cols + j;
        float cost = fabsf(A[i - 1] - B[j - 1]);
        
        float left = dtw[base - 1];
        float up = dtw[base - cols];
        float diagv = dtw[base - cols - 1];
        
        dtw[base] = cost + fminf(left, fminf(up, diagv));
    }
}

// ======================================================
// HÀM CHÍNH - ĐO LƯỜNG CHUẨN HPC
// ======================================================
int main() {
    omp_set_num_threads(8); 
    int omp_block_size = 512;

    cout << "🔥 Kich hoat che do Benchmark loi (Core Compute Isolation)..." << endl;
    vector<string> sizes = {"1k", "2k", "3k", "4k", "5k"};
    
    cout << "\n========================================================================================================\n";
    cout << left << setw(10) << "Size" 
         << setw(15) << "Seq (ms)" 
         << setw(15) << "OpenMP (ms)" 
         << setw(15) << "CUDA (ms)" 
         << setw(20) << "Speedup (OMP)" 
         << setw(20) << "Speedup (CUDA)" << endl;
    cout << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& s : sizes) {
        vector<float> seq = load_from_txt("data/100_" + s + ".txt");
        int n = seq.size();
        int m = seq.size();
        int cols = m + 1;
        long long total = (long long)(n + 1) * cols;

        // =======================================
        // SETUP: CẤP PHÁT BỘ NHỚ TRƯỚC (KHÔNG ĐO)
        // =======================================
        vector<float> dtw_cpu(total, INFINITY); dtw_cpu[0] = 0.0f;
        vector<float> dtw_omp(total, INFINITY); dtw_omp[0] = 0.0f;
        
        float *d_A, *d_B, *d_dtw;
        CUDA_CHECK(cudaMalloc(&d_A, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dtw, total * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, seq.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, seq.data(), m * sizeof(float), cudaMemcpyHostToDevice));

        // Dịch bản đồ CUDA Graph (KHÔNG ĐO)
        cudaStream_t stream; cudaStreamCreate(&stream);
        cudaGraph_t graph; cudaGraphExec_t instance;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        int init_threads = 256;
        int init_blocks = (total + init_threads - 1) / init_threads;
        init_dtw_kernel<<<init_blocks, init_threads, 0, stream>>>(d_dtw, total);

        int threads = 256; 
        for (int diag = 2; diag <= n + m; ++diag) {
            int start_i = max(1, diag - m);
            int end_i   = min(n, diag - 1);
            int num = end_i - start_i + 1;
            if (num > 0) {
                int blocks = (num + threads - 1) / threads;
                dtw_wavefront_kernel<<<blocks, threads, 0, stream>>>(d_dtw, d_A, d_B, n, m, diag);
            }
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        // =======================================
        // BẮT ĐẦU ĐUA: CHỈ ĐO THỜI GIAN THỰC THI 
        // =======================================
        
        // 1. CPU Tuần tự
        auto start_seq = high_resolution_clock::now();
        dtw_sequential_core(seq, seq, dtw_cpu);
        float time_seq = duration_cast<milliseconds>(high_resolution_clock::now() - start_seq).count();

        // 2. CPU OpenMP
        auto start_omp = high_resolution_clock::now();
        dtw_openmp_core(seq, seq, omp_block_size, dtw_omp);
        float time_omp = duration_cast<milliseconds>(high_resolution_clock::now() - start_omp).count();

        // 3. GPU CUDA
        cudaDeviceSynchronize(); // Chắc chắn setup đã xong xuôi
        auto start_cuda = high_resolution_clock::now();
        cudaGraphLaunch(instance, stream); // Bắn lệnh nổ dây chuyền
        cudaStreamSynchronize(stream);     // Chờ tính xong
        float time_cuda = duration_cast<milliseconds>(high_resolution_clock::now() - start_cuda).count();

        // =======================================
        // KẾT THÚC ĐUA: DỌN DẸP & IN KẾT QUẢ
        // =======================================
        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_dtw));

        cout << left << setw(10) << (s + "x" + s)
             << setw(15) << time_seq
             << setw(15) << time_omp
             << "\033[1;32m" << setw(15) << time_cuda << "\033[0m"
             << fixed << setprecision(2)
             << setw(20) << (to_string((time_omp > 0) ? (time_seq / time_omp) : 0) + "x")
             << "\033[1;32m" << setw(20) << (to_string((time_cuda > 0) ? (time_seq / time_cuda) : 0) + "x") << "\033[0m" << endl;
    }
    cout << "========================================================================================================\n";

    return 0;
}