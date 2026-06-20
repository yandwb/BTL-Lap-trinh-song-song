/**
 * DTW CUDA Wavefront Implementation (Optimized for RTX 4060)
 * ==========================================================
 * Data Type: float (4 bytes) to safely fit 30k x 30k (3.6 GB) in 8GB VRAM.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// ─── Macro kiểm tra lỗi CUDA ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d - %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int BLOCK_SIZE     = 256; 
static const int BENCHMARK_RUNS = 5; // Chạy 5 lần lấy trung bình cho ổn định

struct DiagLog {
    int   diag_id;
    int   diag_len;
    int   active_threads;
    int   total_threads;
    float kernel_time_us;
    float efficiency;
};

struct TimingResult {
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
};

// ─── Đọc file dữ liệu từ Python xuất ra ───────────────────────────────────────
vector<float> read_signal_from_file(const string& filename, int length) {
    vector<float> data(length);
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "❌ LOI: Khong the mo file " << filename << "!\n";
        cerr << "Kiem tra lai xem file txt co nam cung cho voi file chay khong.\n";
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < length; i++) {
        file >> data[i];
    }
    file.close();
    return data;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHẦN 1: CUDA KERNEL
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void dtw_wavefront_kernel(
    const float* __restrict__ A_dev,
    const float* __restrict__ B_dev,
    float* dtw_dev,
    int n, int m, int d)          
{
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

    float best = fminf(ins, del);
    best = fminf(best, mat);
    dtw_dev[i*cols + j] = cost + best;
}

__global__ void init_boundary_kernel(float* dtw_dev, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        // INFINITY (Macro của float)
        dtw_dev[idx] = (idx == 0) ? 0.0f : INFINITY; 
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHẦN 2: HOST FUNCTION ĐIỀU PHỐI KERNEL
// ═══════════════════════════════════════════════════════════════════════════════

float dtw_cuda(const vector<float>& A, const vector<float>& B,
                bool collect_log = false,
                vector<DiagLog>* logs = nullptr)
{
    int n = A.size(), m = B.size();
    int rows = n + 1, cols = m + 1;
    size_t mat_bytes = (size_t)rows * cols * sizeof(float);
    size_t seq_bytes_n = n * sizeof(float);
    size_t seq_bytes_m = m * sizeof(float);

    float *A_dev, *B_dev, *dtw_dev;
    CUDA_CHECK(cudaMalloc(&A_dev,   seq_bytes_n));
    CUDA_CHECK(cudaMalloc(&B_dev,   seq_bytes_m));
    CUDA_CHECK(cudaMalloc(&dtw_dev, mat_bytes));

    CUDA_CHECK(cudaMemcpy(A_dev, A.data(), seq_bytes_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_dev, B.data(), seq_bytes_m, cudaMemcpyHostToDevice));

    {
        int total = rows * cols;
        int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_boundary_kernel<<<blocks, BLOCK_SIZE>>>(dtw_dev, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int num_diags = n + m - 1;
    if (collect_log && logs) logs->reserve(num_diags);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // VÒNG LẶP WAVEFRONT
    for (int d = 0; d < num_diags; d++) {
        int i_min    = max(1, d + 2 - m);
        int i_max    = min(n, d + 1);
        int diag_len = i_max - i_min + 1;

        int num_blocks = (diag_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CUDA_CHECK(cudaEventRecord(ev_start));

        dtw_wavefront_kernel<<<num_blocks, BLOCK_SIZE>>>(A_dev, B_dev, dtw_dev, n, m, d);

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        if (collect_log) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            DiagLog log;
            log.diag_id        = d;
            log.diag_len       = diag_len;
            log.active_threads = diag_len;
            log.total_threads  = num_blocks * BLOCK_SIZE;
            log.kernel_time_us = ms * 1000.0f;
            log.efficiency     = (float)diag_len / (float)log.total_threads;
            logs->push_back(log);
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    float result;
    CUDA_CHECK(cudaMemcpy(&result,
                          dtw_dev + (size_t)n*cols + m,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_dev));
    CUDA_CHECK(cudaFree(B_dev));
    CUDA_CHECK(cudaFree(dtw_dev));

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHẦN 3: ĐÁNH GIÁ HIỆU NĂNG (BENCHMARK)
// ═══════════════════════════════════════════════════════════════════════════════

TimingResult benchmark_timing(const vector<float>& A, const vector<float>& B, int runs)
{
    vector<double> times;
    times.reserve(runs);

    for (int r = 0; r < runs; r++) {
        auto t0 = chrono::high_resolution_clock::now();
        
        dtw_cuda(A, B, false, nullptr);
        
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();
        times.push_back(ms);
    }

    double sum = 0, sq_sum = 0;
    for (double t : times) {
        sum += t; sq_sum += t*t;
    }
    double mean = sum / runs;
    double var  = sq_sum / runs - mean * mean;

    return { mean, sqrt(max(var, 0.0)), 0, 0 };
}

void run_speedup_benchmark(const string& csv_path) {
    // Các mốc dữ liệu thực tế trích xuất từ MIT-BIH
    vector<int> sizes = {10000, 15000, 20000, 25000, 30000};

    ofstream f(csv_path);
    f << "n,cuda_mean_ms,cuda_std_ms,giga_cups\n"; 

    cout << "\n[Stress Test Benchmark (Du lieu Y sinh MIT-BIH)]\n";
    cout << setw(8)  << "n"
         << setw(18) << "CUDA Time (ms)"
         << setw(15) << "Thong luong (GCUPS)"
         << "\n";
    cout << string(45, '-') << "\n";

    for (int n : sizes) {
        int k_val = n / 1000;
        
        // Đã sửa đường dẫn: bỏ chữ "data/"
        string file_A = "signal_A_" + to_string(k_val) + "k.txt";
        string file_B = "signal_B_" + to_string(k_val) + "k.txt";

        // Tự động đọc tín hiệu
        vector<float> A = read_signal_from_file(file_A, n);
        vector<float> B = read_signal_from_file(file_B, n);

        auto cuda_res = benchmark_timing(A, B, BENCHMARK_RUNS);

        // Tính GCUPS (Số ô tính toán trong 1 giây / 1 tỷ)
        double total_cells = (double)n * n;
        double gcups = (total_cells / (cuda_res.mean_ms / 1000.0)) / 1e9;

        cout << setw(8)  << n
             << setw(18) << fixed << setprecision(2) << cuda_res.mean_ms
             << setw(15) << fixed << setprecision(3) << gcups
             << "\n";

        f << n << ","
          << fixed << setprecision(4) << cuda_res.mean_ms << ","
          << cuda_res.stddev_ms << ","
          << gcups << "\n";
    }
    cout << "  => CSV da luu: " << csv_path << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

// int main(int argc, char* argv[]) {
//     // 1. In thông số cấu hình GPU
//     cudaDeviceProp prop;
//     CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
//     cout << "--- HE THONG BENCHMARK CUDA DTW ---\n";
//     cout << "GPU Dang chay: " << prop.name << "\n";
//     cout << "Kien truc SM : " << prop.major << "." << prop.minor << "\n";
//     cout << "VRAM Kha dung: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n\n";

//     run_speedup_benchmark("stress_test_results.csv");

//     return 0;
// }
// ═══════════════════════════════════════════════════════════════════════════════
// MAIN FUNCTION (Chiến dịch lấy Log Cân Bằng Tải)
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    // 1. In thông số cấu hình GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "--- HE THONG BENCHMARK CUDA DTW ---\n";
    cout << "GPU Dang chay: " << prop.name << "\n";
    cout << "Kien truc SM : " << prop.major << "." << prop.minor << "\n";
    cout << "VRAM Kha dung: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n\n";

    // 2. Chạy Cân bằng tải (Load-balancing) với N=1000
    int n = 1000; 
    vector<float> A(n), B(n);
    for (int i = 0; i < n; i++) {
        A[i] = sin(i * 0.05f) * 5.0f;
        B[i] = cos(i * 0.05f) * 5.0f;
    }

    cout << "Dang thu thap log Can bang tai (Efficiency) cho TOAN BO Kernel...\n";
    
    // Bật tham số 'true' để xuất log
    vector<DiagLog> logs;
    dtw_cuda(A, B, true, &logs); 

    // Ghi toàn bộ dữ liệu ra file CSV
    ofstream f("occupancy_log.csv");
    f << "diag_id,diag_len,active_threads,total_threads,efficiency\n";
    for (auto& l : logs) {
        f << l.diag_id << "," << l.diag_len << "," 
          << l.active_threads << "," << l.total_threads << "," 
          << fixed << setprecision(4) << l.efficiency << "\n";
    }
    f.close();

    cout << "=> Da luu thanh cong thong so cua " << logs.size() << " kernels vao file occupancy_log.csv!\n";
    cout << "Bay gio anh mo Python chay plot_results.py de ve Histogram nhe!\n";

    return 0;
}