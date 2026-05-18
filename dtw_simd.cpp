#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <omp.h>
#include <cstdio>

using namespace std;
using namespace std::chrono;

inline float calc_distance(float a, float b) {
    return std::abs(a - b);
}

// ---------------------------------------------------------
// THUẬT TOÁN SONG SONG TASK TILING + SIMD (INTEL AVX)
// ---------------------------------------------------------
float dtw_block_task_openmp_simd(const vector<float>& A, const vector<float>& B, int block_size) {
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

    // Lấy con trỏ thô để tối ưu tốc độ đọc/ghi cho SIMD
    float* dtw_ptr = dtw.data();
    const float* A_ptr = A.data();
    const float* B_ptr = B.data();

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
                                     shared(dtw_ptr, A_ptr, B_ptr, sync) \
                                     firstprivate(bx, by, n, m, cols, block_size, sync_cols)
                    {
                        int i_start = bx * block_size + 1;
                        int i_end = std::min(n, (bx + 1) * block_size);
                        
                        int j_start = by * block_size + 1;
                        int j_end = std::min(m, (by + 1) * block_size);

                        // CHIỀU CAO VÀ CHIỀU RỘNG CỦA BLOCK HIỆN TẠI
                        int H = i_end - i_start + 1;
                        int W = j_end - j_start + 1;
                        int num_diagonals = H + W - 1;

                        // BƯỚC ĐỘT PHÁ: QUÉT THEO TỪNG ĐƯỜNG CHÉO BÊN TRONG BLOCK
                        for (int k = 0; k < num_diagonals; ++k) {
                            
                            // Xác định điểm bắt đầu và kết thúc của đường chéo thứ k
                            int local_i_start = std::max(0, k - W + 1);
                            int local_i_end = std::min(k, H - 1);

                            // ÉP XUNG CPU BẰNG TẬP LỆNH SIMD (AVX/AVX2)
                            // Các phần tử trong vòng lặp này hoàn toàn độc lập!
                            #pragma omp simd
                            for (int local_i = local_i_start; local_i <= local_i_end; ++local_i) {
                                int local_j = k - local_i;
                                
                                // Tọa độ thật trên ma trận khổng lồ
                                int i = i_start + local_i;
                                int j = j_start + local_j;

                                float cost = std::abs(A_ptr[i - 1] - B_ptr[j - 1]);
                                
                                float insert_cost = dtw_ptr[(i - 1) * cols + j];     
                                float delete_cost = dtw_ptr[i * cols + (j - 1)];     
                                float match_cost  = dtw_ptr[(i - 1) * cols + (j - 1)]; 
                                
                                dtw_ptr[i * cols + j] = cost + fminf(insert_cost, fminf(delete_cost, match_cost));
                            }
                        }
                    } 
                }
            }
        } 
    }

    return dtw[n * cols + m];
}

int main() {
    const int SIZE = 5000;
    const int BLOCK_SIZE = 64; 
    
    cout << "===========================================" << endl;
    cout << "Kieu du lieu: FLOAT (32-bit)" << endl;
    cout << "Toi uu: Task Tiling + Micro-Wavefront (SIMD AVX)" << endl;
    cout << "Kich thuoc mang: " << SIZE << "x" << SIZE << endl;
    cout << "Kich thuoc Block: " << BLOCK_SIZE << "x" << BLOCK_SIZE << endl;
    cout << "===========================================\n" << endl;
    fflush(stdout);

    vector<float> sequence_A(SIZE);
    vector<float> sequence_B(SIZE);
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int i = 0; i < SIZE; ++i) {
        sequence_A[i] = dist(rng);
        sequence_B[i] = dist(rng);
    }

    vector<int> thread_counts = {1, 2, 4, 8};
    float base_time = 0;

    for (int threads : thread_counts) {
        cout << "--- BAN TASK BLOCK TILING + SIMD (" << threads << " Luong) ---" << endl;
        omp_set_num_threads(threads);
        
        auto start_par = high_resolution_clock::now();
        float result_par = dtw_block_task_openmp_simd(sequence_A, sequence_B, BLOCK_SIZE);
        auto end_par = high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> ms_double = end_par - start_par;
        double duration_par = ms_double.count();

        cout << "Ket qua: " << result_par << endl;
        
        if (threads == 1) {
            base_time = duration_par;
            cout << "Thoi gian: " << duration_par << " ms (Lam moc co so)\n" << endl;
        } else {
            double speedup = base_time / duration_par;
            cout << "Thoi gian: " << duration_par << " ms -> Tang toc (Speedup): " << speedup << "x\n" << endl;
        }
        fflush(stdout);
    }

    return 0;
}