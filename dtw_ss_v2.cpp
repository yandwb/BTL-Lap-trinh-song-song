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

// Dùng Float để giảm một nửa dung lượng RAM cần đọc/ghi
inline float calc_distance(float a, float b) {
    return std::abs(a - b);
}

// ---------------------------------------------------------
// THUẬT TOÁN SONG SONG TASK TILING (BẢN AN TOÀN - DÙNG FLOAT)
// ---------------------------------------------------------
float dtw_block_task_openmp(const vector<float>& A, const vector<float>& B, int block_size) {
    int n = A.size();
    int m = B.size();
    int cols = m + 1;

    // Khởi tạo ma trận float
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

                        // ĐÃ XÓA PRAGMA SIMD Ở ĐÂY - CHẠY TUẦN TỰ TRONG L1 CACHE
                        for (int i = i_start; i <= i_end; ++i) {
                            for (int j = j_start; j <= j_end; ++j) {
                                float cost = std::abs(A[i - 1] - B[j - 1]);
                                
                                float insert_cost = dtw[(i - 1) * cols + j];     
                                float delete_cost = dtw[i * cols + (j - 1)];     
                                float match_cost  = dtw[(i - 1) * cols + (j - 1)]; 
                                
                                // Dùng fminf thay vì min để ép CPU dùng tập lệnh 32-bit nhanh hơn
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

int main() {
    const int SIZE = 5000;
    const int BLOCK_SIZE = 64; 
    
    cout << "===========================================" << endl;
    cout << "Kieu du lieu: FLOAT (32-bit)" << endl;
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
        cout << "--- BAN TASK BLOCK TILING (" << threads << " Luong) ---" << endl;
        omp_set_num_threads(threads);
        
        auto start_par = high_resolution_clock::now();
        float result_par = dtw_block_task_openmp(sequence_A, sequence_B, BLOCK_SIZE);
        auto end_par = high_resolution_clock::now();
        auto duration_par = duration_cast<milliseconds>(end_par - start_par).count();

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