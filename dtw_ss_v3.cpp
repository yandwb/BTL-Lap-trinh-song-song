#pragma GCC optimize("O3,unroll-loops")

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <cstdio>

using namespace std;
using namespace std::chrono;

// ---------------------------------------------------------
// THUẬT TOÁN ĐA LUỒNG TỐI THƯỢNG (EXPLICIT WAVEFRONT TILING)
// Triệt tiêu hoàn toàn Overhead của OpenMP Tasks
// ---------------------------------------------------------
float dtw_explicit_wavefront_tiling(const vector<float>& A, const vector<float>& B, int block_size) {
    int n = A.size();
    int m = B.size();
    int cols = m + 1;

    vector<float> dtw((n + 1) * cols, INFINITY);
    dtw[0 * cols + 0] = 0.0f;

    int num_blocks_x = (n + block_size - 1) / block_size;
    int num_blocks_y = (m + block_size - 1) / block_size;
    int max_k = num_blocks_x + num_blocks_y;

    // Bật đa luồng MỘT LẦN DUY NHẤT
    #pragma omp parallel 
    {
        // k là chỉ số đường chéo (chạy từ 2 đến tổng số block)
        for (int k = 2; k <= max_k; ++k) {
            
            // Toán học: Tính toán ranh giới các Block nằm trên cùng đường chéo k
            int bx_start = max(1, k - num_blocks_y);
            int bx_end = min(num_blocks_x, k - 1);

            // Phân chia trực tiếp các Block trên đường chéo cho các luồng
            // schedule(dynamic) giúp cân bằng tải cực tốt khi các block rìa bị khuyết
            #pragma omp for schedule(dynamic, 1)
            for (int bx = bx_start; bx <= bx_end; ++bx) {
                int by = k - bx;

                // Đổi tọa độ Block (bx, by) thành tọa độ ma trận thực tế
                int i_start = (bx - 1) * block_size + 1;
                int i_end = min(n, bx * block_size);
                
                int j_start = (by - 1) * block_size + 1;
                int j_end = min(m, by * block_size);

                // CPU cắm đầu tính toán liên tục trong L1 Cache, không bị ai làm phiền
                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        float cost = std::abs(A[i - 1] - B[j - 1]);
                        float insert_cost = dtw[(i - 1) * cols + j];     
                        float delete_cost = dtw[i * cols + (j - 1)];     
                        float match_cost  = dtw[(i - 1) * cols + (j - 1)]; 
                        
                        dtw[i * cols + j] = cost + fminf(insert_cost, fminf(delete_cost, match_cost));
                    }
                }
            } // Hết #pragma omp for -> Tự động có rào chắn đợi nhau ở đây
        }
    }

    return dtw[n * cols + m];
}

int main() {
    const int SIZE = 5000;
    const int BLOCK_SIZE = 64; 
    
    cout << "===========================================" << endl;
    cout << "Kien truc: EXPLICIT WAVEFRONT TILING (Khong dung Task)" << endl;
    cout << "Mang: " << SIZE << "x" << SIZE << " | Block: " << BLOCK_SIZE << "x" << BLOCK_SIZE << endl;
    cout << "===========================================\n" << endl;

    vector<float> sequence_A(SIZE);
    vector<float> sequence_B(SIZE);
    
    for (int i = 0; i < SIZE; ++i) {
        sequence_A[i] = (float)(rand() % 100);
        sequence_B[i] = (float)(rand() % 100);
    }

    vector<int> thread_counts = {1, 2, 4, 8};
    float base_time = 0;

    for (int threads : thread_counts) {
        cout << "--- " << threads << " Luong ---" << endl;
        omp_set_num_threads(threads);
        
        auto start = high_resolution_clock::now();
        float result = dtw_explicit_wavefront_tiling(sequence_A, sequence_B, BLOCK_SIZE);
        auto end = high_resolution_clock::now();
        float duration = duration_cast<milliseconds>(end - start).count();

        if (threads == 1) {
            base_time = duration;
            cout << "Thoi gian: " << duration << " ms (Moc co so)" << endl;
        } else {
            double speedup = base_time / duration;
            cout << "Thoi gian: " << duration << " ms -> TANG TOC (SPEEDUP): " << speedup << "x" << endl;
            // Tính toán hiệu suất luồng
            cout << "Hieu suat tren moi luong (Efficiency): " << (speedup / threads) * 100 << "%" << endl;
        }
        cout << "Ket qua: " << result << "\n" << endl;
    }

    return 0;
}