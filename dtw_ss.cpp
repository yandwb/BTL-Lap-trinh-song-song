#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <omp.h>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// PHIÊN BẢN SONG SONG KHÔNG CHIA BLOCK (Cell-level Tasking)
float dtw_cell_task_openmp(const vector<float>& A, const vector<float>& B) {
    int n = A.size();
    int m = B.size();
    int cols = m + 1;

    vector<float> dtw((n + 1) * cols, INFINITY);
    dtw[0 * cols + 0] = 0.0f;

    // LẤY CON TRỎ THÔ (RAW POINTER) ĐỂ OPENMP HIỂU ĐƯỢC ĐỊA CHỈ BỘ NHỚ
    float* dtw_ptr = dtw.data();

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= m; ++j) {
                    
                    // TẠO TASK CHO TỪNG Ô MỘT
                    // Sử dụng dtw_ptr thay vì dtw trong mệnh đề depend
                    #pragma omp task depend(in: dtw_ptr[(i - 1) * cols + j], \
                                                dtw_ptr[i * cols + (j - 1)], \
                                                dtw_ptr[(i - 1) * cols + (j - 1)]) \
                                     depend(out: dtw_ptr[i * cols + j]) \
                                     shared(dtw_ptr, A, B) \
                                     firstprivate(i, j, cols)
                    {
                        float cost = std::abs(A[i - 1] - B[j - 1]);
                        float insert_cost = dtw_ptr[(i - 1) * cols + j];     
                        float delete_cost = dtw_ptr[i * cols + (j - 1)];     
                        float match_cost  = dtw_ptr[(i - 1) * cols + (j - 1)]; 
                        dtw_ptr[i * cols + j] = cost + fminf(insert_cost, fminf(delete_cost, match_cost));
                    }
                }
            }
        }
    }
    return dtw[n * cols + m];
}

int main() {
    // GIẢM SIZE XUỐNG 500 ĐỂ MÁY TÍNH KHÔNG BỊ TREO DO OVERHEAD
    const int SIZE = 500; 
    
    cout << "===========================================" << endl;
    cout << " TEST: SONG SONG TUNG O (KHONG CHIA BLOCK)" << endl;
    cout << " Kich thuoc mang: " << SIZE << "x" << SIZE << " (Tao ra 250,000 Tasks)" << endl;
    cout << "===========================================\n" << endl;

    vector<float> A(SIZE), B(SIZE);
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (int i = 0; i < SIZE; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    vector<int> thread_counts = {1, 2, 4, 8};
    double base_time = 0;

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        
        auto start = high_resolution_clock::now();
        float result = dtw_cell_task_openmp(A, B);
        auto end = high_resolution_clock::now();
        
        // CÁCH TÍNH THỜI GIAN CHUẨN XÁC VÀ AN TOÀN TRÊN MỌI COMPILER
        std::chrono::duration<double, std::milli> ms_double = end - start;
        double duration = ms_double.count();

        cout << "--- " << threads << " Luong ---" << endl;
        cout << "Ket qua: " << result << endl;
        
        if (threads == 1) {
            base_time = duration;
            cout << "Thoi gian: " << duration << " ms (Lam moc)\n" << endl;
        } else {
            double speedup = base_time / duration;
            cout << "Thoi gian: " << duration << " ms -> Tang toc: " << speedup << "x\n" << endl;
        }
    }

    return 0;
}