#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdio>

using namespace std;
using namespace std::chrono;

int main() {
    const int SIZE = 5000;
    int n = SIZE;
    int m = SIZE;
    int cols = m + 1;

    cout << "===========================================" << endl;
    cout << "THUAT TOAN DTW TUAN TU (SEQUENTIAL)" << endl;
    cout << "Kich thuoc mang: " << SIZE << "x" << SIZE << endl;
    cout << "===========================================\n" << endl;
    
    // ---------------------------------------------------------
    // BẮT ĐẦU ĐO: VÙNG KHÔNG THỂ SONG SONG (Setup & Init)
    // ---------------------------------------------------------
    auto start_seq_region = high_resolution_clock::now();

    vector<float> A(SIZE);
    vector<float> B(SIZE);
    
    // Sinh số ngẫu nhiên
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (int i = 0; i < SIZE; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // Cấp phát và khởi tạo ma trận DTW
    vector<float> dtw((n + 1) * cols, INFINITY);
    dtw[0 * cols + 0] = 0.0f;

    auto end_seq_region = high_resolution_clock::now();
    double time_setup = duration_cast<duration<double, milli>>(end_seq_region - start_seq_region).count();

    // ---------------------------------------------------------
    // BẮT ĐẦU ĐO: VÙNG CÓ THỂ SONG SONG (Lõi thuật toán)
    // ---------------------------------------------------------
    auto start_par_region = high_resolution_clock::now();

    // Lõi tính toán (Tương đương vùng được chia Block trong OpenMP)
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = std::abs(A[i - 1] - B[j - 1]);
            
            float insert_cost = dtw[(i - 1) * cols + j];     
            float delete_cost = dtw[i * cols + (j - 1)];     
            float match_cost  = dtw[(i - 1) * cols + (j - 1)]; 
            
            dtw[i * cols + j] = cost + fminf(insert_cost, fminf(delete_cost, match_cost));
        }
    }

    auto end_par_region = high_resolution_clock::now();
    double time_compute = duration_cast<duration<double, milli>>(end_par_region - start_par_region).count();

    // ---------------------------------------------------------
    // IN KẾT QUẢ VÀ THỐNG KÊ
    // ---------------------------------------------------------
    float final_result = dtw[n * cols + m];
    double total_time = time_setup + time_compute;

    cout << "Ket qua DTW: " << final_result << "\n" << endl;
    
    cout << "--- THONG KE THOI GIAN ---" << endl;
    cout << "1. Vung khong the song song (Khoi tao):  " << time_setup << " ms" << endl;
    cout << "2. Vung co the song song (Tinh toan):    " << time_compute << " ms" << endl;
    cout << "-> Tong thoi gian chay:                  " << total_time << " ms\n" << endl;

    // Tính phần trăm
    double percent_setup = (time_setup / total_time) * 100.0;
    double percent_compute = (time_compute / total_time) * 100.0;

    cout << "--- PHAN TICH AMDAHL ---" << endl;
    printf("Phan tram tuan tu (S):     %.3f %%\n", percent_setup);
    printf("Phan tram song song (P):   %.3f %%\n", percent_compute);

    return 0;
}