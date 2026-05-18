#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>
#include <iomanip>

using namespace std;

// ==============================================================================
// 1. CÁC HÀM THUẬT TOÁN ĐỂ KIỂM THỬ
// ==============================================================================

// Phiên bản Chuẩn (Tuần tự) - Dùng làm Thước đo độ chính xác (Ground Truth)
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

// Phiên bản Tối ưu (Song song Task Tiling) - Cần được kiểm tra
float dtw_parallel(const vector<float>& A, const vector<float>& B, int block_size) {
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
                        int i_end = min(n, (bx + 1) * block_size); // Ép chống tràn mảng
                        int j_start = by * block_size + 1;
                        int j_end = min(m, (by + 1) * block_size); // Ép chống tràn mảng

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

// ==============================================================================
// 2. HỆ THỐNG KIỂM THỬ (TESTING FRAMEWORK)
// ==============================================================================

// Hàm so sánh Float với Epsilon (Xử lý sai số làm tròn của CPU)
bool is_equal(float a, float b, float epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

// Biến toàn cục đếm số lượng Test
int tests_passed = 0;
int tests_failed = 0;

// Hàm chạy một Test Case cụ thể
void run_test(string test_name, int size, int block_size, bool identical_arrays = false) {
    vector<float> A(size), B(size);
    mt19937 rng(1337); // Cố định seed để dễ debug nếu có lỗi
    uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int i = 0; i < size; ++i) {
        A[i] = dist(rng);
        B[i] = identical_arrays ? A[i] : dist(rng);
    }

    float seq_result = dtw_sequential(A, B);
    float par_result = dtw_parallel(A, B, block_size);

    cout << left << setw(40) << test_name << " | ";
    if (is_equal(seq_result, par_result)) {
        cout << "\033[1;32m[PASS]\033[0m" << endl; // In màu Xanh lá
        tests_passed++;
    } else {
        cout << "\033[1;31m[FAIL]\033[0m" << endl; // In màu Đỏ
        cout << "   -> Mong doi (Seq): " << seq_result << " | Thuc te (Par): " << par_result << endl;
        tests_failed++;
    }
}

// ==============================================================================
// 3. MAIN: THỰC THI KIỂM THỬ
// ==============================================================================

int main() {
    omp_set_num_threads(4); // Dùng 4 luồng để test (mốc hiệu năng tốt nhất của bạn)
    cout << "==================================================" << endl;
    cout << " BAT DAU KIEM THU THUAT TOAN DTW SONG SONG" << endl;
    cout << "==================================================\n" << endl;

    // --- Nhóm 1: Sanity Checks (Kiểm tra logic cơ bản) ---
    cout << "--- Nhom 1: Kiem tra logic co ban ---" << endl;
    run_test("Mang Giong Nhau (Ket qua phai = 0)", 1000, 64, true);
    run_test("Kich thuoc rat nho (N < Block Size)", 10, 64, false);
    run_test("Block Size = 1 (Test chia nho cuc dai)", 100, 1, false);
    cout << endl;

    // --- Nhóm 2: Edge Cases (Các trường hợp dễ dính lỗi tràn mảng) ---
    cout << "--- Nhom 2: Edge Cases (Goc & Bien) ---" << endl;
    run_test("N vua dung bang Block Size (64x64)", 64, 64, false);
    run_test("Le 1 phan tu (N = 65, Block = 64)", 65, 64, false); // Block viền rất mỏng
    run_test("So nguyen to (N = 1013, Block = 32)", 1013, 32, false);
    run_test("Block Size le (N = 1000, Block = 17)", 1000, 17, false);
    cout << endl;

    // --- Nhóm 3: Fuzz Testing (Stress Test ngẫu nhiên) ---
    cout << "--- Nhom 3: Fuzz Testing (Stress Test ngau nhien) ---" << endl;
    int num_fuzz_tests = 20;
    mt19937 rand_fuzz(42);
    uniform_int_distribution<int> size_dist(500, 2000);
    uniform_int_distribution<int> block_dist(8, 128);

    for (int i = 1; i <= num_fuzz_tests; ++i) {
        int r_size = size_dist(rand_fuzz);
        int r_block = block_dist(rand_fuzz);
        string test_name = "Fuzz Test #" + to_string(i) + " (N=" + to_string(r_size) + ", Blk=" + to_string(r_block) + ")";
        run_test(test_name, r_size, r_block, false);
    }
    cout << endl;

    // --- TỔNG KẾT ---
    cout << "==================================================" << endl;
    cout << " TONG KET KIEM THU:" << endl;
    cout << " - Tong so Test : " << (tests_passed + tests_failed) << endl;
    cout << " - \033[1;32mPASSED\033[0m         : " << tests_passed << endl;
    cout << " - \033[1;31mFAILED\033[0m         : " << tests_failed << endl;
    
    if (tests_failed == 0) {
        cout << "\n \033[1;32m>>> XUAT SAC! THUAT TOAN SONG SONG CHINH XAC 100% <<<\033[0m" << endl;
    } else {
        cout << "\n \033[1;31m>>> CO LOI! Kiem tra lai logic phan chia Block hoac bien gioi (Boundary) <<<\033[0m" << endl;
    }
    cout << "==================================================" << endl;

    return 0;
}