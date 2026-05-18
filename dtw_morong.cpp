#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
#include <chrono>
#include <iomanip>

using namespace std;

// Hàm tính DTW tối ưu hóa với mảng 1 chiều
double dtw_distance(const vector<double>& s, const vector<double>& t) {
    int n = s.size();
    int m = t.size();
    
    // Sử dụng mảng 1 chiều giả lập 2 chiều để tối ưu cache
    vector<double> dtw((n + 1) * (m + 1), numeric_limits<double>::infinity());

    #define DTW(i, j) dtw[(i) * (m + 1) + (j)]

    DTW(0, 0) = 0;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            double cost = abs(s[i - 1] - t[j - 1]);
            DTW(i, j) = cost + min({ DTW(i - 1, j),      // Insertion
                                     DTW(i, j - 1),      // Deletion
                                     DTW(i - 1, j - 1) }); // Match
        }
    }
    return DTW(n, m);
}

void run_benchmark(int num_threads, int num_db, const vector<double>& input, const vector<vector<double>>& database) {
    double min_dist = numeric_limits<double>::infinity();
    int best_idx = -1;

    // Thiết lập số luồng
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    auto start = chrono::high_resolution_clock::now();

    // Nếu num_threads == 0 thì chạy tuần tự, ngược lại chạy song song
    if (num_threads == 0) {
        for (int i = 0; i < num_db; ++i) {
            double dist = dtw_distance(input, database[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
    } else {
        #pragma omp parallel
        {
            double local_min = numeric_limits<double>::infinity();
            int local_best = -1;

            #pragma omp for
            for (int i = 0; i < num_db; ++i) {
                double dist = dtw_distance(input, database[i]);
                if (dist < local_min) {
                    local_min = dist;
                    local_best = i;
                }
            }

            // Cập nhật kết quả chung một cách an toàn
            #pragma omp critical
            {
                if (local_min < min_dist) {
                    min_dist = local_min;
                    best_idx = local_best;
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    if (num_threads == 0)
        cout << left << setw(15) << "Tuan tu" << ": " << time_taken << " ms" << endl;
    else
        cout << left << setw(15) << (to_string(num_threads) + " luong") << ": " << time_taken << " ms" << endl;
}

int main() {
    // Cấu hình thử nghiệm
    const int NUM_DB = 1000;    // 1000 chuỗi trong DB
    const int SEQ_LEN = 150;    // Độ dài mỗi chuỗi

    cout << "--- Bat dau so sanh DTW (DB size: " << NUM_DB << ", Length: " << SEQ_LEN << ") ---" << endl;

    // Khởi tạo dữ liệu giả lập
    vector<double> input_seq(SEQ_LEN);
    for(int i=0; i<SEQ_LEN; ++i) input_seq[i] = i % 10;

    vector<vector<double>> database(NUM_DB, vector<double>(SEQ_LEN));
    for(int i=0; i<NUM_DB; ++i) {
        for(int j=0; j<SEQ_LEN; ++j) {
            database[i][j] = (j + i) % 10; 
        }
    }

    // Chạy các trường hợp
    run_benchmark(0, NUM_DB, input_seq, database); // Tuần tự
    run_benchmark(2, NUM_DB, input_seq, database); // 2 luồng
    run_benchmark(4, NUM_DB, input_seq, database); // 4 luồng
    run_benchmark(8, NUM_DB, input_seq, database); // 8 luồng

    cout << "--------------------------------------------------------" << endl;

    return 0;
}