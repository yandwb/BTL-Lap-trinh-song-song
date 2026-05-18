#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

// Hàm tính khoảng cách giữa hai điểm dữ liệu (Absolute difference cho dữ liệu 1D)
inline double calc_distance(double a, double b) {
    return std::abs(a - b);
}

/**
 * Hàm tính toán DTW tuần tự
 * Trả về chi phí cực tiểu để đồng bộ hai chuỗi A và B.
 */
double dtw_sequential(const vector<double>& A, const vector<double>& B) {
    int n = A.size();
    int m = B.size();
    
    int rows = n + 1;
    int cols = m + 1;

    // Khởi tạo ma trận DTW phẳng 1D với giá trị vô cực.
    // Indexing: dtw[i * cols + j] tương đương với dtw[i][j]
    vector<double> dtw(rows * cols, INFINITY);
    
    // Điều kiện biên ban đầu
    dtw[0 * cols + 0] = 0.0;

    // Lấp đầy ma trận bằng Quy hoạch động (Dynamic Programming)
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            // 1. Tính chi phí tại ô hiện tại
            double cost = calc_distance(A[i - 1], B[j - 1]);
            
            // 2. Lấy 3 giá trị kề trước đó
            double insert_cost = dtw[(i - 1) * cols + j];     // Di chuyển xuống (Insertion)
            double delete_cost = dtw[i * cols + (j - 1)];     // Di chuyển ngang (Deletion)
            double match_cost  = dtw[(i - 1) * cols + (j - 1)]; // Di chuyển chéo (Match)
            
            // 3. Tính toán chi phí tích lũy
            dtw[i * cols + j] = cost + min({insert_cost, delete_cost, match_cost});
        }
    }

    // In ma trận ra để kiểm tra (Tùy chọn debug)
    cout << "Ma tran DTW:" << endl;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cout << setw(5) << dtw[i * cols + j] << " ";
        }
        cout << endl;
    }

    // Kết quả cuối cùng nằm ở góc dưới cùng bên phải của ma trận
    return dtw[n * cols + m];
}

int main() {
    // Testcase: Hai chuỗi có hình dạng giống nhau nhưng lệch pha và độ dài khác nhau
    vector<double> sequence_A = {1, 3, 4, 9, 8, 2, 1, 5, 7, 3};
    vector<double> sequence_B = {1, 6, 2, 3, 0, 9, 4, 3, 6, 3};

    cout << "Chay thu nghiem DTW Tuan tu..." << endl;
    
    double dtw_distance = dtw_sequential(sequence_A, sequence_B);
    
    cout << "-----------------------------------" << endl;
    cout << "Khoang cach DTW cuoi cung: " << dtw_distance << endl;

    return 0;
}