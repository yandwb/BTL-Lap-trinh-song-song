import matplotlib.pyplot as plt

# --- 1. Dữ liệu trích xuất từ bảng mới nhất ---
# Trục X: Số luồng
num_threads = [1, 2, 4, 8]

# Trục Y: Giá trị Speedup thực nghiệm (1x, 2.01282x, 2.512x, 1.18045x)
speedup_values = [1.00000, 2.01282, 2.51200, 1.18045]

# --- 2. Khởi tạo biểu đồ ---
plt.figure(figsize=(10, 6))

# --- 3. Vẽ biểu đồ đường thực nghiệm ---
# Vẫn giữ nguyên format: màu cam ('#f7941d'), đường liền nét, không có đường lý tưởng
plt.plot(num_threads, speedup_values, marker='o', color='#f7941d', linestyle='-', linewidth=3, markersize=10, label='Kết quả thực nghiệm')

# --- 4. Hiển thị giá trị speedup tại mỗi điểm ---
for i, (thread, speedup) in enumerate(zip(num_threads, speedup_values)):
    plt.annotate(f"{speedup:.2f}x", 
                 xy=(thread, speedup), 
                 xytext=(0, 10),  # Đẩy text lên trên điểm 10px để dễ nhìn
                 textcoords='offset points', 
                 ha='center', 
                 va='bottom', 
                 fontsize=11, 
                 fontweight='bold')

# --- 5. Cấu hình tiêu đề và nhãn trục ---
plt.title('Biểu đồ Speedup theo Số luồng (Dữ liệu cập nhật mới)', fontsize=16, fontweight='bold')
plt.xlabel('Số luồng', fontsize=14)
plt.ylabel('Speedup', fontsize=14)

# --- 6. Cấu hình trục X và Y ---
plt.xticks(num_threads, fontsize=12)
# Cập nhật khoảng chia trục Y (giá trị cao nhất là ~2.51, nên để mốc trên cùng là 3.0)
plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=12)

# --- 7. Thêm lưới để dễ quan sát ---
plt.grid(True, linestyle='--', color='#d3d3d3', alpha=0.8)

# --- 8. Tối ưu khoảng cách và hiển thị ---
plt.tight_layout()
plt.show()