# ==========================================
# Makefile cho dự án Autoencoder CPU
# ==========================================

# Trình biên dịch và các cờ (flags)
CXX      := g++
CXXFLAGS := -std=c++11 -O3 -fopenmp -Wall -I./include

# Tên file thực thi đầu ra
TARGET   := autoencoder

# Tự động tìm các file nguồn (.cpp) trong các thư mục
SRCS     := $(wildcard src/*.cpp) \
            $(wildcard src/layers/*.cpp) \
            $(wildcard src/utils/*.cpp)

# Chuyển đổi danh sách file .cpp thành file .o (object)
OBJS     := $(SRCS:.cpp=.o)

# --- Các quy tắc (Rules) ---

# Quy tắc mặc định (khi gõ 'make')
all: $(TARGET)
	@echo "Build successful! Executable created: ./$(TARGET)"

# Quy tắc liên kết (Linking)
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Quy tắc biên dịch (Compiling) từng file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Quy tắc dọn dẹp (xóa file .o và file thực thi)
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Cleaned up build files."

# Quy tắc hiển thị thông tin (để kiểm tra)
info:
	@echo "Source files:"
	@echo $(SRCS)
	@echo "Object files:"
	@echo $(OBJS)

.PHONY: all clean info