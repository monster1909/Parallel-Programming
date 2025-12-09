CXX = g++
CXXFLAGS = -O3 -Wall -fopenmp
INCLUDES = -I./include

# Thư mục đích
BIN_DIR = .
SRC_DIR = src
LIBSVM_DIR = src/libsvm

# Target files
TARGET_TRAIN = train_svm
TARGET_TEST = test_svm

# Sources
SRCS_COMMON = $(SRC_DIR)/svm_wrapper.cpp $(LIBSVM_DIR)/svm.cpp
SRCS_TRAIN = $(SRC_DIR)/main_train.cpp
SRCS_TEST = $(SRC_DIR)/main_test.cpp

all: $(TARGET_TRAIN) $(TARGET_TEST)

$(TARGET_TRAIN):
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS_TRAIN) $(SRCS_COMMON) -o $(BIN_DIR)/$(TARGET_TRAIN)

$(TARGET_TEST):
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS_TEST) $(SRCS_COMMON) -o $(BIN_DIR)/$(TARGET_TEST)

clean:
	rm -f $(TARGET_TRAIN) $(TARGET_TEST) *.o