
CXX      := g++
CXXFLAGS := -std=c++11 -O3 -fopenmp -Wall -I./include

TARGET   := autoencoder

SRCS     := $(wildcard src/*.cpp) \
            $(wildcard src/layers/*.cpp) \
            $(wildcard src/utils/*.cpp)

OBJS     := $(SRCS:.cpp=.o)

all: $(TARGET)
	@echo "Build successful! Executable created: ./$(TARGET)"

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Cleaned up build files."

info:
	@echo "Source files:"
	@echo $(SRCS)
	@echo "Object files:"
	@echo $(OBJS)


.PHONY: all clean info
