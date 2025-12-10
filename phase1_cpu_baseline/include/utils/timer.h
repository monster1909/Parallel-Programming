#ifndef TIMER_H
#define TIMER_H
#include <chrono>
#include <string>
#include <map>
#include <vector>

class Timer {
    std::chrono::high_resolution_clock::time_point start_t;
public:
    Timer();
    void start();
    double get_elapsed_seconds();
};

// New detailed timer for layer-by-layer timing
class DetailedTimer {
private:
    std::map<std::string, std::chrono::duration<double, std::milli>> durations;

public:
    DetailedTimer();
    void reset();
    // Trả về thời điểm bắt đầu để đo thủ công
    std::chrono::high_resolution_clock::time_point start_point(); 
    // Ghi lại thời gian từ điểm bắt đầu đã cho đến hiện tại
    void record_duration(const std::string& layer_name, const std::chrono::high_resolution_clock::time_point& start);
    double get_total_time_ms() const;
    void print_timing_report();
};

#endif