#ifndef TIMER_H
#define TIMER_H
#include <chrono>

class Timer {
    std::chrono::high_resolution_clock::time_point start_t;
public:
    Timer();
    void start();
    double get_elapsed_seconds();
};
#endif