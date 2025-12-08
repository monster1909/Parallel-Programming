#include "utils/timer.h"

Timer::Timer() { start(); }
void Timer::start() { start_t = std::chrono::high_resolution_clock::now(); }
double Timer::get_elapsed_seconds() {
    auto end_t = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_t - start_t;
    return elapsed.count();
}