#include "utils/timer.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace std;

// Basic Timer Implementation
Timer::Timer() { start(); }
void Timer::start() { start_t = std::chrono::high_resolution_clock::now(); }
double Timer::get_elapsed_seconds() {
    auto end_t = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_t - start_t;
    return elapsed.count();
}

// DetailedTimer Implementation
DetailedTimer::DetailedTimer() { reset(); }

void DetailedTimer::reset() {
    durations.clear();
}

std::chrono::high_resolution_clock::time_point DetailedTimer::start_point() {
    return std::chrono::high_resolution_clock::now();
}

void DetailedTimer::record_duration(const std::string& layer_name, const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();

    durations[layer_name] += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
}

double DetailedTimer::get_total_time_ms() const {
    double total = 0.0;
    for (const auto& pair : durations) {
        total += pair.second.count();
    }
    return total;
}

void DetailedTimer::print_timing_report() {
    double total_ms = get_total_time_ms();
    
    // Order of layers for printing to match the Autoencoder structure
    vector<string> layer_order = {
        "Conv1", "ReLU1", "MaxPool1", 
        "Conv2", "ReLU2", "MaxPool2 (Latent)", 
        "DecodeConv1 (Conv3)", "ReLU_Dec1 (ReLU3)", "Upsample1", 
        "DecodeConv2 (Conv4)", "ReLU_Dec2 (ReLU4)", "Upsample2", 
        "FinalConv (Conv5)"
    };

    cout << "\n========================================" << endl;
    cout << "DETAILED LAYER TIMING (1 image)" << endl;
    cout << "========================================" << endl;

    cout << "\n===== FORWARD PASS START =====" << endl;
    cout << "\n===== TIME BREAKDOWN =====" << endl;
    
    // Determine max name length for clean formatting
    size_t max_len = 0;
    for (const string& name : layer_order) {
        max_len = max(max_len, name.length());
    }

    cout << fixed << setprecision(4);
    for (const string& layer_name : layer_order) {
        if (durations.count(layer_name)) {
            double ms = durations.at(layer_name).count();
            double percent = (ms / total_ms) * 100.0;
            cout << left << setw(max_len + 1) << layer_name + ":" 
                 << right << setw(10) << ms << " ms"
                 << right << setw(10) << "(" << setprecision(6) << percent << "%)" << endl;
        }
    }
    
    cout << "----------------------------------" << endl;
    cout << "TOTAL FORWARD TIME:" << right << setw(10) << total_ms << " ms" << endl;
    cout << "==================================" << endl;

}
