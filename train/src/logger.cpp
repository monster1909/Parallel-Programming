#include "../include/logger.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

using namespace std;

Logger::Logger(const string& log_file) : log_file(log_file) {
    log_stream.open(log_file, ios::out | ios::app);
    
    if (!log_stream.is_open()) {
        cerr << "[Logger] Error: Could not open log file: " << log_file << endl;
    } else {
        log_message("=== Logger initialized ===");
    }
}

Logger::~Logger() {
    if (log_stream.is_open()) {
        log_message("=== Logger closed ===");
        log_stream.close();
    }
}

string Logger::get_timestamp() {
    time_t now = time(nullptr);
    tm* ltm = localtime(&now);
    
    ostringstream oss;
    oss << setfill('0')
        << setw(4) << (1900 + ltm->tm_year) << "-"
        << setw(2) << (1 + ltm->tm_mon) << "-"
        << setw(2) << ltm->tm_mday << " "
        << setw(2) << ltm->tm_hour << ":"
        << setw(2) << ltm->tm_min << ":"
        << setw(2) << ltm->tm_sec;
    
    return oss.str();
}

void Logger::log_epoch(int epoch, float loss, float time_seconds) {
    ostringstream oss;
    oss << "Epoch " << setw(3) << epoch 
        << " | Loss: " << fixed << setprecision(6) << loss;
    
    if (time_seconds > 0.0f) {
        oss << " | Time: " << fixed << setprecision(2) << time_seconds << "s";
    }
    
    log_message(oss.str());
    
    // Also print to console
    cout << "[" << get_timestamp() << "] " << oss.str() << endl;
}

void Logger::log_message(const string& msg) {
    if (log_stream.is_open()) {
        log_stream << "[" << get_timestamp() << "] " << msg << endl;
        log_stream.flush();
    }
}

void Logger::log_training_start(int num_epochs, int batch_size, float learning_rate) {
    ostringstream oss;
    oss << "Training started: "
        << "epochs=" << num_epochs 
        << ", batch_size=" << batch_size
        << ", lr=" << fixed << setprecision(6) << learning_rate;
    
    log_message(oss.str());
    cout << "[" << get_timestamp() << "] " << oss.str() << endl;
}

void Logger::log_training_end() {
    log_message("Training completed");
    cout << "[" << get_timestamp() << "] Training completed" << endl;
}

void Logger::log_training_summary(float total_time, const vector<float>& epoch_times, 
                                  float final_loss, size_t gpu_memory_used_mb, 
                                  size_t gpu_memory_total_mb) {
    ostringstream oss;
    oss << "\n========================================\n";
    oss << "TRAINING SUMMARY\n";
    oss << "========================================\n";
    oss << "Total Training Time: " << fixed << setprecision(2) << total_time << " seconds\n";
    oss << "Final Reconstruction Loss: " << fixed << setprecision(6) << final_loss << "\n";
    oss << "\nTraining Time Per Epoch:\n";
    for (size_t i = 0; i < epoch_times.size(); i++) {
        oss << "  Epoch " << setw(3) << (i+1) << ": " << fixed << setprecision(2) 
            << epoch_times[i] << " seconds\n";
    }
    oss << "\nGPU Memory Usage:\n";
    oss << "  Used: " << gpu_memory_used_mb << " MB\n";
    oss << "  Total: " << gpu_memory_total_mb << " MB\n";
    if (gpu_memory_total_mb > 0) {
        oss << "  Usage: " << fixed << setprecision(2) 
            << (100.0f * gpu_memory_used_mb / gpu_memory_total_mb) << "%\n";
    } else {
        oss << "  Usage: N/A (could not get memory info)\n";
    }
    oss << "\nSample Reconstructed Images:\n";
    oss << "  (Check logs directory for sample_original_*.ppm and sample_reconstructed_*.ppm files)\n";
    oss << "========================================\n";
    
    log_message(oss.str());
    cout << oss.str() << endl;
}