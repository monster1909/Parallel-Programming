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
