#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <ctime>

class Logger {
private:
    std::string log_file;
    std::ofstream file;
    
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        localtime_r(&time_t_now, &tm_now);
        
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_now);
        return std::string(buffer);
    }
    
public:
    Logger(const std::string& filename) : log_file(filename) {
        file.open(log_file, std::ios::out | std::ios::app);
        if (!file.is_open()) {
            std::cerr << "[WARNING] Could not open log file: " << log_file << std::endl;
        }
    }
    
    ~Logger() {
        if (file.is_open()) {
            file.close();
        }
    }
    
    void log_message(const std::string& message) {
        if (file.is_open()) {
            file << "[" << get_timestamp() << "] " << message << std::endl;
            file.flush();
        }
    }
    
    void log_training_start(int num_epochs, int batch_size, float learning_rate) {
        if (file.is_open()) {
            file << "\n========================================\n";
            file << "Training Started: " << get_timestamp() << "\n";
            file << "========================================\n";
            file << "Number of epochs: " << num_epochs << "\n";
            file << "Batch size: " << batch_size << "\n";
            file << "Learning rate: " << std::fixed << std::setprecision(6) << learning_rate << "\n";
            file << "========================================\n\n";
            file.flush();
        }
    }
    
    void log_epoch(int epoch, float loss, float time_seconds) {
        if (file.is_open()) {
            file << "Epoch " << std::setw(3) << epoch 
                 << " | Loss: " << std::fixed << std::setprecision(6) << loss
                 << " | Time: " << std::setprecision(2) << time_seconds << "s\n";
            file.flush();
        }
    }
    
    void log_training_end() {
        if (file.is_open()) {
            file << "\n========================================\n";
            file << "Training Ended: " << get_timestamp() << "\n";
            file << "========================================\n\n";
            file.flush();
        }
    }
    
    void log_training_summary(float total_time, const std::vector<float>& epoch_times, 
                             float final_loss, size_t used_mem_mb, size_t total_mem_mb) {
        if (file.is_open()) {
            file << "\n========================================\n";
            file << "       TRAINING SUMMARY\n";
            file << "========================================\n";
            file << "Total training time: " << std::fixed << std::setprecision(2) 
                 << total_time << " seconds\n";
            file << "Number of epochs: " << epoch_times.size() << "\n";
            
            if (!epoch_times.empty()) {
                float avg_time = 0.0f;
                for (float t : epoch_times) avg_time += t;
                avg_time /= epoch_times.size();
                file << "Average time per epoch: " << std::setprecision(2) 
                     << avg_time << " seconds\n";
            }
            
            file << "Final loss: " << std::setprecision(6) << final_loss << "\n";
            file << "GPU Memory Used: " << used_mem_mb << " MB / " << total_mem_mb << " MB\n";
            
            if (total_mem_mb > 0) {
                file << "GPU Memory Usage: " << std::setprecision(1) 
                     << (100.0f * used_mem_mb / total_mem_mb) << "%\n";
            }
            
            file << "========================================\n\n";
            file.flush();
        }
    }
};

#endif // LOGGER_H
