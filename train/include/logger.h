#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <vector>

using namespace std;

class Logger {
public:
    Logger(const string& log_file);
    ~Logger();
    
    void log_epoch(int epoch, float loss, float time_seconds = 0.0f);
    void log_message(const string& msg);
    void log_training_start(int num_epochs, int batch_size, float learning_rate);
    void log_training_end();
    void log_training_summary(float total_time, const vector<float>& epoch_times, 
                              float final_loss, size_t gpu_memory_used_mb, 
                              size_t gpu_memory_total_mb);
    
private:
    string get_timestamp();
    
    ofstream log_stream;
    string log_file;
};

#endif // LOGGER_H
