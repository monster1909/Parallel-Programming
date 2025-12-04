#ifndef TQDM_H
#define TQDM_H

#include <string>
#include <initializer_list>
#include <utility>
#include <chrono>

using namespace std;

class tqdm
{
private:
    size_t total;
    size_t bar_width;
    int refresh_rate;

    chrono::steady_clock::time_point start;
    chrono::steady_clock::time_point last_refresh;

    string join_metrics(initializer_list<pair<string, double>> metrics);

public:
    tqdm(size_t total, size_t bar_width = 75, int refresh_rate_ms = 40);

    void update(size_t current, initializer_list<pair<string, double>> metrics = {});
    
    void end(initializer_list<pair<string, double>> metrics = {});
};

#endif
