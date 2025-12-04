#include "tqdm.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>

using namespace std;

tqdm::tqdm(size_t total, size_t bar_width, int refresh_rate)
    : total(total), bar_width(bar_width), refresh_rate(refresh_rate)
{
    start = chrono::steady_clock::now();
    last_refresh = start;
}

string tqdm::join_metrics(initializer_list<pair<string, double>> metrics)
{
    stringstream ss;
    bool first = true;

    for (auto &m : metrics)
    {
        if (!first)
            ss << "  ";
        first = false;
        ss << m.first << ": "
           << fixed << setprecision(4) << m.second;
    }

    return ss.str();
}

void tqdm::update(size_t current, initializer_list<pair<string, double>> metrics)
{
    if (current > total)
        current = total;

    auto now = chrono::steady_clock::now();
    double ms = chrono::duration<double, milli>(now - last_refresh).count();
    if (ms < refresh_rate && current < total)
        return;
    last_refresh = now;

    float progress = total == 0 ? 1.0f : float(current) / float(total);
    if (progress > 1.0f)
        progress = 1.0f;

    size_t filled = size_t(progress * bar_width);
    double elapsed = chrono::duration<double>(now - start).count();
    double speed = elapsed > 1e-12 ? current / (elapsed) : 0.0;
    double eta = (speed > 1e-12) ? (total - current) / (speed) : 0.0;

    string bar;
    bar.reserve(bar_width);
    for (size_t i = 0; i < bar_width; i++)
        bar += (i < filled ? "█" : " ");

    stringstream prefix_ss;
    prefix_ss << "\r "
              << setw(3) << int(progress * 100) << "% |"
              << bar << "| "
              << "ETA: " << fixed << setprecision(1) << eta << "s  ";
    string prefix = prefix_ss.str();

    if (speed < 1.0)
    {
        double s_per_item = (speed > 1e-12) ? (1.0 / speed) : 0.0;
        cout << prefix
             << "Speed: " << fixed << setprecision(2) << setw(4) << s_per_item << " s/it  "
             << join_metrics(metrics);
    }
    else
    {
        cout << prefix
             << "Speed: " << setw(3) << int(speed) << " it/s  "
             << join_metrics(metrics);
    }

    cout.flush();
}

void tqdm::end(initializer_list<pair<string, double>> metrics)
{
    update(total, metrics);
    cout << "\n";
}
