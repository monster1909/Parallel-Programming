#include "tqdm.h"
#include <thread>
#include <chrono>
#include <iostream>

using namespace std;

int main()
{
    size_t total_steps = 100;
    size_t bar_width = 50;
    tqdm bar(total_steps, bar_width);

    double loss = 1.0;
    double lr = 0.1;
    double acc = 0.0;

    for (size_t i = 0; i <= total_steps; i++)
    {
        bar.update(i, {{"loss", loss}, {"lr", lr}, {"acc", acc}});

        loss *= 0.99;
        lr *= 0.9995;
        acc += 2;

        this_thread::sleep_for(chrono::milliseconds(250));
    }

    bar.end({{"final_loss", loss}, {"final_lr", lr}, {"final_acc", acc}});
    return 0;
}


