#include "utils/weight_init.h"
#include <cstdlib>

float frand(float a, float b) {
    return a + (b - a) * (rand() / (float)RAND_MAX);
}