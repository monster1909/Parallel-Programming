#ifndef CONFIG_H
#define CONFIG_H

// Global configuration
const int IMG_H = 32;
const int IMG_W = 32;
const int IMG_C = 3;

const int F1 = 256;   // conv1 out channels
const int F2 = 128;   // conv2 out channels

const int BATCH_SIZE = 32;
const int EPOCHS = 20;
const float LR = 0.001f;

#endif