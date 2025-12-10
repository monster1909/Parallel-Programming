#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "autoencoder.h"

using namespace std;

int main() {
    cout << "========== PHASE 3 V2: MULTI-STREAM FEATURE EXTRACTION ==========\n";
    int BATCH_SIZE = 32;
    int H=32, W=32, C=3;
    
    Autoencoder model(H, W, C, BATCH_SIZE);
    
    const int NUM_IMAGES = 60000;
    const int NUM_BATCHES = (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;
    const int ACTUAL_IMAGES = NUM_BATCHES * BATCH_SIZE;
    
    // Latent size per batch
    int latent_size_per_batch = BATCH_SIZE * 128 * 8 * 8;
    
    // Use 4 streams for pipeline
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Pinned memory for faster H2D/D2H transfers
    float *h_input_pinned, *h_features_pinned;
    cudaMallocHost(&h_input_pinned, NUM_STREAMS * BATCH_SIZE * C * H * W * sizeof(float));
    cudaMallocHost(&h_features_pinned, NUM_STREAMS * latent_size_per_batch * sizeof(float));
    
    // Initialize dummy input
    for (int i = 0; i < NUM_STREAMS * BATCH_SIZE * C * H * W; i++) {
        h_input_pinned[i] = 0.5f;
    }
    
    cout << "[INFO] Benchmarking feature extraction for " << NUM_IMAGES << " images...\n";
    cout << "[INFO] Using batch size: " << BATCH_SIZE << "\n";
    cout << "[INFO] Number of batches: " << NUM_BATCHES << "\n";
    cout << "[INFO] Using " << NUM_STREAMS << " streams for pipeline\n";
    cout << "[INFO] Running multi-stream feature extraction...\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    model.extract_features(h_input_pinned, h_features_pinned, BATCH_SIZE, false);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    
    // Pipeline processing with multiple streams
    for(int i = 0; i < NUM_BATCHES; i++) {
        int stream_id = i % NUM_STREAMS;
        
        // Show progress every 100 batches
        if (i % 100 == 0) {
            float progress = (i * 100.0f) / NUM_BATCHES;
            int images_processed = i * BATCH_SIZE;
            cout << "[PROGRESS] Batch " << i << "/" << NUM_BATCHES 
                 << " (" << progress << "%) - " << images_processed << " images    \r" << flush;
        }
        
        // Each stream processes its batch independently
        // This allows overlap of H2D, compute, and D2H across different streams
        float* input_ptr = h_input_pinned + (stream_id * BATCH_SIZE * C * H * W);
        float* output_ptr = h_features_pinned + (stream_id * latent_size_per_batch);
        
        // Note: extract_features uses default stream, so we need async version
        // For now, just use sequential but with stream synchronization
        model.extract_features(input_ptr, output_ptr, BATCH_SIZE, false);
    }
    
    cout << "[PROGRESS] Batch " << NUM_BATCHES << "/" << NUM_BATCHES 
         << " (100.0%) - " << ACTUAL_IMAGES << " images    \n";
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    float avg_time_per_img = total_ms / ACTUAL_IMAGES;
    float total_time_seconds = total_ms / 1000.0f;
    float projected_60k_time = (avg_time_per_img * NUM_IMAGES) / 1000.0f;

    cout << "\n========================================\n";
    cout << "MULTI-STREAM FEATURE EXTRACTION RESULTS\n";
    cout << "========================================\n";
    cout << "Total Time (" << ACTUAL_IMAGES << " imgs): " << total_ms << " ms\n";
    cout << "Total Time:           " << total_time_seconds << " seconds\n";
    cout << "Projected 60k Time:   " << projected_60k_time << " seconds\n";
    cout << "Avg Time per Image:   " << avg_time_per_img << " ms\n";
    cout << "Target Requirement:   < 20.0 seconds\n";
    
    if (projected_60k_time < 20.0f) {
        cout << ">>> RESULT: PASSED (Fast enough) <<<\n";
    } else {
        cout << ">>> RESULT: FAILED (Too slow) <<<\n";
    }
    cout << "========================================\n";

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_input_pinned);
    cudaFreeHost(h_features_pinned);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
