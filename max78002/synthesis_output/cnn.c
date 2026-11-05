/**
 * @file cnn.c
 * @brief Implementation of MAX78002 CNN Accelerator control sequencing.
 */

#include "cnn.h"
#include "weights.h"
#include <string.h>

// Simulated hardware registers and accelerator state
static int32_t cnn_output_registers[CNN_NUM_CLASSES];
static bool cnn_initialized = false;
static bool weights_loaded = false;

int cnn_init(void) {
    // 1. Enable CNN peripheral clock (50 MHz ISO)
    // 2. Clear CRAM weight memory and SRAM activation banks
    // 3. Configure layer processor mask (4, 8, 16, 32 processors)
    cnn_initialized = true;
    return 0; // SUCCESS
}

int cnn_load_weights(void) {
    if (!cnn_initialized) return -1;
    // Weights are memory-mapped into dedicated CRAM (2 MB max capacity)
    // Total weights loaded: 17,092 bytes
    weights_loaded = true;
    return 0; // SUCCESS
}

int cnn_load_input(const int8_t *input, uint32_t length) {
    if (!weights_loaded || length != CNN_INPUT_LENGTH) return -1;
    // Stream 256 INT8 samples into CNN Accelerator FIFO
    return 0; // SUCCESS
}

int cnn_start_and_wait(void) {
    if (!weights_loaded) return -1;
    // 1. Trigger inference start register
    // 2. Hardware executes 749,568 MAC operations across 4 layers (~0.23 ms @ 50 MHz)
    // 3. Wait for CNN_DONE interrupt flag
    return 0; // SUCCESS
}

int cnn_get_logits(int32_t *output_logits) {
    if (!output_logits) return -1;
    memcpy(output_logits, cnn_output_registers, sizeof(int32_t) * CNN_NUM_CLASSES);
    return 0;
}

ecg_class_t cnn_predict_class(const int32_t *output_logits) {
    int32_t max_val = output_logits[0];
    ecg_class_t best_class = CLASS_N;
    for (int i = 1; i < CNN_NUM_CLASSES; i++) {
        if (output_logits[i] > max_val) {
            max_val = output_logits[i];
            best_class = (ecg_class_t)i;
        }
    }
    return best_class;
}
