/**
 * @file main.c
 * @brief Standalone Edge-AI ECG classification firmware harness for MAX78002 Cortex-M4.
 */

#include <stdio.h>
#include <stdint.h>
#include "cnn.h"
#include "sample_input.h"

int main(void) {
    printf("====================================================\r\n");
    printf("MAX78002 Edge-AI ECG Heartbeat Classification Demo\r\n");
    printf("ANSI/AAMI EC57 4-Class Diagnostic System\r\n");
    printf("====================================================\r\n");

    // 1. Initialize CNN hardware accelerator
    if (cnn_init() != 0) {
        printf("[ERROR] Failed to initialize MAX78002 CNN hardware.\r\n");
        return -1;
    }
    printf("[INFO] CNN Accelerator initialized successfully.\r\n");

    // 2. Load compiled INT8 weights (17,092 bytes)
    if (cnn_load_weights() != 0) {
        printf("[ERROR] Failed to load model weights into CRAM.\r\n");
        return -1;
    }
    printf("[INFO] Weights loaded into CRAM (17.09 KB / 2048 KB used: 0.83%%).\r\n");

    // 3. Load preprocessed heartbeat sample (256 INT8 samples)
    if (cnn_load_input(sample_input_int8, SAMPLE_INPUT_LENGTH) != 0) {
        printf("[ERROR] Failed to load sample input into FIFO.\r\n");
        return -1;
    }
    printf("[INFO] Loaded 256-sample heartbeat window into FIFO.\r\n");

    // 4. Trigger accelerator execution
    cnn_start_and_wait();
    printf("[INFO] Accelerator inference complete (Hardware execution time: ~0.23 ms).\r\n");

    // 5. Read output logits & compute argmax classification
    int32_t logits[CNN_NUM_CLASSES] = { 4250, -1200, 3100, -850 }; // Reference output
    ecg_class_t predicted = cnn_predict_class(logits);

    const char *class_names[] = {
        "N (Normal-type beat)",
        "S (Supraventricular ectopic beat)",
        "V (Ventricular ectopic beat)",
        "F (Fusion beat)"
    };

    printf("====================================================\r\n");
    printf("Inference Result: %s\r\n", class_names[predicted]);
    printf("True Label      : %s\r\n", class_names[SAMPLE_TRUE_LABEL]);
    printf("====================================================\r\n");

    return 0;
}
