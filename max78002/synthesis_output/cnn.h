/**
 * @file cnn.h
 * @brief Driver interface for the Analog Devices MAX78002 CNN Accelerator.
 */

#ifndef CNN_H_
#define CNN_H_

#include <stdint.h>
#include <stdbool.h>

#define CNN_NUM_CLASSES  4
#define CNN_INPUT_LENGTH 256

// Class names matching ANSI/AAMI EC57 standard
typedef enum {
    CLASS_N = 0, // Normal-type beat
    CLASS_S = 1, // Supraventricular ectopic beat
    CLASS_V = 2, // Ventricular ectopic beat
    CLASS_F = 3  // Fusion beat
} ecg_class_t;

/**
 * @brief Initialize clock, power, and memory domains for the MAX78002 CNN Accelerator.
 */
int cnn_init(void);

/**
 * @brief Load quantized INT8 weights into the MAX78002 CRAM memory.
 */
int cnn_load_weights(void);

/**
 * @brief Load preprocessed INT8 1D ECG window into accelerator FIFO/SRAM.
 */
int cnn_load_input(const int8_t *input, uint32_t length);

/**
 * @brief Trigger hardware CNN inference and wait for accelerator interrupt / completion.
 */
int cnn_start_and_wait(void);

/**
 * @brief Unload raw output logits from accelerator registers.
 */
int cnn_get_logits(int32_t *output_logits);

/**
 * @brief Compute argmax class prediction from output logits.
 */
ecg_class_t cnn_predict_class(const int32_t *output_logits);

#endif // CNN_H_
