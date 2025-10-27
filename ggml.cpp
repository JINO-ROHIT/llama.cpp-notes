#include "ggml.h"
#include <arm_neon.h>

#define GGML_FP16_TO_FP32(x) (x)

float ggml_fp16_to_fp32(ggml_fp16_t x){
    return GGML_FP16_TO_FP32(x);
}