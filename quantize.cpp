#include "ggml.h"
#include "utils.h"

#include <string>

#define QK 32; // block size for quantization

struct llama_hparams{
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // context size?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rotation   = 64;
    int32_t f16     = 1;   // use fp16
};

bool llama_model_quantize(const std::string &input_path,
                        const std::string &output_path,
                        int dtype)
{
    ggml_type type = GGML_TYPE_Q4_1;

    gpt_vocab vocab;

    printf("%s: loading model from %s", __func__, input_path.c_str()); //use c_str to work with printf

    return true;
}

int main(int argc, char **argv){

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    const int dtype = atoi(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    {
        const int64_t t_start_us = ggml_time_us();

        if (!llama_model_quantize(input_path, output_path, dtype)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, input_path.c_str());
            return 1;
        }
        t_quantize_us = ggml_time_us() - t_start_us;
    }

    {
        const int64_t t_main_end_us = ggml_time_us();
        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;

}