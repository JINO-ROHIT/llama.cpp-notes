#pragma once

#include <string>
#include <map>

struct gpt_vocab{
    using id = int32_t; // basically typedef int32_t id;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

int64_t ggml_time_us(void);

size_t ggml_quantize_q4_0(float * src, void * dst, int n, int k, int qk, int64_t * hist);
size_t ggml_quantize_q4_1(float * src, void * dst, int n, int k, int qk, int64_t * hist);
