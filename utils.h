#pragma once

#include <string>
#include <map>
#include <thread>
#include <random>

struct gpt_params {
    int32_t seed      = -1;
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict = 128; // new tokens to predict

    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;

    int32_t n_batch = 8; // batch size for prompt processing

    std::string model = "models/7B/ggml-model-quant.bin";
    std::string prompt;
};

struct gpt_vocab{
    using id = int32_t; // basically typedef int32_t id;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

int64_t ggml_time_us(void);
bool gpt_params_parse(int argc, char ** argv, gpt_params & params);
std::string gpt_random_prompt(std::mt19937 &rng);

size_t ggml_quantize_q4_0(float * src, void * dst, int n, int k, int qk, int64_t * hist);
size_t ggml_quantize_q4_1(float * src, void * dst, int n, int k, int qk, int64_t * hist);

std::vector<gpt_vocab::id> llama_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos);

gpt_vocab::id gpt_sample_top_k_top_p(
        const gpt_vocab & vocab,
        const float * logits,
        int    top_k,
        double top_p,
        double temp,
        std::mt19937 & rng);