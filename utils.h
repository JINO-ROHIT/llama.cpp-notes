#pragma once

#include <string>

struct gpt_vocab{
    using id = int32_t; // basically typedef int32_t id;
    using token = std::string;
};

int64_t ggml_time_us(void);