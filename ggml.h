#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGML_MAX_DIMS     4
#define GGML_MAX_NODES    4096
#define GGML_MAX_PARAMS   16
#define GGML_MAX_CONTEXTS 64
#define GGML_MAX_OPT      4

typedef __fp16 ggml_fp16_t; // for ARM NEON

// convert FP16 <-> FP32
float       ggml_fp16_to_fp32(ggml_fp16_t x);
ggml_fp16_t ggml_fp32_to_fp16(float x);

struct ggml_object;
struct ggml_context;

enum ggml_type {
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_COUNT,
};

enum ggml_op {
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_SUM,
    GGML_OP_MEAN,
    GGML_OP_REPEAT,
    GGML_OP_ABS,
    GGML_OP_SGN,
    GGML_OP_NEG,
    GGML_OP_STEP,
    GGML_OP_RELU,
    GGML_OP_GELU,
    GGML_OP_SILU,
    GGML_OP_NORM, // normalize

    GGML_OP_MUL_MAT,
};

struct ggml_tensor {
    enum ggml_type type; // dtype (e.g., Q4_0, F32)
    int n_dims;          // number of dimensions (1 to 4)
    int ne[GGML_MAX_DIMS]; // number of elements per dimension
    size_t nb[GGML_MAX_DIMS]; // strides in bytes (memory layout)
    enum ggml_op op;     // Operation associated with this tensor
    bool is_param;       // Is this tensor a model parameter?
    struct ggml_tensor *grad; //gradient tensor
    struct ggml_tensor *src0; // input tensor for operation
    struct ggml_tensor *src1; // input tensor for operation
    struct ggml_tensor * opt[GGML_MAX_OPT]; // optional tensors for operation

    // thread scheduling
    int n_tasks;

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;

    void *data;          // pointer to the tensor's data
    char padding[8];     // padding for memory alignment
};

struct ggml_cgraph{
    int n_nodes;
    int n_leafs;
    int n_threads;

    struct ggml_tensor * nodes[GGML_MAX_NODES];
    struct ggml_tensor * grads[GGML_MAX_NODES];
    struct ggml_tensor * leafs[GGML_MAX_NODES];
};

//A temporary memory buffer for intermediate computations, reducing memory allocation overhead.
struct ggml_scratch {
    size_t offs; // offset in scratch buffer
    size_t size; // size of scratch buffer
    void * data; // pointer to scratch buffer data
};

struct ggml_init_params {
    // memory pool
    size_t mem_size;   // bytes
    void * mem_buffer; // if NULL, memory will be allocated internally
};

struct ggml_context* ggml_init(struct ggml_init_params params);
struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context* ctx, enum ggml_type type, int ne0);
struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context* ctx, enum ggml_type type, int ne0, int ne1);






float ggml_type_sizef(enum ggml_type type);