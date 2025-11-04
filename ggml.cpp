#include "ggml.h"
#include "utils.h"

#include <arm_neon.h>
#include <stdatomic.h>

typedef float ggml_float;


#define QK 32 // quantization block size
#define GGML_MEM_ALIGN 16
#define CACHE_LINE_SIZE 64
#define FLT_MAX __FLT_MAX__

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define GGML_FP16_TO_FP32(x) (x)
#define GGML
#define UNUSED(x) (void)(x)
#define GGML_COMPUTE_FP16_TO_FP32(x) (x)
#define GGML_COMPUTE_FP32_TO_FP16(x) (x)
#define GGML_FP16_TO_FP32(x) (x)
#define GGML_FP32_TO_FP16(x) (x)
#define ggml_perf_time_us()       0

//precomputed tables
static ggml_fp16_t table_gelu_f16[1 << 16];
static ggml_fp16_t table_silu_f16[1 << 16];
static ggml_fp16_t table_exp_f16[1 << 16];
static float table_f32_f16[1 << 16];

inline static void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i32(const int n, int32_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }

static inline bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/GGML_BLCK_SIZE[tensor->type] &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

static const ggml_float GELU_COEF_A    = 0.044715;
static const ggml_float SQRT_2_OVER_PI = 0.79788456080286535587989211986876;
inline static float ggml_gelu_f32(float x) {
    return 0.5*x*(1.0 + tanh(SQRT_2_OVER_PI*x*(1.0 + GELU_COEF_A*x*x)));
}
inline static float ggml_silu_f32(float x) {
    return x/(1.0 + exp(-x));
}

static const int GGML_BLCK_SIZE[GGML_TYPE_COUNT] = {
    QK,
    QK,
    1,
    1,
    1,
    1,
    1,
};

static const size_t GGML_TYPE_SIZE[GGML_TYPE_COUNT] = {
    sizeof(float  )   + QK/2,
    sizeof(float  )*2 + QK/2,
    sizeof(int8_t ),
    sizeof(int16_t),
    sizeof(int32_t),
    sizeof(ggml_fp16_t),
    sizeof(float  ),
};

float ggml_fp16_to_fp32(ggml_fp16_t x) {
    return GGML_FP16_TO_FP32(x);
}
ggml_fp16_t ggml_fp32_to_fp16(float x) {
    return GGML_FP32_TO_FP16(x);
}

float ggml_type_sizef(enum ggml_type type) {
    return ((float)(GGML_TYPE_SIZE[type]))/GGML_BLCK_SIZE[type];
}

size_t ggml_element_size(const struct ggml_tensor * tensor) {
    return GGML_TYPE_SIZE[tensor->type];
}

int ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

struct ggml_tensor* ggml_get_rows(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        struct ggml_tensor* b) {

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a->ne[0], b->ne[0]);

    result->op   = GGML_OP_GET_ROWS;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char* const data = static_cast<char*>(tensor->data);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            {
            } break;
        case GGML_TYPE_Q4_1:
            {
            } break;
        case GGML_TYPE_I8:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F32:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_COUNT:
            {
            } break;
    }

    return tensor;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ctx->scratch = ctx->scratch_save;

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor* ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char* const data = static_cast<char*>(tensor->data);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            {
            } break;
        case GGML_TYPE_Q4_1:
            {
            } break;
        case GGML_TYPE_I8:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F32:
            {
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_COUNT:
            {
            } break;
    }

    return tensor;
}

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ctx->scratch = ctx->scratch_save;

    ggml_set_i32(result, value);

    return result;
}

//context management

//manages a memory pool for tensors and other objects, with a linked list of ggml_object entries
struct ggml_context {
    size_t mem_size; // size of memory pool
    void* mem_buffer; // pointer to memory pool
    bool mem_buffer_owned; // does the context own the memory pool?

    int n_objects; // number of allocated objects in the context

    struct ggml_object* objects_begin; // linked list of allocated objects
    struct ggml_object* objects_end; 

    struct ggml_scratch scratch; // current scratch buffer
    struct ggml_scratch scratch_save;
};

//tracks whether a context is in use
struct ggml_context_container {
    bool used;
    struct ggml_context context;
};

//// compute types
enum ggml_task_type {
    GGML_TASK_INIT = 0,
    GGML_TASK_COMPUTE,
    GGML_TASK_FINALIZE,
};

struct ggml_compute_params {
    enum ggml_task_type type;

    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;
};

//global state holding up to GGML_MAX_CONTEXTS contexts.
struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
};

static struct ggml_state g_state;
static atomic_int g_state_barrier = 0;

// barrier via spin lock
inline static void ggml_critical_section_start(void) {
    int processing = atomic_fetch_add(&g_state_barrier, 1);

    while (processing > 0) {
        // wait for other threads to finish
        atomic_fetch_sub(&g_state_barrier, 1);
        processing = atomic_fetch_add(&g_state_barrier, 1);
    }
}

inline static void ggml_critical_section_end(void) {
    atomic_fetch_sub(&g_state_barrier, 1);
}

struct ggml_context* ggml_init(struct ggml_init_params params) {
    // make this function thread safe
    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            ggml_fp16_t ii;
            for (int i = 0; i < (1 << 16); ++i) {
                uint16_t ui = i;
                memcpy(&ii, &ui, sizeof(ii));
                const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
                table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
                table_exp_f16[i]  = GGML_FP32_TO_FP16(exp(f));
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            printf("%s: GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        // initialize g_state
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            g_state = (struct ggml_state) {
                /*.contexts =*/ { { 0 } },
            };

            for (int i = 0; i < GGML_MAX_CONTEXTS; ++i) {
                g_state.contexts[i].used = false;
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            printf("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        is_first_call = false;
    }

    // find non-used context in g_state
    struct ggml_context * ctx = NULL;

    for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
        if (!g_state.contexts[i].used) {
            g_state.contexts[i].used = true;
            ctx = &g_state.contexts[i].context;

            printf("%s: found unused context %d\n", __func__, i);
            break;
        }
    }

    if (ctx == NULL) {
        printf("%s: no unused context found\n", __func__);

        ggml_critical_section_end();

        return NULL;
    }

    *ctx = (struct ggml_context) {
        /*.mem_size         =*/ params.mem_size,
        /*.mem_buffer       =*/ params.mem_buffer ? params.mem_buffer : malloc(params.mem_size),
        /*.mem_buffer_owned =*/ params.mem_buffer ? false : true,
        /*.n_objects        =*/ 0,
        /*.objects_begin    =*/ NULL,
        /*.objects_end      =*/ NULL,
        /*.scratch          =*/ { 0, 0, NULL, },
        /*.scratch_save     =*/ { 0, 0, NULL, },
    };

    printf("%s: context initialized\n", __func__);

    ggml_critical_section_end();

    return ctx;
}

// Each ggml_object node knows:
struct ggml_object {
    size_t offs; // where its data lives (offs)
    size_t size; // how big it is (size)
    struct ggml_object* next; // Next object in linked list 
    char padding[8];
};
static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

//tensor creation
struct ggml_tensor* ggml_new_tensor_impl(
        struct ggml_context* ctx,
        enum   ggml_type type,
        int    n_dims,
        const int* ne,
        void*  data) {

    // always insert objects at the end of the context's memory pool
    struct ggml_object* obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    size_t size_needed = 0;

    if (data == NULL) {
        size_needed += GGML_TYPE_SIZE[type]*(ne[0]/GGML_BLCK_SIZE[type]);
        for (int i = 1; i < n_dims; i++) {
            size_needed *= ne[i];
        }
        // align to GGML_MEM_ALIGN
        size_needed = ((size_needed + GGML_MEM_ALIGN - 1)/GGML_MEM_ALIGN)*GGML_MEM_ALIGN;
    }

    char* const mem_buffer = (char*) ctx->mem_buffer;
    struct ggml_object* const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

    if (ctx->scratch.data == NULL || data != NULL) {
        size_needed += sizeof(struct ggml_tensor);

        if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
            printf("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
            return NULL;
        }

        *obj_new = (struct ggml_object) {
            .offs = cur_end + GGML_OBJECT_SIZE,
            .size = size_needed,
            .next = NULL,
        };
    } else {
        if (ctx->scratch.offs + size_needed > ctx->scratch.size) {
            printf("%s: not enough space in the scratch memory\n", __func__);
            return NULL;
        }

        if (cur_end + sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE > ctx->mem_size) {
            printf("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE, ctx->mem_size);
            return NULL;
        }

        data = (char* const) ctx->scratch.data + ctx->scratch.offs;

        *obj_new = (struct ggml_object) {
            .offs = cur_end + GGML_OBJECT_SIZE,
            .size = sizeof(struct ggml_tensor),
            .next = NULL,
        };

        //printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

        ctx->scratch.offs += size_needed;
    }

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;


    struct ggml_tensor * const result = (struct ggml_tensor *)(mem_buffer + obj_new->offs);

    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.n_dims       =*/ n_dims,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.is_param     =*/ false,
        /*.grad         =*/ NULL,
        /*.src0         =*/ NULL,
        /*.src1         =*/ NULL,
        /*.opt          =*/ { NULL },
        /*.n_tasks      =*/ 0,
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
        /*.data         =*/ data == NULL ? (void *)(result + 1) : data,
        /*.pad          =*/ { 0 },
    };

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = GGML_TYPE_SIZE[type];
    result->nb[1] = result->nb[0]*(result->ne[0]/GGML_BLCK_SIZE[type]);
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}

struct ggml_tensor* ggml_new_tensor(
        struct ggml_context* ctx,
        enum   ggml_type type,
        int    n_dims,
        const int* ne) {
    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

struct ggml_tensor * ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0) {
    return ggml_new_tensor(ctx, type, 1, &ne0);
}

struct ggml_tensor * ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1) {
    const int ne[2] = { ne0, ne1 };
    return ggml_new_tensor(ctx, type, 2, ne);
}

struct ggml_tensor * ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1,
        int    ne2) {
    const int ne[3] = { ne0, ne1, ne2 };
    return ggml_new_tensor(ctx, type, 3, ne);
}

struct ggml_tensor * ggml_view_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        size_t                offset) {
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, &ne0, (char *) a->data + offset);

    result->op   = GGML_OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}


struct ggml_tensor * ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        size_t                nb1,
        size_t                offset) {

    const int ne[GGML_MAX_DIMS] = { ne0, ne1, 1, 1 };

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, (char *) a->data + offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    result->op   = GGML_OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; 

    return result;
}


struct ggml_tensor* ggml_view_tensor(
        struct ggml_context* ctx,
        const struct ggml_tensor* src) {
    return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);
}


struct ggml_tensor * ggml_scale_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool inplace) {

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op   = GGML_OP_SCALE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_scale(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_scale_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, true);
}



struct ggml_tensor * ggml_reshape(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, b->n_dims, b->ne, a->data);

    result->op   = GGML_OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1) {

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    const int ne[2] = { ne0, ne1 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a->data);

    result->op   = GGML_OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2) {

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    const int ne[3] = { ne0, ne1, ne2 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a->data);

    result->op   = GGML_OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}





struct ggml_tensor * ggml_scale_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool inplace) {

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op   = GGML_OP_SCALE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_scale(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_scale_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, true);
}




struct ggml_tensor * ggml_cpy_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool inplace) {

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    // make a view of the destination
    struct ggml_tensor * result = ggml_view_tensor(ctx, b);

    result->op   = GGML_OP_CPY;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor* ggml_cpy(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        struct ggml_tensor* b) {
    return ggml_cpy_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_cpy_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_cpy_impl(ctx, a, b, true);
}


struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    int ne[GGML_MAX_DIMS];
    int nb[GGML_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    result->op   = GGML_OP_PERMUTE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}





struct ggml_tensor* ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src) {
    return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, NULL);
}



int ggml_nelements(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return (ggml_nelements(tensor)*GGML_TYPE_SIZE[tensor->type])/GGML_BLCK_SIZE[tensor->type];
}

int ggml_blck_size(enum ggml_type type) {
    return GGML_BLCK_SIZE[type];
}

size_t ggml_type_size(enum ggml_type type) {
    return GGML_TYPE_SIZE[type];
}

void * ggml_get_data(const struct ggml_tensor* tensor) {
    return tensor->data;
}

size_t ggml_used_mem(const struct ggml_context * ctx) {
    return ctx->objects_end->offs + ctx->objects_end->size;
}

void ggml_free(struct ggml_context * ctx) {
    ggml_critical_section_start();

    bool found = false;

    for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
        if (&g_state.contexts[i].context == ctx) {
            g_state.contexts[i].used = false;

            printf("%s: context %d with %d objects has been freed. memory used = %zu\n",
                    __func__, i, ctx->n_objects, ctx->objects_end->offs + ctx->objects_end->size);

            if (ctx->mem_buffer_owned) {
                free(ctx->mem_buffer);
            }

            found = true;
            break;
        }
    }

    if (!found) {
        printf("%s: context not found\n", __func__);
    }

    ggml_critical_section_end();
}

////// ggml operations

struct ggml_tensor* ggml_add_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_ADD;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_add_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_add_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_add_impl(ctx, a, b, true);
}



struct ggml_tensor * ggml_silu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_SILU;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_silu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_silu_impl(ctx, a, false);
}

struct ggml_tensor * ggml_silu_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_silu_impl(ctx, a, true);
}


struct ggml_tensor * ggml_diag_mask_inf(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);
    struct ggml_tensor * b = ggml_new_i32(ctx, n_past);

    result->op   = GGML_OP_DIAG_MASK_INF;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_soft_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op   = GGML_OP_SOFT_MAX;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor* ggml_norm_impl(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_NORM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor* ggml_norm(
        struct ggml_context* ctx,
        struct ggml_tensor* a) {
    return ggml_norm_impl(ctx, a, false);
}

struct ggml_tensor* ggml_mul_impl(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        struct ggml_tensor* b,
        bool inplace) {

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_MUL;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor* ggml_mul(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        struct ggml_tensor* b) {
    return ggml_mul_impl(ctx, a, b, false);
}

struct ggml_tensor* ggml_mul_inplace(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        struct ggml_tensor* b) {
    return ggml_mul_impl(ctx, a, b, true);
}

struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    const int ne[4] = { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, MIN(a->n_dims, b->n_dims), ne);

    result->op   = GGML_OP_MUL_MAT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

static inline bool ggml_are_same_shape(const struct ggml_tensor* t0, const struct ggml_tensor* t1) {
    return
        (t0->ne[0] == t1->ne[0] ) &&
        (t0->ne[1] == t1->ne[1] ) &&
        (t0->ne[2] == t1->ne[2] ) &&
        (t0->ne[3] == t1->ne[3] );
}

struct ggml_tensor* ggml_repeat(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        struct ggml_tensor* b) {

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    if (ggml_are_same_shape(a, b) && !is_node) {
        return a;
    }

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

    result->op   = GGML_OP_REPEAT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}


struct ggml_tensor* ggml_rope(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past,
        int                   n_dims,
        int                   mode) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }
    struct ggml_tensor* result = ggml_view_tensor(ctx, a);

    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ((int32_t *) b->data)[0] = n_past;
    ((int32_t *) b->data)[1] = n_dims;
    ((int32_t *) b->data)[2] = mode;

    result->op   = GGML_OP_ROPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}




///ggml computation graph and forward pass

static void ggml_compute_forward_add_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    if (nb10 == sizeof(float)) {
        const int j0 = (n/nth)*ith;
        const int j1 = ith == nth - 1 ? n : (n/nth)*(ith + 1);

        for (int j = j0; j < j1; j++) {
            ggml_vec_add_f32(nc,
                    (float *) ((char *) dst->data  + j*nb1),
                    (float *) ((char *) src0->data + j*nb01),
                    (float *) ((char *) src1->data + j*nb11));
        }
    } else {
        // src1 is not contiguous
        for (int j = ith; j < n; j += nth) {
            float * dst_ptr  = (float *) ((char *) dst->data  + j*nb1);
            float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
            for (int i = 0; i < nc; i++) {
                float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);

                dst_ptr[i] = src0_ptr[i] + *src1_ptr;
            }
        }
    }
}

static void ggml_compute_forward_add(
        const struct ggml_compute_params* params,
        const struct ggml_tensor* src0,
        const struct ggml_tensor* src1,
        struct ggml_tensor* dst) 
{
    ggml_compute_forward_add_f32(params, src0, src1, dst);
} 


static void ggml_compute_forward(struct ggml_compute_params* params, struct ggml_tensor* tensor) {
    switch (tensor->op) {
        case GGML_OP_NONE:
            {
                // do nothing
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor->src0, tensor);
            } break;
        case GGML_OP_PERMUTE:
            {
                ggml_compute_forward_permute(params, tensor->src0);
            } break;
        case GGML_OP_RESHAPE:
            {
                ggml_compute_forward_reshape(params, tensor->src0, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor->src0, tensor);
            } break;
        case GGML_OP_SILU:
            {
                ggml_compute_forward_silu(params, tensor->src0, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor->src0, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor->src0, tensor);
            } break;
    }
}


static void ggml_visit_parents(struct ggml_cgraph* cgraph, struct ggml_tensor* node) {
    if (node->grad == NULL) {
        // this usually happens when we generate intermediate nodes from constants in the backward pass
        // it can also happen during forward pass, if the user performs computations with constants
        if (node->op != GGML_OP_NONE) {
            //GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return;
        }
    }

    for (int i = 0; i < cgraph->n_leafs; i++) {
        if (cgraph->leafs[i] == node) {
            return;
        }
    }

    if (node->src0) {
        ggml_visit_parents(cgraph, node->src0);
    }

    if (node->src1) {
        ggml_visit_parents(cgraph, node->src1);
    }

    for (int i = 0; i < GGML_MAX_OPT; ++i) {
        if (node->opt[i]) {
            ggml_visit_parents(cgraph, node->opt[i]);
        }
    }

    if (node->op == GGML_OP_NONE && node->grad == NULL) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {

        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->grads[cgraph->n_nodes] = node->grad;
        cgraph->n_nodes++;
    }
}

static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand) {
    if (!expand) {
        cgraph->n_nodes = 0;
        cgraph->n_leafs = 0;
    }

    const int n0 = cgraph->n_nodes;
    UNUSED(n0);

    ggml_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    printf("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
    }
}

void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true);
}

struct ggml_cgraph ggml_build_forward(struct ggml_tensor* tensor) {
    struct ggml_cgraph result = {
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.n_threads    =*/ 0,
        /*.work_size    =*/ 0,
        /*.work         =*/ NULL,
        /*.nodes        =*/ { NULL },
        /*.grads        =*/ { NULL },
        /*.leafs        =*/ { NULL },
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
    };

    ggml_build_forward_impl(&result, tensor, false);
    return result;
}