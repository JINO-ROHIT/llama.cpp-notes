#include "ggml.h"
#include "utils.h"

#include <arm_neon.h>
#include <stdatomic.h>

typedef float ggml_float;


#define QK 32 // quantization block size
#define GGML_MEM_ALIGN 16

#define GGML_FP16_TO_FP32(x) (x)
#define GGML
#define UNUSED(x) (void)(x)
#define GGML_COMPUTE_FP16_TO_FP32(x) (x)
#define GGML_COMPUTE_FP32_TO_FP16(x) (x)
#define GGML_FP16_TO_FP32(x) (x)
#define GGML_FP32_TO_FP16(x) (x)

//precomputed tables
static ggml_fp16_t table_gelu_f16[1 << 16];
static ggml_fp16_t table_silu_f16[1 << 16];
static ggml_fp16_t table_exp_f16[1 << 16];
static float table_f32_f16[1 << 16];

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

        data = ctx->scratch.data + ctx->scratch.offs;

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