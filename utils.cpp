#include <string>
#include <cassert>

int64_t ggml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}

// the algorithm
// Takes a floating-point input array src of size n.
// Processes it in blocks of size k, with each block divided into nb = k / qk sub-blocks, where each sub-block contains qk elements.
// For each sub-block:

// Computes a scale factor d based on the maximum absolute value (amax) of the qk elements.
// Quantizes the qk elements to 4-bit values (0 to 15, representing -8 to 7 shifted by +8).
// Packs pairs of 4-bit values into bytes (Lower 4 bits (0–3): First value (vi0). Upper 4 bits (4–7): Second value (vi1, shifted left by 4)).

// Stores the scale factors and quantized values in the dst buffer.
// Updates a histogram (hist) of the quantized values.
// Returns the total size of the quantized data in bytes.

size_t ggml_quantize_q4_0(float * src, void * dst, int n, int k, int qk, int64_t * hist) {
    const int nb = k / qk;
    const size_t row_size = nb*(sizeof(float) + sizeof(uint8_t)*qk/2);

    assert(k % qk == 0);

    uint8_t pp[qk/2];

    char * pdst = (char *) dst;

    for (int j = 0; j < n; j += k) {
        float   * pd = (float *)   (pdst + (j/k)*row_size);
        uint8_t * pb = (uint8_t *) (pd + nb);

        for (int i = 0; i < nb; i++) {
            float amax = 0.0f; // absolute max

            {
                for (int l = 0; l < qk; l++) {
                    const float v = src[j + i*qk + l];
                    amax = std::max(amax, fabsf(v));
                }

                const float d = amax / ((1 << 3) - 1);
                const float id = d ? 1.0f/d : 0.0f;

                pd[i] = d;

                for (int l = 0; l < qk; l += 2) {
                    const float v0 = (src[j + i*qk + l + 0])*id;
                    const float v1 = (src[j + i*qk + l + 1])*id;

                    const uint8_t vi0 = ((int8_t) (round(v0))) + 8;
                    const uint8_t vi1 = ((int8_t) (round(v1))) + 8;

                    assert(vi0 >= 0 && vi0 < 16);
                    assert(vi1 >= 0 && vi1 < 16);

                    hist[vi0]++;
                    hist[vi1]++;

                    pp[l/2] = vi0 | (vi1 << 4);
                }

                memcpy(pb + i*qk/2, pp, sizeof(pp));
            }
        }
    }

    return (n/k)*row_size;
}

size_t ggml_quantize_q4_1(float * src, void * dst, int n, int k, int qk, int64_t * hist) {
    return 1;
}
