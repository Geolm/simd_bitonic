

/*
To use this library, do this in *one* C or C++ file:
    #define __SIMD_BITONIC_IMPLEMENTATION__
    #include "simd_bitonic.h"
    
COMPILATION
    
DOCUMENTATION

    int simd_small_sort_max();

        Returns the number of float at max that the library can sort

    int simd_small_sort(float* array, int element_count);

        Sort a small array of float
        Returns an error code
            SIMD_SORT_OK                Everything ok
            SIMD_SORT_TOOMANYELEMENTS   There are too many float to sort in the array, use simd_sort_max() to get the max


    void simd_merge_sort(float* array, int element_count);

        Sort an array of float using a mix merge sort and bitonic sort

*/


#ifndef __SIMD_BITONIC__
#define __SIMD_BITONIC__


//----------------------------------------------------------------------------------------------------------------------
// Prototypes
//----------------------------------------------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

#define SIMD_SORT_OK                (1)
#define SIMD_SORT_TOOMANYELEMENTS   (2)


int simd_small_sort_max();
int simd_small_sort(float* array, int element_count);
void simd_merge_sort(float* array, int element_count);

#ifdef __cplusplus
}
#endif

//----------------------------------------------------------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------------------------------------------------------

#ifdef __SIMD_BITONIC_IMPLEMENTATION__
#undef __SIMD_BITONIC_IMPLEMENTATION__

#include <assert.h>
#include <string.h>
#include <stdlib.h>

// positive infinity float hexadecimal value
#define FLOAT_PINF (0x7F800000) 

//----------------------------------------------------------------------------------------------------------------------
// Neon
//----------------------------------------------------------------------------------------------------------------------
#if defined(__ARM_NEON) && defined(__ARM_NEON__)

#include <arm_neon.h>

#define ALIGN_STRUCT(x) __attribute__((aligned(x)))
#define SIMD_VECTOR_WIDTH (4)

typedef float32x4_t simd_vector;

//----------------------------------------------------------------------------------------------------------------------
static inline float32x4_t vblendq_f32(float32x4_t _a, float32x4_t _b, const char imm8)
{
    const uint32_t ALIGN_STRUCT(16) data[4] = 
    {
        ((imm8) & (1 << 0)) ? UINT32_MAX : 0,
        ((imm8) & (1 << 1)) ? UINT32_MAX : 0,
        ((imm8) & (1 << 2)) ? UINT32_MAX : 0,
        ((imm8) & (1 << 3)) ? UINT32_MAX : 0
    };

    uint32x4_t mask = vld1q_u32(data);
    return vbslq_f32(mask, _b, _a);
}

//----------------------------------------------------------------------------------------------------------------------
static inline float32x4_t simd_sort_1V(float32x4_t input)
{
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, 0xA);
    }
    {
        float32x4_t perm_neigh = __builtin_shufflevector(input, input, 3, 2, 1, 0);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, 0xC);
    }
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, 0xA);
    }
    return input;
}

//----------------------------------------------------------------------------------------------------------------------
static inline float32x4_t simd_aftermerge_1V(float32x4_t input)
{
    {
        float32x4_t perm_neigh = __builtin_shufflevector(input, input, 2, 3, 0, 1);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, 0xC);
    }
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, 0xA);
    }
    return input;
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_permute_minmax_2V(float32x4_t *a, float32x4_t *b)
{
    float32x4_t perm_neigh = __builtin_shufflevector(*b, *b, 3, 2, 1, 0);
    float32x4_t perm_neigh_min = vminq_f32(*a, perm_neigh);
    float32x4_t perm_neigh_max = vmaxq_f32(*a, perm_neigh);
    *a = perm_neigh_min;
    *b = perm_neigh_max;
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_minmax_2V(float32x4_t *a, float32x4_t *b)
{
    float32x4_t perm_neigh_min = vminq_f32(*a, *b);
    float32x4_t perm_neigh_max = vmaxq_f32(*a, *b);
    *a = perm_neigh_min;
    *b = perm_neigh_max;
}

//----------------------------------------------------------------------------------------------------------------------
static inline float32x4_t simd_load_partial(const float* array, int index, int element_count)
{
    int array_index = SIMD_VECTOR_WIDTH * index;
    if (element_count == SIMD_VECTOR_WIDTH)
        return vld1q_f32(array + array_index);
    
    static const uint32_t float_positive_inf = FLOAT_PINF;
    float32x4_t result = vmovq_n_f32(*(float*)&float_positive_inf);
    result = vsetq_lane_f32(array[array_index + 0], result, 0);
    
    if (element_count > 1)
        result = vsetq_lane_f32(array[array_index + 1], result, 1);
    
    if (element_count > 2)
        result = vsetq_lane_f32(array[array_index + 2], result, 2);
    
    return result;
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_store_partial(float* array, float32x4_t a, int index, int element_count)
{   
    int array_index = SIMD_VECTOR_WIDTH * index;
    if (element_count == SIMD_VECTOR_WIDTH)
    {
        vst1q_f32(array + array_index, a);
    }
    else
    {
        array[array_index] = vgetq_lane_f32(a, 0);
        
        if (element_count > 1)
            array[array_index+1] = vgetq_lane_f32(a, 1);
        
        if (element_count > 2)
            array[array_index+2] = vgetq_lane_f32(a, 2);
    }
}

//----------------------------------------------------------------------------------------------------------------------
static inline float32x4_t simd_load_vector(const float* array, int index)
{
    return vld1q_f32(array + SIMD_VECTOR_WIDTH * index);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_store_vector(float* array, float32x4_t a, int index)
{
    vst1q_f32(array + SIMD_VECTOR_WIDTH * index, a);
}

#else

//----------------------------------------------------------------------------------------------------------------------
// AVX
//----------------------------------------------------------------------------------------------------------------------

#include <immintrin.h>
#include <stdint.h>

#define SIMD_VECTOR_WIDTH (8)

typedef __m256 simd_vector;

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector _mm256_swap(__m256 input)
{
    __m128 lo = _mm256_extractf128_ps(input, 0);
    __m128 hi = _mm256_extractf128_ps(input, 1);
    return _mm256_setr_m128(hi, lo);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_minmax_2V(__m256* a, __m256* b)
{
    __m256 a_copy = *a;
    *a = _mm256_min_ps(*b, a_copy);
    *b = _mm256_max_ps(*b, a_copy);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_permute_minmax_2V(__m256* a, __m256* b)
{
    __m256 swap = _mm256_swap(*b);
    __m256 perm_neigh = _mm256_permute_ps(swap, _MM_SHUFFLE(0, 1, 2, 3));
    __m256 perm_neigh_min = _mm256_min_ps(*a, perm_neigh);
    __m256 perm_neigh_max = _mm256_max_ps(*a, perm_neigh);
    *a = perm_neigh_min;
    *b = perm_neigh_max;
}

//----------------------------------------------------------------------------------------------------------------------
static inline __m256 simd_sort_1V(simd_vector input)
{
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(0, 1, 2, 3));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    {
        __m256 swap = _mm256_swap(input);
        __m256 perm_neigh = _mm256_permute_ps(swap, _MM_SHUFFLE(0, 1, 2, 3));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xF0);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(1, 0, 3, 2));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    return input;
}

//----------------------------------------------------------------------------------------------------------------------
static inline __m256 simd_aftermerge_1V(simd_vector a)
{
    {
        __m256 swap = _mm256_swap(a);
        __m256 perm_neigh = _mm256_permute_ps(swap, _MM_SHUFFLE(3, 2, 1, 0));
        __m256 perm_neigh_min = _mm256_min_ps(a, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(a, perm_neigh);
        a = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xF0);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(a, _MM_SHUFFLE(1, 0, 3, 2));
        __m256 perm_neigh_min = _mm256_min_ps(a, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(a, perm_neigh);
        a = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(a, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(a, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(a, perm_neigh);
        a = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    return a;
}

//----------------------------------------------------------------------------------------------------------------------
static inline __m256i loadstore_mask(int element_count)
{
    return _mm256_set_epi32(0, 
                            (element_count>6) ? 0xffffffff : 0,
                            (element_count>5) ? 0xffffffff : 0,
                            (element_count>4) ? 0xffffffff : 0,
                            (element_count>3) ? 0xffffffff : 0,
                            (element_count>2) ? 0xffffffff : 0,
                            (element_count>1) ? 0xffffffff : 0,
                            (element_count>0) ? 0xffffffff : 0);
}

//----------------------------------------------------------------------------------------------------------------------
static inline __m256 simd_load_partial(const float* array, int index, int element_count)
{
    if (element_count == SIMD_VECTOR_WIDTH)
        return _mm256_loadu_ps(array + index * SIMD_VECTOR_WIDTH);
    
    __m256 inf_mask = _mm256_cvtepi32_ps(_mm256_set_epi32(FLOAT_PINF,
                                       (element_count>6) ? 0 : FLOAT_PINF,
                                       (element_count>5) ? 0 : FLOAT_PINF,
                                       (element_count>4) ? 0 : FLOAT_PINF,
                                       (element_count>3) ? 0 : FLOAT_PINF,
                                       (element_count>2) ? 0 : FLOAT_PINF,
                                       (element_count>1) ? 0 : FLOAT_PINF,
                                       (element_count>0) ? 0 : FLOAT_PINF));
    
    __m256 a = _mm256_maskload_ps(array + index * SIMD_VECTOR_WIDTH, loadstore_mask(element_count));
    return _mm256_or_ps(a, inf_mask);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_store_partial(float* array, __m256 a, int index, int element_count)
{
    if (element_count == SIMD_VECTOR_WIDTH)
    {
        _mm256_storeu_ps(array + index * SIMD_VECTOR_WIDTH, a);
    }
    else
    {
        _mm256_maskstore_ps(array + index * SIMD_VECTOR_WIDTH, loadstore_mask(element_count), a);
    }
}

//----------------------------------------------------------------------------------------------------------------------
static inline __m256 simd_load_vector(const float* array, int index)
{
    return _mm256_loadu_ps(array + SIMD_VECTOR_WIDTH * index);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_store_vector(float* array, __m256 a, int index)
{
    _mm256_storeu_ps(array + SIMD_VECTOR_WIDTH * index, a);
}

//----------------------------------------------------------------------------------------------------------------------
#endif // AVX
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_2V(simd_vector *a, simd_vector *b)
{
    simd_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_3V(simd_vector *a, simd_vector *b, simd_vector *c)
{
    simd_minmax_2V(a, c);
    simd_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_4V(simd_vector *a, simd_vector *b, simd_vector *c, simd_vector *d)
{
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_5V(simd_vector *a, simd_vector *b, simd_vector *c, simd_vector *d, simd_vector* e)
{
    simd_minmax_2V(a, e);
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_6V(simd_vector *a, simd_vector *b, simd_vector *c, simd_vector *d, simd_vector* e, simd_vector* f)
{
    simd_minmax_2V(a, e);
    simd_minmax_2V(b, f);
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    simd_minmax_2V(e, f);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
    *f = simd_aftermerge_1V(*f);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_7V(simd_vector *a, simd_vector *b, simd_vector *c, simd_vector *d, simd_vector* e, simd_vector* f, simd_vector *g)
{
    simd_minmax_2V(a, e);
    simd_minmax_2V(b, f);
    simd_minmax_2V(c, g);
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    simd_minmax_2V(e, g);
    simd_minmax_2V(e, f);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
    *f = simd_aftermerge_1V(*f);
    *g = simd_aftermerge_1V(*g);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_8V(simd_vector *a, simd_vector *b, simd_vector *c, simd_vector *d, simd_vector* e, simd_vector* f, simd_vector *g, simd_vector* h)
{
    simd_minmax_2V(a, e);
    simd_minmax_2V(b, f);
    simd_minmax_2V(c, g);
    simd_minmax_2V(d, h);

    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);

    simd_minmax_2V(e, g);
    simd_minmax_2V(f, h);
    simd_minmax_2V(e, f);
    simd_minmax_2V(g, h);

    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
    *f = simd_aftermerge_1V(*f);
    *g = simd_aftermerge_1V(*g);
    *h = simd_aftermerge_1V(*h);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_2V(simd_vector* a, simd_vector* b)
{
    *a = simd_sort_1V(*a);
    *b = simd_sort_1V(*b);
    simd_permute_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_3V(simd_vector* a, simd_vector* b, simd_vector* c)
{
    simd_sort_2V(a, b);
    *c = simd_sort_1V(*c);
    simd_permute_minmax_2V(b, c);
    simd_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_4V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d)
{
    simd_sort_2V(a, b);
    simd_sort_2V(c, d);
    simd_permute_minmax_2V(a, d);
    simd_permute_minmax_2V(b, c);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_5V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e)
{
    simd_sort_4V(a, b, c, d);
    *e = simd_sort_1V(*e);
    simd_permute_minmax_2V(d, e);
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_6V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f)
{
    simd_sort_4V(a, b, c, d);
    simd_sort_2V(e, f);
    simd_permute_minmax_2V(c, f);
    simd_permute_minmax_2V(d, e);
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    simd_minmax_2V(e, f);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
    *f = simd_aftermerge_1V(*f);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_7V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g)
{
    simd_sort_4V(a, b, c, d);
    simd_sort_3V(e, f, g);
    simd_permute_minmax_2V(c, f);
    simd_permute_minmax_2V(d, e);
    simd_permute_minmax_2V(b, g);
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    simd_minmax_2V(e, g);
    simd_minmax_2V(e, f);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
    *f = simd_aftermerge_1V(*f);
    *g = simd_aftermerge_1V(*g);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_8V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h)
{
    simd_sort_4V(a, b, c, d);
    simd_sort_4V(e, f, g, h);
    simd_permute_minmax_2V(a, h);
    simd_permute_minmax_2V(b, g);
    simd_permute_minmax_2V(c, f);
    simd_permute_minmax_2V(d, e);
    simd_minmax_2V(a, c);
    simd_minmax_2V(b, d);
    simd_minmax_2V(a, b);
    simd_minmax_2V(c, d);
    simd_minmax_2V(e, g);
    simd_minmax_2V(f, h);
    simd_minmax_2V(e, f);
    simd_minmax_2V(g, h);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
    *d = simd_aftermerge_1V(*d);
    *e = simd_aftermerge_1V(*e);
    *f = simd_aftermerge_1V(*f);
    *g = simd_aftermerge_1V(*g);
    *h = simd_aftermerge_1V(*h);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_9V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    *i = simd_sort_1V(*i);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    *i = simd_aftermerge_1V(*i);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_10V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i, simd_vector* j)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_2V(i, j);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_2V(i, j);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_11V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i, simd_vector* j, simd_vector* k)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_3V(i, j, k);
    simd_permute_minmax_2V(f, k);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_3V(i, j, k);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_12V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_4V(i, j, k, l);
    simd_permute_minmax_2V(e, l);
    simd_permute_minmax_2V(f, k);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_4V(i, j, k, l);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_13V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_5V(i, j, k, l, m);
    simd_permute_minmax_2V(d, m);
    simd_permute_minmax_2V(e, l);
    simd_permute_minmax_2V(f, k);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_5V(i, j, k, l, m);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_14V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_6V(i, j, k, l, m, n);
    simd_permute_minmax_2V(c, n);
    simd_permute_minmax_2V(d, m);
    simd_permute_minmax_2V(e, l);
    simd_permute_minmax_2V(f, k);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_6V(i, j, k, l, m, n);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_15V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_7V(i, j, k, l, m, n, o);
    simd_permute_minmax_2V(b, o);
    simd_permute_minmax_2V(c, n);
    simd_permute_minmax_2V(d, m);
    simd_permute_minmax_2V(e, l);
    simd_permute_minmax_2V(f, k);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_7V(i, j, k, l, m, n, o);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_16V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h, simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_8V(i, j, k, l, m, n, o, p);
    simd_permute_minmax_2V(a, p);
    simd_permute_minmax_2V(b, o);
    simd_permute_minmax_2V(c, n);
    simd_permute_minmax_2V(d, m);
    simd_permute_minmax_2V(e, l);
    simd_permute_minmax_2V(f, k);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_8V(i, j, k, l, m, n, o, p);
}

//----------------------------------------------------------------------------------------------------------------------
int simd_small_sort_max()
{
    return SIMD_VECTOR_WIDTH * 16;
}

//----------------------------------------------------------------------------------------------------------------------
int simd_small_sort(float* array, int element_count)
{
    if (element_count <= 1)
        return SIMD_SORT_OK;   

    const int last_vec_size = (element_count%SIMD_VECTOR_WIDTH) == 0 ? SIMD_VECTOR_WIDTH : (element_count%SIMD_VECTOR_WIDTH);
    if (element_count <= SIMD_VECTOR_WIDTH)
    {
        simd_vector a = simd_load_partial(array, 0, last_vec_size);
        a = simd_sort_1V(a);
        simd_store_partial(array, a, 0, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 2)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_partial(array, 1, last_vec_size);
        simd_sort_2V(&a, &b);
        simd_store_vector(array, a, 0);
        simd_store_partial(array, b, 1, last_vec_size);
        return SIMD_SORT_OK;
    }
    
    if (element_count <= SIMD_VECTOR_WIDTH * 3)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_partial(array, 2, last_vec_size);
        simd_sort_3V(&a, &b, &c);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_partial(array, c, 2, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 4)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_partial(array, 3, last_vec_size);
        simd_sort_4V(&a, &b, &c, &d);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_partial(array, d, 3, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 5)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_partial(array, 4, last_vec_size);
        simd_sort_5V(&a, &b, &c, &d, &e);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_partial(array, e, 4, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 6)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_partial(array, 5, last_vec_size);
        simd_sort_6V(&a, &b, &c, &d, &e, &f);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_partial(array, f, 5, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 7)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_partial(array, 6, last_vec_size);
        simd_sort_7V(&a, &b, &c, &d, &e, &f, &g);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_partial(array, g, 6, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 8)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_partial(array, 7, last_vec_size);
        simd_sort_8V(&a, &b, &c, &d, &e, &f, &g, &h);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_partial(array, h, 7, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 9)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_partial(array, 8, last_vec_size);
        simd_sort_9V(&a, &b, &c, &d, &e, &f, &g, &h, &i);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_partial(array, i, 8, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 10)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_vector(array, 8);
        simd_vector j = simd_load_partial(array, 9, last_vec_size);
        simd_sort_10V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_vector(array, i, 8);
        simd_store_partial(array, j, 9, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 11)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_vector(array, 8);
        simd_vector j = simd_load_vector(array, 9);
        simd_vector k = simd_load_partial(array, 10, last_vec_size);
        simd_sort_11V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_vector(array, i, 8);
        simd_store_vector(array, j, 9);
        simd_store_partial(array, k, 10, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 12)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_vector(array, 8);
        simd_vector j = simd_load_vector(array, 9);
        simd_vector k = simd_load_vector(array, 10);
        simd_vector l = simd_load_partial(array, 11, last_vec_size);
        simd_sort_12V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_vector(array, i, 8);
        simd_store_vector(array, j, 9);
        simd_store_vector(array, k, 10);
        simd_store_partial(array, l, 11, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 13)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_vector(array, 8);
        simd_vector j = simd_load_vector(array, 9);
        simd_vector k = simd_load_vector(array, 10);
        simd_vector l = simd_load_vector(array, 11);
        simd_vector m = simd_load_partial(array, 12, last_vec_size);
        simd_sort_13V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l, &m);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_vector(array, i, 8);
        simd_store_vector(array, j, 9);
        simd_store_vector(array, k, 10);
        simd_store_vector(array, l, 11);
        simd_store_partial(array, m, 12, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 14)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_vector(array, 8);
        simd_vector j = simd_load_vector(array, 9);
        simd_vector k = simd_load_vector(array, 10);
        simd_vector l = simd_load_vector(array, 11);
        simd_vector m = simd_load_vector(array, 12);
        simd_vector n = simd_load_partial(array, 13, last_vec_size);
        simd_sort_14V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l, &m, &n);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_vector(array, i, 8);
        simd_store_vector(array, j, 9);
        simd_store_vector(array, k, 10);
        simd_store_vector(array, l, 11);
        simd_store_vector(array, m, 12);
        simd_store_partial(array, n, 13, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 15)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_vector(array, 8);
        simd_vector j = simd_load_vector(array, 9);
        simd_vector k = simd_load_vector(array, 10);
        simd_vector l = simd_load_vector(array, 11);
        simd_vector m = simd_load_vector(array, 12);
        simd_vector n = simd_load_vector(array, 13);
        simd_vector o = simd_load_partial(array, 14, last_vec_size);
        simd_sort_15V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l, &m, &n, &o);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_vector(array, i, 8);
        simd_store_vector(array, j, 9);
        simd_store_vector(array, k, 10);
        simd_store_vector(array, l, 11);
        simd_store_vector(array, m, 12);
        simd_store_vector(array, n, 13);
        simd_store_partial(array, o, 14, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= SIMD_VECTOR_WIDTH * 16)
    {
        simd_vector a = simd_load_vector(array, 0);
        simd_vector b = simd_load_vector(array, 1);
        simd_vector c = simd_load_vector(array, 2);
        simd_vector d = simd_load_vector(array, 3);
        simd_vector e = simd_load_vector(array, 4);
        simd_vector f = simd_load_vector(array, 5);
        simd_vector g = simd_load_vector(array, 6);
        simd_vector h = simd_load_vector(array, 7);
        simd_vector i = simd_load_vector(array, 8);
        simd_vector j = simd_load_vector(array, 9);
        simd_vector k = simd_load_vector(array, 10);
        simd_vector l = simd_load_vector(array, 11);
        simd_vector m = simd_load_vector(array, 12);
        simd_vector n = simd_load_vector(array, 13);
        simd_vector o = simd_load_vector(array, 14);
        simd_vector p = simd_load_partial(array, 15, last_vec_size);
        simd_sort_16V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l, &m, &n, &o, &p);
        simd_store_vector(array, a, 0);
        simd_store_vector(array, b, 1);
        simd_store_vector(array, c, 2);
        simd_store_vector(array, d, 3);
        simd_store_vector(array, e, 4);
        simd_store_vector(array, f, 5);
        simd_store_vector(array, g, 6);
        simd_store_vector(array, h, 7);
        simd_store_vector(array, i, 8);
        simd_store_vector(array, j, 9);
        simd_store_vector(array, k, 10);
        simd_store_vector(array, l, 11);
        simd_store_vector(array, m, 12);
        simd_store_vector(array, n, 13);
        simd_store_vector(array, o, 14);
        simd_store_partial(array, p, 15, last_vec_size);
        return SIMD_SORT_OK;
    }
    return SIMD_SORT_TOOMANYELEMENTS;
}

//----------------------------------------------------------------------------------------------------------------------
void merge_arrays(float* array, int left, int middle, int right)
{
    int left_element_count = middle - left + 1;
    int right_element_count = right - middle;

    float* left_array = (float*) malloc(sizeof(float) * left_element_count);
    float* right_array = (float*) malloc(sizeof(float) * right_element_count);

    memcpy(left_array, array + left, sizeof(float) * left_element_count);
    memcpy(right_array, array + middle + 1, sizeof(float) * right_element_count);

    int left_index, right_index, output_index;
    left_index = 0;
    right_index = 0;
    output_index = left;

    while (left_index < left_element_count && right_index < right_element_count) 
    {
        if (left_array[left_index] < right_array[right_index]) 
        {
            array[output_index] = left_array[left_index];
            left_index++;
        } 
        else 
        {
            array[output_index] = right_array[right_index];
            right_index++;
        }
        output_index++;
    }

    while (left_index < left_element_count) 
    {
        array[output_index] = left_array[left_index];
        left_index++;
        output_index++;
    }

    while (right_index < right_element_count) 
    {
        array[output_index] = right_array[right_index];
        right_index++;
        output_index++;
    }

    free(left_array);
    free(right_array);
}

//----------------------------------------------------------------------------------------------------------------------
void merge_sort(float* array, int left, int right) 
{
    if (left < right)
    {
        int middle = left + (right - left) / 2;
        int left_element_count = middle - left + 1;
        int right_element_count = right - middle;

        // if both arrays are small enough, we use the simd bitonic sort
        if (left_element_count <= simd_small_sort_max() && 
            right_element_count <= simd_small_sort_max())
        {
            simd_small_sort(array + left, left_element_count);
            simd_small_sort(array + middle + 1, right_element_count);
        }
        else
        {
            merge_sort(array, left, middle);
            merge_sort(array, middle + 1, right);
        }
        merge_arrays(array, left, middle, right);
    }
}

//----------------------------------------------------------------------------------------------------------------------
void simd_merge_sort(float* array, int element_count)
{
    if (element_count <= simd_small_sort_max())
    {
        simd_small_sort(array, element_count);
    }
    else
    {
        merge_sort(array, 0, element_count - 1);
    }
}

#endif
#endif
