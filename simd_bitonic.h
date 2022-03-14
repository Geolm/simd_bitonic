

/*
To use this library, do this in *one* C or C++ file:
    #define __SIMD_BITONIC_IMPLEMENTATION__
    #include "simd_bitonic.h"
    
COMPILATION
    
DOCUMENTATION

    int simd_small_sort_max();

        Returns the number of float at max that the library can sort

    void simd_small_sort(float* array, int element_count);

        Sort a small array of float. Do nothing if there is too many elements in the array (more than simd_small_sort_max())

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

int simd_small_sort_max();
void simd_small_sort(float* array, int element_count);
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
static inline float32x4_t simd_sort_1V(float32x4_t input)
{
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vtrn2q_f32(perm_neigh_min, perm_neigh_max);
    }
    {
        float32x4_t perm_neigh = __builtin_shufflevector(input, input, 3, 2, 1, 0);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vextq_u64(perm_neigh_min, perm_neigh_max, 1);
    }
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vtrn2q_f32(perm_neigh_min, perm_neigh_max);
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
        input = vextq_u64(perm_neigh_min, perm_neigh_max, 1);
    }
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vtrn2q_f32(perm_neigh_min, perm_neigh_max);
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
    return _mm256_permute2f128_ps(input, input, _MM_SHUFFLE(0, 0, 1, 1));
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
static inline __m256 simd_load_vector(const float* array, int vector_index)
{
    return _mm256_loadu_ps(array + SIMD_VECTOR_WIDTH * vector_index);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_store_vector(float* array, __m256 a, int vector_index)
{
    _mm256_storeu_ps(array + SIMD_VECTOR_WIDTH * vector_index, a);
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
static inline void simd_aftermerge_16V(simd_vector *a, simd_vector *b, simd_vector *c, simd_vector *d, simd_vector* e, simd_vector* f, simd_vector *g, simd_vector* h,
                                       simd_vector *i, simd_vector *j, simd_vector *k, simd_vector *l, simd_vector *m, simd_vector *n, simd_vector *o, simd_vector *p)
{
    simd_minmax_2V(a, i);
    simd_minmax_2V(b, j);
    simd_minmax_2V(c, k);
    simd_minmax_2V(d, l);
    simd_minmax_2V(e, m);
    simd_minmax_2V(f, n);
    simd_minmax_2V(g, o);
    simd_minmax_2V(h, p);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_8V(i, j, k, l, m, n, o, p);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_merge_2V_sorted(simd_vector* a, simd_vector* b)
{
    simd_permute_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_2V(simd_vector* a, simd_vector* b)
{
    *a = simd_sort_1V(*a);
    *b = simd_sort_1V(*b);
    simd_merge_2V_sorted(a, b);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_merge_3V_sorted(simd_vector* a, simd_vector* b, simd_vector* c)
{
    simd_permute_minmax_2V(b, c);
    simd_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_3V(simd_vector* a, simd_vector* b, simd_vector* c)
{
    simd_sort_2V(a, b);
    *c = simd_sort_1V(*c);
    simd_merge_3V_sorted(a, b, c);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_merge_4V_sorted(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d)
{
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
static inline void simd_sort_4V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d)
{
    simd_sort_2V(a, b);
    simd_sort_2V(c, d);
    simd_merge_4V_sorted(a, b, c, d);
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
static inline void simd_sort_17V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    *q = simd_sort_1V(*q);
    simd_permute_minmax_2V(p, q);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    *q = simd_aftermerge_1V(*q);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_18V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q, simd_vector* r)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_sort_2V(q, r);
    simd_permute_minmax_2V(p, q);
    simd_permute_minmax_2V(o, r);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_aftermerge_2V(q, r);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_19V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q, simd_vector* r, simd_vector* s)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_sort_3V(q, r, s);
    simd_permute_minmax_2V(p, q);
    simd_permute_minmax_2V(o, r);
    simd_permute_minmax_2V(n, s);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_aftermerge_3V(q, r, s);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_20V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q, simd_vector* r, simd_vector* s, simd_vector* t)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_sort_4V(q, r, s, t);
    simd_permute_minmax_2V(p, q);
    simd_permute_minmax_2V(o, r);
    simd_permute_minmax_2V(n, s);
    simd_permute_minmax_2V(m, t);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_aftermerge_4V(q, r, s, t);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_21V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q, simd_vector* r, simd_vector* s, simd_vector* t, simd_vector* u)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_sort_5V(q, r, s, t, u);
    simd_permute_minmax_2V(p, q);
    simd_permute_minmax_2V(o, r);
    simd_permute_minmax_2V(n, s);
    simd_permute_minmax_2V(m, t);
    simd_permute_minmax_2V(l, u);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_aftermerge_5V(q, r, s, t, u);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_22V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q, simd_vector* r, simd_vector* s, simd_vector* t, simd_vector* u, simd_vector* v)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_sort_6V(q, r, s, t, u, v);
    simd_permute_minmax_2V(p, q);
    simd_permute_minmax_2V(o, r);
    simd_permute_minmax_2V(n, s);
    simd_permute_minmax_2V(m, t);
    simd_permute_minmax_2V(l, u);
    simd_permute_minmax_2V(k, v);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_aftermerge_6V(q, r, s, t, u, v);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_23V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q, simd_vector* r, simd_vector* s, simd_vector* t, simd_vector* u, simd_vector* v, simd_vector* x)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_sort_7V(q, r, s, t, u, v, x);
    simd_permute_minmax_2V(p, q);
    simd_permute_minmax_2V(o, r);
    simd_permute_minmax_2V(n, s);
    simd_permute_minmax_2V(m, t);
    simd_permute_minmax_2V(l, u);
    simd_permute_minmax_2V(k, v);
    simd_permute_minmax_2V(j, x);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_aftermerge_7V(q, r, s, t, u, v, x);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_24V(simd_vector* a, simd_vector* b, simd_vector* c, simd_vector* d, simd_vector* e, simd_vector* f, simd_vector* g, simd_vector* h,
                                 simd_vector* i, simd_vector* j, simd_vector* k, simd_vector* l, simd_vector *m, simd_vector* n, simd_vector* o, simd_vector* p,
                                 simd_vector* q, simd_vector* r, simd_vector* s, simd_vector* t, simd_vector* u, simd_vector* v, simd_vector* x, simd_vector* y)
{
    simd_sort_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_sort_8V(q, r, s, t, u, v, x, y);
    simd_permute_minmax_2V(p, q);
    simd_permute_minmax_2V(o, r);
    simd_permute_minmax_2V(n, s);
    simd_permute_minmax_2V(m, t);
    simd_permute_minmax_2V(l, u);
    simd_permute_minmax_2V(k, v);
    simd_permute_minmax_2V(j, x);
    simd_permute_minmax_2V(i, y);
    simd_aftermerge_16V(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    simd_aftermerge_8V(q, r, s, t, u, v, x, y);
}

//----------------------------------------------------------------------------------------------------------------------
int simd_small_sort_max()
{
    return SIMD_VECTOR_WIDTH * 24;
}

//----------------------------------------------------------------------------------------------------------------------
void simd_small_sort(float* array, int element_count)
{
    if (element_count <= 1)
        return;
    
    const int full_vec_count = element_count / SIMD_VECTOR_WIDTH;
    const int last_vec_size = element_count - (full_vec_count * SIMD_VECTOR_WIDTH);
    
    simd_vector data[24];
    
    for(int i=0; i<full_vec_count; ++i)
        data[i] = simd_load_vector(array, i);
    
    if (last_vec_size)
        data[full_vec_count] = simd_load_partial(array, full_vec_count, last_vec_size);
    
    if (element_count <= SIMD_VECTOR_WIDTH)
    {
        data[0] = simd_sort_1V(data[0]);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 2)
    {
        simd_sort_2V(data, data+1);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 3)
    {
        simd_sort_3V(data, data+1, data+2);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 4)
    {
        simd_sort_4V(data, data+1, data+2, data+3);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 5)
    {
        simd_sort_5V(data, data+1, data+2, data+3, data+4);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 6)
    {
        simd_sort_6V(data, data+1, data+2, data+3, data+4, data+5);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 7)
    {
        simd_sort_7V(data, data+1, data+2, data+3, data+4, data+5, data+6);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 8)
    {
        simd_sort_8V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 9)
    {
        simd_sort_9V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8); 
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 10)
    {
        simd_sort_10V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 11)
    {
        simd_sort_11V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 12)
    {
        simd_sort_12V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 13)
    {
        simd_sort_13V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 14)
    {
        simd_sort_14V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 15)
    {
        simd_sort_15V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 16)
    {
        simd_sort_16V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 17)
    {
        simd_sort_17V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 18)
    {
        simd_sort_18V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16, data+17);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 19)
    {
        simd_sort_19V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16, data+17, data+18);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 20)
    {
        simd_sort_20V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16, data+17, data+18, data+19);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 21)
    {
        simd_sort_21V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16, data+17, data+18, data+19, data+20);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 22)
    {
        simd_sort_22V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16, data+17, data+18, data+19, data+20, data+21);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 23)
    {
        simd_sort_23V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16, data+17, data+18, data+19, data+20, data+21, data+22);
    }
    else if (element_count <= SIMD_VECTOR_WIDTH * 24)
    {
        simd_sort_24V(data, data+1, data+2, data+3, data+4, data+5, data+6, data+7, data+8, data+9, data+10, data+11, data+12, data+13, data+14, data+15, data+16, data+17, data+18, data+19, data+20, data+21, data+22, data+23);
    }

    for(int i=0; i<full_vec_count; ++i)
        simd_store_vector(array, data[i], i);

    if (last_vec_size)
        simd_store_partial(array, data[full_vec_count], full_vec_count, last_vec_size);
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_load_vector_overflow(const float* array, int size, int* index)
{
    simd_vector result = (*index + SIMD_VECTOR_WIDTH > size) ? simd_load_partial(array + *index, 0, size - *index) : simd_load_vector(array + *index, 0);
    *index += SIMD_VECTOR_WIDTH;
    return result;
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_store_vector_overflow(float* array, int size, int *index, simd_vector a)
{
    if (*index + SIMD_VECTOR_WIDTH > size)
    {
        simd_store_partial(array + *index, a, 0, size - *index);
    }
    else
    {
        simd_store_vector(array + *index, a, 0);
    }
    *index += SIMD_VECTOR_WIDTH;
}

//----------------------------------------------------------------------------------------------------------------------
// based on Efficient Implementation of Sorting on MultiCore SIMD CPU Architecture paper
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

    simd_vector a = simd_load_vector_overflow(left_array, left_element_count, &left_index);
    simd_vector b = simd_load_vector_overflow(right_array, right_element_count, &right_index);

    simd_merge_2V_sorted(&a, &b);
    simd_store_vector_overflow(array, right+1, &output_index, a);

    while (left_index < left_element_count && right_index < right_element_count)
    {
        if (left_array[left_index]<right_array[right_index])
            a = simd_load_vector_overflow(left_array, left_element_count, &left_index);
        else
            a = simd_load_vector_overflow(right_array, right_element_count, &right_index);

        simd_merge_2V_sorted(&a, &b);
        simd_store_vector_overflow(array, right+1, &output_index, a);
    }

    while (left_index < left_element_count) 
    {
        a = simd_load_vector_overflow(left_array, left_element_count, &left_index);
        simd_merge_2V_sorted(&a, &b);
        simd_store_vector_overflow(array, right+1, &output_index, a);
    }

    while (right_index < right_element_count) 
    {
        a = simd_load_vector_overflow(right_array, right_element_count, &right_index);
        simd_merge_2V_sorted(&a, &b);
        simd_store_vector_overflow(array, right+1, &output_index, a);
    }

    simd_store_vector_overflow(array, right+1, &output_index, b);

    free(left_array);
    free(right_array);
}

#define MERGE_SORT_TILE (simd_small_sort_max())

//----------------------------------------------------------------------------------------------------------------------
void merge_sort(float* array, int left, int right) 
{
    if (left < right)
    {
        int middle, left_element_count, right_element_count;
        int element_count = (right - left + 1);

        if (element_count <= (2 * MERGE_SORT_TILE) && element_count > MERGE_SORT_TILE)
        {
            middle = left + MERGE_SORT_TILE - 1;
        }
        else
        {
            middle = left + (right - left) / 2;    
        }

        left_element_count = middle - left + 1;
        right_element_count = right - middle;
        
        if (left_element_count <= MERGE_SORT_TILE)
            simd_small_sort(array + left, left_element_count);
        else
            merge_sort(array, left, middle);

        if (right_element_count <= MERGE_SORT_TILE)
            simd_small_sort(array + middle + 1, right_element_count);
        else
            merge_sort(array, middle + 1, right);
            
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
