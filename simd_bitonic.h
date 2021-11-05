

/*
To use this library, do this in *one* C or C++ file:
    #define __SIMD_BITONIC_IMPLEMENTATION__
    #include "simd_bitonic.h"
    
COMPILATION
    
DOCUMENTATION
*/

// http://performanceguidelines.blogspot.com/2013/08/sorting-algorithms-on-gpu.html


#ifndef __SIMD_BITONIC__
#define __SIMD_BITONIC__


//----------------------------------------------------------------------------------------------------------------------
// Prototypes
//----------------------------------------------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

#define SIMD_SORT_OK                (1)
#define SIMD_SORT_NOTALIGNED        (2)
#define SIMD_SORT_TOOMANYELEMENTS   (3)
#define SIMD_SORT_NOTHINGTOSORT     (4)

// returns SIMD_SORT_OK if suceeded otherwise another error code
int simd_sort_float(float* array, int element_count);

#ifdef __cplusplus
}
#endif

//----------------------------------------------------------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------------------------------------------------------

#ifdef __SIMD_BITONIC_IMPLEMENTATION__
#undef __SIMD_BITONIC_IMPLEMENTATION__

#include <assert.h>

//----------------------------------------------------------------------------------------------------------------------
// Neon
//----------------------------------------------------------------------------------------------------------------------
#if defined(__ARM_NEON) && defined(__ARM_NEON__)

#include <arm_neon.h>

#define ALIGN_STRUCT(x) __attribute__((aligned(x)))

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
static inline float32x4_t simd_sort_4f(float32x4_t input)
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
static inline float32x4_t simd_aftermerge_4f(float32x4_t input)
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
static inline void simd_permute_minmax_8f(float32x4_t *a, float32x4_t *b)
{
    float32x4_t perm_neigh = __builtin_shufflevector(*b, *b, 3, 2, 1, 0);
    float32x4_t perm_neigh_min = vminq_f32(*a, perm_neigh);
    float32x4_t perm_neigh_max = vmaxq_f32(*a, perm_neigh);
    *a = perm_neigh_min;
    *b = perm_neigh_max;
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_1V(float32x4_t *a, float32x4_t *b)
{
    *a = simd_sort_4f(*a);
    *b = simd_sort_4f(*b);
    simd_permute_minmax_8f(a, b);
    *a = simd_aftermerge_4f(*a);
    *b = simd_aftermerge_4f(*b);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_minmax_8f(float32x4_t *a, float32x4_t *b)
{
    float32x4_t perm_neigh_min = vminq_f32(*a, *b);
    float32x4_t perm_neigh_max = vmaxq_f32(*a, *b);
    *a = perm_neigh_min;
    *b = perm_neigh_max;
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_2V(float32x4_t *a, float32x4_t *b, float32x4_t *c, float32x4_t *d)
{
    simd_sort_1V(a, b);
    simd_sort_1V(c, d);
    simd_permute_minmax_8f(a, d);
    simd_permute_minmax_8f(b, c);
    simd_minmax_8f(a, b);
    simd_minmax_8f(c, d);
    *a = simd_aftermerge_4f(*a);
    *b = simd_aftermerge_4f(*b);
    *c = simd_aftermerge_4f(*c);
    *d = simd_aftermerge_4f(*d);
}

//----------------------------------------------------------------------------------------------------------------------
int simd_sort_float(float* array, int element_count)
{
    return SIMD_SORT_TOOMANYELEMENTS;
}

#else

//----------------------------------------------------------------------------------------------------------------------
// AVX
//----------------------------------------------------------------------------------------------------------------------

#include <immintrin.h>
#include <stdint.h>

//----------------------------------------------------------------------------------------------------------------------
static inline __m256 _mm256_swap(__m256 input)
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
static inline __m256 simd_sort_1V(__m256 input)
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
static inline __m256 simd_aftermerge_1V(__m256 a)
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
static inline void simd_aftermerge_2V(__m256 *a, __m256 *b)
{
    simd_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_3V(__m256 *a, __m256 *b, __m256 *c)
{
    simd_minmax_2V(a, c);
    simd_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
    *c = simd_aftermerge_1V(*c);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_aftermerge_4V(__m256 *a, __m256 *b, __m256 *c, __m256 *d)
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
static inline void simd_aftermerge_5V(__m256 *a, __m256 *b, __m256 *c, __m256 *d, __m256* e)
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
static inline void simd_aftermerge_6V(__m256 *a, __m256 *b, __m256 *c, __m256 *d, __m256* e, __m256* f)
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
static inline void simd_aftermerge_7V(__m256 *a, __m256 *b, __m256 *c, __m256 *d, __m256* e, __m256* f, __m256 *g)
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
static inline void simd_aftermerge_8V(__m256 *a, __m256 *b, __m256 *c, __m256 *d, __m256* e, __m256* f, __m256 *g, __m256* h)
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
static inline void simd_sort_2V(__m256* a, __m256* b)
{
    *a = simd_sort_1V(*a);
    *b = simd_sort_1V(*b);
    simd_permute_minmax_2V(a, b);
    *a = simd_aftermerge_1V(*a);
    *b = simd_aftermerge_1V(*b);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_3V(__m256* a, __m256* b, __m256* c)
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
static inline void simd_sort_4V(__m256* a, __m256* b, __m256* c, __m256* d)
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
static inline void simd_sort_5V(__m256* a, __m256* b, __m256* c, __m256* d, __m256* e)
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
static inline void simd_sort_6V(__m256* a, __m256* b, __m256* c, __m256* d, __m256* e, __m256* f)
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
static inline void simd_sort_7V(__m256* a, __m256* b, __m256* c, __m256* d, __m256* e, __m256* f, __m256* g)
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
static inline void simd_sort_8V(__m256* a, __m256* b, __m256* c, __m256* d, __m256* e, __m256* f, __m256* g, __m256* h)
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
static inline void simd_sort_9V(__m256* a, __m256* b, __m256* c, __m256* d, __m256* e, __m256* f, __m256* g, __m256* h, __m256* i)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    *i = simd_sort_1V(*i);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    *i = simd_aftermerge_1V(*i);
}

//----------------------------------------------------------------------------------------------------------------------
static inline void simd_sort_10V(__m256* a, __m256* b, __m256* c, __m256* d, __m256* e, __m256* f, __m256* g, __m256* h, __m256* i, __m256* j)
{
    simd_sort_8V(a, b, c, d, e, f, g, h);
    simd_sort_2V(i, j);
    simd_permute_minmax_2V(g, j);
    simd_permute_minmax_2V(h, i);
    simd_aftermerge_8V(a, b, c, d, e, f, g, h);
    simd_aftermerge_2V(i, j);
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

// positive infinity float hexadecimal value
#define FLOAT_PINF (0x7F800000) 

//----------------------------------------------------------------------------------------------------------------------
static inline __m256 _mm256_load_partial(const float* array, int element_count)
{
    assert(element_count<9 && element_count>0);
    if (element_count == 8)
    {
        return _mm256_load_ps(array);
    }
    else
    {
        __m256 inf_mask = _mm256_cvtepi32_ps(_mm256_set_epi32(FLOAT_PINF, 
                                           (element_count>6) ? 0 : FLOAT_PINF,
                                           (element_count>5) ? 0 : FLOAT_PINF,
                                           (element_count>4) ? 0 : FLOAT_PINF,
                                           (element_count>3) ? 0 : FLOAT_PINF,
                                           (element_count>2) ? 0 : FLOAT_PINF,
                                           (element_count>1) ? 0 : FLOAT_PINF,
                                           (element_count>0) ? 0 : FLOAT_PINF));
        
        __m256 a = _mm256_maskload_ps(array, loadstore_mask(element_count));
        return _mm256_or_ps(a, inf_mask);
    }
}

//----------------------------------------------------------------------------------------------------------------------
static inline void _mm256_store_partial(float* array, __m256 a, int element_count)
{
    assert(element_count<9 && element_count>0);
    if (element_count == 8)
    {
        _mm256_store_ps(array, a);
    }
    else
    {
        _mm256_maskstore_ps(array, loadstore_mask(element_count), a);
    }
}

//----------------------------------------------------------------------------------------------------------------------
int simd_sort_float(float* array, int element_count)
{
    if (!element_count)
        return SIMD_SORT_NOTHINGTOSORT;

    const intptr_t address = (intptr_t) array;
    if (address%32 != 0)
        return SIMD_SORT_NOTALIGNED;

    const int last_vec_size = (element_count%8) == 0 ? 8 : (element_count%8);
    if (element_count <= 8)
    {
        __m256 a = _mm256_load_partial(array, last_vec_size);
        a = simd_sort_1V(a);
        _mm256_store_partial(array, a, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 16)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_partial(array+8, last_vec_size);
        simd_sort_2V(&a, &b);
        _mm256_store_ps(array, a);
        _mm256_store_partial(array+8, b, last_vec_size);
        return SIMD_SORT_OK;
    }
    
    if (element_count <= 24)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_partial(array+16, last_vec_size);
        simd_sort_3V(&a, &b, &c);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_partial(array+16, c, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 32)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_ps(array+16);
        __m256 d = _mm256_load_partial(array+24, last_vec_size);
        simd_sort_4V(&a, &b, &c, &d);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_ps(array+16, c);
        _mm256_store_partial(array+24, d, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 40)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_ps(array+16);
        __m256 d = _mm256_load_ps(array+24);
        __m256 e = _mm256_load_partial(array+32, last_vec_size);
        simd_sort_5V(&a, &b, &c, &d, &e);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_ps(array+16, c);
        _mm256_store_ps(array+24, d);
        _mm256_store_partial(array+32, e, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 48)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_ps(array+16);
        __m256 d = _mm256_load_ps(array+24);
        __m256 e = _mm256_load_ps(array+32);
        __m256 f = _mm256_load_partial(array+40, last_vec_size);
        simd_sort_6V(&a, &b, &c, &d, &e, &f);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_ps(array+16, c);
        _mm256_store_ps(array+24, d);
        _mm256_store_ps(array+32, e);
        _mm256_store_partial(array+40, f, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 56)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_ps(array+16);
        __m256 d = _mm256_load_ps(array+24);
        __m256 e = _mm256_load_ps(array+32);
        __m256 f = _mm256_load_ps(array+40);
        __m256 g = _mm256_load_partial(array+48, last_vec_size);
        simd_sort_7V(&a, &b, &c, &d, &e, &f, &g);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_ps(array+16, c);
        _mm256_store_ps(array+24, d);
        _mm256_store_ps(array+32, e);
        _mm256_store_ps(array+40, f);
        _mm256_store_partial(array+48, g, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 64)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_ps(array+16);
        __m256 d = _mm256_load_ps(array+24);
        __m256 e = _mm256_load_ps(array+32);
        __m256 f = _mm256_load_ps(array+40);
        __m256 g = _mm256_load_ps(array+48);
        __m256 h = _mm256_load_partial(array+56, last_vec_size);
        simd_sort_8V(&a, &b, &c, &d, &e, &f, &g, &h);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_ps(array+16, c);
        _mm256_store_ps(array+24, d);
        _mm256_store_ps(array+32, e);
        _mm256_store_ps(array+40, f);
        _mm256_store_ps(array+48, g);
        _mm256_store_partial(array+56, h, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 72)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_ps(array+16);
        __m256 d = _mm256_load_ps(array+24);
        __m256 e = _mm256_load_ps(array+32);
        __m256 f = _mm256_load_ps(array+40);
        __m256 g = _mm256_load_ps(array+48);
        __m256 h = _mm256_load_ps(array+56);
        __m256 i = _mm256_load_partial(array+64, last_vec_size);
        simd_sort_9V(&a, &b, &c, &d, &e, &f, &g, &h, &i);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_ps(array+16, c);
        _mm256_store_ps(array+24, d);
        _mm256_store_ps(array+32, e);
        _mm256_store_ps(array+40, f);
        _mm256_store_ps(array+48, g);
        _mm256_store_ps(array+56, h);
        _mm256_store_partial(array+64, i, last_vec_size);
        return SIMD_SORT_OK;
    }

    if (element_count <= 80)
    {
        __m256 a = _mm256_load_ps(array);
        __m256 b = _mm256_load_ps(array+8);
        __m256 c = _mm256_load_ps(array+16);
        __m256 d = _mm256_load_ps(array+24);
        __m256 e = _mm256_load_ps(array+32);
        __m256 f = _mm256_load_ps(array+40);
        __m256 g = _mm256_load_ps(array+48);
        __m256 h = _mm256_load_ps(array+56);
        __m256 i = _mm256_load_ps(array+64);
        __m256 j = _mm256_load_partial(array+72, last_vec_size);
        simd_sort_10V(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j);
        _mm256_store_ps(array, a);
        _mm256_store_ps(array+8, b);
        _mm256_store_ps(array+16, c);
        _mm256_store_ps(array+24, d);
        _mm256_store_ps(array+32, e);
        _mm256_store_ps(array+40, f);
        _mm256_store_ps(array+48, g);
        _mm256_store_ps(array+56, h);
        _mm256_store_ps(array+64, i);
        _mm256_store_partial(array+72, j, last_vec_size);
        return SIMD_SORT_OK;
    }

    return SIMD_SORT_TOOMANYELEMENTS;
}

#endif

#endif
#endif
