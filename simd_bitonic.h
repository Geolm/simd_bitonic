

/*
To use this library, do this in *one* C or C++ file:
    #define __SIMD_BITONIC_IMPLEMENTATION__
    #include "simd_bitonic.h"
    
COMPILATION
    
DOCUMENTATION
*/


#ifndef __SIMD_BITONIC__
#define __SIMD_BITONIC__


//----------------------------------------------------------------------------------------------------------------------
// Prototypes
//----------------------------------------------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

void sort_16_floats(float* array);

#ifdef __cplusplus
}
#endif

//----------------------------------------------------------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------------------------------------------------------

#ifdef __SIMD_BITONIC_IMPLEMENTATION__
#undef __SIMD_BITONIC_IMPLEMENTATION__


//----------------------------------------------------------------------------------------------------------------------
// Neon
//----------------------------------------------------------------------------------------------------------------------
#if defined(__ARM_NEON) && defined(__ARM_NEON__)

#include <arm_neon.h>

#define ALIGN_STRUCT(x) __attribute__((aligned(x)))

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

static inline float32x4_t sort_4_floats(float32x4_t input)
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

static inline float32x4_t bitonic_after_merge(float32x4_t input)
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

static inline void sort_8_floats(float32x4_t *a, float32x4_t *b)
{
    *a = sort_4_floats(*a);
    *b = sort_4_floats(*b);
    
    {
        float32x4_t perm_neigh = __builtin_shufflevector(*b, *b, 3, 2, 1, 0);
        float32x4_t perm_neigh_min = vminq_f32(*a, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(*a, perm_neigh);
        *a = perm_neigh_min;
        *b = perm_neigh_max;
    }
    
    *a = bitonic_after_merge(*a);
    *b = bitonic_after_merge(*b);
}

void sort_16_floats(float* array)
{
    float32x4_t a = vld1q_f32(array);
    float32x4_t b = vld1q_f32(array+4);
    float32x4_t c = vld1q_f32(array+8);
    float32x4_t d = vld1q_f32(array+12);
    
    sort_8_floats(&a, &b);
    sort_8_floats(&c, &d);
    
    {
        float32x4_t perm_neigh = __builtin_shufflevector(d, d, 3, 2, 1, 0);
        float32x4_t perm_neigh_min = vminq_f32(a, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(a, perm_neigh);
        a = perm_neigh_min;
        d = perm_neigh_max;
    }
    {
        float32x4_t perm_neigh = __builtin_shufflevector(c, c, 3, 2, 1, 0);
        float32x4_t perm_neigh_min = vminq_f32(b, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(b, perm_neigh);
        b = perm_neigh_min;
        c = perm_neigh_max;
    }
    {
        float32x4_t perm_neigh_min = vminq_f32(a, b);
        float32x4_t perm_neigh_max = vmaxq_f32(a, b);
        a = perm_neigh_min;
        b = perm_neigh_max;
    }
    {
        float32x4_t perm_neigh_min = vminq_f32(c, d);
        float32x4_t perm_neigh_max = vmaxq_f32(c, d);
        c = perm_neigh_min;
        d = perm_neigh_max;
    }
    
    a = bitonic_after_merge(a);
    b = bitonic_after_merge(b);
    c = bitonic_after_merge(c);
    d = bitonic_after_merge(d);
    
    vst1q_f32(array, a);
    vst1q_f32(array+4, b);
    vst1q_f32(array+8, c);
    vst1q_f32(array+12, d);
}

#else

//----------------------------------------------------------------------------------------------------------------------
// AVX
//----------------------------------------------------------------------------------------------------------------------

#include <immintrin.h>

static inline __m128 sort_4_floats(__m128 input)
{
    {
        __m128 perm_neigh = _mm_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 perm_neigh_min = _mm_min_ps(input, perm_neigh);
        __m128 perm_neigh_max = _mm_max_ps(input, perm_neigh);
        input = _mm_blend_ps(perm_neigh_min, perm_neigh_max, 0xA);
    }
    {
        __m128 perm_neigh = _mm_permute_ps(input, _MM_SHUFFLE(0, 1, 2, 3));
        __m128 perm_neigh_min = _mm_min_ps(input, perm_neigh);
        __m128 perm_neigh_max = _mm_max_ps(input, perm_neigh);
        input = _mm_blend_ps(perm_neigh_min, perm_neigh_max, 0xC);
    }
    {
        __m128 perm_neigh = _mm_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 perm_neigh_min = _mm_min_ps(input, perm_neigh);
        __m128 perm_neigh_max = _mm_max_ps(input, perm_neigh);
        input = _mm_blend_ps(perm_neigh_min, perm_neigh_max, 0xA);
    }
    return input;
}

static inline __m256 _mm256_swap(__m256 input)
{
    __m128 lo = _mm256_extractf128_ps(input, 0);
    __m128 hi = _mm256_extractf128_ps(input, 1);
    return _mm256_setr_m128(hi, lo);
}

static inline __m256 sort_8_floats(__m256 input)
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

void sort_16_floats(float* array)
{
    __m256 a = _mm256_load_ps(array);
    __m256 b = _mm256_load_ps(array + 8);
    
    a = sort_8_floats(a);
    b = sort_8_floats(b);

    {
        __m256 swap = _mm256_swap(b);
        __m256 perm_neigh = _mm256_permute_ps(swap, _MM_SHUFFLE(0, 1, 2, 3));
        __m256 perm_neigh_min = _mm256_min_ps(a, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(a, perm_neigh);
        a = perm_neigh_min;
        b = perm_neigh_max;
    }
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
    {
        __m256 swap = _mm256_swap(b);
        __m256 perm_neigh = _mm256_permute_ps(swap, _MM_SHUFFLE(3, 2, 1, 0));
        __m256 perm_neigh_min = _mm256_min_ps(b, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(b, perm_neigh);
        b = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xF0);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(b, _MM_SHUFFLE(1, 0, 3, 2));
        __m256 perm_neigh_min = _mm256_min_ps(b, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(b, perm_neigh);
        b = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(b, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(b, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(b, perm_neigh);
        b = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    
    _mm256_store_ps(array, a);
    _mm256_store_ps(array+8, b);
}

#endif

#endif
#endif
