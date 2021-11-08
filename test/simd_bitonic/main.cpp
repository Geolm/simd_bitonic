#define __SIMD_BITONIC_IMPLEMENTATION__
#include "../../simd_bitonic.h"

#define SOKOL_IMPL
#include "sokol_time.h"

#include "random.h"

#include <vector>
#include <algorithm>
#include <stdio.h>

#define NUMBER_OF_SORTS (1000000)
#define MAX_ARRAY_SIZE (SIMD_VECTOR_WIDTH * 16)

int seed = 0x12345678;

void profile_small(int array_size)
{
    float array[MAX_ARRAY_SIZE];
    
    std::vector<float> vector;
    vector.resize(array_size);
    
    uint64_t start_time;
    uint64_t diff = 0;
    float result = 0.f;
    
    for(int i=0; i<NUMBER_OF_SORTS; ++i)
    {
        for(int j=0; j<array_size; ++j)
            vector[j] = iq_random_float(&seed);
        
        start_time = stm_now();
        
        std::sort(vector.begin(), vector.end());
        
        diff += stm_diff(stm_now(), start_time);
        
        result += vector[0];
    }
    
    float stl_duration = (float)stm_sec(diff);
    
    diff = 0;

    float simd_result = 0.f;
    seed = 0x12345678;
    
    for(int i=0; i<NUMBER_OF_SORTS; ++i)
    {
        for(int j=0; j<array_size; ++j)
            array[j] = iq_random_float(&seed);
        
        start_time = stm_now();
        
        simd_small_sort(array, array_size);
        
        diff += stm_diff(stm_now(), start_time);
        
        simd_result += array[0];
    }
    
    float bitonic_duration = (float)stm_sec(diff);
    printf("%f\n",stl_duration/bitonic_duration);
    //printf("%f times faster for %2d elements\n", stl_duration/bitonic_duration, array_size);
}

void check_correctness(int array_size)
{
    float array[MAX_ARRAY_SIZE];
    
    std::vector<float> vector;
    vector.resize(array_size);

    for(int j=0; j<array_size; ++j)
    {
        vector[j] = (iq_random_float(&seed) - 0.5f) * 10000.f;
        array[j] = vector[j];
    }

    std::sort(vector.begin(), vector.end());
    simd_small_sort(array, array_size);

    for(int j=0; j<array_size; ++j)
    {
        assert(vector[j] == array[j]);
    }
}

void check_error_code()
{
    assert(simd_small_sort(nullptr, 65890) == SIMD_SORT_TOOMANYELEMENTS);
}

void check_merge_sort(int array_size)
{
    float* array = (float*) malloc(sizeof(float) * array_size);
    
    std::vector<float> vector;
    vector.resize(array_size);
    
    for(int j=0; j<array_size; ++j)
    {
        vector[j] = (iq_random_float(&seed) - 0.5f) * 10000.f;
        array[j] = vector[j];
    }

    std::sort(vector.begin(), vector.end());
    
    simd_merge_sort(array, array_size);

    for(int j=0; j<array_size; ++j)
    {
        assert(vector[j] == array[j]);
    }

    
    free(array);
}

void profile_merge_sort(int array_size)
{
    float* array = (float*) malloc(sizeof(float) * array_size);
    
    std::vector<float> vector;
    vector.resize(array_size);

    uint64_t stl_diff = 0;
    uint64_t simd_diff = 0;

    for(int i=0; i<100; ++i)
    {
        for(int j=0; j<array_size; ++j)
        {
            vector[j] = (iq_random_float(&seed) - 0.5f) * 10000.f;
            array[j] = vector[j];
        }

        uint64_t start_time = stm_now();
        
        std::sort(vector.begin(), vector.end());

        stl_diff += stm_diff(stm_now(), start_time);

        start_time = stm_now();

        simd_merge_sort(array, array_size);

        simd_diff += stm_diff(stm_now(), start_time);
    }

    float stl_duration = (float)stm_sec(stl_diff);
    float simd_duration = (float)stm_sec(simd_diff);

    printf("%f\n", stl_duration/simd_duration);
    
    free(array);
}

int main(int argc, const char * argv[])
{
    stm_setup();
    
    seed = (int)stm_now();
    
    printf("checking error codes\n");
    
    check_error_code();

    printf("checking small sort ");
    for(int a=0; a<1000; ++a)
    {
        for(int i=2; i<=simd_small_sort_max(); ++i)
            check_correctness(i);
        
        if (a%100 == 0)
            printf(".");
    }
    
    printf("\nchecking merge sort ");
    
    int size = 3;
    for(int a=0; a<21; ++a)
    {
        check_merge_sort(size);
        
        size *= 2;
        printf(".");
    }

    printf("\nchecking merge bitonic sort performances\n");

    size = 3;
    for(int i=0; i<12; ++i)
    {
        profile_merge_sort(size);
        size *= 3;
    }
    
    seed = 0x12345678;
    
    printf("\nchecking small sort performances\n");
    
    for(int i=1; i<=simd_small_sort_max(); ++i)
        profile_small(i);
    
    return 0;
}
