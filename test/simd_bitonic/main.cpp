#define __SIMD_BITONIC_IMPLEMENTATION__
#include "../../simd_bitonic.h"

#define SOKOL_IMPL
#include "sokol_time.h"

#include "random.h"

#include <vector>
#include <algorithm>
#include <stdio.h>

#define ALIGNED_VARIABLE __attribute__((aligned(32)))
#define NUMBER_OF_SORTS (10000000)
#define ARRAY_SIZE (32)

int seed = 0x12345678;

int main(int argc, const char * argv[])
{
    ALIGNED_VARIABLE float array[ARRAY_SIZE];
    
    stm_setup();
    
    printf("profiling, %d times sorting an array of %d float values\n\n", NUMBER_OF_SORTS, ARRAY_SIZE);
    
    std::vector<float> vector;
    vector.resize(ARRAY_SIZE);
    
    uint64_t current_time = stm_now();
    float result = 0.f;
    
    for(int i=0; i<NUMBER_OF_SORTS; ++i)
    {
        for(int j=0; j<ARRAY_SIZE; ++j)
            vector[j] = iq_random_float(&seed);
        
        std::sort(vector.begin(), vector.end());
        
        result += vector[0];
    }
    
    uint64_t raw_delta_time = stm_since(current_time);
    float duration = (float)stm_sec(raw_delta_time);
    printf("std::vector sort, duration : %f seconds, result : %f\n", duration, result);
    
    
    current_time = stm_now();
    result = 0.f;
    seed = 0x12345678;
    
    for(int i=0; i<NUMBER_OF_SORTS; ++i)
    {
        for(int j=0; j<ARRAY_SIZE; ++j)
            array[j] = iq_random_float(&seed);
        
        simd_sort_float(array, ARRAY_SIZE);
        
        result += array[0];
    }
    
    raw_delta_time = stm_since(current_time);
    duration = (float)stm_sec(raw_delta_time);
    printf("simd_bitonic sort, duration : %f seconds, result : %f\n", duration, result);
    
    return 0;
}
