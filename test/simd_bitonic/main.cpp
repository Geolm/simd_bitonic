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
#define MAX_ARRAY_SIZE (32)

int seed = 0x12345678;

void profile(int array_size)
{
    ALIGNED_VARIABLE float array[MAX_ARRAY_SIZE];
    
    std::vector<float> vector;
    vector.resize(array_size);
    
    uint64_t current_time = stm_now();
    float result = 0.f;
    
    for(int i=0; i<NUMBER_OF_SORTS; ++i)
    {
        for(int j=0; j<array_size; ++j)
            vector[j] = iq_random_float(&seed);
        
        std::sort(vector.begin(), vector.end());
        
        result += vector[0];
    }
    
    uint64_t raw_delta_time = stm_since(current_time);
    float stl_duration = (float)stm_sec(raw_delta_time);
    
    current_time = stm_now();
    float simd_result = 0.f;
    seed = 0x12345678;
    
    for(int i=0; i<NUMBER_OF_SORTS; ++i)
    {
        for(int j=0; j<array_size; ++j)
            array[j] = iq_random_float(&seed);
        
        simd_sort_float(array, array_size);
        
        simd_result += array[0];
    }
    
    raw_delta_time = stm_since(current_time);
    float bitonic_duration = (float)stm_sec(raw_delta_time);
    printf("array of %d elements : simd_bitonic sort is %f times faster than stl\n", array_size, stl_duration/bitonic_duration);
}

int main(int argc, const char * argv[])
{
    
    
    stm_setup();
    
    for(int i=2; i<=32; ++i)
        profile(i);
    
    return 0;
}
