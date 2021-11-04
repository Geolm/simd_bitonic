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
#define MAX_ARRAY_SIZE (64)

int seed = 0x12345678;

void profile(int array_size)
{
    ALIGNED_VARIABLE float array[MAX_ARRAY_SIZE];
    
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
        
        simd_sort_float(array, array_size);
        
        diff += stm_diff(stm_now(), start_time);
        
        simd_result += array[0];
    }
    
    float bitonic_duration = (float)stm_sec(diff);
    printf("%f\n",stl_duration/bitonic_duration);
    //printf("%f times faster for %2d elements\n", stl_duration/bitonic_duration, array_size);
}

int main(int argc, const char * argv[])
{
    
    
    stm_setup();
    
    for(int i=2; i<=MAX_ARRAY_SIZE; ++i)
        profile(i);
    
    return 0;
}
