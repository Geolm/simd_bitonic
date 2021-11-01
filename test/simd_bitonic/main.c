#define __SIMD_BITONIC_IMPLEMENTATION__
#include "../../simd_bitonic.h"

#define ALIGNED_VARIABLE __attribute__((aligned(32)))

int main(int argc, const char * argv[])
{
    ALIGNED_VARIABLE float array[16] = {43, 90, 6, 13, 12, 89, 10, 19, 8, 74,63, 51, 4, 3, 27, -1};
    
    simd_sort_float(array, 15);
    
    return 0;
}
