#ifndef __INSHAPESRANDOM__
#define __INSHAPESRANDOM__

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

// based on https://www.iquilezles.org/www/articles/sfrand/sfrand.htm

static inline int iq_random( int* seed)
{
    *seed *= 16807;
    return (*seed) >> 9;
}

// returns a random number in the range [min, max]
static inline int iq_random_clamped(int* seed, int min, int max)
{
    int range = max - min + 1;
    return abs(iq_random(seed) % range) + min;
}

// returns a random float in the range [0 ; 1]
static inline float iq_random_float(int* seed)
{
    union
    {
        float fres;
        unsigned int ires;
    } float2int;

    *seed *= 16807;

    float2int.ires = ((((unsigned int)*seed)>>9 ) | 0x3f800000);
    return float2int.fres - 1.0f;
}

// returns a random float in the range [0 ; 2pi]
static inline float iq_random_angle(int* seed)
{
    return iq_random_float(seed) * (float)M_PI * 2.f;
}


#endif
