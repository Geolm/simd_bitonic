# SIMD bitonic sort

Sort small arrays of float using simd instructions to parallelize work.

Loosely based on "Fast Sorting Algorithms using AVX-512 on Intel Knights Landing" https://hal.inria.fr/hal-01512970v1/document

AVX (x64) and NEON (arm) implementation using intrinsics.

# Results

## Mac mini 2018, i7, AVX
Profile was done by sorting 10,000,000 times an array from random elements. The simd bitonic sort is almost 6x faster at best.

![AVX chart](/images/AVX_chart.png)

Note that we can clearly see that the sort is more optimal when the array size is multiple of 8 which is the AVX register size.


# Why is it faster?

* SIMD Bitonic sort runs in parallel thanks to SIMD wide register (4 floats NEON / 8 floats AVX)
* There are less access to memory as most of the work is done inside SIMD registers


# Drawbacks
* Only for small arrays, currently only size <= 32 floats are supported
* Sort only floats, cannot sort a structure with a float in for example
* Works only aligned (16 or 32 bytes depending on the platform) array 
* No non-SIMD implementation. Bitonic sort is slower than std::sort if not done in parallel.
