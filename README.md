# SIMD bitonic sort

Sort small arrays of float using simd instructions to parallelize work.

Loosely based on "Fast Sorting Algorithms using AVX-512 on Intel Knights Landing" https://hal.inria.fr/hal-01512970v1/document

AVX (x64) and NEON (arm) implementation using intrinsics.

# Faster than std::sort()

* Running bitonic sort in parallel thanks to SIMD wide register (4 floats NEON / 8 floats AVX)
* Less access to memory as most of the work is done inside SIMD registers


# Drawbacks
* Only for small arrays, current size <= 32 floats
* Sort only floats, cannot sort a structure with a float in for example
* Works only aligned (16 or 32 bytes depending on the platform) array 
* No SSE or non-SIMD implementation. Bitonic sort is slower than std::sort if not done in parallel.
