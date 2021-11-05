# SIMD bitonic sort

Sort small arrays of float using simd instructions to parallelize work.

Loosely based on "Fast Sorting Algorithms using AVX-512 on Intel Knights Landing" https://hal.inria.fr/hal-01512970v1/document

AVX (x64) and NEON (arm) implementation using intrinsics.

# Results

Profile was done by sorting 10,000,000 times an array from random elements. 

## Mac mini 2018, i7, AVX
Array size vary from 2 to 128 elements. The simd bitonic sort is almost 7x faster than std::sort() at best.

![AVX chart](/images/AVX_chart.png)

Note that we can clearly see that the sort is more optimal when the array size is multiple of 8. Because loading data is faster and all float in the registers are used to do the sort.

## M1 macbook air (2020), Neon
Array size vary from 2 to 64 elements.

![Neon chart](/images/NEON_chart.png)

This chart is more all over the place, gain are still impressive though.

# Why is it faster?

* SIMD Bitonic sort runs in parallel thanks to SIMD wide register (4 floats NEON / 8 floats AVX)
* There are less access to memory as most of the work is done inside SIMD registers


# Drawbacks
* Only for small arrays, currently only size <= 64 floats are supported
* Sort only "pure" floats, cannot sort an array of struct {float a; int b;}  for example
* Works only aligned (16 or 32 bytes depending on the platform) array 
* Needs SIMD to be parallel and quick

# What are the use-case examples?
* Sorting values for image compression, usually 8x8 or 4x4 pixels
* Sorting values for kdtree building, for example each leaf of kdtree could have 16 points and when we need to split the node we sort the points using one axis
