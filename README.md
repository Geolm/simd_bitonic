# SIMD bitonic sort

This library works with AVX or NEON instructions.

## Bitonic sort

Sort small arrays of float using SIMD instructions to parallelize work.

Based on "Fast Sorting Algorithms using AVX-512 on Intel Knights Landing" https://hal.inria.fr/hal-01512970v1/document

## Merge sort

Tiled merge sort using SIMD merge sort based on "Efficient Implementation of Sorting on Multi-Core SIMD CPU Architecture" http://www.vldb.org/pvldb/vol1/1454171.pdf


# Library
One C99 header file, simd_bitonic.h. C99 probably needed. Tested with clang -mavx -o3.

* simd_small_sort_max(), returns the maximum number of float at max that be can sorted with the small sort function
* simd_small_sort(), bitonic sort small arrays
* simd_merge_sort(), tiled merge sort

# Results

Profile was done by sorting 10,000,000 times an array from random elements. 

## Mac mini 2018, i7, AVX
Array size vary from 2 to 128 elements. The simd bitonic sort is almost 7x faster than std::sort() at best.

![AVX chart](/images/AVX_chart.png)

Note that we can clearly see that the sort is more optimal when the array size is multiple of 8. Because loading data is faster and all float in the registers are used to do the sort.

## M1 macbook air (2020), NEON
Array size vary from 2 to 64 elements.

![Neon chart](/images/NEON_chart.png)

This chart is more all over the place, gain are still impressive though.

# Why is it faster?
* SIMD Bitonic sort runs in parallel thanks to SIMD wide register (4 floats NEON / 8 floats AVX)
* There are less access to memory as most of the work is done inside SIMD registers

# Drawbacks
* Only for small arrays, merge-sort is not as efficient
* Sort only "pure" floats, cannot sort an array of struct {float a; int b;}  for example

# What are the typical use-case?
* Sorting values for image compression, usually 8x8 or 4x4 pixels
* Sorting values for kdtree building, for example each leaf of kdtree could have 16 points and when we need to split the node we sort the points using one axis
* Sorting values that are already in SIMD registers
