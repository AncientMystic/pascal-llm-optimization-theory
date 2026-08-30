# Pascal Q4 KV Cache Accelerator

**Efficient 4-bit KV cache inference on Pascal GPUs (GTX 10-series, P40, P100) using grouped dequantization.**  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Problem Statement

Large language models (LLMs) require caching of key and value tensors for autoregressive inference. These caches can exceed available GPU memory, especially on older Pascal GPUs (max 12–24 GB VRAM). Quantizing the KV cache to 4 bits reduces memory up to 8× for the packed data itself, but **naive implementations on Pascal are often 10–20× slower than FP32** because the hardware lacks native low-precision arithmetic. This repository provides a custom CUDA kernel that overcomes this slowdown by grouping 8 quantized values into a single 32‑bit word and applying efficient vectorized dequantization.

With this approach, Q4 cache inference can approach FP32 speed while using significantly less memory—**1/8th for the packed values**, plus a small metadata overhead (scales and zero-points).

---

## Theory: Why Naive Q4 is Slow and How Grouping Fixes It

Pascal GPUs execute only FP32 (and some integer) operations natively. When using quantized data, each value must be unpacked from its bit-packed representation and converted to FP32 before multiplication. If done per element with scalar operations, the instruction overhead overwhelms the actual computation, making the kernel compute-bound even though memory traffic is reduced.

**Root causes of slowdown**:

1. **Unpacking overhead** – Extracting a 4‑bit nibble requires mask, shift, and conversion per element.
2. **Poor memory coalescing** – Reading individual bytes instead of 32‑bit words.
3. **Metadata traffic** – Storing a scale/zero per value doubles memory usage, negating quantisation benefits.
4. **Multiple kernel launches** – High‑level frameworks may unpack, scale, and matmul in separate passes.

### Grouped Quantization (Sets of 8)

We quantize values in **groups of 8 consecutive elements** and store them as a single 32‑bit word (little‑endian nibble order). Each group has **one shared float scale** and **one shared float offset** (which we call `zero`, representing the minimum value of the group).

**Advantages**:

- **Vectorized memory access** – One `uint32_t` load per 8 values, and with the proper data layout, loads are coalesced across threads.
- **Efficient unpacking** – All 8 nibbles are extracted simultaneously using bitwise operations.
- **Reduced metadata** – Scales/offsets add only 1/8th the data of the original FP32 tensor.
- **Amortized overhead** – The conversion cost is spread over 8 multiply‑adds, making the kernel memory‑bound for typical head dimensions (≥64).

The dequantization formula is:  
`value_i = nibble_i * scale + zero`  
where `zero` is the group minimum (mapped to 0 during quantization).

---

## Implementation

The core kernel (`q4_dot_product_kernel`) computes dot products between a single FP32 query vector and all Q4‑quantized keys. Each CUDA thread processes one key, looping over its groups.

### Key Features

- **Coalesced global memory access** – data is stored group‑major, so consecutive threads read consecutive 32‑bit words.
- **Unrolled inner loop** – 8 FP32 fused multiply‑adds per group.
- **Read‑only cache usage** – `__restrict__` hints enable texture cache optimizations.
- **Correct dequantization** – values are reconstructed as `q * scale + zero`.

### Data Layout

Assume key shape `[num_keys, head_dim]` with `head_dim` divisible by 8. Let `groups_per_key = head_dim / 8`.

Arrays are stored in **group‑major order** to guarantee coalesced memory access:

- `q4_keys` : `uint32_t` array of size `num_keys * groups_per_key`. For key `k` and group `g`, the index is `g * num_keys + k`.
- `scales`  : `float` array, same indexing; one scale per group.
- `zeros`   : `float` array, same indexing; one offset per group (group minimum).

This layout ensures that for a fixed group `g`, threads with consecutive `k` access consecutive memory addresses.

---

## Performance Expectations

On a **GTX 1080 Ti** (11 GB, 484 GB/s bandwidth), processing a 24k token context with `head_dim = 128` (i.e., `groups_per_key = 16`):

- **Packed Q4 keys**: 24,576 keys × 16 groups × 4 bytes = **1.57 MB**  
- **Scales**: same = **1.57 MB**  
- **Zeros**: same = **1.57 MB**  
- **Total Q4 data**: ~4.7 MB  
- **Equivalent FP32 keys**: 24,576 × 128 × 4 = **12.58 MB**  

The kernel reads ~4.7 MB per query. At peak bandwidth, that takes about **0.01 ms**. Including launch overhead and minor inefficiencies, the kernel should run in **tens of microseconds to a few milliseconds**, depending on context size and GPU model. This is far faster than naive per‑element Q4 implementations (which can be 10–20× slower than FP32) and comparable to FP32 performance for memory‑bound workloads.

Actual speedup depends on workload and GPU model. The memory savings are ~2.7× compared to FP32 when including metadata, but the packed data alone uses 8× less memory—crucial for large contexts.

---

## Limitations

- Requires `head_dim` to be a multiple of 8.
- Only unsigned 4‑bit quantisation (range 0–15) is implemented; signed or asymmetric variants can be added.
- The kernel computes scores for one query vector at a time. For multi‑query attention, tile the query dimension for better reuse.
- Pascal lacks FP16/INT8 tensor cores, so this is a pure CUDA core implementation.

---

## Usage

1. **Include the kernel** in your project or compile as a static library.
2. **Quantize your key cache** offline using `quantize_q4_grouped`.
3. **During inference**, call the kernel for each new query token to obtain attention scores.
4. **Integrate with your attention mechanism** (e.g., softmax, value aggregation).

---

## Full Report with code examples: Accelerating Q4 KV Cache on Pascal GPUs via Grouped Dequantization

### 1. Introduction

Pascal-generation GPUs (GTX 10-series, P40, P100) lack hardware acceleration for low-precision arithmetic (FP16, INT8, INT4). Their CUDA cores are optimised for FP32 operations. When running inference with quantised KV caches (e.g., 4-bit), naive implementations often become **10–20× slower** than FP32, negating the memory savings. However, the slowdown is not due to a fundamental hardware limitation but rather to inefficient dequantization and memory access patterns. By grouping Q4 values into sets of eight (one 32-bit word) and using custom CUDA kernels with a coalesced memory layout, we can achieve near-FP32 performance while using significantly less memory.

This report explains the theoretical basis and provides a fully functional CUDA kernel that computes attention scores between a single FP32 query vector and a Q4-quantised key cache using grouped dequantization.

---

### 2. Theory: Why Naive Q4 is Slow and How Grouping Helps

#### 2.1 Memory vs Compute Trade-off

- **FP32 KV cache**  
  - Memory: 4 bytes per value.  
  - Compute: Native FP32 multiply-add (1 instruction per pair).  
  - Bottleneck: Usually memory bandwidth for large contexts.

- **Naive Q4 KV cache**  
  - Memory: 0.5 bytes per value (packed).  
  - Compute: Each value must be **unpacked** (extract nibble), **dequantised** (scale and offset), then converted to FP32 before the dot product.  
  - If implemented with scalar loads and per‑element operations, the instruction overhead is enormous, making the kernel **compute‑bound** despite the reduced memory traffic.

- **Root causes of slowdown**  
  1. **Unpacking overhead**: Extracting a 4-bit nibble from a byte, masking, shifting.  
  2. **Poor memory coalescing**: Reading individual bytes instead of 32-bit words.  
  3. **Per‑element scale/offset load**: If scales and offsets are stored per value, the metadata traffic doubles memory usage.  
  4. **Kernel launch overhead**: High-level frameworks may use multiple kernels (unpack → scale → matmul) causing multiple passes over data.

#### 2.2 Grouped Q4 (Sets of 8)

Quantise values in **groups of 8 consecutive elements**. For each group:

- Store **one scale** (float) and **one offset** (float, group minimum, called `zero`).  
- Pack the 8 four‑bit values into a single **32‑bit word** (little‑endian nibble order).  

**Benefits**:

- **Vectorised loads**: One `uint32_t` load per 8 values. With group‑major layout, loads are coalesced across threads.  
- **Efficient unpacking**: Use bitwise AND and shifts to extract all 8 nibbles in a few instructions.  
- **Reduced metadata**: Scales/offsets add only 1/8th the data of the original FP32 tensor.  
- **Better arithmetic intensity**: The overhead per group is amortised over 8 multiply‑adds. If the head dimension is large (e.g., 128), the conversion cost becomes negligible compared to the actual dot product.

Mathematically, the dequantised value for the *i*‑th element in a group is:

```
value_i = nibble_i * scale + zero
```

The dot product between a query slice `q` and the dequantised group `k` becomes:

```
sum_{i=0}^{7} q_i * (nibble_i * scale + zero)
     = scale * sum_{i=0}^{7} q_i * nibble_i + zero * sum_{i=0}^{7} q_i
```

This can be further optimised, but the straightforward per‑element dequantisation is often fast enough because the FP32 multiply‑adds dominate.

---

### 3. Data Layout

Assume the key cache has shape `[num_keys, head_dim]`, where `head_dim` is a multiple of 8. Let `groups_per_key = head_dim / 8`. Arrays are stored in **group‑major order**:

- **`q4_keys`**: `uint32_t` array of size `num_keys * groups_per_key`. For key `k` (0 ≤ k < num_keys) and group `g` (0 ≤ g < groups_per_key), the packed word is at index `g * num_keys + k`.
- **`scales`**: `float` array of same size; one scale per group.
- **`zeros`**: `float` array of same size; one offset (group minimum) per group.

This layout ensures that for a fixed `g`, consecutive `k` access consecutive memory addresses, enabling perfect coalescing in the kernel.

---

### 4. CUDA Kernel Design

We present a kernel that computes the dot product of **one FP32 query vector** (`query`) with **all keys** stored in the grouped Q4 format. The output is an array `scores` of length `num_keys`.

**Parallelisation strategy**:

- Each **CUDA thread** handles one key index `k`.  
- The thread loops over all groups belonging to that key (`num_groups_per_key = head_dim / 8`).  
- For each group it:  
  1. Computes the group‑major index `idx = g * num_keys + k`.  
  2. Loads the packed `uint32_t`.  
  3. Loads the corresponding scale and zero.  
  4. Unpacks the 8 nibbles.  
  5. Converts them to floats, multiplies by scale, adds zero.  
  6. Performs the dot product with the corresponding 8‑element slice of `query`.  
- The thread accumulates the total dot product in a register and writes it to `scores[k]`.

This design ensures **coalesced memory access**: for a given group `g`, consecutive threads (with consecutive `k`) read consecutive memory addresses. Because each thread loops over all groups sequentially, the entire array is accessed with perfect coalescing at every iteration.

#### 4.1 Kernel Code

```cpp
#include <cuda_runtime.h>
#include <cstdint>

// Device function: unpack 8 nibbles from a packed 32-bit word and dequantize.
// nibbles are stored in little-endian order: lowest nibble first.
__device__ __forceinline__ void unpack8_dequant(
    uint32_t packed, float scale, float zero, float* vals)
{
    // Extract low nibbles of each byte
    uint32_t low_mask = 0x0F0F0F0F;
    uint32_t low = packed & low_mask;          // each byte contains low nibble (0-15)
    // Extract high nibbles: shift right by 4, then mask
    uint32_t high = (packed >> 4) & low_mask;  // each byte contains high nibble

    // Reinterpret as uchar4 for easy access to individual bytes
    uchar4 low_bytes  = *reinterpret_cast<uchar4*>(&low);
    uchar4 high_bytes = *reinterpret_cast<uchar4*>(&high);

    // Dequantize: value = nibble * scale + zero
    vals[0] = float(low_bytes.x) * scale + zero;
    vals[1] = float(high_bytes.x) * scale + zero;
    vals[2] = float(low_bytes.y) * scale + zero;
    vals[3] = float(high_bytes.y) * scale + zero;
    vals[4] = float(low_bytes.z) * scale + zero;
    vals[5] = float(high_bytes.z) * scale + zero;
    vals[6] = float(low_bytes.w) * scale + zero;
    vals[7] = float(high_bytes.w) * scale + zero;
}

// Kernel: compute dot product between a single query vector and all keys.
// query: [head_dim] (FP32)
// q4_keys: packed uint32 array, size num_keys * groups_per_key (group-major)
// scales, zeros: float arrays, same indexing as q4_keys
// scores: output, size num_keys
// num_keys: number of keys
// groups_per_key = head_dim / 8
__global__ void q4_dot_product_kernel(
    const float* __restrict__ query,
    const uint32_t* __restrict__ q4_keys,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    float* __restrict__ scores,
    int num_keys,
    int groups_per_key)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_keys) return;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_key; ++g) {
        // Group-major index: all keys' group g stored consecutively
        int idx = g * num_keys + k;

        uint32_t packed = q4_keys[idx];
        float scale = scales[idx];
        float zero = zeros[idx];

        float vals[8];
        unpack8_dequant(packed, scale, zero, vals);

        // Query slice corresponding to this group
        const float* q = query + g * 8;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc += q[i] * vals[i];
        }
    }

    scores[k] = acc;
}
```

#### 4.2 Host Code Example (Quantisation and Launch)

To make the example self-contained, we provide a host function that quantises a FP32 key matrix into the grouped Q4 format using simple uniform quantisation per group.

```cpp
#include <algorithm>
#include <cmath>
#include <cstdio>      // for fprintf
#include <stdexcept>   // for std::invalid_argument
#include <cuda_runtime.h>
#include <cstdint>

// Utility: quantise a float array into grouped Q4 (group-major layout).
// Input: keys [num_keys * head_dim] (row-major)
// Output: q4_keys, scales, zeros allocated by caller (size num_keys * groups_per_key)
void quantize_q4_grouped(
    const float* keys,
    int num_keys,
    int head_dim,
    uint32_t* q4_keys,
    float* scales,
    float* zeros)
{
    int groups_per_key = head_dim / 8;
    for (int k = 0; k < num_keys; ++k) {
        for (int g = 0; g < groups_per_key; ++g) {
            const float* group_start = keys + (k * head_dim) + g * 8;
            float min_val = group_start[0], max_val = group_start[0];
            for (int i = 1; i < 8; ++i) {
                min_val = std::fmin(min_val, group_start[i]);
                max_val = std::fmax(max_val, group_start[i]);
            }
            float range = max_val - min_val;
            if (range < 1e-6f) range = 1e-6f;
            float scale = range / 15.0f;
            float zero = min_val;

            uint32_t packed = 0;
            for (int i = 0; i < 8; ++i) {
                float val = group_start[i];
                int q = (int)std::round((val - zero) / scale);
                q = std::max(0, std::min(15, q));
                packed |= (uint32_t)q << (4 * i);
            }

            int idx = g * num_keys + k;
            q4_keys[idx] = packed;
            scales[idx] = scale;
            zeros[idx] = zero;
        }
    }
}

// Example launch with error checking
void compute_scores_q4(
    const float* query,        // [head_dim]
    const uint32_t* q4_keys,   // packed, group-major
    const float* scales,
    const float* zeros,
    float* scores,             // [num_keys]
    int num_keys,
    int head_dim)
{
    if (head_dim % 8 != 0) {
        throw std::invalid_argument("head_dim must be multiple of 8");
    }
    int groups_per_key = head_dim / 8;
    int threads = 256;
    int blocks = (num_keys + threads - 1) / threads;

    q4_dot_product_kernel<<<blocks, threads>>>(
        query, q4_keys, scales, zeros, scores,
        num_keys, groups_per_key
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
}
```

---

### 5. Explanation and Functional Guarantees

- **Unpacking and dequantization**: The `unpack8_dequant` device function uses bitwise operations to extract all eight 4‑bit nibbles from the 32‑bit word, then applies the per‑group scale and offset: `value = nibble * scale + zero`. This matches the quantization formula exactly (assuming no clamping artifacts; clamping only affects accuracy, not correctness).

- **Memory coalescing**: Data is stored in group‑major order. For a fixed group index `g`, consecutive threads (with consecutive `k`) access consecutive memory addresses in `q4_keys`, `scales`, and `zeros`. Thus memory transactions are perfectly coalesced (32‑bit words).

- **Arithmetic intensity**: The inner loop over 8 elements performs 8 FP32 fused multiply‑adds per group. The unpacking overhead (about 6–8 integer instructions) is amortised over 8 FMAs. For typical head dimensions (e.g., 128 → 16 groups), the kernel is **memory‑bound** on the packed data, not compute‑bound.

- **Correctness**: The kernel assumes `head_dim` is a multiple of 8. If not, padding is required. The quantisation function clamps values to [0,15], and the offset is set to the group minimum so that dequantized values are non‑negative, which is standard for unsigned 4‑bit quantisation. The dequantisation formula matches the quantization scheme, ensuring accurate reconstruction within rounding error.

- **Performance expectation**: With the corrected coalesced layout and correct dequantization, the kernel should achieve a large fraction of peak memory bandwidth. The memory traffic per key is `(packed word + scale + zero) = 4 + 4 + 4 = 12 bytes` per group. For `head_dim = 128`, that's `16 × 12 = 192 bytes` per key. For 24k keys, ~4.7 MB total, which can be read in ~10 microseconds at peak bandwidth on a GTX 1080 Ti. Kernel launch overhead dominates, making overall runtime in the tens of microseconds to low milliseconds.

---

### 6. Further Optimisations

While the provided kernel is fully functional, several enhancements can improve performance:

1. **Multiple queries per block**: If you need to compute scores for many query tokens (e.g., the entire query matrix), process a tile of queries per block to reuse loaded key data. This reduces global memory traffic for the keys.

2. **Vectorised loads**: Use `float4` or `int4` loads for query slices when possible, but the current unrolled loop already lets the compiler generate efficient code.

3. **Integer dot products**: Instead of converting nibbles to float, use integer arithmetic to compute `sum(q_int * k_int)` and then apply scaling at the end. This can reduce floating‑point conversion overhead but requires quantising the query as well.

4. **Shared memory for query**: If many threads (keys) access the same query vector, load the query into shared memory once per block to reduce redundant global memory reads.

5. **Use `__ldg`**: The `__restrict__` keyword already hints the compiler to use the read‑only data cache.

6. **Group size larger than 8**: Using groups of 16 or 32 reduces metadata overhead further, but the packing would require two or four 32‑bit words per group. The concept remains identical.

---

### 7. Integration with Inference Frameworks

For use in an actual LLM inference engine, the Q4 KV cache must be stored persistently. The kernel above can be integrated into custom attention implementations. If using PyTorch, one can write a custom CUDA extension using `torch.utils.cpp_extension`. However, for best performance, a fully custom inference engine (e.g., in C++/CUDA) is recommended.

---

### 8. Conclusion

Grouped Q4 quantisation with sets of 8 values per 32‑bit word provides a practical solution for Pascal GPUs: it reduces memory footprint (8× for the packed values, ~2.7× including metadata) while keeping dequantization overhead low enough to remain memory‑bound. The provided kernel demonstrates a fully functional implementation with correct dequantization and coalesced memory access, which should drastically outperform naive Q4 approaches and approach FP32 speeds in memory‑bound attention workloads. The key is to **avoid scalar per‑element unpacking**, **ensure coalesced 32‑bit memory accesses** via a group‑major layout, and **use the correct dequantization formula**.

---

*Note: The code assumes a little‑endian architecture and a CUDA‑capable device of compute capability ≥ 6.0 (Pascal). It has been written for clarity and correctness; production code may include additional checks and optimisations.*
