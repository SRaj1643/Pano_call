# PanoCall — SIFT + CUDA End-Evaluation Study Guide

> **Purpose:** A simple, team-friendly guide for end-evaluation/viva preparation.
>
> **Truth rule:** Separate **implemented project facts** from **general theory**. If a value is not measured or source-confirmed, say **VERIFY FROM CODE** instead of guessing.

## Quick Navigation

1. [Project in 60 Seconds](#1-project-in-60-seconds)
2. [System Pipeline](#2-system-pipeline)
3. [CUDA Basics](#3-cuda-basics)
4. [SIFT CUDA Stages](#4-sift-cuda-stages)
5. [Stitching and Homography](#5-stitching-and-homography)
6. [Performance and Limitations](#6-performance-and-limitations)
7. [30 Core Viva Questions](#7-30-core-viva-questions)
8. [30 Difficult Follow-Ups](#8-30-difficult-follow-ups)
9. [Rapid-Fire Revision](#9-rapid-fire-revision)
10. [If the Evaluator Challenges You](#10-if-the-evaluator-challenges-you)
11. [Final Cheat Sheet](#11-final-cheat-sheet)

---

## 1. Project in 60 Seconds

**PanoCall** is a multi-camera VR calling/stitching prototype. Overlapping camera frames provide correspondences. A custom SIFT pipeline detects and describes features, CUDA accelerates the dense SIFT stages, descriptor matching produces candidate correspondences, and CPU/Numba code performs RANSAC, Hartley-normalized DLT, validation, warping and blending.

The live path caches a valid homography and keeps alignment refresh in the background. This protects display latency. The recorded cached **warp + blend** hot path is about **23 ms (~43 FPS)**.

### One-line pipeline

```text
Camera frames
    ↓
Grayscale
    ↓
CUDA SIFT + descriptor matching
    ↓
CPU/Numba RANSAC + Hartley-normalized DLT
    ↓
Homography validation
    ↓
Cache valid H
    ↓
Inverse warp + feather blend
    ↓
Panorama / display
```

### What is actually CUDA?

- **CUDA C++:** Gaussian scale space through descriptor matching.
- **CPU / NumPy / Numba:** RANSAC, Hartley-normalized DLT, homography validation, inverse warp and blend.
- **Verified real-time path:** two cameras.

---

## 2. System Pipeline

| Component | Role |
|---|---|
| `run_pipeline.py` | Legacy batch path; writes grayscale `.bin`, runs `sift_stitcher.exe`, consumes exports |
| `cpu_pipeline.py` | Parses results, validates indices, estimates/refines H, warps and blends |
| `realtime_stitcher.py` | Numba JIT, Hartley normalization and caching |
| `video_stitcher.py` | Two capture threads, background refresh, overlay and tracking state |
| `homography_validator.py` | Nine geometric validation checks |
| `cuda_bridge.py` | Direct-memory DLL interface specification; native DLL needs an NVIDIA-machine build |

### Real-time design

```text
Capture 0 ─┐
          ├──> latest frames ──> main render ──> cached H ──> warp/blend
Capture 1 ─┘
                    │
                    └──────────> background H refresh
                                 SIFT → matching → RANSAC → validation
```

**Key idea:** the display does not wait for every expensive alignment refresh.

---

## 3. CUDA Basics

### Execution model

- **Host:** CPU code that allocates/copies memory and launches kernels.
- **Device:** GPU where CUDA threads execute.
- **Kernel:** GPU function launched by the host.
- **Grid:** complete set of blocks for one kernel launch.
- **Block:** group of threads scheduled together.
- **Thread:** smallest unit of CUDA work.
- **Warp:** 32 threads executing in SIMT style.
- **SM:** Streaming Multiprocessor that schedules blocks/warps and provides execution resources.

### PanoCall thread mapping

| Work | CUDA mapping |
|---|---|
| Image kernels | `dim3(16,16)` = 256 threads/block |
| Orientation | 1-D, 256 threads/block |
| Descriptor | 1-D, 256 threads/block |
| Matching | 1-D, 256 threads/block |

### Thread indexing

```cpp
x = blockIdx.x * blockDim.x + threadIdx.x;
```

For images, compute both `x` and `y` and always guard against rounded-up grid dimensions:

```cpp
if (x >= width || y >= height) return;
```

### SIMT and divergence

A warp executes together. If threads take different branch paths, the warp can diverge. PanoCall has branches around borders, extrema tests, contrast/edge filters and descriptor boundaries.

### Memory hierarchy

| Memory | PanoCall use |
|---|---|
| Registers | Thread-local coordinates, gradients, descriptor temporaries |
| Shared memory | **No explicit tiling found** |
| Global memory | Images, pyramids, maps, descriptors, matches |
| Constant memory | Gaussian coefficients, radius and sigma levels |
| Host memory | CPU vectors, downloaded results, Python arrays |

### Synchronization and performance

- `__syncthreads()` → synchronizes threads **inside one block**.
- `cudaDeviceSynchronize()` → host-side wait for device work to finish; current wrappers use it after stages.
- CUDA events → correct tool for GPU elapsed-time measurement, but **not implemented in the inspected code**.
- Streams → can overlap independent work, but **not implemented in the inspected code**.
- Occupancy → active resident warps relative to the hardware maximum; higher occupancy is not automatically faster.
- Amdahl's Law → CPU geometry, transfers/I/O and display can limit end-to-end speedup.

---

## 4. SIFT CUDA Stages

### SIFT flow

```text
Gaussian scale space
      ↓
DoG + downsampling
      ↓
Extrema detection
      ↓
Contrast / edge filtering
      ↓
NMS
      ↓
Orientation
      ↓
128-D descriptor
      ↓
Descriptor matching
```

### Stage cards

### Gaussian Scale Space

**What:** It creates progressively blurred versions of an image. PanoCall implements each blur as separable horizontal and vertical Gaussian passes.

**Why:** SIFT needs feature evidence at more than one scale; the blurred images are the inputs to DoG and gradients.

**One thread:** One thread computes one horizontal or vertical output pixel, accumulates the weighted convolution window, and writes one float.

**Block:** A `dim3(16,16)` block covers a 16-by-16 tile of output pixels (256 threads); the grid is rounded up to cover the image.

**Memory:** Input/output image buffers are global memory. `d_gaussianKernel` and `d_kernelRadius` are uploaded with `cudaMemcpyToSymbol` and read from constant memory. The source does not show explicit shared-memory tiling.

**Synchronization:** Threads do not exchange partial sums, so no `__syncthreads()` is required. The host synchronizes between horizontal and vertical launches because the vertical pass consumes the horizontal buffer.

**Risks:** Missing bounds checks, an invalid radius, edge handling errors, and excessive global-memory traffic can break correctness or performance.

**Optimization:** The source uses separability (two 1-D convolutions instead of one 2-D convolution) and constant-memory Gaussian coefficients. It does not implement a shared-memory tile.

**Remaining bottleneck:** Repeated global loads, especially in the vertical pass, plus the current per-stage device synchronizations.

**Measured performance:** **VERIFY FROM CODE:** no CUDA-event kernel time or CPU-vs-GPU Gaussian speedup is recorded.

**Likely viva questions:** Why is Gaussian blur separable? Why constant memory? Why 16-by-16? Why is vertical access harder? Why not claim shared memory?

**Strong answer:** “We map one thread to one output pixel. Separability reduces the convolution work, and the small reusable coefficient array is placed in constant memory. The block size is an implemented choice, not a proven hardware optimum; profiling would be needed to tune it.”

---

### Downsampling and Difference of Gaussian (DoG)

**What:** `downsampleKernel` produces the next octave at half resolution; `dogKernel` computes `G(i+1) - G(i)` at each location.

**Why:** Downsampling expands scale coverage. DoG is the efficient blob-like response used to find candidate SIFT extrema.

**One thread:** For downsampling, one thread writes one half-resolution pixel from its 2-by-2 input region. For DoG, one thread subtracts two same-coordinate Gaussian values and writes one result.

**Block:** It is a 16-by-16 tile of the output image; the launch grid uses ceiling division.

**Memory:** The shown kernels use global input/output buffers and thread registers for indices/temporary values. No explicit shared memory is shown.

**Synchronization:** Not within the kernels. The host must complete producer stages before a dependent downsample/DoG stage begins.

**Risks:** Out-of-range source coordinates, bad output dimensions, or a missing bounds guard can corrupt pyramid storage.

**Optimization:** The project exposes the regular per-pixel work as CUDA kernels using 16-by-16 launches. No further source-confirmed optimization should be claimed.

**Remaining bottleneck:** These operations have low arithmetic intensity and can become memory-bandwidth/launch-overhead limited.

**Measured performance:** **VERIFY FROM CODE:** no stage timing is recorded.

**Likely viva questions:** Why do we downsample? Why is DoG preferable to a direct Laplacian approximation? Is DoG compute- or memory-heavy? Why are guards needed?

**Strong answer:** “The GPU is useful because the same simple operation is repeated at every coordinate. That does not guarantee a dramatic end-to-end speedup because low-work kernels can be dominated by memory and launch costs.”

---

### Gradients

**What:** `gradientKernel` applies Sobel neighbourhood calculations to output gradient magnitude and orientation.

**Why:** Orientation assignment and the descriptor need local gradient magnitude and direction rather than raw intensity.

**One thread:** One thread reads its 3-by-3 neighbourhood, computes `gx`, `gy`, magnitude and angle, normalizes a negative angle to `[0,360)`, then writes two outputs.

**Block:** One 16-by-16 block computes one output tile. The code explicitly uses a rounded grid.

**Memory:** Neighbourhood samples and magnitude/orientation outputs are global-memory accesses; arithmetic temporaries are registers. No shared-memory stencil cache is implemented.

**Synchronization:** No, because a thread writes only its own magnitude/orientation positions. The host synchronizes after the launch for checking.

**Risks:** Border loads are invalid unless handled. The source explicitly fixed the guard for non-multiple-of-16 dimensions and writes border outputs as zero.

**Optimization:** The verified corrective optimization is the grid bounds guard that prevented gradient-pyramid corruption for rounded grids. Do not present this correctness fix as a measured speed optimization.

**Remaining bottleneck:** Neighbour reads and transcendental operations; row-major accesses are friendlier than the vertical components of the stencil.

**Measured performance:** **VERIFY FROM CODE:** no per-kernel timing exists.

**Likely viva questions:** Why Sobel? Why zero borders? What happens at width 640/height 360? Why can a safe rounded grid include extra threads?

**Strong answer:** “The grid intentionally rounds up, so bounds guards are compulsory. We fixed a real corruption risk: threads beyond an octave’s dimensions had been able to write outside the gradient pyramid.”

---

### Extrema, Contrast, and Edge Filtering

**What:** `extremaKernel` tests a DoG pixel against 26 neighbours across three scales; `contrastKernel` and `edgeKernel` remove weak and edge-like candidates.

**Why:** It turns dense DoG responses into repeatable, informative keypoint candidates instead of keeping noise and unstable edge responses.

**One thread:** One thread reads the three DoG levels around its coordinate, tracks maximum/minimum status, and writes an extrema-map byte. The later filters evaluate the same coordinate’s criteria.

**Block:** A 16-by-16 image tile of candidate locations.

**Memory:** DoG images and output maps are global memory; flags/indices are registers. No explicit shared-memory neighbourhood tile is present.

**Synchronization:** No per-thread reduction is used. Dependencies are only between completed pipeline stages.

**Risks:** The source notes that `cudaMalloc` memory is uninitialized: boundary extrema entries must be explicitly zeroed or phantom keypoints can result. Divergent comparisons and border logic are also risks.

**Optimization:** The code explicitly writes border extrema to zero and includes bounds guards. Those are source-confirmed robustness fixes.

**Remaining bottleneck:** High neighbour-read count, branches, and potentially poor cache locality; this is less regular than DoG subtraction.

**Measured performance:** **VERIFY FROM CODE:** no timing/speedup is recorded.

**Likely viva questions:** Why 26 neighbours? Why reject edges? Why can a zeroed map matter? Does every branch cause a race condition?

**Strong answer:** “A race is not the issue here because each thread owns one map location. The real hazards are invalid/border data and divergent control flow. We explicitly clear the border because allocated device memory is not automatically zero.”

---

### Orientation Assignment

**What:** For every NMS keypoint, `orientationKernel` builds a 36-bin, magnitude-weighted local orientation histogram and emits each peak at least 80% of the maximum.

**Why:** Assigning a dominant orientation lets later descriptors be expressed relative to the keypoint, improving rotation robustness.

**One thread:** It reads one NMS keypoint, scans its gradient window, accumulates a private `float histogram[36]`, finds peak bins, then emits one or more oriented keypoints.

**Block:** The actual launcher uses one-dimensional blocks of 256 threads, representing up to 256 input keypoints.

**Memory:** Histogram/local values are thread-local (normally registers or local spill; exact placement **VERIFY FROM CODE**). Gradient pyramids and pointer/dimension arrays are global memory; sigma levels are in constant memory.

**Synchronization:** No block reduction is used because each histogram belongs to one thread. An `atomicAdd` on `orientationCount` reserves a unique output slot when multiple threads emit results.

**Risks:** The variable patch and peak count can diverge. Atomic contention and output overflow are possible; the code has a `maxOrientations` capacity guard after reservation.

**Optimization:** The source caps orientation writes to prevent a flat histogram from producing up to 36 entries and corrupting the four-per-NMS-keypoint allocation.

**Remaining bottleneck:** Irregular per-keypoint work, gradient-pyramid reads, atomics, and a 36-float private histogram can create register pressure/spilling.

**Measured performance:** **VERIFY FROM CODE:** no measured kernel result is present.

**Likely viva questions:** Why 36 bins? Why can one keypoint yield several orientations? Why atomicAdd? Why is a private histogram safer than a shared one here?

**Strong answer:** “The histogram is private to one thread, so it needs no histogram atomic. The only shared item is the global output count; atomicAdd makes each accepted orientation get a unique slot.”

---

### 128-D Descriptor Generation

**What:** `descriptorKernel` constructs one 4-by-4-by-8 (128-value) SIFT descriptor around each oriented keypoint, then L2-normalizes, clamps to 0.2, and normalizes again.

**Why:** The descriptor converts a local patch into a comparable, rotation-referenced feature vector used by matching.

**One thread:** One thread initializes `descriptor.data[128]`, scans a radius-8 patch, rotates coordinates by the keypoint angle, interpolates contributions into 4-by-4-by-8 bins, normalizes/clamps/renormalizes, then stores one descriptor.

**Block:** The launcher uses one-dimensional 256-thread blocks, so it processes up to 256 oriented keypoints per block.

**Memory:** The descriptor accumulator is per-thread; exact register/local-memory placement is **VERIFY FROM CODE**. Pyramids and descriptor output are global memory. The source has no shared-memory tile.

**Synchronization:** No threads collaborate on one descriptor. `atomicAdd(descriptorCount, 1)` reserves an output position, so it is required for safe concurrent append.

**Risks:** Patch boundary checks, angle wrapping, bin bounds, counter/output capacity, register pressure, and small floating-point differences all matter.

**Optimization:** The source uses `#pragma unroll` for several 128-element loops. It does not show a measured register/occupancy tuning pass.

**Remaining bottleneck:** Irregular patch access, transcendental functions, a large private working set, and atomic output reservation.

**Measured performance:** **VERIFY FROM CODE:** no numerical speedup is recorded.

**Likely viva questions:** Why 128 dimensions? Why two normalizations? Why clamp at 0.2? Why one thread per descriptor rather than one thread per element?

**Strong answer:** “One thread owns a complete descriptor, avoiding cross-thread reductions and synchronization. The trade-off is a larger private working set and irregular patch reads, which is why we must not assume this stage has ideal occupancy.”

---

### Descriptor Matching

**What:** `descriptorMatchingKernel` finds the nearest and second-nearest train descriptors for every query descriptor using 128-D Euclidean distance and Lowe’s 0.75 ratio test.

**Why:** Descriptor correspondences supply candidate point pairs for RANSAC/homography estimation.

**One thread:** One thread loads one query descriptor, scans every train descriptor, tracks best and second-best Euclidean distances, applies ratio 0.75, then may append one match.

**Block:** The launcher uses one-dimensional 256-thread blocks: each block handles up to 256 query descriptors.

**Memory:** Query/train descriptors and match output are global memory; best distances and the query copy are private to a thread. No shared train-tile cache is in the inspected source.

**Synchronization:** Threads do not cooperate in a query search. `atomicAdd(matchCount, 1)` is necessary when accepted queries append to the shared match array.

**Risks:** Empty train sets, output-capacity issues, ratio-test ambiguity, atomic contention, and bandwidth-heavy repeated train reads.

**Optimization:** The source unrolls the fixed 128-D distance loop and tracks the two best candidates in one pass. It does not show spatial indexing or a shared-memory training tile.

**Remaining bottleneck:** Exhaustive matching is quadratic in descriptor counts and may be dominated by global-memory traffic.

**Measured performance:** **VERIFY FROM CODE:** there is no event-derived match-kernel timing or speedup percentage.

**Likely viva questions:** Why second-best distance? Why 0.75? Is matching geometrically sufficient? What is its asymptotic complexity? Why use atomicAdd?

**Strong answer:** “The ratio test rejects ambiguous nearest neighbours; it does not prove geometry. RANSAC and the later validator are still required. Each query is independent, but exhaustive scanning remains an honest `O(QT×128)` bottleneck.”

---

### Important SIFT facts to remember

- **Orientation:** 36-bin magnitude-weighted histogram; peaks at 80% of the maximum can produce multiple orientations.
- **Descriptor:** 4 × 4 spatial cells × 8 orientation bins = **128 values**; normalize → clamp at **0.2** → renormalize.
- **Matching:** nearest + second-nearest descriptor, Lowe ratio **0.75**.
- **Matching complexity:** `O(Q × T × 128)`.
- **Do not claim:** a measured CUDA speedup for an individual SIFT stage unless CUDA-event profiling confirms it.

---

## 5. Stitching and Homography

### Homography

A homography maps points using:

```text
p' ~ H p
```

- A 3×3 homography has **8 degrees of freedom** because scale is arbitrary.
- At least **4 non-collinear point pairs** are needed.
- PanoCall uses custom **RANSAC** to handle outliers.
- Realtime DLT uses **Hartley normalization**: center/scale points → solve with SVD → denormalize.

### Why RANSAC?

Descriptor matches can contain outliers. RANSAC finds a consensus geometric model before refinement.

### Why validate H after RANSAC?

A mathematically solvable H can still be physically bad: it can cause extreme perspective, folding, poor coverage or temporal instability.

### Implemented validation checks

- Finite H
- Determinant bounds
- At least **12 inliers**
- At least **30% inlier ratio**
- Median reprojection error **< 3 px**
- Perspective limits
- Corner area / convexity / bounds
- 3×3 coverage
- Temporal corner stability

### Failure handling

```text
Candidate H
   ↓
Validation
   ├── ACCEPT → cache H
   └── REJECT → keep last valid H
```

Tracking states:

`UNINITIALIZED → TRACKING → TRACKING_DEGRADED → TRACKING_LOST`

If there is no valid H, the safe fallback is **side-by-side feeds**.

### Warp and blend

- **Inverse warp:** destination pixel → source coordinate → bilinear sampling.
- **Why inverse warp?** Avoids holes that can occur with forward mapping.
- **Feather blend:** distance-based weights reduce visible seams.
- Cached distance maps avoid repeating blend-distance computation.

---

## 6. Performance and Limitations

### Recorded facts

| Item | Value | What to say in viva |
|---|---:|---|
| Cached warp + blend | **~23 ms** | Measured CPU/Numba hot path |
| Equivalent rate | **~43 FPS** | Below the 33.3 ms 30-FPS budget |
| Legacy refresh | **~1200 ms** | Background refresh path |
| Direct DLL target | **~780 ms** | Expected target, **not hardware-verified** |
| CUDA kernel timing | **VERIFY FROM CODE** | No CUDA-event measurements |

### Why the 23 ms path is fast

- Valid geometry is cached.
- Blend distance maps are cached.
- Warp/blend uses Numba.
- Alignment refresh runs in the background.
- Capture uses separate threads.

### What the 23 ms number does NOT mean

It is **not** the time for CUDA SIFT + matching + RANSAC + validation. It is the recorded **CPU/Numba cached warp + blend hot path**.

### Main limitations

- Verified real-time path is two-camera, although CUDA has `MAX_CAMERAS=6`.
- Parallax can break a single homography model.
- Lens distortion and exposure differences remain issues.
- Camera/USB synchronization matters.
- Matching is exhaustive: `O(Q × T × 128)`.
- Global-memory traffic and repeated synchronization can limit CUDA performance.
- Legacy file/subprocess I/O adds overhead.
- CUDA cannot fix poor calibration or poor camera overlap.

### What to optimize next

1. Build and benchmark the direct CUDA DLL.
2. Add CUDA events / Nsight profiling.
3. Reduce unnecessary copies and synchronizations.
4. Investigate matching memory reuse.
5. Calibrate cameras carefully.
6. Consider shared-memory tiling **only after profiling**.

---

## 7. 30 Core Viva Questions

### Questions 1–20

### 1. What is CUDA?

**Answer:** CUDA is NVIDIA’s platform for host-launched GPU kernels. We use it for custom SIFT stages with repeated independent pixel/keypoint work.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 2. Why CUDA instead of CPU?

**Answer:** The advantage is data-parallel throughput, not merely “more cores”. Gaussian, DoG, gradient, filtering and descriptor work apply the same operation across many elements.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 3. What are grid, block and thread?

**Answer:** A grid is a kernel launch; it contains blocks; blocks contain threads. PanoCall uses 16x16 image blocks and 256-thread 1-D keypoint/matching blocks.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 4. What is a warp?

**Answer:** A warp is 32 SIMT threads scheduled together. Different branch paths inside one warp cause divergence.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 5. What is an SM?

**Answer:** A Streaming Multiprocessor schedules blocks/warps and provides execution resources, registers and shared memory.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 6. Why bounds checks?

**Answer:** Rounded grids launch threads beyond image/keypoint limits. The guard prevents out-of-bounds reads/writes.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 7. What is coalescing?

**Answer:** Adjacent threads accessing adjacent global-memory addresses combine efficiently. Row-wise image output writes are naturally coalesced.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 8. Do we use shared memory?

**Answer:** No explicit shared-memory tiling was found. Do not claim it; current kernels use global memory plus constant Gaussian data.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 9. What is occupancy?

**Answer:** Resident active warps divided by an SM maximum. It can hide latency, but 100% occupancy is not automatically optimal.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 10. What is cudaDeviceSynchronize?

**Answer:** A device-wide wait. The current wrappers use it after stages for correctness/error checks; this prevents overlap and is not a timing tool.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 11. Why CUDA events?

**Answer:** They measure GPU elapsed time accurately. They are not implemented in this repository, so per-kernel timings must be marked VERIFY FROM CODE.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 12. Why SIFT?

**Answer:** It provides repeatable local features under scale and rotation changes, and custom implementation was an explicit project goal.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 13. Why DoG?

**Answer:** Subtracting adjacent Gaussian scales produces scale-space blob responses efficiently.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 14. Why RANSAC?

**Answer:** Appearance-based matches contain outliers. RANSAC finds a consensus geometric model before refinement.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 15. Why Hartley normalization?

**Answer:** Centering/scaling points before SVD improves DLT conditioning and reduces numerical instability.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 16. Why validation after RANSAC?

**Answer:** A solvable RANSAC matrix can still be physically implausible, clustered, folded or temporally unstable.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 17. What does inverse warping do?

**Answer:** It maps each destination pixel back to the source and bilinearly samples it, avoiding forward-mapping holes.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 18. Why feather blending?

**Answer:** Distance-based weights make the overlap transition gradual and reduce visible seams.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 19. How did you reach 23 ms?

**Answer:** The live path caches valid geometry/maps, moves refresh to a background worker, uses threaded capture and Numba warp/blend. The measured warp+blend hot path is about 23 ms.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### 20. What happens on bad alignment?

**Answer:** The nine-stage validator rejects it, retains the last accepted H, increments failures and updates the tracking state.

**Safe response:** Quote only measured timings and inspected implementation facts. Mark unmeasured values **VERIFY FROM CODE**.

**Avoid:** confusing Numba CPU parallelism with CUDA.

### Questions 21–30

### 21. Why is SIFT suitable for CUDA?

**Answer:** many SIFT calculations repeat independently over pixels or keypoints.

**Strong answer:** “PanoCall maps image stages to pixels and later stages to keypoints/queries. That creates parallel work, while the irregular parts—atomics and branch-heavy filtering—remain limits.”

**Follow-up:** Does that mean every stage has equal speedup?

**Follow-up answer:** No—transfers, atomics, irregularity, memory traffic, and CPU stages cap it.

**Avoid:** claiming all SIFT stages are perfectly parallel.

### 22. Which SIFT stage benefits most?

**Answer:** the codebase does not record a measured winner.

**Strong answer:** “Regular dense stages are strong CUDA candidates, but I will not name a fastest stage without CUDA-event profiling.”

**Follow-up:** How would you prove it?

**Follow-up answer:** add warm-up runs and CUDA events around each kernel.

**Avoid:** inventing a percentage.

### 23. How did you parallelize Gaussian filtering?

**Answer:** one output pixel per thread in separate horizontal/vertical passes.

**Strong answer:** “We launch 16-by-16 blocks, read Gaussian coefficients from constant memory, write an intermediate horizontal buffer, then consume it in the vertical pass.”

**Follow-up:** Is shared memory used?

**Follow-up answer:** No explicit shared-memory tiling was found.

**Avoid:** claiming a tiled convolution.

### 24. How did you parallelize DoG?

**Answer:** one thread subtracts two same-coordinate Gaussian values.

**Strong answer:** “DoG is regular, 16-by-16 pixel parallel work with a bounds guard and global buffers.”

**Follow-up:** Why not synchronize?

**Follow-up answer:** no thread shares an output; only stage ordering matters.

**Avoid:** confusing host synchronization with thread synchronization.

### 25. How did you parallelize extrema detection?

**Answer:** one thread tests one DoG location against 26 neighbours.

**Strong answer:** “The thread reads lower/current/upper DoG planes, produces one extrema-map byte, and explicitly clears image borders.”

**Follow-up:** What was a real bug risk?

**Follow-up answer:** uninitialized `cudaMalloc` border values could become phantom keypoints.

**Avoid:** saying the 26-neighbour test uses atomics.

### 26. How did you handle descriptor generation?

**Answer:** one thread owns one oriented keypoint’s 128-D descriptor.

**Strong answer:** “It scans a radius-8 patch, rotates it, distributes weighted gradient votes into 4-by-4-by-8 bins, normalizes, clamps, renormalizes, then atomically reserves output storage.”

**Follow-up:** What is the downside?

**Follow-up answer:** irregular work and register/local-memory pressure.

**Avoid:** claiming 128 threads make one descriptor.

### 27. What was your biggest bottleneck?

**Answer:** no profile identifies a single CUDA-kernel bottleneck.

**Strong answer:** “For matching, asymptotic work is `O(QT×128)`; for live display, the measured hot path is warp+blend at about 23 ms. A kernel-level bottleneck needs profiling.”

**Follow-up:** What would you optimize next?

**Follow-up answer:** profile first; likely reduce repeated transfers/synchronizations and investigate matching memory reuse.

**Avoid:** calling SIFT the measured bottleneck without evidence.

### 28. How did you validate correctness?

**Answer:** validate intermediate/final outputs and geometry, with tolerances.

**Strong answer:** “Use CPU/GPU comparisons of maps, keypoints, descriptors and panorama; use a fixed RANSAC seed; then apply the nine-stage homography validator before caching H.”

**Follow-up:** Why may values differ?

**Follow-up answer:** floating-point order and GPU execution differ.

**Avoid:** expecting bit-identical results.

### 29. How did you achieve approximately 23 ms/frame?

**Answer:** cache accepted geometry and optimize the CPU render path.

**Strong answer:** “The recorded ~23 ms is cached warp+blend, using Numba-compiled loops and cached blend distances; SIFT/RANSAC/validation refresh runs outside that per-frame hot path.”

**Follow-up:** Is it CUDA kernel time?

**Follow-up answer:** No; it is the CPU/Numba live render hot path.

**Avoid:** attributing all 23 ms to CUDA SIFT.

### 30. What exactly did you optimize?

**Answer:** data safety, live-path work, and geometry robustness.

**Strong answer:** “Source-confirmed CUDA changes include rounded-grid guards, zeroed extrema borders, and output capacity protection. The live path adds Numba warp/blend, cache reuse, background refresh, Hartley-normalized DLT, and H validation.”

**Follow-up:** What remains?

**Follow-up answer:** direct CUDA DLL integration and profile-guided GPU tuning.

**Avoid:** saying every optimization was CUDA-specific.

---

## 8. 30 Difficult Follow-Ups

Use the following questions to test whether you can explain the design rather than memorize definitions. For answers, connect back to **CUDA mapping, memory, synchronization, RANSAC/validation, caching and measured limitations**.

1. **Why not 512 threads?**
2. **Why is 100% occupancy not always best?**
3. **Why not move warp/blend entirely to the GPU?**
4. **Why can a valid homography still be wrong?**
5. **Why not just use OpenCV?**
6. **What happens if width is not divisible by 16?**
7. **Why is `__syncthreads()` insufficient for global synchronization?**
8. **Does the ratio test prove geometric correctness?**
9. **What if features are clustered in one corner?**
10. **Why can panoramas ghost?**
11. **What is parallax?**
12. **Why can CPU and GPU outputs differ?**
13. **Why are vertical accesses harder to coalesce?**
14. **What is register pressure?**
15. **Why are repeated device synchronizations expensive?**
16. **How would you profile the CUDA stages?**
17. **How would you prove the 23 ms number?**
18. **Why is the old 25-second value not contradictory?**
19. **How would you debug panorama explosion?**
20. **Why retain the old H?**
21. **Why is descriptor matching expensive?**
22. **Why can display remain fast while refresh is slow?**
23. **Why are minimum inliers not enough?**
24. **Why check corner convexity?**
25. **Why cache blend distances?**
26. **Why use two capture threads?**
27. **What would you optimize next?**
28. **What is the honest limitation of the current CUDA implementation?**
29. **Why does Amdahl's Law matter?**
30. **How can host/device transfers remove GPU speedup?**

**Rule for difficult questions:** never invent a profiler number. If the project does not measure it, say what you would measure and how.

---

## 9. Rapid-Fire Revision

### CUDA

| Term | Remember |
|---|---|
| Host | CPU side |
| Device | GPU side |
| Kernel | GPU function launched by host |
| Thread | One unit of work |
| Block | Group of threads |
| Grid | All blocks in one launch |
| Warp | 32 threads |
| SM | Schedules blocks/warps |
| SIMT | Threads execute the same instruction stream with possible divergence |
| Occupancy | Active resident warps / maximum possible |
| Register pressure | Too much per-thread state can reduce occupancy or cause spills |
| Shared memory | Fast block-local memory; no explicit tiling in current source |
| Global memory | Main GPU buffers |
| Constant memory | Small read-only reused values |
| Coalescing | Adjacent threads access adjacent addresses |
| Bank conflict | Shared-memory access issue; not a current-kernel claim |
| `cudaMemcpy` | Host/device or device/device memory transfer API |
| `cudaDeviceSynchronize` | Waits for device work to finish |
| CUDA event | GPU timing mechanism; not currently used |
| Stream | Execution queue that can enable overlap; not currently used |

### SIFT

| Term | Remember |
|---|---|
| SIFT | Scale/rotation-aware local features |
| Gaussian | Builds scale space |
| DoG | Difference between Gaussian levels |
| Extrema | Candidate local maxima/minima in DoG |
| Contrast filter | Removes weak candidates |
| Edge filter | Removes unstable edge-like responses |
| NMS | Non-maximum suppression |
| Orientation | 36-bin local orientation histogram |
| Descriptor | 128-dimensional feature vector |
| Lowe ratio | 0.75 in current matcher |
| Matching | Query descriptor vs. train descriptors |

### Stitching

| Term | Remember |
|---|---|
| Homography | Projective 3×3 transform, 8 DoF |
| RANSAC | Robustly estimates geometry despite outliers |
| Inlier | Match consistent with the model |
| Reprojection error | Distance between observed and predicted point |
| Hartley normalization | Improves DLT numerical conditioning |
| Inverse warp | Destination → source sampling |
| Bilinear interpolation | Smooth sampling between source pixels |
| Feather blend | Distance-weighted seam reduction |
| Parallax | Different apparent motion due to viewpoint/depth |
| Tracking degraded | Temporary loss of reliable alignment |
| Tracking lost | No valid transform available |

---

## 10. If the Evaluator Challenges You

### “Why not OpenCV?”

The project intentionally implements the SIFT/matching/RANSAC/DLT logic rather than relying on `cv2.SIFT_create` / `findHomography`. OpenCV is used mainly for I/O and visualization.

### “How do you know 23 ms is valid?”

Warm up Numba, time repeated hot-path warp+blend runs with `perf_counter`, and report the configuration, sample count, average and p95. Do not call it a CUDA-kernel time.

### “What is the bottleneck?”

Do not claim a measured CUDA bottleneck without CUDA events/Nsight. Known candidates are `O(Q × T × 128)` matching, global-memory traffic, synchronization, transfers/file I/O and CPU stages.

### “What would you do next?”

Build and benchmark the direct CUDA DLL, add proper GPU profiling, reduce unnecessary copies/synchronization, calibrate cameras and profile before adding more complex CUDA optimizations.

---

## 11. Final Cheat Sheet

### CUDA

**Host → kernel → grid → block → warp → thread; SM schedules blocks.**

PanoCall: **16×16 image blocks**, **256-thread keypoint/matching blocks**, global-memory pyramids, constant Gaussian data, no explicit shared-memory tiling, and mandatory bounds guards.

### SIFT

**Gaussian → DoG → extrema → filters → NMS → orientation → 128-D descriptor → matching.**

### Stitching

**Matches → RANSAC → Hartley DLT → validate H → inverse warp → feather blend.**

### Real-time

**Cache valid H/maps → render quickly → refresh alignment in background → reject bad H → keep last valid H.**

### Numbers to remember

- **30 FPS budget:** 33.3 ms/frame
- **Recorded warp + blend:** ~23 ms/frame
- **Recorded rate:** ~43 FPS
- **Legacy refresh:** ~1200 ms
- **Direct DLL target:** ~780 ms, not hardware-verified
- **Lowe ratio:** 0.75
- **Descriptor:** 128-D
- **Orientation histogram:** 36 bins
- **H minimum pairs:** 4 non-collinear
- **Validator:** ≥12 inliers, ≥30% ratio, median reprojection <3 px, plus geometric/temporal checks

### Final viva rule

> **Be precise about what the project actually implements and measures.**
>
> If you do not have a measurement, say **VERIFY FROM CODE** or explain how you would measure it.

---

*Reformatted and simplified from the uploaded PanoCall SIFT + CUDA End-Evaluation Master Study Guide. Project facts and terminology are kept source-aligned; repetitive template text was removed to make team study easier.*
