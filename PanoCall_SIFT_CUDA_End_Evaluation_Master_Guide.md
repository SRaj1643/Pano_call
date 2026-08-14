# PanoCall SIFT + CUDA End-Evaluation Master Study Guide

> **Truth rule:** **Implemented in our project** means confirmed in the repository/notebook. **Know for viva** means useful theory, not a claim that we implemented it. Values without a measurement are marked **VERIFY FROM CODE**.

## 1. Project Overview
PanoCall is a multi-camera VR calling/stitching prototype. Overlapping camera views supply correspondences; a custom SIFT pipeline finds and describes features; matching, RANSAC and a homography register the views; inverse warping and feather blending create a panorama. A headset/orientation path is intended to select the user’s view. The verified real-time stitching path is two-camera.

```text
Camera frames -> grayscale -> CUDA SIFT/descriptors -> CUDA matching
 -> CPU/Numba RANSAC + DLT -> homography validation -> cached H
 -> inverse warp + feather blend -> panorama/display
```

CUDA enters through descriptor matching: Gaussian scale space through matching are CUDA C++ kernels; RANSAC, Hartley-normalized DLT, validation, current warp and blend are CPU/NumPy/Numba. The hot render target is <33.3 ms (30 FPS); recorded warp+blend is about **23 ms / 43 FPS**.

## 2. Actual System Architecture
`run_pipeline.py` is the legacy batch path: it writes grayscale `.bin` images, builds/runs `sift_stitcher.exe`, and consumes descriptor/match exports. `cpu_pipeline.py` parses them, validates indices, estimates/refines H, warps and blends. `realtime_stitcher.py` adds Numba JIT, Hartley normalization and caching. `video_stitcher.py` owns two capture threads, a background refresh worker, overlay and tracking state. `homography_validator.py` holds nine geometric checks. `cuda_bridge.py` specifies a direct-memory DLL interface, but the native DLL requires an NVIDIA-machine build.

```text
Capture 0 thread ─┐                         ┌-> background H refresh
Capture 1 thread ─┼-> latest frames --------┤   SIFT/match/RANSAC/validate
                  └-> main render ----------└-> cached H -> warp -> blend
```

## 3. CUDA Fundamentals, Architecture and Execution Model
**CUDA:** heterogeneous host-device computing. CPU host code allocates/copies memory and launches `__global__` kernels; GPU device threads execute the kernel.

```text
CPU host -> launch -> GPU grid -> blocks -> warps (32 threads) -> threads
                            └-> blocks scheduled on SMs
```

Thread indexing maps work to data: `x = blockIdx.x * blockDim.x + threadIdx.x`; image kernels similarly compute y. PanoCall image kernels use `dim3(16,16)`, so 256 threads/block. Orientation, descriptor and matching use 1-D 256-thread blocks. A guard such as `if (x >= width || y >= height) return;` is mandatory because grid dimensions round up.

**SIMT/warp divergence:** 32 warp lanes share an instruction stream. PanoCall branches at borders, extrema comparisons, contrast/edge filters and descriptor boundaries; mixed outcomes can diverge but preserve correctness.

## 4. CUDA Memory Hierarchy
| Memory | Scope / property | Actual PanoCall use |
|---|---|---|
| Registers | thread-private, fastest | local coordinates, gradients and descriptor temporaries; exact count **VERIFY FROM CODE** |
| Shared memory | explicit/block-local, synchronized | no explicit tiling found |
| L1/L2 | hardware-managed caches | implicit only; no tuning claim |
| Global memory | device VRAM, high capacity/latency | images, Gaussian/DoG/gradient pyramids, maps, descriptors, matches |
| Constant memory | read-only device storage | Gaussian coefficients, radius, sigma levels via `cudaMemcpyToSymbol` |
| Host memory | CPU RAM | vectors, downloaded results, Python arrays |

**Coalescing:** adjacent row threads tend to access adjacent addresses (`y*width+x`); output writes are good. Vertical/neighborhood reads are less ideal. **Bank conflicts:** shared-memory concept only; not an actual current-kernel issue. **Transfers:** H2D uploads images/keypoints/pointer tables; D2H downloads descriptors/matches/debug buffers. The legacy file/subprocess bridge adds overhead; direct DLL integration targets removal of that I/O.

## 5. Synchronization and Performance Concepts
- `__syncthreads()` is a block-only barrier; it cannot synchronize blocks.
- `cudaDeviceSynchronize()` waits for all device work; current wrappers use it after stages for correctness but it serializes work.
- CUDA events measure elapsed GPU time; streams overlap independent work. **Neither event timing nor streams are implemented in the inspected code.**
- Occupancy is active resident warps/max possible warps. More is not automatically faster: register pressure, bandwidth, dependencies and block size matter.
- Amdahl: `S = 1 / ((1-P) + P/N)`. CPU geometry, transfer/I/O and display limit end-to-end acceleration.

## 6. CPU SIFT Pipeline and GPU/CUDA SIFT Pipeline
SIFT solves repeatable local feature detection/description under scale and rotation changes. Its conceptual sequence is Gaussian scale space -> DoG -> extrema -> contrast/edge rejection -> NMS -> orientation -> 128-D descriptor -> ratio-test matching. The notebook implements CUDA stages through matching; CPU post-processing handles registration/rendering.

### Gaussian scale space
**1. What / why:** Repeated blur at multiple sigma values gives scale invariance. Our CUDA code uses horizontalGaussianKernel followed by verticalGaussianKernel; one thread writes one output pixel. The Gaussian coefficients and radius are in constant memory; image buffers are global memory. Image launches are dim3(16,16). Separable passes and constant coefficients are implemented. No shared-memory tile or measured kernel speedup is recorded.

**2. CPU implementation:** The CPU reference/notebook expresses the same mathematical stage; the production CPU post-processing begins after CUDA descriptors/matches.

**3. Why CPU is expensive:** repeated loops over pixels or keypoints, neighbor reads, scale levels and/or descriptor dimensions.

**4. Why CUDA:** independent output pixels/keypoints can execute concurrently.

**5. Synchronization:** no inter-thread synchronization is required by the shown per-output kernels; host wrappers call device synchronization after launch.

**6. What can go wrong:** bounds errors, global-memory traffic, branch divergence, counter races if multiple threads share a counter, and numerical differences.

**7. Viva answer:** “Our parallel unit is the output pixel or keypoint, not the whole image. GPU throughput helps repeated independent calculations; it does not eliminate data-transfer, synchronization or geometric-validation work.”

### Downsampling and DoG
**1. What / why:** downsampleKernel maps one thread to one output pixel and averages its 2x2 source region. dogKernel maps one thread to one output pixel and subtracts two Gaussian levels. These are naturally data parallel, use global memory and need no inter-thread communication. Bounds checks are essential.

**2. CPU implementation:** The CPU reference/notebook expresses the same mathematical stage; the production CPU post-processing begins after CUDA descriptors/matches.

**3. Why CPU is expensive:** repeated loops over pixels or keypoints, neighbor reads, scale levels and/or descriptor dimensions.

**4. Why CUDA:** independent output pixels/keypoints can execute concurrently.

**5. Synchronization:** no inter-thread synchronization is required by the shown per-output kernels; host wrappers call device synchronization after launch.

**6. What can go wrong:** bounds errors, global-memory traffic, branch divergence, counter races if multiple threads share a counter, and numerical differences.

**7. Viva answer:** “Our parallel unit is the output pixel or keypoint, not the whole image. GPU throughput helps repeated independent calculations; it does not eliminate data-transfer, synchronization or geometric-validation work.”

### Gradients
**1. What / why:** gradientKernel maps one thread to one pixel, reads a Sobel 3x3 neighborhood, then writes magnitude and orientation. It sets border outputs to zero. The source comments document a fixed bounds guard for dimensions not divisible by 16; without it, gradient-pyramid memory was corrupted.

**2. CPU implementation:** The CPU reference/notebook expresses the same mathematical stage; the production CPU post-processing begins after CUDA descriptors/matches.

**3. Why CPU is expensive:** repeated loops over pixels or keypoints, neighbor reads, scale levels and/or descriptor dimensions.

**4. Why CUDA:** independent output pixels/keypoints can execute concurrently.

**5. Synchronization:** no inter-thread synchronization is required by the shown per-output kernels; host wrappers call device synchronization after launch.

**6. What can go wrong:** bounds errors, global-memory traffic, branch divergence, counter races if multiple threads share a counter, and numerical differences.

**7. Viva answer:** “Our parallel unit is the output pixel or keypoint, not the whole image. GPU throughput helps repeated independent calculations; it does not eliminate data-transfer, synchronization or geometric-validation work.”

### Extrema, contrast and edge filtering
**1. What / why:** extremaKernel compares a DoG location across its 3D neighborhood. contrastKernel keeps strong extrema. edgeKernel computes Dxx, Dyy and Dxy, then rejects edge-like/invalid Hessian responses. These kernels are parallel per pixel, but have data-dependent branches and neighbor reads. Maps and DoG images are global memory; no shared-memory optimization is implemented.

**2. CPU implementation:** The CPU reference/notebook expresses the same mathematical stage; the production CPU post-processing begins after CUDA descriptors/matches.

**3. Why CPU is expensive:** repeated loops over pixels or keypoints, neighbor reads, scale levels and/or descriptor dimensions.

**4. Why CUDA:** independent output pixels/keypoints can execute concurrently.

**5. Synchronization:** no inter-thread synchronization is required by the shown per-output kernels; host wrappers call device synchronization after launch.

**6. What can go wrong:** bounds errors, global-memory traffic, branch divergence, counter races if multiple threads share a counter, and numerical differences.

**7. Viva answer:** “Our parallel unit is the output pixel or keypoint, not the whole image. GPU throughput helps repeated independent calculations; it does not eliminate data-transfer, synchronization or geometric-validation work.”

### NMS and orientation
**1. What / why:** NMS produces keypoints, then orientationKernel maps one thread to one NMS keypoint. It reads magnitude/orientation pyramid pointers and writes oriented keypoints. The host uses one-dimensional 256-thread launches. Do not claim the exact counter/atomic mechanism without rereading the final kernel.

**2. CPU implementation:** The CPU reference/notebook expresses the same mathematical stage; the production CPU post-processing begins after CUDA descriptors/matches.

**3. Why CPU is expensive:** repeated loops over pixels or keypoints, neighbor reads, scale levels and/or descriptor dimensions.

**4. Why CUDA:** independent output pixels/keypoints can execute concurrently.

**5. Synchronization:** no inter-thread synchronization is required by the shown per-output kernels; host wrappers call device synchronization after launch.

**6. What can go wrong:** bounds errors, global-memory traffic, branch divergence, counter races if multiple threads share a counter, and numerical differences.

**7. Viva answer:** “Our parallel unit is the output pixel or keypoint, not the whole image. GPU throughput helps repeated independent calculations; it does not eliminate data-transfer, synchronization or geometric-validation work.”

### 128-D descriptor
**1. What / why:** descriptorKernel maps one thread to one oriented keypoint. It initializes 128 values with an unrolled loop, reads the relevant gradient pyramid and writes a Descriptor. This is irregular and can be register-heavy; exact register count/occupancy is not measured. The current project does not show shared-memory tiling.

**2. CPU implementation:** The CPU reference/notebook expresses the same mathematical stage; the production CPU post-processing begins after CUDA descriptors/matches.

**3. Why CPU is expensive:** repeated loops over pixels or keypoints, neighbor reads, scale levels and/or descriptor dimensions.

**4. Why CUDA:** independent output pixels/keypoints can execute concurrently.

**5. Synchronization:** no inter-thread synchronization is required by the shown per-output kernels; host wrappers call device synchronization after launch.

**6. What can go wrong:** bounds errors, global-memory traffic, branch divergence, counter races if multiple threads share a counter, and numerical differences.

**7. Viva answer:** “Our parallel unit is the output pixel or keypoint, not the whole image. GPU throughput helps repeated independent calculations; it does not eliminate data-transfer, synchronization or geometric-validation work.”

### Descriptor matching
**1. What / why:** descriptorMatchingKernel maps one thread to one query descriptor. The thread scans every train descriptor, computes 128-D Euclidean distances and keeps best/second-best candidates. It is O(Q*T*128), with repeated global-memory reads. The code uses a 256-thread 1-D launch and a single scan for the two best distances.

**2. CPU implementation:** The CPU reference/notebook expresses the same mathematical stage; the production CPU post-processing begins after CUDA descriptors/matches.

**3. Why CPU is expensive:** repeated loops over pixels or keypoints, neighbor reads, scale levels and/or descriptor dimensions.

**4. Why CUDA:** independent output pixels/keypoints can execute concurrently.

**5. Synchronization:** no inter-thread synchronization is required by the shown per-output kernels; host wrappers call device synchronization after launch.

**6. What can go wrong:** bounds errors, global-memory traffic, branch divergence, counter races if multiple threads share a counter, and numerical differences.

**7. Viva answer:** “Our parallel unit is the output pixel or keypoint, not the whole image. GPU throughput helps repeated independent calculations; it does not eliminate data-transfer, synchronization or geometric-validation work.”

## 7. CPU vs CUDA Comparison
| Stage | CPU/reference | CUDA | Main limit |
|---|---|---|---|
| Gaussian/DoG/gradient | conceptual pixel loops | explicit kernels, 16x16 blocks | global reads, synchronization |
| Extrema/filters | neighborhood tests | explicit per-pixel kernels | branches/neighbor access |
| Orientation/descriptor | keypoint loops | 256-thread keypoint kernels | irregular work/registers |
| Matching | distance loops | one query thread scans train set | O(Q*T*128) |
| RANSAC/DLT | custom CPU | Numba JIT, not CUDA | iterative/control-heavy |
| Warp/blend | CPU reference | Numba parallel CPU, not CUDA | output-pixel memory traffic |

Do not claim exact GPU speedups: no CUDA-event kernel measurements are present.

## 8. Registration, Warping, Blending and Failure Handling
**Homography:** `p_prime ~ H p`; a 3x3 matrix up to scale has eight degrees of freedom and needs four non-collinear pairs. Custom RANSAC samples four matches, scores inliers by reprojection error, then refines using all inliers. Realtime DLT uses Hartley normalization: center/scale points, solve by SVD, then denormalize.

**Warp:** current `inverse_warp_vectorised` is CPU/Numba: invert H, map each destination pixel backwards, bilinearly sample source. **Blend:** distance transforms yield feather weights; `_blend_kernel` fuses weights/RGB and cached distance maps avoid repeat work.

**Implemented validator:** finite H, determinant bounds, >=12 inliers, >=30% ratio, median reprojection <3 px, perspective limits, corner area/convexity/bounds, 3x3 coverage, temporal corner stability.

```text
Low overlap -> weak/clustered matches -> unstable H -> bad warp/distortion
Candidate H -> validation -> ACCEPT: cache H | REJECT: keep last valid H
```

States: UNINITIALIZED -> TRACKING -> TRACKING_DEGRADED -> TRACKING_LOST. With no valid H, safe fallback is side-by-side feeds.

## 9. Performance, Correctness and Limitations
**Recorded facts:** cached warp+blend ~23.0 ms (~43 FPS); 30 FPS budget 33.3 ms; validation background-only. Legacy background refresh ~1200 ms. Direct DLL target ~780 ms is expected, not hardware-verified. The earlier ~25-second value is batch/development end-to-end work, not per-frame hot-path latency.

| Stage | Value | Honest statement |
|---|---:|---|
| CUDA kernels | VERIFY | no event timing / no speedup claim |
| RANSAC+DLT | collected metric | configuration dependent |
| Warp+blend | ~23 ms | measured hot path |
| Refresh | ~1200 ms legacy | direct bridge target ~780 ms |

**Correctness:** compare CPU/GPU maps/counts/descriptors/final panorama with tolerances; use fixed RANSAC seed; validate indices, inliers, reprojection and corners. Small CPU/GPU differences can arise from floating-point order/FMA/parallel rounding.

**Limitations:** verified two-camera path although CUDA has `MAX_CAMERAS=6`; parallax, lens distortion/exposure, USB synchronization, O(Q*T) match work, global traffic, repeated synchronization and legacy file I/O. CUDA does not fix calibration or poor overlap.

## 10. The Why Behind Every Design Decision
| Why? | Strong answer |
|---|---|
| SIFT | repeatable scale/rotation-aware local features and custom-learning goal |
| Gaussian/DoG | scale-space blobs efficiently |
| CUDA | dense data-parallel stage throughput |
| 16x16 / 256 blocks | actual practical mappings; multiples of warp size; not claimed globally optimal |
| constant Gaussian data | small read-only reused coefficients |
| no shared-memory claim | source has no explicit tiling; profile before adding complexity |
| RANSAC | visual similarity does not prove geometric consistency |
| validation | solvable H can still explode globally |
| inverse warp | avoids holes |
| feather blend | suppresses seam visibility |
| caching/background refresh | protects display latency |

## 11. 30 Core Viva Questions

### 1. What is CUDA?
**What evaluator tests:** project understanding.

**Short answer:** CUDA is NVIDIA’s platform for host-launched GPU kernels. We use it for custom SIFT stages with repeated independent pixel/keypoint work.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 2. Why CUDA instead of CPU?
**What evaluator tests:** project understanding.

**Short answer:** The advantage is data-parallel throughput, not merely “more cores”. Gaussian, DoG, gradient, filtering and descriptor work apply the same operation across many elements.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 3. What are grid, block and thread?
**What evaluator tests:** project understanding.

**Short answer:** A grid is a kernel launch; it contains blocks; blocks contain threads. PanoCall uses 16x16 image blocks and 256-thread 1-D keypoint/matching blocks.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 4. What is a warp?
**What evaluator tests:** project understanding.

**Short answer:** A warp is 32 SIMT threads scheduled together. Different branch paths inside one warp cause divergence.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 5. What is an SM?
**What evaluator tests:** project understanding.

**Short answer:** A Streaming Multiprocessor schedules blocks/warps and provides execution resources, registers and shared memory.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 6. Why bounds checks?
**What evaluator tests:** project understanding.

**Short answer:** Rounded grids launch threads beyond image/keypoint limits. The guard prevents out-of-bounds reads/writes.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 7. What is coalescing?
**What evaluator tests:** project understanding.

**Short answer:** Adjacent threads accessing adjacent global-memory addresses combine efficiently. Row-wise image output writes are naturally coalesced.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 8. Do we use shared memory?
**What evaluator tests:** project understanding.

**Short answer:** No explicit shared-memory tiling was found. Do not claim it; current kernels use global memory plus constant Gaussian data.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 9. What is occupancy?
**What evaluator tests:** project understanding.

**Short answer:** Resident active warps divided by an SM maximum. It can hide latency, but 100% occupancy is not automatically optimal.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 10. What is cudaDeviceSynchronize?
**What evaluator tests:** project understanding.

**Short answer:** A device-wide wait. The current wrappers use it after stages for correctness/error checks; this prevents overlap and is not a timing tool.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 11. Why CUDA events?
**What evaluator tests:** project understanding.

**Short answer:** They measure GPU elapsed time accurately. They are not implemented in this repository, so per-kernel timings must be marked VERIFY FROM CODE.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 12. Why SIFT?
**What evaluator tests:** project understanding.

**Short answer:** It provides repeatable local features under scale and rotation changes, and custom implementation was an explicit project goal.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 13. Why DoG?
**What evaluator tests:** project understanding.

**Short answer:** Subtracting adjacent Gaussian scales produces scale-space blob responses efficiently.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 14. Why RANSAC?
**What evaluator tests:** project understanding.

**Short answer:** Appearance-based matches contain outliers. RANSAC finds a consensus geometric model before refinement.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 15. Why Hartley normalization?
**What evaluator tests:** project understanding.

**Short answer:** Centering/scaling points before SVD improves DLT conditioning and reduces numerical instability.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 16. Why validation after RANSAC?
**What evaluator tests:** project understanding.

**Short answer:** A solvable RANSAC matrix can still be physically implausible, clustered, folded or temporally unstable.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 17. What does inverse warping do?
**What evaluator tests:** project understanding.

**Short answer:** It maps each destination pixel back to the source and bilinearly samples it, avoiding forward-mapping holes.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 18. Why feather blending?
**What evaluator tests:** project understanding.

**Short answer:** Distance-based weights make the overlap transition gradual and reduce visible seams.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 19. How did you reach 23 ms?
**What evaluator tests:** project understanding.

**Short answer:** The live path caches valid geometry/maps, moves refresh to a background worker, uses threaded capture and Numba warp/blend. The measured warp+blend hot path is about 23 ms.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

### 20. What happens on bad alignment?
**What evaluator tests:** project understanding.

**Short answer:** The nine-stage validator rejects it, retains the last accepted H, increments failures and updates the tracking state.

**Strong answer:** Connect the definition to the actual PanoCall kernel/pipeline above.

**Likely follow-up:** What must you not claim?

**Ideal reply:** “I will quote only measured timings and inspected implementation facts; kernel events, occupancy and speedups are VERIFY FROM CODE.”

**Common mistake:** confusing Numba CPU parallelism with CUDA.

## 12. 30 Difficult Follow-Up Questions

1. **Why not 512 threads?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

2. **Why is 100% occupancy not always best?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

3. **Why not all-GPU warp/blend?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

4. **Why can a valid homography be wrong?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

5. **Why not just OpenCV?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

6. **What if width is not divisible by 16?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

7. **Why is __syncthreads insufficient globally?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

8. **Does the ratio test prove geometry?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

9. **What if features are clustered in one corner?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

10. **Why can panoramas ghost?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

11. **What is parallax?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

12. **Why can CPU/GPU outputs differ?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

13. **Why are vertical accesses harder to coalesce?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

14. **What is register pressure?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

15. **Why are repeated device synchronizations expensive?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

16. **How would you profile?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

17. **How do you prove 23 ms?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

18. **Why is the old 25-second value not contradictory?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

19. **How would you debug panorama explosion?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

20. **Why retain old H?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

21. **Why is matching expensive?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

22. **Why can display remain fast while refresh is slow?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

23. **Why are minimum inliers insufficient?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

24. **Why corner convexity?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

25. **Why cache blend distances?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

26. **Why two capture threads?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

27. **What would you optimize next?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

28. **What is the honest CUDA limitation?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

29. **Why does Amdahl’s Law matter?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

30. **Why can transfers remove GPU speedup?**  Strong answer: GPU performance depends on parallel work, global-memory behavior, transfers, synchronization and serial stages; answer with the relevant project fact above and never invent a profiler result. For geometric questions: use RANSAC plus the implemented nine-stage validation, retained-H rule and tracking state.

## 13. Rapid-Fire Revision (50+)

1. **CUDA?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

2. **host?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

3. **device?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

4. **kernel?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

5. **thread?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

6. **block?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

7. **grid?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

8. **warp?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

9. **SM?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

10. **SIMT?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

11. **occupancy?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

12. **register pressure?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

13. **shared memory?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

14. **global memory?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

15. **constant memory?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

16. **coalescing?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

17. **bank conflict?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

18. **divergence?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

19. **latency hiding?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

20. **cudaMemcpy?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

21. **cudaDeviceSynchronize?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

22. **CUDA event?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

23. **stream?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

24. **SIFT?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

25. **octave?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

26. **scale?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

27. **Gaussian blur?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

28. **DoG?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

29. **extremum?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

30. **contrast filter?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

31. **edge filter?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

32. **NMS?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

33. **orientation?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

34. **descriptor?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

35. **Lowe ratio?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

36. **matching?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

37. **homography?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

38. **RANSAC?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

39. **inlier?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

40. **reprojection error?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

41. **Hartley normalization?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

42. **inverse warp?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

43. **bilinear interpolation?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

44. **canvas?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

45. **feather blend?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

46. **latency?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

47. **throughput?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

48. **Amdahl’s Law?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

49. **parallax?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

50. **tracking degraded?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

51. **tracking lost?** Review its definition and PanoCall relevance in Sections 3-9. Give a one-sentence definition, then one project example.

## 14. Explain the Project in 60 Seconds
“PanoCall is a multi-camera VR calling prototype. Overlapping frames are registered using our custom SIFT pipeline, matched, filtered by RANSAC and mapped with a homography before inverse warping and feather blending. CUDA accelerates the dense SIFT stages through descriptor matching, where each pixel or keypoint is an independent parallel unit. The live path then caches validated geometry and runs warp/blend separately from background alignment refresh. We added Hartley-normalized DLT, nine geometric validation checks and tracking states so a bad transform cannot corrupt the display. The recorded cached warp-and-blend path is about 23 ms, around 43 FPS.”

### 2-minute answer
Add block sizes, constant/global memory, CPU/Numba geometry, legacy file bridge and pending direct DLL compilation.

### 5-minute answer
Walk: capture -> CUDA SIFT -> matching -> CPU/Numba RANSAC/DLT -> validation -> cache -> inverse warp -> blend -> tracking/recovery -> timing/limitations. Explain Gaussian or extrema as one concrete kernel.

## 15. If the Evaluator Challenges Me
**“Why not OpenCV?”** The project intentionally implements SIFT/matching/RANSAC/DLT logic rather than relying on `cv2.SIFT_create`/`findHomography`; OpenCV is used mainly for I/O/visualization.

**“How do you know 23 ms is valid?”** Warm up Numba, time repeated hot-path warp+blend with `perf_counter`, report configuration/sample count/average/p95. Do not call it a CUDA-kernel time.

**“What is the bottleneck?”** Do not claim a measured CUDA bottleneck without events/Nsight. Known candidates are O(Q*T*128) matching, global traffic, sync, transfers/file I/O and CPU stages.

**“What next?”** Build/benchmark direct CUDA DLL; add events/Nsight; reduce debug copies/synchronization; calibrate cameras; profile before considering shared-memory tiling.

## 16. Final One-Page Cheat Sheet
**CUDA:** host -> kernel -> grid/block/warp/thread; SM schedules; 16x16 images, 256 keypoints; global pyramids; constant Gaussian values; no explicit shared tiles; bounds guards; events time kernels.

**SIFT:** Gaussian -> DoG -> extrema -> filters -> NMS -> orientation -> 128-D descriptor -> ratio match.

**Stitch:** matches -> RANSAC -> Hartley DLT -> validate -> inverse warp -> feather blend.

**Real time:** cache valid H/maps; background refresh; retain prior H after rejection; ~23 ms/~43 FPS hot path; do not fabricate speedups.

## 17. Stage-by-Stage CUDA Examination Cards (Source-Checked)

This section is the authoritative viva format for the computational stages. It deliberately separates source facts from engineering implications. `sift_stitcher.cu` uses explicit `cudaDeviceSynchronize()` calls after its launched stages for error detection; those calls are host-side stage boundaries, not intra-kernel cooperation.

### Gaussian Scale Space

#### 1. What is this stage?
It creates progressively blurred versions of an image. PanoCall implements each blur as separable horizontal and vertical Gaussian passes.

#### 2. Why is this stage necessary?
SIFT needs feature evidence at more than one scale; the blurred images are the inputs to DoG and gradients.

#### 3. CPU implementation
The mathematical CPU equivalent loops over pixels and a Gaussian window. In the inspected production source, the operational implementation is CUDA; do not claim a separately profiled CPU Gaussian implementation.

#### 4. Why is CPU computation expensive here?
Every output pixel reads several neighbours, and this is repeated for several scales and octaves.

#### 5. Why is CUDA useful here?
Each output pixel is independent once the input image is available, giving a large regular data-parallel workload.

#### 6. What does one CUDA thread do?
One thread computes one horizontal or vertical output pixel, accumulates the weighted convolution window, and writes one float.

#### 7. What does one block do?
A `dim3(16,16)` block covers a 16-by-16 tile of output pixels (256 threads); the grid is rounded up to cover the image.

#### 8. What memory is used?
Input/output image buffers are global memory. `d_gaussianKernel` and `d_kernelRadius` are uploaded with `cudaMemcpyToSymbol` and read from constant memory. The source does not show explicit shared-memory tiling.

#### 9. Is synchronization required?
Threads do not exchange partial sums, so no `__syncthreads()` is required. The host synchronizes between horizontal and vertical launches because the vertical pass consumes the horizontal buffer.

#### 10. What can go wrong?
Missing bounds checks, an invalid radius, edge handling errors, and excessive global-memory traffic can break correctness or performance.

#### 11. What optimization did we perform?
The source uses separability (two 1-D convolutions instead of one 2-D convolution) and constant-memory Gaussian coefficients. It does not implement a shared-memory tile.

#### 12. What bottleneck remains?
Repeated global loads, especially in the vertical pass, plus the current per-stage device synchronizations.

#### 13. What performance improvement did we obtain?
**VERIFY FROM CODE:** no CUDA-event kernel time or CPU-vs-GPU Gaussian speedup is recorded.

#### 14. Likely viva questions
Why is Gaussian blur separable? Why constant memory? Why 16-by-16? Why is vertical access harder? Why not claim shared memory?

#### 15. Strong viva answers
“We map one thread to one output pixel. Separability reduces the convolution work, and the small reusable coefficient array is placed in constant memory. The block size is an implemented choice, not a proven hardware optimum; profiling would be needed to tune it.”

### Downsampling and Difference of Gaussian (DoG)

#### 1. What is this stage?
`downsampleKernel` produces the next octave at half resolution; `dogKernel` computes `G(i+1) - G(i)` at each location.

#### 2. Why is this stage necessary?
Downsampling expands scale coverage. DoG is the efficient blob-like response used to find candidate SIFT extrema.

#### 3. CPU implementation
The CPU reference is per-output-pixel averaging/subtraction. The inspected implementation launches the corresponding CUDA kernels rather than a timed CPU version.

#### 4. Why is CPU computation expensive here?
The work spans every pixel of every scale/octave, even though each individual subtraction is small.

#### 5. Why is CUDA useful here?
There is no dependency between output pixels within either kernel.

#### 6. What does one CUDA thread do?
For downsampling, one thread writes one half-resolution pixel from its 2-by-2 input region. For DoG, one thread subtracts two same-coordinate Gaussian values and writes one result.

#### 7. What does one block do?
It is a 16-by-16 tile of the output image; the launch grid uses ceiling division.

#### 8. What memory is used?
The shown kernels use global input/output buffers and thread registers for indices/temporary values. No explicit shared memory is shown.

#### 9. Is synchronization required?
Not within the kernels. The host must complete producer stages before a dependent downsample/DoG stage begins.

#### 10. What can go wrong?
Out-of-range source coordinates, bad output dimensions, or a missing bounds guard can corrupt pyramid storage.

#### 11. What optimization did we perform?
The project exposes the regular per-pixel work as CUDA kernels using 16-by-16 launches. No further source-confirmed optimization should be claimed.

#### 12. What bottleneck remains?
These operations have low arithmetic intensity and can become memory-bandwidth/launch-overhead limited.

#### 13. What performance improvement did we obtain?
**VERIFY FROM CODE:** no stage timing is recorded.

#### 14. Likely viva questions
Why do we downsample? Why is DoG preferable to a direct Laplacian approximation? Is DoG compute- or memory-heavy? Why are guards needed?

#### 15. Strong viva answers
“The GPU is useful because the same simple operation is repeated at every coordinate. That does not guarantee a dramatic end-to-end speedup because low-work kernels can be dominated by memory and launch costs.”

### Gradients

#### 1. What is this stage?
`gradientKernel` applies Sobel neighbourhood calculations to output gradient magnitude and orientation.

#### 2. Why is this stage necessary?
Orientation assignment and the descriptor need local gradient magnitude and direction rather than raw intensity.

#### 3. CPU implementation
The CPU equivalent applies Sobel arithmetic per non-border pixel. The GPU source is the executed custom implementation described here.

#### 4. Why is CPU computation expensive here?
It repeats nine-neighbour reads, arithmetic, `sqrtf`, and `atan2f` across the pyramid.

#### 5. Why is CUDA useful here?
Each output pixel has a fixed local stencil and can be evaluated independently.

#### 6. What does one CUDA thread do?
One thread reads its 3-by-3 neighbourhood, computes `gx`, `gy`, magnitude and angle, normalizes a negative angle to `[0,360)`, then writes two outputs.

#### 7. What does one block do?
One 16-by-16 block computes one output tile. The code explicitly uses a rounded grid.

#### 8. What memory is used?
Neighbourhood samples and magnitude/orientation outputs are global-memory accesses; arithmetic temporaries are registers. No shared-memory stencil cache is implemented.

#### 9. Is synchronization required?
No, because a thread writes only its own magnitude/orientation positions. The host synchronizes after the launch for checking.

#### 10. What can go wrong?
Border loads are invalid unless handled. The source explicitly fixed the guard for non-multiple-of-16 dimensions and writes border outputs as zero.

#### 11. What optimization did we perform?
The verified corrective optimization is the grid bounds guard that prevented gradient-pyramid corruption for rounded grids. Do not present this correctness fix as a measured speed optimization.

#### 12. What bottleneck remains?
Neighbour reads and transcendental operations; row-major accesses are friendlier than the vertical components of the stencil.

#### 13. What performance improvement did we obtain?
**VERIFY FROM CODE:** no per-kernel timing exists.

#### 14. Likely viva questions
Why Sobel? Why zero borders? What happens at width 640/height 360? Why can a safe rounded grid include extra threads?

#### 15. Strong viva answers
“The grid intentionally rounds up, so bounds guards are compulsory. We fixed a real corruption risk: threads beyond an octave’s dimensions had been able to write outside the gradient pyramid.”

### Extrema, Contrast, and Edge Filtering

#### 1. What is this stage?
`extremaKernel` tests a DoG pixel against 26 neighbours across three scales; `contrastKernel` and `edgeKernel` remove weak and edge-like candidates.

#### 2. Why is this stage necessary?
It turns dense DoG responses into repeatable, informative keypoint candidates instead of keeping noise and unstable edge responses.

#### 3. CPU implementation
The CPU equivalent is nested neighbourhood comparisons and Hessian/threshold tests. The inspected project launches CUDA per-pixel filtering kernels.

#### 4. Why is CPU computation expensive here?
The extrema test performs many neighbour comparisons at every location and its branch outcomes vary by image content.

#### 5. Why is CUDA useful here?
Each candidate location can test its own local 3-D neighbourhood independently.

#### 6. What does one CUDA thread do?
One thread reads the three DoG levels around its coordinate, tracks maximum/minimum status, and writes an extrema-map byte. The later filters evaluate the same coordinate’s criteria.

#### 7. What does one block do?
A 16-by-16 image tile of candidate locations.

#### 8. What memory is used?
DoG images and output maps are global memory; flags/indices are registers. No explicit shared-memory neighbourhood tile is present.

#### 9. Is synchronization required?
No per-thread reduction is used. Dependencies are only between completed pipeline stages.

#### 10. What can go wrong?
The source notes that `cudaMalloc` memory is uninitialized: boundary extrema entries must be explicitly zeroed or phantom keypoints can result. Divergent comparisons and border logic are also risks.

#### 11. What optimization did we perform?
The code explicitly writes border extrema to zero and includes bounds guards. Those are source-confirmed robustness fixes.

#### 12. What bottleneck remains?
High neighbour-read count, branches, and potentially poor cache locality; this is less regular than DoG subtraction.

#### 13. What performance improvement did we obtain?
**VERIFY FROM CODE:** no timing/speedup is recorded.

#### 14. Likely viva questions
Why 26 neighbours? Why reject edges? Why can a zeroed map matter? Does every branch cause a race condition?

#### 15. Strong viva answers
“A race is not the issue here because each thread owns one map location. The real hazards are invalid/border data and divergent control flow. We explicitly clear the border because allocated device memory is not automatically zero.”

### Orientation Assignment

#### 1. What is this stage?
For every NMS keypoint, `orientationKernel` builds a 36-bin, magnitude-weighted local orientation histogram and emits each peak at least 80% of the maximum.

#### 2. Why is this stage necessary?
Assigning a dominant orientation lets later descriptors be expressed relative to the keypoint, improving rotation robustness.

#### 3. CPU implementation
The CPU equivalent scans a local window, accumulates a histogram, finds its peaks, and may make multiple oriented keypoints. The implemented CUDA form assigns that entire local task to one thread.

#### 4. Why is CPU computation expensive here?
Each keypoint scans a radius-dependent patch, evaluates Gaussian weights, and builds a histogram; the number of retained orientations is data-dependent.

#### 5. Why is CUDA useful here?
Different keypoints are independent, so many keypoint histograms can be evaluated concurrently.

#### 6. What does one CUDA thread do?
It reads one NMS keypoint, scans its gradient window, accumulates a private `float histogram[36]`, finds peak bins, then emits one or more oriented keypoints.

#### 7. What does one block do?
The actual launcher uses one-dimensional blocks of 256 threads, representing up to 256 input keypoints.

#### 8. What memory is used?
Histogram/local values are thread-local (normally registers or local spill; exact placement **VERIFY FROM CODE**). Gradient pyramids and pointer/dimension arrays are global memory; sigma levels are in constant memory.

#### 9. Is synchronization required?
No block reduction is used because each histogram belongs to one thread. An `atomicAdd` on `orientationCount` reserves a unique output slot when multiple threads emit results.

#### 10. What can go wrong?
The variable patch and peak count can diverge. Atomic contention and output overflow are possible; the code has a `maxOrientations` capacity guard after reservation.

#### 11. What optimization did we perform?
The source caps orientation writes to prevent a flat histogram from producing up to 36 entries and corrupting the four-per-NMS-keypoint allocation.

#### 12. What bottleneck remains?
Irregular per-keypoint work, gradient-pyramid reads, atomics, and a 36-float private histogram can create register pressure/spilling.

#### 13. What performance improvement did we obtain?
**VERIFY FROM CODE:** no measured kernel result is present.

#### 14. Likely viva questions
Why 36 bins? Why can one keypoint yield several orientations? Why atomicAdd? Why is a private histogram safer than a shared one here?

#### 15. Strong viva answers
“The histogram is private to one thread, so it needs no histogram atomic. The only shared item is the global output count; atomicAdd makes each accepted orientation get a unique slot.”

### 128-D Descriptor Generation

#### 1. What is this stage?
`descriptorKernel` constructs one 4-by-4-by-8 (128-value) SIFT descriptor around each oriented keypoint, then L2-normalizes, clamps to 0.2, and normalizes again.

#### 2. Why is this stage necessary?
The descriptor converts a local patch into a comparable, rotation-referenced feature vector used by matching.

#### 3. CPU implementation
The CPU equivalent loops through the rotated 16-by-16 patch and its spatial/orientation bins. The current CUDA kernel keeps one descriptor build within one thread.

#### 4. Why is CPU computation expensive here?
Each keypoint has patch traversal, coordinate rotation, Gaussian weighting, trilinear-style bin contributions, and several 128-element passes.

#### 5. Why is CUDA useful here?
Descriptors for different oriented keypoints are independent and can execute concurrently.

#### 6. What does one CUDA thread do?
One thread initializes `descriptor.data[128]`, scans a radius-8 patch, rotates coordinates by the keypoint angle, interpolates contributions into 4-by-4-by-8 bins, normalizes/clamps/renormalizes, then stores one descriptor.

#### 7. What does one block do?
The launcher uses one-dimensional 256-thread blocks, so it processes up to 256 oriented keypoints per block.

#### 8. What memory is used?
The descriptor accumulator is per-thread; exact register/local-memory placement is **VERIFY FROM CODE**. Pyramids and descriptor output are global memory. The source has no shared-memory tile.

#### 9. Is synchronization required?
No threads collaborate on one descriptor. `atomicAdd(descriptorCount, 1)` reserves an output position, so it is required for safe concurrent append.

#### 10. What can go wrong?
Patch boundary checks, angle wrapping, bin bounds, counter/output capacity, register pressure, and small floating-point differences all matter.

#### 11. What optimization did we perform?
The source uses `#pragma unroll` for several 128-element loops. It does not show a measured register/occupancy tuning pass.

#### 12. What bottleneck remains?
Irregular patch access, transcendental functions, a large private working set, and atomic output reservation.

#### 13. What performance improvement did we obtain?
**VERIFY FROM CODE:** no numerical speedup is recorded.

#### 14. Likely viva questions
Why 128 dimensions? Why two normalizations? Why clamp at 0.2? Why one thread per descriptor rather than one thread per element?

#### 15. Strong viva answers
“One thread owns a complete descriptor, avoiding cross-thread reductions and synchronization. The trade-off is a larger private working set and irregular patch reads, which is why we must not assume this stage has ideal occupancy.”

### Descriptor Matching

#### 1. What is this stage?
`descriptorMatchingKernel` finds the nearest and second-nearest train descriptors for every query descriptor using 128-D Euclidean distance and Lowe’s 0.75 ratio test.

#### 2. Why is this stage necessary?
Descriptor correspondences supply candidate point pairs for RANSAC/homography estimation.

#### 3. CPU implementation
The CPU equivalent uses nested query/train loops and a 128-element distance computation. The project implements the shown exhaustive search as a CUDA kernel.

#### 4. Why is CPU computation expensive here?
Its cost is `O(Q × T × 128)` and repeated reads of train descriptors dominate as query/train counts grow.

#### 5. Why is CUDA useful here?
Queries are independent, so all query descriptors can scan the train set concurrently.

#### 6. What does one CUDA thread do?
One thread loads one query descriptor, scans every train descriptor, tracks best and second-best Euclidean distances, applies ratio 0.75, then may append one match.

#### 7. What does one block do?
The launcher uses one-dimensional 256-thread blocks: each block handles up to 256 query descriptors.

#### 8. What memory is used?
Query/train descriptors and match output are global memory; best distances and the query copy are private to a thread. No shared train-tile cache is in the inspected source.

#### 9. Is synchronization required?
Threads do not cooperate in a query search. `atomicAdd(matchCount, 1)` is necessary when accepted queries append to the shared match array.

#### 10. What can go wrong?
Empty train sets, output-capacity issues, ratio-test ambiguity, atomic contention, and bandwidth-heavy repeated train reads.

#### 11. What optimization did we perform?
The source unrolls the fixed 128-D distance loop and tracks the two best candidates in one pass. It does not show spatial indexing or a shared-memory training tile.

#### 12. What bottleneck remains?
Exhaustive matching is quadratic in descriptor counts and may be dominated by global-memory traffic.

#### 13. What performance improvement did we obtain?
**VERIFY FROM CODE:** there is no event-derived match-kernel timing or speedup percentage.

#### 14. Likely viva questions
Why second-best distance? Why 0.75? Is matching geometrically sufficient? What is its asymptotic complexity? Why use atomicAdd?

#### 15. Strong viva answers
“The ratio test rejects ambiguous nearest neighbours; it does not prove geometry. RANSAC and the later validator are still required. Each query is independent, but exhaustive scanning remains an honest `O(QT×128)` bottleneck.”

## 18. Core Viva Questions 21–30

### 21. Why is SIFT suitable for CUDA?
**What the evaluator is testing:** whether you can connect algorithm shape to hardware.  
**Short answer:** many SIFT calculations repeat independently over pixels or keypoints.  
**Strong answer:** “PanoCall maps image stages to pixels and later stages to keypoints/queries. That creates parallel work, while the irregular parts—atomics and branch-heavy filtering—remain limits.”  
**Technical explanation:** Gaussian/DoG/gradient are regular; descriptor/orientation are keypoint-parallel.  
**Likely follow-up:** Does that mean every stage has equal speedup?  
**Ideal follow-up answer:** No—transfers, atomics, irregularity, memory traffic, and CPU stages cap it.  
**Common mistake:** claiming all SIFT stages are perfectly parallel.

### 22. Which SIFT stage benefits most?
**What the evaluator is testing:** performance honesty.  
**Short answer:** the codebase does not record a measured winner.  
**Strong answer:** “Regular dense stages are strong CUDA candidates, but I will not name a fastest stage without CUDA-event profiling.”  
**Technical explanation:** regular work maps cleanly, but time depends on image size, radii, occupancy, and memory.  
**Likely follow-up:** How would you prove it?  
**Ideal follow-up answer:** add warm-up runs and CUDA events around each kernel.  
**Common mistake:** inventing a percentage.

### 23. How did you parallelize Gaussian filtering?
**What the evaluator is testing:** kernel mapping.  
**Short answer:** one output pixel per thread in separate horizontal/vertical passes.  
**Strong answer:** “We launch 16-by-16 blocks, read Gaussian coefficients from constant memory, write an intermediate horizontal buffer, then consume it in the vertical pass.”  
**Technical explanation:** separability lowers convolution work.  
**Likely follow-up:** Is shared memory used?  
**Ideal follow-up answer:** No explicit shared-memory tiling was found.  
**Common mistake:** claiming a tiled convolution.

### 24. How did you parallelize DoG?
**What the evaluator is testing:** simple data mapping.  
**Short answer:** one thread subtracts two same-coordinate Gaussian values.  
**Strong answer:** “DoG is regular, 16-by-16 pixel parallel work with a bounds guard and global buffers.”  
**Technical explanation:** output values have no inter-thread dependency.  
**Likely follow-up:** Why not synchronize?  
**Ideal follow-up answer:** no thread shares an output; only stage ordering matters.  
**Common mistake:** confusing host synchronization with thread synchronization.

### 25. How did you parallelize extrema detection?
**What the evaluator is testing:** 3-D neighbourhood understanding.  
**Short answer:** one thread tests one DoG location against 26 neighbours.  
**Strong answer:** “The thread reads lower/current/upper DoG planes, produces one extrema-map byte, and explicitly clears image borders.”  
**Technical explanation:** 3-by-3 in each of three scales, excluding duplicate centre comparison.  
**Likely follow-up:** What was a real bug risk?  
**Ideal follow-up answer:** uninitialized `cudaMalloc` border values could become phantom keypoints.  
**Common mistake:** saying the 26-neighbour test uses atomics.

### 26. How did you handle descriptor generation?
**What the evaluator is testing:** descriptor details and atomics.  
**Short answer:** one thread owns one oriented keypoint’s 128-D descriptor.  
**Strong answer:** “It scans a radius-8 patch, rotates it, distributes weighted gradient votes into 4-by-4-by-8 bins, normalizes, clamps, renormalizes, then atomically reserves output storage.”  
**Technical explanation:** thread ownership avoids cross-thread descriptor reductions.  
**Likely follow-up:** What is the downside?  
**Ideal follow-up answer:** irregular work and register/local-memory pressure.  
**Common mistake:** claiming 128 threads make one descriptor.

### 27. What was your biggest bottleneck?
**What the evaluator is testing:** evidence discipline.  
**Short answer:** no profile identifies a single CUDA-kernel bottleneck.  
**Strong answer:** “For matching, asymptotic work is `O(QT×128)`; for live display, the measured hot path is warp+blend at about 23 ms. A kernel-level bottleneck needs profiling.”  
**Technical explanation:** pipeline bottleneck differs from a kernel bottleneck.  
**Likely follow-up:** What would you optimize next?  
**Ideal follow-up answer:** profile first; likely reduce repeated transfers/synchronizations and investigate matching memory reuse.  
**Common mistake:** calling SIFT the measured bottleneck without evidence.

### 28. How did you validate correctness?
**What the evaluator is testing:** engineering rigor.  
**Short answer:** validate intermediate/final outputs and geometry, with tolerances.  
**Strong answer:** “Use CPU/GPU comparisons of maps, keypoints, descriptors and panorama; use a fixed RANSAC seed; then apply the nine-stage homography validator before caching H.”  
**Technical explanation:** numerical closeness and physically plausible geometry are separate checks.  
**Likely follow-up:** Why may values differ?  
**Ideal follow-up answer:** floating-point order and GPU execution differ.  
**Common mistake:** expecting bit-identical results.

### 29. How did you achieve approximately 23 ms/frame?
**What the evaluator is testing:** separation of hot path and refresh path.  
**Short answer:** cache accepted geometry and optimize the CPU render path.  
**Strong answer:** “The recorded ~23 ms is cached warp+blend, using Numba-compiled loops and cached blend distances; SIFT/RANSAC/validation refresh runs outside that per-frame hot path.”  
**Technical explanation:** this is approximately 43 FPS and below the 33.3-ms 30-FPS budget.  
**Likely follow-up:** Is it CUDA kernel time?  
**Ideal follow-up answer:** No; it is the CPU/Numba live render hot path.  
**Common mistake:** attributing all 23 ms to CUDA SIFT.

### 30. What exactly did you optimize?
**What the evaluator is testing:** ownership and concrete changes.  
**Short answer:** data safety, live-path work, and geometry robustness.  
**Strong answer:** “Source-confirmed CUDA changes include rounded-grid guards, zeroed extrema borders, and output capacity protection. The live path adds Numba warp/blend, cache reuse, background refresh, Hartley-normalized DLT, and H validation.”  
**Technical explanation:** these changes improve different things—correctness, latency, or failure containment.  
**Likely follow-up:** What remains?  
**Ideal follow-up answer:** direct CUDA DLL integration and profile-guided GPU tuning.  
**Common mistake:** saying every optimization was CUDA-specific.

## 19. Final One-Page Cheat Sheet

### CUDA

- **CUDA:** NVIDIA's host/device programming platform; PanoCall uses custom CUDA C++ SIFT kernels.
- **Kernel / grid / block / thread:** a host launches a kernel as a grid of blocks; a thread owns one unit of work.
- **Warp / SM / SIMT:** 32 threads execute in a warp on an SM; different branch paths cause divergence.
- **PanoCall mappings:** image kernels use 16-by-16 blocks; orientation, descriptor, and matching kernels use 256-thread 1-D blocks.
- **Registers / shared / global / constant:** private temporaries; no explicit shared tiling found; image/pyramid/descriptor buffers; Gaussian/sigma constants.
- **Coalescing:** adjacent threads accessing adjacent row addresses is favourable. **Occupancy:** useful for latency hiding, not a performance guarantee.
- **Synchronization:** `__syncthreads()` is block-local; `cudaDeviceSynchronize()` is host-side device completion. Atomics safely reserve output slots.
- **CUDA events:** correct tool for GPU kernel timing; no event data is currently recorded.

### SIFT

- **Gaussian scale space -> DoG -> extrema -> contrast/edge filtering -> NMS -> orientation -> 128-D descriptor -> matching.**
- **Orientation:** 36-bin weighted histogram; peaks at 80% of its maximum can yield multiple orientations.
- **Descriptor:** 4-by-4 spatial cells times 8 orientation bins; normalize, clamp at 0.2, renormalize.
- **Matching:** one query thread exhaustively checks train descriptors; nearest/second-nearest with Lowe ratio 0.75; complexity is `O(Q x T x 128)`.

### Stitching

- **Matches -> RANSAC -> Hartley-normalized DLT -> validate H -> inverse warp -> feather blend.**
- **Homography:** `p' ~ H p`; needs four non-collinear pairs. A mathematically solvable H is not automatically physically valid.
- **Validator:** inliers, ratio, reprojection, perspective, corner sanity, coverage, and temporal stability protect the cached transform.
- **Failure rule:** reject a bad candidate H and retain the prior valid H; fall back safely if tracking is lost.

### Performance and Viva Truth Rule

- **~23 ms/frame (~43 FPS)** is the recorded cached **CPU/Numba warp + blend hot path**, not a CUDA SIFT-kernel measurement.
- Background SIFT/matching/RANSAC/validation refresh is separate from display. Caching and precomputed blend distances protect per-frame latency.
- Never invent kernel speedups, occupancy, register counts, stream usage, or a single biggest CUDA bottleneck. Say **VERIFY FROM CODE / profile**.
- GPU acceleration helps data-parallel work; transfers, synchronization, irregular branches, matching complexity, RANSAC, validation, camera overlap, and parallax still matter.
