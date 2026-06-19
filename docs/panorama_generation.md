# Panorama Generation: Homography, Warping and Blending

## Project Overview

After developing a robust SIFT descriptor pipeline, the next objective was to transform feature matches into a seamless panorama.

The descriptor stage answered:

> Which points correspond between two images?

The panorama stage answers:

* How do we geometrically align the images?
* How do we place them into a common coordinate system?
* How do we combine them into a seamless panorama?

This stage involved:

* Custom RANSAC
* Custom DLT Homography Estimation
* Homography Refinement
* Error Validation
* Inverse Warping
* Dynamic Panorama Canvas Generation
* Feather Blending

---

# Final Panorama Pipeline

```text
Descriptor Matching
        ↓
Lowe Ratio Test
        ↓
Custom RANSAC
        ↓
Best Inlier Set
        ↓
Homography Refinement
        ↓
Error Validation
        ↓
Corner Projection
        ↓
Canvas Generation
        ↓
Translation Matrix
        ↓
Panorama Homography
        ↓
Inverse Warping
        ↓
Bilinear Interpolation
        ↓
Mask Generation
        ↓
Distance Transform
        ↓
Feather Blending
        ↓
Final Panorama
```

---

# Stage 1: Descriptor Matching

Using the 128-dimensional SIFT descriptors generated in the previous stage:

For every descriptor:

1. Find nearest neighbor
2. Find second nearest neighbor
3. Apply Lowe Ratio Test

### Result

```text
193 candidate matches
```

These matches represent potential correspondences between the two images.

However:

* Not all matches are correct
* Many matches are outliers

---

# Stage 2: Custom RANSAC

## Why RANSAC is Needed

Suppose we have:

```text
193 matches
```

Many correspondences may be incorrect due to:

* Repetitive textures
* Illumination changes
* Descriptor ambiguity
* Background clutter

If we directly compute a homography using all matches,

the resulting transformation will be incorrect.

---

## RANSAC Methodology

Each iteration performs:

```text
Randomly choose 4 matches
        ↓
Compute Homography
        ↓
Project all points
        ↓
Compute Reprojection Error
        ↓
Count Inliers
        ↓
Store Best Model
```

### Configuration

```text
Iterations = 2000
Threshold  = 5 pixels
```

### Why Only 4 Matches?

A homography contains:

```text
8 independent parameters
```

Each point correspondence provides:

```text
2 equations
```

Therefore:

```text
4 correspondences
=
8 equations
```

which is sufficient to solve the homography.

### Results

```text
Initial Matches    = 193
RANSAC Inliers     = 102
Rejected Outliers  = 91
```

Approximately:

```text
47% outliers removed
```

---

# Stage 3: Homography Refinement

## Problem

The homography obtained during RANSAC is estimated from:

```text
Only 4 randomly selected points
```

Although geometrically valid,

it is not necessarily the most accurate model.

---

## Solution

Use:

```text
ALL 102 inliers
```

to recompute the homography.

### Refinement Procedure

```text
Build DLT Matrix
        ↓
Solve using SVD
        ↓
Normalize Matrix
        ↓
Final Homography
```

### Why Refinement Helps

Using all inliers:

* Averages measurement noise
* Reduces estimation variance
* Improves alignment accuracy

### Result

A more stable and accurate homography.

---

## Final Homography

```text
[[ 1.2168   0.0409 -336.65 ]
 [ 0.1590   1.1098  -41.29 ]
 [ 0.00033 -0.00003  1.0   ]]
```

### Interpretation

The transformation contains:

* Scale change
* Small rotation
* Perspective distortion
* Translation

This matrix is image-dependent.

For every new image pair:

* A new homography is estimated
* The algorithm remains identical
* Only the numerical values change

---

# Stage 4: Error Validation

Computing a homography alone is not sufficient.

We must verify its accuracy.

---

## Forward Reprojection Error

Project a source point:

```text
p → H → p'
```

Compare the projected point with the true destination point.

Error:

```text
|| predicted - actual ||
```

This measures alignment quality.

---

## Symmetric Reprojection Error

Forward error alone can be misleading.

Therefore we evaluate:

```text
Forward Projection
+
Backward Projection
```

using:

```text
H
and
H⁻¹
```

Error:

```text
||p' − Hp||
+
||p − H⁻¹p'||
```

### Results

```text
Average Error            = 1.23 px
Average Symmetric Error  = 1.74 px
Maximum Error            = 4.15 px
```

These values indicate strong geometric consistency and excellent alignment quality.

---

# Stage 5: Corner Projection

Before creating the panorama, we must determine:

> Where Image 1 lands inside the coordinate system of Image 2.

### Procedure

Take the four image corners:

* Top Left
* Top Right
* Bottom Right
* Bottom Left

Apply the homography:

```text
Corner
  ↓
  H
  ↓
Projected Corner
```

### Why This Step Matters

Corner projection determines:

* Panorama width
* Panorama height
* Boundary limits

Without corner projection,

the canvas size cannot be computed.

---

# Stage 6: Dynamic Panorama Canvas Generation

## Problem

Warped images often extend outside the original image boundaries.

Examples:

```text
Negative X coordinate
Negative Y coordinate
```

Direct placement would cause cropping.

---

## Solution

Compute:

```text
Minimum X
Maximum X

Minimum Y
Maximum Y
```

across:

* Projected Image 1
* Image 2

Then generate a canvas large enough to contain both.

### Benefits

* No cropping
* Automatic expansion
* Works for arbitrary image pairs

---

# Stage 7: Translation Matrix

Even after generating the canvas,

projected coordinates may remain negative.

Example:

```text
x = -336
y = -41
```

### Solution

Apply a translation matrix:

```text
T =
[1 0 tx]
[0 1 ty]
[0 0 1 ]
```

where:

```text
tx = -min_x
ty = -min_y
```

### Result

All coordinates are shifted into positive space,

making image placement possible.

---

# Stage 8: Panorama Homography

Combine:

```text
Translation Matrix
+
Homography
```

to obtain:

```text
H_panorama = T × H
```

This becomes the final transformation used during panorama generation.

---

# Stage 9: Inverse Warping

## Why Not Forward Warping?

Forward mapping often produces:

* Holes
* Overlapping pixels
* Missing regions

---

## Solution

Use inverse warping.

For every destination pixel:

```text
Destination Pixel
        ↓
Apply H⁻¹
        ↓
Source Coordinate
        ↓
Sample Source Image
```

### Benefits

* No holes
* Complete coverage
* Stable rendering

This is the industry-standard approach used in image stitching systems.

---

# Stage 10: Bilinear Interpolation

After inverse warping,

source coordinates are often fractional.

Example:

```text
(105.3, 220.7)
```

No exact pixel exists at that location.

### Solution

Use the four neighboring pixels

to estimate the intensity value.

### Benefits

* Smooth images
* Reduced aliasing
* Sharper details

---

# Stage 11: Mask Generation

Before blending,

we must determine:

* Which pixels belong to Image 1
* Which pixels belong to Image 2

### Solution

Generate binary masks.

```text
1 → Valid Pixel
0 → Empty Pixel
```

These masks identify overlap regions.

---

# Stage 12: Distance Transform

## Problem

Direct averaging often produces visible seams.

### Solution

Compute a distance transform.

Each pixel receives a weight proportional to:

```text
Distance from image boundary
```

Pixels near the center receive:

```text
Higher Weight
```

Pixels near boundaries receive:

```text
Lower Weight
```

### Why It Works

Center pixels are generally:

* More reliable
* Less distorted

Therefore they should contribute more to the final panorama.

---

# Stage 13: Feather Blending

Feather blending combines images using normalized distance weights.

Formula:

```text
Panorama =
w1 × Image1
+
w2 × Image2
```

where:

```text
w1 + w2 = 1
```

### Benefits

* Smooth seams
* Reduced brightness discontinuities
* Natural transitions

---

# Final Results

## Matching Statistics

```text
Initial Matches   = 193
RANSAC Inliers    = 102
Rejected Outliers = 91
```

---

## Validation Statistics

```text
Average Error            = 1.23 px
Average Symmetric Error  = 1.74 px
Maximum Error            = 4.15 px
```

---

# Panorama Features Achieved

* ✅ Custom DLT Homography
* ✅ Custom RANSAC
* ✅ Homography Refinement
* ✅ Error Validation
* ✅ Dynamic Canvas Generation
* ✅ Translation-Corrected Homography
* ✅ Inverse Warping
* ✅ Bilinear Interpolation
* ✅ Distance-Based Feather Blending
* ✅ Seamless Panorama Generation

---

# Key Takeaway

The panorama generation stage transforms feature correspondences into a geometrically consistent visual representation.

Starting from **193 descriptor matches**, the pipeline uses **RANSAC** to remove outliers, refines the homography using **102 inliers**, validates alignment through reprojection errors, constructs a dynamic panorama canvas, performs inverse warping with bilinear interpolation, and finally blends images using distance-weighted feather blending.

The resulting system produces smooth, accurately aligned panoramas and establishes the core framework required for future:

* VR Scene Reconstruction
* Visual Odometry
* SLAM-Based Mapping Systems
* Augmented Reality Tracking
* Large-Scale Image Stitching Applications
