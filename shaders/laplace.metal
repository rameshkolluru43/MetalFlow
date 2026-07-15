/**
 * @file laplace.metal
 * @brief MetalFlow kernels for 2-D Laplace / CG (and MG helpers).
 *
 * Used by examples/laplace_solver.cpp via metalflow::MetalCompute.
 */

// Geometric multigrid / CG kernels for 2-D Laplace: ∇²u = 0
// Structured grid, Dirichlet boundaries held fixed (copied through).

#include <metal_stdlib>
using namespace metal;

inline bool onBoundary(uint i, uint j, uint nx, uint ny) {
    return i == 0 || j == 0 || i + 1 == nx || j + 1 == ny;
}

/// Weighted Jacobi smooth for Laplace (f = 0):
/// u_new = (1-w)*u + w*0.25*(N+S+E+W)
kernel void laplace_jacobi(device const float* u [[buffer(0)]],
                           device float* uNew [[buffer(1)]],
                           constant uint& nx [[buffer(2)]],
                           constant uint& ny [[buffer(3)]],
                           constant float& omega [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;
    uint idx = j * nx + i;
    if (onBoundary(i, j, nx, ny)) {
        uNew[idx] = u[idx];
        return;
    }
    float avg = 0.25f * (u[idx - 1] + u[idx + 1] + u[idx - nx] + u[idx + nx]);
    uNew[idx] = (1.0f - omega) * u[idx] + omega * avg;
}

/// Red-black Gauss–Seidel (color = 0 red, 1 black). In-place update.
kernel void laplace_rbgs(device float* u [[buffer(0)]],
                         constant uint& nx [[buffer(1)]],
                         constant uint& ny [[buffer(2)]],
                         constant uint& color [[buffer(3)]],
                         constant float& omega [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;
    if (((i + j) & 1u) != color) return;
    if (onBoundary(i, j, nx, ny)) return;

    uint idx = j * nx + i;
    float avg = 0.25f * (u[idx - 1] + u[idx + 1] + u[idx - nx] + u[idx + nx]);
    u[idx] = (1.0f - omega) * u[idx] + omega * avg;
}

/// Stencil residual ρ = 4u − N − S − E − W  (homogeneous Laplace; h-independent)
/// Boundary residual set to 0.
kernel void laplace_residual(device const float* u [[buffer(0)]],
                             device float* r [[buffer(1)]],
                             constant uint& nx [[buffer(2)]],
                             constant uint& ny [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;
    uint idx = j * nx + i;
    if (onBoundary(i, j, nx, ny)) {
        r[idx] = 0.0f;
        return;
    }
    r[idx] = 4.0f * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - nx] - u[idx + nx];
}

/// Squared residual contribution per cell (interior only) → for L2 norm
kernel void laplace_res2(device const float* r [[buffer(0)]],
                         device float* partial [[buffer(1)]],
                         constant uint& nx [[buffer(2)]],
                         constant uint& ny [[buffer(3)]],
                         uint i [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint tid [[threadgroup_position_in_grid]],
                         uint tgSize [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float v = 0.0f;
    uint n = nx * ny;
    if (i < n) {
        uint ii = i % nx;
        uint jj = i / nx;
        if (!onBoundary(ii, jj, nx, ny)) {
            float ri = r[i];
            v = ri * ri;
        }
    }
    shared[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partial[tid] = shared[0];
}

/// Full-weighting restriction: fine residual → coarse (half resolution, 2^k+1 grids)
/// nxc = (nxf - 1)/2 + 1
kernel void restrict_fw(device const float* fine [[buffer(0)]],
                        device float* coarse [[buffer(1)]],
                        constant uint& nxf [[buffer(2)]],
                        constant uint& nyf [[buffer(3)]],
                        constant uint& nxc [[buffer(4)]],
                        constant uint& nyc [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint ic = gid.x;
    uint jc = gid.y;
    if (ic >= nxc || jc >= nyc) return;
    uint cidx = jc * nxc + ic;
    if (onBoundary(ic, jc, nxc, nyc)) {
        coarse[cidx] = 0.0f;
        return;
    }
    uint i = ic * 2;
    uint j = jc * 2;
    // full weighting stencil
    float v = 0.0f;
    v += 4.0f * fine[j * nxf + i];
    v += 2.0f * (fine[j * nxf + (i - 1)] + fine[j * nxf + (i + 1)] +
                 fine[(j - 1) * nxf + i] + fine[(j + 1) * nxf + i]);
    v += fine[(j - 1) * nxf + (i - 1)] + fine[(j - 1) * nxf + (i + 1)] +
         fine[(j + 1) * nxf + (i - 1)] + fine[(j + 1) * nxf + (i + 1)];
    coarse[cidx] = v * (1.0f / 16.0f);
}

/// Bilinear prolongation: coarse correction → fine, subtract into u (u -= P e)
/// because residual ρ = A u and we solve A e = R ρ, so u ← u − e.
kernel void prolong_bilinear(device const float* coarse [[buffer(0)]],
                             device float* fine [[buffer(1)]],
                             constant uint& nxc [[buffer(2)]],
                             constant uint& nyc [[buffer(3)]],
                             constant uint& nxf [[buffer(4)]],
                             constant uint& nyf [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nxf || j >= nyf) return;
    if (onBoundary(i, j, nxf, nyf)) return;

    uint ic = i / 2;
    uint jc = j / 2;
    float corr = 0.0f;

    if ((i % 2u) == 0u && (j % 2u) == 0u) {
        corr = coarse[jc * nxc + ic];
    } else if ((i % 2u) == 1u && (j % 2u) == 0u) {
        corr = 0.5f * (coarse[jc * nxc + ic] + coarse[jc * nxc + (ic + 1)]);
    } else if ((i % 2u) == 0u && (j % 2u) == 1u) {
        corr = 0.5f * (coarse[jc * nxc + ic] + coarse[(jc + 1) * nxc + ic]);
    } else {
        corr = 0.25f * (coarse[jc * nxc + ic] + coarse[jc * nxc + (ic + 1)] +
                        coarse[(jc + 1) * nxc + ic] + coarse[(jc + 1) * nxc + (ic + 1)]);
    }
    fine[j * nxf + i] -= corr;
}

/// Jacobi for stencil equation 4e − N − S − E − W = rhs
/// e_new = 0.25 * (N + S + E + W + rhs)
kernel void laplace_jacobi_rhs(device const float* e [[buffer(0)]],
                               device float* eNew [[buffer(1)]],
                               device const float* rhs [[buffer(2)]],
                               constant uint& nx [[buffer(3)]],
                               constant uint& ny [[buffer(4)]],
                               constant float& omega [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;
    uint idx = j * nx + i;
    if (onBoundary(i, j, nx, ny)) {
        eNew[idx] = 0.0f;
        return;
    }
    float avg = 0.25f * (e[idx - 1] + e[idx + 1] + e[idx - nx] + e[idx + nx] + rhs[idx]);
    eNew[idx] = (1.0f - omega) * e[idx] + omega * avg;
}

kernel void zero_buffer(device float* a [[buffer(0)]],
                        constant uint& n [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < n) a[i] = 0.0f;
}

/// y = A x for 5-point Laplacian stencil A = (4, -1, -1, -1, -1)
/// with homogeneous treatment of Dirichlet: boundary y = 0, and
/// off-boundary neighbour contributions from Dirichlet nodes omitted
/// when those neighbours are boundary (they are absorbed into RHS).
/// For pure Laplace with BCs baked into u, use apply to interior only.
kernel void laplace_matvec(device const float* x [[buffer(0)]],
                           device float* y [[buffer(1)]],
                           constant uint& nx [[buffer(2)]],
                           constant uint& ny [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;
    uint idx = j * nx + i;
    if (onBoundary(i, j, nx, ny)) {
        y[idx] = 0.0f;
        return;
    }
    y[idx] = 4.0f * x[idx] - x[idx - 1] - x[idx + 1] - x[idx - nx] - x[idx + nx];
}

/// SAXPY: y = a*x + y
kernel void laplace_saxpy(device const float* x [[buffer(0)]],
                          device float* y [[buffer(1)]],
                          constant float& a [[buffer(2)]],
                          constant uint& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = a * x[i] + y[i];
}

/// y = x + a*y
kernel void laplace_xpay(device const float* x [[buffer(0)]],
                         device float* y [[buffer(1)]],
                         constant float& a [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = x[i] + a * y[i];
}

/// y = x (copy)
kernel void laplace_copy(device const float* x [[buffer(0)]],
                         device float* y [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = x[i];
}

/// Partial dot product
kernel void laplace_dot_partial(device const float* x [[buffer(0)]],
                                device const float* y [[buffer(1)]],
                                device float* partial [[buffer(2)]],
                                constant uint& n [[buffer(3)]],
                                uint i [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tid [[threadgroup_position_in_grid]],
                                uint tgSize [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float v = 0.0f;
    if (i < n) v = x[i] * y[i];
    shared[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partial[tid] = shared[0];
}

/// Pointwise |u - uexact| 
kernel void abs_err(device const float* u [[buffer(0)]],
                    device const float* exact [[buffer(1)]],
                    device float* absdiff [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    absdiff[i] = fabs(u[i] - exact[i]);
}
