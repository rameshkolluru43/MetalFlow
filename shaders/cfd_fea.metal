/**
 * @file cfd_fea.metal
 * @brief MetalFlow CFD / FEA oriented Metal compute kernels (float32).
 *
 * Used with metalflow::MetalCompute. Kernels: saxpy, axpby, hadamard,
 * copy_f32, fill_f32, dot_partial, csr_matvec, jacobi2d, diffuse2d.
 */

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// BLAS-like primitives used everywhere in CFD/FEA
// -----------------------------------------------------------------------------

/// y = a*x + y   (SAXPY)
kernel void saxpy(device const float* x [[buffer(0)]],
                  device float* y [[buffer(1)]],
                  constant float& a [[buffer(2)]],
                  constant uint& n [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = a * x[i] + y[i];
}

/// y = a*x + b*y
kernel void axpby(device const float* x [[buffer(0)]],
                  device float* y [[buffer(1)]],
                  constant float& a [[buffer(2)]],
                  constant float& b [[buffer(3)]],
                  constant uint& n [[buffer(4)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = a * x[i] + b * y[i];
}

/// z = x * y  (Hadamard / element-wise product)
kernel void hadamard(device const float* x [[buffer(0)]],
                     device const float* y [[buffer(1)]],
                     device float* z [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    z[i] = x[i] * y[i];
}

/// y = x  (copy)
kernel void copy_f32(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = x[i];
}

/// y = c  (fill)
kernel void fill_f32(device float* y [[buffer(0)]],
                     constant float& c [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = c;
}

/// Partial dot product: out[tg] = sum of threadgroup chunk of x·y
/// Launch with n threads; then reduce out[] on CPU or a follow-up kernel.
kernel void dot_partial(device const float* x [[buffer(0)]],
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

// -----------------------------------------------------------------------------
// Sparse / FEA: CSR matrix-vector product  y = A * x
// rowPtr length = nrows+1, colInd / values length = nnz
// -----------------------------------------------------------------------------

kernel void csr_matvec(device const float* values [[buffer(0)]],
                       device const uint* colInd [[buffer(1)]],
                       device const uint* rowPtr [[buffer(2)]],
                       device const float* x [[buffer(3)]],
                       device float* y [[buffer(4)]],
                       constant uint& nrows [[buffer(5)]],
                       uint row [[thread_position_in_grid]]) {
    if (row >= nrows) return;
    float sum = 0.0f;
    uint start = rowPtr[row];
    uint end = rowPtr[row + 1];
    for (uint j = start; j < end; ++j)
        sum += values[j] * x[colInd[j]];
    y[row] = sum;
}

// -----------------------------------------------------------------------------
// CFD: 2-D Jacobi / Laplace smoother on a structured grid
// u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1))
// Interior only; boundaries unchanged.
// -----------------------------------------------------------------------------

kernel void jacobi2d(device const float* u [[buffer(0)]],
                     device float* uNew [[buffer(1)]],
                     constant uint& nx [[buffer(2)]],
                     constant uint& ny [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;

    uint idx = j * nx + i;
    if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
        uNew[idx] = u[idx];
        return;
    }
    float left = u[idx - 1];
    float right = u[idx + 1];
    float down = u[idx - nx];
    float up = u[idx + nx];
    uNew[idx] = 0.25f * (left + right + down + up);
}

/// Explicit diffusion step: u_new = u + nu*dt * Laplacian(u)  (5-point)
kernel void diffuse2d(device const float* u [[buffer(0)]],
                      device float* uNew [[buffer(1)]],
                      constant uint& nx [[buffer(2)]],
                      constant uint& ny [[buffer(3)]],
                      constant float& alpha [[buffer(4)]], // nu*dt / h^2
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= nx || j >= ny) return;

    uint idx = j * nx + i;
    if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
        uNew[idx] = u[idx];
        return;
    }
    float c = u[idx];
    float lap = u[idx - 1] + u[idx + 1] + u[idx - nx] + u[idx + nx] - 4.0f * c;
    uNew[idx] = c + alpha * lap;
}
