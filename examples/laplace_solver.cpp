/**
 * @file laplace_solver.cpp
 * @brief MetalFlow 2-D Laplace solver (∇²u=0) with Dirichlet BCs via GPU CG.
 * @see laplace_solver_doc
 */
#include <metalflow/metalflow.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace metalflow;

namespace {

constexpr float kPi = 3.14159265358979323846f;

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void check(MetalCompute& gpu, const char* step) {
    if (!gpu.lastError().empty()) {
        std::cerr << step << ": " << gpu.lastError() << "\n";
        std::exit(1);
    }
}

inline float exact_xx_yy(float x, float y) { return x * x - y * y; }

inline float exact_sinh(float x, float y) {
    return std::sinh(kPi * x) * std::sin(kPi * y) / std::sinh(kPi);
}

float reduceSum(const float* p, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += p[i];
    return static_cast<float>(s);
}

float maxAbsError(const float* u, const float* exact, uint32_t n) {
    float m = 0.f;
    for (uint32_t i = 0; i < n; ++i)
        m = std::max(m, std::abs(u[i] - exact[i]));
    return m;
}

struct LaplaceCG {
    MetalCompute& gpu;
    uint32_t nx, ny, n;
    float h, inv_h;
    GpuBuffer u, exact, r, p, Ap, partial;
    uint32_t nPartial;

    LaplaceCG(MetalCompute& g, uint32_t nx_, uint32_t ny_)
        : gpu(g), nx(nx_), ny(ny_), n(nx_ * ny_) {
        h = 1.0f / float(nx - 1);
        inv_h = 1.0f / h;
        const size_t bytes = size_t(n) * sizeof(float);
        u = gpu.mallocShared(bytes);
        exact = gpu.mallocShared(bytes);
        r = gpu.mallocShared(bytes);
        p = gpu.mallocShared(bytes);
        Ap = gpu.mallocShared(bytes);
        nPartial = (n + 255) / 256;
        partial = gpu.mallocShared(nPartial * sizeof(float));
        std::memset(u.contents(), 0, bytes);
        std::memset(r.contents(), 0, bytes);
        std::memset(p.contents(), 0, bytes);
        std::memset(Ap.contents(), 0, bytes);
        u.didModify();
        r.didModify();
        p.didModify();
        Ap.didModify();
    }

    void matvec(const GpuBuffer& x, GpuBuffer& y) {
        gpu.setKernel("laplace_matvec");
        gpu.setBuffer(x, 0);
        gpu.setBuffer(y, 1);
        gpu.setValue(nx, 2);
        gpu.setValue(ny, 3);
        gpu.launch2D(nx, ny);
    }

    float dot(const GpuBuffer& a, const GpuBuffer& b) {
        gpu.setKernel("laplace_dot_partial");
        gpu.setBuffer(a, 0);
        gpu.setBuffer(b, 1);
        gpu.setBuffer(partial, 2);
        gpu.setValue(n, 3);
        gpu.launch(n, 256);
        gpu.synchronize();
        return reduceSum(partial.data<float>(), nPartial);
    }

    void saxpy(float a, const GpuBuffer& x, GpuBuffer& y) {
        gpu.setKernel("laplace_saxpy");
        gpu.setBuffer(x, 0);
        gpu.setBuffer(y, 1);
        gpu.setValue(a, 2);
        gpu.setValue(n, 3);
        gpu.launch(n);
    }

    void xpay(const GpuBuffer& x, float a, GpuBuffer& y) {
        gpu.setKernel("laplace_xpay");
        gpu.setBuffer(x, 0);
        gpu.setBuffer(y, 1);
        gpu.setValue(a, 2);
        gpu.setValue(n, 3);
        gpu.launch(n);
    }

    void copy(const GpuBuffer& x, GpuBuffer& y) {
        gpu.setKernel("laplace_copy");
        gpu.setBuffer(x, 0);
        gpu.setBuffer(y, 1);
        gpu.setValue(n, 2);
        gpu.launch(n);
    }

    /// Physical L2 residual ≈ ||∇²u||_L2 from stencil ρ = A u
    float residualL2() {
        matvec(u, r);
        // ||ρ/h²||_L2 = sqrt(h² * sum (ρ/h²)²) = sqrt(sum ρ²) / h
        gpu.setKernel("laplace_dot_partial");
        gpu.setBuffer(r, 0);
        gpu.setBuffer(r, 1);
        gpu.setBuffer(partial, 2);
        gpu.setValue(n, 3);
        gpu.launch(n, 256);
        gpu.synchronize();
        return std::sqrt(reduceSum(partial.data<float>(), nPartial)) * inv_h;
    }

    /// Solve ∇²u = 0 with Dirichlet BCs already stored in u (boundary).
    /// CG on A e = −A u0 with homogeneous Dirichlet for e; u ← u + e.
    int solve(float tol, int maxIter, bool verbose) {
        // b = −A u,  r0 = b (since e0 = 0), p0 = r0
        matvec(u, Ap);          // Ap = A u
        copy(Ap, r);
        saxpy(-2.0f, Ap, r);    // r = Ap - 2 Ap = −A u
        copy(r, p);
        gpu.synchronize();

        float rr = dot(r, r);
        const float rr0 = rr;
        if (verbose)
            std::printf("iter %5d  ||r||_2 = %.6e  (stencil)\n", 0, std::sqrt(rr));

        int it = 0;
        for (it = 1; it <= maxIter; ++it) {
            matvec(p, Ap);
            const float pAp = dot(p, Ap);
            if (!(pAp > 0.f) || !std::isfinite(pAp)) {
                std::cerr << "CG breakdown at iter " << it << "\n";
                return it;
            }
            const float alpha = rr / pAp;

            saxpy(alpha, p, u);    // u += α p  (boundary of p is 0)
            saxpy(-alpha, Ap, r);  // r -= α Ap

            const float rrNew = dot(r, r);
            const int printEvery = (n > 5'000'000) ? 200 : 50;
            if (verbose && (it % printEvery == 0 || it == 1)) {
                const float err = maxAbsError(u.data<float>(), exact.data<float>(), n);
                std::printf("iter %5d  ||r||_2 = %.6e  max|u−uexact| = %.6e\n",
                            it, std::sqrt(rrNew), err);
            }

            if (rrNew < tol * tol * rr0 || rrNew < tol * tol)
                return it;

            const float beta = rrNew / rr;
            xpay(r, beta, p); // p = r + β p
            rr = rrNew;
        }
        return it;
    }
};

} // namespace

/*
 * Full 2-D Laplace solver on Apple Silicon GPU (Metal):
 *   ∇²u = 0 on (0,1)²
 *   u = g on ∂Ω   (Dirichlet from an exact harmonic solution)
 *
 * Method: conjugate gradient on the 5-point stencil with homogeneous
 * corrections (boundary values held fixed in the solution vector).
 *
 * Usage:
 *   ./laplace_solver [shader.metal] [p] [tol] [exact_id]
 *     p        : 8→257², 9→513², 10→1025², 11→2049²  (default 10)
 *     tol      : relative CG residual                   (default 1e-8)
 *     exact_id : 0 = x²−y², 1 = sinh(πx)sin(πy)/sinh(π)
 */
int main(int argc, char** argv) {
    const char* shaderPath = (argc > 1) ? argv[1] : "shaders/laplace.metal";
    int p = (argc > 2) ? std::atoi(argv[2]) : 10;
    float tol = (argc > 3) ? static_cast<float>(std::atof(argv[3])) : 1e-8f;
    int whichExact = (argc > 4) ? std::atoi(argv[4]) : 0;
    p = std::clamp(p, 5, 14); // 14 → 16385²; ~5.4 GB buffers (needs ≥16 GB Mac)

    MetalCompute gpu;
    if (!gpu.ok()) {
        std::cerr << "Metal init failed: " << gpu.lastError() << "\n";
        return 1;
    }
    std::cout << "Device: " << gpu.deviceName() << "\n";

    gpu.compileSource(readFile(shaderPath));
    check(gpu, "compileSource");

    const uint32_t N = 1u << p;
    const uint32_t nx = N + 1;
    const uint32_t ny = N + 1;

    LaplaceCG sol(gpu, nx, ny);

    float* uh = sol.u.data<float>();
    float* ex = sol.exact.data<float>();
    for (uint32_t j = 0; j < ny; ++j) {
        for (uint32_t i = 0; i < nx; ++i) {
            const float x = i * sol.h;
            const float y = j * sol.h;
            const float ue = (whichExact == 0) ? exact_xx_yy(x, y) : exact_sinh(x, y);
            ex[j * nx + i] = ue;
            const bool bnd = (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny);
            uh[j * nx + i] = bnd ? ue : 0.0f;
        }
    }
    sol.u.didModify();
    sol.exact.didModify();

    const int maxIter = static_cast<int>(4 * nx); // ample for 5-point CG

    std::cout << "\n=== MetalFlow  2-D Laplace  ∇²u = 0  with Dirichlet BCs ===\n";
    std::cout << "CUDA-style Metal layer for CFD/FEA\n";
    std::cout << "Solver: Conjugate Gradient (Metal GPU)\n";
    std::cout << "Exact : " << (whichExact == 0 ? "u = x² − y²" : "u = sinh(πx)sin(πy)/sinh(π)")
              << "\n";
    std::cout << "Grid  : " << nx << " × " << ny << "   h = " << sol.h
              << "   (" << 1e-6 * double(sol.n) << " M nodes)\n";
    std::cout << "Tol   : " << tol << "\n\n";

    const float res0 = sol.residualL2();
    std::printf("Initial residual_L2 (∇²) = %.6e\n", res0);
    std::printf("Initial max|u−uexact|    = %.6e\n\n",
                maxAbsError(uh, ex, sol.n));

    auto t0 = std::chrono::steady_clock::now();
    const int iters = sol.solve(tol, maxIter, true);
    gpu.synchronize();
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const float res = sol.residualL2();
    const float err = maxAbsError(sol.u.data<float>(), sol.exact.data<float>(), sol.n);

    float bndErr = 0.f;
    const float* ufin = sol.u.data<float>();
    for (uint32_t i = 0; i < nx; ++i) {
        bndErr = std::max(bndErr, std::abs(ufin[i] - ex[i]));
        bndErr = std::max(bndErr, std::abs(ufin[(ny - 1) * nx + i] - ex[(ny - 1) * nx + i]));
    }
    for (uint32_t j = 0; j < ny; ++j) {
        bndErr = std::max(bndErr, std::abs(ufin[j * nx] - ex[j * nx]));
        bndErr = std::max(bndErr, std::abs(ufin[j * nx + nx - 1] - ex[j * nx + nx - 1]));
    }

    const uint32_t ic = nx / 2, jc = ny / 2;
    const float uc = ufin[jc * nx + ic];
    const float uec = ex[jc * nx + ic];

    std::printf("\n--- Summary ---\n");
    std::printf("Grid             : %u × %u  (%.3f M nodes)\n", nx, ny, 1e-6 * double(sol.n));
    std::printf("CG iterations    : %d\n", iters);
    std::printf("Final residual   : %.6e\n", res);
    std::printf("Max abs error    : %.6e\n", err);
    std::printf("Boundary max err : %.6e  (Dirichlet)\n", bndErr);
    std::printf("Center u / exact : %.10f / %.10f\n", uc, uec);
    std::printf("Wall time        : %.2f ms\n", ms);
    std::printf("Status           : %s\n", (err < 1e-3f || iters < maxIter) ? "CONVERGED" : "MAX ITER");
    return (iters <= maxIter && err < 1e-2f) ? 0 : 1;
}
