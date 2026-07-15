/**
 * @file demo.cpp
 * @brief MetalFlow demo: SAXPY, CSR SpMV, Jacobi CFD step.
 */
#include <metalflow/metalflow.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace metalflow;

static std::string readFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void check(MetalCompute& gpu, const char* step) {
    if (!gpu.lastError().empty()) {
        std::cerr << step << ": " << gpu.lastError() << "\n";
        std::exit(1);
    }
}

/// CUDA-style usage demo: SAXPY + CSR matvec + Jacobi CFD step.
int main(int argc, char** argv) {
    const char* shaderPath = (argc > 1) ? argv[1] : "shaders/cfd_fea.metal";

    MetalCompute gpu;
    if (!gpu.ok()) {
        std::cerr << "Metal init failed: " << gpu.lastError() << "\n";
        return 1;
    }
    std::cout << "Device: " << gpu.deviceName() << "\n";
    std::cout << "Recommended working set: "
              << (gpu.recommendedMaxWorkingSetSize() / (1024.0 * 1024.0 * 1024.0))
              << " GB\n";

    gpu.compileSource(readFile(shaderPath));
    check(gpu, "compileSource");

    // -------------------------------------------------------------------------
    // 1) SAXPY: y = a*x + y   (same pattern as CUDA runtime)
    // -------------------------------------------------------------------------
    {
        const uint32_t n = 1 << 20; // 1M
        const float a = 2.0f;

        GpuBuffer d_x = gpu.mallocShared(n * sizeof(float));
        GpuBuffer d_y = gpu.mallocShared(n * sizeof(float));

        float* x = d_x.data<float>();
        float* y = d_y.data<float>();
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = 1.0f;
            y[i] = 1.0f;
        }

        gpu.setKernel("saxpy");
        check(gpu, "setKernel(saxpy)");
        gpu.setBuffer(d_x, 0);
        gpu.setBuffer(d_y, 1);
        gpu.setValue(a, 2);
        gpu.setValue(n, 3);
        gpu.launch(n);
        gpu.synchronize();

        double err = 0.0;
        for (uint32_t i = 0; i < n; i += n / 16)
            err += std::abs(y[i] - 3.0f);
        std::printf("SAXPY  n=%u  sample_err=%.3e\n", n, err);

        gpu.free(d_x);
        gpu.free(d_y);
    }

    // -------------------------------------------------------------------------
    // 2) FEA: CSR SpMV  y = A * x   (3x3 diagonal-ish demo matrix)
    // -------------------------------------------------------------------------
    {
        // A = [[2,1,0],[1,2,1],[0,1,2]]
        const uint32_t nrows = 3;
        const float valuesArr[] = {2, 1, 1, 2, 1, 1, 2};
        const uint32_t colIndArr[] = {0, 1, 0, 1, 2, 1, 2};
        const uint32_t rowPtrArr[] = {0, 2, 5, 7};
        const float xHost[] = {1, 2, 3};
        // Expected y = (4, 8, 8)

        GpuBuffer values = gpu.mallocShared(sizeof(valuesArr));
        GpuBuffer colInd = gpu.mallocShared(sizeof(colIndArr));
        GpuBuffer rowPtr = gpu.mallocShared(sizeof(rowPtrArr));
        GpuBuffer x = gpu.mallocShared(nrows * sizeof(float));
        GpuBuffer y = gpu.mallocShared(nrows * sizeof(float));

        std::memcpy(values.contents(), valuesArr, sizeof(valuesArr));
        std::memcpy(colInd.contents(), colIndArr, sizeof(colIndArr));
        std::memcpy(rowPtr.contents(), rowPtrArr, sizeof(rowPtrArr));
        std::memcpy(x.contents(), xHost, sizeof(xHost));

        gpu.setKernel("csr_matvec");
        check(gpu, "setKernel(csr_matvec)");
        gpu.setBuffer(values, 0);
        gpu.setBuffer(colInd, 1);
        gpu.setBuffer(rowPtr, 2);
        gpu.setBuffer(x, 3);
        gpu.setBuffer(y, 4);
        gpu.setValue(nrows, 5);
        gpu.launch(nrows);
        gpu.synchronize();

        float* yh = y.data<float>();
        std::printf("CSR SpMV y = [%.1f, %.1f, %.1f] (expect 4, 8, 8)\n",
                    yh[0], yh[1], yh[2]);

        gpu.free(values);
        gpu.free(colInd);
        gpu.free(rowPtr);
        gpu.free(x);
        gpu.free(y);
    }

    // -------------------------------------------------------------------------
    // 3) CFD: Jacobi sweep on a small 2-D grid
    // -------------------------------------------------------------------------
    {
        const uint32_t nx = 32, ny = 32;
        const size_t n = size_t(nx) * ny;

        GpuBuffer u = gpu.mallocShared(n * sizeof(float));
        GpuBuffer uNew = gpu.mallocShared(n * sizeof(float));
        float* uh = u.data<float>();
        for (uint32_t j = 0; j < ny; ++j)
            for (uint32_t i = 0; i < nx; ++i)
                uh[j * nx + i] = (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny)
                                     ? 1.0f
                                     : 0.0f;

        gpu.setKernel("jacobi2d");
        check(gpu, "setKernel(jacobi2d)");

        for (int iter = 0; iter < 500; ++iter) {
            gpu.setBuffer(u, 0);
            gpu.setBuffer(uNew, 1);
            gpu.setValue(nx, 2);
            gpu.setValue(ny, 3);
            gpu.launch2D(nx, ny);
            gpu.synchronize();
            std::swap(u, uNew);
        }

        float center = u.data<float>()[(ny / 2) * nx + (nx / 2)];
        std::printf("Jacobi2D %ux%u center=%.4f (boundary=1)\n", nx, ny, center);

        gpu.free(u);
        gpu.free(uNew);
    }

    std::cout << "OK\n";
    return 0;
}
