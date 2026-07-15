/**
 * @file MetalCompute.hpp
 * @brief MetalFlow core types: GpuBuffer and MetalCompute.
 *
 * @details
 * **MetalFlow** is a CUDA-style Metal layer for CFD/FEA on Apple Silicon.
 * This header is the primary public C++ surface. Implementation lives in
 * `src/metalflow/MetalCompute.mm` (Objective-C++ / Metal).
 *
 * Typical workflow:
 * @code
 * #include <metalflow/MetalCompute.hpp>
 * using namespace metalflow;
 *
 * MetalCompute gpu;
 * gpu.compileSource(mslSource);
 * GpuBuffer a = gpu.mallocShared(n * sizeof(float));
 * a.data<float>()[0] = 1.0f;
 * a.didModify();
 * gpu.setKernel("my_kernel");
 * gpu.setBuffer(a, 0);
 * gpu.launch(n);
 * gpu.synchronize();
 * @endcode
 *
 * @see metalflow::MetalCompute
 * @see metalflow::GpuBuffer
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @namespace metalflow
 * @brief CUDA-style Metal layer for CFD/FEA on Apple Silicon.
 *
 * Provides a thin runtime-like API (`MetalCompute`, `GpuBuffer`) that hides
 * Metal / Objective-C++ details while exposing buffer allocation, kernel
 * dispatch, and synchronization suitable for scientific computing.
 */
namespace metalflow {

/**
 * @class GpuBuffer
 * @brief Opaque GPU buffer handle with optional unified-memory host access.
 *
 * @details
 * Created only via MetalCompute::mallocShared() or MetalCompute::mallocDevice().
 * Non-copyable; movable. On Apple Silicon, shared buffers map into the same
 * physical pages seen by CPU and GPU (UMA).
 *
 * @warning After writing through contents() / data() on a shared buffer,
 *          call didModify() before the next GPU dispatch.
 *
 * @note Prefer mallocShared for CFD/FEA setup on Apple Silicon.
 */
class GpuBuffer {
public:
    GpuBuffer() = default;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    /** @brief Move-construct; transfers ownership of the Metal buffer. */
    GpuBuffer(GpuBuffer&& other) noexcept
        : metalBuffer_(other.metalBuffer_), size_(other.size_) {
        other.metalBuffer_ = nullptr;
        other.size_ = 0;
    }

    /** @brief Move-assign; transfers ownership of the Metal buffer. */
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            metalBuffer_ = other.metalBuffer_;
            size_ = other.size_;
            other.metalBuffer_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /** @return Allocated size in bytes (0 if empty). */
    size_t size() const { return size_; }

    /** @return True if no Metal buffer is owned. */
    bool empty() const { return size_ == 0 || metalBuffer_ == nullptr; }

    /**
     * @brief Host pointer into unified memory (shared buffers).
     * @return Pointer valid until free() / destruction; nullptr if empty.
     * @warning Do not read/write while GPU work using this buffer is in flight
     *          unless you have synchronized.
     */
    void* contents() const;

    /**
     * @brief Notify Metal that the CPU updated the shared buffer contents.
     * @details Required for correct coherency on Apple Silicon after host writes.
     */
    void didModify() const;

    /**
     * @brief Typed view of contents().
     * @tparam T Element type (e.g. float).
     */
    template <typename T>
    T* data() const {
        return static_cast<T*>(contents());
    }

    /**
     * @brief Number of elements of type T that fit in the buffer.
     * @tparam T Element type.
     */
    template <typename T>
    size_t count() const {
        return size_ / sizeof(T);
    }

private:
    friend class MetalCompute;
    void* metalBuffer_ = nullptr; ///< Opaque id&lt;MTLBuffer&gt;.
    size_t size_ = 0;             ///< Byte length.
};

/**
 * @class MetalCompute
 * @brief MetalFlow compute runtime — CUDA-style dispatch for Apple GPUs.
 *
 * @details
 * Hides Metal / Objective-C types behind a C++ API aimed at CFD and FEA.
 * One instance owns the default system GPU device and a command queue.
 *
 * Lifecycle of a dispatch:
 * @verbatim
 *   compileSource / loadLibrary
 *        → setKernel(name)
 *        → setBuffer / setValue  (binds [[buffer(i)]])
 *        → launch / launch2D
 *        → synchronize
 * @endverbatim
 *
 * @par Thread safety
 * Not thread-safe. Use one MetalCompute per host thread, or external locking.
 *
 * @par Error handling
 * Failures set lastError(); check after compile / setKernel / launch.
 */
class MetalCompute {
public:
    /**
     * @brief Select the default system Metal device (Apple GPU on M-series).
     * @post ok() is true on success.
     */
    MetalCompute();
    ~MetalCompute();

    MetalCompute(const MetalCompute&) = delete;
    MetalCompute& operator=(const MetalCompute&) = delete;
    MetalCompute(MetalCompute&&) noexcept;
    MetalCompute& operator=(MetalCompute&&) noexcept;

    /** @return True if a Metal device and queue were created. */
    bool ok() const { return device_ != nullptr; }

    /** @return Human-readable GPU name (e.g. "Apple M2"). */
    std::string deviceName() const;

    /** @return Device-recommended maximum working-set size in bytes. */
    uint64_t recommendedMaxWorkingSetSize() const;

    // --- Memory ---

    /**
     * @brief Allocate a shared (unified) buffer — preferred on Apple Silicon.
     * @param bytes Allocation size.
     * @return Movable GpuBuffer; empty on failure (see lastError()).
     */
    GpuBuffer mallocShared(size_t bytes);

    /**
     * @brief Allocate a private GPU-only buffer.
     * @param bytes Allocation size.
     * @note Use memcpyHtoD / memcpyDtoH for transfers.
     */
    GpuBuffer mallocDevice(size_t bytes);

    /** @brief Release a buffer and reset the handle. */
    void free(GpuBuffer& buf);

    /**
     * @brief Copy host → device (no-op path for shared storage).
     * @param dst Destination GPU buffer.
     * @param src Host pointer.
     * @param bytes Number of bytes (must be ≤ dst.size()).
     */
    void memcpyHtoD(GpuBuffer& dst, const void* src, size_t bytes);

    /**
     * @brief Copy device → host (no-op path for shared storage).
     */
    void memcpyDtoH(void* dst, const GpuBuffer& src, size_t bytes);

    /** @brief Fill the first @p bytes of @p buf with @p value (byte pattern). */
    void memset(GpuBuffer& buf, int value, size_t bytes);

    // --- Kernels ---

    /**
     * @brief Compile Metal Shading Language source at runtime (NVRTC-like).
     * @param metalSource Full MSL translation unit.
     * @param options Reserved for future compile flags.
     */
    void compileSource(const std::string& metalSource,
                       const std::string& options = "");

    /**
     * @brief Load a precompiled metallib from disk.
     * @param metallibPath Path to `.metallib` from `xcrun metal` / `metallib`.
     */
    void loadLibrary(const std::string& metallibPath);

    /**
     * @brief Select a kernel by name and build its compute pipeline.
     * @param name Exact MSL kernel function name (e.g. `"saxpy"`).
     */
    void setKernel(const std::string& name);

    /**
     * @brief Bind a buffer to `[[buffer(index)]]` for the next launch.
     * @param buf GPU buffer.
     * @param index Shader buffer index.
     */
    void setBuffer(const GpuBuffer& buf, uint32_t index);

    /**
     * @brief Bind small POD / constant bytes to `[[buffer(index)]]`.
     */
    void setBytes(const void* data, size_t size, uint32_t index);

    /**
     * @brief Bind a value by value (sizeof(T) bytes) to buffer index.
     * @tparam T Trivially copyable type (float, uint32_t, …).
     */
    template <typename T>
    void setValue(const T& value, uint32_t index) {
        setBytes(&value, sizeof(T), index);
    }

    /**
     * @brief 1-D dispatch: one thread per work item.
     * @param nThreads Total threads (grid width).
     * @param threadsPerGroup Threadgroup size; 0 = auto (capped at 256).
     */
    void launch(size_t nThreads, size_t threadsPerGroup = 0);

    /**
     * @brief 2-D dispatch for structured grids.
     * @param width Grid width in threads.
     * @param height Grid height in threads.
     * @param tgWidth Threadgroup width (default 16).
     * @param tgHeight Threadgroup height (default 16).
     */
    void launch2D(size_t width, size_t height,
                  size_t tgWidth = 16, size_t tgHeight = 16);

    /**
     * @brief Block until all committed GPU work on this queue completes.
     * @details Analogous to cudaDeviceSynchronize().
     */
    void synchronize();

    /** @return Last error string; empty if none. */
    const std::string& lastError() const { return lastError_; }

private:
    void release();
    void beginEncode();
    void endEncodeAndCommit();

    void* device_ = nullptr;    ///< id&lt;MTLDevice&gt;
    void* queue_ = nullptr;     ///< id&lt;MTLCommandQueue&gt;
    void* library_ = nullptr;   ///< id&lt;MTLLibrary&gt;
    void* pipeline_ = nullptr;  ///< id&lt;MTLComputePipelineState&gt;
    void* cmdBuffer_ = nullptr; ///< id&lt;MTLCommandBuffer&gt;
    void* encoder_ = nullptr;   ///< id&lt;MTLComputeCommandEncoder&gt;

    std::string lastError_;
};

} // namespace metalflow
