/**
 * @file MetalCompute.mm
 * @brief MetalFlow Objective-C++ / Metal implementation of MetalCompute and GpuBuffer.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <metalflow/MetalCompute.hpp>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

inline id<MTLDevice> asDevice(void* p) {
    return (__bridge id<MTLDevice>)p;
}
inline id<MTLCommandQueue> asQueue(void* p) {
    return (__bridge id<MTLCommandQueue>)p;
}
inline id<MTLLibrary> asLibrary(void* p) {
    return (__bridge id<MTLLibrary>)p;
}
inline id<MTLComputePipelineState> asPipeline(void* p) {
    return (__bridge id<MTLComputePipelineState>)p;
}
inline id<MTLCommandBuffer> asCmd(void* p) {
    return (__bridge id<MTLCommandBuffer>)p;
}
inline id<MTLComputeCommandEncoder> asEncoder(void* p) {
    return (__bridge id<MTLComputeCommandEncoder>)p;
}
inline id<MTLBuffer> asBuffer(void* p) {
    return (__bridge id<MTLBuffer>)p;
}

inline void* retainObj(id obj) {
    return (__bridge_retained void*)obj;
}
inline void releaseObj(void*& p) {
    if (p) {
        (void)(__bridge_transfer id)p;
        p = nullptr;
    }
}

} // namespace

namespace metalflow {

void* GpuBuffer::contents() const {
    if (!metalBuffer_) return nullptr;
    return [asBuffer(metalBuffer_) contents];
}

void GpuBuffer::didModify() const {
    if (!metalBuffer_) return;
    id<MTLBuffer> buf = asBuffer(metalBuffer_);
    if (buf.storageMode == MTLStorageModeShared)
        [buf didModifyRange:NSMakeRange(0, buf.length)];
}

MetalCompute::MetalCompute() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            lastError_ = "No Metal GPU device found";
            return;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            lastError_ = "Failed to create Metal command queue";
            return;
        }
        device_ = retainObj(device);
        queue_ = retainObj(queue);
    }
}

MetalCompute::~MetalCompute() {
    release();
}

MetalCompute::MetalCompute(MetalCompute&& other) noexcept {
    *this = std::move(other);
}

MetalCompute& MetalCompute::operator=(MetalCompute&& other) noexcept {
    if (this == &other) return *this;
    release();
    device_ = other.device_;
    queue_ = other.queue_;
    library_ = other.library_;
    pipeline_ = other.pipeline_;
    cmdBuffer_ = other.cmdBuffer_;
    encoder_ = other.encoder_;
    lastError_ = std::move(other.lastError_);
    other.device_ = other.queue_ = other.library_ = other.pipeline_ =
        other.cmdBuffer_ = other.encoder_ = nullptr;
    return *this;
}

void MetalCompute::release() {
    @autoreleasepool {
        if (encoder_) {
            [asEncoder(encoder_) endEncoding];
            releaseObj(encoder_);
        }
        releaseObj(cmdBuffer_);
        releaseObj(pipeline_);
        releaseObj(library_);
        releaseObj(queue_);
        releaseObj(device_);
    }
}

std::string MetalCompute::deviceName() const {
    if (!device_) return {};
    @autoreleasepool {
        return std::string([[asDevice(device_) name] UTF8String]);
    }
}

uint64_t MetalCompute::recommendedMaxWorkingSetSize() const {
    if (!device_) return 0;
    return [asDevice(device_) recommendedMaxWorkingSetSize];
}

GpuBuffer MetalCompute::mallocShared(size_t bytes) {
    GpuBuffer buf;
    if (!device_ || bytes == 0) return buf;
    @autoreleasepool {
        id<MTLBuffer> mtl = [asDevice(device_) newBufferWithLength:bytes
                                                          options:MTLResourceStorageModeShared];
        if (!mtl) {
            lastError_ = "mallocShared failed";
            return buf;
        }
        buf.metalBuffer_ = retainObj(mtl);
        buf.size_ = bytes;
    }
    return buf;
}

GpuBuffer MetalCompute::mallocDevice(size_t bytes) {
    GpuBuffer buf;
    if (!device_ || bytes == 0) return buf;
    @autoreleasepool {
        // Private storage: GPU-only; use blit/memcpy for transfers.
        id<MTLBuffer> mtl = [asDevice(device_) newBufferWithLength:bytes
                                                          options:MTLResourceStorageModePrivate];
        if (!mtl) {
            lastError_ = "mallocDevice failed";
            return buf;
        }
        buf.metalBuffer_ = retainObj(mtl);
        buf.size_ = bytes;
    }
    return buf;
}

void MetalCompute::free(GpuBuffer& buf) {
    @autoreleasepool {
        releaseObj(buf.metalBuffer_);
        buf.size_ = 0;
    }
}

void MetalCompute::memcpyHtoD(GpuBuffer& dst, const void* src, size_t bytes) {
    if (!dst.metalBuffer_ || !src || bytes == 0) return;
    if (bytes > dst.size_) {
        lastError_ = "memcpyHtoD: size exceeds buffer";
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> mtl = asBuffer(dst.metalBuffer_);
        if (mtl.storageMode == MTLStorageModeShared) {
            std::memcpy([mtl contents], src, bytes);
            return;
        }
        // Staging via temporary shared buffer + blit
        id<MTLBuffer> staging =
            [asDevice(device_) newBufferWithBytes:src
                                          length:bytes
                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [asQueue(queue_) commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:staging
                sourceOffset:0
                    toBuffer:mtl
           destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void MetalCompute::memcpyDtoH(void* dst, const GpuBuffer& src, size_t bytes) {
    if (!src.metalBuffer_ || !dst || bytes == 0) return;
    if (bytes > src.size_) {
        lastError_ = "memcpyDtoH: size exceeds buffer";
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> mtl = asBuffer(src.metalBuffer_);
        if (mtl.storageMode == MTLStorageModeShared) {
            std::memcpy(dst, [mtl contents], bytes);
            return;
        }
        id<MTLBuffer> staging =
            [asDevice(device_) newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [asQueue(queue_) commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:mtl
                sourceOffset:0
                    toBuffer:staging
           destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        std::memcpy(dst, [staging contents], bytes);
    }
}

void MetalCompute::memset(GpuBuffer& buf, int value, size_t bytes) {
    if (!buf.metalBuffer_ || bytes == 0) return;
    bytes = std::min(bytes, buf.size_);
    @autoreleasepool {
        id<MTLBuffer> mtl = asBuffer(buf.metalBuffer_);
        if (mtl.storageMode == MTLStorageModeShared) {
            std::memset([mtl contents], value, bytes);
            return;
        }
        // Fill private buffer via staging
        std::vector<uint8_t> tmp(bytes, static_cast<uint8_t>(value));
        memcpyHtoD(buf, tmp.data(), bytes);
    }
}

void MetalCompute::compileSource(const std::string& metalSource,
                                 const std::string& options) {
    if (!device_) return;
    @autoreleasepool {
        releaseObj(pipeline_);
        releaseObj(library_);

        NSError* error = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (!options.empty()) {
            // Keep simple: language version only; extend as needed.
            (void)options;
        }
        NSString* src = [NSString stringWithUTF8String:metalSource.c_str()];
        id<MTLLibrary> lib =
            [asDevice(device_) newLibraryWithSource:src options:opts error:&error];
        if (!lib) {
            lastError_ = error ? [[error localizedDescription] UTF8String]
                               : "compileSource failed";
            return;
        }
        library_ = retainObj(lib);
        lastError_.clear();
    }
}

void MetalCompute::loadLibrary(const std::string& metallibPath) {
    if (!device_) return;
    @autoreleasepool {
        releaseObj(pipeline_);
        releaseObj(library_);

        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallibPath.c_str()]];
        id<MTLLibrary> lib =
            [asDevice(device_) newLibraryWithURL:url error:&error];
        if (!lib) {
            lastError_ = error ? [[error localizedDescription] UTF8String]
                               : "loadLibrary failed";
            return;
        }
        library_ = retainObj(lib);
        lastError_.clear();
    }
}

void MetalCompute::setKernel(const std::string& name) {
    if (!device_ || !library_) {
        lastError_ = "setKernel: no library loaded";
        return;
    }
    @autoreleasepool {
        // Finish any open encode before switching pipelines.
        if (encoder_) {
            [asEncoder(encoder_) endEncoding];
            releaseObj(encoder_);
            if (cmdBuffer_) [asCmd(cmdBuffer_) commit];
        }
        releaseObj(pipeline_);
        NSString* fname = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> fn = [asLibrary(library_) newFunctionWithName:fname];
        if (!fn) {
            lastError_ = "Kernel not found: " + name;
            return;
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pso =
            [asDevice(device_) newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            lastError_ = error ? [[error localizedDescription] UTF8String]
                               : "Failed to create pipeline";
            return;
        }
        pipeline_ = retainObj(pso);
        lastError_.clear();
    }
}

void MetalCompute::beginEncode() {
    if (encoder_) return;
    @autoreleasepool {
        // Drop retain on a previously committed buffer. The queue is in-order, so
        // waiting on the latest command buffer in synchronize() covers earlier work.
        releaseObj(cmdBuffer_);
        id<MTLCommandBuffer> cmd = [asQueue(queue_) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:asPipeline(pipeline_)];
        cmdBuffer_ = retainObj(cmd);
        encoder_ = retainObj(enc);
    }
}

void MetalCompute::endEncodeAndCommit() {
    @autoreleasepool {
        if (encoder_) {
            [asEncoder(encoder_) endEncoding];
            releaseObj(encoder_);
        }
        if (cmdBuffer_) {
            [asCmd(cmdBuffer_) commit];
            // Keep cmdBuffer_ until synchronize() so waitUntilCompleted works.
        }
    }
}

void MetalCompute::setBuffer(const GpuBuffer& buf, uint32_t index) {
    if (!pipeline_ || !buf.metalBuffer_) {
        lastError_ = "setBuffer: missing pipeline or buffer";
        return;
    }
    @autoreleasepool {
        beginEncode();
        [asEncoder(encoder_) setBuffer:asBuffer(buf.metalBuffer_)
                                offset:0
                               atIndex:index];
    }
}

void MetalCompute::setBytes(const void* data, size_t size, uint32_t index) {
    if (!pipeline_ || !data || size == 0) return;
    @autoreleasepool {
        beginEncode();
        [asEncoder(encoder_) setBytes:data length:size atIndex:index];
    }
}

void MetalCompute::launch(size_t nThreads, size_t threadsPerGroup) {
    if (!pipeline_ || nThreads == 0) {
        lastError_ = "launch: no pipeline or zero threads";
        return;
    }
    @autoreleasepool {
        beginEncode();
        id<MTLComputePipelineState> pso = asPipeline(pipeline_);
        NSUInteger tg = threadsPerGroup;
        if (tg == 0) {
            tg = pso.maxTotalThreadsPerThreadgroup;
            if (tg > 256) tg = 256;
        }
        if (tg > pso.maxTotalThreadsPerThreadgroup)
            tg = pso.maxTotalThreadsPerThreadgroup;

        MTLSize grid = MTLSizeMake(nThreads, 1, 1);
        MTLSize group = MTLSizeMake(tg, 1, 1);
        [asEncoder(encoder_) dispatchThreads:grid threadsPerThreadgroup:group];
        endEncodeAndCommit();
    }
}

void MetalCompute::launch2D(size_t width, size_t height,
                            size_t tgWidth, size_t tgHeight) {
    if (!pipeline_ || width == 0 || height == 0) {
        lastError_ = "launch2D: invalid size";
        return;
    }
    @autoreleasepool {
        beginEncode();
        id<MTLComputePipelineState> pso = asPipeline(pipeline_);
        NSUInteger maxTg = pso.maxTotalThreadsPerThreadgroup;
        if (tgWidth * tgHeight > maxTg) {
            tgWidth = 16;
            tgHeight = maxTg / tgWidth;
            if (tgHeight == 0) tgHeight = 1;
        }
        MTLSize grid = MTLSizeMake(width, height, 1);
        MTLSize group = MTLSizeMake(tgWidth, tgHeight, 1);
        [asEncoder(encoder_) dispatchThreads:grid threadsPerThreadgroup:group];
        endEncodeAndCommit();
    }
}

void MetalCompute::synchronize() {
    @autoreleasepool {
        if (encoder_) {
            [asEncoder(encoder_) endEncoding];
            releaseObj(encoder_);
        }
        if (cmdBuffer_) {
            [asCmd(cmdBuffer_) waitUntilCompleted];
            releaseObj(cmdBuffer_);
        }
    }
}

} // namespace metalflow
