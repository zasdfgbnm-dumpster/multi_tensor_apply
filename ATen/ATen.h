#pragma once

#include <c10/macros/Macros.h>
#include <vector>
#include <functional>
#include <iostream>
#include <cuda_runtime.h>

struct Scalar {
    float value;
    Scalar(float v): value(v) {}
    template<typename T>
    T to() {
        return (T)value;
    }
};

struct Tensor {
    int n;
    float *data;
    int numel() {
        return n;
    }
    float *data_ptr() {
        return data;
    }
};

using TensorList = std::vector<Tensor>;

template<typename T>
using ArrayRef = std::vector<T>;

namespace at {

using Scalar = ::Scalar;

using Tensor = ::Tensor;

using TensorList = ::TensorList;

template<typename T>
using ArrayRef = ::ArrayRef<T>;

} // namespace at

#define TORCH_CHECK(...)
#define C10_CUDA_KERNEL_LAUNCH_CHECK(...)


namespace at { namespace native {

Tensor arange(int size) {
    using T = float;
    T *buf = new T[size];
    for (int64_t i = 0; i < size; i++) {
        buf[i] = T(i);
    }
    T *ret;
    int64_t size_ = size * sizeof(T);
    cudaMalloc(&ret, size_);
    cudaMemcpy(ret, buf, size_, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    delete [] buf;
    // who cares about cudaFree :P LOL
    return Tensor {size, ret};
}

Tensor zeros(int size) {
    using T = float;
    T *buf = new T[size];
    for (int64_t i = 0; i < size; i++) {
        buf[i] = 0;
    }
    T *ret;
    int64_t size_ = size * sizeof(T);
    cudaMalloc(&ret, size_);
    cudaMemcpy(ret, buf, size_, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    delete [] buf;
    // who cares about cudaFree :P LOL
    return Tensor {size, ret};
}

Tensor ones(int size) {
    using T = float;
    T *buf = new T[size];
    for (int64_t i = 0; i < size; i++) {
        buf[i] = 1;
    }
    T *ret;
    int64_t size_ = size * sizeof(T);
    cudaMalloc(&ret, size_);
    cudaMemcpy(ret, buf, size_, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    delete [] buf;
    // who cares about cudaFree :P LOL
    return Tensor {size, ret};
}

Tensor empty_like(Tensor t) {
    using T = float;
    int size = t.n;
    T *ret;
    int64_t size_ = size * sizeof(T);
    cudaMalloc(&ret, size_);
    cudaDeviceSynchronize();
    // who cares about cudaFree :P LOL
    return Tensor {size, ret};
}

}}

std::ostream &operator<<(std::ostream &os, Tensor t) {
    using T = float;
    int size = t.n;
    T *data = t.data;
    T *buf = new T[size];
    int64_t size_ = size * sizeof(T);
    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaMemcpy(buf, data, size_, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int64_t i = 0; i < size; i++) {
        os << buf[i] << ", ";
    }
    os << std::endl;
    delete [] buf;
    return os;
}