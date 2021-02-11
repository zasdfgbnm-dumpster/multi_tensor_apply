#pragma once

#include <c10/macros/Macros.h>
#include <vector>
#include <functional>

struct Scalar {
    float value;
    Scalar(float v): value(v) {}
    template<typename T>
    T to() {
        return (T)value;
    }
};

struct Tensor {

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
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
