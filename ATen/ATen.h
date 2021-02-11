#pragma once

#include <vector>

class Tensor {

};

using TensorList = std::vector<Tensor>;

template<typename T>
using ArrayRef = std::vector<T>;

namespace at {

using Tensor = ::Tensor;

using TensorList = ::TensorList;

template<typename T>
using ArrayRef = ::ArrayRef<T>;

} // namespace at

#define TORCH_CHECK(...)
