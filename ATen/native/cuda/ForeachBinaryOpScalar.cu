#include <ATen/native/cuda/ForeachFunctors.cuh>

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    using scalar_t = float;

    using opmath_t = get_opmath_t<scalar_t>::opmath_t;
    multi_tensor_apply<2>(tensor_lists,
                          BinaryOpScalarFunctor<scalar_t,
                                                /* depth */ 2,
                                                /* r_args_depth */ 1,
                                                /* res_arg_index */ 1>(),
                          Op<opmath_t>(),
                          scalar.to<opmath_t>());
    return tensor_lists[1];
}

int main() {
    TensorList tensors;
    Scalar scalar = 10.0f;                                                                                       \
    return foreach_binary_op<std::multiplies>(tensors, scalar);                                                  \
}
