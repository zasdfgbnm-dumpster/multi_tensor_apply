clear
nvcc -I. ATen/native/cuda/ForeachBinaryOpScalar.cu -o test
./test
