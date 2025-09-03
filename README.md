This repository provides two compressed storage schemes for Piecewise Linear Approximations (PLAs), which are commonly used in many learned data structures nowadays.  

1. The first scheme achieves the theoretical lower bound described in the accompanying paper (up to lower-order terms), thus achieving succinct space usage.  

2. The second scheme uses a greedy (optimal) algorithm to select segment slopes that minimize the entropy of their mantissae in the floating-point representation, and then compresses them using an entropy coder.  

Both schemes deliver practical query times, making them suitable for integration into existing learned data structures.

## Building the project

```bash
git clone --recursive https://github.com/FilippoLari/PLA-compressor
cd PLA-compressor
mkdir build
cd build
cmake ..
make -j8
```

## Credits

If you use this project in a scientific article, please cite the following paper:

```tex
@inproceedings{FerraginaL25,
    author = {Paolo Ferragina and Filippo Lari},
    title = {Compressibility Measures and Succinct Data Structures for Piecewise Linear Approximations},
    booktitle = {Proc. 36th International Symposium on Algorithms and Computation},
    year = {2025}
}
```