FROM nvidia/cuda:11.7.1-devel-ubuntu22.04


# LLVM 14 toolchain + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        clang-14 \
        cmake \
        g++ \
        git \
        libclang-14-dev \
        llvm-14 \
        llvm-14-dev \
        make \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*


# Point bare names to the -14 variants (CuPBoP's build calls clang / llvm-config without a suffix)
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100 && \
    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-14 100 && \
    update-alternatives --install /usr/bin/llc llc /usr/bin/llc-14 100


# Clone CuPBoP at a pinned commit + submodules (the SHA freezes the tree and the submodules)
WORKDIR /opt
RUN git clone https://github.com/cupbop/CuPBoP.git && \
    cd CuPBoP && \
    git checkout 508bd62e928bea3b5f0633c8fa63b5f42f3b4da0 && \
    git submodule update --init --recursive
ENV CuPBoP_PATH=/opt/CuPBoP
ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH=/opt/CuPBoP/build/runtime:/opt/CuPBoP/build/runtime/threadPool


# Build CuPBoP
WORKDIR /opt/CuPBoP/build
RUN cmake .. && \
    make -j$(nproc)
WORKDIR /work