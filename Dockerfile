# Rust builder
FROM asia-south1-docker.pkg.dev/nstream-mvp-stag/nstreamai-common/rust:latest AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

# Install 'gcc-c++' in the chef image
RUN yum install -y gcc-c++ openssl-devel && \
    yum clean all && rm -rf /var/cache/yum

# Planning stage using Cargo Chef
FROM chef as planner
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY router router
COPY launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

# Builder stage
FROM chef AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

# Install 'unzip' and any other required packages
RUN yum install -y unzip && \
    yum clean all && rm -rf /var/cache/yum

# Install protoc (Protocol Buffers compiler)
RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY router router
COPY launcher launcher
RUN cargo build --release

# Python builder with PyTorch and CUDA
FROM nvidia/cuda:12.4.0-devel-ubi9 as pytorch-install

ARG PYTORCH_VERSION=2.4.0
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=12.4.0
ARG MAMBA_VERSION=23.3.0-0
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH


RUN yum update -y && yum install -y \
    make \
    wget \
    git \
    gcc \
    gcc-c++ \
    openssl-devel \
    python3-pip \
    python3-devel \
    unzip \
    bzip2 \
    && yum clean all && rm -rf /var/cache/yum

# Install Conda
RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    wget -O ~/mambaforge.sh "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh"
RUN chmod +x ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

# Install pytorch
# On arm64 we exit with an error code
RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -c anaconda -c conda-forge -y "python=${PYTHON_VERSION}" "pytorch=$PYTORCH_VERSION" "pytorch-cuda=$(echo $CUDA_VERSION | cut -d'.' -f 1-2)"  ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

# Install build tools for CUDA kernels
RUN yum install -y ninja-build cmake && \
    yum clean all && rm -rf /var/cache/yum

# Build Flash Attention CUDA kernels
FROM pytorch-install as flash-att-builder
WORKDIR /usr/src
COPY server/Makefile-flash-att Makefile
RUN make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM pytorch-install as flash-att-v2-builder
WORKDIR /usr/src
COPY server/Makefile-flash-att-v2 Makefile
RUN make build-flash-attention-v2-cuda

# Build exllama kernels
FROM pytorch-install as exllama-kernels-builder
WORKDIR /usr/src
COPY server/exllama_kernels/ .
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build exllamav2 kernels
FROM pytorch-install as exllamav2-kernels-builder
WORKDIR /usr/src
COPY server/exllamav2_kernels/ .
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build AWQ kernels
FROM pytorch-install as awq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-awq Makefile
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" make build-awq

# Build custom CUDA kernels
FROM pytorch-install as custom-kernels-builder
WORKDIR /usr/src
COPY server/custom_kernels/ .
RUN python setup.py build

# Build vLLM CUDA kernels
FROM pytorch-install as vllm-builder
WORKDIR /usr/src
RUN yum install -y wget && \
    yum clean all && rm -rf /var/cache/yum
RUN wget 'https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.tar.gz' && \
    tar xzvf 'cmake-3.30.0-linux-x86_64.tar.gz' && \
    ln -s "$(pwd)/cmake-3.30.0-linux-x86_64/bin/cmake" /usr/local/bin/cmake
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
COPY server/Makefile-vllm Makefile
RUN make build-vllm-cuda

# Build Megablocks kernels
FROM pytorch-install as megablocks-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-megablocks Makefile
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN make build-megablocks

# Build Punica CUDA kernels
FROM pytorch-install as punica-builder
WORKDIR /usr/src
COPY server/punica_kernels/ .
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN  python setup.py build

# Build EETQ kernels
FROM pytorch-install as eetq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-eetq Makefile
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" make build-eetq

# Base image
FROM nvidia/cuda:12.4.0-base-ubi9 as base

# Conda environment
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# LoRAX environment variables
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

WORKDIR /usr/src

RUN yum update -y && yum install -y \
    openssl-devel \
    make \
    sudo \
    && yum clean all && rm -rf /var/cache/yum

# Copy Conda environment with PyTorch installed
COPY --from=pytorch-install /opt/conda /opt/conda

# Copy build artifacts from kernel builders
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=flash-att-v2-builder /usr/src/flash-attention-v2/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=custom-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=exllamav2-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=awq-kernels-builder /usr/src/llm-awq/awq/kernels/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=vllm-builder /usr/src/vllm/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=megablocks-kernels-builder /usr/src/megablocks/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=punica-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=eetq-kernels-builder /usr/src/eetq/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Install Flash Attention dependencies
RUN pip install einops --no-cache-dir

# Install FlashInfer
RUN pip install --no-cache-dir https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.5/flashinfer-0.1.5+cu124torch2.4-cp310-cp310-linux_x86_64.whl#sha256=f22ccbd800b3d74f01f95c68c5c04b5eb64b181ebf18c3a8ad4f3dc15805de59

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile

RUN cd server && \
    make gen-server && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install ".[bnb, accelerate, quantize, peft, outlines]" --no-cache-dir

# Install router and launcher binaries
COPY --from=builder /usr/src/target/release/lorax-router /usr/local/bin/lorax-router
COPY --from=builder /usr/src/target/release/lorax-launcher /usr/local/bin/lorax-launcher

RUN yum install -y \
    gcc \
    gcc-c++ \
    && yum clean all && rm -rf /var/cache/yum

# Final image
FROM base
ENV CUDA_HOME=/usr/local/cuda
LABEL source="https://github.com/predibase/lorax"

RUN yum update -y && yum install -y sudo unzip time && \
    yum clean all && rm -rf /var/cache/yum

COPY container-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
COPY sync.sh sync.sh
RUN chmod +x sync.sh

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

ENTRYPOINT ["lorax-launcher"]
CMD ["--json-output"]