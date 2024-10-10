FROM registry.access.redhat.com/ubi9/ubi

# Install required packages and Rust
RUN yum update -y && \
    yum install -y gcc-c++ gcc make && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y

# Add Rust to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify Rust installation
RUN rustc --version

# Set the CHEF_TAG environment variable
ENV CHEF_TAG="0.1.68"

# Install cargo-chef and clean up
RUN cargo install cargo-chef --locked --version $CHEF_TAG \
    && rm -rf $CARGO_HOME/registry/
