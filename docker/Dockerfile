FROM ghcr.io/onnxmlir/onnx-mlir-dev
WORKDIR /workdir
ENV HOME=/workdir

# 1) Install packages.
ENV PATH=$PATH:/workdir/bin
RUN apt-get update
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-pip
RUN python -m pip install --upgrade pip
RUN apt-get install -y gdb
RUN apt-get install -y lldb

# 2) Instal optional packages, comment/uncomment/add as you see fit.
RUN apt-get install -y vim
RUN apt-get install -y emacs
RUN apt-get install -y valgrind
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y clang-format
RUN python -m pip install wheel
RUN python -m pip install numpy
RUN python -m pip install torch==2.0.0+cpu torchvision==0.15.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/onnx/tutorials.git
# Install clang
RUN apt-get install -y lsb-release wget software-properties-common
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
# For development
RUN apt-get install -y ssh-client

# 3) When using vscode, copy your .vscode in the Dockerfile dir and
#    uncomment the two lines below.
# WORKDIR /workdir/.vscode
# ADD .vscode /workdir/.vscode

# 4) When using a personal workspace folder, set your workspace sub-directory
#    in the Dockerfile dir and uncomment the two lines below.
# WORKDIR /workdir/workspace
# ADD workspace /workdir/workspace

# 5) Fix git by reattaching head and making git see other branches than main.
WORKDIR /workdir/onnx-mlir
# Add optional personal fork and disable pushing to upstream (best practice).
# RUN git remote add origin https://github.com/<<user>>/onnx-mlir.git
# RUN git remote set-url --push upstream no_push

# 6) Set the PATH environment vars for make/debug mode. Replace Debug
#    with Release in the PATH below when using Release mode.
WORKDIR /workdir
ENV NPROC=40
RUN ls

RUN git clone -n https://github.com/llvm/llvm-project.git llvm
#RUN cd llvm-project && git checkout 00128a20eec27246719d73ba427bf821883b00b4 && cd ..
#RUN mkdir llvm-project/build && cd llvm-project/build
RUN cd llvm-project && git checkout 00128a20eec27246719d73ba427bf821883b00b4 && \
    rm -rf build && mkdir build
RUN cd llvm-project/build && cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DENABLE_LIBOMPTARGET=OFF \
   -DLLVM_ENABLE_LIBEDIT=OFF

RUN cd llvm-project/build && cmake --build . -- ${MAKEFLAGS}
RUN cd llvm-project/build && cmake --build . --target check-mlir

RUN rm -rf /workdir/onnx-mlir

ENV PATH=$PATH:/workdir/onnx-mlir/build/Debug/bin/:/workdir/onnx-mlir/build/Debug/lib:/workdir/llvm-project/build/bin

RUN apt-get update && apt-get install -y openssh-server git && \
    mkdir -p /run && \
    mkdir -p /var/run && \
    mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config && \
    passwd -d root

RUN apt-get install -y libprotobuf-dev python3.12 zip

# RUN ACCEL=ARX NPROC=80 WORKDIR=/workdir LLVM_PROJECT_ROOT=${WORKDIR}/llvm-project \
#     cd ${WORKDIR} && \
#     git clone --recursive https://gitlab.com/ones-ai/mlir-arx -b chacha/mlir-arx && \
#     cd mlir-arx && \
#     mkdir build && \
#     cd build && \
#     cmake -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
#                 -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
#                 -DCMAKE_BUILD_TYPE=Debug \
#                 -DLLVM_ENABLE_ASSERTIONS=OFF \
#                 -DMLIR_DIR=${LLVM_PROJECT_ROOT}/build/lib/cmake/mlir  \
#                 -DONNX_MLIR_ACCELERATORS=${ACCEL} \
#                 ..  && \
#     cmake --build .

CMD ["/usr/sbin/sshd", "-D"]
