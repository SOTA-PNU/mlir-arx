FROM ubuntu:22.04

WORKDIR /work

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get install -y git dos2unix

RUN apt install -y curl wget lsb-release software-properties-common gnupg python3-pip python3-dev \
                    ninja-build vim gcc cmake make

RUN python3 -m pip install --upgrade pip && \
        pip install onnx

# When you use the command below, replace {intern} with your own username.
RUN mkdir -p /home/intern/app/3rdparty

COPY ./3rdparty/build_mlir.sh /home/intern/app/3rdparty
COPY ./3rdparty/build_onnxmlir.sh /home/intern/app/3rdparty
COPY ./setup.sh /home/intern/app

RUN chmod 655 /home/intern/app/setup.sh && chmod 655 /home/intern/app/3rdparty/build_mlir.sh && chmod 655 /home/intern/app/3rdparty/build_onnxmlir.sh

RUN dos2unix /home/intern/app/setup.sh && dos2unix /home/intern/app/3rdparty/build_mlir.sh && dos2unix /home/intern/app/3rdparty/build_onnxmlir.sh

CMD ["tail", "-f", "/dev/null"]