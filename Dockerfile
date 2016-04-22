FROM andrewosh/binder-base

MAINTAINER Anton Kiselev <straw.berry.pie@ya.ru>

USER root

RUN apt-get update

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN apt-get install -y git-lfs
RUN git lfs install

# CUDA
RUN apt-get install nvidia-cuda-toolkit

USER main
RUN git lfs fetch
