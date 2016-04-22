FROM andrewosh/binder-base

MAINTAINER Anton Kiselev <straw.berry.pie@ya.ru>

USER root

RUN apt-get update

# Git LFS
RUN wget https://packagecloud.io/github/git-lfs/packages/debian/jessie/git-lfs_1.2.0_amd64.deb/download
RUN chmod +x download && dpkg -i download
RUN git lfs install

# CUDA
RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb
RUN apt-get update
RUN apt-get install cuda

USER main
RUN git lfs fetch
