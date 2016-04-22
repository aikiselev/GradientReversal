FROM andrewosh/binder-base

MAINTAINER Anton Kiselev <straw.berry.pie@ya.ru>

USER root

RUN apt-get update

# Git LFS
RUN wget https://packagecloud.io/github/git-lfs/packages/debian/jessie/git-lfs_1.2.0_amd64.deb/download
RUN chmod +x download && dpkg -i download
RUN git lfs install

# CUDA
RUN wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1504-7-5-local_7.5-18_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1504-7-5-local_7.5-18_amd64.deb
RUN apt-get update
RUN apt-get install -y cuda

USER main
RUN git lfs fetch
