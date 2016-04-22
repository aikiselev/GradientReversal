FROM andrewosh/binder-base

MAINTAINER Anton Kiselev <straw.berry.pie@ya.ru>

USER root

RUN apt-get update

# Git LFS
RUN wget https://packagecloud.io/github/git-lfs/packages/debian/jessie/git-lfs_1.2.0_amd64.deb/download
RUN chmod +x download && dpkg -i download
RUN git lfs install

USER main
