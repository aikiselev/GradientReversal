FROM andrewosh/binder-base

MAINTAINER Anton Kiselev <straw.berry.pie@ya.ru>

USER root

RUN apt-get update
RUN apt-get install -y git-lfs cuda 

USER main
RUN git lfs fetch
