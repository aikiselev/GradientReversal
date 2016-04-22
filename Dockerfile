FROM andrewosh/binder-base

MAINTAINER Anton Kiselev <straw.berry.pie@ya.ru>

USER root

RUN apt-get update

# Git LFS
RUN pip install --upgrade theano keras pandas
USER main
