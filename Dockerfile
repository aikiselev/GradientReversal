FROM aikiselev/GradientReverse

USER root

RUN apt-get update
RUN apt-get install -y git-lfs cuda 

USER main
RUN git lfs fetch