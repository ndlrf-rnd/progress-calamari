ARG BASE_CONTAINER=centos:7.6.1810
FROM ${BASE_CONTAINER}

LABEL maintainer="Ilya Kutukov <i@leninka.ru>"
LABEL org="https://rusneb.ru"
LABEL repo="https://github.com/ndlrf-rnd/progress-calamari"
LABEL description="NDL RF - Progress - OCR training subsystem"

ENV CONTAINER docker

RUN rpm --import http://mirror.centos.org/centos/RPM-GPG-KEY-CentOS-5 && \
    yum update -y && \
#    yum install -y \
    virtualenv -p python3 /home/progress/venv/ && \
    source /home/progress/venv/bin/activate && \
    python -m pip install -U pip && \
    pip install -r /home/progress/requirements.txt && \
    pip uninstall pillow && \
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd && \

#      sudo \
#      g++ \
#      wget \
#      git \
#      make \
#      unzip \
#      bsdtar \
#      java-1.8.0-openjdk \
#      gcc-c++ \
#      cairo-devel \
#      pango-devel \
#      libjpeg-turbo-devel \
#      giflib-devel \
#      tree \
#      dos2unix \
      && \
    yum clean all && \
    rm -rf /var/cache/yum



COPY /*.* /home/progress/
COPY /dockerscripts/ /home/progress/dockerscripts/

RUN mkdir -p /home/progress/contrib/ && \
RUN mkdir /output/ &&
    mkdir /input/

WORKDIR /home/progress
VOLUME /input/
VOLUME /cache/
VOLUME /output/

ENTRYPOINT ["/home/progress/dockerscripts/entrypoint.sh"]
