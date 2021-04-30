FROM openvino/ubuntu20_dev:2021.3

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake \
        ffmpeg \
        libgtk-3-dev \
        libboost-all-dev \
        python3-pip \
        python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /src

COPY ./src /src

COPY ./model /model

COPY ./video /video

CMD tail -f /dev/null
