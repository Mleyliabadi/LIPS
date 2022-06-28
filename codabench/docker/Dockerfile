# Use an official Python runtime as a parent image
FROM ubuntu:latest

MAINTAINER Jérôme PICAULT <jerome.picault@rte-france.com>

ENV DEBIAN_FRONTEND noninteractive

# generic install
RUN apt-get update && \
    apt-get install -y \
    python3.8 python3.8-dev python3-pip\
    less \
    apt-transport-https \
    git\
    ssh \
    tar

# to gather proxy details during the build (beware of arguments when building the image, prefer to use the provided script)
ARG HTTP_PROXY
ENV HTTP_PROXY=$HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
RUN export http_proxy=$HTTP_PROXY; export https_proxy=$HTTPS_PROXY

# install additional packages
RUN pip install --no-cache-dir loguru filelock PyYAML json2table numba

# sparse checkout repository
# !! token required to access private github
RUN git clone -b  main --single-branch https://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@github.com/Mleyliabadi/LIPS.git --no-checkout --depth 1

# get LIPS from repo and install recommended dependencies
WORKDIR /LIPS
# NB. git sparse-checkout requires git >=2.25
# start fetching only root files, then list subfolders to checkout and do checkout
RUN git sparse-checkout init --cone && \
    git sparse-checkout set lips reference_data && \
    git checkout -f

# install LIPS package and dependencies
RUN pip install --no-cache-dir -e .[recommended]

# workaround to have the environment
RUN python3 -c "import grid2op; env=grid2op.make('l2rpn_case14_sandbox')"
RUN python3 -c "import grid2op; env=grid2op.make('l2rpn_neurips_2020_track1_small')"

# artificial command to force next run (git update)
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
# force pull
RUN git pull
# reinstall LIPS package and dependencies (in case of git update)
RUN pip install --no-cache-dir -e .[recommended]

WORKDIR /

# Make port 80 available to the world outside this container
EXPOSE 80

ENV LIPS_HOME=/LIPS

ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""


