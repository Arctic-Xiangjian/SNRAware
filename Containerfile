FROM mcr.microsoft.com/mirror/docker/library/ubuntu:24.04

RUN <<HEREDOC
# install package dependencies and create nonroot user
set -eux

useradd -u 1001 -m nonroot

apt-get update

apt-get install -yqq --no-install-recommends \
    python3-minimal \
    python3-venv \
    python3-pip

apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
rm -rf /root/.cache
HEREDOC

# copy the minimal number of files to install the devtools
# this way we can cache as many container layers as possible
COPY pyproject.toml /opt/snraware/pyproject.toml
COPY README.md /opt/snraware/README.md
# create the skeleton of the packages so they install successfully
RUN <<HEREDOC
# create skeleton of python modules
set -eux
mkdir -p /opt/snraware/src/snraware
touch /opt/snraware/src/snraware/__init__.py
HEREDOC

WORKDIR /opt/snraware

RUN <<HEREDOC
# install python package
set -eux

python3 -m venv .venv
.venv/bin/pip install .

rm -rf /root/.cache
HEREDOC

COPY . /opt/snraware
ENV PATH="$PATH:/opt/snraware/.venv/bin"

# nothing should be installed after this step
USER nonroot
