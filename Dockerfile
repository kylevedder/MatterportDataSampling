FROM nvidia/cudagl:11.2.0-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

RUN nvcc --version

# System packages 
RUN apt update
RUN apt install -y software-properties-common wget curl gpg gcc git make libssl-dev libmodule-install-perl libboost-all-dev libopenblas-dev

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Install prerequisites needed for spconv and second.pytorch.
RUN conda install \
habitat-sim \
headless \
pip \
pytorch=1.9 \
torchvision \
cudnn \
openblas-devel \
scikit-image \
scipy \
numba \
pillow \
matplotlib \
seaborn \
psutil \
-c pytorch -c conda-forge -c aihabitat -c anaconda -c defaults

RUN python -c "import torch; print(torch.__version__)"

RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git habitat-lab
WORKDIR habitat-lab
# RUN git reset --hard d6ed1c0a0e786f16f261de2beafe347f4186d0d8
RUN pip install -e .
RUN pip install open3d pyntcloud pandas
WORKDIR /project