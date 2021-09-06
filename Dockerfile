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
habitat-sim=0.2.0 \
headless=1.0 \
pip=21.1.3 \
pytorch=1.9 \
torchvision=0.2.2 \
cudnn=8.2.1.32 \
openblas-devel=0.3.2 \
scikit-image=0.18.2 \
scipy=1.5.3 \
numba=0.53.1 \
pillow=8.2.0 \
matplotlib=3.4.2 \
seaborn=0.11.1 \
psutil=5.8.0 \
-c pytorch -c conda-forge -c aihabitat -c anaconda -c defaults

RUN python -c "import torch; print(torch.__version__)"

RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git habitat-lab
WORKDIR habitat-lab
# RUN git reset --hard d6ed1c0a0e786f16f261de2beafe347f4186d0d8
RUN pip install -e .
RUN pip install open3d pyntcloud pandas
WORKDIR /project