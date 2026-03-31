FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt update

RUN apt install -y wget git ffmpeg libsm6 libxext6
ENV NVIDIA_DRIVER_CAPABILITIES=all

SHELL ["/bin/bash", "--login", "-c"]

# Install (mini) conda
RUN cd / && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN /bin/bash -c "source ~/.bashrc && \
    /opt/conda/bin/conda config --set always_yes yes --set changeps1 no && \
    /opt/conda/bin/conda config --add channels defaults && \
    /opt/conda/bin/conda config --set channel_alias https://conda.anaconda.org && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r"

# install dependencies
RUN source ~/.bashrc && \
    /opt/conda/bin/conda update -n base -c defaults conda -y &&\
    conda create -n gizmo python=3.10

RUN echo "conda activate gizmo" >> ~/.bashrc

RUN conda activate gizmo &&\
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN conda activate gizmo &&\
    conda install ninja &&\
    MAX_JOBS=8 pip install flash-attn --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN conda activate gizmo &&\
    pip install -r /tmp/requirements.txt

# support up to H100, do not support RTX 5090 or RTX PRO 6000
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# install tapip3d
RUN conda activate gizmo &&\
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
RUN conda activate gizmo &&\
    pip install xformers>=0.0.27 torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124


RUN cd /tmp && \
    git clone https://github.com/zbw001/TAPIP3D.git && \
    cd TAPIP3D && \
    conda activate gizmo && \
    cd third_party/pointops2 && \
    LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install

RUN cd /tmp/TAPIP3D/third_party/megasam/base && \
    apt install ninja-build -y && \
    conda activate gizmo && \
    LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install

# install pyroki
RUN conda activate gizmo &&\
    pip install git+https://github.com/lsun-bdai/pyroki.git@stable git+https://github.com/huggingface/diffusers.git ray[serve] pyzmq google-genai google-cloud-storage


# install spatialtracker dependencies
RUN conda activate gizmo && \
    pip install pycolmap==3.11.1 git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d scikit-learn flow_vis moviepy==1.0.0 pyceres==2.4 decord