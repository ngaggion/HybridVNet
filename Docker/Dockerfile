FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

RUN /opt/conda/bin/conda init bash

# Install necessary packages
RUN chmod 1777 /tmp && \
    apt-get update && \
    apt-get install -y wget git nano gedit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 

WORKDIR /

# Create and activate Conda environment 
RUN /opt/conda/bin/conda install -n base conda-libmamba-solver && \
    /opt/conda/bin/conda config --set solver libmamba

# Set up conda-forge and pyg channels
RUN conda config --add channels conda-forge && \
    conda config --add channels pyg && \
    conda config --set channel_priority flexible

# Install packages
RUN conda install -n base -y \
        pyg=2.4 \
        scikit-image \
        meshio \
        pandas \
        matplotlib && \
    conda clean -a -y

# Install OpenCV and SimpleITK with pip, then remove unnecessary files
RUN pip install opencv-python simpleitk && \
    pip cache purge && \
    rm -rf /root/.cache/pip

RUN apt-get update && \
    apt-get install -y libboost-dev build-essential gcc make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 

RUN pip install trimesh vedo && \
    pip cache purge && \
    rm -rf /root/.cache/pip

RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz && \
    tar -xvf git-lfs-linux-amd64-v3.5.1.tar.gz && \
    rm git-lfs-linux-amd64-v3.5.1.tar.gz && \
    cd git-lfs-3.5.1 && \
    ./install.sh && \
    cd .. && \
    rm -rf git-lfs-3.5.1 && \
    git lfs install && \
    git clone https://huggingface.co/datasets/ngaggion/HybridVNet_weights

RUN git clone https://github.com/ngaggion/HybridGNet_3D.git

# Set the working directory
WORKDIR /HybridGNet_3D

RUN git pull