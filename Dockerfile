FROM hjtcjsfmswd853.xuanyuan.run/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main \
        --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n mamba python=3.10 && \
    conda clean -afy

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate mamba" \
    > /etc/profile.d/conda_mamba.sh

SHELL ["/bin/bash", "-c", "-l"]
ENV CONDA_DEFAULT_ENV=mamba
ENV PATH=/opt/conda/envs/mamba/bin:$PATH

RUN conda install numpy
RUN pip install --no-cache-dir --prefer-binary -U \
        cyvcf2==0.31.1 scipy==1.15.3 pandas==2.3.2 scikit-learn==1.7.2 tqdm==4.67.1 \
        causal-conv1d>=1.4.0
        
RUN pip install --no-cache-dir -U \
        torch torchvision \
        accelerate \
        deepspeed \
        mamba-ssm==2.2.5

CMD ["conda","activate", "evofill"]