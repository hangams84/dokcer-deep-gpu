FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# hadolint ignore=DL3008
RUN set -x && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
        vim \
        curl \
        unzip \
        unrar \
        libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O anaconda.sh -q && \
    echo 2751ab3d678ff0277ae80f9e8a74f218cfc70fe9a9cdc7bb1c137d7e47e33d53 anaconda.sh > shasum && \
    sha256sum --check --status shasum && \
    /bin/bash anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy 

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# ==================================================================
# darknet
# ------------------------------------------------------------------

    $GIT_CLONE https://github.com/AlexeyAB/darknet ~/darknet && \
    cd ~/darknet && \
    sed -i 's/GPU=0/GPU=1/g' ~/darknet/Makefile && \
    sed -i 's/CUDNN=0/CUDNN=1/g' ~/darknet/Makefile && \
    make -j"$(nproc)" && \
    cp ~/darknet/include/* /usr/local/include && \
    cp ~/darknet/darknet /usr/local/bi && \

# ==================================================================
# python
# ------------------------------------------------------------------
     DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm \
        && \

# ==================================================================
# boost
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libboost-all-dev \
        && \

# ==================================================================
# chainer
# ------------------------------------------------------------------

    $PIP_INSTALL \
        cupy \
        chainer \
        && \

# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyter \
        && \

# ==================================================================
# mxnet
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libatlas-base-dev \
        graphviz \
        && \

    $PIP_INSTALL \
        mxnet-cu110 \
        graphviz \
        && \

# ==================================================================
# onnx
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        protobuf-compiler \
        libprotoc-dev \
        && \

    $PIP_INSTALL \
        --no-binary onnx onnx \
        && \

    $PIP_INSTALL \
        onnxruntime \
        && \

# ==================================================================
# paddle
# ------------------------------------------------------------------

    $PIP_INSTALL \
        paddlepaddle-gpu \
        && \

# ==================================================================
# pytorch
# https://pytorch.org/get-started/previous-versions/
# ------------------------------------------------------------------

    $PIP_INSTALL \
        future \
        numpy==1.15.2 \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        --pre torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f \
        https://download.pytorch.org/whl/torch_stable.html 


RUN python3.6 -m pip install --upgrade pip

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
       
# ==================================================================
# tensorflow
# ------------------------------------------------------------------
        #pip uninstall tensorflow && \
        $PIP_INSTALL tensorflow-gpu==2.2.0 \
        #ERROR: No matching distribution found for tensorflow-gpu==2.1.0
        #$PIP_INSTALL --pre tensorflow==2.1.0 -f https://files.pythonhosted.org/packages/85/d4/c0cd1057b331bc38b65478302114194bd8e1b9c2bbc06e300935c0e93d90/tensorflow-2.1.0-cp36-cp36m-manylinux2010_x86_64.whl \
        && \

# ==================================================================
# theano
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libblas-dev \
        && \

    wget -qO- https://github.com/Theano/libgpuarray/archive/v0.7.6.tar.gz | tar xz -C ~ && \
    cd ~/libgpuarray* && mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j"$(nproc)" install && \
    cd ~/libgpuarray* && \
    python setup.py build && \
    python setup.py install && \

    printf '[global]\nfloatX = float32\ndevice = cuda0\n\n[dnn]\ninclude_path = /usr/local/cuda/targets/x86_64-linux/include\n' > ~/.theanorc && \

    $PIP_INSTALL \
        https://github.com/Theano/Theano/archive/master.zip \
        && \

# ==================================================================
# jupyterlab
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
        && \

# ==================================================================
# keras
# ------------------------------------------------------------------

    $PIP_INSTALL \
        h5py \
        keras \
        && \

# ==================================================================
# lasagne
# ------------------------------------------------------------------

    $GIT_CLONE https://github.com/Lasagne/Lasagne ~/lasagne && \
    cd ~/lasagne && \
    $PIP_INSTALL \
        . && \

# ==================================================================
# opencv
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \      
    $GIT_CLONE --branch 4.5.2 https://github.com/opencv/opencv ~/opencv && \
    mkdir -p ~/opencv/build && cd ~/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_EXAMPLES=OFF \
          .. && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2

# ==================================================================
# sonnet
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \  
    $PIP_INSTALL \
        tensorflow_probability \
        "dm-sonnet>=2.0.0b0" --pre \
        && \

# ==================================================================
# caffe
# ------------------------------------------------------------------

    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        caffe-cuda \
        && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*


#출처: https://sukill.tistory.com/27 [sukill 의 블로그]
# install SSH Service
#RUN apt-get update && apt-get install -y openssh-server 
RUN apt-get update && apt-get install -y ssh
# apt-get update로 기존설치된 패키지들의 업데이트 버전이 있을 경우 업데이트 
# openssh-server 설치 (-y 옵션은 프롬프트에 자동으로 yes라고 응답해줌)

RUN mkdir /var/run/sshd 
# /var/run/sshd 디렉토리 생성


# Install sudo
RUN apt-get update && \
    apt-get install -y sudo \
    xterm \
    curl \
    apt-utils

    
# Configure user
ARG user=keti
ARG passwd=keti
ARG uid=1000
ARG gid=1000

ENV USER=$user
ENV PASSWD=$passwd
ENV UID=$uid
ENV GID=$gid
RUN groupadd $USER && \
    useradd --create-home --no-log-init -g $USER $USER && \
    usermod -aG sudo $USER && \
    echo "$PASSWD:$PASSWD" | sudo chpasswd && \
    chsh -s /bin/bash $USER && \
    # Replace 1000 with your user/group id
    usermod  --uid $UID $USER && \
    groupmod --gid $GID $USER

RUN adduser $USER sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


RUN sed -ie 's/keti:!/keti:\$6\$pIPlIFrg\$WZRTmOoouwD6L\/IDgz8fw97agQjdeXticOJyayjUYwmiuNKkJE92rBtr0XDXhVE4rSi9PkWv8rtnA0rCuR6ts\//g' /etc/shadow


RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config 
# openssh 7.0버전 부터 PermitRootLogin 옵션이 'prohibit-password'를 인자로 받아들인다. (without-password 와 살짝 비슷) 
# sed -i 옵션은 타겟 파일에서 문자열을 검색한 후 변경한다. 
# sshd_config 파일에서 PermitRootLogin prohibit-password를 PermitRootLogin yes 로 변경 
# pw 없이 접속 가능하던 것을 root 계정으로 로그인 하도록 변경 
# https://www.openssh.com/txt/release-7.0 
# permitrootlogin : http://manpages.ubuntu.com/manpages/trusty/en/man5/sshd_config.5.html


# SSH login fix. Otherwise user is kicked off after login 
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd 
# s/대신 s@로 표현되었다(s 는 substitue를 뜻함) 
# / 대신 백슬래쉬와 개행문자를 제외한 어느 것도 delimiter로 사용가능하다. 문자 같은 경우는 백슬래쉬 앞에 함게 써야 함 
# delimiter 안쪽의 문자열들은 Regex 표현식이 사용됨 
# http://pubs.opengroup.org/onlinepubs/9699919799/utilities/sed.html 
#'session required pam_loginuid.so' 문자열을 #'session optional pam_loginuid.so' 문자열로 교체 
# 표현식 내부의 \s는 공백을 의미 # *는 반복을 의미, \s*는 공백이 0번이상 반복 
# pam ( Pluggable Authentication Modules)

#ENV NOTVISIBLE "in users profile"
# 환경변수 설정, NOTVISIBLE이란 환경변수를 무슨 용도로 쓰는지는 불명
# 어쩌면 그냥 환경변수 설정 예를 보여주는데 그 값이 user profile에 저장된다는 의미인걸까..
# 
#RUN echo "export VISIBLE=now" >> /etc/profile
# /etc/profile 파일은 user shell에 시스템 전역 환경변수를 설정하는데 사용된다.
# >>, 파일 뒷부분에 내용 추가

RUN apt install ssh-askpass

RUN /usr/bin/ssh-keygen -A 
#RUN sudo ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -y
#RUN sudo ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -y
#RUN sudo ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -y
#RUN sudo ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -y


EXPOSE 22
# 호스트와 연결할 포트번호

#RUN CONDA_INSTALL="conda install" && \
#    $CONDA_INSTALL python=3.6.5 \
#    tensorflow-gpu=2.1.0 \
#    cudnn=7.6.5 \
#    numpy=1.15 \
#    pytorch=1.7.1 \
#    tensorboard=2.3.0

# for jupyter notebook
EXPOSE 8888

# for tensorboard
EXPOSE 6006

RUN mkdir /notebook
VOLUME ["/notebook"]

ENTRYPOINT jupyter notebook --no-browser --ip="0.0.0.0" --port=8888 --notebook-dir="/notebook"

WORKDIR /notebook

USER $USER

CMD [ "/bin/bash" ]
