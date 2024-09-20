FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y htop
RUN apt-get install -y nano
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ARG UNAME=ngenze
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
RUN pip3 install pandas==2.2.2 --user
RUN pip3 install numpy==1.22.0 --user
RUN pip3 install albumentations==1.4.13 --user
RUN pip3 install tqdm==4.66.4 --user
RUN pip3 install scikit-image==0.24.0 --user
RUN pip3 install scikit-learn==1.4.2 --user
RUN pip3 install matplotlib==3.9.0 --user
RUN pip3 install seaborn==0.13.2 --user
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@92ae9f0b92aba5867824b4f12aa06a22a60a45d3'
ENV PATH="${PATH}:/root/.local/bin"

