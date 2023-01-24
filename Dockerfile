FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
WORKDIR /usr/src/app
RUN apt-get update
RUN apt-get install -y python3 python3-pip git

# Timesone setting for system libararies required by opencv
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Dubai
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

RUN pip install ipdb numpy matplotlib tqdm wandb ipdb
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install gym[atari]==0.23.1
RUN pip install gym[accept-rom-license]
RUN pip install tensorboard pygame==2.1.0
RUN pip install stable-baselines3==1.2.0
RUN sed -i 's/randint(1/integers(1/g' /usr/local/lib/python3.8/dist-packages/stable_baselines3/common/atari_wrappers.py

#COPY . .
CMD nvidia-smi

#CMD nvidia-smi && \
#python3 dqn_atari.py  --env-id BreakoutNoFrameskip-v4
# --track --wandb-project-name=cleanrl
