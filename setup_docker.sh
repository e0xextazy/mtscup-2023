echo "version: \"3.8\"
services:
  $1:
    build: .
    ipc: \"host\"
    network_mode: host
    container_name: $1
    restart: always
    volumes:
      - '.:/app'
    deploy:
      resources:
        reservations:
          devices:
            - capabilities:
                - gpu
    environment:
      - VERSION=latest
      - NVIDIA_VISIBLE_DEVICES=0 
      - NVIDIA_DRIVER_CAPABILITIE=all
    entrypoint: tail -f /bin/bash

" > docker-compose.yml

echo "FROM python:3.8

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install screen sox libopenblas-base cmake python3-pip mc wget libportaudio2 ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# COPY requirements.txt /app 

RUN pip install llvmlite --ignore-installed
RUN pip3 install --upgrade pip
# RUN pip3 install -r requirements.txt

ENTRYPOINT [\"pwd\"]" > Dockerfile 

echo "up:
	docker compose up --build -d --remove-orphans
in:
	docker exec -it $1 bash
down:
	docker stop $1" > Makefile
