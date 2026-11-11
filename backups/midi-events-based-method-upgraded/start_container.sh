#!/bin/bash

IMAGE="musicgen:dev"

docker build -t $IMAGE .

docker run -it --rm \
    --gpus '"device=0"' \
    --shm-size=4g \
    -v "$(pwd):/app/" \
    -w /app \
    $IMAGE \
    /bin/bash

