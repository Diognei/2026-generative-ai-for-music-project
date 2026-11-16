#!/bin/bash

IMAGE="genaiformusic-project:dev"

docker build -t $IMAGE .

docker run -it --rm \
    --gpus '"device=0"' \
    --shm-size=16g \
    -v "$(pwd):/app/" \
    -w /app \
    $IMAGE \
    /bin/bash

