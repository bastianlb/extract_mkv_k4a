PROJECT ?= export_mkv
REGISTRY = artekmedregistry.campar.in.tum.de
DOCKER_IMAGE ?= ${REGISTRY}/${PROJECT}
TAG ?= latest
VOLUME=/atlasarchive/atlas/03_animal_trials/210810_animal_trial_01/recordings/

DOCKER_ARGS = --label ${PROJECT}:${TAG}
RUNTIME_ARGS = --runtime nvidia --privileged --network host --gpus all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /dev/bus/usb:/dev/bus/usb -v /etc/localtime:/etc/localtime:ro \
		-v /data/develop/extract_mkv_pcpd/scripts:/deploy \
		-v /data/datasets/atlas_export:/data/export/ \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
		--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl \
		-v ${VOLUME}:/data/input

SHELL=/bin/bash
UID := $(shell id -u)

all:
	clean
	docker-base
	docker-extract
	docker-run

clean:
	# TODO: re-add project name here
	docker stop /extract_mkv || : \
	docker rm /extract_mkv --force || : \

docker-base:
	DOCKER_BUILDKIT=1 docker build \
		-f docker/base/Dockerfile \
		-t ${DOCKER_IMAGE}_base --ssh github=/artekmed/config/credentials/id_rsa \
		--no-cache .
	# --no-cache .

docker-extract:
	DOCKER_BUILDKIT=1 docker build \
		-f docker/run/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-start:
	docker exec -it extract_mkv /bin/sh

docker-run: clean
	export XAUTHORITY=/run/user/${UID}/gdm/Xauthority
	export DISPLAY=:0
	/usr/bin/xhost +
	docker run --name extract_mkv --rm \
		${RUNTIME_ARGS} ${DOCKER_IMAGE}

