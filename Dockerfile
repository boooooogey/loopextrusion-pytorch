# DLEM image
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
USER root

ENV APT_PACKAGES=" \
		software-properties-common \
		apt-transport-https \
		ca-certificates \
		locales \
		fonts-liberation \
		apt-utils \
		libreadline-dev \
		zlib1g-dev \
		libgsl0-dev \
		wget \
		bzip2 \
		curl \
		zip \
		git \
		"

#Set up shell for install
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash","-c"]
ENV BASH_EN=~/.bashrc 
ENV SHELL=/bin/bash

#Get apt packages
RUN apt-get update && \
	apt-get install -y --no-install-recommends $APT_PACKAGES && \
 	rm -rf "/var/lib/apt/lists/*" && \
	apt-get clean && \
	rm -rf /var/cache/apt

#Get micromamba
COPY environment.yml /tmp/environment.yml
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1 && \
	chmod 755 ./micromamba && \
	mkdir -p $(dirname $MAMBA_ROOT_PREFIX) && \
	./micromamba shell init -s bash -r $MAMBA_ROOT_PREFIX && \
	echo "micromamba activate base" >> /root/.bashrc && \
	source ~/.bashrc && \
	./micromamba install -y -n base -f /tmp/environment.yml && \
	ln -s /micromamba /usr/bin/conda && \
	ln -s /micromamba /opt/conda/bin/conda && \
	./micromamba clean --all --yes && \
	rm ${MAMBA_ROOT_PREFIX}/lib/*.a && \
	rm -rf ${MAMBA_ROOT_PREFIX}/pkgs

# Set DLEM as working directory, copy local, and install in container
ENV LOCAL_BASE="/data"
ENV LOCAL_NAME="loopextrusion-pytorch"
ENV LOCAL_REPO="${LOCAL_BASE}/${LOCAL_NAME}"
ENV PYTHON_NAME="dlem"

RUN --mount=type=secret,id=github-token,dst=/run/secrets/git_token \
    GIT_ACCESS_TOKEN=`cat /run/secrets/git_token` && \
	mkdir -p ${LOCAL_REPO} && \
	cd ${LOCAL_BASE} && \
	pip install git+https://github.com/boooooogey/loopextrusion-pytorch.git && \
	git clone https://github.com/boooooogey/loopextrusion-pytorch.git && \
	chmod -R 0777 ${LOCAL_REPO} && \
	unset GIT_ACCESS_TOKEN
	
#Copy local changes in
COPY ./Dockerfile ./environment.yml ./pyproject.toml ${LOCAL_REPO}
COPY ./${PYTHON_NAME}/ ${LOCAL_REPO}/${PYTHON_NAME}
#Install package locally
RUN cd ${LOCAL_REPO} && \
	pip install . 
ENV PYTHONPATH "${LOCAL_REPO}/${PYTHON_NAME}"
WORKDIR ${LOCAL_REPO}


ENTRYPOINT ["DLEM"]