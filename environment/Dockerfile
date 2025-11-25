# hash:sha256:cbbd04f8ad1dd66188afd5b753d9f7943b427f165fc644e535a0f8204bed7321
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    ipykernel \
    matplotlib \
    pandas \
    seaborn
