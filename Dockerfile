# ARG PROJECT_PATH

FROM python:3.9-slim

SHELL ["/bin/bash", "-c"]

# COPY $PROJECT_PATH /root/texture_classification
COPY . /root/application
WORKDIR /root/application

RUN rm -rf __pycache__ && \ 
    rm -rf ./textile_classification/__pycache__ && \ 
    rm -rf ./tests/__pycache__ && \ 
    rm -rf ./textile_classification/data_setters/__pycache__ && \ 
    rm -rf ./textile_classification/utils/__pycache__ && \ 
    rm -rf ./textile_classification/model/__pycache__ && \ 
    rm -rf ./hub/pretrained_models/__pycache__ && \ 
    rm -rf ./venv && \ 
    rm -rf ./*.egg-info/ && \ 
    rm -rf ./.python-version && \ 
    rm -rf ./.git && \ 
    rm -rf ./temporary_test.py

# RUN apt-get update && apt-get install -y vim

RUN python -m venv venv && \ 
    ./venv/bin/python -m pip install pip setuptools wheel && \ 
    ./venv/bin/python -m pip install --upgrade pip setuptools wheel && \ 
    ./venv/bin/python -m pip install -e .

RUN chmod +x docker_help.sh

ENV PATH="/root/application/venv/bin:${PATH}"
# avoid mlflow rasing a warning regarding to git.
ENV GIT_PYTHON_REFRESH=0

CMD ["./docker_help.sh"]

VOLUME /root/application