ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
WORKDIR /tdecomp
COPY . .
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir ".[experiments]"
ENTRYPOINT ["python", "-m", "examples.tensorgrad_train"]
CMD ["--device", "cpu", "--output", "/outputs/tensorgrad-training.json"]
