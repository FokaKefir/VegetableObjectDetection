FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py

RUN mkdir -p /data /outputs /tmp/vegetable-dataset

ENTRYPOINT ["python", "main.py"]
