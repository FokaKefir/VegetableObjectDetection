FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./requirements.txt
RUN python -m venv "$VIRTUAL_ENV" \
	&& pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py

RUN mkdir -p /data /outputs /tmp/vegetable-dataset

ENTRYPOINT ["python", "main.py"]
