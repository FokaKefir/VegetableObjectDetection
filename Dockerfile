FROM pytorch/pytorch:2.5.1-cpu

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py

RUN mkdir -p /data /outputs /tmp/vegetable-dataset

ENTRYPOINT ["python", "main.py"]
