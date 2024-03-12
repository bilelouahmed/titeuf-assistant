FROM python:3.9.6

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5050

ENTRYPOINT ["python3", "transcript.py"]