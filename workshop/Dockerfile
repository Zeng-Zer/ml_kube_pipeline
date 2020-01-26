FROM python:3.7.6-slim-stretch

ADD ./app /app
VOLUME /app
WORKDIR /app

RUN pip install -r ./models/requirements.txt
