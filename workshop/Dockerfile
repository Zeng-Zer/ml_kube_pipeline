FROM python:3.7.6-slim-stretch

ADD ./app /app
VOLUME /app

WORKDIR /app

EXPOSE 5000

RUN pip install -r ./models/requirements.txt
