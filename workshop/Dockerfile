FROM tiangolo/uwsgi-nginx-flask:python3.6


ADD ./app /app
VOLUME /app

WORKDIR /app

RUN pip install -r ./models/requirements.txt
