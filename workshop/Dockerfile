FROM tiangolo/uwsgi-nginx-flask:python3.6


ADD ./app /app
VOLUME /app

WORKDIR /app

RUN pip install -r ./models/requirements.txt

RUN chmod +x main.py

ARG TRAINED_MODEL_PATH=/app/models/model_train
ARG DATA_PATH=/app/models/kc_house_data.csv

RUN python3 train.py

ENTRYPOINT ["python3"]
CMD ["main.py"]