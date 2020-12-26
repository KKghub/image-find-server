FROM python:3.8.2-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python ./train_and_build.py

CMD [ "python", "./train_and_build.py" ]