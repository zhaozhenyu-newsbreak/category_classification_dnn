FROM python:3.7

WORKDIR /app

ADD . /app

RUN pip install -r environment.txt

