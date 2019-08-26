FROM ubuntu:18.04

MAINTAINER ONUR DUNDAR "onur.dundar1@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY flask_app.py /app/

EXPOSE 8000

CMD ["python3", "/app/flask_app.py", "-p 8000"]