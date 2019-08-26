# Dockerize AI Applications

AWS, Google Cloud & Azure already provides their services for AI. 

They mainly aims to enable you to use services to send data and get the predictions. 

Let's try to utilize this face detection modules to be used as a docker service.

## Services

1. Face Detection Service 

I want to deploy a face detection service which I can connect a network stream to get the detected faces from that video stream. 

2. Age Gender Detection Service

I want to deploy a age-gender detection service which I can send face frames to get the age and gender data.

# System Design

## How to send data?

JSON is used to send data, which can also include the remote video stream or remote image url to send to docker application to read from. 

## How to receive data?

JSON files can be send to user to get the information; 

- For face detection, number of faces detected each second etc.
- For age-gender detection, age and gender information.

