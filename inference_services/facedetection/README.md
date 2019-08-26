# Face Detection Service with Docker

Here I have a very basic application developed with OpenVINO(TM) & Flask to start inference on given source.

This app can handle single request at one time, multiple requests are not possible. 

# Build & Start Docker with WebCam

Note: I had problems with webcam loading on docker so feel free to test as below.

```bash
docker build -t facedetection:latest .

docker run -d -p 8000:8000 facedetection --device=/dev/video0:/dev/video0
```

# JSON Config

JSON Configuration have to made beforeh and in order to correctly start the inference

# Curl Request to Start Face Detection Application

```bash
curl --header "Content-Type: application/json" --request POST --data '@inference_config.json' http://127.0.0.1:8000/

curl --header "Content-Type: application/json" --request POST --data '@/home/intel/Projects/face_detection/inference_services/facedetection/inference_config.json' http://127.0.0.1:8000
```

# Retrieve Status

You can retrieve the frame-id and face coordinates by a get request from:

```bash
curl --request GET http://127.0.0.1:8000/status
```

# Retrieve Results

You can retrieve the frame-id and face coordinates by a get request from:

```bash
curl --request GET http://127.0.0.1:8000/results
```

# Check Logs

```bash
curl --request GET http://127.0.0.1:8000/logs
```

# Play Video

NOT IMPLEMENTED