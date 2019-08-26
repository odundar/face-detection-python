# Run Application

This is a test application to run face-age-gender detections configured with the json file stored in config folder.

Before running this application make sure OpenVINO environment has been setup correctly and `detection`, `utils` modules are in same directory as `app` folder.

```bash
python3 app/face_detection_openvino.py config/config.json
```

If you run inside the `app` folder make sure `detection` and `utils` modules are included in `PYTHONPATH`



