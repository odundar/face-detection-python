# Python AI Project with Intel(R) OpenVINO(TM) Inference Engine Python API

Here is a template project to reuse for production ready applications to use Deep Learning models.

# OpenVINO(TM) Toolkit Installation and Configuration for Ubuntu 18.04

 ``OpenVINO(TM) Version: 2019.2.242``

You can install OpenVINO(TM) by following the instructions published online documentation.

- https://docs.openvinotoolkit.org/

- https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html

Before running this Python application:

1. Set Environment Variables on the current workspace:

```bash
source /opt/intel/openvino/bin/setupvars.sh
```

**OR**

2. Set Environment Variables System Wide

Copy `files/intel-openvino.sh` & `files/intel-openvino.conf` file as shown below:

```bash
sudo cp files/intel-openvino.sh /etc/profile.d/
sudo cp files/intel-openvino.conf /etc/ld.so.conf.d/ 
sudo reboot
```

## Clone This Repository

```bash
git clone https://github.com/odundar/face_detection.git
```

## Quick Run for Face Detection Application
If all setup completed successfully, you can use the default configurations to give a start for face detection application. 

```bash
python3 face_detection_openvino.py config/config.json
```

## app/

`app` folder includes apps & services, ready to run for face, age, gender detection applications.

## config/

`config` folder includes app and service default configurations to be used as template.  

## detection/

`detection` folder contains the modules and classes to reuse for inference application development.

## docker/


## files/

## models/
 
