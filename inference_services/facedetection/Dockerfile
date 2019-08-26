FROM ubuntu:18.04

MAINTAINER ONUR DUNDAR "onur.dundar1@gmail.com"

# ARG DOWNLOAD_LINK=~/Projects/workshop_installation/openvino_installer/l_openvino_toolkit_p_2019.2.242.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR

COPY l_openvino_toolkit_p_2019.2.242.tgz $TEMP_DIR

WORKDIR $TEMP_DIR

RUN tar xf l_openvino_toolkit_p_2019.2.242.tgz && \
    cd l_openvino_toolkit_p_2019.2.242 && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR

RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev libgtk-3-0

WORKDIR /app
COPY requirements.txt /app

RUN pip3 install -r requirements.txt

ENV PATH="$PATH:/opt/intel/openvino/deployment_tools/model_optimizer"
ENV PYTHONPATH="/opt/intel/openvino/python/python3.6:/opt/intel/openvino/deployment_tools/model_optimizer:/opt/intel/openvino/python/python3:"
ENV OpenCV_DIR="/opt/intel/openvino/opencv/cmake"
ENV InferenceEngine_DIR="/opt/intel/openvino/deployment_tools/inference_engine/share"
ENV IE_PLUGINS_PATH="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/openvino/opencv/lib:/opt/intel/opencl:/opt/intel/openvino/deployment_tools/inference_engine/external/hddl/lib"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/openvino/deployment_tools/inference_engine/external/gna/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/openvino/openvx/lib"
ENV INTEL_OPENVINO_DIR="/opt/intel/openvino"
ENV INTEL_CVSDK_DIR="/opt/intel/openvino"
ENV HDDL_INSTALL_DIR="/opt/intel/openvino/deployment_tools/inference_engine/external/hddl"

COPY detection /app/detection
COPY utils /app/utils
COPY models /app/models
COPY videos /app/videos

COPY face_detection_service.py /app

EXPOSE 8000

CMD ["python3", "/app/face_detection_service.py", "-p 8000"]