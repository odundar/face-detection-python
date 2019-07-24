# MIT License
#
# Copyright (c) 2019 Onur Dundar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Using Multiple Models

This sample application implemented to guide engineers who works with OpenVINO(TM) Toolkit more familiar
with using pre-trained models from OpenVINO(TM) model zoo and custom models trained by engineers itself.

We intend to show all details of Python API to ease the development process.
"""

# TODO: Dynamic Batch Support for Multiple Faces ?
# TODO: Async Execution for GA Model
# TODO: INT8 Tests

import sys, time
import cv2 as cv
import numpy as np
import json

# Import OpenVINO
# Make sure environment variables set correctly for this to work
# Check on README.md file
from openvino.inference_engine import IENetwork, IEPlugin


def prepare_openvino(model_path,
                     model_name,
                     target_device="CPU",
                     plugin_libraries_dir='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/',
                     cpu_extension='/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so'):
    """
    Method used to load given IR files using the target device plugin
    :param model_path: Full path to model file, need to end with /
    :param model_name: name of model, without .xml or .bin extension
    :param target_device: CPU or GPU or MYRIAD etc.
    :param plugin_libraries_dir: Paths to Inference Engine Library Paths
    :param cpu_extension: Path to CPU Extension path
    :return: True/False
    """

    if model_path is None or model_name is None:
        return None

    # Model File Paths
    model_file = model_path + model_name + '.xml'
    model_weights = model_path + model_name + '.bin'

    # Load Networks
    try:
        network = IENetwork(model=model_file, weights=model_weights)
    except FileNotFoundError:
        print(FileNotFoundError.strerror, " ", FileNotFoundError.filename)
        exit(-1)

    # Load Plugin
    try:
        plugin = IEPlugin(device=target_device, plugin_dirs=plugin_libraries_dir)
    except OSError:
        print(OSError.strerror)
        exit(-1)

    # Loading Custom Extension Libraries if required
    if target_device.find("CPU") != -1:
        plugin.add_cpu_extension(cpu_extension)

    # Input / Output Memory Allocations to feed input or get output values
    input_blobs = next(iter(network.inputs))
    output_blobs = next(iter(network.outputs))

    # Load
    try:
        openvino_net = plugin.load(network=network)
    except OSError:
        print(OSError.strerror)
        exit(-1)

    return input_blobs, output_blobs, openvino_net, network, plugin


# Print Frame Features required to be extended
def print_frame_features(frame):
    print('Image Shape: ', frame.shape)
    return None


def prepare_input_image(original_frame, network, input_blobs):
    """
    Resize, transform input image as required by deep learning model.
    :param original_frame: ndarray , cv image
    :param network: CNNNetwork
    :param input_blob: input layer definition
    :return:
    """

    n, c, h, w = network.inputs[input_blobs].shape
    # print('N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)
    r_frame = cv.resize(original_frame, (w, h))
    r_frame = cv.cvtColor(r_frame, cv.COLOR_BGR2RGB)
    r_frame = np.transpose(r_frame,(2, 0, 1))
    r_frame = np.expand_dims(r_frame, axis=0)

    return r_frame


def draw_rectangle(frame, x1, y1, x2, y2, normalized=True):
    """
    Draw Rectangle with given Normalized
    :param frame:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param normalized:
    :return:
    """
    if normalized:
        h = frame.shape[0]
        w = frame.shape[1]

        x1 = int(x1 * w)
        x2 = int(x2 * w)

        y1 = int(y1 * h)
        y2 = int(y2 * h)

    cv.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

def draw_rectangles(frame, coordinates, normalized=True):
    """
    Draw Rectangles with given Normalized
    :param frame:
    :param coordinates:
    :param normalized:
    :return:
    """
    for coordinate in coordinates:
        x1 = coordinate[0]
        y1 = coordinate[1]
        x2 = coordinate[2]
        y1 = coordinate[3]

        if normalized:
            h = frame.shape[0]
            w = frame.shape[1]

            x1 = int(x1 * w)
            x2 = int(x2 * w)

            y1 = int(y1 * h)
            y2 = int(y2 * h)

        cv.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)



def draw_text(frame, text, x1, y1, x2, y2, normalized=True):
    """
    Draw text with cv.puttext method
    :param frame:
    :param text:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param normalized:
    :return:
    """
    if normalized:
        h = frame.shape[0]
        w = frame.shape[1]

        x1 = int(x1 * w)
        x2 = int(x2 * w)

        y1 = int(y1 * h)
        y2 = int(y2 * h)

    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x2, y1+10)
    fontScale = 0.5
    fontColor = draw_color
    lineType = 2

    cv.putText(frame,
               text,
               bottomLeftCornerOfText,
               font,
               fontScale,
               fontColor,
               lineType)


def crop_frame(frame, x1, y1, x2, y2, normalized=True):
    """
    Crop Frame as Given Coordinates
    :param frame:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param normalized:
    :return:
    """
    if normalized:
        h = frame.shape[0]
        w = frame.shape[1]

        x1 = int(x1 * w)
        x2 = int(x2 * w)

        y1 = int(y1 * h)
        y2 = int(y2 * h)

    return frame[y1:y2, x1:x2]


def run_app():
    """
    Runs Face Detection Application
    """
    facedetection_models = prepare_openvino(face_detection_model_path, face_detection_model_name, target_device=inference_device)

    '''Get Face Detection Data'''
    face_input_blobs = facedetection_models[0]
    face_output_blobs = facedetection_models[1]
    facedetection_executable = facedetection_models[2]
    facedetection_network = facedetection_models[3]

    n, c, h, w = facedetection_network.inputs[face_input_blobs].shape
    print('Face Detection Input Shape: ','N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)

    '''Get Age/Gender Detection Data'''
    if run_ga:
        agedetection_model = prepare_openvino(age_gender_model_path, age_gender_model_name, target_device=ag_inference_device)
        ag_input_blobs = agedetection_model[0]
        ag_output_blobs = agedetection_model[1]
        ag_executable = agedetection_model[2]
        ag_network = agedetection_model[3]

        n, c, h, w = ag_network.inputs[ag_input_blobs].shape
        print('Age/Gender Detection Input Shape: ', 'N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)

    if facedetection_models is None:
        return

    '''Open Web Cam (change 0 to any video file if required)'''
    capture = cv.VideoCapture(input_path)

    has_frame, frame = capture.read()

    print_frame_features(frame)

    # Performance Metrics Counters
    TotalFaceInferenceElapsedTime = 0.0
    FaceInferenceFrameCount = 0.0

    # TODO: Batch Faces Together with Dynamic Batching
    TotalAGInferenceElapsedTime = 0.0
    AGInferenceFrameCount = 0.0

    '''Read Frames from Video Cam'''
    while has_frame:
        FaceInferenceFrameCount += 1

        start_fi = time.time()

        resized_frame = prepare_input_image(frame, facedetection_network, face_input_blobs)
        # Run inference
        facedetection_executable.infer(inputs={face_input_blobs: resized_frame})
        # Get Detections
        detections = facedetection_executable.requests[0].outputs[face_output_blobs]

        end_fi = time.time()
        TotalFaceInferenceElapsedTime += (end_fi - start_fi)

        for detection_vector in detections[0][0]:
            # If face detected with over confidence
            if detection_vector[2] > face_detection_threshold:
                # print(detection_vector)
                if run_ga:
                    # Crop image if face detected and feed to age/gender model
                    cropped_image = crop_frame(frame, detection_vector[3], detection_vector[4], detection_vector[5], detection_vector[6])

                    if cropped_image.size != 0:
                        AGInferenceFrameCount += 1
                        # face_image = prepare_input_image(cropped_image, ag_network, ag_input_blobs)
                        start_fi = time.time()

                        r_frame = prepare_input_image(cropped_image, ag_network, ag_input_blobs)
                        # cv.imwrite("frame%d.jpg" % AGInferenceFrameCount, r_frame)
                        # Start Inference
                        ag_executable.infer(inputs={ag_input_blobs: r_frame})
                        ag_detections = ag_executable.requests[0].outputs[ag_output_blobs]

                        # Parse detection vector to get age and gender
                        g = ag_detections[:,0:2].flatten()
                        gender = int(np.argmax(g))

                        gender_text = 'female'
                        if gender == 1:
                            gender_text = 'male'

                        a = ag_detections[:,2:202].reshape((100, 2))
                        a = np.argmax(a, axis=1)
                        age = int(sum(a))

                        end_fi = time.time()
                        TotalAGInferenceElapsedTime += (end_fi - start_fi)

                        text = '{}, {}'.format(age, gender_text)

                        # Draw gender-age text
                        draw_rectangle(frame, detection_vector[3], detection_vector[4], detection_vector[5], detection_vector[6])
                        draw_text(frame, text, detection_vector[3], detection_vector[4], detection_vector[5], detection_vector[6])
                else:
                    draw_rectangle(frame, detection_vector[3], detection_vector[4], detection_vector[5],detection_vector[6])

        cv.imshow(cv_window_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        has_frame, frame = capture.read()

    print('Inference Per Image for Face Detection: ', (TotalFaceInferenceElapsedTime / FaceInferenceFrameCount), " Seconds")
    print('Inference Per Image for Age/Gender Detection: ', (TotalAGInferenceElapsedTime / AGInferenceFrameCount), " Seconds")

    return None


'''
We don't deal with parameters, hard coded arguments.
'''
cv_window_name = 'Face Detection'
face_detection_threshold = 0.99
run_ga = True
# Model is pre-trained model from Model Zoo: https://github.com/opencv/open_model_zoo/tree/master/intel_models/face-detection-adas-0001
face_detection_model_path = '/home/intel/openvino_models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/'
face_detection_model_name = 'face-detection-adas-0001'
age_gender_model_path = '/home/intel/Downloads/gamodel-r50/FP32/'
age_gender_model_name = 'model-0000'
draw_color = (0, 125, 255)

input_type = "file"
input_path = "/home/intel/Videos/facedetection.mp4"

inference_device = "CPU"


def parse_config_file(config_json='config.json'):
    """
    Parse Config File
    :param config_json:
    :return:
    """

    with open(config_json) as json_file:
        data = json.load(json_file)

        global cv_window_name
        cv_window_name = data['window_name']

        global face_detection_threshold
        face_detection_threshold = float(data['face_detection_threshold'])

        global run_ga
        if data['run_ga'] is "False":
            run_ga = False

        global face_detection_model_path
        face_detection_model_path = data["face_detection_model_path"]

        global face_detection_model_name
        face_detection_model_name = data["face_detection_model_name"]

        global age_gender_model_path
        age_gender_model_path = data["age_gender_model_path"]

        global age_gender_model_name
        age_gender_model_name = data["age_gender_model_name"]

        global input_path
        input_path = data["input_path"]

        global input_type
        input_type = data["input_type"]

        global inference_device
        inference_device = data["face_detection_inference_device"]

        global ag_inference_device
        ag_inference_device = data["age_gender_inference_device"]


def help():
    print('Usage: python3 face_detection_openvino.py <config_file.json>')


# Application Entry Point
if __name__ == "__main__":
    if len(sys.argv) is not 2:
        help()
        exit(-1)

    parse_config_file(sys.argv[1])

    run_app()
