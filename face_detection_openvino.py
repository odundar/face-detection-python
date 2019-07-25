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
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork


class ImageUtil(object):
    @staticmethod
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

    @staticmethod
    def draw_text(frame, text, x1, y1, x2, y2, line_color=(0,255,124),normalized=True):
        """
        Draw text with cv.puttext method
        :param frame:
        :param text:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param line_color:
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
        bottomLeftCornerOfText = (x2, y1 + 10)
        fontScale = 0.5
        fontColor = line_color
        lineType = 2

        cv.putText(frame,
                   text,
                   bottomLeftCornerOfText,
                   font,
                   fontScale,
                   fontColor,
                   lineType)

    @staticmethod
    def draw_rectangles(frame, coordinates, line_color=(0,255,124), normalized=True):
        """
        Draw Rectangles with given Normalized
        :param frame:
        :param coordinates:
        :param line_color:
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

            cv.rectangle(frame, (x1, y1), (x2, y2), line_color, 2)

    @staticmethod
    def draw_rectangle(frame, x1, y1, x2, y2, line_color=(0,255,124), normalized=True):
        """
        Draw Rectangle with given Normalized
        :param frame:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param line_color:
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

        cv.rectangle(frame, (x1, y1), (x2, y2), line_color, 2)


class InferenceConfig(object):
    ModelPath = str()
    ModelName = str()
    TargetDevice = str()
    Async = False
    BatchSize = 1
    CpuExtension = True
    CpuExtensionPath = "/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"


class FaceInferenceConfig(InferenceConfig):
    FaceDetectionThreshold = 1.0


class AgeGenderInferenceConfig(InferenceConfig):
    RunAgeGenderDetection = False


class OpenVINOInferenceBase(object):
    Config = InferenceConfig()
    OpenVinoIE = IECore()
    OpenVinoNetwork = IENetwork()
    OpenVinoExecutable = ExecutableNetwork()

    InputLayer = str()
    OutputLayer = str()

    ElapsedInferenceTime = 0.0
    InferenceCount = 0.0

    InputShape = None
    OutputShape = None

    def __init__(self, infer_config):
        self.Config = infer_config
        self.prepare_detector()

    def prepare_detector(self):
        if self.Config.ModelPath is None or self.Config.ModelName is None:
            return None

        # Model File Paths
        model_file = self.Config.ModelPath + self.Config.ModelName + '.xml'
        model_weights = self.Config.ModelPath + self.Config.ModelName + '.bin'

        self.OpenVinoIE = IECore()

        if self.Config.CpuExtension and 'CPU' in self.Config.TargetDevice:
            self.OpenVinoIE.add_extension(self.Config.CpuExtensionPath, "CPU")

        try:
            self.OpenVinoNetwork = IENetwork(model=model_file, weights=model_weights)
        except FileNotFoundError:
            print(FileNotFoundError.strerror, " ", FileNotFoundError.filename)
            exit(-1)

        if "CPU" in self.Config.TargetDevice:
            supported_layers = self.OpenVinoIE.query_network(self.OpenVinoNetwork, "CPU")
            not_supported_layers = [l for l in self.OpenVinoNetwork.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                print("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.Config.TargetDevice, ', '.join(not_supported_layers)))
                print(
                    "Please try to specify cpu extensions library path in config.json file ")

        # Input / Output Memory Allocations to feed input or get output values
        self.InputLayer = next(iter(self.OpenVinoNetwork.inputs))
        self.OutputLayer = next(iter(self.OpenVinoNetwork.outputs))

        self.InputShape = self.OpenVinoNetwork.inputs[self.InputLayer].shape
        n, c, h, w = self.OpenVinoNetwork.inputs[self.InputLayer].shape
        print('Input Shape: ', 'N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)

        self.OutputShape = self.OpenVinoNetwork.outputs[self.OutputLayer].shape

        self.OpenVinoExecutable = self.OpenVinoIE.load_network(network=self.OpenVinoNetwork, device_name=self.Config.TargetDevice)

        return None

    def infer(self, inputs):
        self.InferenceCount += 1
        start = time.time()

        # Resize
        n, c, h, w = self.OpenVinoNetwork.inputs[self.InputLayer].shape
        # print('N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)
        r_frame = cv.resize(inputs, (w, h))
        r_frame = cv.cvtColor(r_frame, cv.COLOR_BGR2RGB)
        r_frame = np.transpose(r_frame, (2, 0, 1))
        r_frame = np.expand_dims(r_frame, axis=0)

        res = self.OpenVinoExecutable.infer(inputs={self.InputLayer: r_frame})

        end = time.time()
        self.ElapsedInferenceTime += (end - start)

        return res[self.OutputLayer]

    def print_inference_performance_metrics(self):
        print('Average Inference Time Per Request : {}'.format(self.ElapsedInferenceTime / self.InferenceCount))


class FaceDetectionInference(OpenVINOInferenceBase):

    def get_faces(self, images):
        """
        Get Detected Face Coordinates
        :param images:
        :return:
        """
        res = self.infer(images)

        face_coordinates = []

        for r in res[0][0]:
            if r[2] > self.Config.FaceDetectionThreshold:
                face_coordinates.append([r[3], r[4], r[5], r[6]])

        return face_coordinates


class AgeGenderDetectionInference(OpenVINOInferenceBase):

    def get_age_gender(self, images):

        detection = self.infer(images)
        # Parse detection vector to get age and gender
        gender_vector = detection[:, 0:2].flatten()
        gender = int(np.argmax(gender_vector))

        gender_text = 'female'
        if gender == 1:
            gender_text = 'male'

        age_matrix = detection[:, 2:202].reshape((100, 2))
        ages = np.argmax(age_matrix, axis=1)
        age = int(sum(ages))

        return age, gender_text

# TODO: Async, Multi-Batch Implementation


def run_app():

    """
    Runs Face Detection Application
    """
    face_infer_cfg = FaceInferenceConfig()
    face_infer_cfg.ModelPath = face_detection_model_path
    face_infer_cfg.ModelName = face_detection_model_name
    face_infer_cfg.TargetDevice = face_detection_inference_device
    face_infer_cfg.Async = face_detection_async
    face_infer_cfg.BatchSize = face_detection_batch_size
    face_infer_cfg.CpuExtension = face_detection_cpu_extension
    face_infer_cfg.CpuExtensionPath = face_detection_cpu_extension_path
    face_infer_cfg.FaceDetectionThreshold = face_detection_threshold

    face_infer = FaceDetectionInference(face_infer_cfg)

    '''Get Age/Gender Detection Data'''
    if run_age_gender:
        age_gender_infer_cfg = AgeGenderInferenceConfig()
        age_gender_infer_cfg.ModelPath = age_gender_model_path
        age_gender_infer_cfg.ModelName = age_gender_model_name
        age_gender_infer_cfg.TargetDevice = age_gender_inference_device
        age_gender_infer_cfg.Async = age_gender_async
        age_gender_infer_cfg.BatchSize = age_gender_batch_size
        age_gender_infer_cfg.CpuExtension = age_gender_cpu_extension
        age_gender_infer_cfg.CpuExtensionPath = age_gender_cpu_extension_path
        age_gender_infer_cfg.RunAgeGenderDetection = run_age_gender

        age_gender_infer = AgeGenderDetectionInference(age_gender_infer_cfg)

    '''Open Web Cam (change 0 to any video file if required)'''
    capture = cv.VideoCapture(input_path)
    has_frame, frame = capture.read()

    '''Read Frames from Video Cam'''
    while has_frame:

        faces = face_infer.get_faces(frame)

        for face in faces:
            age_gender_text = 'Age: {} - Gender: {}'
            if run_age_gender:
                cropped_image = ImageUtil.crop_frame(frame, face[0], face[1], face[2], face[3])
                age, gender = age_gender_infer.get_age_gender(cropped_image)
                age_gender_text = age_gender_text.format(age, gender)
            else:
                age_gender_text = age_gender_text.format('None', 'None')

            ImageUtil.draw_rectangle(frame, face[0], face[1], face[2], face[3])
            ImageUtil.draw_text(frame, age_gender_text, face[0], face[1], face[2], face[3])

        cv.imshow(cv_window_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        has_frame, frame = capture.read()

    face_infer.print_inference_performance_metrics()
    if run_age_gender:
        age_gender_infer.print_inference_performance_metrics()

    return None


"""
Global Parameters Used for Application Configuration
"""

cv_window_name = 'Face Detection'

input_type = "webcam"
input_path = "0"

face_detection_model_path = '/home/intel/openvino_models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/'
face_detection_model_name = 'face-detection-adas-0001'
face_detection_threshold = 0.99
face_detection_inference_device = "CPU"
face_detection_async = False
face_detection_cpu_extension = False
face_detection_cpu_extension_path = "/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"
face_detection_batch_size = 1
face_detection_dynamic_batch = False

run_age_gender = True
age_gender_model_path = '/home/intel/Downloads/gamodel-r50/FP32/'
age_gender_model_name = 'model-0000'
age_gender_inference_device = "CPU"
age_gender_cpu_extension = False
age_gender_cpu_extension_path = "/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"
age_gender_async = False
age_gender_dynamic_batch = False
age_gender_batch_size = 1


def parse_config_file(config_json='config.json'):
    """
    Parse Config File
    :param config_json:
    :return:
    """

    with open(config_json) as json_file:
        data = json.load(json_file)

        global cv_window_name
        cv_window_name = data['output_window_name']

        global input_path
        input_path = data["input_path"]

        global input_type
        input_type = data["input_type"]

        #################################################################

        global face_detection_model_path
        face_detection_model_path = data["face_detection_model_path"]

        global face_detection_model_name
        face_detection_model_name = data["face_detection_model_name"]

        global face_detection_threshold
        face_detection_threshold = float(data['face_detection_threshold'])

        global face_detection_inference_device
        face_detection_inference_device = data["face_detection_inference_device"]

        global face_detection_async
        if data["face_detection_async"] == "True":
            face_detection_async = True

        global face_detection_cpu_extension
        if data["face_detection_cpu_extension"] == "True":
            face_detection_cpu_extension = True

        global face_detection_cpu_extension_path
        face_detection_cpu_extension_path = data["face_detection_cpu_extension_path"]

        global face_detection_dynamic_batch
        if data["face_detection_dynamic_batch"] == "True":
            face_detection_dynamic_batch = True

        global face_detection_batch_size
        face_detection_batch_size = int(data['face_detection_batch_size'])

        #################################################################

        global run_age_gender
        if data['run_age_gender'] is "False":
            run_age_gender = False

        global age_gender_model_path
        age_gender_model_path = data["age_gender_model_path"]

        global age_gender_model_name
        age_gender_model_name = data["age_gender_model_name"]

        global age_gender_inference_device
        age_gender_inference_device = data["age_gender_inference_device"]

        global age_gender_cpu_extension
        if data["age_gender_cpu_extension"] == "True":
            age_gender_cpu_extension = True

        global age_gender_cpu_extension_path
        age_gender_cpu_extension_path = data["age_gender_cpu_extension_path"]

        global age_gender_async
        if data["age_gender_async"] == "True":
            age_gender_async = True

        global age_gender_dynamic_batch
        if data["age_gender_dynamic_batch"] == "True":
            age_gender_dynamic_batch = True

        global age_gender_batch_size
        age_gender_batch_size = int(data["age_gender_batch_size"])


def help():
    print('Usage: python3 face_detection_openvino.py <config_file.json>')


# Application Entry Point
if __name__ == "__main__":

    if len(sys.argv) is not 2:
        help()
        parse_config_file()
    else:
        parse_config_file(sys.argv[1])

    run_app()