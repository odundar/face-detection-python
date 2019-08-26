# MIT License
#
# Copyright (c) 2019 Onur Dundar onur.dundar1@gmail.com
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

import logging
import math
import json
import time
import cv2 as cv
import numpy as np
from PIL import Image

from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork
from .detection_base_ov import InferenceConfig, InferenceBase


class FaceDetectionModelTypes:
    """
    Face Detection Model Type to Be Used
    """
    # MTCNN Face Detection
    MTCNN = 'mtcnn_facedetection'
    # Open Model Zoo Face Detection
    OPENMODELZOO = 'omz_facedetection'


class FaceDetectionConfig(InferenceConfig):
    """
    Face Detection Module Configurations based on Open Model Zoo Face Detection Model
    """
    ModelType = FaceDetectionModelTypes.OPENMODELZOO
    FaceDetectionThreshold = 1.0
    InputHeight = 720
    InputWidth = 1080

    def parse_json(self, json_file):
        try:
            logging.log(logging.INFO, "Loading JSON File {}".format(json_file))
            logging.log(logging.INFO, "Model Type {}".format(self.ModelType))

            with open(json_file) as json_file:
                data = json.load(json_file)

                self.ModelPath = data[self.ModelType]["model_path"]
                self.ModelName = data[self.ModelType]["model_name"]

                self.TargetDevice = data[self.ModelType]["target_device"]

                self.FaceDetectionThreshold = data[self.ModelType]["face_detection_threshold"]

                if data[self.ModelType]["async"] == "True":
                    self.Async = True

                self.RequestCount = int(data[self.ModelType]["request_count"])

                self.BatchSize = int(data[self.ModelType]["batch_size"])

                if data[self.ModelType]["cpu_extension"] == "True":
                    self.CpuExtension = True

                self.CpuExtensionPath = data[self.ModelType]["cpu_extension_path"]

                if data[self.ModelType]["dynamic_batch"] == "True":
                    self.DynamicBatch = True

                if data[self.ModelType]["limit_cpu_threads"] == "True":
                    self.LimitCPUThreads = True

                self.CPUThreadNum = int(data[self.ModelType]["number_of_cpu_threads"])

                if data[self.ModelType]["bind_cpu_threads"] == "True":
                    self.LimitCPUThreads = True

                self.CPUStream = data[self.ModelType]["cpu_stream"]

        except FileNotFoundError:
            logging.log(logging.ERROR, '{} FileNotFound'.format(json_file))
            exit(-1)

    def read_dict(self, data=None):
        """
        Used When JSON Already Parsed as Dict
        :return:
        """
        if data is None:
            logging.log(logging.ERROR, "No Parameters Passed")
            exit(-1)

        self.ModelPath = data[self.ModelType]["model_path"]
        self.ModelName = data[self.ModelType]["model_name"]

        self.TargetDevice = data[self.ModelType]["target_device"]

        self.FaceDetectionThreshold = data[self.ModelType]["face_detection_threshold"]

        if data[self.ModelType]["async"] == "True":
            self.Async = True

        self.RequestCount = int(data[self.ModelType]["request_count"])

        self.BatchSize = int(data[self.ModelType]["batch_size"])

        if data[self.ModelType]["cpu_extension"] == "True":
            self.CpuExtension = True

        self.CpuExtensionPath = data[self.ModelType]["cpu_extension_path"]

        if data[self.ModelType]["dynamic_batch"] == "True":
            self.DynamicBatch = True

        if data[self.ModelType]["limit_cpu_threads"] == "True":
            self.LimitCPUThreads = True

        self.CPUThreadNum = int(data[self.ModelType]["number_of_cpu_threads"])

        if data[self.ModelType]["bind_cpu_threads"] == "True":
            self.LimitCPUThreads = True

        self.CPUStream = data[self.ModelType]["cpu_stream"]


class OpenMZooFaceDetection(InferenceBase):
    """
    Face Detection Module Configured to Get Results using Open Model Zoo Face Detection Model
    """
    Config = FaceDetectionConfig()

    def __init__(self, config=FaceDetectionConfig()):
        super(OpenMZooFaceDetection, self).__init__(config)
        self.Config = config

    def get_face_detection_data(self, request_id=0):
        """
        Parse Face Detection Output
        :param output_layer:
        :param request_id:
        :return: face coordinates
        """
        face_coordinates = []

        detections = self.get_results(self.OutputLayer, request_id)[0][0]

        logging.log(logging.INFO, "Fetched Face Detection Results")

        for detection in detections:
            if detection[2] > self.Config.FaceDetectionThreshold:
                face_coordinates.append([detection[3], detection[4], detection[5], detection[6]])

        logging.log(logging.INFO, "Number of Detected Faces: {}".format(len(face_coordinates)))
        return face_coordinates


class MTCNNFaceDetectionConfig(InferenceConfig):
    """
    Face Detection Module Configurations based on MTCNN Face Detection Model
    """
    ModelType = FaceDetectionModelTypes.MTCNN

    InputHeight = 720
    InputWidth = 1080

    PNetworkThreshold = 0.6
    RNetworkThreshold = 0.7
    ONetworkThreshold = 0.8

    NMSThresholds = [0.7, 0.7, 0.7]
    MinDetectionSize = 12
    Factor = 0.707

    MinimumFaceSize = 15.0
    MinLength = 720
    FactorCount = 0

    RInputBatchSize = 128
    OInputBatchSize = 128

    PModelFileName = "det1-0001"
    RModelFileName = "det2-0001"
    OModelFileName = "det3-0001"

    def parse_json(self, json_file):
        try:
            logging.log(logging.INFO, "Loading JSON File {}".format(json_file))
            logging.log(logging.INFO, "Model Type {}".format(self.ModelType))
            with open(json_file) as json_file:
                data = json.load(json_file)

                self.ModelPath = data[self.ModelType]["model_path"]
                self.PModelFileName = data[self.ModelType]["p_model_file_name"]
                self.RModelFileName = data[self.ModelType]["r_model_file_name"]
                self.OModelFileName = data[self.ModelType]["o_model_file_name"]

                self.TargetDevice = data[self.ModelType]["target_device"]

                if data[self.ModelType]["cpu_extension"] == "True":
                    self.CpuExtension = True

                self.CpuExtensionPath = data[self.ModelType]["cpu_extension_path"]

                self.PNetworkThreshold = float(data[self.ModelType]["p_network_threshold"])
                self.RNetworkThreshold = float(data[self.ModelType]["r_network_threshold"])
                self.ONetworkThreshold = float(data[self.ModelType]["o_network_threshold"])

                self.MinimumFaceSize = float(data[self.ModelType]["minimum_face_size"])
                self.MinLength = float(data[self.ModelType]["minimum_length"])
                self.FactorCount = float(data[self.ModelType]["factor_count"])
                self.Factor = float(data[self.ModelType]["factor"])
                self.MinDetectionSize = int(data[self.ModelType]["min_detection_size"])

                self.NMSThresholds = list(data[self.ModelType]["nms_thresholds"])

                self.RInputBatchSize = int(data[self.ModelType]["r_input_batch_size"])
                self.OInputBatchSize = int(data[self.ModelType]["o_input_batch_size"])

                if data[self.ModelType]["limit_cpu_threads"] == "True":
                    self.LimitCPUThreads = True
                self.CPUThreadNum = int(data[self.ModelType]["number_of_cpu_threads"])
                if data[self.ModelType]["bind_cpu_threads"] == "True":
                    self.LimitCPUThreads = True
                self.CPUStream = data[self.ModelType]["cpu_stream"]

        except FileNotFoundError:
            logging.log(logging.ERROR, '{} FileNotFound'.format(json_file))
            exit(-1)

    def read_dict(self, data=None):
        """
        Used When JSON Already Parsed as Dict
        :return:
        """
        if data is None:
            logging.getLogger(name="face_detection").log(logging.ERROR, "No Parameters Passed")
            exit(-1)

        self.ModelPath = data[self.ModelType]["model_path"]
        self.PModelFileName = data[self.ModelType]["p_model_file_name"]
        self.RModelFileName = data[self.ModelType]["r_model_file_name"]
        self.OModelFileName = data[self.ModelType]["o_model_file_name"]

        self.TargetDevice = data[self.ModelType]["target_device"]

        if data[self.ModelType]["cpu_extension"] == "True":
            self.CpuExtension = True

        self.CpuExtensionPath = data[self.ModelType]["cpu_extension_path"]

        self.PNetworkThreshold = float(data[self.ModelType]["p_network_threshold"])
        self.RNetworkThreshold = float(data[self.ModelType]["r_network_threshold"])
        self.ONetworkThreshold = float(data[self.ModelType]["o_network_threshold"])

        self.MinimumFaceSize = float(data[self.ModelType]["minimum_face_size"])
        self.MinLength = float(data[self.ModelType]["minimum_length"])
        self.FactorCount = float(data[self.ModelType]["factor_count"])
        self.Factor = float(data[self.ModelType]["factor"])
        self.MinDetectionSize = int(data[self.ModelType]["min_detection_size"])

        self.NMSThresholds = list(data[self.ModelType]["nms_thresholds"])

        self.RInputBatchSize = int(data[self.ModelType]["r_input_batch_size"])
        self.OInputBatchSize = int(data[self.ModelType]["o_input_batch_size"])

        if data[self.ModelType]["limit_cpu_threads"] == "True":
            self.LimitCPUThreads = True
        self.CPUThreadNum = int(data[self.ModelType]["number_of_cpu_threads"])
        if data[self.ModelType]["bind_cpu_threads"] == "True":
            self.LimitCPUThreads = True
        self.CPUStream = data[self.ModelType]["cpu_stream"]


class MtCNNFaceDetection(InferenceBase):

    Config = MTCNNFaceDetectionConfig()

    OpenVinoExecutablesP = list()
    OpenVinoExecutableR = ExecutableNetwork()
    OpenVinoExecutableO = ExecutableNetwork()

    OpenVinoNetworkP = IENetwork()
    OpenVinoNetworkR = IENetwork()
    OpenVinoNetworkO = IENetwork()

    Scales = []

    RINPUT = []
    OINPUT = []

    LastFaceDetections = []
    LastLandmarkDetections = []

    InputLayerP = str()
    InputLayerR = str()
    InputLayerO = str()

    OutputLayersP = list()
    OutputLayersR = list()
    OutputLayersO = list()

    InputShapeP = []
    InputShapeR = []
    InputShapeO = []

    def __init__(self, config=MTCNNFaceDetectionConfig()):
        super(MtCNNFaceDetection, self).__init__(config)
        self.Config = config

    def prepare_detector(self):
        """
        Override Base Class Since MTCNN works with three different model
        :return: None
        """

        if self.Config.ModelPath is None or self.Config.ModelName is None:
            return None

        logging.log(logging.INFO, "Setting Up R - O Network Input Storage")
        self.RINPUT = np.zeros(dtype=float, shape=(self.Config.RInputBatchSize, 3, 24, 24))
        self.OINPUT = np.zeros(dtype=float, shape=(self.Config.OInputBatchSize, 3, 48, 48))

        self.OpenVinoIE = IECore()

        if self.Config.CpuExtension and 'CPU' in self.Config.TargetDevice:
            logging.log(logging.INFO, "CPU Extensions Added")
            self.OpenVinoIE.add_extension(self.Config.CpuExtensionPath, "CPU")

        try:
            # Model File Paths
            model_file = self.Config.ModelPath + self.Config.PModelFileName + ".xml"
            model_weights = self.Config.ModelPath + self.Config.PModelFileName + ".bin"
            logging.log(logging.INFO, "Loading Models File {}".format(model_file))
            logging.log(logging.INFO, "Loading Weights File {}".format(model_weights))

            self.OpenVinoNetworkP = IENetwork(model=model_file, weights=model_weights)
            logging.log(logging.INFO, "Loading P Network")

            model_file = self.Config.ModelPath + self.Config.RModelFileName + ".xml"
            model_weights = self.Config.ModelPath + self.Config.RModelFileName + ".bin"
            logging.log(logging.INFO, "Loading Models File {}".format(model_file))
            logging.log(logging.INFO, "Loading Weights File {}".format(model_weights))

            self.OpenVinoNetworkR = IENetwork(model=model_file, weights=model_weights)
            self.OpenVinoNetworkR.batch_size = self.Config.RInputBatchSize
            logging.log(logging.INFO, "Loading R Network")

            model_file = self.Config.ModelPath + self.Config.OModelFileName + ".xml"
            model_weights = self.Config.ModelPath + self.Config.OModelFileName + ".bin"
            logging.log(logging.INFO, "Loading Models File {}".format(model_file))
            logging.log(logging.INFO, "Loading Weights File {}".format(model_weights))

            self.OpenVinoNetworkO = IENetwork(model=model_file, weights=model_weights)
            self.OpenVinoNetworkO.batch_size = self.Config.OInputBatchSize
            logging.log(logging.INFO, "Loading O Network")

        except FileNotFoundError:
            logging.log(logging.ERROR, FileNotFoundError.strerror, " ", FileNotFoundError.filename)
            exit(-1)

        if "CPU" in self.Config.TargetDevice:
            supported_layers = self.OpenVinoIE.query_network(self.OpenVinoNetworkP, "CPU")
            not_supported_layers = [l for l in self.OpenVinoNetworkP.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                logging.log(logging.INFO, "Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(self.Config.TargetDevice, ', '.join(not_supported_layers)))
                logging.log(logging.INFO, "Please try to specify cpu extensions library path in config.json file ")

        # Input / Output Memory Allocations to feed input or get output values
        self.InputLayerP = next(iter(self.OpenVinoNetworkP.inputs))
        self.InputLayerR = next(iter(self.OpenVinoNetworkP.inputs))
        self.InputLayerO = next(iter(self.OpenVinoNetworkP.inputs))

        self.OutputLayersP = list(self.OpenVinoNetworkP.outputs)
        self.OutputLayersR = list(self.OpenVinoNetworkR.outputs)
        self.OutputLayersO = list(self.OpenVinoNetworkO.outputs)

        self.InputShapeP = self.OpenVinoNetworkP.inputs[self.InputLayerP].shape
        self.InputShapeR = self.OpenVinoNetworkR.inputs[self.InputLayerR].shape
        self.InputShapeO = self.OpenVinoNetworkO.inputs[self.InputLayerO].shape

        # Enable Dynamic Batch By Default
        config = {"DYN_BATCH_ENABLED": "YES"}

        self.OpenVinoExecutableR = self.OpenVinoIE.load_network(network=self.OpenVinoNetworkR,
                                                                device_name=self.Config.TargetDevice,
                                                                config=config,
                                                                num_requests=self.Config.RequestCount)
        logging.log(logging.INFO, "Created R Network Executable")

        self.OpenVinoExecutableO = self.OpenVinoIE.load_network(network=self.OpenVinoNetworkO,
                                                                device_name=self.Config.TargetDevice,
                                                                config=config,
                                                                num_requests=self.Config.RequestCount)
        logging.log(logging.INFO, "Created O Network Executable")

        self.Config.MinLength = min(self.Config.InputHeight, self.Config.InputWidth)
        M = self.Config.MinDetectionSize / self.Config.MinimumFaceSize
        self.Config.MinLength *= M

        while self.Config.MinLength > self.Config.MinDetectionSize:
            scale = (M*self.Config.Factor**self.Config.FactorCount)
            self.Scales.append(scale)
            self.Config.MinLength *= self.Config.Factor
            self.Config.FactorCount += 1

            sw, sh = math.ceil(self.Config.InputWidth * scale), math.ceil(self.Config.InputHeight * scale)

            self.OpenVinoNetworkP.reshape({self.InputLayerP: (1, 3, sh, sw)})

            self.OpenVinoExecutablesP.append(self.OpenVinoIE.load_network(network=self.OpenVinoNetworkP,
                                                                          device_name=self.Config.TargetDevice,
                                                                          num_requests=self.Config.RequestCount))

        logging.log(logging.INFO, "Created Scaled P Networks {}".format(len(self.OpenVinoExecutablesP)))

    def run_mtcnn_face_detection(self, images, request_id=0):
        """
        Get Detected Face Coordinates
        :param images:
        :param request_id:
        :return:
        """
        self.InferenceCount += 1
        start_time = time.time()
        bounding_boxes = []
        landmarks = []

        cv_img = cv.cvtColor(images, cv.COLOR_BGR2RGB)
        image = Image.fromarray(cv_img)

        none_count = 0

        for i, scale in enumerate(self.Scales):
            width, height = image.size
            sw, sh = math.ceil(width * scale), math.ceil(height * scale)
            img = image.resize((sw, sh), Image.BILINEAR)
            img = np.asarray(img, 'float32')
            img = self.preprocess(img)

            output = self.OpenVinoExecutablesP[i].infer({self.InputLayerP: img})

            probs = output["prob1"][0, 1, :, :]
            offsets = output["conv4_2"]

            boxes = self.generate_bboxes(probs, offsets, scale, self.Config.PNetworkThreshold)

            if len(boxes) == 0:
                bounding_boxes.append(None)
                none_count += 1
            else:
                keep = self.nms(boxes[:, 0:5], overlap_threshold=0.5)
                bounding_boxes.append(boxes[keep])

        if len(bounding_boxes) > none_count:
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)
            keep = self.nms(bounding_boxes[:, 0:5], self.Config.NMSThresholds[0])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = self.calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            bounding_boxes = self.convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            img_boxes = self.get_image_boxes(bounding_boxes, image, size=24)

            if img_boxes.shape[0] > 0:
                shp = img_boxes.shape
                self.RINPUT[0:shp[0], ] = img_boxes
                self.OpenVinoExecutableR.requests[request_id].set_batch(shp[0])
                self.OpenVinoExecutableR.requests[request_id].infer({self.InputLayerR: self.RINPUT})

                offsets = self.OpenVinoExecutableR.requests[0].outputs['conv5_2'][:shp[0], ]
                probs = self.OpenVinoExecutableR.requests[0].outputs['prob1'][:shp[0]]

                keep = np.where(probs[:, 1] > self.Config.RNetworkThreshold)[0]
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
                offsets = offsets[keep]
                keep = self.nms(bounding_boxes, self.Config.NMSThresholds[1])
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes = self.calibrate_box(bounding_boxes, offsets[keep])
                bounding_boxes = self.convert_to_square(bounding_boxes)
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
                img_boxes = self.get_image_boxes(bounding_boxes, image, size=48)

                if img_boxes.shape[0] > 0:
                    shp = img_boxes.shape
                    self.OINPUT[0:shp[0], ] = img_boxes

                    self.OpenVinoExecutableO.requests[0].set_batch(shp[0])
                    self.OpenVinoExecutableO.requests[0].infer({self.InputLayerO: self.OINPUT})

                    landmarks = self.OpenVinoExecutableO.requests[0].outputs['conv6_3'][:shp[0]]
                    offsets = self.OpenVinoExecutableO.requests[0].outputs['conv6_2'][:shp[0]]
                    probs = self.OpenVinoExecutableO.requests[0].outputs['prob1'][:shp[0]]

                    keep = np.where(probs[:, 1] > self.Config.ONetworkThreshold)[0]
                    bounding_boxes = bounding_boxes[keep]
                    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
                    offsets = offsets[keep]
                    landmarks = landmarks[keep]
                    # compute landmark points
                    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
                    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
                    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
                    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
                    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
                    bounding_boxes = self.calibrate_box(bounding_boxes, offsets)
                    keep = self.nms(bounding_boxes, self.Config.NMSThresholds[2], mode='min')
                    bounding_boxes = bounding_boxes[keep]
                    landmarks = landmarks[keep]

        none_count = 0

        face_detections = []
        landmark_detections = []
        i = 0
        for box in bounding_boxes:
            if type(box) is type(None):
                none_count += 1
            else:
                scale = box[4]
                xmin = float((box[0] / scale) / self.Config.InputWidth)
                ymin = float((box[1] / scale) / self.Config.InputHeight)
                xmax = float((box[2] / scale) / self.Config.InputWidth)
                ymax = float((box[3] / scale) / self.Config.InputHeight)
                face_detections.append([xmin, ymin, xmax, ymax])
                lands = []
                for l in range(5):
                    lands.append(float((landmarks[i][l] / scale) / self.Config.InputWidth))
                    lands.append(float((landmarks[i][l + 5] / scale) / self.Config.InputHeight))

                landmark_detections.append(lands)
                i += 1

        if none_count == len(bounding_boxes):
            return [], []

        self.LastFaceDetections = face_detections
        self.LastLandmarkDetections = landmark_detections

        self.ElapsedInferenceTime += (time.time() - start_time)

    def infer(self, images, request_id=0):
        """
        Run inference
        :param images: image to get faces
        :param request_id: request id
        :return:
        """
        self.run_mtcnn_face_detection(images, request_id=0)

    def request_ready(self, request_id):
        """
        This is true by default since there is no ASYNC mode for MTCNN
        :param request_id:
        :return:
        """
        return True

    def get_face_detection_data(self, request_id=0):
        """
        Get Latest Results for Face Coordinates
        :param request_id:
        :return:
        """
        return self.LastFaceDetections

    def get_face_landmarks_data(self, request_id=0):
        """
        Get Latest Results for Landmark Coordinates
        :param request_id:
        :return:
        """
        return self.LastLandmarkDetections

    @staticmethod
    def preprocess(img):
        """Preprocessing step before feeding the network.

        Arguments:
            img: a float numpy array of shape [h, w, c].

        Returns:
            a float numpy array of shape [1, c, h, w].
        """
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = (img - 127.5) * 0.0078125
        return img

    @staticmethod
    def generate_bboxes(probs, offsets, scale, threshold):
        """Generate bounding boxes at places
        where there is probably a face.

        Arguments:
            probs: a float numpy array of shape [n, m].
            offsets: a float numpy array of shape [1, 4, n, m].
            scale: a float number,
                width and height of the image were scaled by this number.
            threshold: a float number.

        Returns:
            a float numpy array of shape [n_boxes, 9]
        """

        # applying P-Net is equivalent, in some sense, to
        # moving 12x12 window with stride 2
        stride = 2
        cell_size = 12

        # indices of boxes where there is probably a face
        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        # transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images
        # so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score, offsets
        ])
        # why one is added?

        return bounding_boxes.T

    @staticmethod
    def nms(boxes, overlap_threshold=0.5, mode='union'):
        """Non-maximum suppression.

        Arguments:
            boxes: a float numpy array of shape [n, 5],
                where each row is (xmin, ymin, xmax, ymax, score).
            overlap_threshold: a float number.
            mode: 'union' or 'min'.

        Returns:
            list with indices of the selected boxes
        """

        # if there are no boxes, return the empty list
        if len(boxes) == 0:
            return []

        # list of picked indices
        pick = []

        # grab the coordinates of the bounding boxes
        x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

        area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        ids = np.argsort(score)  # in increasing order

        while len(ids) > 0:

            # grab index of the largest value
            last = len(ids) - 1
            i = ids[last]
            pick.append(i)

            # compute intersections
            # of the box with the largest score
            # with the rest of boxes

            # left top corner of intersection boxes
            ix1 = np.maximum(x1[i], x1[ids[:last]])
            iy1 = np.maximum(y1[i], y1[ids[:last]])

            # right bottom corner of intersection boxes
            ix2 = np.minimum(x2[i], x2[ids[:last]])
            iy2 = np.minimum(y2[i], y2[ids[:last]])

            # width and height of intersection boxes
            w = np.maximum(0.0, ix2 - ix1 + 1.0)
            h = np.maximum(0.0, iy2 - iy1 + 1.0)

            # intersections' areas
            inter = w * h
            if mode == 'min':
                overlap = inter / np.minimum(area[i], area[ids[:last]])
            elif mode == 'union':
                # intersection over union (IoU)
                overlap = inter / (area[i] + area[ids[:last]] - inter)

            # delete all boxes where overlap is too big
            ids = np.delete(
                ids,
                np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
            )

        return pick

    @staticmethod
    def calibrate_box(bboxes, offsets):
        """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.

        Arguments:
            bboxes: a float numpy array of shape [n, 5].
            offsets: a float numpy array of shape [n, 4].

        Returns:
            a float numpy array of shape [n, 5].
        """
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        w = np.expand_dims(w, 1)
        h = np.expand_dims(h, 1)

        # this is what happening here:
        # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
        # x1_true = x1 + tx1*w
        # y1_true = y1 + ty1*h
        # x2_true = x2 + tx2*w
        # y2_true = y2 + ty2*h
        # below is just more compact form of this

        # are offsets always such that
        # x1 < x2 and y1 < y2 ?

        translation = np.hstack([w, h, w, h]) * offsets
        bboxes[:, 0:4] = bboxes[:, 0:4] + translation
        return bboxes

    @staticmethod
    def convert_to_square(bboxes):
        """Convert bounding boxes to a square form.

        Arguments:
            bboxes: a float numpy array of shape [n, 5].

        Returns:
            a float numpy array of shape [n, 5],
                squared bounding boxes.
        """

        square_bboxes = np.zeros_like(bboxes)
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = np.maximum(h, w)
        square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
        square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
        return square_bboxes

    @staticmethod
    def correct_bboxes(bboxes, width, height):
        """Crop boxes that are too big and get coordinates
        with respect to cutouts.

        Arguments:
            bboxes: a float numpy array of shape [n, 5],
                where each row is (xmin, ymin, xmax, ymax, score).
            width: a float number.
            height: a float number.

        Returns:
            dy, dx, edy, edx: a int numpy arrays of shape [n],
                coordinates of the boxes with respect to the cutouts.
            y, x, ey, ex: a int numpy arrays of shape [n],
                corrected ymin, xmin, ymax, xmax.
            h, w: a int numpy arrays of shape [n],
                just heights and widths of boxes.

            in the following order:
                [dy, edy, dx, edx, y, ey, x, ex, w, h].
        """

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
        num_boxes = bboxes.shape[0]

        # 'e' stands for end
        # (x, y) -> (ex, ey)
        x, y, ex, ey = x1, y1, x2, y2

        # we need to cut out a box from the image.
        # (x, y, ex, ey) are corrected coordinates of the box
        # in the image.
        # (dx, dy, edx, edy) are coordinates of the box in the cutout
        # from the image.
        dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
        edx, edy = w.copy() - 1.0, h.copy() - 1.0

        # if box's bottom right corner is too far right
        ind = np.where(ex > width - 1.0)[0]
        edx[ind] = w[ind] + width - 2.0 - ex[ind]
        ex[ind] = width - 1.0

        # if box's bottom right corner is too low
        ind = np.where(ey > height - 1.0)[0]
        edy[ind] = h[ind] + height - 2.0 - ey[ind]
        ey[ind] = height - 1.0

        # if box's top left corner is too far left
        ind = np.where(x < 0.0)[0]
        dx[ind] = 0.0 - x[ind]
        x[ind] = 0.0

        # if box's top left corner is too high
        ind = np.where(y < 0.0)[0]
        dy[ind] = 0.0 - y[ind]
        y[ind] = 0.0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
        return_list = [i.astype('int32') for i in return_list]

        return return_list

    @staticmethod
    def get_image_boxes(bounding_boxes, img, size=24):
        """Cut out boxes from the image.

        Arguments:
            bounding_boxes: a float numpy array of shape [n, 5].
            img: an instance of PIL.Image.
            size: an integer, size of cutouts.

        Returns:
            a float numpy array of shape [n, 3, size, size].
        """

        num_boxes = len(bounding_boxes)
        width, height = img.size

        [dy, edy, dx, edx, y, ey, x, ex, w, h] = MtCNNFaceDetection.correct_bboxes(bounding_boxes, width, height)
        img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

        for i in range(num_boxes):
            if h[i] <= 0 or w[i] <= 0:
                continue
            img_box = np.zeros((h[i], w[i], 3), 'uint8')

            img_array = np.asarray(img, 'uint8')
            img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
                img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

            # resize
            img_box = Image.fromarray(img_box)
            img_box = img_box.resize((size, size), Image.BILINEAR)
            img_box = np.asarray(img_box, 'float32')

            img_boxes[i, :, :, :] = MtCNNFaceDetection.preprocess(img_box)

        return img_boxes