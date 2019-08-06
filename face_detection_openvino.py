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

import sys, time
import cv2 as cv
import numpy as np
import json
import math

from PIL import Image

# Import OpenVINO
# Make sure environment variables set correctly for this to work
# Check on README.md file
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork


class ImageUtil(object):
    @staticmethod
    def crop_frame(frame, coordinate, normalized=True):
        """
        Crop Frame as Given Coordinates
        :param frame:
        :param coordinate:
        :param normalized:
        :return:
        """

        x1 = coordinate[0]
        y1 = coordinate[1]
        x2 = coordinate[2]
        y2 = coordinate[3]

        if normalized:
            h = frame.shape[0]
            w = frame.shape[1]

            x1 = int(x1 * w)
            x2 = int(x2 * w)

            y1 = int(y1 * h)
            y2 = int(y2 * h)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def draw_text(frame, text, coordinate, line_color=(0, 255, 124), normalized=True):
        """
        Draw text with cv.puttext method
        :param frame:
        :param text:
        :param coordinate:
        :param line_color:
        :param normalized:
        :return:
        """

        x1 = coordinate[0]
        y1 = coordinate[1]
        x2 = coordinate[2]
        y2 = coordinate[3]

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
    def draw_rectangles(frame, coordinates, line_color=(0, 255, 124), normalized=True):
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
            y2 = coordinate[3]

            if normalized:
                h = frame.shape[0]
                w = frame.shape[1]

                x1 = int(x1 * w)
                x2 = int(x2 * w)

                y1 = int(y1 * h)
                y2 = int(y2 * h)

            cv.rectangle(frame, (x1, y1), (x2, y2), line_color, 2)

    @staticmethod
    def draw_rectangle(frame, coordinate, line_color=(0, 255, 124), normalized=True):
        """
        Draw Rectangle with given Normalized
        :param frame:
        :param coordinate
        :param line_color:
        :param normalized:
        :return:
        """

        x1 = coordinate[0]
        y1 = coordinate[1]
        x2 = coordinate[2]
        y2 = coordinate[3]

        if normalized:
            h = frame.shape[0]
            w = frame.shape[1]

            x1 = int(x1 * w)
            x2 = int(x2 * w)

            y1 = int(y1 * h)
            y2 = int(y2 * h)

        cv.rectangle(frame, (x1, y1), (x2, y2), line_color, 2)

    @staticmethod
    def draw_ellipse(frame, coordinate, line_color=(124, 0, 0), normalized=True):
        """
        Draw Rectangle with given Normalized
        :param frame:
        :param coordinate
        :param line_color:
        :param normalized:
        :return:
        """

        x1 = coordinate[0]
        y1 = coordinate[1]

        if normalized:
            h = frame.shape[0]
            w = frame.shape[1]

            x1 = int(x1 * w)
            y1 = int(y1 * h)

        cv.circle(frame, (x1, y1), radius=1, color=line_color, thickness=1)


class InferenceConfig(object):
    ModelPath = str()
    ModelName = str()
    TargetDevice = str()
    Async = False
    AsyncReq = 1
    BatchSize = 1
    CpuExtension = True
    CpuExtensionPath = "/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"


class FaceInferenceConfig(InferenceConfig):
    FaceDetectionThreshold = 1.0
    InputHeight = 720
    InputWidth = 1080
    pass


class AgeGenderInferenceConfig(InferenceConfig):
    RunAgeGenderDetection = False
    pass


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
        n, c, h, w = self.InputShape
        print('Input Shape: ', 'N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)

        self.OutputShape = self.OpenVinoNetwork.outputs[self.OutputLayer].shape

        if self.Config.Async:
            self.OpenVinoExecutable = self.OpenVinoIE.load_network(network=self.OpenVinoNetwork,
                                                                   device_name=self.Config.TargetDevice,
                                                                   num_requests=self.Config.AsyncReq)
        else:
            self.OpenVinoExecutable = self.OpenVinoIE.load_network(network=self.OpenVinoNetwork,
                                                                   device_name=self.Config.TargetDevice)

        return None

    def infer(self, inputs, request_id=0):
        if self.Config.Async:
            self.infer_async(inputs, request_id)
        else:
            self.infer_sync(inputs)

    def infer_async(self, inputs, request_id=0):
        n, c, h, w = self.OpenVinoNetwork.inputs[self.InputLayer].shape

        # print('N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)
        r_frame = cv.resize(inputs, (w, h))
        r_frame = cv.cvtColor(r_frame, cv.COLOR_BGR2RGB)
        r_frame = np.transpose(r_frame, (2, 0, 1))
        r_frame = np.expand_dims(r_frame, axis=0)

        self.OpenVinoExecutable.requests[request_id].async_infer(inputs={self.InputLayer: r_frame})

    def infer_sync(self, inputs):
        self.InferenceCount += 1
        start = time.time()

        # Resize
        n, c, h, w = self.OpenVinoNetwork.inputs[self.InputLayer].shape
        # print('N: ', n, ' C: ', c, ' H: ', h, ' W: ', w)
        r_frame = cv.resize(inputs, (w, h))
        r_frame = cv.cvtColor(r_frame, cv.COLOR_BGR2RGB)
        r_frame = np.transpose(r_frame, (2, 0, 1))
        r_frame = np.expand_dims(r_frame, axis=0)

        self.OpenVinoExecutable.infer(inputs={self.InputLayer: r_frame})

        end = time.time()
        self.ElapsedInferenceTime += (end - start)

    def print_inference_performance_metrics(self):
        if self.Config.Async:
            print('Async Mode No Metrics')
        else:
            print('Average Inference Time Per Request : {}'.format(self.ElapsedInferenceTime / self.InferenceCount))


class FaceDetectionInference(OpenVINOInferenceBase):

    def get_faces_sync(self, images):
        """
        Get Detected Face Coordinates
        :param images:
        :return:
        """

        face_coordinates = []

        self.infer(images)

        res = self.OpenVinoExecutable.requests[0].outputs[self.OutputLayer]

        for r in res[0][0]:
            if r[2] > self.Config.FaceDetectionThreshold:
                face_coordinates.append([r[3], r[4], r[5], r[6]])

        return face_coordinates, []

    def get_faces_async(self, images, request_id=0):
        self.infer(images, request_id)

    # TODO: Make Callback Function
    def get_faces_after_async(self, request_id):
        face_coordinates = []

        res = self.OpenVinoExecutable.requests[request_id].outputs[self.OutputLayer]

        for r in res[0][0]:
            if r[2] > self.Config.FaceDetectionThreshold:
                face_coordinates.append([r[3], r[4], r[5], r[6]])

        return face_coordinates, []


class MtCNNFaceDetectionInference(OpenVINOInferenceBase):

    OpenVinoExecutablesP = list()

    OpenVinoExecutableR = ExecutableNetwork()
    OpenVinoExecutableO = ExecutableNetwork()

    OpenVinoNetworkP = IENetwork()
    OpenVinoNetworkR = IENetwork()
    OpenVinoNetworkO = IENetwork()

    Scales = []
    Thresholds = [0.6, 0.7, 0.8]

    NMSThresholds = [0.7, 0.7, 0.7]
    MinDetectionSize = 12
    Factor = 0.707

    MinimumFaceSize = 15.0
    MinLength = 720
    FactorCount = 0

    RIMGBOXES = []
    OIMGBOXES = []

    def prepare_detector(self):
        if self.Config.ModelPath is None or self.Config.ModelName is None:
            return None

        self.RIMGBOXES = np.zeros(dtype=float, shape=(self.Config.BatchSize, 3, 24, 24))
        self.OIMGBOXES = np.zeros(dtype=float, shape=(self.Config.BatchSize, 3, 48, 48))

        self.OpenVinoIE = IECore()

        if self.Config.CpuExtension and 'CPU' in self.Config.TargetDevice:
            self.OpenVinoIE.add_extension(self.Config.CpuExtensionPath, "CPU")

        try:
            # Model File Paths
            model_file = self.Config.ModelPath + "det1-0001.xml"
            model_weights = self.Config.ModelPath + "det1-0001.bin"
            self.OpenVinoNetworkP = IENetwork(model=model_file, weights=model_weights)

            model_file = self.Config.ModelPath + "det2-0001.xml"
            model_weights = self.Config.ModelPath + "det2-0001.bin"
            self.OpenVinoNetworkR = IENetwork(model=model_file, weights=model_weights)
            self.OpenVinoNetworkR.batch_size = self.Config.BatchSize

            model_file = self.Config.ModelPath + "det3-0001.xml"
            model_weights = self.Config.ModelPath + "det3-0001.bin"
            self.OpenVinoNetworkO = IENetwork(model=model_file, weights=model_weights)
            self.OpenVinoNetworkO.batch_size = self.Config.BatchSize
        except FileNotFoundError:
            print(FileNotFoundError.strerror, " ", FileNotFoundError.filename)
            exit(-1)

        if "CPU" in self.Config.TargetDevice:
            supported_layers = self.OpenVinoIE.query_network(self.OpenVinoNetworkP, "CPU")
            not_supported_layers = [l for l in self.OpenVinoNetworkP.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                print("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(self.Config.TargetDevice, ', '.join(not_supported_layers)))
                print(
                    "Please try to specify cpu extensions library path in config.json file ")

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

        config = {"DYN_BATCH_ENABLED" : "YES"}

        if self.Config.Async:

            self.OpenVinoExecutableR = self.OpenVinoIE.load_network(network=self.OpenVinoNetworkR,
                                                                    device_name=self.Config.TargetDevice,
                                                                    config=config,
                                                                    num_requests=self.Config.AsyncReq)

            self.OpenVinoExecutableO = self.OpenVinoIE.load_network(network=self.OpenVinoNetworkO,
                                                                    device_name=self.Config.TargetDevice,
                                                                    config=config,
                                                                    num_requests=self.Config.AsyncReq)
        else:
            self.OpenVinoExecutableR = self.OpenVinoIE.load_network(network=self.OpenVinoNetworkR,
                                                                    config=config,
                                                                    device_name=self.Config.TargetDevice)

            self.OpenVinoExecutableO = self.OpenVinoIE.load_network(network=self.OpenVinoNetworkO,
                                                                    config=config,
                                                                    device_name=self.Config.TargetDevice)

        self.MinLength = min(self.Config.InputHeight, self.Config.InputWidth)
        M = self.MinDetectionSize / self.MinimumFaceSize
        self.MinLength *= M

        while self.MinLength > self.MinDetectionSize:
            scale = (M*self.Factor**self.FactorCount)
            self.Scales.append(scale)
            self.MinLength *= self.Factor
            self.FactorCount += 1

            sw, sh = math.ceil(self.Config.InputWidth * scale), math.ceil(self.Config.InputHeight * scale)

            self.OpenVinoNetworkP.reshape({self.InputLayerP: (1, 3, sh, sw)})

            if self.Config.Async:
                self.OpenVinoExecutablesP.append(self.OpenVinoIE.load_network(network=self.OpenVinoNetworkP,
                                                                        device_name=self.Config.TargetDevice,
                                                                        num_requests=self.Config.AsyncReq))
            else:
                self.OpenVinoExecutablesP.append(self.OpenVinoIE.load_network(network=self.OpenVinoNetworkP,
                                                                              device_name=self.Config.TargetDevice))

    def get_faces_sync(self, images):
        """
        Get Detected Face Coordinates
        :param images:
        :return:
        """

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

            boxes = self.generate_bboxes(probs, offsets, scale, self.Thresholds[0])

            if len(boxes) == 0:
                bounding_boxes.append(None)
                none_count += 1
            else:
                keep = self.nms(boxes[:, 0:5], overlap_threshold=0.5)
                bounding_boxes.append(boxes[keep])

        if len(bounding_boxes) > none_count:
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)
            keep = self.nms(bounding_boxes[:, 0:5], self.NMSThresholds[0])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = self.calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            bounding_boxes = self.convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            img_boxes = self.get_image_boxes(bounding_boxes, image, size=24)

            if img_boxes.shape[0] > 0:
                shp = img_boxes.shape
                self.RIMGBOXES[0:shp[0], ] = img_boxes
                self.OpenVinoExecutableR.requests[0].set_batch(shp[0])
                self.OpenVinoExecutableR.requests[0].infer({self.InputLayerR: self.RIMGBOXES})

                offsets = self.OpenVinoExecutableR.requests[0].outputs['conv5_2'][:shp[0], ]
                probs = self.OpenVinoExecutableR.requests[0].outputs['prob1'][:shp[0]]

                keep = np.where(probs[:, 1] > self.Thresholds[1])[0]
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
                offsets = offsets[keep]
                keep = self.nms(bounding_boxes, self.NMSThresholds[1])
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes = self.calibrate_box(bounding_boxes, offsets[keep])
                bounding_boxes = self.convert_to_square(bounding_boxes)
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
                img_boxes = self.get_image_boxes(bounding_boxes, image, size=48)

                if img_boxes.shape[0] > 0:
                    shp = img_boxes.shape
                    self.OIMGBOXES[0:shp[0], ] = img_boxes

                    self.OpenVinoExecutableO.requests[0].set_batch(shp[0])
                    self.OpenVinoExecutableO.requests[0].infer({self.InputLayerO: self.OIMGBOXES})

                    landmarks = self.OpenVinoExecutableO.requests[0].outputs['conv6_3'][:shp[0]]
                    offsets = self.OpenVinoExecutableO.requests[0].outputs['conv6_2'][:shp[0]]
                    probs = self.OpenVinoExecutableO.requests[0].outputs['prob1'][:shp[0]]

                    keep = np.where(probs[:, 1] > self.Thresholds[2])[0]
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
                    keep = self.nms(bounding_boxes, self.NMSThresholds[2], mode='min')
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

        return face_detections, landmark_detections

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

        [dy, edy, dx, edx, y, ey, x, ex, w, h] = MtCNNFaceDetectionInference.correct_bboxes(bounding_boxes, width, height)
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

            img_boxes[i, :, :, :] = MtCNNFaceDetectionInference.preprocess(img_box)

        return img_boxes


class AgeGenderDetectionInference(OpenVINOInferenceBase):

    def get_age_gender(self, images):
        self.infer(images)
        detection = self.OpenVinoExecutable.requests[0].outputs[self.OutputLayer]
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

    def get_age_gender_async(self, images, request_id):
        self.infer(images, request_id)

    def get_age_gender_after_async(self, request_id):
        detection = self.OpenVinoExecutable.requests[request_id].outputs[self.OutputLayer]

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


def run_app():
    """
    Runs Face Detection Application
    """
    face_infer_cfg = FaceInferenceConfig()
    face_infer_cfg.ModelPath = face_detection_model_path
    face_infer_cfg.ModelName = face_detection_model_name
    face_infer_cfg.TargetDevice = face_detection_inference_device
    face_infer_cfg.Async = face_detection_async
    face_infer_cfg.AsyncReq = face_detection_async_req
    face_infer_cfg.BatchSize = face_detection_batch_size
    face_infer_cfg.CpuExtension = face_detection_cpu_extension
    face_infer_cfg.CpuExtensionPath = face_detection_cpu_extension_path
    face_infer_cfg.FaceDetectionThreshold = face_detection_threshold

    '''Get Age/Gender Detection Data'''
    age_gender_infer = None

    if run_age_gender:
        age_gender_infer_cfg = AgeGenderInferenceConfig()
        age_gender_infer_cfg.ModelPath = age_gender_model_path
        age_gender_infer_cfg.ModelName = age_gender_model_name
        age_gender_infer_cfg.TargetDevice = age_gender_inference_device
        age_gender_infer_cfg.Async = age_gender_async
        age_gender_infer_cfg.AsyncReq = age_gender_async_req
        age_gender_infer_cfg.BatchSize = age_gender_batch_size
        age_gender_infer_cfg.CpuExtension = age_gender_cpu_extension
        age_gender_infer_cfg.CpuExtensionPath = age_gender_cpu_extension_path
        age_gender_infer_cfg.RunAgeGenderDetection = run_age_gender

        age_gender_infer = AgeGenderDetectionInference(age_gender_infer_cfg)

    '''Open Web Cam (change 0 to any video file if required)'''

    for input_path in input_paths:
        if input_type == "video" or input_type == "webcam":
            capture = cv.VideoCapture(input_path)
            has_frame, frame = capture.read()
        elif input_type == "image":
            frame = cv.imread(input_path)
        else:
            print("Invalid Input Type: {}".format(input_type))
            exit(-1)

        face_infer_cfg.InputHeight = frame.shape[0]
        face_infer_cfg.InputWidth = frame.shape[1]

        face_infer = MtCNNFaceDetectionInference(face_infer_cfg)

        face_request_order = list()
        face_process_order = list()

        if face_infer.Config.Async:
            for i in range(face_infer.Config.AsyncReq):
                face_request_order.append(i)

        age_gender_request_order = list()
        age_gender_process_order = list()

        if age_gender_infer is not None and age_gender_infer.Config.Async:
            for i in range(age_gender_infer.Config.AsyncReq):
                age_gender_request_order.append(i)

        faces = []

        cv.namedWindow(cv_window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(cv_window_name, 800, 600)

        if input_type == "video" or input_type == "webcam":
            while has_frame:
                if face_infer.Config.Async:
                    req_id = face_request_order[0]
                    face_request_order.pop(0)
                    face_infer.get_faces_async(frame, req_id)
                    face_process_order.append(req_id)

                    if len(face_process_order) > 0:
                        first = face_process_order[0]
                        if face_infer.OpenVinoExecutable.requests[first].wait() == 0:
                            faces, landmarks = face_infer.get_faces_after_async(first)
                            face_process_order.pop(0)
                            face_request_order.append(first)
                else:
                    faces, landmarks = face_infer.get_faces_sync(frame)

                if len(faces) > 0:
                    for idx, face in enumerate(faces):
                        ImageUtil.draw_rectangle(frame, (face[0], face[1], face[2], face[3]))

                        for coordinate in range(0, len(landmarks[idx]), 2):
                            ImageUtil.draw_ellipse(frame, [landmarks[idx][coordinate], landmarks[idx][coordinate + 1]])

                        if run_age_gender:
                            cropped_image = ImageUtil.crop_frame(frame, (face[0], face[1], face[2], face[3]))
                            if age_gender_infer.Config.Async:
                                ag_req_id = age_gender_request_order[0]
                                age_gender_request_order.pop(0)
                                age_gender_infer.get_age_gender_async(frame, ag_req_id)
                                age_gender_process_order.append(ag_req_id)

                                if len(age_gender_process_order) > 0:
                                    ag_first = age_gender_process_order[0]
                                    if age_gender_infer.OpenVinoExecutable.requests[ag_first].wait() == 0:
                                        age, gender = age_gender_infer.get_age_gender_after_async(ag_first)
                                        age_gender_process_order.pop(0)
                                        age_gender_request_order.append(ag_first)
                                        age_gender_text = '{} - {}'
                                        age_gender_text = age_gender_text.format(age, gender)
                                        ImageUtil.draw_text(frame, age_gender_text,
                                                            (face[0], face[1], face[2], face[3]))
                            else:
                                age, gender = age_gender_infer.get_age_gender(cropped_image)
                                age_gender_text = '{} - {}'
                                age_gender_text = age_gender_text.format(age, gender)
                                ImageUtil.draw_text(frame, age_gender_text, (face[0], face[1], face[2], face[3]))

                    faces = []

                cv.imshow(cv_window_name, frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                has_frame, frame = capture.read()
        else:
            if face_infer.Config.Async:
                req_id = face_request_order[0]
                face_request_order.pop(0)
                face_infer.get_faces_async(frame, req_id)
                face_process_order.append(req_id)

                if len(face_process_order) > 0:
                    first = face_process_order[0]
                    if face_infer.OpenVinoExecutable.requests[first].wait() == 0:
                        faces, landmarks = face_infer.get_faces_after_async(first)
                        face_process_order.pop(0)
                        face_request_order.append(first)
            else:
                t_start = time.time()
                faces, landmarks= face_infer.get_faces_sync(frame)
                t_end = time.time()
                print("Inference Time: ", t_end - t_start)

            if len(faces) > 0:
                print("Detected {} Faces with {} Threshold".format(len(faces), face_infer.Config.FaceDetectionThreshold))
                for idx, face in enumerate(faces):
                    ImageUtil.draw_rectangle(frame, (face[0], face[1], face[2], face[3]))
                    for coordinate in range(0, len(landmarks[idx]), 2):
                        ImageUtil.draw_ellipse(frame, [landmarks[idx][coordinate], landmarks[idx][coordinate+1]])

                    if run_age_gender:
                        cropped_image = ImageUtil.crop_frame(frame, (face[0], face[1], face[2], face[3]))
                        if age_gender_infer.Config.Async:
                            ag_req_id = age_gender_request_order[0]
                            age_gender_request_order.pop(0)
                            age_gender_infer.get_age_gender_async(frame, ag_req_id)
                            age_gender_process_order.append(ag_req_id)

                            if len(age_gender_process_order) > 0:
                                ag_first = age_gender_process_order[0]
                                if age_gender_infer.OpenVinoExecutable.requests[ag_first].wait() == 0:
                                    age, gender = age_gender_infer.get_age_gender_after_async(ag_first)
                                    age_gender_process_order.pop(0)
                                    age_gender_request_order.append(ag_first)
                                    age_gender_text = '{} - {}'
                                    age_gender_text = age_gender_text.format(age, gender)
                                    ImageUtil.draw_text(frame, age_gender_text,
                                                        (face[0], face[1], face[2], face[3]))
                        else:
                            age, gender = age_gender_infer.get_age_gender(cropped_image)
                            age_gender_text = '{} - {}'
                            age_gender_text = age_gender_text.format(age, gender)
                            ImageUtil.draw_text(frame, age_gender_text, (face[0], face[1], face[2], face[3]))

                faces = []

            cv.imshow(cv_window_name, frame)
            cv.waitKey(0)


        face_infer.print_inference_performance_metrics()
        if run_age_gender:
            age_gender_infer.print_inference_performance_metrics()

    return None


"""
Global Parameters Used for Application Configuration
"""

cv_window_name = 'Face-Detection'

face_detection_model_path = '/home/intel/openvino_models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/'
face_detection_model_name = 'face-detection-adas-0001'
face_detection_threshold = 0.99
face_detection_inference_device = "CPU"
face_detection_async = False
face_detection_cpu_extension = False
face_detection_cpu_extension_path = "/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"
face_detection_batch_size = 1
face_detection_dynamic_batch = False
face_detection_async_req = 1

run_age_gender = True
age_gender_model_path = '/home/intel/Downloads/gamodel-r50/FP32/'
age_gender_model_name = 'model-0000'
age_gender_inference_device = "CPU"
age_gender_cpu_extension = False
age_gender_cpu_extension_path = "/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"
age_gender_async = False
age_gender_dynamic_batch = False
age_gender_batch_size = 1
age_gender_async_req = 1

input_type = "image"
input_paths = []
web_cam_index = 0


def parse_config_file(config_json='config.json'):
    """
    Parse Config File
    :param config_json:
    :return:
    """
    try:
        with open(config_json) as json_file:
            data = json.load(json_file)

            global cv_window_name
            cv_window_name = data['output_window_name']

            global input_paths
            input_paths = data["input_paths"]

            global input_type
            input_type = data["input_type"]

            global web_cam_index
            web_cam_index = int(data["web_cam_index"])

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

            global face_detection_async_req
            face_detection_async_req = int(data['face_detection_async_req'])

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
            if data['run_age_gender'] == "False":
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

            global age_gender_async_req
            age_gender_async_req = int(data['age_gender_async_req'])

            global age_gender_dynamic_batch
            if data["age_gender_dynamic_batch"] == "True":
                age_gender_dynamic_batch = True

            global age_gender_batch_size
            age_gender_batch_size = int(data["age_gender_batch_size"])

    except FileNotFoundError:
        print('{} FileNotFound'.format(config_json))
        exit(-1)


def print_help():
    print('Usage: python3 face_detection_openvino.py <config_file.json>')


# Application Entry Point
if __name__ == "__main__":

    if len(sys.argv) is not 2:
        print_help()
        print('using default config file')
        parse_config_file()
    else:
        parse_config_file(sys.argv[1])

    # Run FD App
    run_app()
