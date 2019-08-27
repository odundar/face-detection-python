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

import time, logging, json
import cv2 as cv
import numpy as np

# Import OpenVINO
# Make sure environment variables set correctly for this to work
# Check on README.md file
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork


class InferenceConfig(object):
    """
    Inference Configuration Model
    """
    ModelType = ""
    ModelPath = str()
    ModelName = str()
    TargetDevice = str()
    Async = False
    RequestCount = 1
    DynamicBatch = False
    BatchSize = 1
    CpuExtension = False
    CpuExtensionPath = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so"
    LimitCPUThreads = False
    CPUThreadNum = 1
    BindCPUThreads = True
    CPUStream = "AUTO"

    def parse_json(self, json_file):
        """
        Parse JSON Parameters
        :param json_file:
        :return:
        """
        try:
            logging.log(logging.INFO, "Loading JSON File".format(json_file))
            with open(json_file) as json_file:
                data = json.load(json_file)

                self.ModelPath = data[self.ModelType]["model_path"]
                self.ModelName = data[self.ModelType]["model_name"]
                self.TargetDevice = data[self.ModelType]["target_device"]

                if data[self.ModelType]["async"] == "True":
                    self.Async = True

                self.RequestCount = int(data[self.ModelType]["request_count"])
                self.BatchSize = int(data[self.ModelType]["batch_size"])

                if data[self.ModelType]["cpu_extension"] == "True":
                    self.CpuExtension = True

                self.CpuExtensionPath = data["cpu_extension_path"]

                if data[self.ModelType]["dynamic_batch"] == "True":
                    self.DynamicBatch = True

                if data[self.ModelType]["limit_cpu_threads"] == "True":
                    self.LimitCPUThreads = True

                self.CPUThreadNum = int(data[self.ModelType]["cpu_thread_num"])

                if data[self.ModelType]["bind_cpu_threads"] == "True":
                    self.LimitCPUThreads = True

                self.CPUStream = data[self.ModelType]["cpu_stream"]

        except FileNotFoundError:
            logging.log(logging.ERROR,'{} FileNotFound'.format(json_file))
            exit(-1)

    def read_dict(self, data=None):
        """
        Used When JSON Already Parsed as Dict
        :return:
        """
        if data is None:
            data = dict()
        self.ModelPath = data[self.ModelType]["model_path"]
        self.ModelName = data[self.ModelType]["model_name"]
        self.TargetDevice = data[self.ModelType]["target_device"]

        if data[self.ModelType]["async"] == "True":
            self.Async = True

        self.RequestCount = int(data[self.ModelType]["request_count"])
        self.BatchSize = int(data[self.ModelType]["batch_size"])

        if data[self.ModelType]["cpu_extension"] == "True":
            self.CpuExtension = True

        self.CpuExtensionPath = data["cpu_extension_path"]

        if data[self.ModelType]["dynamic_batch"] == "True":
            self.DynamicBatch = True

        if data[self.ModelType]["limit_cpu_threads"] == "True":
            self.LimitCPUThreads = True

        self.CPUThreadNum = int(data[self.ModelType]["cpu_thread_num"])

        if data[self.ModelType]["bind_cpu_threads"] == "True":
            self.LimitCPUThreads = True

        self.CPUStream = data[self.ModelType]["cpu_stream"]


class InferenceBase(object):
    """
    Base Class to Load a Model with Inference Engine
    """

    Config = InferenceConfig()

    '''Inference Engine Components'''
    OpenVinoIE = IECore()
    OpenVinoNetwork = IENetwork()
    OpenVinoExecutable = ExecutableNetwork()

    '''Model Components'''
    InputLayer = str()
    InputLayers = list()
    OutputLayer = str()
    OutputLayers = list()
    InputShape = None
    OutputShape = None

    '''Performance Metrics Storage'''
    ElapsedInferenceTime = 0.0
    InferenceCount = 0.0

    def __init__(self, infer_config):
        self.Config = infer_config
        self.prepare_detector()

    def prepare_detector(self):
        """
        Load Model, Libraries According to Given Configuration.
        :return:
        """
        if self.Config.ModelPath is None or self.Config.ModelName is None:
            return None

        ''' Model File Paths '''
        model_file = self.Config.ModelPath + self.Config.ModelName + '.xml'
        model_weights = self.Config.ModelPath + self.Config.ModelName + '.bin'

        logging.log(logging.INFO, "Model File {}".format(model_file))
        logging.log(logging.INFO, "Model Weights {}".format(model_weights))

        ''' Create IECore Object '''
        self.OpenVinoIE = IECore()

        ''' If target device is CPU add extensions '''
        if self.Config.CpuExtension and 'CPU' in self.Config.TargetDevice:
            logging.log(logging.INFO, "Adding CPU Extensions, Path {}".format(self.Config.CpuExtensionPath))
            self.OpenVinoIE.add_extension(self.Config.CpuExtensionPath, "CPU")

        ''' Try loading network '''
        try:
            self.OpenVinoNetwork = IENetwork(model=model_file, weights=model_weights)
            logging.log(logging.INFO, "Loaded IENetwork")
        except FileNotFoundError:
            logging.log(logging.ERROR, FileNotFoundError.strerror + " " + FileNotFoundError.filename)
            logging.log(logging.ERROR, "Exiting ....")
            exit(-1)

        ''' Print supported/not-supported layers '''
        if "CPU" in self.Config.TargetDevice:
            supported_layers = self.OpenVinoIE.query_network(self.OpenVinoNetwork, "CPU")
            not_supported_layers = [l for l in self.OpenVinoNetwork.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                logging.log(logging.WARN, "Following layers are not supported by the plugin for specified device {}:\n {}".format(self.Config.TargetDevice, ', '.join(not_supported_layers)))
                logging.log(logging.WARN, "Please try to specify cpu extensions library path in config.json file ")

        '''Input / Output Memory Allocations to feed input or get output values'''
        self.InputLayer = next(iter(self.OpenVinoNetwork.inputs))
        logging.log(logging.INFO, "Input Layer ".format(self.InputLayer))

        N, C, H, W = self.OpenVinoNetwork.inputs[self.InputLayer].shape

        if self.Config.BatchSize > N:
            self.OpenVinoNetwork.batch_size = self.Config.BatchSize
        else:
            self.Config.BatchSize = self.OpenVinoNetwork.batch_size

        self.OutputLayer = next(iter(self.OpenVinoNetwork.outputs))
        logging.log(logging.INFO, "Output Layer ".format(self.OutputLayer))

        self.InputLayers = list(self.OpenVinoNetwork.inputs)
        logging.log(logging.INFO, "Input Layers ".format(self.InputLayers))

        self.OutputLayers = list(self.OpenVinoNetwork.outputs)
        logging.log(logging.INFO, "Output Layers ".format(self.OutputLayers))

        self.InputShape = self.OpenVinoNetwork.inputs[self.InputLayer].shape
        logging.log(logging.INFO, "Input Shape: {}".format(self.InputShape))

        self.OutputShape = self.OpenVinoNetwork.outputs[self.OutputLayer].shape
        logging.log(logging.INFO, "Output Shape: {}".format(self.OutputShape))

        '''Set Configurations'''

        config = {}

        if self.Config.DynamicBatch:
            config["DYN_BATCH_ENABLE"] = "YES"
            logging.log(logging.INFO, "Enabling Dynamic Batch Mode")

        if self.Config.Async:
            logging.log(logging.INFO, "Async Mode Enabled")

        self.OpenVinoExecutable = self.OpenVinoIE.load_network(network=self.OpenVinoNetwork,
                                                               device_name=self.Config.TargetDevice,
                                                               config=config,
                                                               num_requests=self.Config.RequestCount)

        logging.log(logging.INFO, "Completed Loading Neural Network")

        return None

    def preprocess_input(self, input_data):
        """
        Pre-process Input According to Loaded Network
        :param input_data:
        :return:
        """

        n, c, h, w = self.OpenVinoNetwork.inputs[self.InputLayer].shape
        logging.log(logging.DEBUG, "Pre-processing Input to Shape {}".format(self.OpenVinoNetwork.inputs[self.InputLayer].shape))

        resized = cv.resize(input_data, (w, h))
        color_converted = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        transposed = np.transpose(color_converted, (2, 0, 1))
        reshaped = np.expand_dims(transposed, axis=0)

        return reshaped

    def infer(self, input_data, request_id=0):
        """
        Used to send data to network and start forward propagation.
        :param input_data:
        :param request_id:
        :return:
        """
        if self.Config.Async:
            logging.log(logging.DEBUG, "Async Infer Request Id {}".format(request_id))
            self.infer_async(input_data, request_id)
        else:
            logging.log(logging.DEBUG, "Infer Request Id {}".format(request_id))
            self.infer_sync(input_data, request_id)

    def infer_async(self, input_data, request_id=0):
        """
        Start Async Infer for Given Request Id
        :param input_data:
        :param request_id:
        :return:
        """
        self.InferenceCount += 1
        processed_input = self.preprocess_input(input_data)
        self.OpenVinoExecutable.requests[request_id].async_infer(inputs={self.InputLayer: processed_input})

    def infer_sync(self, input_data, request_id=0):
        """
        Start Sync Infer
        :param input_data:
        :param request_id:
        :return:
        """
        self.InferenceCount += 1
        processed_input = self.preprocess_input(input_data)
        start = time.time()
        self.OpenVinoExecutable.requests[request_id].infer(inputs={self.InputLayer: processed_input})
        end = time.time()
        self.ElapsedInferenceTime += (end - start)

    def request_ready(self, request_id):
        """
        Check if request is ready
        :param request_id: id to check request
        :return: bool
        """
        if self.Config.Async:
            if self.OpenVinoExecutable.requests[request_id].wait(0) == 0:
                return True
        else:
            return True

        return False

    def get_results(self, output_layer, request_id=0):
        """
        Get results from the network.
        :param output_layer: output layer
        :param request_id: request id
        :return:
        """
        logging.log(logging.DEBUG, "Getting Results Request Id {}".format(request_id))
        return self.OpenVinoExecutable.requests[request_id].outputs[output_layer]

    def print_inference_performance_metrics(self):
        """
        Print Performance Data Collection
        :return:
        """
        if self.Config.Async:
            logging.log(logging.WARN, 'Async Mode Inferred Frame Count {}'.format(self.InferenceCount))
        else:
            logging.log(logging.WARN, "Sync Mode Inferred Frame Count {}".format(self.InferenceCount))
            logging.log(logging.WARN, "Inference Per Input: {} MilliSeconds".format((self.ElapsedInferenceTime / self.InferenceCount) * 1000))
