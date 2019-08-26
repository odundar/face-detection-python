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

import json
import numpy as np
import logging

from .detection_base_ov import InferenceBase, InferenceConfig


class AgeGenderDetectionTypes:
    MTCNN = "mtcnn_age_gender"
    OPENMODELZOO = "omz_age_gender"


class AgeGenderConfig(InferenceConfig):
    ModelType = AgeGenderDetectionTypes.OPENMODELZOO

    def parse_json(self, json_file):
        try:
            logging.log(logging.INFO, "Loading JSON File {}".format(json_file))
            logging.log(logging.INFO, "Model Type {}".format(self.ModelType))

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

        self.CpuExtensionPath = data[self.ModelType]["cpu_extension_path"]

        if data[self.ModelType]["dynamic_batch"] == "True":
            self.DynamicBatch = True

        if data[self.ModelType]["limit_cpu_threads"] == "True":
            self.LimitCPUThreads = True

        self.CPUThreadNum = int(data[self.ModelType]["number_of_cpu_threads"])

        if data[self.ModelType]["bind_cpu_threads"] == "True":
            self.LimitCPUThreads = True

        self.CPUStream = data[self.ModelType]["cpu_stream"]


class MTCNNAgeGenderConfig(AgeGenderConfig):
    ModelType = AgeGenderDetectionTypes.MTCNN


class MTCNNAgeGenderDetection(InferenceBase):

    Config = MTCNNAgeGenderConfig()

    def get_age_gender_data(self, request_id=0):
        """
        Parse Output Data for Age-Gender Detection Model
        :param request_id:
        :return:
        """
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


class AgeGenderDetection(InferenceBase):

    Config = AgeGenderConfig()

    def get_age_gender_data(self, request_id=0):
        """
        Parse Output Data for Age-Gender Detection Model
        :param request_id:
        :return:
        """
        age = int(self.OpenVinoExecutable.requests[request_id].outputs["age_conv3"][0][0][0][0] * 100)
        genders = self.OpenVinoExecutable.requests[request_id].outputs["prob"]
        # Parse detection vector to get age and gender

        gender_text = 'female'
        if genders[0][0][0][0] < genders[0][1][0][0]:
            gender_text = 'male'

        return age, gender_text