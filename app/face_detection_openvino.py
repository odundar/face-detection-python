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

import sys
import cv2 as cv
import json
import logging

from detection.face_detection_ov import FaceDetectionConfig, OpenMZooFaceDetection, FaceDetectionModelTypes, MtCNNFaceDetection, MTCNNFaceDetectionConfig
from detection.age_gender_detection_ov import AgeGenderConfig, MTCNNAgeGenderDetection, AgeGenderDetectionTypes, MTCNNAgeGenderConfig, AgeGenderDetection
from utils.image_utils import ImageUtil


def prepare_configs():
    """
    Set Configurations for Face, Age Gender Models
    :return: face config, age_gender config
    """
    logging.log(logging.INFO, "Setting Configurations")
    if face_detection_model == FaceDetectionModelTypes.MTCNN:
        face_infer_cfg = MTCNNFaceDetectionConfig()
    else:
        face_infer_cfg = FaceDetectionConfig()

    face_infer_cfg.parse_json(config_file)

    age_gender_cfg = None

    if run_age_gender:
        if age_gender_model == AgeGenderDetectionTypes.MTCNN:
            age_gender_cfg = MTCNNAgeGenderConfig()
        else:
            age_gender_cfg = AgeGenderConfig()
        age_gender_cfg.parse_json(config_file)

    return face_infer_cfg, age_gender_cfg


def run_app():
    """
    Runs Face Detection Application
    """
    face_cfg, age_cfg = prepare_configs()

    '''Open Web Cam (change 0 to any video file if required)'''

    if input_type == "video":
        capture = cv.VideoCapture(input_path)
        has_frame, frame = capture.read()
    elif input_type == "webcam":
        capture = cv.VideoCapture(web_cam_index)
        has_frame, frame = capture.read()
    elif input_type == "image":
        frame = cv.imread(input_path)
    else:
        logging.log(logging.ERROR, "Invalid Input Type: {}".format(input_type))
        exit(-1)

    face_cfg.InputHeight = frame.shape[0]
    face_cfg.InputWidth = frame.shape[1]

    if face_detection_model == FaceDetectionModelTypes.MTCNN:
        face_infer = MtCNNFaceDetection(face_cfg)
    else:
        face_infer = OpenMZooFaceDetection(face_cfg)

    if run_age_gender:
        if age_gender_model == AgeGenderDetectionTypes.MTCNN:
            age_gender_infer = MTCNNAgeGenderDetection(age_cfg)
        else:
            age_gender_infer = AgeGenderDetection(age_cfg)

    face_request_order = list()
    face_process_order = list()

    for i in range(face_infer.Config.RequestCount):
        face_request_order.append(i)

    cv.namedWindow(cv_window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(cv_window_name, 800, 600)

    frame_order = []
    frame_id = 1

    if input_type == "video" or input_type == "webcam":
        while has_frame:
            logging.log(logging.DEBUG, "Processing Frame {}".format(frame_id))
            if len(face_request_order) > 0:
                req_id = face_request_order[0]
                face_request_order.pop(0)
                face_infer.infer(frame, req_id)
                face_process_order.append(req_id)
                frame_order.append(frame)

            if len(face_process_order) > 0:
                first = face_process_order[0]
                if face_infer.request_ready(request_id=first):
                    detected_faces = face_infer.get_face_detection_data(first)
                    if face_cfg.ModelType == FaceDetectionModelTypes.MTCNN:
                        face_landmarks = face_infer.get_face_landmarks_data(first)
                    face_process_order.pop(0)
                    face_request_order.append(first)
                    show_frame = frame_order[0]
                    frame_order.pop(0)
                    if len(detected_faces) > 0:
                        for idx, face in enumerate(detected_faces):
                            ImageUtil.draw_rectangle(show_frame, (face[0], face[1], face[2], face[3]))

                            if face_cfg.ModelType == FaceDetectionModelTypes.MTCNN:
                                for coordinate in range(0, len(face_landmarks[idx]), 2):
                                    ImageUtil.draw_ellipse(show_frame, [face_landmarks[idx][coordinate], face_landmarks[idx][coordinate + 1]])

                            if run_age_gender:
                                cropped_image = ImageUtil.crop_frame(show_frame, (face[0], face[1], face[2], face[3]))
                                if cropped_image.size > 0:
                                    age_gender_infer.infer(cropped_image)
                                    age, gender = age_gender_infer.get_age_gender_data()
                                    age_gender_text = '{} - {}'
                                    age_gender_text = age_gender_text.format(age, gender)
                                    ImageUtil.draw_text(show_frame, age_gender_text, (face[0], face[1], face[2], face[3]))

                    cv.imshow(cv_window_name, show_frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

            if len(face_request_order) > 0:
                has_frame, frame = capture.read()
                frame_id += 1
    else:
        face_infer.infer(frame)
        faces = face_infer.get_face_detection_data()
        if face_cfg.ModelType == FaceDetectionModelTypes.MTCNN:
            landmarks = face_infer.get_face_landmarks_data()

        if len(faces) > 0:
            print("Detected {} Faces with {} Threshold".format(len(faces), face_infer.Config.FaceDetectionThreshold))
            for idx, face in enumerate(faces):
                ImageUtil.draw_rectangle(frame, (face[0], face[1], face[2], face[3]))
                if face_cfg.ModelType == FaceDetectionModelTypes.MTCNN:
                    for coordinate in range(0, len(landmarks[idx]), 2):
                        ImageUtil.draw_ellipse(frame, [landmarks[idx][coordinate], landmarks[idx][coordinate+1]])

                if run_age_gender:
                    cropped_image = ImageUtil.crop_frame(frame, (face[0], face[1], face[2], face[3]))
                    if cropped_image.size > 0:
                        age_gender_infer.infer(cropped_image)
                        age, gender = age_gender_infer.get_age_gender_data()
                        age_gender_text = '{} - {}'
                        age_gender_text = age_gender_text.format(age, gender)
                        ImageUtil.draw_text(frame, age_gender_text,
                                            (face[0], face[1], face[2], face[3]))

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
run_age_gender = False
input_type = "image"
input_path = ''
web_cam_index = 0

face_detection_model = FaceDetectionModelTypes.OPENMODELZOO
age_gender_model = AgeGenderDetectionTypes.OPENMODELZOO

config_file = "~/Projects/face_detection/config/config.json"


def parse_config_file(config_json='config.json'):
    """
    Parse Config File
    :param config_json:
    :return:
    """
    global config_file
    config_file = config_json

    try:
        with open(config_json) as json_file:
            data = json.load(json_file)

            global cv_window_name
            cv_window_name = data['output_window_name']

            global input_path
            input_path = data["input_path"]

            global input_type
            input_type = data["input_type"]

            global web_cam_index
            web_cam_index = int(data["web_cam_index"])

            global run_age_gender
            if data['run_age_gender'] == "True":
                run_age_gender = True

            global face_detection_model
            if data['face_detection_model'] == FaceDetectionModelTypes.MTCNN:
                face_detection_model = FaceDetectionModelTypes.MTCNN

            global age_gender_model
            if data['age_gender_detection_model'] == AgeGenderDetectionTypes.MTCNN:
                age_gender_model = AgeGenderDetectionTypes.MTCNN

            if data["log_level"] == "DEBUG":
                logging.basicConfig(level=logging.DEBUG)
            elif data["log_level"] == "INFO":
                logging.basicConfig(level=logging.INFO)
            elif data["log_level"] == "WARN":
                logging.basicConfig(level=logging.WARN)
            else:
                logging.basicConfig(level=logging.ERROR)

            logging.log(logging.WARN, "Log Level Set to: {}".format(data["log_level"]))

    except FileNotFoundError:
        print('{} FileNotFound'.format(config_json))
        exit(-1)


def print_help():
    print('Usage: python3 face_detection_openvino.py <config_file.json>')


# Application Entry Point
if __name__ == "__main__":

    if len(sys.argv) is not 2:
        print_help()
        print('Using default config file: {}'.format(config_file))
        parse_config_file(config_file)
    else:
        parse_config_file(sys.argv[1])

    # Run FD App
    run_app()
