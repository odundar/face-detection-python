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

from flask import Flask, request, jsonify, make_response, redirect
import logging
import sys
import optparse
import time
import cv2 as cv
import asyncio
import threading

from detection.face_detection_ov import FaceDetectionConfig, OpenMZooFaceDetection, FaceDetectionModelTypes, MtCNNFaceDetection, MTCNNFaceDetectionConfig
from detection.age_gender_detection_ov import AgeGenderConfig, MTCNNAgeGenderDetection, AgeGenderDetectionTypes, MTCNNAgeGenderConfig, AgeGenderDetection
from utils.image_utils import ImageUtil

app = Flask(__name__)

start = int(round(time.time()))

loop = asyncio.get_event_loop()

thread = threading.Thread()

class AppStatus:
    STARTED = "STARTED"
    FINISHED = "FINISHED"
    NOTSTARTED = "NOTSTARTED"
    STOPREQUEST = "STOPREQUESTED"


def prepare_configs():
    """
    Set Configurations for Face, Age Gender Models
    :return: face config, age_gender config
    """
    logging.getLogger(name="inference").log(logging.INFO, "Setting Configurations")

    if face_detection_model == FaceDetectionModelTypes.MTCNN:
        face_infer_cfg = MTCNNFaceDetectionConfig()
    else:
        face_infer_cfg = FaceDetectionConfig()

    face_infer_cfg.read_dict(json_req)

    logging.getLogger(name="inference").log(logging.INFO, "Configuration Set Completed...")

    return face_infer_cfg


async def inference():
    if inference_status == AppStatus.FINISHED or inference_status == AppStatus.NOTSTARTED:
        run_inference()
    else:
        logging.log(logging.WARN, "Inference Already Running ... ")
    loop.stop()
    return "OK"


def run_inference():
    """
    Runs Face Detection Application with the Requested JSON
    :return:
    """

    face_cfg = prepare_configs()

    if input_type == "video":
        logging.log(logging.INFO, "Video File Input Selected")
        capture = cv.VideoCapture(input_path)
        has_frame, frame = capture.read()
    elif input_type == "webcam":
        logging.log(logging.INFO, "Webcam Video Selected")
        capture = cv.VideoCapture(web_cam_index)
        has_frame, frame = capture.read()
    elif input_type == "image":
        logging.log(logging.INFO, "Single Image Inference Selected")
        frame = cv.imread(input_path)
    else:
        logging.log(logging.ERROR, "Invalid Input Type: {}".format(input_type))
        exit(-1)

    face_cfg.InputHeight = frame.shape[0]
    face_cfg.InputWidth = frame.shape[1]

    logging.getLogger(name="inference").log(logging.INFO, "Input Frame H: {} W: {}".format(face_cfg.InputHeight, face_cfg.InputWidth))

    if face_detection_model == FaceDetectionModelTypes.MTCNN:
        face_infer = MtCNNFaceDetection(face_cfg)
    else:
        face_infer = OpenMZooFaceDetection(face_cfg)

    face_request_order = list()
    face_process_order = list()

    for i in range(face_infer.Config.RequestCount):
        face_request_order.append(i)

    frame_order = []
    frame_id = 1

    global inference_status
    inference_status = AppStatus.STARTED

    if save_roi_text:
        roi_file = open(output_dir + roi_text_filename, 'w')
        roi = "{};{};{};{};{}\n".format("frameid","xmin","ymin","xmax","ymax")
        roi_file.write(roi)

    if save_roi_video:
        fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')
        roi_video = cv.VideoWriter(output_dir + roi_video_filename, fourcc, 10, (face_cfg.InputWidth, face_cfg.InputHeight ))

    if input_type == "video" or input_type == "webcam":
        while has_frame:

            if inference_status == AppStatus.STOPREQUEST:
                break

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
                                    ImageUtil.draw_ellipse(show_frame, [face_landmarks[idx][coordinate],
                                                                        face_landmarks[idx][coordinate + 1]])

                            if save_roi_text:
                                roi = "{};{};{};{};{}\n".format(frame_id, face[0], face[1], face[2], face[3])
                                roi_file.write(roi)

                    if save_only_frames and not save_roi_video and len(detected_faces) > 0:
                        cv.imwrite(output_dir + roi_frame_filename + "_{}.png".format(frame_id), show_frame)
                    elif save_roi_video:
                        roi_video.write(show_frame)

                    # Required Since
                    face_infer.LastFaceDetections = []
                    face_infer.LastLandmarkDetections = []

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
                        ImageUtil.draw_ellipse(frame, [landmarks[idx][coordinate], landmarks[idx][coordinate + 1]])

                if save_roi_text:
                    roi = "{};{};{};{};{}\n".format(frame_id, face[0], face[1], face[2], face[3])
                    roi_file.write(roi)

                if save_only_frames:
                    cv.imwrite(output_dir + roi_frame_filename + "_{}.png".format(frame_id), frame)

    face_infer.print_inference_performance_metrics()

    inference_status = AppStatus.FINISHED

    roi_file.close()
    roi_video.release()


inference_status = AppStatus.NOTSTARTED

input_type = "image"
input_path = ''
web_cam_index = 0
face_detection_model = FaceDetectionModelTypes.OPENMODELZOO
logfile_name = "log.txt" # "/app/log.txt"
json_req = None

output_dir = "./"
roi_text_filename = "inference_roi.txt"
roi_video_filename = "inference_roi.mp4"
roi_frame_filename = "inference_frame"

save_roi_video = False
save_only_frames = False
save_roi_text = True


@app.route("/", methods=['GET', 'POST'])
def start():
    if request.is_json:
        # Parse the JSON into a Python dictionary
        req = request.json
        try:
            if req["log_level"] == "DEBUG":
                logging.basicConfig(filename=logfile_name,
                                    level=logging.DEBUG,
                                    filemode='a',
                                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                    datefmt='%H:%M:%S')
            elif req["log_level"] == "INFO":
                logging.basicConfig(filename=logfile_name,
                                    level=logging.INFO,
                                    filemode='a',
                                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                    datefmt='%H:%M:%S')
            elif req["log_level"] == "WARN":
                logging.basicConfig(filename=logfile_name,
                                    level=logging.WARN,
                                    filemode='a',
                                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                    datefmt='%H:%M:%S')
            else:
                logging.basicConfig(filename=logfile_name,
                                    level=logging.ERROR,
                                    filemode='a',
                                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                    datefmt='%H:%M:%S')

            logging.log(logging.WARN, "Log Level Set to: {}".format(req["log_level"]))

            global input_path
            input_path = req["input_path"]

            logging.log(logging.WARN, "Input Path: {}".format(req["input_path"]))

            global input_type
            input_type = req["input_type"]

            logging.log(logging.WARN, "Input Type: {}".format(req["input_type"]))

            global web_cam_index
            web_cam_index = int(req["web_cam_index"])

            logging.log(logging.WARN, "Web Cam {}".format(req["web_cam_index"]))

            global face_detection_model
            if req['face_detection_model'] == FaceDetectionModelTypes.MTCNN:
                face_detection_model = FaceDetectionModelTypes.MTCNN

            logging.log(logging.WARN, "Face Detection Model {}".format(req["face_detection_model"]))

            global save_roi_video
            if req["save_roi_video"] == "True":
                save_roi_video = True

            global save_only_frames
            if req["save_only_frames"] == "True":
                save_only_frames = True

            global save_roi_text
            if req["save_roi"] == "False":
                save_roi_text = False

            res = make_response(jsonify({"message": "INFERENCE STARTED"}), 200)

            global json_req
            json_req = req

            #threading.Thread(target=run_inference()).start()
            # Start Async Thread
            logging.log(logging.WARN, "Starting Inference ...")
            task = loop.create_task(inference())

            if not loop.is_running():
                loop.run_forever()
            else:
                logging.log(logging.WARN, "Thread Loop Running ...")

            return res
        except KeyError:
            logging.log(logging.ERROR, "Key Not Found Error")
            exit(-1)
        except Exception as e:
            logging.log(logging.ERROR, e.__str__())
            exit(-1)
        # Return a string along with an HTTP status code

    else:
        # The request body wasn't JSON so return a 400 HTTP status code
        return "Request was not JSON", 400


@app.route("/status", methods=["GET"])
def status():
    """
    Get App Status
    :return:
    """
    logging.log(logging.WARN, "STATUS CALLED")
    return jsonify(inference_status), 200


@app.route("/stop_inference", methods=["POST"])
def stop_inference():
    """
    Get App Status
    :return:
    """

    global inference_status
    inference_status = AppStatus.STOPREQUEST
    logging.log(logging.WARN, "STOPPING INFERENCE ... ")
    return jsonify(inference_status), 200


@app.route("/logs", methods=["GET"])
def logs():
    """
    Show Logs
    :return:
    """
    with open(logfile_name) as f:
        file_content = f.read()

    return file_content, 200


@app.route("/results", methods=["GET"])
def results():
    """
    Get Latest Results
    :return:
    """

    roifile = output_dir + roi_text_filename
    with open(roifile) as f:
        file_content = f.read()

    return file_content


@app.route('/play_roi', methods=["GET"])
def play_roi():
    return redirect(output_dir + roi_video_filename)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage="python3 /app/face_detection_service.py -p ")
    parser.add_option('-p', '--port', action='store', dest='port', help='The port to listen on.')

    (args, _) = parser.parse_args()

    if args.port is None:
        print("Missing required argument: -p/--port")
        sys.exit(1)

    app.run(host='0.0.0.0', port=int(args.port), debug=True, threaded=True)