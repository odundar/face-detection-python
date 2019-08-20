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

import cv2 as cv
import numpy as np


class ImageUtil(object):
    @staticmethod
    def crop_frame(frame, coordinate, normalized=True):
        """
        Crop Frame
        :param frame: cv mat object
        :param coordinate: x,y coordinates [xmin, ymin, xmax, ymax]
        :param normalized: if values normalized
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
        :param frame: cv mat object
        :param coordinate: x,y coordinates [xmin, ymin, xmax, ymax]
        :param normalized: if values normalized
        :param text: Text to write on image
        :param line_color: color of text
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
        bottom_left_corner_of_text = (x2, y1 + 10)
        font_scale = 0.4
        font_color = line_color
        line_type = 1

        cv.putText(frame,
                   text,
                   bottom_left_corner_of_text,
                   font,
                   font_scale,
                   font_color,
                   line_type)

    @staticmethod
    def draw_rectangles(frame, coordinates, line_color=(0, 255, 124), normalized=True):
        """
        Draw Rectangles with given Normalized
        :param frame: cv mat object
        :param coordinates: x,y coordinates [xmin, ymin, xmax, ymax]
        :param normalized: if values normalized
        :param line_color: color of rectangle
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
        :param frame: cv mat object
        :param coordinate: x,y coordinates [xmin, ymin, xmax, ymax]
        :param normalized: if values normalized
        :param line_color: color of rectangle
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
    def draw_ellipse(frame, coordinate, line_color=(124, 0, 0), radius=1, normalized=True):
        """
        Draw Circle with given values
        :param frame: cv mat object
        :param coordinate: x,y coordinates [xmin, ymin, xmax, ymax]
        :param normalized: if values normalized
        :param line_color: color of rectangle
        :param radius: radius of circle
        :return:
        """

        x1 = coordinate[0]
        y1 = coordinate[1]

        if normalized:
            h = frame.shape[0]
            w = frame.shape[1]

            x1 = int(x1 * w)
            y1 = int(y1 * h)

        cv.circle(frame, (x1, y1), radius=radius, color=line_color, thickness=1)
