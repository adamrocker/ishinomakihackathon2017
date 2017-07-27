# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np


class Poly(object):
    def __init__(self, contour):
        self.__contour = contour
        arclen = cv2.arcLength(contour, True)
        self.__polygon = cv2.approxPolyDP(contour, 0.02 * arclen, True)

    def isQuadrangle(self):
        return len(self.__polygon) == 4

    def getQuadrangle(self):
        points = list(map(lambda x: x[0], self.__polygon))
        points_h = sorted(points, key=lambda x: x[0])
        lt, lb = points_h[0], points_h[1]
        if lb[1] < lt[1]:
            lt, lb = lb, lt
        rt, rb = points_h[2], points_h[3]
        if rb[1] < rt[1]:
            rt, rb = rb, rt
        quad = [lb, rb, rt, lt]
        return quad

    def getRectPoints(self):
        points = list(map(lambda x: x[0], self.__polygon))
        points_h = sorted(points, key=lambda x: x[0])
        points_v = sorted(points, key=lambda x: x[1])
        l = points_h[0][0]
        r = points_h[-1][0]
        t = points_v[0][1]
        b = points_v[-1][1]
        lt = (l, t)
        rb = (r, b)
        return lt, rb

    def getPerspectiveTransformMatrix(self):
        pp1 = self.getQuadrangle()
        pt1, pt2 = self.getRectPoints()
        pp2 = [[pt1[0], pt2[1]], [pt2[0], pt2[1]], [pt2[0], pt1[1]], [pt1[0], pt1[1]]]
        psp_matrix = cv2.getPerspectiveTransform(np.float32(pp1), np.float32(pp2))
        return psp_matrix


class Image(object):

    def __init__(self, filename, flags=cv2.IMREAD_COLOR):
        # type: (str) -> None
        self.__orig = cv2.imread(filename, flags)
        print "<< original >>"
        print self.__orig[0]
        print "row length=%d" % len(self.__orig[0])
        print "pixel:{} / {}".format(self.__orig[0][0], type(self.__orig[0][0]))
        self.__img = cv2.imread(filename, flags)
        self.__size = tuple([self.__orig.shape[1], self.__orig.shape[0]])
        self.__polys = []

    def show(self, img=None):
        if img is None:
            img = self.__img
        cv2.imshow('image', img)
        cv2.waitKey(0)

    def grayscale(self):
        self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)

    def blur(self):
        self.__img = cv2.GaussianBlur(self.__img, (11, 11), 0)

    def binarization(self):
        ret, img = cv2.threshold(self.__img, 150, 255, cv2.THRESH_TOZERO_INV)
        cv2.bitwise_not(img, self.__img)
        ret, img = cv2.threshold(self.__img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.__img = img

    def edge(self):
        img = cv2.Canny(self.__img, 50, 110)
        cv2.bitwise_not(img, img)
        for y, row in enumerate(img):
            for x, color in enumerate(row):
                if color == 0:  # black
                    print "pixel[{}][{}]: {}".format(x, y, type(self.__img[y][x]))
                    self.__img[y][x] = 0

    def contours(self):
        # rect
        img, contours, hierarchy = cv2.findContours(self.__img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        print("contours=%d, hierarchy=%d" % (len(contours), len(hierarchy)))

        # area size filtering
        limited_contours = [c for c in contours if 10000 <= cv2.contourArea(c, 0)]
        print("limited_contours=%d" % (len(limited_contours)))

        # contour rect
        for contour in limited_contours:
            poly = Poly(contour)
            self.__polys.append(poly)
        print("quadrangle=%d" % (len(self.__polys)))

    def drawPolys(self):
        color = (0, 0, 255)
        thickness = 1

        img = self.__orig.copy()
        for poly in self.__polys:
            pt1, pt2 = poly.getRectPoints()
            img = cv2.rectangle(img, pt1, pt2, color)

        self.__img = img

    def warpPerspective(self):
        imgs = []
        for poly in self.__polys:
            matrix = poly.getPerspectiveTransformMatrix()
            psp_img = cv2.warpPerspective(self.__orig, matrix, self.__size)
            pt1, pt2 = poly.getRectPoints()
            img = psp_img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            imgs.append(img)
            # self.__orig = cv2.rectangle(self.__orig, pt1, pt2, color)
        return imgs


def main(args):
    img = Image(args.filename)
    img.grayscale()
    img.blur()
    # img.edge()
    img.binarization()
    img.contours()
    img.drawPolys()
    img.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text processing from an image')
    parser.add_argument('filename', help="input file path")
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')  # version
    args = parser.parse_args()  # コマンドラインの引数を解釈します

    main(args)
