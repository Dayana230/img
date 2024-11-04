import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
from matplotlib.path import Path
import ast

# Class and function definitions remain mostly unchanged

# Here is the core class and functions. Add them before your main Streamlit code.
class Triangle:
    def __init__(self, vertices):
        if isinstance(vertices, np.ndarray) == 0:
            raise ValueError("Input argument is not of type np.array.")
        if vertices.shape != (3, 2):
            raise ValueError("Input argument does not have the expected dimensions.")
        if vertices.dtype != np.float64:
            raise ValueError("Input argument is not of type float64.")
        self.vertices = vertices
        self.minX = int(self.vertices[:, 0].min())
        self.maxX = int(self.vertices[:, 0].max())
        self.minY = int(self.vertices[:, 1].min())
        self.maxY = int(self.vertices[:, 1].max())

    def getPoints(self):
        xList = range(self.minX, self.maxX + 1)
        yList = range(self.minY, self.maxY + 1)
        emptyList = list((x, y) for x in xList for y in yList)

        points = np.array(emptyList, np.float64)
        p = Path(self.vertices)
        grid = p.contains_points(points)
        mask = grid.reshape(self.maxX - self.minX + 1, self.maxY - self.minY + 1)

        trueArray = np.where(np.array(mask) == True)
        coordArray = np.vstack((trueArray[0] + self.minX, trueArray[1] + self.minY, np.ones(trueArray[0].shape[0])))

        return coordArray


class Morpher:
    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):
        if type(leftImage) != np.ndarray:
            raise TypeError('Input leftImage is not an np.ndarray')
        if leftImage.dtype != np.uint8:
            raise TypeError('Input leftImage is not of type np.uint8')
        if type(rightImage) != np.ndarray:
            raise TypeError('Input rightImage is not an np.ndarray')
        if rightImage.dtype != np.uint8:
            raise TypeError('Input rightImage is not of type np.uint8')
        if type(leftTriangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for j in leftTriangles:
            if isinstance(j, Triangle) == 0:
                raise TypeError('Element of input leftTriangles is not of Class Triangle')
        if type(rightTriangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for k in rightTriangles:
            if isinstance(k, Triangle) == 0:
                raise TypeError('Element of input rightTriangles is not of Class Triangle')
        self.leftImage = np.ndarray.copy(leftImage)
        self.leftTriangles = leftTriangles  # Not of type np.uint8
        self.rightImage = np.ndarray.copy(rightImage)
        self.rightTriangles = rightTriangles  # Not of type np.uint8
        self.leftInterpolation = RectBivariateSpline(np.arange(self.leftImage.shape[0]), np.arange(self.leftImage.shape[1]), self.leftImage)
        self.rightInterpolation = RectBivariateSpline(np.arange(self.rightImage.shape[0]), np.arange(self.rightImage.shape[1]), self.rightImage)

    def getImageAtAlpha(self, alpha):
        for leftTriangle, rightTriangle in zip(self.leftTriangles, self.rightTriangles):
            self.interpolatePoints(leftTriangle, rightTriangle, alpha)
        blendARR = ((1 - alpha) * self.leftImage + alpha * self.rightImage)
        blendARR = blendARR.astype(np.uint8)
        return blendARR

    def interpolatePoints(self, leftTriangle, rightTriangle, alpha):
        targetTriangle = Triangle(leftTriangle.vertices + (rightTriangle.vertices - leftTriangle.vertices) * alpha)
        targetVertices = targetTriangle.vertices.reshape(6, 1)
        tempLeftMatrix = np.array([[leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1, 0, 0, 0],
                                    [0, 0, 0, leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1],
                                    [leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1, 0, 0, 0],
                                    [0, 0, 0, leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1],
                                    [leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1, 0, 0, 0],
                                    [0, 0, 0, leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1]])
        tempRightMatrix = np.array([[rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1, 0, 0, 0],
                                     [0, 0, 0, rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1],
                                     [rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1, 0, 0, 0],
                                     [0, 0, 0, rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1],
                                     [rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1, 0, 0, 0],
                                     [0, 0, 0, rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1]])
        lefth = np.linalg.solve(tempLeftMatrix, targetVertices)
        righth = np.linalg.solve(tempRightMatrix, targetVertices)
        leftH = np.array([[lefth[0][0], lefth[1][0], lefth[2][0]], [lefth[3][0], lefth[4][0], lefth[5][0]], [0, 0, 1]])
        rightH = np.array([[righth[0][0], righth[1][0], righth[2][0]], [righth[3][0], righth[4][0], righth[5][0]], [0, 0, 1]])
        leftinvH = np.linalg.inv(leftH)
        rightinvH = np.linalg.inv(rightH)
        targetPoints = targetTriangle.getPoints()

        leftSourcePoints = np.transpose(np.matmul(leftinvH, targetPoints))
        rightSourcePoints = np.transpose(np.matmul(rightinvH, targetPoints))
        targetPoints = np.transpose(targetPoints)

        for x, y, z in zip(targetPoints, leftSourcePoints, rightSourcePoints):
            self.leftImage[int(x[1])][int(x[0])] = self.leftInterpolation(y[1], y[0])
            self.rightImage[int(x[1])][int(x[0])] = self.rightInterpolation(z[1], z[0])


def autofeaturepoints(leimg, riimg, featuregridsize, showfeatures):
    result = [[], []]
    for idx, img in enumerate([leimg, riimg]):
        try:
            if showfeatures:
                print(img.shape)

            result[idx] = [[0, 0], [(img.shape[1] - 1), 0], [0, (img.shape[0] - 1)], [(img.shape[1] - 1), (img.shape[0] - 1)]]

            h = int(img.shape[0] / featuregridsize) - 1
            w = int(img.shape[1] / featuregridsize) - 1

            for i in range(0, featuregridsize):
                for j in range(0, featuregridsize):
                    crop_img = img[(j * h):(j * h) + h, (i * w):(i * w) + w]
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    featurepoints = cv2.goodFeaturesToTrack(gray, 1, 0.1, 10)  # TODO: parameters can be tuned
                    if featurepoints is None:
                        featurepoints = [[[h / 2, w / 2]]]
                    featurepoints = np.int0(featurepoints)

                    for featurepoint in featurepoints:
                        x, y = featurepoint.ravel()
                        y = y + (j * h)
                        x = x + (i * w)
                        if showfeatures:
                            cv2.circle(img, (x, y), 3, 255, -1)
                        result[idx].append([x, y])

            if showfeatures:
                cv2.imshow("", img)
                cv2.waitKey(0)

        except Exception as ex:
            print(ex)
    return result


def loadTriangles (limg, rimg, featuregridsize, showfeatures) -> tuple:
    leftTriList = []
    rightTriList = []

    lrlists = autofeaturepoints(limg, rimg, featuregridsize
