import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import List
import sns_toolbox
import sns_toolbox.networks
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.neurons import NonSpikingNeuron
import itertools
from scipy.interpolate import interp1d
import scipy as sp
import sklearn


def trajectoryInput(t, T, theta_list, func_type, to_plot=False):
    if func_type == 'sinusoid':
        '''
        in this case, thetamin = theta_list[0] and thetamax = theta_list[1]
        '''
        thetamin = theta_list[0]
        thetamax = theta_list[1]
        inputT = (thetamin + thetamax) / 2 + (thetamax - thetamin) / 2 * np.sin(2 * np.pi * t / T)
    elif func_type == 'steps':
        '''
        replace thetamin and thetamax with vectors
        make the signal the 0th value for T, then the 1st value for T after that, etc.
        '''
        tmax = np.max(t)
        inputT = np.zeros_like(t)
        num_steps = int(np.ceil(tmax / T))
        num_vals = np.size(theta_list)
        for i in np.arange(num_steps):
            bin_mask = np.floor((t / T)) == i
            inputT[bin_mask] = theta_list[np.mod(i, num_vals)]

    else:
        inputT = None

    if to_plot:
        qw = 1
        # plt.figure()
        # plt.plot(t, inputT)
        # plt.draw()
        # plt.show()
    return inputT

def bell_curve(magnitude, width, theta, shift):
    return magnitude * np.exp(-width * pow((theta - shift), 2))

def SkewMatrix(x):
    skew = [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]
    return skew

class LegConfig:
    '''
        legMatConfig returns 3x4 matrix with xyz coordinates of the origin and all servos(joints)
        in the zero position configuration
        legAxes returns 3x3 matrix with unit vectors directions for each joint
    '''

    def __init__(self, name):
        self.name = name

    def legMatConfig(self):
        L1 = 52.52
        L2 = 60
        L3 = 95

        x0 = 125
        y02 = 100
        y01 = 65
        y03 = 65
        # zero position config
        p0 = np.array([[0],
                       [0],
                       [0]])
        if self.name == 'R1':
            # R1 leg
            p1R1 = np.array([[x0],
                             [-y01],
                             [0]])
            p2R1 = np.array([[x0 + L1 * np.cos(45 * np.pi / 180)],
                             [-y02 - L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p3R1 = np.array([[x0 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p4R1 = np.array([[x0 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm
            pR1 = np.concatenate((np.concatenate((np.concatenate((p1R1, p2R1), axis=1), p3R1), axis=1), p4R1), axis=1)
            return pR1
        elif self.name == 'R2':
            # R2 leg
            p1R2 = np.array([[0],
                             [-y02],
                             [0]])
            p2R2 = np.array([[0],
                             [-y02 - L1],
                             [0]])  # mm
            p3R2 = np.array([[0],
                             [-y02 - (L1 + L2)],
                             [-15]])  # mm
            p4R2 = np.array([[0],
                             [-y02 - (L1 + L2 + L3)],
                             [-15 - 90]])  # mm
            pR2 = np.concatenate((np.concatenate((np.concatenate((p1R2, p2R2), axis=1), p3R2), axis=1), p4R2), axis=1)
            return pR2
        elif self.name == 'R3':
            # R3 leg
            p1R3 = np.array([[-x0],
                             [-y03],
                             [0]])
            p2R3 = np.array([[-x0 - L1 * np.cos(45 * np.pi / 180)],
                             [-y02 - L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p3R3 = np.array([[-x0 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p4R3 = np.array([[-x0 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm
            pR3 = np.concatenate((np.concatenate((np.concatenate((p1R3, p2R3), axis=1), p3R3), axis=1), p4R3), axis=1)
            return pR3
        elif self.name == 'L1':
            # L1 leg
            p1L1 = np.array([[x0],
                             [y01],
                             [0]])
            p2L1 = np.array([[x0 + L1 * np.cos(45 * np.pi / 180)],
                             [y02 + L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p3L1 = np.array([[x0 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p4L1 = np.array([[x0 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm
            pL1 = np.concatenate((np.concatenate((np.concatenate((p1L1, p2L1), axis=1), p3L1), axis=1), p4L1), axis=1)
            return pL1
        elif self.name == 'L2':
            # L2 leg
            p1L2 = np.array([[0],
                             [y02],
                             [0]])
            p2L2 = np.array([[0],
                             [y02 + L1],
                             [0]])  # mm
            p3L2 = np.array([[0],
                             [y02 + (L1 + L2)],
                             [-15]])  # mm
            p4L2 = np.array([[0],
                             [y02 + (L1 + L2 + L3)],
                             [-15 - 90]])  # mm
            pL2 = np.concatenate((np.concatenate((np.concatenate((p1L2, p2L2), axis=1), p3L2), axis=1), p4L2), axis=1)
            return pL2
        elif self.name == 'L3':
            # L3 leg
            p1L3 = np.array([[-x0],
                             [y03],
                             [0]])
            p2L3 = np.array([[-x0 - L1 * np.cos(45 * np.pi / 180)],
                             [y02 + L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p3L3 = np.array([[-x0 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p4L3 = np.array([[-x0 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm
            pL3 = np.concatenate((np.concatenate((np.concatenate((p1L3, p2L3), axis=1), p3L3), axis=1), p4L3), axis=1)
            return pL3
        else:
            print('Available options are: R1-R3, L1-L3')

    def legAxes(self):
        w1 = np.array([[0],
                       [0],
                       [1]])
        if self.name == 'R1':
            # R1 leg
            w2R1 = np.array([[1 / np.sqrt(2)],
                             [1 / np.sqrt(2)],
                             [0]])
            wR1 = np.concatenate((np.concatenate((w1, w2R1), axis=1), w2R1), axis=1)
            return wR1
        elif self.name == 'L1':
            # L1 leg
            w2L1 = np.array([[-1 / np.sqrt(2)],
                             [1 / np.sqrt(2)],
                             [0]])
            wL1 = np.concatenate((np.concatenate((w1, w2L1), axis=1), w2L1), axis=1)
            return wL1
        elif self.name == 'R2':
            # R2 leg
            w2R2 = np.array([[1 / np.sqrt(2)],
                             [0], [0]])
            wR2 = np.concatenate((np.concatenate((w1, w2R2), axis=1), w2R2), axis=1)
            return wR2
        elif self.name == 'L2':
            # L2 leg
            w2L2 = np.array([[-1 / np.sqrt(2)],
                             [0], [0]])
            wL2 = np.concatenate((np.concatenate((w1, w2L2), axis=1), w2L2), axis=1)
            return wL2
        elif self.name == 'R3':
            # R3 leg
            w2L1 = np.array([[-1 / np.sqrt(2)],
                             [1 / np.sqrt(2)],
                             [0]])
            wR3 = np.concatenate((np.concatenate((w1, -w2L1), axis=1), -w2L1), axis=1)
            return wR3
        elif self.name == 'L3':
            # L3 leg
            w2L3 = np.array([[-1 / np.sqrt(2)],
                             [-1 / np.sqrt(2)],
                             [0]])
            wL3 = np.concatenate((np.concatenate((w1, w2L3), axis=1), w2L3), axis=1)
            return wL3
        else:
            print('Available options are: wR1-wR3, wL1-wL3')

class LegConfigTestOnly:
    '''
        legMatConfig returns 3x4 matrix with xyz coordinates of the origin and all servos(joints)
        in the zero position configuration
        legAxes returns 3x3 matrix with unit vectors directions for each joint
    '''

    def __init__(self, name):
        self.name = name

    def legMatConfig(self):
        L1 = 52.52
        L2 = 60
        L3 = 95
        L4 = L1 / 2
        L5 = L2 / 3
        L6 = L3 / 2

        x0 = 125
        y02 = 100
        y01 = 65
        y03 = 65
        # zero position config
        p0 = np.array([[0],
                       [0],
                       [0]])
        if self.name == 'R1':
            # R1 leg
            p1R1 = np.array([[x0],
                             [-y01],
                             [0]])
            p5R1 = np.array([[x0 + (L1 - L5) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 - L5) * np.cos(45 * np.pi / 180)],
                             [0]])
            p2R1 = np.array([[x0 + L1 * np.cos(45 * np.pi / 180)],
                             [-y02 - L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p6R1 = np.array([[x0 + (L1 + L4) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L4) * np.cos(45 * np.pi / 180)],
                             [0]])
            p3R1 = np.array([[x0 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p7R1 = np.array([[x0 + (L1 + L2 + L5) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2 + L5) * np.cos(45 * np.pi / 180)],
                             [-15]])
            p4R1 = np.array([[x0 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm

            p1_1 = np.concatenate((p1R1, p5R1), axis=1)
            p2_1 = np.concatenate((p1_1, p2R1), axis=1)
            p3_1 = np.concatenate((p2_1, p6R1), axis=1)
            p4_1 = np.concatenate((p3_1, p3R1), axis=1)
            p5_1 = np.concatenate((p4_1, p7R1), axis=1)

            pR1 = np.concatenate((p5_1, p4R1), axis=1)

            # pR1 = np.concatenate((np.concatenate((np.concatenate((p1R1, p2R1), axis=1), p3R1), axis=1), p4R1), axis=1)
            return pR1
        # if self.name == 'R1':
        #     # R1 leg
        #     p1R1 = np.array([[x0],
        #                      [-y01],
        #                      [0]])
        #     p2R1 = np.array([[x0 + L1 * np.cos(45 * np.pi / 180)],
        #                      [-y02 - L1 * np.cos(45 * np.pi / 180)],
        #                      [0]])  # mm
        #     p3R1 = np.array([[x0 + (L1 + L2) * np.cos(45 * np.pi / 180)],
        #                      [-y02 - (L1 + L2) * np.cos(45 * np.pi / 180)],
        #                      [-15]])  # mm
        #     p4R1 = np.array([[x0 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
        #                      [-y02 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
        #                      [-15 - 90]])  # mm
        #     pR1 = np.concatenate((np.concatenate((np.concatenate((p1R1, p2R1), axis=1), p3R1), axis=1), p4R1), axis=1)
        #     return pR1
        elif self.name == 'R2':
            # R2 leg
            p1R2 = np.array([[0],
                             [-y02],
                             [0]])
            p2R2 = np.array([[0],
                             [-y02 - L1],
                             [0]])  # mm
            p3R2 = np.array([[0],
                             [-y02 - (L1 + L2)],
                             [-15]])  # mm
            p4R2 = np.array([[0],
                             [-y02 - (L1 + L2 + L3)],
                             [-15 - 90]])  # mm
            pR2 = np.concatenate((np.concatenate((np.concatenate((p1R2, p2R2), axis=1), p3R2), axis=1), p4R2), axis=1)
            return pR2
        elif self.name == 'R3':
            # R3 leg
            p1R3 = np.array([[-x0],
                             [-y03],
                             [0]])
            p2R3 = np.array([[-x0 - L1 * np.cos(45 * np.pi / 180)],
                             [-y02 - L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p3R3 = np.array([[-x0 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p4R3 = np.array([[-x0 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm
            pR3 = np.concatenate((np.concatenate((np.concatenate((p1R3, p2R3), axis=1), p3R3), axis=1), p4R3), axis=1)
            return pR3
        elif self.name == 'L1':
            # L1 leg
            p1L1 = np.array([[x0],
                             [y01],
                             [0]])
            p2L1 = np.array([[x0 + L1 * np.cos(45 * np.pi / 180)],
                             [y02 + L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p3L1 = np.array([[x0 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p4L1 = np.array([[x0 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm
            pL1 = np.concatenate((np.concatenate((np.concatenate((p1L1, p2L1), axis=1), p3L1), axis=1), p4L1), axis=1)
            return pL1
        elif self.name == 'L2':
            # L2 leg
            p1L2 = np.array([[0],
                             [y02],
                             [0]])
            p2L2 = np.array([[0],
                             [y02 + L1],
                             [0]])  # mm
            p3L2 = np.array([[0],
                             [y02 + (L1 + L2)],
                             [-15]])  # mm
            p4L2 = np.array([[0],
                             [y02 + (L1 + L2 + L3)],
                             [-15 - 90]])  # mm
            pL2 = np.concatenate((np.concatenate((np.concatenate((p1L2, p2L2), axis=1), p3L2), axis=1), p4L2), axis=1)
            return pL2
        elif self.name == 'L3':
            # L3 leg
            p1L3 = np.array([[-x0],
                             [y03],
                             [0]])
            p2L3 = np.array([[-x0 - L1 * np.cos(45 * np.pi / 180)],
                             [y02 + L1 * np.cos(45 * np.pi / 180)],
                             [0]])  # mm
            p3L3 = np.array([[-x0 - (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2) * np.cos(45 * np.pi / 180)],
                             [-15]])  # mm
            p4L3 = np.array([[-x0 - (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [y02 + (L1 + L2 + L3) * np.cos(45 * np.pi / 180)],
                             [-15 - 90]])  # mm
            pL3 = np.concatenate((np.concatenate((np.concatenate((p1L3, p2L3), axis=1), p3L3), axis=1), p4L3), axis=1)
            return pL3
        else:
            print('Available options are: R1-R3, L1-L3')

    def legAxes(self):
        w1 = np.array([[0],
                       [0],
                       [1]])
        if self.name == 'R1':
            # R1 leg
            w2R1 = np.array([[1 / np.sqrt(2)],
                             [1 / np.sqrt(2)],
                             [0]])
            w1_1R1 = np.concatenate((w1, w2R1), axis=1)
            w2_1R1 = np.concatenate((w1_1R1, w2R1), axis=1)
            w3_1R1 = np.concatenate((w2_1R1, w2R1), axis=1)
            w4_1R1 = np.concatenate((w3_1R1, w1), axis=1)
            wR1 = np.concatenate((w4_1R1, w2R1), axis=1)
            # wR1 = np.concatenate((np.concatenate((w1, w2R1), axis=1), w2R1), axis=1)
            return wR1
        elif self.name == 'L1':
            # L1 leg
            w2L1 = np.array([[-1 / np.sqrt(2)],
                             [1 / np.sqrt(2)],
                             [0]])
            wL1 = np.concatenate((np.concatenate((w1, w2L1), axis=1), w2L1), axis=1)
            return wL1
        elif self.name == 'R2':
            # R2 leg
            w2R2 = np.array([[1 / np.sqrt(2)],
                             [0], [0]])
            wR2 = np.concatenate((np.concatenate((w1, w2R2), axis=1), w2R2), axis=1)
            return wR2
        elif self.name == 'L2':
            # L2 leg
            w2L2 = np.array([[-1 / np.sqrt(2)],
                             [0], [0]])
            wL2 = np.concatenate((np.concatenate((w1, w2L2), axis=1), w2L2), axis=1)
            return wL2
        elif self.name == 'R3':
            # R3 leg
            w2L1 = np.array([[-1 / np.sqrt(2)],
                             [1 / np.sqrt(2)],
                             [0]])
            wR3 = np.concatenate((np.concatenate((w1, -w2L1), axis=1), -w2L1), axis=1)
            return wR3
        elif self.name == 'L3':
            # L3 leg
            w2L3 = np.array([[-1 / np.sqrt(2)],
                             [-1 / np.sqrt(2)],
                             [0]])
            wL3 = np.concatenate((np.concatenate((w1, w2L3), axis=1), w2L3), axis=1)
            return wL3
        else:
            print('Available options are: wR1-wR3, wL1-wL3')

def JacobianFunc(w, p, TmatArray):  # R1, R2, p1Final, p2Final):
    '''
    :param w:
    :param p: zero pos config
    :param TmatArray: translational matrix
    :return: 6x3 jacobian matrix
    '''
    numJoints = TmatArray.shape[0]
    numLayers = numJoints - 1
    sCross = np.zeros([3, numJoints])
    Jmat = np.zeros([6, numJoints])
    pSkew = np.zeros([numJoints - 1, 3, 3])
    Rmat = np.zeros([numJoints - 1, 3, 3])
    adjointMat = np.zeros([numJoints - 1, 6, 6])

    for joint in range(numJoints):
        sCross[:, joint] = np.cross(-w[:, joint], p[:, joint])
        Jmat[:, joint] = np.concatenate((sCross[:, joint], w[:, joint]), axis=0)

    for joint in range(numLayers):
        pSkew[joint, :, :] = SkewMatrix(TmatArray[joint, 0:3, 3])
        Rmat[joint, :, :] = np.matmul(pSkew[joint, :, :], TmatArray[joint, 0:3, 0:3])
        adjointMat[joint, :, :] = np.concatenate(
            (np.concatenate((TmatArray[joint, 0:3, 0:3], Rmat[joint, :, :]), axis=1),
             np.concatenate((np.zeros([3, 3]), TmatArray[joint, 0:3, 0:3]), axis=1)), axis=0)  # 6x6

    jac = np.zeros([numLayers, 6, 1])
    for joint in range(numJoints - 1):
        jac[joint, :, :] = np.matmul(adjointMat[joint, :, :], Jmat[:, joint + 1].reshape((6, 1)))  # 6x1

    jacobianMatrix = Jmat[:, 0].reshape((6, 1))
    for joint in range(numLayers):
        jacobianMatrix = np.concatenate((jacobianMatrix, jac[joint, :, :]), axis=1)
    return jacobianMatrix

def RodriguesFunc(w, theta, p):
    Scross = np.cross(-w, p)
    Smatrix = np.array([[0.0, -w[2], w[1]],
                        [w[2], 0.0, -w[0]],
                        [-w[1], w[0], 0.0]])  # Skew-symmetric
    Rmatrix = np.eye(3, dtype=float) + Smatrix * np.sin(theta) + (Smatrix @ Smatrix) * (1 - np.cos(theta))
    c = ((np.eye(3, dtype=float) - Rmatrix) @ Smatrix) @ Scross
    d = w @ w.reshape((3, 1))
    e = np.dot(theta, np.dot(d.item(), Scross))
    TranslVec = c + e
    f = np.concatenate((Rmatrix, TranslVec.reshape((3, 1))), axis=1)
    expoFinal = np.concatenate((f, np.array([[0, 0, 0, 1]])), axis=0)
    return expoFinal

def Gmax2(k, R, delE):
    maxConduct = k * R / (delE - 2 * k * R)
    return maxConduct

def FrwrdKnmtcsFunc(w, thetaValue, p):
    translMatArray = np.zeros([thetaValue.shape[0], 4, 4], dtype=float)
    expoOne = np.eye(4, dtype=float)
    if hasattr(thetaValue, "__len__"):
        for index in range(len(thetaValue)):
            expo = RodriguesFunc(w[:, index], thetaValue[index], p[:, index])
            expoOne = expoOne @ expo
            translMatArray[index, :, :] = expoOne
    else:
        expo = RodriguesFunc(w[:, 0], thetaValue, p[:, 0])
        expoOne = expoOne @ expo
        translMatArray[:, :] = expoOne
    positionM = np.atleast_2d(p[:, -1]).T
    M = np.concatenate((np.concatenate((np.eye(3, dtype=float), positionM), axis=1),
                        np.array([[0, 0, 0, 1]])), axis=0)
    TranslMat = expoOne @ M

    JacobianMatrix = JacobianFunc(w, p, translMatArray)
    return (TranslMat, JacobianMatrix)

def normVec(vec, maxRange, minRange):
    normalized = ((vec - min(vec)) * (maxRange - minRange)) / (max(vec) - min(vec)) + minRange
    return normalized

def ComboVec(input1bell, input2bell, mag, gMax, delE):
    # position
    if np.ndim(input2bell) == 1:
        input2bell = np.reshape(input2bell, [1, -1])

    temp_mat2 = np.transpose((input1bell / mag * gMax * delE + np.transpose(input2bell) / mag * gMax * delE) / (
                1 + gMax * input1bell / mag + gMax * np.transpose(input2bell) / mag)) - mag
    comboSS2 = np.reshape(temp_mat2, [-1, 1])
    comboSS2[comboSS2 < 0] = 0
    comboSS2[comboSS2 > mag] = mag
    return comboSS2

class NonSpikingInput(nn.Module):

    def __init__(self,
                 num_neurons_per_joint: int,
                 num_inputs: int = 1,
                 parameters: dict = None,
                 device: torch.device = torch.device('mps')):

        super().__init__()

        self.device = device
        # Default parameters
        if parameters is None: parameters = {}
        self.dt = parameters.get('dt', 0.1)
        c_m = parameters.get('c_m', 1.0)
        e_rest = parameters.get('e_rest', 0.0)
        g_m_leak = parameters.get('g_m_leak', 5.0)
        e_syn = parameters.get('e_syn', 20.0)
        R_op_range = parameters.get('R_op_range', 1.0)
        i_bias = parameters.get('i_bias', 0.0)

        self.num_inputs = num_inputs  # number of joints
        self.N = num_neurons_per_joint  # N neurons per joint

        self.num_neurons = num_neurons_per_joint

        # Neuron parameters
        self.c_m = torch.full((self.num_neurons, 1), c_m, device=self.device)
        self.e_rest = torch.full((self.num_neurons, 1), e_rest, device=self.device)
        self.g_m_leak = torch.full((self.num_neurons, 1), g_m_leak, device=self.device)
        self.e_syn_delta = torch.full((self.num_neurons, 1), e_rest + e_syn, device=self.device)
        self.R_op_range = torch.full((self.num_neurons, 1), R_op_range, device=self.device)
        self.i_bias = torch.full((self.num_neurons, 1), i_bias, device=self.device)
        self.v_m = self.e_rest.clone().squeeze(1)

    '''
        num integration
    '''

    def _dv_dt(self, v_post, i_ext):
        i_leak = (v_post - self.e_rest) * self.g_m_leak
        return (self.i_bias + i_ext - i_leak) / self.c_m

    '''
        runge-kutta 4th order integration
    '''

    def _rk4_forward_pass(self, i_ext_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_ext_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_ext_values).squeeze(1)
        k3 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k2.unsqueeze(1), i_ext_values).squeeze(1)
        k4 = self._dv_dt(v_m_unsqueezed + self.dt * k3.unsqueeze(1), i_ext_values).squeeze(1)

        self.v_m += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return self.v_m

    '''
        runge-kutta 2th order integration (modified euler)
    '''

    def _rk2_forward_pass(self, i_ext_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_ext_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_ext_values).squeeze(1)

        self.v_m += self.dt * k2

        return self.v_m

    def forward(self, i_ext_vector: torch.Tensor) -> torch.Tensor:

        num_steps = i_ext_vector.shape[1]

        v_m_Post = torch.zeros(self.num_neurons, num_steps, device=self.device)

        for t in range(num_steps):
            i_ext_t = i_ext_vector[:, t].unsqueeze(1)

            self.v_m = self._rk2_forward_pass(i_ext_values=i_ext_t)

            v_m_Post[:, t] = self.v_m

        return v_m_Post

class NonSpikingInterneuron(nn.Module):

    def __init__(self,
                 num_neurons_per_joint: int,
                 num_inputs: int = 1,
                 parameters: dict = None,
                 connection_type: str = 'row_col',
                 device: torch.device = torch.device('mps'),
                 k_syn_init: torch.Tensor = None,
                 ode_method: str = 'rk4'):

        super().__init__()

        self.device = device
        self.connection_type = connection_type
        self.ode_method = ode_method

        # Default parameters
        if parameters is None: parameters = {}
        self.dt = parameters.get('dt', 0.1)
        c_m = parameters.get('c_m', 1.0)
        e_rest = parameters.get('e_rest', 0.0)
        g_m_leak = parameters.get('g_m_leak', 5.0)
        e_syn = parameters.get('e_syn', 20.0)
        R_op_range = parameters.get('R_op_range', 1.0)
        i_bias = parameters.get('i_bias', 0.0)

        self.num_inputs = num_inputs  # number of joints
        self.N = num_neurons_per_joint  # N neurons per joint

        self.num_neurons = self.N * self.N  # Total  NSI neurons in the N x N grid

        # Neuron parameters
        self.c_m = torch.full((self.num_neurons, 1), c_m, device=self.device)
        self.e_rest = torch.full((self.num_neurons, 1), e_rest, device=self.device)
        self.g_m_leak = torch.full((self.num_neurons, 1), g_m_leak, device=self.device)
        self.e_syn_delta = torch.full((self.num_neurons, 1), e_rest + e_syn, device=self.device)
        self.R_op_range = torch.full((self.num_neurons, 1), R_op_range, device=self.device)
        self.i_bias = torch.full((self.num_neurons, 1), i_bias, device=self.device)
        self.v_m = self.e_rest.clone().squeeze(1)

        # k_syn (for input connections to the layer) as a learnable parameter for backprop?

        expected_shape = (self.num_inputs, self.N)

        if k_syn_init is None:
            # initialized default k_syn
            initial_k_syn = torch.ones(expected_shape, device=self.device)
        else:
            initial_k_syn = k_syn_init.to(device=self.device)
        # For NSI, k_syn has a gain for each of the N neurons from the 2 input sources.
        self.k_syn = initial_k_syn

    '''
        synaptic conductance calculation
    '''

    def g_max(self, k_syn_gain_t: torch.Tensor, num_active_inputs: int = 1) -> torch.Tensor:
        g_max_factor_num = k_syn_gain_t * self.R_op_range.T
        g_max_factor_den = self.e_syn_delta.T - g_max_factor_num * num_active_inputs

        if torch.any(g_max_factor_den <= 1e-9):
            raise ValueError("Denominator in g_max is near zero.")
        return g_max_factor_num / g_max_factor_den

    def g_syn(self, u_pre: torch.Tensor, g_max_t: torch.Tensor) -> torch.Tensor:
        activation = torch.clamp(u_pre / self.R_op_range.T, min=0.0, max=1.0)
        return g_max_t * activation

    '''
        connections tuning
    '''

    def _row_col_syn_connection(self, u_pre_0: torch.Tensor, u_pre_1: torch.Tensor,
                                k_syn_0: torch.Tensor, k_syn_1: torch.Tensor, v_m_grid: torch.Tensor) -> torch.Tensor:

        u0_grid = u_pre_0.view(self.N, 1).expand(-1, self.N)
        u1_grid = u_pre_1.view(1, self.N).expand(self.N, -1)

        k0_grid = k_syn_0.view(self.N, 1).expand(-1, self.N)
        k1_grid = k_syn_1.view(1, self.N).expand(self.N, -1)

        g_max0 = self.g_max(k0_grid.flatten().unsqueeze(0), 2)
        g_max1 = self.g_max(k1_grid.flatten().unsqueeze(0), 2)

        g_syn0 = self.g_syn(u0_grid.flatten().unsqueeze(0), g_max0).view(self.N, self.N)
        g_syn1 = self.g_syn(u1_grid.flatten().unsqueeze(0), g_max1).view(self.N, self.N)

        i_syn_0 = g_syn0 * (self.e_syn_delta.view(self.N, self.N) - v_m_grid)
        i_syn_1 = g_syn1 * (self.e_syn_delta.view(self.N, self.N) - v_m_grid)

        i_row_col_syn_total = (i_syn_0 + i_syn_1).flatten().unsqueeze(1)

        return i_row_col_syn_total

    '''
        num integration
    '''

    def _dv_dt(self, v_post, i_syn_values=None):
        i_leak = (v_post - self.e_rest) * self.g_m_leak
        return (self.i_bias + i_syn_values - i_leak) / self.c_m

    '''
        runge-kutta 4th order integration
    '''

    def _rk4_forward_pass(self, i_syn_total_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_syn_total_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_syn_total_values).squeeze(1)
        k3 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k2.unsqueeze(1), i_syn_total_values).squeeze(1)
        k4 = self._dv_dt(v_m_unsqueezed + self.dt * k3.unsqueeze(1), i_syn_total_values).squeeze(1)

        self.v_m += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return self.v_m

    '''
        runge-kutta 2th order integration (modified euler)
    '''

    def _rk2_forward_pass(self, i_syn_total_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_syn_total_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_syn_total_values).squeeze(1)

        self.v_m += self.dt * k2

        return self.v_m

    def forward(self, u_pre_vector: List[torch.Tensor]):
        num_steps = u_pre_vector[0].shape[1]
        # Create a V_output tensor
        v_m_Post = torch.zeros(self.num_neurons, num_steps, device=self.device)

        # should think about how to make this part modifiable so we can have >2 inputs
        u_pre_0, u_pre_1 = u_pre_vector[0], u_pre_vector[1]
        k_syn_0, k_syn_1 = self.k_syn[0, :], self.k_syn[1, :]

        for t in range(num_steps):
            u_pre_0_t, u_pre_1_t = u_pre_0[:, t], u_pre_1[:, t]

            v_m_grid = self.v_m.view(self.N, self.N)

            i_syn_total = self._row_col_syn_connection(u_pre_0=u_pre_0_t, u_pre_1=u_pre_1_t,
                                                       k_syn_0=k_syn_0, k_syn_1=k_syn_1,
                                                       v_m_grid=v_m_grid)

            self.v_m = self._rk2_forward_pass(i_syn_total_values=i_syn_total)
            v_m_Post[:, t] = self.v_m

        return v_m_Post.view(self.N, self.N, num_steps)

class NonSpikingOutput(nn.Module):

    def __init__(self,
                 num_neurons: int,
                 num_steps: int,
                 num_outputs: int = 3,
                 parameters: dict = None,
                 device: torch.device = torch.device('mps'),
                 k_syn_init: torch.Tensor = None,
                 e_syn_init: torch.Tensor = None,
                 ode_method: str = 'rk2'):

        super().__init__()

        self.device = device
        self.ode_method = ode_method
        self.num_outputs = num_outputs
        self.num_steps = num_steps

        # Default parameters
        if parameters is None: parameters = {}
        self.dt = parameters.get('dt', 0.1)
        c_m = parameters.get('c_m', 1.0)
        e_rest = parameters.get('e_rest', 0.0)
        g_m_leak = parameters.get('g_m_leak', 5.0)
        # e_syn = parameters.get('e_syn', 20.0)
        R_op_range = parameters.get('R_op_range', 1.0)
        i_bias = parameters.get('i_bias', 0.0)

        self.num_neurons = num_neurons  # Total  NSI neurons in the N x N grid

        self.expected_shape = (self.num_neurons, self.num_outputs)  # 25x3
        # Neuron parameters
        self.c_m = torch.full(self.expected_shape, c_m, device=self.device)
        self.e_rest = torch.full(self.expected_shape, e_rest, device=self.device)
        self.g_m_leak = torch.full(self.expected_shape, g_m_leak, device=self.device)
        # self.e_syn_delta = torch.full(self.expected_shape, e_rest + e_syn, device=self.device)
        self.e_syn_delta = e_syn_init.to(device=self.device)
        self.R_op_range = torch.full(self.expected_shape, R_op_range, device=self.device)
        self.i_bias = torch.full(self.expected_shape, i_bias, device=self.device)

        self.v_m = self.e_rest.clone().squeeze(1)

        if k_syn_init is None:
            size = (self.num_neurons, self.num_outputs, self.num_steps)
            # initialized default k_syn
            self.k_syn = torch.ones(size, device=self.device)
        else:
            self.k_syn = k_syn_init.to(device=self.device)

    def _g_max(self, k_syn_gain_t: torch.Tensor, e_syn_delta, num_active_inputs: int = 1) -> torch.Tensor:
        g_max_factor_num = k_syn_gain_t * self.R_op_range
        g_max_factor_den = e_syn_delta - g_max_factor_num * num_active_inputs

        # if torch.any(g_max_factor_den <= 1e-9) and torch.any(g_max_factor_den <= -1e-9):
        #     g = self.R_op_range/(e_syn_delta-self.R_op_range)
        #         raise ValueError("Denominator in g_max is near zero.")
        # else:
        g = g_max_factor_num / g_max_factor_den
        return g

    def _g_syn(self, u_pre: torch.Tensor, g_max: torch.Tensor) -> torch.Tensor:
        activation = torch.clamp(u_pre / self.R_op_range, min=0.0, max=1.0)
        return g_max * activation

    def _dv_dt(self, v_post, i_syn_values):
        i_leak = (v_post - self.e_rest) * self.g_m_leak
        return (self.i_bias + i_syn_values - i_leak) / self.c_m

    def _connection(self, u_pre: torch.Tensor,
                    k_syn: torch.Tensor, e_syn: torch.Tensor) -> torch.Tensor:

        g_max = self._g_max(k_syn, e_syn)

        g_syn = self._g_syn(u_pre, g_max)

        i_full_syn_total = g_syn * (e_syn - self.v_m)

        return i_full_syn_total

    '''
        runge-kutta 4th order integration
    '''

    def _rk4_forward_pass(self, i_syn_total_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_syn_total_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_syn_total_values).squeeze(1)
        k3 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k2.unsqueeze(1), i_syn_total_values).squeeze(1)
        k4 = self._dv_dt(v_m_unsqueezed + self.dt * k3.unsqueeze(1), i_syn_total_values).squeeze(1)

        self.v_m += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return self.v_m

    '''
        runge-kutta 2th order integration (modified euler)
    '''

    def _rk2_forward_pass(self, i_syn_total_values: torch.Tensor):

        k1 = self._dv_dt(self.v_m, i_syn_total_values).squeeze(1)
        k2 = self._dv_dt(self.v_m + 0.5 * self.dt * k1, i_syn_total_values).squeeze(1)

        self.v_m += self.dt * k2

        return self.v_m

    def forward(self, u_pre_vector: torch.Tensor):

        v_m_Post = torch.zeros(self.num_outputs, self.num_steps, device=self.device)

        for t in range(self.num_steps):
            u_pre_t = u_pre_vector[:, :, t].flatten().unsqueeze(1).expand(self.expected_shape)

            i_syn_total = self._connection(u_pre=u_pre_t, k_syn=self.k_syn[:, :, t], e_syn=self.e_syn_delta[:, :, t])

            self.v_m = self._rk2_forward_pass(i_syn_total_values=i_syn_total)

            res = self.v_m.sum(dim=0)

            v_m_Post[:, t] = res

        return v_m_Post.view(self.num_outputs, self.num_steps)

    '''
    make k_syn learnable over time? (later)

    calc FK, and then having results for k_syn create vectors for those for x, y and z separately


    so we can reduce num layers for 3 joints
    maybe have a custom loss function

    '''

if __name__ == '__main__':
    # --- Setup (Device, Parameters, Inputs) ---
    # (This part is the same as the previous version)
    # input = torch.jit.script(NonSpikingInput())
    # nsi = torch.jit.script(NonSpikingInterneuron())
    wR1 = LegConfig('R1').legAxes()
    pR1 = LegConfig('R1').legMatConfig()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) accelerator.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")

    # device = torch.device("cpu")
    dt = 1.0
    tMax = 3e3
    t = np.arange(0, tMax, dt)
    numSteps = len(t)
    numJoints = 3
    numNeurons = 10
    numOutputs = 3
    c_m, width, mag = 5.0, 7, 1.0
    delEex, delEin = 20.0, -20.0
    thetaMin, thetaMax = -1.6, 1.6
    jointTheta = np.linspace(thetaMin, thetaMax, numNeurons)

    thetaMat = np.zeros([np.power(numNeurons, numJoints), numJoints])
    pMat = np.zeros([numJoints - 1, np.power(numNeurons, numJoints), 3])
    # jMat: first column is not changing, for other columns there are two velocity components: linear&angular (6 in total)
    jMat = np.zeros([numJoints - 1, np.power(numNeurons, numJoints), 6])
    thetaCombinations = itertools.product(jointTheta, repeat=numJoints)  # Dynamic handling of theta length

    n = 0  # Initialize index
    for combination in thetaCombinations:
        # Store theta values dynamically
        thetaMat[n, :] = combination

        # Compute forward kinematics for full joint configuration
        resultsMat = FrwrdKnmtcsFunc(wR1, thetaMat[n, :], pR1)
        endTranslMat = np.array(resultsMat[0])

        # Store final transformed positions
        pMat[0, n, :] = [endTranslMat[0, 3], endTranslMat[1, 3], endTranslMat[2, 3]]
        if numJoints > 2:
            for index in range(numJoints - 2):
                # Adjust joint selection dynamically for sub matrices
                sub_wR1 = wR1[:, :numJoints - (index + 1)]
                sub_pR1 = pR1[:, :numJoints - (index + 1)]
                sub_theta = np.array(combination[:numJoints - (index + 2)])  # Select appropriate theta values
                # Compute hidden forward kinematics
                [endTranslMat_hidden, _] = FrwrdKnmtcsFunc(sub_wR1, sub_theta, sub_pR1)

                # Compute relative transformation
                R = endTranslMat_hidden[0:3, 0:3]
                pMat[index + 1, n, :] = np.matmul(R.T, endTranslMat[0:3, 3] - endTranslMat_hidden[
                    0:3, 3])  # xyzEnd - xyzHidden
        n += 1  # Increment index

    outputP = np.zeros([np.power(numNeurons, numJoints), 4])

    """
    for all combinations we calculated position of an endpoint
    """

    for j in range(numOutputs):
        pm = pMat[1, :, j]
        outputP[:, j] = (pm - np.min(pm)) / (np.max(pm) - np.min(pm))
        # outputP[:, j] = pm
    outputP[:, -1] = 1.0

    thetaBell = np.zeros([numJoints, thetaMat.shape[0], numNeurons])

    for joint in range(numJoints):
        for i in range(thetaMat.shape[0]):
            thetaBell[joint, i, :] = bell_curve(magnitude=1, theta=thetaMat[i, joint], shift=jointTheta,
                                                width=width)


    nsi = np.zeros([numNeurons * numNeurons, thetaMat.shape[0]])
    kSyn = np.zeros([numNeurons * numNeurons, numOutputs])  # Each row is one nsi compartment, each column is one input scenario
    mag = 1

    for i in range(thetaMat.shape[0]):
        '''
            Generate response of each NSI neuron given a particular combination of input states. Check out ComboVec()
            for more details.
        '''
        in1list = np.reshape(thetaBell[-2, i, :], [1, -1])
        in2list = np.reshape(thetaBell[-1, i, :], [1, -1])

        # neural activity positions
        nsi[:, i] = np.squeeze(
            ComboVec(input1bell=in1list, input2bell=in2list, mag=mag, gMax=Gmax2(1, mag, 20), delE=20))
    kSyn = np.zeros([numNeurons * numNeurons, numOutputs, numSteps])
    delESyn = np.zeros_like(kSyn)

    if numNeurons * numNeurons == thetaMat.shape[0]:
        print('Finding unique solution')
        for i in range(numOutputs):
            res = sp.linalg.solve(nsi, outputP[:, i])
            for j in range(numSteps):
                kSyn[:, i, j] = res
    else:
        print('Finding least squares solution')
        for i in range(numOutputs):
            res = sp.linalg.lstsq(np.transpose(nsi), outputP[:, i])[0]
            for j in range(numSteps):
                kSyn[:, i, j] = res

    for i in range(numNeurons * numNeurons):
        for j in range(numOutputs):
            for k in range(numSteps):
                if kSyn[i, j, k] < 0:
                    delESyn[i, j, k] = delEin
                elif kSyn[i, j, k] > 0:
                    delESyn[i, j, k] = delEex
                else:
                    continue

    T1, T2, T3 = 450, 600, 750
    theta1vec = trajectoryInput(t, T1, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
    theta2vec = trajectoryInput(t, T2, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
    theta3vec = trajectoryInput(t, T3, theta_list=[thetaMin, thetaMax], func_type='sinusoid')


    networkInputs = np.array([np.zeros_like(theta1vec), theta2vec, theta3vec])
    neuron_parameters = {'c_m': c_m, 'e_rest': 0.0, 'g_m_leak': 1.0, 'e_syn': 20.0, 'dt': dt, 'R_op_range': mag}
    inputBellResponsesNP = np.zeros([numJoints, numNeurons, numSteps])
    for joint in range(numJoints):
        for neuron in range(numNeurons):
            inputBellResponsesNP[joint, neuron, :] = (
                bell_curve(magnitude=mag, theta=networkInputs[joint, :], shift=jointTheta[neuron],
                           width=width))

    print("Starting simulation...")
    start = time.time()
    # --- Create Input Layers ---
    '''2nd approach'''
    input_layer_joint1 = torch.jit.script(NonSpikingInput(num_neurons_per_joint=numNeurons,
                                                          parameters=neuron_parameters,
                                                          device=device))
    input_layer_joint2 = torch.jit.script(NonSpikingInput(num_neurons_per_joint=numNeurons,
                                                          parameters=neuron_parameters,
                                                          device=device))

    # --- Create NSI Layer ---

    nsi_layer = torch.jit.script(NonSpikingInterneuron(num_neurons_per_joint=numNeurons,
                                                       connection_type='row_col',
                                                       parameters=neuron_parameters, device=device,
                                                       num_inputs=numJoints - 1, ode_method='rk2'))

    out_layer = NonSpikingOutput(num_neurons=numNeurons * numNeurons, parameters=neuron_parameters,
                                 device=device, ode_method='rk2', num_outputs=3, num_steps=numSteps,
                                 k_syn_init=torch.Tensor(kSyn), e_syn_init=torch.Tensor(delESyn))
    # --- Run Input Layers ---
    v_m_out_joint1 = input_layer_joint1(i_ext_vector=torch.tensor(inputBellResponsesNP[1], device='mps', dtype=torch.float32))
    v_m_out_joint2 = input_layer_joint2(i_ext_vector=torch.tensor(inputBellResponsesNP[2], device='mps', dtype=torch.float32))

    # # --- Run NSI Layer ---
    v_m_to_nsi = [v_m_out_joint1, v_m_out_joint2]

    v_m_out_nsi = nsi_layer(u_pre_vector=v_m_to_nsi)

    v_m_out_final = out_layer(u_pre_vector=v_m_out_nsi)

    print("\nSimulation finished.")
    print(f"--- {time.time() - start:.2f} seconds ---")

    theta_all = np.array([np.zeros_like(theta1vec), theta2vec, theta3vec])
    # Compute Forward Kinematics for all time steps simultaneously
    results = [FrwrdKnmtcsFunc(wR1, theta_all[:, i], pR1) for i in range(len(t))]

    # Separate the endpoints and jacobians into two NumPy arrays.
    endPointActual = np.array([res[0] for res in results])
    ActualXYZ = np.zeros([numSteps, 3])
    # Extract (x, y, z) coordinates efficiently
    for index in range(3):
        ActualXYZ[:, index] = endPointActual[:, index, 3]  # going through x, y, z

    plt.figure()
    plt.plot(t, ActualXYZ[:, 0])
    plt.plot(t, ActualXYZ[:, 1])
    plt.plot(t, ActualXYZ[:, 2])
    plt.xlabel('time, ms ')
    plt.ylabel('position, mm')
    plt.title('FK actual')
    plt.show()

    plt.figure()
    plt.plot(kSyn[:, 0, 0])
    plt.title('X torch')
    plt.show()

    plt.figure()
    plt.plot(kSyn[:, 1, 0])
    plt.title('Y torch')
    plt.show()
    plt.figure()
    plt.plot( kSyn[:, 2, 0])
    plt.title('Z torch')
    plt.show()

    out_1 = v_m_out_final[0, :].cpu().detach().numpy()
    out_2 = v_m_out_final[1, :].cpu().detach().numpy()
    out_3 = v_m_out_final[2, :].cpu().detach().numpy()
    plt.figure()
    plt.plot(t, out_1)
    plt.plot(t, out_2)
    plt.plot(t, out_3)
    plt.xlabel('time, ms ')
    plt.ylabel('voltage, mV')
    plt.title('FK output net')
    plt.show()

    # --- Plotting Results ---
    fig, axs = plt.subplots(3, 2, figsize=(22, 24), gridspec_kw={'height_ratios': [1, 1, 1.5]})
    fig.suptitle('Two-Layer Network Simulation', fontsize=16)

    # Plot joint angles
    for j in range(numJoints - 1):
        axs[0, 0].plot(t, networkInputs[j, :], label=f'Input joint {j}')
    axs[0, 0].set_title('Joint Angles Inputs')
    axs[0, 0].set_ylabel('Angles (rad)')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].grid(True)
    # axs[0, 0].set_xlim([0, 2e3])
    axs[0, 0].legend()

    # Plot a few neurons from the NSI grid over time
    axs[0, 1].set_title('NSI Layer - Membrane Potentials')
    for i in range(0, numNeurons):
        for j in range(0, numNeurons):
            axs[0, 1].plot(t, v_m_out_nsi[i, j, :].cpu().detach().numpy(), label=f'V_m of NSI Neuron ({i},{j})')
    axs[0, 1].set_ylabel('Voltage (mV)')
    axs[0, 1].set_xlabel('Time (ms)')
    axs[0, 1].grid(True)

    axs[0, 1].set_xlim([0, 2000])
    axs[0, 1].legend()

    # Plot input layer 1 voltages
    axs[1, 0].set_title('Input Layer 1 - Postsynaptic Potentials')
    for neuron in range(0, numNeurons):
        axs[1, 0].plot(t, v_m_out_joint1[neuron, :].cpu().detach().numpy())
    axs[1, 0].set_ylabel('Voltage (mV)')
    axs[1, 0].set_xlabel('Time (ms)')
    # axs[1, 0].set_xlim([0, 2e3])
    axs[1, 0].grid(True)

    # Plot input layer 2 voltages
    axs[1, 1].set_title('Input Layer 2 - Postsynaptic Potentials')
    for neuron in range(0, numNeurons):
        axs[1, 1].plot(t, v_m_out_joint2[neuron, :].cpu().detach().numpy())
    axs[1, 1].set_ylabel('Voltage (mV)')
    axs[1, 1].set_xlabel('Time (ms)')
    # axs[1, 1].set_xlim([0, 2e3])
    axs[1, 1].grid(True)

    # Plot input layer 2 voltages
    axs[2, 0].set_title('OUTPUT positions XYZ')
    for neuron in range(0, numNeurons):
        axs[2, 0].plot(t, v_m_out_final[0, :].cpu().detach().numpy())
        axs[2, 0].plot(t, v_m_out_final[1, :].cpu().detach().numpy())
        axs[2, 0].plot(t, v_m_out_final[2, :].cpu().detach().numpy())
    axs[2, 0].set_ylabel('Voltage (mV)')
    axs[2, 0].set_xlabel('Time (ms)')
    # axs[1, 1].set_xlim([0, 2e3])
    axs[2, 0].grid(True)
    axs[2, 1].set_title('FK Actual')
    for neuron in range(0, numNeurons):
        axs[2, 1].plot(t, ActualXYZ[:, 0])
        axs[2, 1].plot(t, ActualXYZ[:, 1])
        axs[2, 1].plot(t, ActualXYZ[:, 2])
    axs[2, 1].set_ylabel('Voltage (mV)')
    axs[2, 1].set_xlabel('Time (ms)')
    # axs[1, 1].set_xlim([0, 2e3])
    axs[2, 1].grid(True)
    # Add a heatmap of NSI activity at a specific time
    # time_point_to_plot = int(numSteps / 4)
    # ax_heatmap = plt.subplot(212)  # Span the whole bottom row
    # fig.add_axes(ax_heatmap)
    # # ax_heatmap.set_position([0.125, 0.1, 0.775, 0.25])  # Manually position it
    # im = ax_heatmap.imshow(v_m_out_nsi[:, :, time_point_to_plot].cpu().detach().numpy(),
    #                        cmap='viridis', aspect='auto', origin='lower',
    #                        extent=[0, numNeurons - 1, 0, numNeurons - 1])
    # ax_heatmap.set_title(f'NSI Grid Activity Heatmap at t = {t[time_point_to_plot]} ms')
    # ax_heatmap.set_xlabel('Neuron Index (from Input 2)')
    # ax_heatmap.set_ylabel('Neuron Index (from Input 1)')
    # fig.colorbar(im, ax=ax_heatmap, label='Voltage (mV)')

    # Remove the now-empty axes
    # fig.delaxes(axs[1, 0])
    # fig.delaxes(axs[1, 1])

    # Set shared x-limits for time plots
    for i in range(2):
        for j in range(2):
            if (i, j) not in [(2, 0), (2, 1)]:
                axs[i, j].set_xlim([0, tMax / 2])

    plt.tight_layout(rect=[0, 0.3, 1, 0.96])
    plt.show()

    plt.figure()
    plt.plot(t, ((ActualXYZ[:, 0] - min(ActualXYZ[:, 0])) / (max(ActualXYZ[:, 0]) - min(ActualXYZ[:, 0]))))
    plt.plot(t, ((out_1 - min(out_1)) / (max(out_1) - min(out_1))))
    plt.plot(t, ((ActualXYZ[:, 1] - min(ActualXYZ[:, 1])) / (max(ActualXYZ[:, 1]) - min(ActualXYZ[:, 1]))))
    plt.plot(t, ((out_2 - min(out_2)) / (max(out_2) - min(out_2))))
    plt.xlabel('time, ms ')
    plt.ylabel('normalized position')
    plt.title('Out vs Actual, X and Y coords')
    plt.show()
    plt.figure()
    plt.plot(t, ((ActualXYZ[:, 1] - min(ActualXYZ[:, 1])) / (max(ActualXYZ[:, 1]) - min(ActualXYZ[:, 1]))))
    plt.plot(t, ((out_2 - min(out_2)) / (max(out_2) - min(out_2))))
    plt.xlabel('time, ms ')
    plt.ylabel('normalized position')
    plt.title('Out vs Actual, Y coord')
    plt.show()
    plt.figure()
    plt.plot(t, ((ActualXYZ[:, 2] - min(ActualXYZ[:, 2])) / (max(ActualXYZ[:, 2]) - min(ActualXYZ[:, 2]))))
    plt.plot(t[1:], ((out_3[1:] - min(out_3[1:])) / (max(out_3[1:]) - min(out_3[1:]))))
    plt.xlabel('time, ms ')
    plt.ylabel('normalized position')
    plt.title('Out vs Actual, Z coord')
    plt.show()
    # clean VRAM and cache, delete all tensors/layers
    del networkInputs
    del inputBellResponsesNP
    del nsi_layer
    del v_m_to_nsi
    del v_m_out_nsi
    del v_m_out_joint1
    del v_m_out_joint2
    del input_layer_joint1
    del input_layer_joint2
    del v_m_out_final
    del out_layer
    torch.mps.empty_cache()

'''
to do 
compare sns NSI outputs to this network - done
'''

'''
to do 
output part with NSI tuning
allow backprop for Kgain?

'''
'''
to do 
compare sns NSI outputs to this network - done
'''

'''
to do 
output part with NSI tuning
allow backprop for Kgain?

'''

'''
change C_M
find a "bandwidth" c_m values
'''
