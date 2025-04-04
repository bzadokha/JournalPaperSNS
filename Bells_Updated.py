import numpy as np
import scipy as sp
from sns_toolbox.connections import NonSpikingSynapse
import sns_toolbox.networks
import matplotlib.pyplot as plt
from sns_toolbox.neurons import NonSpikingNeuron
import itertools
import sklearn
import time

'''
Bohdan Zadokha and Nicholas Szczecinski
Feb 10 2024
WVU NeuroMINT
Simplified network that calculates FK (x, y, z) or jacobian
using inputs from 3 joints
'''

'''
Generalized network function that can design the whole network for n-joints (single leg)
'''
start = time.time()
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
        L4 = L1/2
        L5 = L2/3
        L6 = L3/2


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
            p6R1 = np.array([[x0 + (L1+L4) * np.cos(45 * np.pi / 180)],
                             [-y02 - (L1+L4) * np.cos(45 * np.pi / 180)],
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

def RodriguesFunc(w, theta, p):
    Scross = np.cross(-w, p)
    Smatrix = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])  # Skew-symmetric
    Rmatrix = np.eye(3, dtype=float) + np.dot(Smatrix, np.sin(theta)) + np.dot(np.matmul(Smatrix, Smatrix),
                                                                               (1 - np.cos(theta)))
    c = np.matmul(np.matmul(np.eye(3, dtype=float) - Rmatrix, Smatrix), Scross)
    d = np.matmul(w, w.reshape((3, 1)))
    e = np.dot(theta, np.dot(d.item(), Scross))
    TranslVec = c + e
    f = np.concatenate((Rmatrix, TranslVec.reshape((3, 1))), axis=1)
    expoFinal = np.concatenate((f, np.array([[0, 0, 0, 1]])), axis=0)
    return expoFinal

def FrwrdKnmtcsFunc(w, thetaValue, p):
    translMatArray = np.zeros([thetaValue.shape[0], 4, 4], dtype=float)
    expoOne = np.eye(4, dtype=float)
    if hasattr(thetaValue, "__len__"):
        for index in range(len(thetaValue)):
            expo = RodriguesFunc(w[:, index], thetaValue[index], p[:, index])
            expoOne = np.matmul(expoOne, expo)
            translMatArray[index, :, :] = expoOne
    else:
        expo = RodriguesFunc(w[:, 0], thetaValue, p[:, 0])
        expoOne = np.matmul(expoOne, expo)
        translMatArray[:, :] = expoOne
    positionM = np.atleast_2d(p[:, -1]).T
    M = np.concatenate((np.concatenate((np.eye(3, dtype=float), positionM), axis=1),
                        np.array([[0, 0, 0, 1]])), axis=0)
    TranslMat = np.matmul(expoOne, M)

    JacobianMatrix = JacobianFunc(w, p, translMatArray)
    return (TranslMat, JacobianMatrix)

def JacobianFunc(w, p, TmatArray):#R1, R2, p1Final, p2Final):
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
    pSkew = np.zeros([numJoints-1, 3, 3])
    Rmat = np.zeros([numJoints-1, 3, 3])
    adjointMat = np.zeros([numJoints-1, 6, 6])

    for joint in range(numJoints):
        sCross[:, joint] = np.cross(-w[:, joint], p[:, joint])
        Jmat[:, joint] = np.concatenate((sCross[:, joint], w[:, joint]), axis=0)

    for joint in range(numLayers):
        pSkew[joint, :, :] = SkewMatrix(TmatArray[joint, 0:3, 3])
        Rmat[joint, :, :] = np.matmul(pSkew[joint, :, :], TmatArray[joint, 0:3, 0:3])
        adjointMat[joint, :, :] = np.concatenate((np.concatenate((TmatArray[joint, 0:3, 0:3], Rmat[joint, :, :]), axis=1),
                                    np.concatenate((np.zeros([3, 3]), TmatArray[joint, 0:3, 0:3]), axis=1)), axis=0)  # 6x6

    jac = np.zeros([numLayers, 6, 1])
    for joint in range(numJoints-1):
        jac[joint, :, :] = np.matmul(adjointMat[joint, :, :], Jmat[:, joint+1].reshape((6, 1))) #6x1

    jacobianMatrix = Jmat[:, 0].reshape((6, 1))
    for joint in range(numLayers):
        jacobianMatrix = np.concatenate((jacobianMatrix, jac[joint, :, :]), axis=1)
    return jacobianMatrix

def sigm_curve(magnitude, width, theta, shift, jointThetaDiff):
    return magnitude / 2 * (1 + np.tanh(width * (theta - shift + jointThetaDiff / 2)))

def bell_curve(magnitude, width, theta, shift):
    return magnitude * np.exp(-width * pow((theta - shift), 2))

def rmse(pred, actual):
    result = np.sqrt(((actual - pred) ** 2).mean())
    return result

def Gmax(k, R, delE):
    maxConduct = k * R / (delE - k * R)
    return maxConduct

def Gmax2(k, R, delE):
    maxConduct = k * R / (delE - 2 * k * R)
    return maxConduct

def Normalize(matrix, minNorm, maxNorm):
    vector = np.matrix(matrix).getA1()
    vectorNorm = ((vector - min(vector)) / (max(vector) - min(vector))) * (maxNorm - minNorm) + minNorm
    return vectorNorm, max(vector), min(vector)

def ComboVec(input1bell, input2bell, mag, gMax, delE):
    # position
    if np.ndim(input2bell) == 1:
        input2bell = np.reshape(input2bell, [1, -1])

    # temp_mat, comboSS, comboVec are the old version. This should be removed.
    temp_mat = np.transpose(input1bell + np.transpose(input2bell) - mag)
    comboSS = np.reshape(temp_mat, [-1, 1])
    comboSS[comboSS < 0] = 0
    comboSS[comboSS > mag] = mag
    comboVec = gMax * comboSS / mag * delE / (1 + 2 * gMax)
    comboVec[comboVec < 0] = 0

    temp_mat2 = np.transpose((input1bell/mag*gMax*delE + np.transpose(input2bell)/mag*gMax*delE)/(1 + gMax*input1bell/mag + gMax*np.transpose(input2bell)/mag)) - mag
    comboSS2 = np.reshape(temp_mat2, [-1, 1])
    comboSS2[comboSS2 < 0] = 0
    comboSS2[comboSS2 > mag] = mag
    comboVec2 = comboSS2
    return comboVec2

def SkewMatrix(x):
    skew = [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]
    return skew

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
        num_steps = int(np.ceil(tmax/T))
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

def synTuning(netModel, nsiNeuron, inputData1, inputData2, outputData, inputStr1, inputStr2, outputStr,
                        nsiStr, gI, synI, dEex, dEin):
    '''
    We gotta pass in the "bellCurve" data, so this function can focus on setting parameter values. When too many
    processes are bundled up into one function, we can't reuse useful pieces! :)

    :param netModel: model object
    :param nsiNeuron: non-spiking neuron object from the SNS-toolbox
    :param inputData1: 2D matrix of inputs with shape (num data points, num input neurons)
    :param inputData2: 2D matrix of inputs with shape (num data points, num input neurons)
    :param outputData: 2D matrix of inputs with shape (num data points, num output neurons)
    :param inputStr1: String name of input 1's neurons, used to add synapses to the network. This function assumes the
    input neurons are named like inputStr1_0, inputStr1_1, etc.
    :param inputStr1: String name of input 2's neurons, used to add synapses to the network. This function assumes the
    input neurons are named like inputStr2_0, inputStr2_1, etc.
    :param outputStr: String name of output neurons, used to add neurons (outputStr_0, outputStr_1, etc.) to the
    network, and to add synapses from the "grid" to the output.
    :param nsiStr: String name of neurons in the "grid". These will be named nsiStr_0_0, nsiStr_0_1, etc.
    :param gI: synaptic conductance = Gmax|k=1
    :param synI: synaptic connection object
    :param dEex: excitatory synapse potential
    :param dEin: inhibitory synapse potential
    :return: synaptic conductance surfaces, to_simulate values, and the indices of the output neurons (which this function will add).
    '''
    if inputData1.shape[0] != inputData2.shape[0]:
        raise Exception('inputData1 and inputData2 must have the same number of rows.')
    else:
        numDataPoints = inputData1.shape[0]
        numInputs1 = inputData1.shape[1]
        numInputs2 = inputData2.shape[1]

    if outputData.shape[0] != inputData1.shape[0]:
        raise Exception('inputData1, inputData2, and outputData must have the same number of rows.')
    else:
        numOutputs = outputData.shape[1]

    '''
    We will project inputData1 and inputData2 into the "grid" of NSIs. NSIs will connect all-to-all to the outputs
    '''
    nsi = np.zeros([numInputs1*numInputs2, numDataPoints])  # Each row is one nsi compartment, each column is one input scenario
    kSyn = np.zeros([numInputs1*numInputs2, numOutputs])  # Each row is one synapse from nsi, each column is the output neuron
    output_inds = np.zeros(numOutputs, dtype=int)  # np array to store the indices of the output neurons from this operation.
    mag = 1
    for i in range(numDataPoints):
        '''
            Generate response of each NSI neuron given a particular combination of input states. Check out ComboVec()
            for more details.
        '''
        in1list = np.reshape(inputData1[i, :], [1, -1])
        in2list = np.reshape(inputData2[i, :], [1, -1])

        # neural activity positions
        nsi[:, i] = np.squeeze(ComboVec(input1bell=in1list, input2bell=in2list, mag=mag, gMax=gI, delE=dEex))

    '''
    solving a linear equation nsi*k=output -> k = nsi^(-1)*output
    '''

    if numInputs1 * numInputs2 == numDataPoints:
        print('Finding unique solution')
        for i in range(numOutputs):
            kSyn[:, i] = sp.linalg.solve(nsi, outputData[:, i])
    else:
        '''
        If there are more datapoints than synapses, solve using least squares. This avoids overfitting.
        '''
        print('Finding least squares solution')
        for i in range(numOutputs):
            kSyn[:, i] = sp.linalg.lstsq(np.transpose(nsi), outputData[:, i])[0]

            # plt.figure()
            # plt.plot(outputData[:, i], np.matmul(np.transpose(nsi), kSyn[:, i]), '.', label='nsi * ksyn')
            # temp_x = np.array([np.min(outputData[:, i]), np.max(outputData[:, i])])
            # plt.plot(temp_x, temp_x, label='ideal fit')
            # plt.xlabel('desired network output ("training" data)')
            # plt.ylabel('steady-state network output')
            # plt.legend()
            # plt.title('fit for output ' + str(i))
        # plt.show()

    # Use kSyn values to select synaptic reversal potenteials and compute synaptic conductances.
    if np.any(kSyn > dEex / mag):
        to_simulate = False
        print('IMPOSSIBLY HIGH GAINS, CANNOT SIMULATE.')
    else:
        to_simulate = True
        print('Valid gains, will simulate.')

    '''
    Build the SNS-Toolbox model. Add the NSI neurons in a grid. Follow the naming provided by the user. All the synapses
    from input neurons to the central grid have the same "identity" tuning.
    '''
    for i in range(numInputs1):
        for j in range(numInputs2):
            nameStr = nsiStr + '_' + str(i) + '_' + str(j)  # between theta2 and theta3 (1st layer)

            # from input sensory neurons to NSIs 1st layer
            netModel.add_neuron(neuron_type=nsiNeuron, name=nameStr)

            netModel.add_connection(synI, source=inputStr1 + '_' + str(i), destination=nameStr)
            try:
                netModel.add_connection(synI, source=inputStr2 + '_' + str(j), destination=nameStr)
            except ValueError:
                print('caught an error')

            netModel.add_output(source=nameStr)

    # Matrices to hold synaptic conductance values and reversal potentials.
    g = np.zeros([numInputs1, numInputs2, numOutputs])
    dE = np.zeros([numInputs1, numInputs2, numOutputs])

    # Loop through the NSI neurons and add their synapses to the output neurons.
    n = 0
    for i in range(numInputs1):
        for j in range(numInputs2):
            sourceNameStr = nsiStr + '_' + str(i) + '_' + str(j)
            for k in range(numOutputs):
                # Name the output neuron
                destNameStr = outputStr + '_' + str(k)
                if i == 0 and j == 0:
                    # If this is the first time making output connections from the NSI grid, create the output neuron
                    netModel.add_neuron(neuron_type=nsiNeuron, name=destNameStr)
                    output_inds[k] = int(netModel.get_num_outputs())
                    netModel.add_output(source=destNameStr, name=destNameStr)
                    print('Neuron ' + destNameStr + ' is output number ' + str(output_inds[k]))

                '''
                    SNS-Toolbox does not enable synapses with max_cond = 0, so if kSyn = 0, we must pass it machine epsilon instead.
                '''
                if kSyn[n, k] > 0:
                    dE[i, j, k] = dEex
                elif kSyn[n, k] < 0:
                    dE[i, j, k] = dEin
                else:
                    continue

                '''
                Synapses from the combo neurons to the output neuron(s) each have unique conductance value that corresponds to
                the desired output in that scenario. Note that the e_lo for the synapses = mag and e_hi = 2*mag, because
                multiple combo neurons will be active at any time, but we only care about the most active, so setting e_lo this 
                way serves as a threshold mechanism.
                '''
                g[i, j, k] = Gmax(kSyn[n, k], mag, dE[i, j, k])

                if g[i, j, k] > 0:
                    synOutput = NonSpikingSynapse(max_conductance=float(g[i, j, k]), reversal_potential=float(dE[i, j, k]),
                                                  e_lo=mag, e_hi=2 * mag)
                    netModel.add_connection(synOutput, source=sourceNameStr, destination=destNameStr)
                else:
                    print('Negative g value encountered; synapse not added. i = ' + str(i) + ', j = ' + str(j) + ', k = ' + str(k))

            #  n only increases when i and j increment, not when k increments.
            n += 1
    return to_simulate, g, output_inds

def normVec(vec):
    normalized = ((vec - min(vec)) / (max(vec) - min(vec)))
    return normalized

pR1 = LegConfig('R1').legMatConfig()
pR2 = LegConfig('R2').legMatConfig()
pR3 = LegConfig('R3').legMatConfig()
pL1 = LegConfig('L1').legMatConfig()
pL2 = LegConfig('L2').legMatConfig()
pL3 = LegConfig('L3').legMatConfig()


wR1 = LegConfig('R1').legAxes()
wR2 = LegConfig('R2').legAxes()
wR3 = LegConfig('R3').legAxes()
wL1 = LegConfig('L1').legAxes()
wL2 = LegConfig('L2').legAxes()
wL3 = LegConfig('L3').legAxes()

# time vec
dt = 0.1
tMax = 3e3
t = np.arange(0, tMax, dt)
numSteps = len(t)

# network's parameters

numJoints = 3
numNeuronsInit = 9  # 9
thetaMin = -1.6
thetaMax = 1.6
jointTheta = np.linspace(thetaMin, thetaMax, numNeuronsInit)
jointThetaDiff = (thetaMax - thetaMin) / (numNeuronsInit - 1)
mem_capacitance = 2
widthbellInit = 4  # 7
delaySlow = 1000

# inputs
T1 = 450
T2 = 600  # 99.8 # 99.8
T3 = 750  # 102.5 # 102.5

theta1vec = trajectoryInput(t, T1, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta2vec = trajectoryInput(t, T2, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta3vec = trajectoryInput(t, T3, theta_list=[thetaMin, thetaMax], func_type='sinusoid')

networkInputs = np.squeeze(np.array([[theta1vec.transpose()], [theta2vec.transpose()], [theta3vec.transpose()]]))

def theNetwork(netName, numJoints, numNeurons, delay, widthCurve, jointInputs, jointTheta, wConfig, pConfig):

    if numJoints < 2:
        print("The number of joints should be at least 2.")
        return 0
    else:
        wR1 = wConfig
        pR1 = pConfig
        mag = 1
        delEex = 20
        delEin = -20
        kIdent = 1
        numLayers = numJoints - 1
        numHiddenLayers = numJoints - 2
        '''
            Create sensory input matrices for n joints. This matrix stores bell responses for all joints
        '''
        inputs = np.zeros([numJoints, numSteps, numNeurons])
        '''
            Create an input vector for a network of n joints. 
            It requires to 'squeeze' all inputs into one-dimensional vector by numSteps.
        '''
        inputNet = np.zeros([numSteps, numNeurons * numJoints])
        '''
        Actual X, Y, Z values as a function of time. We need a n-1 numJoints surfaces.
        First surface is a xyz result of a whole leg.
        All next surfaces are xyz positions of a numJoint - 2 frames
        '''
        ActualXYZ = np.zeros([numLayers, numSteps, 3])
        ActualJacobian = np.zeros([numSteps, 6*numLayers])
        '''
            For the network to perform the desired calculation, the synapses from each combo neuron to the output neuron should have
            effective gains (i.e., Vout/Vcombo) that are proportional to the value that the output neuron encodes. Here, we take all
            the "training data", that is, the actual X coordinate of the leg, normalize it to values between 0 and 1, and use those
            to set the unique properties of the synapse from each combo neuron to the output neuron.
        '''

        for joint in range(numJoints):
            for neuron in range(numNeurons):
                inputs[joint, :, neuron] = bell_curve(magnitude=mag, theta=jointInputs[joint, :], shift=jointTheta[neuron], width=widthCurve)

        for i in range(numSteps):
            inputNet[i, :] = np.concatenate(inputs[:, i, :], axis=None)

        theta_all = jointInputs[:numJoints, :]  # Get all joint inputs dynamically

        # Compute Forward Kinematics for all time steps simultaneously
        results = [FrwrdKnmtcsFunc(wR1, theta_all[:, i], pR1) for i in range(len(t))]

        # Separate the endpoints and jacobians into two NumPy arrays.
        endPointActual = np.array([res[0] for res in results])
        jacobianActual = np.array([res[1] for res in results])

        # Extract (x, y, z) coordinates efficiently
        for index in range(3):
            ActualXYZ[0, :, index] = endPointActual[:, index, 3]  # going through x, y, z

        for i in range(numSteps):
            ActualJacobian[i, :] = np.matrix(jacobianActual[i, :, 1:numJoints].transpose()).getA1()

        # If numJoints > 2, compute all hidden transformations efficiently
        if numJoints > 2:
            for layer in range(numHiddenLayers):
                # Extract relevant sub-matrices
                sub_wR1 = wR1[:, :numJoints - (layer + 1)]
                sub_pR1 = pR1[:, :numJoints - (layer + 1)]
                sub_theta_all = theta_all[:numJoints - (layer + 2), :]  # Dynamically slice theta

                # Compute all hidden layer transformations in one go
                results_hidden = [FrwrdKnmtcsFunc(sub_wR1, sub_theta_all[:, i], sub_pR1) for i in range(len(t))]

                # Separate the endpoint and jacobian into two NumPy arrays.
                endTranslMat_hidden = np.array([res_hidden[0] for res_hidden in results_hidden])

                # Store hidden positions efficiently
                for index in range(3):
                    ActualXYZ[layer + 1, :, index] = endTranslMat_hidden[:, index, 3]

                # Compute relative transformations using matrix multiplications
                R_hidden = endTranslMat_hidden[:, 0:3, 0:3]  # Extract rotation matrices
                pos_diff = endPointActual[:, 0:3, 3] - endTranslMat_hidden[:, 0:3, 3]  # Compute (xyzEnd - xyzHidden)

                # Efficient batch-wise transformation using Einstein summation (fastest method)
                ActualXYZ[layer + 1, :, :] = np.einsum("bij,bj->bi", R_hidden.transpose(0, 2, 1), pos_diff)
        '''
            Create a network object.
        '''
        net = sns_toolbox.networks.Network(name=netName)
        '''
           Define key neuron and synapse parameters. 
           'mag' is like R in the functional subnetwork publications: it is the maximum expected neural activation. 
           delE is the synaptic reversal potential relative to the rest potential of the neurons.
        '''
        '''
            Create a basic nonspiking neuron type.
        '''

        # gIdent = Gmax(kIdent, mag, delEex)  # From Szczecinski, Hunt, and Quinn 2017
        gIdent = Gmax2(kIdent, mag, delEex)  # UPDATED: solve for g to make postsyn neuron U = 2*mag.

        identitySyn = NonSpikingSynapse(max_conductance=gIdent, reversal_potential=delEex, e_lo=0, e_hi=mag)

        bellNeuron = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1)

        NSIneuron = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1, bias=0)
        '''
            Create the "bell" neurons, which are the sensory neurons. Each sensory neuron has a bell-curve receptive field
            with a unique "preferred angle", that is, the angle at which the peak response is evoked.
        '''
        for joint in range(numJoints):
            # input neurons
            for neuron in range(numNeurons):
                name = 'Bell_' + str(joint+1) + '_' + str(neuron)
                net.add_neuron(neuron_type=bellNeuron, name=name)
                net.add_input(name)
                net.add_output(name)
                print(name)
        '''
            Each sensory neuron synapses onto a number of "NSI" neurons. Each NSI neuron receives synaptic input from one
            sensory neuron from each joint. All these synapses may be identical, because they are simply generating all the possible
            combinations of the joint angles/independent variables. All the combo neurons may be identical, because they are simply
            integrating the joint angles.
        '''
        '''
            widthCurve is the width of the sensory-encoding activation functions. 
            Note that these have been normalized to the range of theta and the number of sensory neurons. 
            Also note that smaller value of widthCurve makes the curves broader.
        '''
        '''
            Initialize empty arrays to store input current for each set of sensory neurons (theta2 = input2, theta3 = input3), the
            input current for the whole network (inputX), and the validation data X(t).
        '''
        '''
            Create the "training data" for the network, that is, the known X, Y, Z coordinates given the lists of theta values.
            For 3+ joints we need to generate surfaces for each joint 1 angles: for X and Y coordinates
            Z axis does not depend on joint1
        '''
        thetaMat = np.zeros([np.power(numNeurons, numJoints), numJoints])
        pMat = np.zeros([numLayers, np.power(numNeurons, numJoints), 3])
        # jMat: first column is not changing, for other columns there are two velocity components: linear&angular (6 in total)
        jMat = np.zeros([numLayers, np.power(numNeurons, numJoints), 6])
        thetaCombinations = itertools.product(jointTheta, repeat=numJoints)  # Dynamic handling of theta length

        n = 0  # Initialize index
        for combination in thetaCombinations:
            # Store theta values dynamically
            thetaMat[n, :] = combination

            # Compute forward kinematics for full joint configuration
            resultsMat = FrwrdKnmtcsFunc(wR1, thetaMat[n, :], pR1)
            endTranslMat = np.array(resultsMat[0])
            jacMat = np.array(resultsMat[1])

            # Store final transformed positions
            pMat[0, n, :] = [endTranslMat[0, 3], endTranslMat[1, 3], endTranslMat[2, 3]]
            for joint in range(1, numJoints):
                jMat[joint-1, n, :] = np.matrix(jacMat[:, joint:joint+1].transpose()).getA1()
            #jMat[1, n, :] = np.matrix(jacMat[:, 1:numJoints].transpose()).getA1()
            # If numJoints > 2, compute hidden transformations dynamically
            if numJoints > 2:
                for index in range(numHiddenLayers):
                    # Adjust joint selection dynamically for sub matrices
                    sub_wR1 = wR1[:, :numJoints - (index + 1)]
                    sub_pR1 = pR1[:, :numJoints - (index + 1)]
                    sub_theta = np.array(combination[:numJoints - (index + 2)])  # Select appropriate theta values
                    # Compute hidden forward kinematics
                    [endTranslMat_hidden, _] = FrwrdKnmtcsFunc(sub_wR1, sub_theta, sub_pR1)

                    # Compute relative transformation
                    R = endTranslMat_hidden[0:3, 0:3]
                    pMat[index + 1, n, :] = np.matmul(R.T, endTranslMat[0:3, 3] - endTranslMat_hidden[0:3, 3])  # xyzEnd - xyzHidden
            n += 1  # Increment index

        jMatLinear = jMat[:, :, 0:3]
        jMatAngular = jMat[:, :, 3:]
        jMatLinNorm = np.zeros_like(jMatLinear)
        jMatAngNorm = np.zeros_like(jMatAngular)

        '''normalize jacobian columns'''
        for layer in range(numLayers):
            for column in range(numJoints):
                jMatLinNorm[layer, :, :] = sklearn.preprocessing.minmax_scale(jMatLinear[layer, :, :], axis=0)
                jMatAngNorm[layer, :, :] = sklearn.preprocessing.minmax_scale(jMatAngular[layer, :, :], axis=0)

        thetaBell = np.zeros([numJoints, thetaMat.shape[0], numNeurons])
        bellShift = jointTheta

        for joint in range(numJoints):
            for i in range(thetaMat.shape[0]):
                thetaBell[joint, i, :] = bell_curve(magnitude=mag, theta=thetaMat[i, joint], shift=bellShift, width=widthCurve)

        outputHiddenP = np.zeros([numHiddenLayers, np.power(numNeurons, numJoints), 4])
        outputJ = np.zeros([numLayers, np.power(numNeurons, numJoints), 6])
        outputHiddenP[:, :,-1] = 1.0

        # layerNorm = numHiddenLayers
        # for layerJ in range(numLayers):
        #     outputJ[layerJ, :, :] = np.concatenate
        #     layerNorm -=1
        '''
        jacobian 
        1st layer -1
        2nd layer -2
        etc
        '''
        '''modify synTuning such that it has 3 outputs: position, lin velocity, ang velocity'''

        outputP = np.zeros([np.power(numNeurons, numJoints), 3])

        for j in range(pMat.shape[-1]):
            pm = pMat[0, :, j] # translational mat (go through columns)
            outputP[:, j] = (pm - np.min(pm)) / (np.max(pm) - np.min(pm)) # only for the final layer


        for i in range(numHiddenLayers):
            for j in range(pMat.shape[-1]):
                pm = pMat[i+1, :, j]  # translational mat (go through columns)
                outputHiddenP[i, :, j] = (pm - np.min(pm)) / (np.max(pm) - np.min(pm))

        for layer in range(numLayers):
            if layer == 0:
                input1data = thetaBell[-1, :, :]
                input1string = 'Bell_' + str(numJoints)
                input2data=thetaBell[-2, :, :]
                input2string='Bell_'+str(numLayers)
                outputDataNetP = outputHiddenP[layer, :, :]
                output1strP = 'p_' + str(layer)
            elif layer == 1 and layer != numHiddenLayers:
                input1data = thetaBell[-3, :, :]
                input1string = 'Bell_' + str(numJoints-2)
                input2data = outputHiddenP[layer-1, :, :]
                input2string = output1strP
                outputDataNetP = outputHiddenP[layer, :, :]
                # outputDataNetP = sklearn.preprocessing.minmax_scale(outputDataNetP, axis=0)
                output1strP = 'p_' + str(layer)
            elif layer == numHiddenLayers:
                input1data = thetaBell[-3, :, :]
                input1string = 'Bell_' + str(numJoints-(layer+1))
                input2data = outputHiddenP[-1, :, :]
                input2string = output1strP
                outputDataNetP = outputP[:, :]
                # outputDataNetP = sklearn.preprocessing.minmax_scale(outputDataNetP, axis=0)
                output1strP = 'p_Final' + str(layer)
            else:
                input1data = thetaBell[numJoints-layer, :, :]
                input1string = 'Bell_' + str(numJoints-(layer+1))
                input2data = outputHiddenP[layer - 1, :, :]
                input2string = output1strP
                outputDataNetP = outputHiddenP[layer, :, :]
                # outputDataNetP = sklearn.preprocessing.minmax_scale(outputDataNetP, axis=0)
                output1strP = 'p_' + str(layer)
            print('Layer_'+str(layer))
            print('input1 = '+ input1string)
            print('input2 = '+ input2string)
            print('output = ' + output1strP)
            nsiStr = 'nsi_' + str(layer)
            '''positions part'''

            to_simulate, g1, output_inds = synTuning(netModel=net, nsiNeuron=NSIneuron,
                                                        inputData1=input1data,
                                                        inputData2=input2data,
                                                        inputStr1=input1string,
                                                        inputStr2=input2string,
                                                        outputData=outputDataNetP,
                                                        outputStr=output1strP,
                                                        nsiStr=nsiStr, gI=gIdent, synI=identitySyn, dEex=delEex, dEin=delEin)

            # Simulate the network response
        model = net.compile(backend='numpy', dt=dt)
        numOut = net.get_num_outputs()
        PredictedOut = np.zeros([numSteps, numOut])

        if to_simulate:
            for i in range(len(t)):
                PredictedOut[i, :] = model(inputNet[i, :])
            print('Num outputs: ', str(net.get_num_outputs))
            predXYZ = PredictedOut[:, output_inds]
            predX = predXYZ[:, 0]
            predY = predXYZ[:, 1]
            predZ = predXYZ[:, 2]

            fig, (axX, axY, axZ) = plt.subplots(3, 1, figsize=(7, 8))
            # ax1 = plt.subplot(311)
            # plt.title('Predicted vs Actual x, y, and z endpoint positions')
            axX.plot(t, ((predX - min(predX)) / (max(predX) - min(predX))))
            axX.plot(t, ((ActualXYZ[0, :, 0] - min(ActualXYZ[0, :, 0])) / (max(ActualXYZ[0, :, 0]) - min(ActualXYZ[0, :, 0]))))
            axY.plot(t, ((predY - min(predY)) / (max(predY) - min(predY))))
            axY.plot(t,
                ((ActualXYZ[0, :, 1] - min(ActualXYZ[0, :, 1])) / (max(ActualXYZ[0, :, 1]) - min(ActualXYZ[0, :, 1]))))
            axZ.plot(t, ((predZ - min(predZ)) / (max(predZ) - min(predZ))))
            axZ.plot(t,
                ((ActualXYZ[0, :, 2] - min(ActualXYZ[0, :, 2])) / (max(ActualXYZ[0, :, 2]) - min(ActualXYZ[0, :, 2]))))

            plt.xlabel('time, ms')
            axX.set_ylabel('X')
            axY.set_ylabel('Y')
            axZ.set_ylabel('Z')
            plt.show()

            plt.figure()
            plt.plot(ActualXYZ[0, :, 0])
            plt.plot(ActualXYZ[0, :, 1])
            plt.plot(ActualXYZ[0, :, 2])
            plt.title('Actual')
            plt.show()
        else:
            print('Simulation can not be performed.')

    return PredictedOut, ActualXYZ

result, actualPos = theNetwork('R1', 3, numNeuronsInit, 3, widthbellInit, networkInputs, jointTheta, wR1, pR1)
# predX = result[:, -3]
# predY = result[:, -2]
'''predZ = result[:, -1]

plt.figure()
plt.plot(t,((actualPos[0, :, 2] - min(actualPos[0, :, 2])) / (max(actualPos[0, :, 2]) - min(actualPos[0, :, 2]))))
# plt.plot(t, ((predZ - min(predZ)) / (max(predZ) - min(predZ))))
plt.plot(t, ((result[:, 110] - min(result[:, 110])) / (max(result[:, 110]) - min(result[:, 110]))))
plt.show()
startNeuron = 4
endNeuron = 31
startWidth = 2
endWidth = 15
numNeuronsArray = np.arange(startNeuron, endNeuron, 2)
widthBellArray = np.arange(startWidth, endWidth, 2)

rmseX = np.zeros([len(numNeuronsArray), len(widthBellArray)])
rmseY = np.zeros_like(rmseX)
rmseZ = np.zeros_like(rmseX)

'''
'''
for i in range(len(numNeuronsArray)):
    for j in range(len(widthBellArray)):
        jointTheta = np.linspace(thetaMin, thetaMax, numNeuronsArray[i])
        jointThetaDiff = (thetaMax - thetaMin) / (numNeuronsArray[i] - 1)
        predicted, actual = theNetwork('R1', 3, numNeuronsArray[i], 3, widthBellArray[j], networkInputs, jointTheta,
                                       wConfig=wR1, pConfig=pR1)
        predX = normVec(predicted[:, -3])
        predY = normVec(predicted[:, -3])
        predZ = normVec(predicted[:, -3])

        actualX = normVec(actual[0, :, 0])
        actualY = normVec(actual[0, :, 1])
        actualZ = normVec(actual[0, :, 2])

        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        # fig.suptitle('Results')
        # ax1.plot(predX)
        # ax1.plot(actualX)
        # ax2.plot(predY)
        # ax2.plot(actualY)
        # ax3.plot(predZ)
        # ax3.plot(actualZ)
        # plt.show()
        rmseX[i, j] = rmse(predX, actualX)
        rmseY[i, j] = rmse(predY, actualY)
        rmseZ[i, j] = rmse(predZ, actualZ)

cc=np.linspace(startWidth, endWidth, len(widthBellArray))
colors = plt.get_cmap('jet', len(widthBellArray))
fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(7, 8))
# ax1 = plt.subplot(311)
# plt.title('RMSE. Varying width and num sens per joint, X coord')
for widthVar in range(len(widthBellArray)):
    ax4.plot(numNeuronsArray, rmseX[:, widthVar], c=colors(widthVar))
    ax5.plot(numNeuronsArray, rmseY[:, widthVar], c=colors(widthVar))
    ax6.plot(numNeuronsArray, rmseZ[:, widthVar], c=colors(widthVar))
plt.xlabel('Number of sensory neurons per joint')
ax4.set_ylabel('RMSE, X')
# for widthVar in range(len(widthBellArray)):
#     ax5.plot(numNeuronsArray, rmseY[widthVar, :], c=colors(widthVar))
# plt.xlabel('Number of sensory neurons per joint')
ax5.set_ylabel('RMSE, Y')
# plt.xlabel('Number of sensory neurons per joint')
ax6.set_ylabel('RMSE, Z')
sm = matplotlib.cm.ScalarMappable(cmap=colors, norm=colorsPlt.Normalize(vmin=widthBellArray[0], vmax=widthBellArray[-1]))
sm.set_array([])
plt.show()

rmseTotal = rmseX + rmseY + rmseZ

# predX = Normalize(predX, 0, 1)
# predY = Normalize(predY, 0, 1)
# predZ = Normalize(predZ, 0, 1)

'''
'''TESTING 6 JOINTS'''
'''
pR1Test = LegConfigTestOnly('R1').legMatConfig()
wR1Test = LegConfigTestOnly('R1').legAxes()
# inputs
T1 = 450
T2 = 600  # 99.8 # 99.8
T3 = 750  # 102.5 # 102.5
T4 = 350
T5 = 550
T6 = 820

theta1vec = trajectoryInput(t, T1, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta2vec = trajectoryInput(t, T2, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta3vec = trajectoryInput(t, T3, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta4vec = trajectoryInput(t, T4, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta5vec = trajectoryInput(t, T5, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta6vec = trajectoryInput(t, T6, theta_list=[thetaMin, thetaMax], func_type='sinusoid')

networkInputsTest_6 = np.squeeze(np.array([[theta1vec.transpose()], [theta2vec.transpose()], [theta3vec.transpose()],
                                     [theta4vec.transpose()], [theta5vec.transpose()], [theta6vec.transpose()]]))
#
# result_6, actualXYZ_6 = theNetwork('R1Test_6', 6, 7, 3, widthbellInit, networkInputsTest_6,
#                                    jointTheta, wConfig=wR1Test, pConfig=pR1Test)
# predX = result[:, -3]
# predY = result[:, -2]
# predZ = result[:, -1]

print('done')
# to do
'''
'''add fast slow neurons in between layers'''
''' incorporate jacobian part'''
'''closed-loop?'''
'''filtration?'''
'''
make a net that uses xyz (or thetas?) and jacobian to tune connections using 1 layer if it is possible'''
print("--- %s seconds ---" % (time.time() - start))