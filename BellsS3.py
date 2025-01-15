import numpy as np
import scipy as sp
from sns_toolbox.connections import NonSpikingSynapse
import sns_toolbox.networks
import matplotlib.pyplot as plt
from sns_toolbox.neurons import NonSpikingNeuron

'''
Bohdan Zadokha and Nicholas Szczecinski
05 Dec 2024
WVU NeuroMINT
Simplified network that calculates only FK (x, y, z) coords (no jacobian)
using inputs from 3 joints
'''

'''
An updated version with only 2 output neurons for 1st layer
'''


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
tMax = 1e3
t = np.arange(0, tMax, dt)
numSteps = len(t)

numJoints = 3
numSensPerJointInit = 10  # 10
thetaMin = -1.6
thetaMax = 1.6
jointTheta = np.linspace(thetaMin, thetaMax, numSensPerJointInit)
jointThetaDiff = (thetaMax - thetaMin) / (numSensPerJointInit - 1)


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
    expoOne = np.eye(4, dtype=float)
    if hasattr(thetaValue, "__len__"):
        for index in range(len(thetaValue)):
            expo = RodriguesFunc(w[:, index], thetaValue[index], p[:, index])
            expoOne = np.matmul(expoOne, expo)
    else:
        expo = RodriguesFunc(w[:, 0], thetaValue, p[:, 0])
        expoOne = np.matmul(expoOne, expo)
    positionM = np.atleast_2d(p[:, -1]).T
    M = np.concatenate((np.concatenate((np.eye(3, dtype=float), positionM), axis=1),
                        np.array([[0, 0, 0, 1]])), axis=0)
    TranslMat = np.matmul(expoOne, M)
    return TranslMat


def sigm_curve(magnitude, width, theta, shift, jointThetaDiff):
    return magnitude / 2 * (1 + np.tanh(width * (theta - shift + jointThetaDiff / 2)))


def bell_curve(magnitude, width, theta, shift):
    return magnitude * np.exp(-width * pow((theta - shift), 2))


def rmse(pred, actual):
    result = np.sqrt(((actual - pred) ** 2).mean())
    return result


def synConnectMap(netModel, nsiNeuron, numNeurons,
                  nsiX, nsiY, nsiZ, nsi1,
                  mapX, mapY, mapZ,
                  gI, synI, dEex, dEin,
                  bellShift, jointTheta, bellWidth, deltaX, deltaY):
    '''
    :param netModel: model object
    :param nsiNeuron: non-spiking neuron object from the SNS-toolbox
    :param numNeurons: number of sensory neurons
    :param nsiX: matrix that represents neuronal activity for Px (empty) #can move it to this func
    :param nsiY: matrix that represents neuronal activity for Py (empty) #can move it to this func
    :param nsiZ: matrix that represents predicted (final) neuronal activity for Pz (empty) #can move it to this func
    :param nsi1:
    :param mapX: calculated surfaces from FK func(theta2, theta3) based on different theta1 values (Px)
    :param mapY: calculated surfaces from FK func(theta2, theta3) based on different theta1 values (Py)
    :param mapZ: calculated surfaces from FK func(theta2, theta3) based on different theta1 values (Pz)
    :param gI: synaptic conductance = Gmax|k=1
    :param synI: synaptic connection
    :param dEex: excitatory synapse potential
    :param dEin: inhibitory synapse potential
    :param bellShift: shift of the input neurons
    :param jointTheta: range of the shift for sensory(input) neurons
    :param bellWidth: width of the bell curve function
    :return: synaptic conductance surfaces and to_simulate values
    '''
    kSyn = np.zeros([2, pow(numNeurons, numJoints - 1)])

    # 1 layer (theta2 & theta3)

    k = 0
    for i in range(numNeurons):
        for j in range(numNeurons):
            '''
                generate response of each NSI neuron given a particular 
            '''
            th2 = bell_curve(mag, theta=jointTheta[i], shift=bellShift, width=bellWidth)
            th3 = bell_curve(mag, theta=jointTheta[j], shift=bellShift, width=bellWidth)

            '''
                This calculate requires refinement. Use the steady-state equations properly.
            '''
            # neural activity positions
            nsiX[k, :] = ComboVec(theta2Bell=th2, theta3Bell=th3, mag=mag, comboAct=nsiX, gMax=gI, delE=dEex)
            nsiY[k, :] = ComboVec(theta2Bell=th2, theta3Bell=th3, mag=mag, comboAct=nsiY, gMax=gI, delE=dEex)
            nsiZ[k, :] = ComboVec(theta2Bell=th2, theta3Bell=th3, mag=mag, comboAct=nsiZ, gMax=gI, delE=dEex)
            nameStr = 'NSI' + str(i) + str(j)  # between theta2 and theta3 (1st layer)

            # from input sensory neurons to NSIs 1st layer
            netModel.add_neuron(neuron_type=nsiNeuron, name=nameStr)

            netModel.add_connection(synI, source='Bell2' + str(i), destination=nameStr)
            netModel.add_connection(synI, source='Bell3' + str(j), destination=nameStr)

            # Increment k, which is a linear counting variable through all the for loops.
            k += 1

    # 2nd layer outputs from 1st P1, P2 & theta1
    k1 = 0
    th4 = bell_curve(1, theta=1, shift=1, width=bellWidth)
    for j in range(numNeurons):
        th1 = bell_curve(mag, theta=jointTheta[j], shift=bellShift, width=bellWidth)
        nsi1[k1, :] = ComboVec(theta2Bell=th1, theta3Bell=th4, mag=mag, comboAct=nsi1, gMax=gI, delE=dEex)
        k1+=1

    # theta 1 bell neurons
    for i in range(numJoints - 1):
        for j in range(numNeurons):
            nameStr2 = 'NSIth1' + str(i) + str(j)
            netModel.add_neuron(neuron_type=nsiNeuron, name=nameStr2)
            netModel.add_connection(synI, source='Bell1' + str(j), destination=nameStr2)
            netModel.add_output('NSIth1' + str(i) + str(j))

    for n in range(numNeurons - 3):
        netModel.add_connection(synI, source='OutputP1', destination='NSIth11' + str(n))
        netModel.add_connection(synI, source='OutputP2', destination='NSIth10' + str(n + 3))

    for n in range(3):
        netModel.add_connection(synI, source='OutputP1', destination='NSIth10' + str(n))

    for n in range(numNeurons - 1, numNeurons - 4, -1):
        netModel.add_connection(synI, source='OutputP2', destination='NSIth11' + str(n))

    # for index in range(numNeurons):
    #     nameStrX = 'X' + str(index)
    #     netModel.add_output(nameStrX)
    #
    # for index in range(numNeurons):
    #     nameStrY = 'Y' + str(index)
    #     netModel.add_output(nameStrY)
    netModel.add_output('OutputEndX')
    netModel.add_output('OutputEndY')
    netModel.add_output('OutputEndZ')

    for i in range(numJoints - 1):
        kSyn[i, :] = sp.linalg.solve(nsiX, mapX[i, :])
    kSynX = kSyn[0, :]
    kSynY = kSyn[1, :]
    kSynZ = sp.linalg.solve(nsiZ, mapZ)

    if np.any(kSynX > dEex / mag) or np.any(kSynY > dEex / mag) or np.any(kSynZ > dEex / mag):
        to_simulate = False
        print('SKIPPING THIS ONE')
    else:
        to_simulate = True

    gX = np.zeros([numNeurons, numNeurons])
    gY = np.zeros([numNeurons, numNeurons])
    gZ = np.zeros([numNeurons, numNeurons])

    dEX = np.zeros([numNeurons, numNeurons])
    dEY = np.zeros([numNeurons, numNeurons])
    dEZ = np.zeros([numNeurons, numNeurons])
    n = 0
    for j in range(numNeurons):
        for k in range(numNeurons):
            nameStr = 'NSI' + str(j) + str(k)
            '''
                SNS-Toolbox does not enable synapses with max_cond = 0, so if kSyn = 0, we must pass it machine epsilon instead.
            '''
            if kSynX[n] > 0:
                dEX[j, k] = dEex
            elif kSynX[n] < 0:
                dEX[j, k] = dEin
            else:
                continue

            if kSynY[n] > 0:
                dEY[j, k] = dEex
            elif kSynY[n] < 0:
                dEY[j, k] = dEin
            else:
                continue

            if kSynZ[n] > 0:
                dEZ[j, k] = dEex
            elif kSynZ[n] < 0:
                dEZ[j, k] = dEin
            else:
                continue

            '''
            Synapses from the combo neurons to the output neuron(s) each have unique conductance value that corresponds to
            the desired output in that scenario. Note that the e_lo for the synapses = mag and e_hi = 2*mag, because
            multiple combo neurons will be active at any time, but we only care about the most active, so setting e_lo this 
            way serves as a threshold mechanism.
            '''
            gX[j, k] = Gmax(kSynX[n], mag, dEX[j, k])
            gY[j, k] = Gmax(kSynY[n], mag, dEY[j, k])
            gZ[j, k] = Gmax(kSynZ[n], mag, dEZ[j, k])

            if gX[j, k] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gX[j, k]), reversal_potential=float(dEX[j, k]),
                                              e_lo=mag, e_hi=2 * mag)
                netModel.add_connection(synOutput, source=nameStr, destination='OutputP1')
            else:
                1 + 1
            if gY[j, k] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gY[j, k]), reversal_potential=float(dEY[j, k]),
                                              e_lo=mag, e_hi=2 * mag)
                netModel.add_connection(synOutput, source=nameStr, destination='OutputP2')
            else:
                1 + 1
            if gZ[j, k] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gZ[j, k]), reversal_potential=float(dEZ[j, k]),
                                              e_lo=mag, e_hi=2 * mag)
                netModel.add_connection(synOutput, source=nameStr, destination='OutputEndZ')
            else:
                1 + 1
            n += 1

    kSynX1 = sp.linalg.solve(nsi1, deltaX)
    kSynY1 = np.flip(kSynX1, axis=0)
    print(kSynX1)
    print(kSynY1)

    gX1 = np.zeros([numNeurons])
    gY1 = np.zeros([numNeurons])

    dEX1 = np.zeros([numNeurons])
    dEY1 = np.zeros([numNeurons])
    '''
    2nd layer output connections
    '''
    for j in range(numNeurons):
        nameStrX = 'NSIth10' + str(j)
        nameStrY = 'NSIth11' + str(j)
        '''
            SNS-Toolbox does not enable synapses with max_cond = 0, so if kSyn = 0, we must pass it machine epsilon instead.
        '''
        if kSynX1[j] > 0:
            dEX1[j] = dEex
        elif kSynX1[j] < 0:
            dEX1[j] = dEin
        else:
            1 + 1
        if kSynY1[j] > 0:
            dEY1[j] = dEex
        elif kSynY1[j] < 0:
            dEY1[j] = dEin
        else:
            1 + 1

        gX1[j] = Gmax(kSynX1[j], mag, dEX1[j])
        gY1[j] = Gmax(kSynY1[j], mag, dEX1[j])

        if gX1[j] > 0:
            synOutput = NonSpikingSynapse(max_conductance=float(gX1[j]), reversal_potential=float(dEX1[j]),
                                          e_lo=mag, e_hi=2 * mag)
            netModel.add_connection(synOutput, source=nameStrX, destination='OutputEndX')
        else:
            1 + 1

        if gY1[j] > 0:
            synOutput = NonSpikingSynapse(max_conductance=float(gY1[j]), reversal_potential=float(dEY1[j]),
                                          e_lo=mag, e_hi=2 * mag)
            netModel.add_connection(synOutput, source=nameStrY, destination='OutputEndY')
        else:
            1 + 1

    plt.figure()
    plt.plot(kSynX1)
    plt.plot(kSynY1)
    plt.show()

    return to_simulate, gX, gY, gZ, gX1, gY1


def FKJSNS3D(delay, numJoints, numSensPerJoint, numSteps,
             widthBell, servo1vec, servo2vec, servo3vec, jointTheta):
    # Create a network object. Here, it is called 'netX', because it computes the x-coordinate of the foot.
    # However, it should be noted that the same network could be used to compute just about anything
    # by re-using the combo neurons (described below) and creating an additional output neuron with unique synapses
    # from the combo neurons.
    net = sns_toolbox.networks.Network(name='TheLegR1')
    """
       Define key neuron and synapse parameters. 
       'mag' is like R in the functional subnetwork publications: it is the maximum expected neural activation. 
       delE is the synaptic reversal potential relative to the rest potential of the neurons.
    """

    '''
        Create a basic nonspiking neuron type.
    '''
    mag = 1
    delEex = 20
    delEin = -20
    kIdent = 1
    gIdent = Gmax(kIdent, mag, delEex)  # From Szczecinski, Hunt, and Quinn 2017

    identitySyn = NonSpikingSynapse(max_conductance=gIdent, reversal_potential=delEex, e_lo=0, e_hi=mag)

    bellNeur = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1)

    NSIneuron = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1, bias=0)
    # thetaDiff = (thetaMax - thetaMin) / (numSensPerJoint - 1)
    '''
    Create the "bell" neurons, which are the sensory neurons. Each sensory neuron has a bell-curve receptive field
    with a unique "preferred angle", that is, the angle at which the peak response is evoked.
    '''
    # joint 1
    for index in range(numSensPerJoint):
        nameStr1 = 'Bell1' + str(index)
        net.add_neuron(neuron_type=bellNeur, name=nameStr1)
        net.add_input(nameStr1)
        net.add_output(nameStr1)
    # joint 2
    for index in range(numSensPerJoint):
        nameStr2 = 'Bell2' + str(index)
        net.add_neuron(neuron_type=bellNeur, name=nameStr2)
        net.add_input(nameStr2)
        net.add_output(nameStr2)
    # joint 3
    for index in range(numSensPerJoint):
        nameStr3 = 'Bell3' + str(index)
        net.add_neuron(neuron_type=bellNeur, name=nameStr3)
        net.add_input(nameStr3)
        net.add_output(nameStr3)
    '''
        Each sensory neuron synapses onto a number of "combo" neurons. Each combo neuron receives synaptic input from one
        sensory neuron from each joint. All these synapses may be identical, because they are simply generating all the possible
        combinations of the joint angles/independent variables. All the combo neurons may be identical, because they are simply
        integrating the joint angles.
    '''
    # initialize neurons (they are outputs for 1st layer [X and Y normalized] and inputs for 2nd layer)
    out1Neurons = 2

    for index in range(out1Neurons):
        nameX = 'OutputP' + str(index + 1)
        net.add_neuron(neuron_type=bellNeur, name=nameX)
        net.add_output(nameX)

    # outputs for 2nd layer

    # for index in range(numSensPerJoint):
    #     nameStrX = 'X' + str(index)
    #     net.add_neuron(neuron_type=bellNeur, name=nameStrX)
    #
    # for index in range(numSensPerJoint):
    #     nameStrY = 'Y' + str(index)
    #     net.add_neuron(neuron_type=bellNeur, name=nameStrY)

    net.add_neuron(neuron_type=bellNeur, name='OutputEndX')
    net.add_neuron(neuron_type=bellNeur, name='OutputEndY')
    net.add_neuron(neuron_type=bellNeur, name='OutputEndZ')

    '''
        widthBell is the width of the sensory-encoding bell curves. 
        Note that these have been normalized to the range of theta and the number of sensory neurons. 
        Also note that smaller value of widthBell makes the curves broader.
    '''
    '''
        Initialize empty arrays to store input current for each set of sensory neurons (theta2 = input2, theta3 = input3), the
        input current for the whole network (inputX), and the validation data X(t).
    '''
    # replace with for loop
    input1 = np.zeros([numSteps, numSensPerJoint])  # bell responses joint 1
    input2 = np.zeros([numSteps, numSensPerJoint])  # bell responses joint 2
    input3 = np.zeros([numSteps, numSensPerJoint])  # bell responses joint 3

    inputNet = np.zeros([numSteps, numSensPerJoint * numJoints])
    # Actual X, Y, Z values as a function of time
    ActualT = np.zeros([numSteps, 3])
    '''
        For the network to perform the desired calculation, the synapses from each combo neuron to the output neuron should have
        effective gains (i.e., Vout/Vcombo) that are proportional to the value that the output neuron encodes. Here, we take all
        the "training data", that is, the actual X coordinate of the leg, normalize it to values between 0 and 1, and use those
        to set the unique properties of the synapse from each combo neuron to the output neuron.
    '''
    # position map
    # mapping surfaces for syn connections
    ActualXmap = np.zeros([len(jointTheta), len(jointTheta), len(jointTheta)])
    ActualYmap = np.zeros([len(jointTheta), len(jointTheta), len(jointTheta)])
    ActualZmap = np.zeros([len(jointTheta), len(jointTheta)])

    '''
    Create the "training data" for the network, that is, the known X, Y, Z coordinates given the lists of theta values.
    For 3+ joints we need to generate surfaces for each joint 1 angles: for X and Y coordinates
    Z axis does not depend on joint1

    '''
    for i in range(numSensPerJoint):
        for j in range(numSensPerJoint):
            for k in range(numSensPerJoint):
                theta = np.array([jointTheta[i], jointTheta[j], jointTheta[k]])
                endPointTrnslMat = FrwrdKnmtcsFunc(wR1, theta, pR1)
                ActualXmap[i, j, k] = endPointTrnslMat[0, 3]  # x coord
                ActualYmap[i, j, k] = endPointTrnslMat[1, 3]  # y coord
                ActualZmap[j, k] = endPointTrnslMat[2, 3]  # z coord

    mapNormX = np.zeros([len(jointTheta), pow(numSensPerJoint, numJoints - 1)])
    mapNormY = np.zeros([len(jointTheta), pow(numSensPerJoint, numJoints - 1)])

    maxCoef = np.zeros([2, numSensPerJointInit])
    minCoef = np.zeros([2, numSensPerJointInit])

    for i in range(numSensPerJoint):
        mapNormX[i, :], maxCoef[0, i], minCoef[0, i] = Normalize(ActualXmap[i, :, :], 0, 1)
        mapNormY[i, :], maxCoef[1, i], minCoef[1, i] = Normalize(ActualYmap[i, :, :], 0, 1)

    mapNormZ, _, _ = Normalize(ActualZmap, 0, 1)
    '''
        to tune a 1st layer of the network we need only 2 surfaces
    '''
    mapSetX = np.array([mapNormX[0, :], mapNormX[numSensPerJoint - 1, :]])
    mapSetY = np.array([mapNormY[0, :], mapNormX[numSensPerJoint - 1, :]])

    XcA = maxCoef[0, :] - minCoef[0, :]
    YcA = maxCoef[1, :] - minCoef[1, :]

    XcANorm = ((XcA - min(XcA)) / (max(XcA) - min(XcA)))
    YcANorm = np.flip(XcANorm)

    print('XcANorm', XcANorm)
    print('YcANorm', YcANorm)

    '''
        "nsi..." matrices store neural activity values for each NSI 
    '''

    nsi23X = np.empty(
        shape=[pow(numSensPerJoint, numJoints - 1), pow(numSensPerJoint, numJoints - 1)])  # theta2 and theta3, Xcoord
    nsi23Y = np.empty_like(nsi23X)  # theta2 and theta3, Ycoord
    nsiZ = np.empty_like(nsi23X)  # theta2 and theta3,
    # Zcoord does not depend on theta1
    '''ask about this shape'''
    nsiP1 = np.empty(shape=[numSensPerJoint, numSensPerJoint])  # theta1 and P1&P2 from the n-1 layer 23

    jointThetaMat = np.empty([numSensPerJoint, 1])
    jointThetaMat[:, 0] = jointTheta
    '''
    Now we create the each combo neuron, its input connections, and its output connections. Here, we have 2 nested for
    loops, one for each joint/independent variable. As stated above, a higher-dimensional network would require more for 
    loops or an approach in which we NDgrid the joint angles/sensory neurons and then use a single for loop.
    '''
    '''
        Setting up synaptic conductance values for NSIs between theta 2 and theta 3

    '''
    to_simulate, gX, gY, gZ, gX1, gY1 = synConnectMap(net, NSIneuron, numSensPerJoint,
                                                      nsiX=nsi23X, nsiY=nsi23Y, nsiZ=nsiZ, nsi1=nsiP1,
                                                      mapX=mapSetX, mapY=mapSetY, mapZ=mapNormZ,
                                                      gI=gIdent, synI=identitySyn, dEex=delEex, dEin=delEin,
                                                      bellShift=jointThetaMat, jointTheta=jointTheta,
                                                      bellWidth=widthBell,
                                                      deltaX=XcANorm, deltaY=YcANorm)

    # Concatenate the inputs into one network input array. Calculate the "Actual X, Y, Z values as a function of time",
    # ActualT.
    # Calculate and store the input current for the theta2 and theta3 sensory neurons using the bellCurve function.
    for i in range(numSensPerJoint):
        input1[:, i] = bell_curve(magnitude=mag, theta=servo1vec, shift=jointTheta[i], width=widthBell)
        input2[:, i] = bell_curve(magnitude=mag, theta=servo2vec, shift=jointTheta[i], width=widthBell)
        input3[:, i] = bell_curve(magnitude=mag, theta=servo3vec, shift=jointTheta[i], width=widthBell)

    # Concatenate the inputs into one network input array. Calculate the "Actual X value as a function of time", ActualXt.

    # Compile the network model
    model = net.compile(backend='numpy', dt=dt)
    numOut = net.get_num_outputs()
    PredictedOut = np.zeros([numSteps, numOut])
    if to_simulate:
        for i in range(len(t)):
            inputNet[i, :] = np.concatenate((input1[i, :], np.concatenate((input2[i, :], input3[i, :]), axis=None)),
                                            axis=None)
            endPointTrnslMat = FrwrdKnmtcsFunc(wR1, np.array([theta1vec[i], theta2vec[i], theta3vec[i]]), pR1)
            ActualT[i, 0] = endPointTrnslMat[0, 3]  # x coord
            ActualT[i, 1] = endPointTrnslMat[1, 3]  # y coord
            ActualT[i, 2] = endPointTrnslMat[2, 3]  # z coord

        # Simulate the network response
        for i in range(len(t)):
            PredictedOut[i, :] = model(inputNet[i, :])
    return PredictedOut, ActualT, gX, gY, gZ, model, to_simulate, ActualXmap, ActualYmap, ActualZmap, maxCoef, minCoef, gX1, gY1


def Gmax(k, R, delE):
    maxConduct = k * R / (delE - k * R)
    return maxConduct

def Normalize(matrix, minNorm, maxNorm):
    vector = np.matrix(matrix).getA1()
    vectorNorm = ((vector - min(vector)) / (max(vector) - min(vector))) * (maxNorm - minNorm) + minNorm
    return vectorNorm, max(vector), min(vector)

def ComboVec(theta2Bell, theta3Bell, mag, comboAct, gMax, delE):
    # position
    comboSS = np.reshape(theta2Bell + np.transpose(theta3Bell) - mag, [np.size(comboAct, 1), ])
    comboSS[comboSS < 0] = 0
    comboVec = gMax * comboSS / mag * delE / (1 + 2 * gMax)
    comboVec[comboVec < 0] = 0
    return comboVec

def SkewMatrix(x):
    skew = [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]
    return skew

def JacobianFunc(w, p, R1, R2, p1Final, p2Final):
    S1cross = np.cross(-w[:, 0], p[:, 0])
    S2cross = np.cross(-w[:, 1], p[:, 1])
    S3cross = np.cross(-w[:, 2], p[:, 2])

    p1Skew = SkewMatrix(p1Final)
    p2Skew = SkewMatrix(p2Final)

    m1 = np.matmul(p1Skew, R1)
    m2 = np.matmul(p2Skew, R2)

    adjointS1_1 = np.concatenate((R1, m1), axis=1)
    adjointS1_2 = np.concatenate((np.zeros([3, 3]), R1), axis=1)
    adjointS1 = np.concatenate((adjointS1_1, adjointS1_2), axis=0)

    adjointS2_1 = np.concatenate((R2, m2), axis=1)
    adjointS2_2 = np.concatenate((np.zeros([3, 3]), R2), axis=1)
    adjointS2 = np.concatenate((adjointS2_1, adjointS2_2), axis=0)

    J1 = np.concatenate((S1cross, w[:, 0]), axis=0)
    J2 = np.concatenate((S2cross, w[:, 1]), axis=0)
    J3 = np.concatenate((S3cross, w[:, 2]), axis=0)

    jac1 = np.matmul(adjointS1, J2.reshape((6, 1)))
    jac2 = np.matmul(adjointS2, J3.reshape((6, 1)))

    jacobianMatrix = np.concatenate((np.concatenate((J1.reshape((6, 1)),
                                                     jac1), axis=1),
                                     jac2), axis=1)
    return jacobianMatrix

def trajectoryInput(t, T, thetamin, thetamax):
    inputT = (thetamin + thetamax) / 2 + (thetamax - thetamin) / 2 * np.sin(2 * np.pi * t / T)
    return inputT


'''
    Generating two patterns of joint angles over time. Here, we have two sinusoidal functions of 
    time with different periods. In this way, we sample the entire workspace.
'''

T1 = 150
T2 = 200  # 99.8 # 99.8
T3 = 250  # 102.5 # 102.5

theta1vec = trajectoryInput(t, T1, thetamin=thetaMin, thetamax=thetaMax)
theta2vec = trajectoryInput(t, T2, thetamin=thetaMin, thetamax=thetaMax)
# theta2vec = np.zeros_like(theta1vec)
theta3vec = trajectoryInput(t, T3, thetamin=thetaMin, thetamax=thetaMax)
# theta3vec = np.zeros_like(theta1vec)

mag = 1
delayInit = 2
widthbellInit = 6  # 6
delaySlow = 1000

'''
    write a func for plots
'''

print('smth')
PredictedOutOne, ActualTOne, kSynX, kSynY, kSynZ, modelFKJ, _, xMap, yMap, zMap, maxC, minC, gX1, gY1 = FKJSNS3D(
    delay=delayInit, numJoints=numJoints,
    numSensPerJoint=numSensPerJointInit,
    numSteps=numSteps,
    widthBell=widthbellInit,
    servo3vec=theta3vec,
    servo2vec=theta2vec,
    servo1vec=theta1vec,
    jointTheta=jointTheta)
# XcA = maxC[0, :] - minC[0, :]
# YcA = maxC[1, :] - minC[1, :]
# XcADenorm = min(XcA)+(max(XcA)-min(XcA))*XcA
# YcADenorm = min(YcA)+(max(YcA)-min(YcA))*YcA
predOutXOne = min(ActualTOne[:, 0]) + (max(ActualTOne[:, 0]) - min(ActualTOne[:, 0])) * PredictedOutOne[:, -3] / mag
predOutYOne = min(ActualTOne[:, 1]) + (max(ActualTOne[:, 1]) - min(ActualTOne[:, 1])) * PredictedOutOne[:, -2] / mag
predOutZOne = min(ActualTOne[:, 2]) + (max(ActualTOne[:, 2]) - min(ActualTOne[:, 2])) * PredictedOutOne[:, -1] / mag

plt.figure()
plt.plot(predOutZOne)
plt.plot(ActualTOne[:, 2])
plt.show()


plt.figure()
plt.plot(PredictedOutOne[:, -3], color='red')
# plt.plot(Px)
plt.plot((ActualTOne[:, 0] - min(ActualTOne[:, 0])) / (max(ActualTOne[:, 0]) - min(ActualTOne[:, 0])))
# plt.plot(ActualTOne[:, 0])
plt.show()

plt.figure()
plt.plot(PredictedOutOne[:, -2], color='k')
plt.plot((ActualTOne[:, 1] - min(ActualTOne[:, 1])) / (max(ActualTOne[:, 1]) - min(ActualTOne[:, 1])))
# plt.plot(ActualTOne[:, 1])
plt.show()
#
plt.figure()
for i in range(numSensPerJointInit):
    P1 = PredictedOutOne[:, 3 * numSensPerJointInit + 2 + i]
    plt.plot(P1)
    # plt.xlim([0, 1000])
plt.show()
plt.figure()
for i in range(numSensPerJointInit):
    P2 = PredictedOutOne[:, 4 * numSensPerJointInit + 2 + i]
    plt.plot(P2)
    plt.title('Y')
    # plt.xlim([0, 1000])
plt.show()



theta22, theta33 = np.meshgrid(jointTheta, jointTheta)
# theta22rot, theta33rot = np.meshgrid(jointThetaRot1, jointThetaRot2)
ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(theta22, theta33, np.transpose(kSynY), edgecolor='tomato', lw=0.5, rstride=11, cstride=11,
                alpha=0.5, label='yActual')

# ax.plot_surface(theta22, theta22, kSynX.transpose(), edgecolor='tomato', lw=0.5, rstride=11, cstride=11,
#                 alpha=0.5, label='yRot')
# ax.view_init(90, -90, 0)
# ax.contour(theta22, theta33, xMap, zdir='z', offset=-120, cmap='autumn')
# ax.contour(theta22, theta33, xMap, zdir='x', offset=-1.7,cmap='autumn')

# ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), zlim=(-120, 400),
#        xlabel='Theta2, rad', ylabel='Theta3, rad', zlabel='X pos, mm')
# plt.savefig('mapXfk.svg', dpi=500)
plt.legend()
plt.show()
Pxx = np.zeros([numSensPerJointInit, 3])

for i in range(numSensPerJointInit):
    theta = np.array([jointTheta[i], 0, 0])
    endPointTrnslMat = FrwrdKnmtcsFunc(wR1, theta, pR1)
    Pxx[i, :] = endPointTrnslMat[0:3, 3]
plt.figure()
plt.plot(jointTheta, Pxx[:, 0])
# plt.show()


