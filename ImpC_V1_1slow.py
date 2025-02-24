import numpy as np
import scipy as sp
from sns_toolbox.connections import NonSpikingSynapse
import sns_toolbox.networks
import matplotlib.pyplot as plt
from sns_toolbox.neurons import NonSpikingNeuron
import matplotlib
'''
Bohdan Zadokha and Nicholas Szczecinski
16 Sep 2024
WVU NeuroMINT
Impedance control using virtual environmental stiffness and 
an SNS-model for forward kinematics and jacobian calculations'''
# adding slow neurons

# 3d space

L1 = 52.52
L2 = 60
L3 = 95

x0 = 125
y02 = 100
y01 = 65
y03 = 65

w1 = np.array([[0], [0], [1]])
# R1 leg
w2R1 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)], [0]])
wR1 = np.concatenate((np.concatenate((w1, w2R1), axis=1), w2R1), axis=1)

# zero position R1 leg

p1R1 = np.array([[x0], [-y01], [0]])
p2R1 = np.array([[x0+L1*np.cos(45*np.pi/180)],
                 [-y02-L1*np.cos(45*np.pi/180)],
                 [0]])  # mm
p3R1 = np.array([[x0+(L1+L2)*np.cos(45*np.pi/180)],
                 [-y02-(L1+L2)*np.cos(45*np.pi/180)],
                 [-15]])  # mm
p4R1 = np.array([[x0+(L1+L2+L3)*np.cos(45*np.pi/180)],
                 [-y02-(L1+L2+L3)*np.cos(45*np.pi/180)],
                 [-15-90]])  # mm
pR1 = np.concatenate((np.concatenate((np.concatenate((p1R1, p2R1), axis=1), p3R1), axis=1), p4R1), axis=1)

# time vec
dt = 1
tMax = 2e3
t = np.arange(0, tMax, dt)
numSteps = len(t)

numJoints = 2
numSensPerJointInit = 6 # 6
thetaMin = -1.6
thetaMax = 1.6
jointTheta = np.linspace(thetaMin, thetaMax, numSensPerJointInit)
jointThetaDiff = (thetaMax - thetaMin)/(numSensPerJointInit-1)
# print(jointThetaDiff)

def RodriguesFunc(w, theta, p):
    Scross = np.cross(-w, p)
    Smatrix = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]  # Skew-symmetric
    Rmatrix = np.eye(3, dtype=float)+np.dot(Smatrix, np.sin(theta))+np.dot(np.matmul(Smatrix, Smatrix), (1-np.cos(theta)))
    c = np.matmul(np.matmul(np.eye(3, dtype=float)-Rmatrix, Smatrix), Scross)
    d = np.matmul(w, w.reshape((3, 1)))
    e = np.dot(theta, np.dot(d.item(), Scross))
    TranslVec = c+e
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
    # print(positionM)
    M = np.concatenate((np.concatenate((np.eye(3, dtype=float), positionM), axis=1),
                        np.array([[0, 0, 0, 1]])), axis=0)
    TranslMat = np.matmul(expoOne, M)
    return TranslMat

def sigm_curve(magnitude, width, theta, shift, jointThetaDiff):
    return magnitude / 2 * (1 + np.tanh(width * (theta - shift + jointThetaDiff/2)))
    #  return magnitude*np.exp(-width*pow((theta-shift), 2))

def bell_curve(magnitude, width, theta, shift):
    return magnitude*np.exp(-width*pow((theta-shift), 2))

def rmse(pred, actual):
    # return np.sqrt(np.sum(pow(actual-pred, 2))/numSteps)
    return np.sqrt(((actual - pred) ** 2).mean())

def FKJSNS3D(delay, delaySlow, numJoints, numSensPerJoint, numSteps,
             widthBell, ActualMapX, ActualMapY, ActualMapZ,
             ActualJ1map, ActualJ2map, ActualJ3map,
             servo2vec, servo3vec, jointTheta, kInhib):
    '''Create a network object. Here, it is called 'netX', because it computes the x-coordinate of the foot.
    However, it should be noted that the same network could be used to compute just about anything
    by re-using the combo neurons (described below) and creating an additional output neuron with unique synapses
    from the combo neurons. '''
    net = sns_toolbox.networks.Network(name='SigmoidPrediction')
    '''
       Define key neuron and synapse parameters. 
       'mag' is like R in the functional subnetwork publications: it is the maximum expected neural activation. 
       delE is the synaptic reversal potential relative to the rest potential of the neurons.
    '''
    '''
    Create a basic nonspiking neuron type.
    '''
    mag = 1
    delEex = 20
    delEin = -20
    kIdent = 1
    kInh = kInhib
    gIdent = Gmax(kIdent, mag, delEex)  # From Szczecinski, Hunt, and Quinn 2017
    gInh = Gmax(kInh, mag, delEex)

    identitySyn = NonSpikingSynapse(max_conductance=gIdent, reversal_potential=delEex, e_lo=0, e_hi=mag)
    inhSyn = NonSpikingSynapse(max_conductance=gInh, reversal_potential=delEin, e_lo=0, e_hi=mag)

    sigmNeur = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1)
    slowNeur = NonSpikingNeuron(membrane_capacitance=delaySlow, membrane_conductance=1)
    comboNeur = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1, bias=0)

    numTest = 100
    thetaTest = np.zeros([numTest, 1])
    thetaTest[:, 0] = np.linspace(thetaMin, thetaMax, num=numTest)
    '''
    Create the "sigmoid" neurons, which are the sensory neurons. Each sensory neuron has a bell-curve receptive field
    with a unique "preferred angle", that is, the angle at which the peak response is evoked.
    '''
    for index in range(numSensPerJoint):
        nameStr2 = 'Sigm2' + str(index)
        net.add_neuron(neuron_type=sigmNeur, name=nameStr2)
        net.add_input(nameStr2)
        net.add_output(nameStr2)

    for index in range(numSensPerJoint):
        nameStr3 = 'Sigm3' + str(index)
        net.add_neuron(neuron_type=sigmNeur, name=nameStr3)
        net.add_input(nameStr3)
        net.add_output(nameStr3)

    for index in range(numSensPerJoint):
        nameStr2slow = 'Sigm2slow' + str(index)
        net.add_neuron(neuron_type=slowNeur, name=nameStr2slow)

    for index in range(numSensPerJoint):
        nameStr3slow = 'Sigm3slow' + str(index)
        net.add_neuron(neuron_type=slowNeur, name=nameStr3slow)

    for index in range(numSensPerJoint):
        net.add_connection(identitySyn, source='Sigm2' + str(index), destination='Sigm2slow' + str(index))
        net.add_connection(identitySyn, source='Sigm3' + str(index), destination='Sigm3slow' + str(index))

        net.add_connection(inhSyn, source='Sigm2slow' + str(index), destination='Sigm2' + str(index))
        net.add_connection(inhSyn, source='Sigm3slow' + str(index), destination='Sigm3' + str(index))

    '''
        Each sensory neuron synapses onto a number of "combo" neurons. Each combo neuron receives synaptic input from one
        sensory neuron from each joint. All these synapses may be identical, because they are simply generating all the possible
        combinations of the joint angles/independent variables. All the combo neurons may be identical, because they are simply
        integrating the joint angles.
    '''
    net.add_neuron(neuron_type=sigmNeur, name='OutputEndX')
    net.add_neuron(neuron_type=sigmNeur, name='OutputEndY')
    net.add_neuron(neuron_type=sigmNeur, name='OutputEndZ')
    net.add_neuron(neuron_type=sigmNeur, name='OutputJ1')
    net.add_neuron(neuron_type=sigmNeur, name='OutputJ2')
    net.add_neuron(neuron_type=sigmNeur, name='OutputJ3')

    '''
    widthBell is the width of the sensory-encoding bell curves. 
    Note that these have been normalized to the range of theta and the number of sensory neurons. 
    Also note that smaller value of widthBell makes the curves broader.
    '''
    '''
    Initialize empty arrays to store input current for each set of sensory neurons (theta2 = input2, theta3 = input3), the
    input current for the whole network (inputX), and the validation data X(t).
    '''
    input2 = np.zeros([numSteps, numSensPerJoint])  # sigmoid responses
    input3 = np.zeros([numSteps, numSensPerJoint])  # sigmoid responses
    inputNet = np.zeros([numSteps, numSensPerJoint * numJoints])
    # Actual X, Y, Z values as a function of time
    ActualT = np.zeros([numSteps, 3])
    ActualJ = np.zeros([numSteps, 3])
    '''
    For the network to perform the desired calculation, the synapses from each combo neuron to the output neuron should have
    effective gains (i.e., Vout/Vcombo) that are proportional to the value that the output neuron encodes. Here, we take all
    the "training data", that is, the actual X coordinate of the leg, normalize it to values between 0 and 1, and use those
    to set the unique properties of the synapse from each combo neuron to the output neuron.
    '''
    # position map
    mapNormX = Normalize(ActualMapX)
    mapNormY = Normalize(ActualMapY)
    mapNormZ = Normalize(ActualMapZ)

    comboActX = np.empty(shape=[pow(numSensPerJoint, numJoints), pow(numSensPerJoint, numJoints)])
    comboActY = np.empty(shape=[pow(numSensPerJoint, numJoints), pow(numSensPerJoint, numJoints)])
    comboActZ = np.empty(shape=[pow(numSensPerJoint, numJoints), pow(numSensPerJoint, numJoints)])

    # jacobian map
    mapNormJ1 = Normalize(ActualJ1map)
    mapNormJ2 = Normalize(ActualJ2map)
    mapNormJ3 = Normalize(ActualJ3map)

    comboActJ1 = np.empty(shape=[pow(numSensPerJoint, numJoints), pow(numSensPerJoint, numJoints)])
    comboActJ2 = np.empty(shape=[pow(numSensPerJoint, numJoints), pow(numSensPerJoint, numJoints)])
    comboActJ3 = np.empty(shape=[pow(numSensPerJoint, numJoints), pow(numSensPerJoint, numJoints)])

    jointThetaMat = np.empty([numSensPerJoint, 1])
    jointThetaMat[:, 0] = jointTheta
    '''
    Now we create the each combo neuron, its input connections, and its output connections. Here, we have 2 nested for
    loops, one for each joint/independent variable. As stated above, a higher-dimensional network would require more for 
    loops or an approach in which we NDgrid the joint angles/sensory neurons and then use a single for loop.
    '''
    thetaDiff = (thetaMax - thetaMin)/(numSensPerJoint-1)
    k = 0
    for i in range(numSensPerJoint):
        for j in range(numSensPerJoint):
            '''
            generate response of each combo neuron given a particular 
            '''
            th2 = sigm_curve(mag, widthBell, theta=jointTheta[i], shift=jointThetaMat, jointThetaDiff=thetaDiff)
            th3 = sigm_curve(mag, widthBell, theta=jointTheta[j], shift=jointThetaMat, jointThetaDiff=thetaDiff)
            '''
            This calculate requires refinement. Use the steady-state equations properly.
            '''
            # position
            comboActX[k, :] = ComboVec(theta2=th2, theta3=th3, mag=mag, comboAct=comboActX, gMax=gIdent, delE=delEex)
            comboActY[k, :] = ComboVec(theta2=th2, theta3=th3, mag=mag, comboAct=comboActY, gMax=gIdent, delE=delEex)
            comboActZ[k, :] = ComboVec(theta2=th2, theta3=th3, mag=mag, comboAct=comboActZ, gMax=gIdent, delE=delEex)
            # jacobian
            comboActJ1[k, :] = ComboVec(theta2=th2, theta3=th3, mag=mag, comboAct=comboActJ1, gMax=gIdent, delE=delEex)
            comboActJ2[k, :] = ComboVec(theta2=th2, theta3=th3, mag=mag, comboAct=comboActJ2, gMax=gIdent, delE=delEex)
            comboActJ3[k, :] = ComboVec(theta2=th2, theta3=th3, mag=mag, comboAct=comboActJ3, gMax=gIdent, delE=delEex)

            nameStr = 'combo' + str(i) + str(j)

            net.add_neuron(neuron_type=comboNeur, name=nameStr)
            net.add_connection(identitySyn, source='Sigm2' + str(i), destination=nameStr)
            net.add_connection(identitySyn, source='Sigm3' + str(j), destination=nameStr)
            net.add_output(nameStr)

            # Increment k, which is a linear counting variable through all the for loops.
            k += 1

    net.add_output('OutputEndX')
    net.add_output('OutputEndY')
    net.add_output('OutputEndZ')
    net.add_output('OutputJ1')
    net.add_output('OutputJ2')
    net.add_output('OutputJ3')

    #fk syn cond
    kSynX = sp.linalg.solve(comboActX, mapNormX)
    kSynY = sp.linalg.solve(comboActY, mapNormY)
    kSynZ = sp.linalg.solve(comboActZ, mapNormZ)

    gX = np.zeros([numSensPerJoint, numSensPerJoint])
    gY = np.zeros([numSensPerJoint, numSensPerJoint])
    gZ = np.zeros([numSensPerJoint, numSensPerJoint])

    k = 0
    delEX = np.zeros([numSensPerJoint, numSensPerJoint])
    delEY = np.zeros([numSensPerJoint, numSensPerJoint])
    delEZ = np.zeros([numSensPerJoint, numSensPerJoint])

    # jacobian syn cond
    kSynJ1 = sp.linalg.solve(comboActJ1, mapNormJ1)
    kSynJ2 = sp.linalg.solve(comboActJ2, mapNormJ2)
    kSynJ3 = sp.linalg.solve(comboActJ3, mapNormJ3)

    gJ1 = np.zeros([numSensPerJoint, numSensPerJoint])
    gJ2 = np.zeros([numSensPerJoint, numSensPerJoint])
    gJ3 = np.zeros([numSensPerJoint, numSensPerJoint])

    k = 0
    delEJ1 = np.zeros([numSensPerJoint, numSensPerJoint])
    delEJ2 = np.zeros([numSensPerJoint, numSensPerJoint])
    delEJ3 = np.zeros([numSensPerJoint, numSensPerJoint])

    for i in range(numSensPerJoint):
        for j in range(numSensPerJoint):
            nameStr = 'combo' + str(i) + str(j)
            '''
            SNS-Toolbox does not enable synapses with max_cond = 0, so if kSyn = 0, we must pass it machine epsilon instead.
            '''
            if kSynX[k] > 0:
                delEX[i, j] = delEex
            elif kSynX[k] < 0:
                delEX[i, j] = delEin
            else:
                1 + 1

            if kSynY[k] > 0:
                delEY[i, j] = delEex
            elif kSynY[k] < 0:
                delEY[i, j] = delEin
            else:
                1 + 1

            if kSynZ[k] > 0:
                delEZ[i, j] = delEex
            elif kSynZ[k] < 0:
                delEZ[i, j] = delEin
            else:
                1 + 1

            if kSynJ1[k] > 0:
                delEJ1[i, j] = delEex
            elif kSynJ1[k] < 0:
                delEJ1[i, j] = delEin
            else:
                #  kSyn[k] = 0, so g = 0.
                1 + 1

            if kSynJ2[k] > 0:
                delEJ2[i, j] = delEex
            elif kSynJ2[k] < 0:
                delEJ2[i, j] = delEin
            else:
                1 + 1
            if kSynJ3[k] > 0:
                delEJ3[i, j] = delEex
            elif kSynJ3[k] < 0:
                delEJ3[i, j] = delEin
            else:
                1 + 1
            '''
            Synapses from the combo neurons to the output neuron(s) each have unique conductance value that corresponds to
            the desired output in that scenario. Note that the e_lo for the synapses = mag and e_hi = 2*mag, because
            multiple combo neurons will be active at any time, but we only care about the most active, so setting e_lo this 
            way serves as a threshold mechanism.
            '''
            gX[i, j] = Gmax(kSynX[k], mag, delEX[i, j])
            gY[i, j] = Gmax(kSynY[k], mag, delEY[i, j])
            gZ[i, j] = Gmax(kSynZ[k], mag, delEZ[i, j])

            gJ1[i, j] = Gmax(kSynJ1[k], mag, delEJ1[i, j])
            gJ2[i, j] = Gmax(kSynJ2[k], mag, delEJ2[i, j])
            gJ3[i, j] = Gmax(kSynJ3[k], mag, delEJ3[i, j])

            if gX[i, j] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gX[i, j]), reversal_potential=float(delEX[i, j]),
                                              e_lo=mag, e_hi=2 * mag)
                net.add_connection(synOutput, source=nameStr, destination='OutputEndX')
            else:
                1 + 1
            if gY[i, j] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gY[i, j]), reversal_potential=float(delEY[i, j]),
                                              e_lo=mag, e_hi=2 * mag)
                net.add_connection(synOutput, source=nameStr, destination='OutputEndY')
            else:
                1 + 1
            if gZ[i, j] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gZ[i, j]), reversal_potential=float(delEZ[i, j]),
                                              e_lo=mag, e_hi=2 * mag)
                net.add_connection(synOutput, source=nameStr, destination='OutputEndZ')
            else:
                1 + 1

            if gJ1[i, j] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gJ1[i, j]), reversal_potential=float(delEJ1[i, j]),
                                              e_lo=mag, e_hi=2 * mag)
                net.add_connection(synOutput, source=nameStr, destination='OutputJ1')
            else:
                1 + 1
            if gJ2[i, j] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gJ2[i, j]), reversal_potential=float(delEJ2[i, j]),
                                              e_lo=mag, e_hi=2 * mag)
                net.add_connection(synOutput, source=nameStr, destination='OutputJ2')
            else:
                1 + 1
            if gJ3[i, j] > 0:
                synOutput = NonSpikingSynapse(max_conductance=float(gJ3[i, j]), reversal_potential=float(delEJ3[i, j]),
                                              e_lo=mag, e_hi=2 * mag)
                net.add_connection(synOutput, source=nameStr, destination='OutputJ3')
            else:
                1 + 1
            # Increment k, which is a linear counting variable through all the for loops.
            k += 1
    # Concatenate the inputs into one network input array. Calculate the "Actual X, Y, Z values as a function of time",
    # ActualT.
    # Calculate and store the input current for the theta2 and theta3 sensory neurons using the bellCurve function.
    for i in range(numSensPerJoint):
        input2[:, i] = sigm_curve(mag, widthBell, theta=servo2vec, shift=jointTheta[i], jointThetaDiff=thetaDiff)
        input3[:, i] = sigm_curve(mag, widthBell, theta=servo3vec, shift=jointTheta[i], jointThetaDiff=thetaDiff)

    # Concatenate the inputs into one network input array. Calculate the "Actual X value as a function of time", ActualXt.

    for i in range(len(t)):
        inputNet[i, :] = np.concatenate((input2[i, :], input3[i, :]), axis=None)
        endPointTrnslMat = FrwrdKnmtcsFunc(wR1, np.array([0, theta2vec[i], theta3vec[i]]), pR1)
        ActualT[i, 0] = endPointTrnslMat[0, 3]  # x coord
        ActualT[i, 1] = endPointTrnslMat[1, 3]  # y coord
        ActualT[i, 2] = endPointTrnslMat[2, 3]  # z coord

        p1TrnslMat = FrwrdKnmtcsFunc(w1, 0, pR1[:, 0:2])
        p2TrnslMat = FrwrdKnmtcsFunc(np.concatenate((w1, w2R1), axis=1), [0, theta2vec[i]], pR1[:, 0:3])

        xyzPresentP1 = p1TrnslMat[0:3, 3]
        xyzPresentP2 = p2TrnslMat[0:3, 3]
        # xyzEndPoint = endPointTrnslMat[0:3, 3]
        R1s1 = p1TrnslMat[0:3, 0:3]
        R1s2 = p2TrnslMat[0:3, 0:3]
        # jacobian
        jacobian = JacobianFunc(wR1, pR1, R1s1, R1s2, xyzPresentP1, xyzPresentP2)
        ActualJ[i, 0] = jacobian[0, 2]
        ActualJ[i, 1] = jacobian[1, 2]
        ActualJ[i, 2] = jacobian[2, 2]

    numOut = net.get_num_outputs()
    # Compile the network model
    model = net.compile(backend='numpy', dt=dt)

    PredictedOut = np.zeros([numSteps, numOut])
    # Simulate the network response
    for i in range(len(t)):
        PredictedOut[i, :] = model(inputNet[i, :])
    return  PredictedOut, ActualT, ActualJ, gX, gY, gZ, model


def Gmax(k, R, delE):
    maxConduct = k * R / (delE - k * R)
    return maxConduct

def Normalize(matrix):
    vector = np.matrix(matrix).getA1()
    vectorNorm = (vector - min(vector)) / (max(vector - min(vector)))
    return vectorNorm

def ComboVec(theta2, theta3, mag, comboAct, gMax, delE):
    # position
    comboSS = np.reshape(theta2 + np.transpose(theta3) - mag, [np.size(comboAct, 1), ])
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

    jac1 = np.matmul(adjointS1, J2.reshape((6,1)))
    jac2 = np.matmul(adjointS2, J3.reshape((6,1)))

    jac1_1 = np.concatenate((J1.reshape((6,1)), jac1), axis=1)
    jacobianMatrix = np.concatenate((jac1_1, jac2), axis=1)
    return jacobianMatrix


# output: xyz2, xyz3, xyzEnd
ActualX = np.zeros([len(jointTheta), len(jointTheta)])
ActualY = np.zeros([len(jointTheta), len(jointTheta)])
ActualZ = np.zeros([len(jointTheta), len(jointTheta)])

# mapping matricies for Jacobian
jacobian3 = np.zeros([len(jointTheta), len(jointTheta)])
jacobian6 = np.zeros([len(jointTheta), len(jointTheta)])
jacobian9 = np.zeros([len(jointTheta), len(jointTheta)])

'''
Create the "training data" for the network, that is, the known X, Y, Z coordinates given the lists of theta values.
To fully generalize this approach to more than 2 joints/independent variables, we could do 2 things:
1. nest another for loop for each joint, but this approach may be difficult to implement
2. first, meshgrid (or NDgrid) all the joint angle lists into one master list, then use a single for loop.
'''
for i in range(len(jointTheta)):
    for j in range(len(jointTheta)):
        theta = np.array([0, jointTheta[i], jointTheta[j]])
        endPointTrnslMat = FrwrdKnmtcsFunc(wR1, theta, pR1)
        ActualX[i, j] = endPointTrnslMat[0, 3]  # x coord
        ActualY[i, j] = endPointTrnslMat[1, 3]  # y coord
        ActualZ[i, j] = endPointTrnslMat[2, 3]  # z coord

        p1TrnslMat = FrwrdKnmtcsFunc(w1, theta[0], pR1[:, 0:2])
        p2TrnslMat = FrwrdKnmtcsFunc(np.concatenate((w1, w2R1), axis=1), [theta[0], theta[1]], pR1[:, 0:3])

        xyzPresentP1 = p1TrnslMat[0:3, 3]
        xyzPresentP2 = p2TrnslMat[0:3, 3]
        xyzEndPoint = endPointTrnslMat[0:3, 3]
        # rotation mat
        R1s1 = p1TrnslMat[0:3, 0:3]
        R1s2 = p2TrnslMat[0:3, 0:3]
        # jacobian
        jacobian = JacobianFunc(wR1, pR1, R1s1, R1s2, xyzPresentP1, xyzPresentP2)
        jacobian3[i, j] = jacobian[0, 2]
        jacobian6[i, j] = jacobian[1, 2]
        jacobian9[i, j] = jacobian[2, 2]

'''
Generating two patterns of joint angles over time. Here, we have two sinusoidal functions of 
time with different periods. In this way, we sample the entire workspace.
'''

T2 = 99.8
T3 = 102.5
theta2vec = (thetaMin + thetaMax) / 2 + (thetaMax - thetaMin) / 2 * np.sin(2 * np.pi * t / T2)
theta3vec = (thetaMin + thetaMax) / 2 + (thetaMax - thetaMin) / 2 * np.sin(2 * np.pi * t / T3)

mag = 1
delayInit = 1
widthbellInit = 5  # 5
delaySlow = 1000

PredictedOutOne, ActualTOne, ActualJOne, kSynX, kSynY, kSynZ, modelFKJ = FKJSNS3D(delay=delayInit, delaySlow=delaySlow,
                                                                                  numJoints=numJoints,
                                                                                  numSensPerJoint=numSensPerJointInit,
                                                                                  numSteps=numSteps, widthBell=widthbellInit,
                                                                                  ActualMapX=ActualX, ActualMapY=ActualY,
                                                                                  ActualMapZ=ActualZ,
                                                                                  servo3vec=theta3vec, servo2vec=theta2vec,
                                                                                  jointTheta=jointTheta,
                                                                                  ActualJ1map=jacobian3, ActualJ2map=jacobian6,
                                                                                  ActualJ3map=jacobian9, kInhib=0)

predOutXOne = min(ActualTOne[:, 0]) + (max(ActualTOne[:, 0]) - min(ActualTOne[:, 0])) * PredictedOutOne[:, -6] / mag
predOutYOne = min(ActualTOne[:, 1]) + (max(ActualTOne[:, 1]) - min(ActualTOne[:, 1])) * PredictedOutOne[:, -5] / mag
predOutZOne = min(ActualTOne[:, 2]) + (max(ActualTOne[:, 2]) - min(ActualTOne[:, 2])) * PredictedOutOne[:, -4] / mag

predOutJ1 = min(ActualJOne[:, 0]) + (max(ActualJOne[:, 0]) - min(ActualJOne[:, 0])) * PredictedOutOne[:, -3] / mag
predOutJ2 = min(ActualJOne[:, 1]) + (max(ActualJOne[:, 1]) - min(ActualJOne[:, 1])) * PredictedOutOne[:, -2] / mag
predOutJ3 = min(ActualJOne[:, 2]) + (max(ActualJOne[:, 2]) - min(ActualJOne[:, 2])) * PredictedOutOne[:, -1] / mag

figBell, ax1 = plt.subplots()

ax1.set_xlabel('Time, ms')
ax1.set_ylabel('Sensory neural activation, mV', color='b')
jj = [-1.6, -.96, -.32, .32, .96, 1.6]
for i in range(len(np.linspace(0,numSensPerJointInit, numSensPerJointInit))):
# lns1=ax1.plot(t, PredictedOutOne[:, numSensPerJointInit*numJoints], label='Interneuron', linewidth=6, color='b')
    lns2=ax1.plot(t, PredictedOutOne[:, i], label='Activation of the ${\Theta}_{2}$ at '+str(jj[i])+' rads.', linewidth=1.0)#, color='tab:olive')
    # ax1.legend()
    # plt.legend()

# lns3=ax1.plot(t, PredictedOutOne[:, numSensPerJointInit-1], label='Activation of the ${\Theta}_{3}$ at -1.6 rads', linewidth=2.5, color='tab:orange')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.set_ylabel('Joint angle [${\Theta}$], rad', color='k')
lns4=ax2.plot(t, theta2vec, '--', color='k', label='Input ${\Theta}_{2}$', linewidth=2.0)
# lns5=ax2.plot(t, theta3vec, '--', color='orange', label='Input ${\Theta}_{3}$', linewidth=2.0)
ax2.tick_params(axis='y', color='k')
# lns = lns1+lns2+lns3+lns4+lns5
# labs = [l.get_label() for l in lns]
# ax1.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(20, 10)
plt.xlim([1600, 2000])
# plt.legend()

figBell.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Theta2respSigmSfN.svg', dpi=500)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# ax1 = plt.subplot(311)
# plt.title('Predicted vs Actual position X')
ax1.plot(t, predOutJ1, label='Predicted', color='red')
ax1.plot(t, ActualJOne[:, 0], label='Actual', color='k')
# plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
# ax1.set_ylabel('X, mm')
# plt.ylabel('X, mm')
# plt.savefig('PredActX.png', bbox_inches='tight')
# plt.show()

# ax2 = plt.subplot(312)
# plt.title('Predicted vs Actual position Y')
ax2.plot(t, predOutJ2, label='Predicted', color='limegreen')
ax2.plot(t, ActualJOne[:, 1], label='Actual', color='k')
plt.xlabel('Time, ms')
# plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
# ax2.set_ylabel('Y, mm')
# plt.savefig('PredActY.png', bbox_inches='tight')
# plt.show()

# ax3 = plt.subplot(313)
# plt.title('Predicted vs Actual position Z')
ax3.plot(t, predOutJ3, label='Predicted', color='royalblue')
ax3.plot(t, ActualJOne[:, 2], label='Actual', color='k')
plt.xlabel('Time, ms')
plt.legend()
# plt.ylabel('Z, mm')
plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
# plt.savefig('PredActJacSfN.svg', bbox_inches='tight', dpi=500)
plt.show()

delaySlowList = np.logspace(1.0, 4.0, num=4)
# checking inhibitory mem capacitance values
kInhList = np.linspace(0.1, 0.8, 8)
'''
for ind in range(len(delaySlowList)):
    for k in range(len(kInhList)):
        PredictedOutOne, ActualTOne, ActualJOne, kSynX, kSynY, kSynZ, modelFKJ = FKJSNS3D(delay=delayInit,
                                                                                        delaySlow=delaySlowList[ind],
                                                                                        numJoints=numJoints,
                                                                                        numSensPerJoint=numSensPerJointInit,
                                                                                        numSteps=numSteps,
                                                                                        widthBell=widthbellInit,
                                                                                        ActualMapX=ActualX,
                                                                                        ActualMapY=ActualY,
                                                                                        ActualMapZ=ActualZ,
                                                                                        servo3vec=theta3vec,
                                                                                        servo2vec=theta2vec,
                                                                                        jointTheta=jointTheta,
                                                                                        ActualJ1map=jacobian3,
                                                                                        ActualJ2map=jacobian6,
                                                                                        ActualJ3map=jacobian9,
                                                                                        kInhib=kInhList[k])

        predOutXOne = min(ActualTOne[:, 0]) + (max(ActualTOne[:, 0]) - min(ActualTOne[:, 0])) * PredictedOutOne[:, -6] / mag
        predOutYOne = min(ActualTOne[:, 1]) + (max(ActualTOne[:, 1]) - min(ActualTOne[:, 1])) * PredictedOutOne[:, -5] / mag
        predOutZOne = min(ActualTOne[:, 2]) + (max(ActualTOne[:, 2]) - min(ActualTOne[:, 2])) * PredictedOutOne[:, -4] / mag
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1 = plt.subplot(311)
        ax1.set_title('Pred vs Act position. Inh delay= '+ str(delaySlowList[ind])+ ' ms. kInh= '+ str(kInhList[k]))
        # plt.title('Predicted vs Actual position. Inh delay= '+ str(delaySlowList[ind]))
        ax1.plot(t, predOutXOne, label='Predicted', color='red')
        ax1.plot(t, ActualTOne[:, 0], label='Actual', color='k')
        plt.legend()
        ax1.set_ylabel('X, mm')
        # plt.ylabel('X, mm')
        # plt.savefig('PredActX.png', bbox_inches='tight')
        # plt.show()

        ax2 = plt.subplot(312)
        # plt.title('Predicted vs Actual position Y')
        ax2.plot(t, predOutYOne, label='Predicted', color='limegreen')
        ax2.plot(t, ActualTOne[:, 1], label='Actual', color='k')
        plt.xlabel('Time, ms')
        plt.legend()
        ax2.set_ylabel('Y, mm')
        # plt.savefig('PredActY.png', bbox_inches='tight')
        # plt.show()

        ax3 = plt.subplot(313)
        # plt.title('Predicted vs Actual position. Inh delay= '+ str(delaySlowList[ind]))
        ax3.plot(t, predOutZOne, label='Predicted', color='royalblue')
        ax3.plot(t, ActualTOne[:, 2], label='Actual', color='k')
        plt.xlabel('Time, ms')
        plt.legend()
        plt.ylabel('Z, mm')
        # plt.savefig('PredActErrorsSfN'+ str(delaySlowList[ind])+ 'k'+ str(kInhList[k])+ '.svg', bbox_inches='tight', dpi=500)
        # plt.show()
        
'''

widthMin = 5
widthMax = 31 #5
widthList = np.arange(widthMin, widthMax, 1) #5
numSensPerJointList = np.arange(5, 24, 1) #6

# widthList = np.array([10, 6]) #5
# numSensPerJointList = np.array([11, 6]) #6
# coords
rmseSensX = np.zeros([len(widthList), len(numSensPerJointList)])
rmseSensY = np.zeros([len(widthList), len(numSensPerJointList)])
rmseSensZ = np.zeros([len(widthList), len(numSensPerJointList)])
# jacobian
rmseSensJ1 = np.zeros([len(widthList), len(numSensPerJointList)])
rmseSensJ2 = np.zeros([len(widthList), len(numSensPerJointList)])
rmseSensJ3 = np.zeros([len(widthList), len(numSensPerJointList)])

for widthVar in range(len(widthList)):
    for k in range(len(numSensPerJointList)):
        jointThetaVar = np.linspace(thetaMin, thetaMax, numSensPerJointList[k])
        ActualXVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        ActualYVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        ActualZVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])

        jacobian3Var = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        jacobian6Var = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        jacobian9Var = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        for i in range(len(jointThetaVar)):
            for j in range(len(jointThetaVar)):
                endPointTrnslMat = FrwrdKnmtcsFunc(wR1, np.array([0, jointThetaVar[i], jointThetaVar[j]]), pR1)
                ActualXVar[i, j] = endPointTrnslMat[0, 3]  # x coord
                ActualYVar[i, j] = endPointTrnslMat[1, 3]  # y coord
                ActualZVar[i, j] = endPointTrnslMat[2, 3]  # z coord
                p1TrnslMat = FrwrdKnmtcsFunc(w1, 0, pR1[:, 0:2])
                p2TrnslMat = FrwrdKnmtcsFunc(np.concatenate((w1, w2R1), axis=1), [0, jointThetaVar[i]], pR1[:, 0:3])

                xyzPresentP1 = p1TrnslMat[0:3, 3]
                xyzPresentP2 = p2TrnslMat[0:3, 3]
                xyzEndPoint = endPointTrnslMat[0:3, 3]
                # rotation mat
                R1s1 = p1TrnslMat[0:3, 0:3]
                R1s2 = p2TrnslMat[0:3, 0:3]
                # jacobian
                jacobian = JacobianFunc(wR1, pR1, R1s1, R1s2, xyzPresentP1, xyzPresentP2)
                jacobian3Var[i, j] = jacobian[0, 2]
                jacobian6Var[i, j] = jacobian[1, 2]
                jacobian9Var[i, j] = jacobian[2, 2]

        PredictedOut, ActualT, ActualJ, kSynXvar, kSynYvar, kSynZvar, _ = FKJSNS3D(delay=delayInit, delaySlow=delaySlow,
                                                                                numJoints=numJoints, numSensPerJoint=numSensPerJointList[k],
                                                                                numSteps=numSteps, widthBell=widthList[widthVar],
                                                                                ActualMapX=ActualXVar, ActualMapY=ActualYVar,
                                                                                ActualMapZ=ActualZVar, servo2vec=theta2vec,
                                                                                servo3vec=theta3vec, jointTheta=jointThetaVar,
                                                                                ActualJ1map=jacobian3Var, ActualJ2map=jacobian6Var,
                                                                                ActualJ3map=jacobian9Var, kInhib=0)
        predOutX = min(ActualT[:, 0]) + (max(ActualT[:, 0]) - min(ActualT[:, 0])) * PredictedOut[:, -6] / mag
        predOutY = min(ActualT[:, 1]) + (max(ActualT[:, 1]) - min(ActualT[:, 1])) * PredictedOut[:, -5] / mag
        predOutZ = min(ActualT[:, 2]) + (max(ActualT[:, 2]) - min(ActualT[:, 2])) * PredictedOut[:, -4] / mag

        predOutJ1 = min(ActualJ[:, 0]) + (max(ActualJ[:, 0]) - min(ActualJ[:, 0])) * PredictedOut[:, -6] / mag
        predOutJ2 = min(ActualJ[:, 1]) + (max(ActualJ[:, 1]) - min(ActualJ[:, 1])) * PredictedOut[:, -5] / mag
        predOutJ3 = min(ActualJ[:, 2]) + (max(ActualT[:, 2]) - min(ActualJ[:, 2])) * PredictedOut[:, -4] / mag

        # slopeXstat[widthVar, k], _, rXstat[widthVar, k], _, _ = sp.stats.linregress(ActualT[:, 0], predOutX)
        # slopeYstat[widthVar, k], _, rYstat[widthVar, k], _, _ = sp.stats.linregress(ActualT[:, 1], predOutY)
        # slopeZstat[widthVar, k], _, rZstat[widthVar, k], _, _ = sp.stats.linregress(ActualT[:, 2], predOutZ)

        rmseSensX[widthVar, k] = rmse(predOutX, ActualT[:, 0])
        rmseSensY[widthVar, k] = rmse(predOutY, ActualT[:, 1])
        rmseSensZ[widthVar, k] = rmse(predOutZ, ActualT[:, 2])

        # jacobian errors
        # slopeJ1stat[widthVar, k], _, rJ1stat[widthVar, k], _, _ = sp.stats.linregress(ActualJ[:, 0], predOutJ1)
        # slopeJ2stat[widthVar, k], _, rJ2stat[widthVar, k], _, _ = sp.stats.linregress(ActualJ[:, 1], predOutJ2)
        # slopeJ3stat[widthVar, k], _, rJ3stat[widthVar, k], _, _ = sp.stats.linregress(ActualJ[:, 2], predOutJ3)

        rmseSensJ1[widthVar, k] = rmse(predOutJ1, ActualJ[:, 0])
        rmseSensJ2[widthVar, k] = rmse(predOutJ2, ActualJ[:, 1])
        rmseSensJ3[widthVar, k] = rmse(predOutJ3, ActualJ[:, 2])

        print('X width ' + str(widthList[widthVar]) + '. Num sens ' + str(numSensPerJointList[k]) + ' done')
        print('Y width ' + str(widthList[widthVar]) + '. Num sens ' + str(numSensPerJointList[k]) + ' done')
        print('Z width ' + str(widthList[widthVar]) + '. Num sens ' + str(numSensPerJointList[k]) + ' done')

rmseSens = rmseSensX+rmseSensY+rmseSensZ
rmseSensJ = rmseSensJ1+rmseSensJ2+rmseSensJ3
rmseTotal = rmseSens+rmseSensJ

cc=np.linspace(widthMin, 1, widthMax)
colors2 = plt.cm.jet(cc)
colors = plt.get_cmap('jet', widthMax)
fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(7, 8))

for widthVar in range(len(widthList)):
    if widthVar==0:
        ax4.plot(numSensPerJointList, rmseSensX[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax4.plot(numSensPerJointList, rmseSensX[widthVar, :], c=colors(widthVar))
# plt.xlabel('Number of sensory neurons per joint')
ax4.set_ylabel('RMSE, X, mm')
sm = matplotlib.cm.ScalarMappable(cmap=colors, norm=matplotlib.colors.Normalize(vmin=widthMin, vmax=widthMax))
sm.set_array([])
plt.colorbar(sm, ax=ax4)
ax4.set_ylim([25, 200])
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumX.png', bbox_inches='tight')
# plt.show()

# ax1 = plt.subplot(312)
# plt.title('RMSE. Varying width and num sens per joint, Y coord')
for widthVar in range(len(widthList)):
    if widthVar == 0:
        ax5.plot(numSensPerJointList, rmseSensY[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax5.plot(numSensPerJointList, rmseSensY[widthVar, :], c=colors(widthVar))
# plt.xlabel('Number of sensory neurons per joint')
ax5.set_ylabel('RMSE, Y, mm')
plt.colorbar(sm, ax=ax5)
ax5.set_ylim([25, 200])
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumY.png', bbox_inches='tight')
# plt.show()

# ax1 = plt.subplot(313)
# plt.title('RMSE. Varying c and num sens per joint, Z coord')
for widthVar in range(len(widthList)):
    if widthVar == 0:
        ax6.plot(numSensPerJointList, rmseSensZ[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax6.plot(numSensPerJointList, rmseSensZ[widthVar, :], c=colors(widthVar))
plt.xlabel('Number of sensory neurons per joint')
ax6.set_ylabel('RMSE, Z, mm')
plt.colorbar(sm, ax=ax6)
ax6.set_ylim([25, 250])
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc=7)

# plt.savefig('WidthNumSigmUpdFKJFKSfN.svg', dpi=500)
bbox_inches='tight'
plt.show()

cc=np.linspace(widthMin, 1, widthMax)
colors2 = plt.cm.jet(cc)
colors = plt.get_cmap('jet', widthMax)

fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(7, 8))
for widthVar in range(len(widthList)):
    if widthVar==0:
        ax4.plot(numSensPerJointList, rmseSensJ1[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax4.plot(numSensPerJointList, rmseSensJ1[widthVar, :], c=colors(widthVar))
# plt.xlabel('Number of sensory neurons per joint')
ax4.set_ylabel('RMSE, J1, mm')
# ax4.set_ylim([0, 1500])
ax4.set_ylim([10, 50])
sm = matplotlib.cm.ScalarMappable(cmap=colors, norm=matplotlib.colors.Normalize(vmin=widthMin, vmax=widthMax))
sm.set_array([])
plt.colorbar(sm, ax=ax4)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumSigmJ1SfN.svg', bbox_inches='tight')
# plt.show()

# ax1 = plt.subplot(312)
# plt.title('RMSE. Varying width and num sens per joint, Y coord')
for widthVar in range(len(widthList)):
    if widthVar == 0:
        ax5.plot(numSensPerJointList, rmseSensJ2[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax5.plot(numSensPerJointList, rmseSensJ2[widthVar, :], c=colors(widthVar))
# plt.xlabel('Number of sensory neurons per joint')
ax5.set_ylabel('RMSE, J2')
ax5.set_ylim([10, 50])
# ax5.set_ylim([0, 1500])
plt.colorbar(sm, ax=ax5)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumSigmJ2SfN.svg', bbox_inches='tight')
# plt.show()

# ax1 = plt.subplot(313)
# plt.title('RMSE. Varying c and num sens per joint, Z coord')
for widthVar in range(len(widthList)):
    if widthVar == 0:
        ax6.plot(numSensPerJointList, rmseSensJ3[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax6.plot(numSensPerJointList, rmseSensJ3[widthVar, :], c=colors(widthVar))
plt.xlabel('Number of sensory neurons per joint')
ax6.set_ylabel('RMSE, J3')
plt.colorbar(sm, ax=ax6)
ax6.set_ylim([100, 220])
# plt.ylim([0, 1500])
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc=7)

# plt.savefig('WidthNumSigmUpdJSfNupd2221.svg', dpi=500)
bbox_inches='tight'
plt.show()

