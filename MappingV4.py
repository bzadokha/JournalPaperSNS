# Living Machines Paper 2024
# using a sns-toolbox to map xyz
# final version
import matplotlib.cm
import scipy
import numpy as np
from sns_toolbox.connections import NonSpikingSynapse
import sns_toolbox.networks
import matplotlib.pyplot as plt
from sns_toolbox.neurons import NonSpikingNeuron
import time
import seaborn as sns

'''
Bohdan Zadokha and Nicholas Szczecinski
26 March 2024
WVU NeuroMINT
'''
startTime = time.time()

# 3d space
# R1 leg
L1 = 52.52 # mm
L2 = 60 # mm
L3 = 95 # mm
# distance from the CoM (0,0,0)
x0 = 125 # mm
y02 = 100 # mm
y01 = 65 # mm
y03 = 65 # mm

w1 = np.array([[0], [0], [1]])
# R1 leg
w2R1 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)], [0]])
wR1 = np.concatenate((np.concatenate((w1, w2R1), axis=1), w2R1), axis=1)

# zero position of the R1 leg
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

# time vector for simulation
dt = 1 # step
tMax = 2e3 # time vector length (2000 ms)
t = np.arange(0, tMax, dt)
numSteps = len(t)

numJoints = 2 # number of joints
numSensPerJoint = 11 # number of interneuron compartments

thetaMin = -1.6 # min joint angle
thetaMax = 1.6 # max joint angle
# joint angles vector for mapping
jointTheta = np.linspace(thetaMin, thetaMax, numSensPerJoint)

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


# output: xyz2, xyz3, xyzEnd
ActualX = np.zeros([len(jointTheta), len(jointTheta)])
ActualY = np.zeros([len(jointTheta), len(jointTheta)])
ActualZ = np.zeros([len(jointTheta), len(jointTheta)])

'''
Create the "training data" for the network, that is, the known X, Y, Z coordinates given the lists of theta values.
To fully generalize this approach to more than 2 joints/independent variables, we could do 2 things:
1. nest another for loop for each joint, but this approach may be difficult to implement
2. first, meshgrid (or NDgrid) all the joint angle lists into one master list, then use a single for loop.
'''
for i in range(len(jointTheta)):
    for j in range(len(jointTheta)):
        p1TrnslMat = FrwrdKnmtcsFunc(wR1, np.array([0, jointTheta[i], jointTheta[j]]), pR1)
        ActualX[i, j] = p1TrnslMat[0, 3]  # x coord
        ActualY[i, j] = p1TrnslMat[1, 3]  # y coord
        ActualZ[i, j] = p1TrnslMat[2, 3]  # z coord

'''
Generating two patterns of joint angles over time. Here, we have two sinusoidal functions of 
time with different periods. In this way, we sample the entire workspace.
'''
T2 = 400
T3 = 500
theta2vec = (thetaMin + thetaMax) / 2 + (thetaMax - thetaMin) / 2 * np.sin(2 * np.pi * t / T2)
theta3vec = (thetaMin + thetaMax) / 2 + (thetaMax - thetaMin) / 2 * np.sin(2 * np.pi * t / T3)

'''
Define the "bell curve" function that maps joint angle into external current for a sensory neuron.
'''
def bell_curve(magnitude, width, theta, shift):
    return magnitude*np.exp(-width*pow((theta-shift), 2))

'''
Define the root mean square error function.
'''
def rmse(pred, actual):
    return np.sqrt(((actual - pred) ** 2).mean())

def frwrdKnmtcsSNS3D(delay, numJoints, numSensPerJoint, numSteps,
                   widthBell, ActualMapX, ActualMapY, ActualMapZ,
                   servo2vec, servo3vec, jointTheta):
    # Create a network object. Here, it is called 'netX', because it computes the x-coordinate of the foot.
    # However, it should be noted that the same network could be used to compute just about anything
    # by re-using the combo neurons (described below) and creating an additional output neuron with unique synapses
    # from the combo neurons.
    net = sns_toolbox.networks.Network(name='endPointPrediction')

    # Create a basic nonspiking neuron type.
    bellNeur = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1)
    '''
    Create the "bell curve" neurons, which are the sensory neurons. Each sensory neuron has a bell-curve receptive field
    with a unique "preferred angle", that is, the angle at which the peak response is evoked.
    '''
    for index in range(numSensPerJoint):
        nameStr2 = 'Bell2' + str(index)
        net.add_neuron(neuron_type=bellNeur, name=nameStr2)
        net.add_input(nameStr2)
        net.add_output(nameStr2)

    for index in range(numSensPerJoint):
        nameStr3 = 'Bell3' + str(index)
        net.add_neuron(neuron_type=bellNeur, name=nameStr3)
        net.add_input(nameStr3)
        net.add_output(nameStr3)

    '''
    Define key neuron and synapse parameters. 
    'mag' is like R in the functional subnetwork publications: it is the maximum expected neural activation. 
    delE is the synaptic reversal potential relative to the rest potential of the neurons.
    widthBell is the width of the sensory-encoding bell curves. 
    Note that these have been normalized to the range of theta and the number of sensory neurons. 
    Also note that smaller value of widthBell makes the curves broader.
    '''
    mag = 1
    delE = 20
    '''
    Initialize empty arrays to store input current for each set of sensory neurons (theta2 = input2, theta3 = input3), the
    input current for the whole network (inputX), and the validation data X(t).
    '''
    input2 = np.zeros([numSteps, numSensPerJoint])  # bell curve responses
    input3 = np.zeros([numSteps, numSensPerJoint])  # bell curve responses
    inputNet = np.zeros([numSteps, numSensPerJoint * numJoints])
    # Actual X, Y, Z values as a function of time
    ActualT = np.zeros([numSteps, 3])
    '''
    Each sensory neuron synapses onto a number of "combo" neurons. Each combo neuron receives synaptic input from one
    sensory neuron from each joint. All these synapses may be identical, because they are simply generating all the possible
    combinations of the joint angles/independent variables. All the combo neurons may be identical, because they are simply
    integrating the joint angles.
    '''
    kIdent = 1
    gIdent = kIdent * mag / (delE - kIdent * mag)  # From Szczecinski, Hunt, and Quinn 2017
    identitySyn = NonSpikingSynapse(max_conductance=gIdent, reversal_potential=delE, e_lo=0, e_hi=mag)
    comboNeur = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1, bias=0)
    net.add_neuron(neuron_type=bellNeur, name='OutputEndX')
    net.add_neuron(neuron_type=bellNeur, name='OutputEndY')
    net.add_neuron(neuron_type=bellNeur, name='OutputEndZ')
    '''
    For the network to perform the desired calculation, the synapses from each combo neuron to the output neuron should have
    effective gains (i.e., Vout/Vcombo) that are proportional to the value that the output neuron encodes. Here, we take all
    the "training data", that is, the actual X coordinate of the leg, normalize it to values between 0 and 1, and use those
    to set the unique properties of the synapse from each combo neuron to the output neuron.
    '''
    mapVectorX = np.matrix(ActualMapX).getA1()
    mapVectorY = np.matrix(ActualMapY).getA1()
    mapVectorZ = np.matrix(ActualMapZ).getA1()

    mapNormX = (mapVectorX - min(mapVectorX)) / (max(mapVectorX - min(mapVectorX)))
    mapNormY = (mapVectorY - min(mapVectorY)) / (max(mapVectorY - min(mapVectorY)))
    mapNormZ = (mapVectorZ - min(mapVectorZ)) / (max(mapVectorZ - min(mapVectorZ)))
    '''
    Now we create the each combo neuron, its input connections, and its output connections. Here, we have 2 nested for
    loops, one for each joint/independent variable. As stated above, a higher-dimensional network would require more for 
    loops or an approach in which we NDgrid the joint angles/sensory neurons and then use a single for loop.
    '''
    k = 0
    for i in range(numSensPerJoint):
        for j in range(numSensPerJoint):
            nameStr = 'combo' + str(i) + str(j)
            net.add_neuron(neuron_type=comboNeur, name=nameStr)
            net.add_connection(identitySyn, source='Bell2' + str(i), destination=nameStr)
            net.add_connection(identitySyn, source='Bell3' + str(j), destination=nameStr)
            net.add_output(nameStr)
            # SNS-Toolbox does not enable synapses with max_cond = 0, so if kSyn = 0, we must pass it machine epsilon instead.
            
            kSynX = max(mapNormX[k], np.finfo(float).eps)
            kSynY = max(mapNormY[k], np.finfo(float).eps)
            kSynZ = max(mapNormZ[k], np.finfo(float).eps)
            gX = kSynX * mag / (delE - kSynX * mag)
            gY = kSynY * mag / (delE - kSynX * mag)
            gZ = kSynZ * mag / (delE - kSynX * mag)
            # print('Synapse from neuron ' + nameStr + ' has g = ' + str(g) + 'uS.')

            
            # Synapses from the combo neurons to the output neuron(s) each have unique conductance value that corresponds to
            # the desired output in that scenario. Note that the e_lo for the synapses = mag and e_hi = 2*mag, because
            # multiple combo neurons will be active at any time, but we only care about the most active, so setting e_lo this 
            # way serves as a threshold mechanism.
            
            synOutputX = NonSpikingSynapse(max_conductance=gX, reversal_potential=delE, e_lo=mag, e_hi=2 * mag)
            synOutputY = NonSpikingSynapse(max_conductance=gY, reversal_potential=delE, e_lo=mag, e_hi=2 * mag)
            synOutputZ = NonSpikingSynapse(max_conductance=gZ, reversal_potential=delE, e_lo=mag, e_hi=2 * mag)
            net.add_connection(synOutputX, source=nameStr, destination='OutputEndX')
            net.add_connection(synOutputY, source=nameStr, destination='OutputEndY')
            net.add_connection(synOutputZ, source=nameStr, destination='OutputEndZ')
            # Increment k, which is a linear counting variable through all the for loops.
            k += 1
    net.add_output('OutputEndX')
    net.add_output('OutputEndY')
    net.add_output('OutputEndZ')

    # Calculate and store the input current for the theta2 and theta3 sensory neurons using the bellCurve function.
    for i in range(numSensPerJoint):
        input2[:, i] = bell_curve(mag, widthBell, theta=servo2vec, shift=jointTheta[i])
        input3[:, i] = bell_curve(mag, widthBell, theta=servo3vec, shift=jointTheta[i])

    # Concatenate the inputs into one network input array. Calculate the "Actual X, Y, Z values as a function of time",
    # ActualT.
    for i in range(len(t)):
        inputNet[i, :] = np.concatenate((input2[i, :], input3[i, :]), axis=None)
        p1TrnslMat = FrwrdKnmtcsFunc(wR1, np.array([0, theta2vec[i], theta3vec[i]]), pR1)
        ActualT[i, 0] = p1TrnslMat[0, 3]  # x coord
        ActualT[i, 1] = p1TrnslMat[1, 3]  # y coord
        ActualT[i, 2] = p1TrnslMat[2, 3]  # z coord

    numOut = net.get_num_outputs()
    PredictedOut = np.zeros([numSteps, numOut])

    # Compile the network model
    model = net.compile(backend='numpy', dt=dt)
    # render(net, view=True, save=True, filename='MapNet', img_format='png')
    # Simulate the network response
    for i in range(len(t)):
        PredictedOut[i, :] = model(inputNet[i, :])
    return PredictedOut, ActualT
# same func but only for one output: x, y, or z

mag = 1
delayInit = 2
numSensPerJointInit = 11 # 11
widthbellInit = 20 # 20

widthList = np.arange(8, 35, 1)
delayList = np.arange(2, 45, 2)
numSensPerJointList = np.arange(5, 20, 1)

rmseDelay = np.zeros([3, len(delayList)])
rmseSensX = np.zeros([len(widthList), len(numSensPerJointList)])
rmseSensY = np.zeros([len(widthList), len(numSensPerJointList)])
rmseSensZ = np.zeros([len(widthList), len(numSensPerJointList)])

slopeXstat = np.zeros([len(widthList), len(numSensPerJointList)])
slopeYstat = np.zeros([len(widthList), len(numSensPerJointList)])
slopeZstat = np.zeros([len(widthList), len(numSensPerJointList)])

rXstat = np.zeros([len(widthList), len(numSensPerJointList)])
rYstat = np.zeros([len(widthList), len(numSensPerJointList)])
rZstat = np.zeros([len(widthList), len(numSensPerJointList)])

PredictedOutOne, ActualTOne = frwrdKnmtcsSNS3D(delay=delayInit, numJoints=numJoints, numSensPerJoint=numSensPerJointInit,
                                       numSteps=numSteps, widthBell=widthbellInit,
                                       ActualMapX=ActualX, ActualMapY=ActualY, ActualMapZ=ActualZ,
                                       servo2vec=theta2vec, servo3vec=theta3vec, jointTheta=jointTheta)
predOutXOne = min(ActualTOne[:, 0]) + (max(ActualTOne[:, 0]) - min(ActualTOne[:, 0])) * PredictedOutOne[:, -3] / mag
predOutYOne = min(ActualTOne[:, 1]) + (max(ActualTOne[:, 1]) - min(ActualTOne[:, 1])) * PredictedOutOne[:, -2] / mag
predOutZOne = min(ActualTOne[:, 2]) + (max(ActualTOne[:, 2]) - min(ActualTOne[:, 2])) * PredictedOutOne[:, -1] / mag

rmseXcoord = rmse(predOutXOne, ActualTOne[:, 0])
rmseYcoord = rmse(predOutYOne, ActualTOne[:, 1])
rmseZcoord = rmse(predOutZOne, ActualTOne[:, 2])
print('rmseX= '+str(rmseXcoord))
print('rmseY= '+str(rmseYcoord))
print('rmseZ= '+str(rmseZcoord))
errX = (ActualTOne[:, 0] - predOutXOne)
errY = (ActualTOne[:, 1] - predOutYOne)
errZ = (ActualTOne[:, 2] - predOutZOne)

# corr stat
# x
slopeX, interceptX, rX, pX, stderrX = scipy.stats.linregress(ActualTOne[:, 0], predOutXOne)
lineX = f'Regression line: y={interceptX:.2f}+{slopeX:.2f}*Actual, r={rX:.2f}'
figX, axX = plt.subplots()
axX.plot(ActualTOne[:, 0], predOutXOne, linewidth=0, marker='s', label='Data points', color='r')
axX.plot(ActualTOne[:, 0], interceptX + slopeX * ActualTOne[:, 0], label='Regression line', color='k', linewidth=2)
axX.set_xlabel('Actual, mm')
axX.set_ylabel('Predicted, mm')
# plt.title('X lin reg')
axX.legend(facecolor='white')
# plt.savefig('X lin reg.png', bbox_inches='tight', dpi=500)
plt.show()

# y
slopeY, interceptY, rY, pY, stderrY = scipy.stats.linregress(ActualTOne[:, 1], predOutYOne)
lineY = f'Regression line: y={interceptY:.2f}+{slopeY:.2f}*Actual, r={rY:.2f}'
figY, axY = plt.subplots()
axY.plot(ActualTOne[:, 1], predOutYOne, linewidth=0, marker='s', label='Data points', color='g')
axY.plot(ActualTOne[:, 1], interceptY + slopeY * ActualTOne[:, 1], label='Regression line', color='k', linewidth=2)
axX.set_xlabel('Actual, mm')
axX.set_ylabel('Predicted, mm')
# plt.title('Y lin reg')
axY.legend(facecolor='white')
# plt.savefig('Y lin reg.png', bbox_inches='tight', dpi=500)
plt.show()

# z
slopeZ, interceptZ, rZ, pZ, stderrZ = scipy.stats.linregress(ActualTOne[:, 2], predOutZOne)
lineZ = f'Regression line: y={interceptZ:.2f}+{slopeZ:.2f}*Actual, r={rZ:.2f}'
figZ, axZ = plt.subplots()
axZ.plot(ActualTOne[:, 2], predOutZOne, linewidth=0, marker='s', label='Data points', color='b')
axZ.plot(ActualTOne[:, 2], interceptZ + slopeZ * ActualTOne[:, 2], label='Regression line', color='k', linewidth=2)
axX.set_xlabel('Actual, mm')
axX.set_ylabel('Predicted, mm')
# plt.title('Z lin reg')
axZ.legend(facecolor='white')
# plt.savefig('Z lin reg.png', bbox_inches='tight', dpi=500)
plt.show()

# histogram stat
fig12, axs = plt.subplots(3, 1, figsize=(10, 10))
# plt.figure()
# sns.set_style('whitegrid')
sns.histplot(errX, kde=True, color='red', ax=axs[0])
axs[0].axvline(errX.mean(), color='k', linestyle='dashed', linewidth=2.5)
axs[0].axvline(np.median(errX), color='m', linestyle='dashdot', linewidth=2.5)
axs[0].set_ylabel('Count, X coord.')
axs[0].set_xlabel('Error, mm')
axs[0].set_xlim([-150, 150])
# plt.xlim(right=150)
# plt.show()
# plt.figure()
sns.histplot(errY, kde=True, color='green', ax=axs[1])
axs[1].axvline(errY.mean(), color='k', linestyle='dashed', linewidth=2.5)
axs[1].axvline(np.median(errY), color='m', linestyle='dashdot', linewidth=2.5)
axs[1].set_xlim([-150, 150])
# axs[1].text(-15, 7, 'Mean: {:.2f}'.format(errY.mean()))
axs[1].set_xlabel('Error, mm')
axs[1].set_ylabel('Count, Y coord.')
# plt.xlim(right=150)
# plt.show()
# plt.figure()
sns.histplot(errZ, kde=True, color='blue', ax=axs[2])
axs[2].axvline(errZ.mean(), color='k', linestyle='dashed', linewidth=2.5)
axs[2].axvline(np.median(errZ), color='m', linestyle='dashdot', linewidth=2.5)
axs[2].set_xlim([-150, 150])
axs[2].set_xlabel('Error, mm')
axs[2].set_ylabel('Count, Z coord.')
# plt.savefig('HistUpd.png', bbox_inches='tight', dpi=500)
# plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# ax1 = plt.subplot(311)
# plt.title('Predicted vs Actual position X')
ax1.plot(t, predOutXOne, label='Predicted', color='red')
ax1.plot(t, ActualTOne[:, 0], label='Actual', color='k')
# plt.legend()
ax1.set_ylabel('X, mm')
# plt.ylabel('X, mm')
# plt.savefig('PredActX.png', bbox_inches='tight')
# plt.show()

# ax2 = plt.subplot(312)
# plt.title('Predicted vs Actual position Y')
ax2.plot(t, predOutYOne, label='Predicted', color='limegreen')
ax2.plot(t, ActualTOne[:, 1], label='Actual', color='k')
plt.xlabel('Time, ms')
# plt.legend()
ax2.set_ylabel('Y, mm')
# plt.savefig('PredActY.png', bbox_inches='tight')
# plt.show()

# ax3 = plt.subplot(313)
# plt.title('Predicted vs Actual position Z')
ax3.plot(t, predOutZOne, label='Predicted', color='royalblue')
ax3.plot(t, ActualTOne[:, 2], label='Actual', color='k')
plt.xlabel('Time, ms')
# plt.legend()
plt.ylabel('Z, mm')
# plt.savefig('PredActUpd3.png', bbox_inches='tight', dpi=500)
plt.show()

# checking mem capacitance values
'''
for ind in range(len(delayList)):
    PredictedOut, ActualT = frwrdKnmtcsSNS3D(delay=delayList[ind], numJoints=numJoints, numSensPerJoint=numSensPerJoint,
                                           numSteps=numSteps, widthBell=widthbellInit,
                                           ActualMapX=ActualX, ActualMapY=ActualY, ActualMapZ=ActualZ,
                                           servo2vec=theta2vec,servo3vec=theta3vec, jointTheta=jointTheta)
    predOutX = min(ActualT[:, 0]) + (max(ActualT[:, 0]) - min(ActualT[:, 0])) * PredictedOut[:, -3] / mag
    predOutY = min(ActualT[:, 1]) + (max(ActualT[:, 1]) - min(ActualT[:, 1])) * PredictedOut[:, -2] / mag
    predOutZ = min(ActualT[:, 2]) + (max(ActualT[:, 2]) - min(ActualT[:, 2])) * PredictedOut[:, -1] / mag
    rmseDelay[0, ind] = rmse(predOutX, ActualT[:, 0], numSteps=numSteps)
    rmseDelay[1, ind] = rmse(predOutY, ActualT[:, 1], numSteps=numSteps)
    rmseDelay[2, ind] = rmse(predOutZ, ActualT[:, 2], numSteps=numSteps)
plt.figure()
plt.plot(delayList, rmseDelay[0, :], label='x', color='red')
plt.plot(delayList, rmseDelay[1, :], label='y', color='limegreen')
plt.plot(delayList, rmseDelay[2, :], label='z', color='royalblue')
plt.xlabel('Membrane capacitance of the neuron, nF')
plt.ylabel('RMSE, mm')
plt.legend()
plt.savefig('RMSEDelayUpd.png', bbox_inches='tight', dpi=500)
plt.show()
'''
# bell curve width & num sens neurons per joint
'''
for widthVar in range(len(widthList)):
    for k in range(len(numSensPerJointList)):
        jointThetaVar = np.linspace(thetaMin, thetaMax, numSensPerJointList[k])
        ActualXVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        ActualYVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        ActualZVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        for i in range(len(jointThetaVar)):
            for j in range(len(jointThetaVar)):
                p1TrnslMat = FrwrdKnmtcsFunc(wR1, np.array([0, jointThetaVar[i], jointThetaVar[j]]), pR1)
                ActualXVar[i, j] = p1TrnslMat[0, 3]  # x coord
                ActualYVar[i, j] = p1TrnslMat[1, 3]  # y coord
                ActualZVar[i, j] = p1TrnslMat[2, 3]  # z coord
        PredictedOut, ActualT = frwrdKnmtcsSNS3D(delay=delayInit, numJoints=numJoints, numSensPerJoint=numSensPerJointList[k],
                                                 numSteps=numSteps, widthBell=widthList[widthVar],
                                                 ActualMapX=ActualXVar, ActualMapY=ActualYVar, ActualMapZ=ActualZVar,
                                                 servo2vec=theta2vec, servo3vec=theta3vec, jointTheta=jointThetaVar)
        predOutX = min(ActualT[:, 0]) + (max(ActualT[:, 0]) - min(ActualT[:, 0])) * PredictedOut[:, -3] / mag
        predOutY = min(ActualT[:, 1]) + (max(ActualT[:, 1]) - min(ActualT[:, 1])) * PredictedOut[:, -2] / mag
        predOutZ = min(ActualT[:, 2]) + (max(ActualT[:, 2]) - min(ActualT[:, 2])) * PredictedOut[:, -1] / mag
        slopeXstat[widthVar, k], _, rXstat[widthVar, k], _, _ = scipy.stats.linregress(ActualT[:, 0], predOutX)
        slopeYstat[widthVar, k], _, rYstat[widthVar, k], _, _ = scipy.stats.linregress(ActualT[:, 1], predOutY)
        slopeZstat[widthVar, k], _, rZstat[widthVar, k], _, _ = scipy.stats.linregress(ActualT[:, 2], predOutZ)
        rmseSensX[widthVar, k] = rmse(predOutX, ActualT[:, 0], numSteps=numSteps)
        rmseSensY[widthVar, k] = rmse(predOutY, ActualT[:, 1], numSteps=numSteps)
        rmseSensZ[widthVar, k] = rmse(predOutZ, ActualT[:, 2], numSteps=numSteps)
        print('X width ' + str(widthList[widthVar]) + '. Num sens ' + str(numSensPerJointList[k]) + ' done')
        print('Y width ' + str(widthList[widthVar]) + '. Num sens ' + str(numSensPerJointList[k]) + ' done')
        print('Z width ' + str(widthList[widthVar]) + '. Num sens ' + str(numSensPerJointList[k]) + ' done')
'''
'''
cc=np.linspace(8, 1, 35)
colors2 = plt.cm.jet(cc)
colors = plt.get_cmap('jet', 35)
fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(7, 8))
# ax1 = plt.subplot(311)
# plt.title('RMSE. Varying width and num sens per joint, X coord')
for widthVar in range(len(widthList)):
    if widthVar==12:
        ax4.plot(numSensPerJointList, rmseSensX[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax4.plot(numSensPerJointList, rmseSensX[widthVar, :], c=colors(widthVar))
# plt.xlabel('Number of sensory neurons per joint')
ax4.set_ylabel('RMSE, X, mm')

sm = matplotlib.cm.ScalarMappable(cmap=colors, norm=matplotlib.colors.Normalize(vmin=8, vmax=35))
sm.set_array([])
plt.colorbar(sm, ax=ax4)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumX.png', bbox_inches='tight')
# plt.show()
print('done')

# ax1 = plt.subplot(312)
# plt.title('RMSE. Varying width and num sens per joint, Y coord')
for widthVar in range(len(widthList)):
    if widthVar == 12:
        ax5.plot(numSensPerJointList, rmseSensY[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax5.plot(numSensPerJointList, rmseSensY[widthVar, :], c=colors(widthVar))
# plt.xlabel('Number of sensory neurons per joint')
ax5.set_ylabel('RMSE, Y, mm')
plt.colorbar(sm, ax=ax5)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumY.png', bbox_inches='tight')
# plt.show()

# ax1 = plt.subplot(313)
# plt.title('RMSE. Varying c and num sens per joint, Z coord')
for widthVar in range(len(widthList)):
    if widthVar == 12:
        ax6.plot(numSensPerJointList, rmseSensZ[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax6.plot(numSensPerJointList, rmseSensZ[widthVar, :], c=colors(widthVar))
plt.xlabel('Number of sensory neurons per joint')
ax6.set_ylabel('RMSE, Z, mm')

plt.colorbar(sm, ax=ax6)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc=7)

plt.savefig('WidthNumUpd2.png', dpi=500)
bbox_inches='tight'
plt.show()

rmseSens = rmseSensX+rmseSensY+rmseSensZ


cc=np.linspace(8, 1, 35)
colors2 = plt.cm.jet(cc)
colors = plt.get_cmap('jet', 35)
fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(7, 8))
# ax1 = plt.subplot(311)
# plt.title('RMSE. Varying width and num sens per joint, X coord')
for widthVar in range(len(widthList)):
    if widthVar==12:
        ax4.plot(numSensPerJointList, slopeXstat[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax4.plot(numSensPerJointList, slopeXstat[widthVar, :], c=colors(widthVar))
ax4.axhline(y=1, color='r', linestyle='dashed', linewidth=2.5)
ax4.axvline(x=11, color='r', linestyle='dashed', linewidth=2.5)
# plt.xlabel('Number of sensory neurons per joint')
ax4.set_ylabel('Slope, X')

sm = matplotlib.cm.ScalarMappable(cmap=colors, norm=matplotlib.colors.Normalize(vmin=8, vmax=35))
sm.set_array([])
plt.colorbar(sm, ax=ax4)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumX.png', bbox_inches='tight')
# plt.show()
print('done')

# ax1 = plt.subplot(312)
# plt.title('RMSE. Varying width and num sens per joint, Y coord')
for widthVar in range(len(widthList)):
    if widthVar == 12:
        ax5.plot(numSensPerJointList, slopeYstat[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax5.plot(numSensPerJointList, slopeYstat[widthVar, :], c=colors(widthVar))
ax5.axhline(y=1, color='r', linestyle='dashed', linewidth=2.5)
ax5.axvline(x=11, color='r', linestyle='dashed', linewidth=2.5)
# plt.xlabel('Number of sensory neurons per joint')
ax5.set_ylabel('Slope, Y')
plt.colorbar(sm, ax=ax5)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('WidthnumY.png', bbox_inches='tight')
# plt.show()

# ax1 = plt.subplot(313)
# plt.title('RMSE. Varying c and num sens per joint, Z coord')
for widthVar in range(len(widthList)):
    if widthVar == 12:
        ax6.plot(numSensPerJointList, slopeZstat[widthVar, :], c='k', linewidth=4)#, colors=colors2[widthVar])
    else:
        ax6.plot(numSensPerJointList, slopeZstat[widthVar, :], c=colors(widthVar))
ax6.axhline(y=1, color='r', linestyle='dashed', linewidth=2.5)
ax6.axvline(x=11, color='r', linestyle='dashed', linewidth=2.5)
plt.xlabel('Number of sensory neurons per joint')
ax6.set_ylabel('Slope, Z')

plt.colorbar(sm, ax=ax6)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig2.legend(loc=7)

plt.savefig('WidthNumStat.png', dpi=500)
bbox_inches='tight'
plt.show()







'''

figBell, ax1 = plt.subplots()

ax1.set_xlabel('Time, ms')
ax1.set_ylabel('Sensory neural activation, mV', color='tab:red')
for index in range(len(jointTheta)):
    ax1.plot(t[0:1000], PredictedOutOne[0:1000, index])
ax1.tick_params(axis='y', labelcolor='tab:red')
ax2 = ax1.twinx()
ax2.set_ylabel('Joint angle [${\Theta}_{2}$], rad', color='k')
ax2.plot(t[0:1000], theta2vec[0:1000], color='k', label='Input', linewidth=4.0)
ax2.tick_params(axis='y', color='k')
# plt.savefig('Fig5.png', bbox_inches='tight', dpi=400)
figBell.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
'''
plt.figure()
for index in range(len(jointTheta)):
    plt.plot(t, PredictedOutOne[:, index])
plt.plot(t, theta2vec, label='Input')
plt.ylabel('Rads')
plt.xlabel('Time, ms')
plt.title('Theta 2 bell curves neuron responses')
plt.legend()
# # plt.savefig('Theta2respAll.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('Combined neuron2 and neuron3 responses')
plt.ylabel('Activation, mV')
plt.xlabel('Time, ms')
plt.plot(t, PredictedOutOne[:, numSensPerJoint*numJoints], label='Interneuron', linewidth=6, color='b')
plt.plot(t, PredictedOutOne[:, 0], label='Theta 2 = -1.6 rads', linewidth=2.5, color='tab:olive')
plt.plot(t, PredictedOutOne[:, numSensPerJoint], label='Theta 3 = -1.6 rads', linewidth=2.5, color='tab:orange')
plt.legend()
# plt.savefig('CombinedResp.png', bbox_inches='tight')
plt.show()
'''
figBell, ax1 = plt.subplots()

ax1.set_xlabel('Time, ms')
ax1.set_ylabel('Sensory neural activation, mV', color='b')
lns1=ax1.plot(t, PredictedOutOne[:, numSensPerJoint*numJoints], label='Interneuron', linewidth=6, color='b')
lns2=ax1.plot(t, PredictedOutOne[:, 0], label='Activation of the ${\Theta}_{2}$ at -1.6 rads', linewidth=2.5, color='tab:olive')
lns3=ax1.plot(t, PredictedOutOne[:, numSensPerJoint], label='Activation of the ${\Theta}_{3}$ at -1.6 rads', linewidth=2.5, color='tab:orange')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.set_ylabel('Joint angle [${\Theta}$], rad', color='k')
lns4=ax2.plot(t, theta2vec, '--', color='tab:olive', label='Input ${\Theta}_{2}$', linewidth=2.0)
lns5=ax2.plot(t, theta3vec, '--', color='orange', label='Input ${\Theta}_{3}$', linewidth=2.0)
ax2.tick_params(axis='y', color='k')
lns = lns1+lns2+lns3+lns4+lns5
labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc='center left', bbox_to_anchor=(1.15, 0.5))
plt.xlim([1300, 2000])
# plt.legend()

# figBell.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Theta2respAllUpd2.png', bbox_inches='tight', dpi=500)
# plt.show()

# widthListBell = np.arange(8, 35, 1)
# numSensPerJointListBell = np.arange(5, 17, 1)
'''
for i in range(len(widthListBell)):
    for j in range(len(numSensPerJointListBell)):
        jointThetaList = np.linspace(thetaMin, thetaMax, numSensPerJointListBell[j])
        colors = plt.cm.jet(np.linspace(0, 1, len(jointThetaList)))
        bellvec = np.arange(-2.5, 2.5, 0.05)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular' }
        plt.rcParams.update(params)
        ax.set_ylabel('Magnitude [$I_{ext}$(${\Theta}$)], nA')
        for k in range(len(jointThetaList)):
            ax.plot(bellvec, bell_curve(mag, widthListBell[i], theta=bellvec, shift=jointThetaList[k]), color=colors[k])
            ax.set_xlabel('Joint angle [${\Theta}$], rad')
        plt.title('nums= '+ str(numSensPerJointListBell[j])+'. width '+str(widthListBell[i]))
# plt.ylabel('Magnitude [Iext(${\Theta}$)], nA')
# plt.xlabel('Joint angle [$\dot{\Theta}$], rad')
# plt.legend()
# plt.savefig('BellRespUpd.png', dpi=300)
# plt.show()

'''
print("--- %s seconds ---" % (time.time() - startTime))

# mapping surfaces for joint angles

theta22, theta33 = np.meshgrid(jointTheta, jointTheta)

ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(theta22, theta33, ActualX, edgecolor='tomato', lw=0.5, rstride=11, cstride=11,
                alpha=0.5)


ax.contour(theta22, theta33, ActualX, zdir='z', offset=-120, cmap='autumn')
ax.contour(theta22, theta33, ActualX, zdir='x', offset=-1.7,cmap='autumn')

ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), zlim=(-120, 400),
       xlabel='Theta2, rad', ylabel='Theta3, rad', zlabel='X pos, mm')
# plt.savefig('mapXfk.svg', dpi=500)
plt.show()

ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(theta22, theta33, ActualY, edgecolor='yellowgreen', lw=0.5,rstride=11, cstride=11,
                alpha=0.5)

ax.contour(theta22, theta33, ActualY, zdir='z', offset=-400, cmap='summer')
ax.contour(theta22, theta33, ActualY, zdir='x', offset=-1.7,cmap='summer')
ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), zlim=(-400, 100),
       xlabel='Theta2, rad', ylabel='Theta3, rad', zlabel='Y pos, mm')
# plt.savefig('mapYfk.svg', dpi=500)
plt.show()

ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(theta22, theta33, ActualZ, edgecolor='royalblue', lw=0.5, rstride=11, cstride=11,
                alpha=0.5)

ax.contour(theta22, theta33, ActualZ, zdir='z', offset=-350, cmap='winter')
ax.contour(theta22, theta33, ActualZ, zdir='x', offset=-1.7,cmap='winter')
ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), zlim=(-350, 350),
       xlabel='Theta2, rad', ylabel='Theta3, rad', zlabel='Z pos, mm')
# plt.savefig('mapZfk.svg', dpi=500)
plt.show()



fig, ax = plt.subplots()
CS = ax.contour(jointTheta, jointTheta, ActualX, cmap='autumn')
# ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('X coord')
ax.set(xlabel='${\Theta}_{2}$, rad')
ax.set(ylabel='${\Theta}_{3}$, rad')
# plt.savefig('FKXcontour.svg', dpi=500)
plt.show()
fig, ax = plt.subplots()
CS = ax.contour(jointTheta, jointTheta, ActualY, cmap='summer')
# ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Y coord')
ax.set(xlabel='${\Theta}_{2}$, rad')
ax.set(ylabel='${\Theta}_{3}$, rad')
# plt.savefig('FKYcontour.svg', dpi=500)
plt.show()
fig, ax = plt.subplots()
CS = ax.contour(jointTheta, jointTheta, ActualZ, cmap='winter')
# ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Z coord')
ax.set(xlabel='${\Theta}_{2}$, rad')
ax.set(ylabel='${\Theta}_{3}$, rad')
# plt.savefig('FKZcontour.svg', dpi=500)
plt.show()


# Nick's request for the meeting
# widthListN = np.array([10, 20, 30])  # 20
widthListN = np.array([20])  # 20
numSensPerJointListN = np.array([6, 11, 15])  # 11
# numSensPerJointListN = np.array([11])  # 11
'''
plt.figure()
for widthVar in range(len(widthListN)):
    for k in range(len(numSensPerJointListN)):
        jointThetaVar = np.linspace(thetaMin, thetaMax, numSensPerJointListN[k])
        ActualXVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        ActualYVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        ActualZVar = np.zeros([len(jointThetaVar), len(jointThetaVar)])
        for i in range(len(jointThetaVar)):
            for j in range(len(jointThetaVar)):
                p1TrnslMat = FrwrdKnmtcsFunc(wR1, np.array([0, jointThetaVar[i], jointThetaVar[j]]), pR1)
                ActualXVar[i, j] = p1TrnslMat[0, 3]  # x coord
                ActualYVar[i, j] = p1TrnslMat[1, 3]  # y coord
                ActualZVar[i, j] = p1TrnslMat[2, 3]  # z coord
        PredictedOut, ActualT = frwrdKnmtcsSNS3D(delay=delayInit, numJoints=numJoints, numSensPerJoint=numSensPerJointListN[k],
                                                 numSteps=numSteps, widthBell=widthListN[widthVar],
                                                 ActualMapX=ActualXVar, ActualMapY=ActualYVar, ActualMapZ=ActualZVar,
                                                 servo2vec=theta2vec, servo3vec=theta3vec, jointTheta=jointThetaVar)
        predOutX = min(ActualT[:, 0]) + (max(ActualT[:, 0]) - min(ActualT[:, 0])) * PredictedOut[:, -3] / mag
        plt.plot(t, predOutX,  linewidth=1.75, label='Neurons Per Joint = '+str(numSensPerJointListN[k]))
plt.plot(t, ActualT[:, 0], label='Actual', color='k', linewidth=3)
plt.xlabel('T, ms')
plt.ylabel('X, mm')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
# plt.savefig('PredActForNickVarNumNeur.png', bbox_inches='tight', dpi=500)
# plt.show()'''
