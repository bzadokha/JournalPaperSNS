# imported from Matlab
# forward kinematics & jacobian calculations using dynamixel sdk
# trossen robotics hexapod IV


import os
import dynamixel_sdk
from dynamixel_sdk import *
import numpy as np
import sys, tty, termios
# if os.name == 'nt':
#     import msvcrt
#     def getch():
#         return msvcrt.getch().decode()
# else:
#
# fd = sys.stdin.fileno()
# old_settings = termios.tcgetattr(fd)
# def getch():
#     try:
#         tty.setraw(sys.stdin.fileno())
#         ch = sys.stdin.read(1)
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#     return ch


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
# L1 leg
w2L1 = np.array([[-1/np.sqrt(2)], [1/np.sqrt(2)], [0]])
wL1 = np.concatenate((np.concatenate((w1, w2L1), axis=1), w2L1), axis=1)
# R2 leg
w2R2 = np.array([[1/np.sqrt(2)], [0], [0]])
wR2 = np.concatenate((np.concatenate((w1, w2R2), axis=1), w2R2), axis=1)
# L2 leg
wL2 = np.concatenate((np.concatenate((w1, -w2R2), axis=1), -w2R2), axis=1)
# R3 leg
wR3 = np.concatenate((np.concatenate((w1, -w2L1), axis=1), -w2L1), axis=1)
# L3 leg
wL3 = np.concatenate((np.concatenate((w1, -w2R1), axis=1), -w2R1), axis=1)

# zero position config
p0 = np.array([[0], [0], [0]])
# R1 leg
p1R1 = np.array([[x0], [-y01], [0]])
p2R1 = np.array([[x0+L1*np.cos(45*np.pi/180)], [-y02-L1*np.cos(45*np.pi/180)], [0]])  # mm
p3R1 = np.array([[x0+(L1+L2)*np.cos(45*np.pi/180)], [-y02-(L1+L2)*np.cos(45*np.pi/180)], [-15]])  # mm
p4R1 = np.array([[x0+(L1+L2+L3)*np.cos(45*np.pi/180)], [-y02-(L1+L2+L3)*np.cos(45*np.pi/180)], [-15-90]])  # mm
pR1 = np.concatenate((np.concatenate((np.concatenate((p1R1, p2R1), axis=1), p3R1), axis=1), p4R1), axis=1)
# R2 leg
p1R2 = np.array([[0], [-y02], [0]])
p2R2 = np.array([[0], [-y02-L1], [0]])  # mm
p3R2 = np.array([[0], [-y02-(L1+L2)], [-15]])  # mm
p4R2 = np.array([[0], [-y02-(L1+L2+L3)], [-15-90]])  # mm
pR2 = np.concatenate((np.concatenate((np.concatenate((p1R2, p2R2), axis=1), p3R2), axis=1), p4R2), axis=1)
# R3 leg
p1R3 = np.array([[-x0], [-y03], [0]])
p2R3 = np.array([[-x0-L1*np.cos(45*np.pi/180)], [-y02-L1*np.cos(45*np.pi/180)], [0]])  # mm
p3R3 = np.array([[-x0-(L1+L2)*np.cos(45*np.pi/180)], [-y02-(L1+L2)*np.cos(45*np.pi/180)], [-15]])  # mm
p4R3 = np.array([[-x0-(L1+L2+L3)*np.cos(45*np.pi/180)], [-y02-(L1+L2+L3)*np.cos(45*np.pi/180)], [-15-90]])  # mm
pR3 = np.concatenate((np.concatenate((np.concatenate((p1R3, p2R3), axis=1), p3R3), axis=1), p4R3), axis=1)
# L1 leg
p1L1 = np.array([[x0], [y01], [0]])
p2L1 = np.array([[x0+L1*np.cos(45*np.pi/180)], [y02+L1*np.cos(45*np.pi/180)], [0]])  # mm
p3L1 = np.array([[x0+(L1+L2)*np.cos(45*np.pi/180)], [y02+(L1+L2)*np.cos(45*np.pi/180)], [-15]])  # mm
p4L1 = np.array([[x0+(L1+L2+L3)*np.cos(45*np.pi/180)], [y02+(L1+L2+L3)*np.cos(45*np.pi/180)], [-15-90]])  # mm
pL1 = np.concatenate((np.concatenate((np.concatenate((p1L1, p2L1), axis=1), p3L1), axis=1), p4L1), axis=1)
# L2 leg
p1L2 = np.array([[0], [y02], [0]])
p2L2 = np.array([[0], [y02+L1], [0]])  # mm
p3L2 = np.array([[0], [y02+(L1+L2)], [-15]])  # mm
p4L2 = np.array([[0], [y02+(L1+L2+L3)], [-15-90]])  # mm
pL2 = np.concatenate((np.concatenate((np.concatenate((p1L2, p2L2), axis=1), p3L2), axis=1), p4L2), axis=1)
# L3 leg
p1L3 = np.array([[-x0], [y03], [0]])
p2L3 = np.array([[-x0-L1*np.cos(45*np.pi/180)], [y02+L1*np.cos(45*np.pi/180)], [0]])  # mm
p3L3 = np.array([[-x0-(L1+L2)*np.cos(45*np.pi/180)], [y02+(L1+L2)*np.cos(45*np.pi/180)], [-15]])  # mm
p4L3 = np.array([[-x0-(L1+L2+L3)*np.cos(45*np.pi/180)], [y02+(L1+L2+L3)*np.cos(45*np.pi/180)], [-15-90]])  # mm
pL3 = np.concatenate((np.concatenate((np.concatenate((p1L3, p2L3), axis=1), p3L3), axis=1), p4L3), axis=1)

# impedance settings
kEn = 1  # Nmm/rad environmental stiffness
kpS = 1e3  # Nmm/rad servo's Kp
Kleg = np.diag([kEn, kEn, kEn])  # what we want to have
KpMatrix = np.diag([kpS, kpS, kpS])  # servo
KpInv = np.linalg.inv(KpMatrix)
portName = 'COM5'
servoID = np.array([10, 11, 12])

def initialization430w250t(ID, portName, threshold, KpMat):
    IDsize = np.size(servoID)

    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    LEN_GOAL_POSITION = 4  # Data Byte Length
    ADDR_PRESENT_POSITION = 132
    LEN_PRESENT_POSITION = 4  # Data Byte Length
    ADDR_P_GAIN = 84
    BAUDRATE = 1e6
    protocol = 2.0
    DEVICENAME = portName

    TORQUE_ENABLE = 1  # Value for enabling the torque
    DXL_MOVING_STATUS_THRESHOLD = threshold  # Dynamixel moving status threshold
    dxl_p_gain = KpMat[0,0]/100
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(protocol)
    groupSyncWrite = GroupSyncWrite(portHandler, PacketHandler(protocol), ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    groupSyncWriteKpGain = GroupSyncWrite(portHandler, PacketHandler(protocol), ADDR_P_GAIN, LEN_GOAL_POSITION)
    groupSyncRead = GroupSyncRead(portHandler, PacketHandler(protocol), ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    # Open port
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        # getch()
        quit()
    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        # getch()
        quit()
    for index in np.arange(0, IDsize, 1):
        enableTorque(portHandler, ID[index], ADDR_TORQUE_ENABLE, TORQUE_ENABLE, COMM_SUCCESS, packetHandler)
    # Add parameter storage for Dynamixel#1 present position value
    for index in np.arange(0, IDsize, 1):
        dxl_addparam_result = groupSyncRead.addParam(ID[index])
        if dxl_addparam_result != True:
            print(str(ID[index]) + " groupSyncRead addparam failed")
            quit()
    return portHandler, protocol, groupSyncRead, dxl_p_gain, groupSyncWrite, groupSyncWriteKpGain, \
        ADDR_PRESENT_POSITION, LEN_GOAL_POSITION, ADDR_TORQUE_ENABLE, \
        LEN_PRESENT_POSITION, COMM_SUCCESS
def enableTorque(port, ID, torqueAddress, torqueEnable, commSuccess, packet):
    dxl_comm_result, dxl_error = packet.write1ByteTxRx(port, ID, torqueAddress, torqueEnable)
    if dxl_comm_result != commSuccess:
        return print(str(packet.getTxRxResult(dxl_comm_result)))
    elif dxl_error != 0:
        return print(str(packet.getRxPacketError(dxl_error)))
    else:
        return print("Dynamixel " + str(ID) + "has been successfully connected")
def RodriguesFunc(w, theta, p):
    Scross = np.cross(-w, p)
    Smatrix = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]  # Skew-symmetric
    Rmatrix = np.eye(3, dtype=float)+np.dot(Smatrix, np.sin(theta))+np.dot(np.matmul(Smatrix, Smatrix), (1-np.cos(theta)))
    wrenchSquare = np.matmul(w, w.reshape((3, 1)))
    TranslVec = np.matmul(np.matmul(np.eye(3, dtype=float)-Rmatrix, Smatrix), Scross)+np.dot(theta, np.dot(wrenchSquare.item(), Scross))
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
    M = np.concatenate((np.concatenate((np.eye(3, dtype=float), np.atleast_2d(p[:, -1]).T), axis=1),
                        np.array([[0, 0, 0, 1]])), axis=0)
    TranslMat = np.matmul(expoOne, M)
    return TranslMat
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
def positionToRads(servoPos):
    if servoPos > 2047 or servoPos < 2047:
        posDeg = (servoPos-2047)*0.088
        rad = posDeg*np.pi/180
    else:
        rad = 0
    return rad
def radsToPosition(rad):
    posDeg = rad/np.pi*180
    posServo = np.minimum(4095, (2047+np.floor(posDeg/0.088)))
    return posServo
def disableTorque(portHandler, ID, ADDR_TORQUE_ENABLE, packet):
    TORQUE_DISABLE = 0
    IDsize = np.size(servoID)
    for index in np.arange(0, IDsize, 1):
        dxl_comm_result, dxl_error = packet.write1ByteTxRx(portHandler, ID[index], ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            return print("%s" % packet.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            return print("%s" % packet.getRxPacketError(dxl_error))


servo1 = 2047
servo2 = 2055
servo3 = 2065

theta = np.array([0, positionToRads(servo2), positionToRads(servo3)])

endPointTrnslMat = FrwrdKnmtcsFunc(wR1, theta, pR1)
# print(endPointTrnslMat)
p1TrnslMat = FrwrdKnmtcsFunc(w1, theta[0], pR1[:, 0:2])
print(p1TrnslMat)
print(pR1)
p2TrnslMat = FrwrdKnmtcsFunc(np.concatenate((w1, w2R1), axis=1), [theta[0], theta[1]], pR1[:, 0:3])
print(p2TrnslMat)

xyzPresentP1 = p1TrnslMat[0:3, 3]
xyzPresentP2 = p2TrnslMat[0:3, 3]
xyzEndPoint = endPointTrnslMat[0:3, 3]
# rotation mat
R1s1 = p1TrnslMat[0:3, 0:3]
R1s2 = p2TrnslMat[0:3, 0:3]
# jacobian
jacobian = JacobianFunc(wR1, pR1, R1s1, R1s2, xyzPresentP1, xyzPresentP2)

print(jacobian)

'''
deltaXYZ = xyzEndPoint - xyzStartPos
f = np.matmul(-Kleg, deltaXYZ)
wrench = np.concatenate((f, np.cross(xyzEndPoint, f)), axis=0)
deltaTheta = np.matmul(np.matmul(KpInv, jacobian.T), wrench)
thetaCommand = deltaTheta + actualPosServoTheta
servoGoalPos = radsToPosition(thetaCommand)
print(jacobian)
'''

# Close port
# dynamixel_sdk.PortHandler.closePort(portName)