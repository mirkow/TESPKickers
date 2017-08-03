import nxt
from nxt.motor import *
import time
import numpy as np
import circle_detector

import sys
from PyQt4.QtCore import pyqtSlot
import PyQt4.QtCore
from PyQt4.QtGui import *

import PyQt4.QtGui

brick = nxt.find_one_brick(None, "NXT", False, False, True)
motorA = nxt.Motor(brick, PORT_A)
motorB = nxt.Motor(brick, PORT_B)
motorC = nxt.Motor(brick, PORT_C)

detector = circle_detector.circle_detector()

@pyqtSlot()
def printRotationCountMotorA():
    print("rotation motor tachocount: " + str(motorC.get_tacho()))

@pyqtSlot()
def resetMotorATacho():
    motorA.reset_position(False)
    time.sleep(0.5)
    printRotationCountMotorA()

time.sleep(1)
printRotationCountMotorA()
# time.sleep(0.5)
# printRotationCountMotorA()


@pyqtSlot()
def on_click_lift():
    print("before: " + str(motorB.get_tacho()))
    # brick.mc.move_to(PORT_B, 80, 10,1,1,1)
    # time.sleep(2)
    print("after: " + str(motorB.get_tacho()))

    motorB.run(-60, True)
    time.sleep(0.5)
    motorB.run(-127, True)
    # motorB.run(-127)
    time.sleep(1)
    motorB.run(-30)
    # time.sleep(0.1)
    # motorB.brake()





# Create an PyQT4 application object.
a = PyQt4.QtGui.QApplication(sys.argv)



# The QWidget widget is the base class of all user interface objects in PyQt4.
w = PyQt4.QtGui.QWidget()

btnTracking = PyQt4.QtGui.QPushButton('Start Tracking', w)

timer = PyQt4.QtCore.QTimer(w)
@pyqtSlot()
def track_and_command():
    # print "detecting..."
    (x_pred, pos) = detector.detect()

    if btnTracking.isChecked() and pos is not None and pos[2] > -1.8:
        # on_click_move_right()
        command = x_pred * -100 / 0.07
        command = np.clip(command, -400,300)
        print "xpred: " + str(x_pred)
        print "ball pos" + str(pos)
        brick.mc.move_to(PORT_B, 50, 120,1,1,1)
        on_click_move_robot(None, command)
        time.sleep(0.4)
        pass

timer.timeout.connect(track_and_command)
timer.start(30)

# Set window size.
w.resize(320, 240)

# Set window title
w.setWindowTitle("TESP Goalie")
layout = PyQt4.QtGui.QGridLayout(w)
# Add a button

angleEdit = PyQt4.QtGui.QSpinBox(w)
angleEdit.setMinimum(-10000)
angleEdit.setMaximum(10000)
angleEdit.setValue(00)
layout.addWidget(angleEdit)


btnTracking.setCheckable(True)

@pyqtSlot()
def toggleTracking(toggled):
    if toggled:
        print "starting goalie behavior"
        # timer.start(30)
    else:
        print "stopping goalie behavior"
        # timer.stop()

btnTracking.toggled.connect(toggleTracking)
layout.addWidget(btnTracking)

# btnLift = PyQt4.QtGui.QPushButton('Lift', w)
# btnLift.clicked.connect(on_click_lift)
# layout.addWidget(btnLift)


@pyqtSlot()
def on_click_move_robot(value = None, command = None):

    #nxt.locator.make_config()
    # brick.mc.start()
    # print("before: " + str(motorB.get_tacho()))
    #
    # # brick.
    # # brick.mc.move_to(PORT_A, 100,30000)
    # # brick.mc.move_to(PORT_B, 100,50)
    # print("after: " + str(motorB.get_tacho()))
    if command is None:
        command = angleEdit.value()
    print "moving " + str(command)
    brick.mc.move_to(PORT_C, 50, command)
    # motorC.run(127)
    # time.sleep(0.35)
    # motorC.run(0)

    # return
    # brick.
    # brick.play_tone_and_wait(100,1000)
    # brick.mc.start()
    # time.sleep(1)
    # motor.run(100)
    # # motor.turn(127, 10000, False)
    # for i in range(0,10):
    #     time.sleep(1)
    #     print(motor.get_tacho())
    # motor.brake()
    # for i in range(5,8):
    #
    #     time.sleep(0.4)
    #     print(motor.get_tacho())
    # motor.run(0)

    # brick.mc.cmd(PORT_A, 10, 100)
    # for i in range(0,10):
    #     # time.sleep(0.1)
    #     print(motor.get_tacho())
    # motor.brake()
    print("after: " + str(motorA.get_tacho()))


btnKick = PyQt4.QtGui.QPushButton('Move right', w)
btnKick.clicked.connect(on_click_move_robot)
layout.addWidget(btnKick)




btnRotate = PyQt4.QtGui.QPushButton('Rotate', w)
layout.addWidget(btnRotate)


def on_click_rotate_foot():
    print("before: " + str(motorB.get_tacho()))
    brick.mc.move_to(PORT_B, 20, angleEdit.value(),1,1,1)
    time.sleep(2)

    # brick.
    # brick.mc.move_to(PORT_A, 100,30000)
    # brick.mc.move_to(PORT_B, 100,50)
    print("after: " + str(motorB.get_tacho()))
btnRotate.clicked.connect(on_click_rotate_foot)


btnRotationInfo = PyQt4.QtGui.QPushButton('Show rotation motor tacho count', w)
layout.addWidget(btnRotationInfo)
btnRotationInfo.clicked.connect(printRotationCountMotorA)





# Show window
w.show()

sys.exit(a.exec_())

#rotateFoot()
# playing_around()