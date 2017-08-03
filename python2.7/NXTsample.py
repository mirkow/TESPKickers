import nxt
from nxt.motor import *
import time


import sys
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import *
import PyQt4.QtGui

brick = nxt.find_one_brick(None, "NXT-4", False, False, True)
motorA = nxt.Motor(brick, PORT_A)
motorB = nxt.Motor(brick, PORT_B)
motorC = nxt.Motor(brick, PORT_C)



@pyqtSlot()
def printRotationCountMotorA():
    print("rotation motor tachocount: " + str(motorA.get_tacho()))

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
    print "lifting"
    print("before: " + str(motorB.get_tacho()))
    # brick.mc.move_to(PORT_B, 80, 10,1,1,1)
    # time.sleep(2)
    print("after: " + str(motorB.get_tacho()))

    motorB.run(-80, False)
    motorC.run(-80, False)
    time.sleep(0.5)
    # motorB.run(-127, True)
    # motorC.run(-127, True)
    # motorB.run(-127)
    # time.sleep(1)
    motorB.run(-50)
    motorC.run(-50)
    # time.sleep(0.1)
    # motorB.brake()


@pyqtSlot()
def on_click_kick():
    print "kicking"
    #nxt.locator.make_config()
    # brick.mc.start()
    # print("before: " + str(motorB.get_tacho()))
    #
    # # brick.
    # # brick.mc.move_to(PORT_A, 100,30000)
    # # brick.mc.move_to(PORT_B, 100,50)
    # print("after: " + str(motorB.get_tacho()))
    power = 90

    motorB.run(power)
    motorC.run(power)
    time.sleep(0.35)
    motorB.run(0)
    motorC.run(0)

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

# Create an PyQT4 application object.
a = PyQt4.QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = PyQt4.QtGui.QWidget()

# Set window size.
w.resize(320, 240)

# Set window title
w.setWindowTitle("TESP Kicker")
layout = PyQt4.QtGui.QGridLayout(w)
# Add a button

btnReset = PyQt4.QtGui.QPushButton('Reset Rotation Motor', w)
btnReset.clicked.connect(resetMotorATacho)
layout.addWidget(btnReset)

btnLift = PyQt4.QtGui.QPushButton('Lift', w)
btnLift.clicked.connect(on_click_lift)
layout.addWidget(btnLift)


btnKick = PyQt4.QtGui.QPushButton('Kick', w)
btnKick.clicked.connect(on_click_kick)
layout.addWidget(btnKick)



angleEdit = PyQt4.QtGui.QSpinBox(w)
angleEdit.setMinimum(-10000)
angleEdit.setMaximum(10000)
angleEdit.setValue(0)
layout.addWidget(angleEdit)
btnRotate = PyQt4.QtGui.QPushButton('Rotate', w)
layout.addWidget(btnRotate)


def on_click_rotate_foot():
    print("before: " + str(motorA.get_tacho()))
    brick.mc.move_to(PORT_A, 20, angleEdit.value(),1,1,1)
    time.sleep(2)

    # brick.
    # brick.mc.move_to(PORT_A, 100,30000)
    # brick.mc.move_to(PORT_B, 100,50)
    print("after: " + str(motorA.get_tacho()))
btnRotate.clicked.connect(on_click_rotate_foot)


btnRotationInfo = PyQt4.QtGui.QPushButton('Show rotation motor tacho count', w)
layout.addWidget(btnRotationInfo)
btnRotationInfo.clicked.connect(printRotationCountMotorA)





# Show window
w.show()

sys.exit(a.exec_())

#rotateFoot()
# playing_around()