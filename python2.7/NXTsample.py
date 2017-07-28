import nxt
from nxt.motor import *
import time
import bluetooth
def playing_around():
    #nxt.locator.make_config()
    brick = nxt.find_one_brick(None, None, False, False, True)
    # brick.mc.start()
    motor = nxt.Motor(brick, PORT_A)
    motorB = nxt.Motor(brick, PORT_B)
    print("before: " + str(motor.get_tacho()))

    # brick.
    # brick.mc.move_to(PORT_A, 100,30000)
    # brick.mc.move_to(PORT_B, 100,50)
    print("after: " + str(motor.get_tacho()))

    motorB.run(70)
    time.sleep(1)
    motorB.run(-127)
    time.sleep(0.2)
    motorB.run(0)

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
    print("after: " + str(motor.get_tacho()))

def rotateFoot():
    brick = nxt.find_one_brick(None, None, False, False, True)
    motorB = nxt.Motor(brick, PORT_B)
    motorB.run(-127)
    time.sleep(2)
    motorB.run(0)
rotateFoot()