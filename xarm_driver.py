# import streamlit as st
# import serial
# import time

# import xarm

# # arm is the first xArm detected which is connected to USB
# arm = xarm.Controller('USB')
# print('Battery voltage in volts:', arm.getBatteryVoltage())
# import xarm

# arm = xarm.Controller('USB', debug=True)

# import xarm

# arm = xarm.Controller('USB')

# # define servo as servo ID 1 with position 300
# servo = xarm.Servo(1, 300)

# print('servo ID:', servo.servo_id)
# print('servo position:', servo.position)
# print('servo angle:', servo.angle)


# import xarm

# arm = xarm.Controller('USB')

# servo1 = xarm.Servo(1)       # assumes default unit position 500
# servo2 = xarm.Servo(2, 300)  # unit position 300
# servo3 = xarm.Servo(3, 90.0) # angle 90 degrees

# # sets servo 1 to unit position 300 and waits the default 1 second
# # before returning
# arm.setPosition(1, 300, wait=True)

# # sets servo 2 to unit position 700 and moves the servo at a
# # rate of 2 seconds
# arm.setPosition(2, 700, 2000, True)

# # sets servo1 to 45 degrees and waits the default 1 second
# # before returning
# arm.setPosition(3, 45.0, wait=True) 

# # sets servo 2 to position 300 as defined above but continues to
# # the next method before completing movement
# arm.setPosition(servo2) 

# # sets servos 1-3 as defined and continues without waiting
# arm.setPosition([servo1, servo3])

# # sets servos 1 to unit position 200 and servo 2 to 90 degrees
# arm.setPosition([[1, 200], [2, 90.0]], wait=True) 

# # Servo object and servo ID/position pairs can be combined
# arm.setPosition([servo1, [2, 500], [3, 0.0]], 2000)


# import xarm

# arm = xarm.Controller('USB')

# servo1 = xarm.Servo(1)
# servo2 = xarm.Servo(2)
# servo3 = xarm.Servo(3)

# # Gets the position of servo 1 in units
# position = arm.getPosition(1)
# print('Servo 1 position:', position)

# # Gets the position of servo 2 as defined above
# position = arm.getPosition(servo2)
# print('Servo 2 position:', position)

# # Gets the position of servo 3 in degrees
# position = arm.getPosition(3, True)
# print('Servo 3 position (degrees):', position)

# # Gets the position of servo 2 as defined above
# # It is not necessary to set the degreees parameter
# # because the Servo object performes that conversion
# position = arm.getPosition([servo1, servo2, servo3])
# print('Servo 1 position (degrees):', servo1.angle)
# print('Servo 2 position (degrees):', servo2.angle)
# print('Servo 3 position (degrees):', servo3.angle)



import tkinter as tk
import xarm
class GUI():
    def __init__(self):
        self.robot = self.connect_to_robot()
        if self.robot is None:
            print("Exiting application due to connection failure.")
            exit()
        print(f"{self.robot} connected Successfully")
        self.mw =tk.Tk()
        self.mw.geometry("400x400")
        self.mw.title("xArm Control Panel")
        #Labels
        self.label_gripper =tk.Label(self.mw, text = "Gripper")
        self.label_link2 =tk.Label(self.mw, text = "Link 2")
        self.label_link3 =tk.Label(self.mw, text = "Link 3")
        self.label_link4 =tk.Label(self.mw, text = "Link 4")
        self.label_link5 =tk.Label(self.mw, text = "Link 5")
        self.label_link6 =tk.Label(self.mw, text = "Link 6")

        #Slidebars = Scales
        self.scale_gripper = tk.Scale(self.mw, from_=0, to=1000, orient=tk.HORIZONTAL, command = self.send_val)
        self.scale_link2 = tk.Scale(self.mw, from_=0, to=1000, orient=tk.HORIZONTAL, command = self.send_val)
        self.scale_link3 = tk.Scale(self.mw, from_=0, to=1000, orient=tk.HORIZONTAL, command = self.send_val)
        self.scale_link4 = tk.Scale(self.mw, from_=0, to=1000, orient=tk.HORIZONTAL, command = self.send_val)
        self.scale_link5 = tk.Scale(self.mw, from_=0, to=1000, orient=tk.HORIZONTAL, command = self.send_val)
        self.scale_link6 = tk.Scale(self.mw, from_=0, to=1000, orient=tk.HORIZONTAL, command = self.send_val)

        self.scale_gripper.set(500)
        self.scale_link2.set(500)
        self.scale_link3.set(500)
        self.scale_link4.set(500)
        self.scale_link5.set(500)
        self.scale_link6.set(500)

        self.label_gripper.place(x=10,y=10)
        self.label_link2.place(x=10,y=60)
        self.label_link3.place(x=10,y=110)
        self.label_link4.place(x=10,y=160)
        self.label_link5.place(x=10,y=210)
        self.label_link6.place(x=10,y=260)

        self.scale_gripper.place(x=100,y=0)
        self.scale_link2.place(x=100,y=50)
        self.scale_link3.place(x=100,y=100)
        self.scale_link4.place(x=100,y=150)
        self.scale_link5.place(x=100,y=200)
        self.scale_link6.place(x=100,y=250)

        tk.mainloop()

    def send_val(self, event):
        gripper_val = self.scale_gripper.get()
        link2_val = self.scale_link2.get()
        link3_val = self.scale_link3.get()
        link4_val = self.scale_link4.get()
        link5_val = self.scale_link5.get()
        link6_val = self.scale_link6.get()

        self.robot.setPosition(1, gripper_val, wait=False)
        self.robot.setPosition(2, link2_val, wait=False)
        self.robot.setPosition(3, link3_val, wait=False)
        self.robot.setPosition(4, link4_val, wait=False)
        self.robot.setPosition(5, link5_val, wait=False)
        self.robot.setPosition(6, link6_val, wait=False)

        

    def connect_to_robot(self):
        try:
            robot = xarm.Controller('USB')
            print("Connected to xArm robot.")
            return robot
        except:
            print(f"Failed to connect to xArm robot")
            return None
        


if __name__ == "__main__":
    gui = GUI()