# -*- coding: utf-8 -*-
"""
Created on January 2020

@author: jordi; jordi.fonollosa.m@upc.edu
"""

import time
from datetime import datetime
import serial
import os



os.chdir('/home/pi/Documents/test_v3')


## Boolean variable that will represent 
## whether or not the arduino is connected
connected = False

## open the serial port that your ardiono 
## is connected to.
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=0.5); # Establish the connection on a specific port


print(ser.readline())
time.sleep(.1)
print(ser.readline())
time.sleep(.1)


tini=time.strftime("%d-%m-%Y // %H:%M:%S")
print ("###"+tini)

fname=time.strftime("%d%m%Y")

with open(fname, 'a') as fa:
    fa.write("###"+tini+'\n')

i=0
while True: #start signal acquisition. This is an infinte loop. 

    tini=time.strftime("%d-%m-%Y // %H:%M:%S")
    tini=datetime.now().strftime("%d-%m-%Y // %H:%M:%S.%f")
    ser.write("R\n".encode())  #request data
    #print(ser.readline()) #uncomment this to print data on the terminal
    with open(fname, 'a') as fa:
        #x = ser.readline()
        fa.write(str(ser.readline()[0:-2]) + ',' + tini + '\n') #read incoming data and write it to the txt file
    i=i+1
    print(i) #print control variable
    time.sleep(0.25) #wait before requesting next sensor data
    
    


## close the port and end the program
ser.close()




