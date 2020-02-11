# code to display sensor voltage values over time

import serial
import time



timestamps = 50  #use DEFINE here 

ser = serial.Serial("/dev/ttyACM0", 115200, timeout=10); # Establish the connection on a specific port

rawdata = []
count = 0


while count < timestamps: # change to 200 timestamps to get full profile
    #print(str(ser.readline()))
    rawdata.append(str(ser.readline()))
    count += 1
    
#print(rawdata)

def clean(L):
    newl = []
    for i in range(len(L)):
        temp= L[i][2:]
        newl.append(temp[:-5])
    return newl

cleandata = clean(rawdata)

def write(L):
    file = open("data.txt", mode = 'w')
    for i in range(len(L)):
        file.write(L[i] + '\n')
    file.close()

write(cleandata)


