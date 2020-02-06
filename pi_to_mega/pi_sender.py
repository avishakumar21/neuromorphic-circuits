# rpi_sender
# sends one byte over a 9600-baud serial link every two seconds

from __future__ import print_function
import serial
from time import sleep


port = "/dev/ttyACM2"  
baudrate = 9600
ser = serial.Serial( port, baudrate )


count = 0
while True:
    #ser.write( chr( count ) )
    ser.write( "count".encode() )
    count = ( count + 1) % 255
    print( "sent", count )
    sleep(2) # send data every two seconds
    