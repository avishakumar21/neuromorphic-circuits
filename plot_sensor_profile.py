import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt



def read_data(file_name):
    data = np.genfromtxt(file_name, delimiter = ' ', skip_header=2 ,skip_footer=1)
    return data

data = read_data('/home/pi/Documents/Avisha_project/data.txt')

#want to plot all 16 channels vs time

# xaxis = time
# yaxis = voltage

y0 = data[:,0]
y1 = data[:,1]
y2 = data[:,2]
y3 = data[:,3]
y4 = data[:,4]
y5 = data[:,5]
y6 = data[:,6]
y7 = data[:,7]
y8 = data[:,8]
y9 = data[:,9]
y10 = data[:,10]
y11 = data[:,11]
y12 = data[:,12]
y13 = data[:,13]
y14 = data[:,14]
y15 = data[:,15]

max_array = [max(y0), max(y1), max(y2), max(y3), max(y4), max(y5), max(y6), max(y7), max(y8), max(y9), max(y10), max(y11), max(y12), max(y13), max(y14), max(y15)]
print(max_array)  # return this into run_parallel


val = len(y0)
x = range(0,val)
#print(len(x))

fig = plt.figure()

ax1 = fig.add_subplot(111)


ax1.set_title("Sensor Profile")    
ax1.set_xlabel('timestamp')
ax1.set_ylabel('sensor voltage')
ax1.plot(x,y0,x,y1,x,y2,x,y3,x,y4,x,y5,x,y6,x,y7,x,y8,x,y9,x,y10,x,y11,x,y12,x,y13,x,y14,x,y15, label='the data')



plt.show()
