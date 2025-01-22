# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 01:15:06 2022

@author: guria
"""

#THIS CODE WILL WORK JUST FINE FOR THE HALF HOURLY DATA. 
#WE WOULD NEED TO REMOVE THE NOISE IN THE HIGH FEEQUENCY DOMAIN 
#AND EVALUATE THE COUNT OF HIGH RAIN EVENTS APART FROM THE TOTAL VOLUME OF RAIN.
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import scipy.interpolate as interpolate
import datetime
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft


diri=r"C:\Users\guria\trmm"
valid_paths = next(os.walk(diri), (None, None, []))[2]


index_lat= lambda lat: int((lat+49.875)/0.25)
index_lon= lambda lon: int((lon+179.875)/0.25)
lat= lambda index_lat: -49.875+index_lat*0.25
lon= lambda index_lon: -179.875+index_lon*0.25
Bangalore_lat_index=[index_lat(11.375),index_lat(14.375)]
Bangalore_lon_index=[index_lon(76.375),index_lon(79.375)]


start = datetime.datetime.strptime("19980101", "%Y%m%d")
end = datetime.datetime.strptime("20191230", "%Y%m%d")
date_generated = pd.date_range(start, end)
dates=date_generated.strftime("%Y%m%d")


daily_data=[]
data_shape=[]
for date in dates:
    search_path='3B42_Daily.'+date+'.7.nc4'
    if search_path in valid_paths:
        ds=xr.open_dataset(r"C:\Users\guria\trmm"+"\\"+search_path,engine="netcdf4")
        rain=ds['precipitation'].values        
        daily_data.append(rain[Bangalore_lon_index[0]:Bangalore_lon_index[1]+1,Bangalore_lat_index[0]:Bangalore_lat_index[1]+1])
        data_shape=daily_data[-1].shape
    else:
        daily_data.append("dummy")  
missed=[i for i,x in enumerate(daily_data) if x=="dummy"]
x=[i for i in range(len(daily_data)) if i not in missed]
y=list(daily_data)

try:
    while True:
        y.remove("dummy")
except ValueError:
    pass
y=np.array(y)
daily_recon=np.zeros(shape=(len(daily_data),)+data_shape)
x_1=np.array([i for i in range(len(daily_data))])
for i in range(data_shape[0]):
    for j in range(data_shape[1]):
        f=interpolate.interp1d(x,y[:,i,j])        
        daily_recon[:,i,j]=f(x_1)


    
#plt.plot(np.sum(daily_recon,axis=(1,2)))
rain_data=np.sum(daily_recon,axis=(1,2))

x=[lon(i) for i in range(Bangalore_lon_index[0],Bangalore_lon_index[1]+1)]
y=[lat(i) for i in range(Bangalore_lat_index[0],Bangalore_lat_index[1]+1)]
X,Y=np.meshgrid(x,y)

print("done here")
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

period=365
spacing=1
cycles=rain_data.shape[0]//period

periodic_sliced=[]
periodic_sliced_ma=[]
Y=[]
for i in range(cycles):
    periodic_sliced.append(rain_data[i*period:(i+1)*period])
    periodic_sliced_ma.append(moving_average(rain_data[i*period:(i+1)*period],20))
    Y.append((2/period)*np.abs(fft.fft(periodic_sliced[-1])[:period//2]))
Y_av=sum(Y)/cycles

rain_av=sum(periodic_sliced)/cycles
periodic_sliced=np.array(periodic_sliced)
periodic_sliced_ma=np.array(periodic_sliced_ma)
std_dev=np.std(periodic_sliced,axis=1)
std_dev_ma=np.std(periodic_sliced_ma,axis=1)
rain_av_ma=moving_average(rain_av, 20)

'''
yf=(2/200)*np.abs(fft.fft(rain_av_ma[100:300])[:200//2])
xf=fft.fftfreq(200,spacing)[:200//2]

plt.plot(xf,yf)
peaks,_=signal.find_peaks(yf)
peak_freq=[xf[i] for i in peaks]
print(peak_freq)
print([1/f for f in peak_freq])
'''
plt.plot(rain_av_ma)














