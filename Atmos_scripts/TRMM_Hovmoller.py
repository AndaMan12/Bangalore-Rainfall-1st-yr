# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:04:24 2022

@author: guria
"""

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

for date in dates:
    search_path='3B42_Daily.'+date+'.7.nc4'
    if search_path in valid_paths:
        ds=xr.open_dataset(r"C:\Users\guria\trmm"+"\\"+search_path,engine="netcdf4")
        rain=ds['precipitation'].values        
        daily_data.append(rain[Bangalore_lon_index[0]:Bangalore_lon_index[1]+1,Bangalore_lat_index[0]:Bangalore_lat_index[1]+1])
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
daily_recon=np.zeros(shape=(len(daily_data),13,13))
x_1=np.array([i for i in range(len(daily_data))])
for i in range(13):
    for j in range(13):
        f=interpolate.interp1d(x,y[:,i,j])        
        daily_recon[:,i,j]=f(x_1)

period=365
time=[i+1 for i in range(period)]
time=np.array(time)
lonx=[lon(i) for i in range(Bangalore_lon_index[0],Bangalore_lon_index[1]+1)]
laty=[lat(i) for i in range(Bangalore_lat_index[0],Bangalore_lat_index[1]+1)]

time=[i+1 for i in range(75)]
X,Tx=np.meshgrid(lonx,time)
Y,Ty=np.meshgrid(laty,time)

def data_com(data):
    S=0
    Sx=0
    for i in range(data.shape[0]):
        S+=data[i]
        Sx+=i*data[i]
    com=(Sx/S)
    return(com)
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

spacing=1
cycles=daily_recon.shape[0]//period

periodic_sliced=[]
periodic_sliced_ma=[]

for i in range(cycles):
    periodic_sliced.append(daily_recon[i*period+100:(i)*period+175,:,:])
periodic_sliced=np.array(periodic_sliced)
mean_dist=np.mean(periodic_sliced,axis=0)

mean_EW=np.mean(mean_dist,axis=2)
mean_NS=np.mean(mean_dist,axis=1)
com_EW=np.array([0.25*(data_com(mean_EW[i,:]))+lonx[0] for i in range(mean_EW.shape[0])])
com_NS=np.array([0.25*(data_com(mean_NS[i,:]))+laty[0] for i in range(mean_NS.shape[0])])
figEW, axEW = plt.subplots(1, 1)
figNS, axNS = plt.subplots(1, 1)
axEW.contourf(Tx,X,signal.fftconvolve(mean_EW,np.ones(shape=(20,)+mean_EW.shape[1:]),"valid")/20,100,cmap=plt.cm.twilight)
axNS.contourf(Ty,Y,signal.fftconvolve(mean_NS,np.ones(shape=(20,)+mean_NS.shape[1:]),"valid")/20,100,cmap=plt.cm.twilight)
axEW.set_title("East_West propagation")
axNS.set_title("North_South propagation")
plt.show()




























