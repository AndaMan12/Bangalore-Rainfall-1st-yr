# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:26:16 2022

@author: guria
"""
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.interpolate as interpolate
import datetime
import pandas as pd



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


    
#plt.plot(np.sum(daily_recon,axis=(1,2)))
rain_data=np.sum(daily_recon,axis=(1,2))

x=[lon(i) for i in range(Bangalore_lon_index[0],Bangalore_lon_index[1]+1)]
y=[lat(i) for i in range(Bangalore_lat_index[0],Bangalore_lat_index[1]+1)]
X,Y=np.meshgrid(x,y)

print("done here")




fps = 10
frn = daily_recon.shape[0]


zarray = daily_recon

def t_partial (data): 
    #function for calculating time derivative of a date-labbeled scalar field
    data_partial_t=[]
    data_partial_t.append(data[1,:,:]-data[0,:,:])
    for t in range (1,frn-1):
        data_partial_t.append(data[t+1,:,:]-data[t-1,:,:])
    data_partial_t.append(data[frn-1,:,:]-data[frn-2,:,:])
    return(np.array(data_partial_t))

rain_del_t=t_partial(daily_recon)# Evaluating time derivative

grad_rain_del_t=np.gradient(rain_del_t, axis=(1,2))


def change_plot(frame_number, grad_rain_del_t):
   ax.clear()
   plt.quiver(X,Y,grad_rain_del_t[0][frame_number]/(np.mean(np.abs(grad_rain_del_t[0][frame_number]))+1),grad_rain_del_t[1][frame_number]/(np.mean(np.abs(grad_rain_del_t[1][frame_number]))+1))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(grad_rain_del_t, ), interval=1000 / fps)



plt.show()