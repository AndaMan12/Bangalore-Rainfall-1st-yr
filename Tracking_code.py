#fetching data
events_tracking=dict()
def weight(data,pt_id):
    wt=0
    try:
        for i in range(-1,2):
            for j in range(-1,2):
                try:
                    wt+=data[pt_id[0]+i,pt_id[1]+j]
                except:
                    wt=data[pt_id[0],pt_id[1]]
                    break
        return(wt)
    except:
        return(0)

for i in range(2008,2009):
    yr=str(i)
    print(yr,"...",end='')
    diri="/home/anindya/GPM_mar_may_halfhr/"+yr+"_half_hr"
    valid_paths = next(os.walk(diri), (None, None, []))[2]
    ds=nc.Dataset(diri+"/"+valid_paths[0])
    latitude=np.array(ds['lat'][0])
    longitude=np.array(ds['lon'][0])
    def index_lat(lat):
        
        return( round((lat-latitude)/0.1))
    def index_lon(lon):
        
        return( round((lon-longitude)/0.1))

    Bangalore_lat_index=[index_lat(12.05),index_lat(14.05)]
    Bangalore_lon_index=[index_lon(76.55),index_lon(78.55)]
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def Filter (data,cutoff,order,mode="low"):
        sos = signal.butter(order,cutoff, mode, output='sos')
        filtered = signal.sosfilt(sos, data)
        return(filtered)

    def checker(point,P0,rad):
        
        if (point[0]-P0[0])**2 + (point[1]-P0[1])**2 <=rad**2:
            return True
        else:
            return False
    B_points=[]
    lons=[76.55+i*0.1 for i in range(20)]
    lats=[12.05+i*0.1 for i in range(20)]
    for y in lats:
        for x in lons:
            if checker([x,y],[77.55,13.05],2**0.5):
                B_points.append([x,y])
    def points_rad(point,rad,sep):
        X=np.arange(point[0]-rad,point[0]+rad,sep)
       
        Y=np.arange(point[1]-rad,point[1]+rad,sep)
        points=[]
        for x in X:
            
            for y in Y:
                
                if checker(np.array([x,y]),point,rad):
                    points.append(np.array([float(format(x, '.2f')),float(format(y, '.2f'))]))
        return(points)
    def circle_peak(dataset,circ_points):
        pt=[index_lon(float(circ_points[0][0])),index_lat(float(circ_points[0][1]))]
        max_val=weight(dataset,pt)
        point_max=circ_points[0]
        for point in circ_points:
            val=weight(dataset,[index_lon(float(point[0])),index_lat(float(point[1]))])
            if val>=max_val:
                point_max=point
                max_val=val
        return(max_val,point_max)
    def end_judge(dataset,max_val,t_index_now,point_thread):
        print(point_thread)
        point_max=point_thread[-1]
        circ_points=points_rad(point_max,1,0.1)
        data=dataset[t_index_now,:,:]
        max_now=circle_peak(data,circ_points)
        
   
    set_times = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range(yr+'-03-01T00:00:00Z', yr+'-05-31T23:30:00Z',
                                      freq='30T'))
       .between_time('00:00','23:30')
       .index.strftime('%Y%m%d-S%H%M%S')
       .tolist())

    daily_rain_data=[]
    rain_data=[]
    for time in set_times:
        for path in valid_paths:
            if path[0]=='3' and path[21:37]==time:
                ds=nc.Dataset(diri+"/"+path)
                daily_rain_data.append(np.array(ds['precipitationCal'][:]))
                #B_pt_rain=[]
                rain_sum=0
                for point in B_points:
                    #B_pt_rain.append((point,daily_rain_data[-1][index_lon(point[0]),index_lat(point[1])]))
                    rain_sum+=daily_rain_data[-1][0,index_lon(float(point[0])),index_lat(float(point[1]))]
                rain_data.append(rain_sum)  
    
    daily_rain_data=np.array(daily_rain_data)[:,0,:,:]
    rain_data=np.array(rain_data)
    peaks,_=signal.find_peaks(rain_data)
    
    for peak in peaks:
        maxp=circle_peak(daily_rain_data[peak,:,:],B_points)
        
        if maxp[0]>=np.average(daily_rain_data[peak,:,:]):
            point_thread.append(maxp[1])
        else:
            continue
        point_thread=[]
        t_index=peak-1
        
        while t_index>0 and maxp[0]>=np.average(daily_rain_data[t_index+1,:,:])+0.25*np.std(daily_rain_data[t_index+1,:,:]) :
            
            data=daily_rain_data[t_index]
            circ_points=points_rad(maxp[1],1,0.1)
            maxp=circle_peak(data,circ_points)
            t_index-=1
            point_thread.append(list(maxp[1]))
        events_tracking[set_times[peak]]=point_thread
    print('done')
    
