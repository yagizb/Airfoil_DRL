import numpy as np

rho = 1		#Density
Uinf = 1	#Freestream velocity

c = 1		#Chord
Lz = 0.2 	#Spanwidth

S_ref = c * Lz 	#Reference surface
tAVG_start = 0.0

# #Literature coefficients
# Cl_ref = 1.05700 #Jones M 0.4
# Cd_ref = 0.01190

##FUNCTIONS ----------------------------
def compute_runmean(force, time=0, t_start=0):
      
	#Start averaging from t_start
	if t_start > 0:
		mask = np.where(time >= t_start)[0]
		time = time[mask]
		force = force[mask]
      
	#Perform average
	nsample = force.shape[0]
	runmean = np.zeros((nsample,))

	for i in range(nsample):

		if i == 0:
			runmean[i] = force[i]

		else:
			runmean[i] = np.mean(force[:i+1])

	return runmean, time

def compute_runStandardDeviation(force, time=0, t_start=0):

	#Start averaging from t_start
	if t_start > 0:
		mask = np.where(time >= t_start)[0]
		time = time[mask]
		force = force[mask]
      
	#Perform standard deviation
	nsample = force.shape[0]
	runEps = np.zeros((nsample,))

	for i in range(nsample):

		if i == 0:
			runEps[i] = np.std(force[0])

		else:
			runEps[i] = np.std(force[:i+1])

	return runEps, time	

def calclcd(output_geo,angle_of_attack):

    AoA = np.deg2rad(angle_of_attack)

    #Read Surface data [Iter, Time, Area, Fp_x, Fp_y, Fp_z, Fv_x, Fv_y, Fx_z]
    surf_data = np.genfromtxt(f"surf_code_1-{output_geo}-4.dat", delimiter = ',', skip_header=1)


    #Filter NaNs from points where the simulation diverged
    surf_data = surf_data[~np.isnan(surf_data).any(axis=1)]


    #Filter backwards steps
    time = surf_data[:,1]

    mask = np.ones_like(time, dtype=bool)
    recovery_indices = []
    max_value = time[0]
    for i in range(1, len(time)):
        if time[i] < time[i - 1]:
            mask[i] = False
        else:
            max_value = max(max_value, time[i])
            if time[i] < max_value:
                mask[i] = False
                recovery_indices.append(i)
    surf_data = surf_data[mask]


    #Take iterations and time
    iter = surf_data[:,0]
    time = surf_data[:,1] * Uinf/c


    #Compute Lift Coefficient
    Cl_pres = ( -surf_data[:,3]*np.sin(AoA) + surf_data[:,4]*np.cos(AoA) ) / (0.5*rho*Uinf*Uinf*S_ref)
    Cl_visc = ( -surf_data[:,6]*np.sin(AoA) + surf_data[:,7]*np.cos(AoA) ) / (0.5*rho*Uinf*Uinf*S_ref)

    Cl = Cl_pres + Cl_visc


    #Compute Drag Coefficient
    Cd_pres = ( surf_data[:,3]*np.cos(AoA) + surf_data[:,4]*np.sin(AoA) ) / (0.5*rho*Uinf*Uinf*S_ref)
    Cd_visc = ( surf_data[:,6]*np.cos(AoA) + surf_data[:,7]*np.sin(AoA) ) / (0.5*rho*Uinf*Uinf*S_ref)

    Cd = Cd_pres + Cd_visc


    #Compute running mean
    Cl_avg, time_Clavg = compute_runmean(Cl, time = time, t_start = tAVG_start)
    # Cl_avg_pres = compute_runmean(Cl_pres, time = time, t_start = tAVG_start)[0]
    # Cl_avg_visc = compute_runmean(Cl_visc, time = time, t_start = tAVG_start)[0]

    Cd_avg, time_Cdavg = compute_runmean(Cd, time = time, t_start = tAVG_start)
    # Cd_avg_pres = compute_runmean(Cd_pres, time = time, t_start = tAVG_start)[0]
    # Cd_avg_visc = compute_runmean(Cd_visc, time = time, t_start = tAVG_start)[0]

    # #Compute running standard deviation 
    # Cl_std, time_Clstd = compute_runStandardDeviation(Cl, time = time, t_start = tAVG_start)
    # Cd_std, time_Cdstd = compute_runStandardDeviation(Cd, time = time, t_start = tAVG_start)

    # #PRINTS ------------------------
    # print('------------Average Results ---------------')
    # print('Cd: ', Cd_avg[-1])
    # print('Cd PRESSURE: ', Cd_avg_pres[-1])
    # print('Cd VISCOUS: ', Cd_avg_visc[-1])
    # print('C\'d, rms: ', Cd_std[-1])


    # print('')
    # print('Cl: ', Cl_avg[-1])
    # print('Cl PRESSURE: ', Cl_avg_pres[-1])
    # print('Cl VISCOUS: ', Cl_avg_visc[-1])
    # print('C\'l, rms: ', Cl_std[-1])

    # print('-------------Relative error------------------')
    # print('Cd error (%): ', (np.abs(Cd_avg[-1] - Cd_ref)/Cd_ref)*100)
    # print('Cl error (%): ', (np.abs(Cl_avg[-1] - Cl_ref)/Cl_ref)*100)

    return Cl_avg[-1], Cd_avg[-1]
# output_geo = ""
# angle_of_attack = 0
# Cl, Cd = calclcd(output_geo, angle_of_attack)
# print(f"Cl: {Cl}, Cd: {Cd}")
