import numpy as np
import pandas as pd


# --- 1. Load your data into these Python structures ---
G = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
F = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
P = [0,1,2,3,4,5,6,7,8]

# Parameters (all dicts keyed by tuples or single keys):
### Koen :
# aj[j], dj[j]                   : expected arrival/departure time of flight j
# na[p,j], nd[p,j], nt[p,j,j2]   : number of arriving/departing/transferring passengers
# wa[i], wd[i]                   : walking distances for arriving/departing
# wt[i,i2]                       : walking distance between gate i and i2



aj = pd.read_excel('Full_data.xlsx', sheet_name='aj').to_numpy()[0]
dj = pd.read_excel('Full_data.xlsx', sheet_name='dj').to_numpy()[0]
na = pd.read_excel('Full_data.xlsx', sheet_name='na',usecols="B:W").to_numpy()
nd = pd.read_excel('Full_data.xlsx', sheet_name='nd',usecols="B:W").to_numpy() 
data_for_nt = pd.read_excel('Full_data.xlsx', sheet_name='nt', usecols="B:W",).to_numpy()
wa = pd.read_excel('Full_data.xlsx', sheet_name='wa').to_numpy()[0]
wd = pd.read_excel('Full_data.xlsx', sheet_name='wd').to_numpy()[0]
wt = pd.read_excel('Full_data.xlsx', sheet_name='wt',usecols="B:AH").to_numpy()

nt = np.zeros((9, len(F), len(F)))
# for nt, we need to fill the 3D array with the values from the data_for_nt
for i in [6,7,8]:
    for j in range(len(F)):
        j_nt = j +(i-6)*(2+len(F))
        for j2 in range(len(F)):
            nt[i,j,j2] = data_for_nt[j_nt][j2]

ra = pd.read_excel('Full_data.xlsx', sheet_name='ra',usecols="B:AH").to_numpy()
rd = pd.read_excel('Full_data.xlsx', sheet_name='rd',usecols="B:AH").to_numpy()
rt = pd.read_excel('Full_data.xlsx', sheet_name='rt',usecols="B:AH").to_numpy()
ca, cd, ct = 0.012, 0.012, 0.012
gi = pd.read_excel('Full_data.xlsx', sheet_name='gi').to_numpy()[0]
gS = pd.read_excel('Full_data.xlsx', sheet_name='gS').to_numpy()[0]
fj = pd.read_excel('Full_data.xlsx', sheet_name='fj').to_numpy()[0]
fS = pd.read_excel('Full_data.xlsx', sheet_name='fS').to_numpy()[0]

theta = pd.read_excel('Full_data.xlsx', sheet_name='thetai').to_numpy()[0]
delta = pd.read_excel('Full_data.xlsx', sheet_name='deltai').to_numpy()[0]
tau_t = pd.read_excel('Full_data.xlsx', sheet_name='taut',usecols="B:AH").to_numpy()
tau_i = 5
xp = np.zeros((len(G),len(F)), dtype=int)
xp[5,0] = 1
xp[7,1] = 1
xp[0,2] = 1
xp[8,3] = 1
xp[1,4] = 1
xp[14,5] = 1
xp[6,6] = 1
xp[4,7] = 1
xp[3,8] = 1	
xp[2,9] = 1
xp[11,10] = 1
xp[9,11] = 1
xp[12,12] = 1
xp[10,13] = 1
xp[13,14] = 1
# xp[25,15] = 1
# xp[5,16] = 1
# xp[16,17] = 1
M = 1e6 #Big M constant

### Returns a list of array variantions with the same total passengers but different distribution for each gate ###
def sensitivity_analysis_passengers(parameter, change):

    #Calculate average distribution of passengers
    passenger_distribution = []
    passenger_array = []
    distribution_array = []
    for flight in range(len(F)):
        number_passengers = np.sum(parameter[:, flight])
        if number_passengers != 0:
            passenger_distribution.append((parameter[:, flight] / number_passengers))

    dist_matrix = np.column_stack(passenger_distribution)
    
    row_avg = dist_matrix.mean(axis=1)

    #Vary the distribution of passengers
    for i in range(len(row_avg)):
        for j in range(len(row_avg)):
            if i != j:
                if row_avg[i]> 0 and row_avg[j] > 0:
                    new_distribution = row_avg.copy()
                    new_distribution[i] += change
                    new_distribution[j] -= change
                    distribution_array.append(new_distribution)


    #Calculate new passenger distribution for each flight
    for variation in range(len(distribution_array)):
        var_distribution = distribution_array[variation]
        for flight in range(len(F)):
            passenger_distribution = np.rint(var_distribution * number_passengers)
            passenger_array.append(passenger_distribution)


    return passenger_array

### Returns a list of array variantions with the same total revenue but different distribution for each gate ###
def sensitivity_analysis_revenue(parameter, change): #Change no more than 0.05
    revenue_distribution = []
    revenue_array = []
    distribution_array = []
    #Calculate average distribution of revenue
    for gate in range(len(G)):
        total_revenue = np.sum(parameter[:, gate])
        if total_revenue != 0:
            revenue_distribution.append((parameter[:, gate] / total_revenue))

    dist_matrix = np.column_stack(revenue_distribution)
    
    row_avg = dist_matrix.mean(axis=1)

    #Vary the distribution of passengers
    for i in range(len(row_avg)):
        for j in range(len(row_avg)):
            if i != j:
                if row_avg[i]> 0 and row_avg[j] > 0:
                    new_distribution = row_avg.copy()
                    new_distribution[i] += change
                    new_distribution[j] -= change
                    distribution_array.append(new_distribution)


    #Calculate new passenger distribution for each flight
    for variation in range(len(distribution_array)):
        var_distribution = distribution_array[variation]
        for gate in range(len(G)):
            passenger_distribution = var_distribution * total_revenue
            revenue_array.append(passenger_distribution)


    return revenue_array