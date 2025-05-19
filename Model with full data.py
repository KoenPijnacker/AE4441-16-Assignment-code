from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np

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
xp[6,1] = 1
# xp[8,2] = 1
# xp[1,3] = 1
# xp[9,4] = 1
# xp[2,5] = 1
# xp[15,6] = 1
# xp[7,7] = 1
# xp[5,8] = 1
# xp[4,9] = 1	
# xp[3,10] = 1
# xp[12,11] = 1
# xp[10,12] = 1
# xp[13,13] = 1
# xp[11,14] = 1
# xp[14,15] = 1
# xp[26,16] = 1
xp[6,17] = 1
# xp[17,18] = 1
M = 1e6 #Big M constant

### Bradut :
# tau_t[i,i2], tau_i[i]          : min transfer and min free-gate times

# (You can load these from CSV, a database, etc.)


# --- 2. Build the model ---
m = Model('GateAssignment_RevenueMax_FullData')

# 2.1 Decision variables
# x[i,j] = 1 if flight j is assigned to gate i
x = m.addVars(G, F, vtype=GRB.BINARY, name='x')

# y[j,j2] = 1 if flight j departs no later than flight j2 lands
y = m.addVars(F, F, vtype=GRB.BINARY, name='y')

# z[i,i2,j,j2] = 1 if flight j at i and flight j2 at i2 (transfer pair)
z = m.addVars(G, G, F, F, vtype=GRB.BINARY, name='z')

# b[j], c[j]: continuous, disembark/embark times
b = m.addVars(F, lb=0.0, name='b')
c = m.addVars(F, lb=0.0, name='c')

# --- 3. Objective (maximize revenues minus walking-costs) ---
# O1: transfers revenue
O1 = quicksum(nt[p,j,j2]*rt[p,i]*z[i,i2,j,j2]
              for p in P for i in G for i2 in G for j in F for j2 in F)

# O2: arriving revenue
O2 = quicksum(na[p,j]*ra[p,i]*x[i,j] for p in P for i in G for j in F)

# O3: departing revenue
O3 = quicksum(nd[p,j]*rd[p,i]*x[i,j] for p in P for i in G for j in F)

# O4: transfer walking-cost
O4 = quicksum(nt[p,j,j2]*ct*wt[i,i2]*z[i,i2,j,j2]
              for p in P for i in G for i2 in G for j in F for j2 in F)

# O5: arriving walking-cost
O5 = quicksum(na[p,j]*ca*wa[i]*x[i,j] for p in P for i in G for j in F)

# O6: departing walking-cost
O6 = quicksum(nd[p,j]*cd*wd[i]*x[i,j] for p in P for i in G for j in F)

m.setObjective(O1 + O2 + O3 - O4 - O5 - O6, GRB.MAXIMIZE)

# --- 4. Constraints ---

# 4.1 Each flight assigned to exactly one gate
m.addConstrs((quicksum(x[i,j] for i in G) == 1 for j in F), name='assign_once')

# 4.2 Linking z to x
m.addConstrs((z[i,i2,j,j2] <= x[i,j]
               for i in G for i2 in G for j in F for j2 in F),
             name='z_le_x1')
m.addConstrs((z[i,i2,j,j2] <= x[i2,j2]
               for i in G for i2 in G for j in F for j2 in F),
             name='z_le_x2')
m.addConstrs((z[i,i2,j,j2] >= x[i,j] + x[i2,j2] - 1
               for i in G for i2 in G for j in F for j2 in F),
             name='z_ge_xx')

# 4.3 Compute b[j] and c[j]
m.addConstrs((b[j] == aj[j] + quicksum((theta[i]+delta[i])*x[i,j] for i in G)
               for j in F), name='def_b')
m.addConstrs((c[j] == dj[j] - quicksum((theta[i]+delta[i])*x[i,j] for i in G)
               for j in F), name='def_c')

# 4.4 Order variables y
m.addConstrs((c[j] - b[j2] + y[j,j2]*M >= 0
               for j in F for j2 in F), name='order1')
m.addConstrs((c[j] - b[j2] - (1-y[j,j2])*M <= 0
               for j in F for j2 in F), name='order2')

# 4.5 Prevent same-gate conflict
m.addConstrs((y[j,j2] + y[j2,j] >= z[i,i,j,j2]
               for i in G for j in F for j2 in F if j!=j2),
             name='no_conflict')

# 4.6 Gate/flight compatibility
m.addConstrs((gi[i] >= fj[j]*x[i,j]
               for i in G for j in F), name='size_compat')
m.addConstrs((x[i,j] == 0
               for i in G for j in F if gS[i] != fS[j]),
             name='schengen_compat')

# 4.7 Transfer time feasibility
#valid pairs to be determined first
valid_pairs = [(i, i2, j, j2) for i in G for i2 in G for j in F for j2 in F
               if np.sum(nt[1:9, j, j2]) > 0]

m.addConstrs(
    (c[j2] - b[j] >= tau_t[i, i2] * z[i, i2, j, j2]
     for i, i2, j, j2 in valid_pairs),
    name='transfer_time'
)

# 4.8 Minimum free-gate time
m.addConstrs((
    b[j2] - delta[i] - tau_i - c[j] - delta[i] >=
      -M*(2 - x[i,j] - x[i,j2])
    for i in G for j in F for j2 in F if j!=j2 and aj[j] <= aj[j2]),
    name='free_gate')

# 4.9 Pre-assigned gates locked out
m.addConstrs((x[i,j] >= xp[i,j] for i in G for j in F),
             name='preassigned_lock')

# --- 5. Optimize ---
m.params.LogFile='GateAssignment_RevenueMax.log'
m.Params.TimeLimit = 600     # e.g. 10-minute time limit
m.optimize()

# --- 6. Extract solution ---
if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
    # Recompute O1 to O6 using the values of the variables
    O1_val = sum(nt[p, j, j2] * rt[p, i] * z[i, i2, j, j2].X
                 for p in P for i in G for i2 in G for j in F for j2 in F)

    O2_val = sum(na[p, j] * ra[p, i] * x[i, j].X
                 for p in P for i in G for j in F)

    O3_val = sum(nd[p, j] * rd[p, i] * x[i, j].X
                 for p in P for i in G for j in F)

    O4_val = sum(nt[p, j, j2] * ct * wt[i, i2] * z[i, i2, j, j2].X
                 for p in P for i in G for i2 in G for j in F for j2 in F)

    O5_val = sum(na[p, j] * ca * wa[i] * x[i, j].X
                 for p in P for i in G for j in F)

    O6_val = sum(nd[p, j] * cd * wd[i] * x[i, j].X
                 for p in P for i in G for j in F)

    print(f"O1 (Transfer Revenue):         {O1_val:.2f}")
    print(f"O2 (Arrival Revenue):          {O2_val:.2f}")
    print(f"O3 (Departure Revenue):        {O3_val:.2f}")
    print(f"O4 (Transfer Walking Cost):   -{O4_val:.2f}")
    print(f"O5 (Arrival Walking Cost):    -{O5_val:.2f}")
    print(f"O6 (Departure Walking Cost):  -{O6_val:.2f}")
    print(f"Total Objective:               {(O1_val + O2_val + O3_val - O4_val - O5_val - O6_val):.2f}")
    assign = {(i,j): x[i,j].X for i in G for j in F if x[i,j].X > 0.5}
    print("Gate→Flight assignment:", assign)
    print("Objective value:", m.objVal)
else:
    print("Optimization was not successful.")
