from gurobipy import Model, GRB, quicksum

# --- 1. Load your data into these Python structures ---
# G = list of gate IDs, e.g. [1,2,...,NG]
# F = list of flight IDs, e.g. [1,2,...,NF]
# P = list of passenger categories, e.g. [1,2,...,NPC]

# Parameters (all dicts keyed by tuples or single keys):
### Koen :
# aj[j], dj[j]                   : expected arrival/departure time of flight j
# na[p,j], nd[p,j], nt[p,j,j2]   : number of arriving/departing/transferring passengers
# wa[i], wd[i]                   : walking distances for arriving/departing
# wt[i,i2]                       : walking distance between gate i and i2
### Mariska :
# ra[p,i], rd[p,i], rt[p,i]      : revenue per passenger of category p at gate i
# ca[p], cd[p], ct[p]            : cost per meter for each passenger type p
# gi[i], gS[i]                   : gate i classifications
# fj[j], fS[j]                   : flight j classifications
### Bradut :
# theta[i], delta[i]             : taxi/prep times for gate i
# tau_t[i,i2], tau_i[i]          : min transfer and min free-gate times
# xP[i,j]                        : 1 if gate i already occupied by flight j
# M                              : a big-M constant (e.g. 1e5)

# (You can load these from CSV, a database, etc.)

# --- 2. Build the model ---
m = Model('GateAssignment_RevenueMax')

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
O4 = quicksum(nt[p,j,j2]*ct[p]*wt[i,i2]*z[i,i2,j,j2]
              for p in P for i in G for i2 in G for j in F for j2 in F)

# O5: arriving walking-cost
O5 = quicksum(na[p,j]*ca[p]*wa[i]*x[i,j] for p in P for i in G for j in F)

# O6: departing walking-cost
O6 = quicksum(nd[p,j]*cd[p]*wd[i]*x[i,j] for p in P for i in G for j in F)

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
m.addConstrs((c[j2] - b[j] >= tau_t[i,i2]*z[i,i2,j,j2]
               for i in G for i2 in G for j in F for j2 in F
               if nt[p,j,j2] > 0), name='transfer_time')

# 4.8 Minimum free-gate time
m.addConstrs((
    b[j2] - delta[i] - tau_i[i] - c[j] - delta[i] >=
      -M*(2 - x[i,j] - x[i,j2])
    for i in G for j in F for j2 in F if j!=j2 and aj[j] <= aj[j2]),
    name='free_gate')

# 4.9 Pre-assigned gates locked out
m.addConstrs((x[i,j] <= xP[i,j] for i in G for j in F),
             name='preassigned_lock')

# --- 5. Optimize ---
m.Params.TimeLimit = 600     # e.g. 10-minute time limit
m.optimize()

# --- 6. Extract solution ---
if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
    assign = {(i,j): x[i,j].X for i in G for j in F if x[i,j].X > 0.5}
    print("Flight→Gate assignment:", assign)
    print("Objective value:", m.objVal)
