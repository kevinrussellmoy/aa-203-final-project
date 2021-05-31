#%% File to compute optimal LMP dispatch from load data and tariff rate pricing
# Kevin Moy, 11/3/2020

import cvxpy as cp
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% Set environment variables:
# LMP_LEN = lmp.size  # length of optimization
BAT_KW = 5.0  # Rated power of battery, in kW, continuous power for the Powerwall
BAT_KWH = 14.0  # Rated energy of battery, in kWh.
# Note Tesla Powerwall rates their energy at 13.5kWh, but at 100% DoD,
# but I have also seen that it's actually 14kwh, 13.5kWh usable
BAT_KWH_MIN = 0.0 * BAT_KWH  # Minimum SOE of battery, 10% of rated
BAT_KWH_MAX = 1.0 * BAT_KWH  # Maximum SOE of battery, 90% of rated
BAT_KWH_INIT = 0.5 * BAT_KWH  # Starting SOE of battery, 50% of rated
HR_FRAC = (
    60 / 60
)  # Data at 60 minute intervals, which is 1 hours. Need for conversion between kW <-> kWh


#%% Import load and tariff rate data; convert to numpy array and get length
df = pd.read_csv("df_LMP.csv")
# load = df.gridnopv[0:288].to_numpy()
# tariff = df.tariff[0:288].to_numpy()
# times = pd.to_datetime(df.local_15min[0:288])
lmp = df.LMP_kWh.to_numpy()
times = pd.to_datetime(df.DATETIME)

#%% TODO: Solve for one week using Gurobi (will become one iteration of MPC)

week_len = 24*7*40
lmp_wk = lmp[:week_len]

# Create a new model

m = gp.Model('lmp')

# Create variables for:
# each power flow
# Power dispatched from ESS (positive=discharge, negative=charge)
ess_c = m.addMVar(week_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_c')
ess_d = m.addMVar(week_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_d')
ess_p = m.addMVar(week_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_d')

#Energy stored in ESS
E = m.addMVar(week_len, lb=0, vtype=GRB.CONTINUOUS, name='E')

m.addConstr(E[0] == BAT_KWH_INIT)

for t in range(week_len):
    # ESS power constraints
    m.addConstr(ess_d[t] <= BAT_KW)
    m.addConstr(ess_c[t] <= BAT_KW)
    m.addConstr(ess_p[t] == ess_d[t] - ess_c[t])

    m.addConstr(E[t] <= BAT_KWH)
    m.addConstr(E[t] >= 0) 
    m.addConstr(ess_c[t] >= 0)
    m.addConstr(ess_d[t] >= 0)

    # Time evolution of stored energy
    if t > 0:
        m.addConstr(E[t] == E[t-1] + HR_FRAC*(ess_c[t-1] - ess_d[t-1]))

# #Ensure non-simultaneous charge and discharge aka why I downloaded Gurobi
m.addConstrs(0 == ess_d[i] @ ess_c[i] for i in range(week_len))

# Objective function
m.setObjective(HR_FRAC*(sum(lmp_wk[i] * (ess_c[i] - ess_d[i]) for i in range(week_len))), GRB.MINIMIZE)

# Solve the optimization
m.params.NonConvex = 2
m.optimize()

plt.plot(ess_d.getAttr('x'))
plt.plot(ess_c.getAttr('x'))
#%%
# Create optimization variables.
chg_pow = cp.Variable(week_len)  # Power charged to the battery
dch_pow = cp.Variable(week_len)  # Power discharged from the battery
bat_eng = cp.Variable(week_len)  # Energy stored in the battery

# Create constraints.
constraints = [bat_eng[0] == BAT_KWH_INIT]

for i in range(week_len):
    constraints += [
        chg_pow[i] <= BAT_KW,
        dch_pow[i] <= BAT_KW,
        bat_eng[i] <= BAT_KWH_MAX,  # Prevent overcharging
        bat_eng[i] >= BAT_KWH_MIN,  # Prevent undercharging
        bat_eng[i]
        >= HR_FRAC * dch_pow[i],  # Prevent undercharging from overdischarging
        # Convexity requirements:
        chg_pow[i] >= 0,
        dch_pow[i] >= 0,
        bat_eng[i] >= 0,
    ]

for i in range(1, week_len):
    constraints += [
        bat_eng[i]
        == HR_FRAC * chg_pow[i - 1] + (bat_eng[i - 1] - HR_FRAC * dch_pow[i - 1])
    ]  # Energy flow constraints

print("constraints complete")

# Form objective.
obj = cp.Maximize(lmp_wk.T @ (dch_pow - chg_pow))
# obj = cp.Minimize(lod_pow.T @ np.ones(LOAD_LEN))


# Form and solve problem.
prob = cp.Problem(obj, constraints)
print("solving...")
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)

# Calculate relevant quantities.
bat_pow = dch_pow.value - chg_pow.value
cumulative_revenue = np.cumsum(bat_pow * lmp)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
p1 = ax1.plot(bat_pow)


#%% Save output to CSV.
print("saving to CSV")
outputdf = pd.DataFrame(
    np.transpose([bat_pow, bat_eng.value, lmp, cumulative_revenue])
)
outputdf.columns = [
    "battery_power",
    "battery_energy",
    "lmp",
    "cumulative_cost",
]
outputdf.set_index(times, inplace=True)
outputdf.to_csv("opt_lmp_5kW_14kWh.csv")


#%% PLOTTING !

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
p1 = ax1.plot(times, bat_pow)

color = "tab:red"
ax2 = ax1.twinx()
ax2.set_ylabel("Energy Price, $/kWh", color=color)
p4 = ax2.plot(times, lmp, color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim([0, 1.1 * max(lmp)])
ax2.xaxis.set_major_formatter(xfmt)

plt.legend(
    (p1[0]),
    ("Battery Power"),
    loc="best",
)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig("opt_ex_lmp.png")

# for i in range(len(data_frame)):
#     if i % 1000 == 0: print(i)
#     constraints += [rate[i] <= discharge_max,  # Rate should be lower than or equal to max rate,
#                     rate[i] >= charge_max,
#                     E[i] <= SOC_max,  # Overall kW should be within the range of [SOC_min,SOC_max]
#                     E[i] >= SOC_min]
#     revenue += prices[i] * (
#     rate[i])  # Revenue = sum of (prices ($/kWh) * (energy sold (kW) * 1hr - energy bought (kW) * 1hr) at timestep t)
# for i in range(1, len(data_frame)):
#     if i % 1000 == 0: print(i)
#     constraints += [E[i] == E[i - 1] + rate[i - 1]]  # Current SOC constraint
# constraints += [E[0] == random.uniform(SOC_min, SOC_max), rate[0] == 0]  # create first time step constraints