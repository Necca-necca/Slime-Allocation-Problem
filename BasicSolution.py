# Basic Solution
import numpy as np
import gurobipy

distance = np.load('distance.npy')
time = np.load('time.npy')
train_init = np.load('train_init.npy')
rate = np.load('rate.npy')
train_demand = np.load('train_demand.npy')

avg_slime_begin = np.around(np.mean(train_init*rate, 0))
avg_demand_per_day = np.around(np.mean(train_demand, 0))

trans_space_fee = 2.28
trans_time_fee = 0.63
dist_time_fee = 0.70

MODEL = gurobipy.Model("Q1")

# Varibles
x_trans = MODEL.addVars(8, 8, 6, vtype=gurobipy.GRB.INTEGER, name='x_trans')
x_dist = MODEL.addVars(8, 8, 6, vtype=gurobipy.GRB.INTEGER, name='x_dist')

MODEL.update()

# Objective
MODEL.setObjective(trans_space_fee * gurobipy.quicksum(x_trans[i,j,t] * distance[i,j] for i in range(0,8) for j in range(0,8) for t in range(0,6))
				   + trans_time_fee * gurobipy.quicksum(x_trans[i,j,t] * time[i,j] for i in range(0,8) for j in range(0,8) for t in range(0,6))
				   - dist_time_fee * gurobipy.quicksum(x_dist[i,j,t] * distance[i,j] for i in range(0,8) for j in range(0,8) for t in range(0,6))
				   , gurobipy.GRB.MAXIMIZE)

# Constraints
MODEL.addConstrs(x_trans[i,j,t] <= avg_demand_per_day[i,j,t+1] for i in range(0,8) for j in range(0,8) for t in range(0,6))

MODEL.addConstrs(gurobipy.quicksum(x_trans[i,j,0] for j in range(0,8)) + gurobipy.quicksum(x_dist[i,j,0] for j in range(0,8))
				 == avg_slime_begin[i] for i in range(0,8))

MODEL.addConstrs((gurobipy.quicksum(x_trans[i,j,t] for j in range(0,8)) + gurobipy.quicksum(x_dist[i,j,t] for j in range(0,8)))
				 == (gurobipy.quicksum(x_trans[j,i,t-1] for j in range(0,8)) + gurobipy.quicksum(x_dist[j,i,t-1] for j in range(0,8)))
				 for i in range(0,8) for t in range(1,6))

MODEL.optimize()
print("Obj:", MODEL.objVal)
