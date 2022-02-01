# Extension
import numpy as np
import gurobipy


def train():
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

	MODEL = gurobipy.Model("Q2")

	# Varibles
	x_trans = MODEL.addVars(8, 8, 6, vtype=gurobipy.GRB.CONTINUOUS, name='x_trans')
	x_dist = MODEL.addVars(8, 8, 6, vtype=gurobipy.GRB.CONTINUOUS, name='x_dist')
	a_dist = MODEL.addVars(8, 8, vtype=gurobipy.GRB.CONTINUOUS, name='a_dist')
	b_dist = MODEL.addVars(8, 8, 6, vtype=gurobipy.GRB.CONTINUOUS, name='b_dist')
	d_dist = MODEL.addVars(8, 8, 6, vtype=gurobipy.GRB.CONTINUOUS, name='d_dist')

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

	MODEL.addConstrs(gurobipy.quicksum(x_trans[i,j,t] for j in range(0,8)) + gurobipy.quicksum(x_dist[i,j,t] for j in range(0,8))
					 == gurobipy.quicksum(x_trans[j,i,t-1] for j in range(0,8)) + gurobipy.quicksum(x_dist[j,i,t-1] for j in range(0,8))
					 for i in range(0,8) for t in range(1,6))

	MODEL.addConstrs(a_dist[i,j] + b_dist[i,j,t]*avg_demand_per_day[i,j,t]
					 + d_dist[i,j,t] * gurobipy.quicksum(avg_slime_begin[k] for k in range(0,8))
					 == x_dist[i,j,t] for i in range(0, 8) for j in range(0, 8) for t in range(0, 6))

	MODEL.optimize()
	print("Obj:", MODEL.objVal)
	return np.array(MODEL.getAttr('X', a_dist.values())).reshape(8,8),\
		   np.array(MODEL.getAttr('X', b_dist.values())).reshape(8,8,6),\
		   np.array(MODEL.getAttr('X', d_dist.values())).reshape(8,8,6)


def lack_distribution(i, t, total_num, demand):
	temp = np.zeros(8)
	sum = 0
	for j in range(0,8):
		sum += demand[i, j, t]
		if sum <= total_num:
			temp[j] = demand[i, j, t]
		else:
			temp[j] = total_num - (sum-demand[i, j, t])
			break
	return temp


def test(a_dist, b_dist, d_dist):
	distance = np.load('distance.npy')
	time = np.load('time.npy')
	test_init = np.load('test_init.npy')
	rate = np.load('rate.npy')
	test_demand = np.load('test_demand.npy')

	trans_space_fee = 2.28
	trans_time_fee = 0.63
	dist_time_fee = 0.70

	taxi_begin = np.around(test_init*rate, 0)
	sum = 0

	for day in range(0,10):
		slime_begin_today = np.squeeze(taxi_begin[day,:])
		slimenum = np.sum(slime_begin_today)
		demand_today = np.squeeze(test_demand[day, :, :, :])
		x_trans = np.zeros((8, 8, 6))
		x_dist = np.zeros((8, 8, 6))
		x_dist_plan = np.maximum(np.round(np.dstack((a_dist, a_dist, a_dist, a_dist, a_dist, a_dist))
								+ b_dist * demand_today[:, :, :-1]
								+ d_dist * np.tile(np.sum(slime_begin_today),(8,8,6))
								), 0)
		for t in range(0, 6):
			for i in range(0, 8):
				slime_need_start_i = np.sum(demand_today[i, :, t+1])
				if slime_need_start_i <= slime_begin_today[i]:
					x_trans[i, :, t] = demand_today[i, :, t+1]
					slime_dist_start_i = slime_begin_today[i] - slime_need_start_i
					x_dist[i, :, t] = lack_distribution(i, t, slime_dist_start_i, x_dist_plan)
					if slime_dist_start_i > np.sum(x_dist[i, :, t]):
						x_dist[i, i, t] += slime_dist_start_i - np.sum(x_dist[i, :, t])
				else:
					x_trans[i, :, t] = lack_distribution(i, t, slime_begin_today[i], demand_today[:, :, 1:])
					x_dist[i, :, t] = 0

			cnt = 0
			for b in range(0, 8):
				slime_begin_today[b] = np.sum(x_trans[:, b, t]) + np.sum(x_dist[:, b, t])
				cnt += slime_begin_today[b]
			print("Day " + str(day + 1) + " Duration " + str(
				t + 1) + ": Check today's Total Slime Number (should be " + str(slimenum) + "), now is " + str(cnt))

		today_sum = trans_space_fee * np.sum(
			x_trans * np.dstack((distance, distance, distance, distance, distance, distance))) \
					+ trans_time_fee * np.sum(x_trans * np.dstack((time, time, time, time, time, time))) \
					- dist_time_fee * np.sum(x_dist * np.dstack((distance, distance, distance, distance, distance, distance)))
		sum += today_sum
		print("------------Day " + str(day + 1) + " Summary---------------")
		print("Need_times = " + str(np.sum(demand_today[:, :, 1:])))
		print("Trans_times = " + str(np.sum(x_trans)))
		print("Dist_times = " + str(np.sum(x_dist * np.dstack((time, time, time, time, time, time)))))
		print("Stay_locally_times = " + str(np.sum(x_dist)-np.sum(x_dist*np.dstack((time,time,time,time,time,time)))))
		print("Today's_Profit = " + str(today_sum))
		print("---------Day " + str(day + 1) + " Summary Over---------")

	print("10_Days_Totol_Profit = " + str(sum))


if __name__ == "__main__":
	a, b, d = train()
	test(a, b, d)