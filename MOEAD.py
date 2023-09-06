#  Creator: Payne_Lee
#  Institution: NUAA
#  Date: 2023/5/11

import random
import math
import copy
import numpy as np
import random
import os
import QVNS_NSGA_II as Q
import Scheduling
import pandas as pd
import time

# -*- coding: utf-8 -*-

############################################################################
"""===================Initialization and Parameter setting==================="""

'''-------Scheduling parameter-------'''
JOB_LIST = [10, 20, 50]
STAGE_LIST = [3, 5]
MACHINE_LIST = [3, 4, 6]
OBJECTIVE = ['TT', 'TEC', 'CTC']
carbon_rate = Q.carbon_rate = 0.2  # 碳排放系数
carbon_price = Q.carbon_price = 30  # 每单位碳排放的价格

d_max = 8  # 经过实验以后d_max确定为8个
random.seed(42)

# raw_input is used in python 3
population_size = int(input('Please input the size of population: ') or 120)  # default value is 20
crossover_rate = float(input('Please input the size of Crossover Rate: ') or 0.85)  # default value is 0.8
mutation_rate = float(input('Please input the size of Mutation Rate: ') or 0.2)  # default value is 0.3
num_generations = int(input('Please input number of generations: ') or 100)  # default value is 1000
neighborhood_size = int(input('Please input number of neighborhood size: ') or 10)  # default value is 1000
max_episodes = int(input('Please input number of max iterations: ') or 10)
TOU_mode = int(input('Please input TOU_mode: ') or 4)  # default value is 4, spring and fall`
CT = float(input('Please input CT: ') or 0.8)  # default value is 0.8`

n_objectives = 3  # number of objectives


def para_initialization(job_num, stage_num, machine_num, TOU_mode):
    scale = str(job_num) + '-' + str(stage_num) + '-' + str(machine_num)  # 记录问题规模
    read_path = r'C:\Users\Lipei\Desktop' + '\\Program\\Data\\' + scale + '.xlsx'
    '''------read excel------'''
    Machine_num = [machine_num] * stage_num  # [3,3,3], machine list
    EA = job_num * stage_num
    '''-------Duedate parameter-------'''
    duedate_tmp = pd.read_excel(read_path, sheet_name='Duedate', index_col=[0])
    duedate = list(
        map(int, duedate_tmp.values[:, 0]))  # [9, 17, 9, 24, 18, 18, 27, 16, 8, 19]  # 预设的duedate

    '''-------process time parameter-------'''
    pt_tmp = pd.read_excel(read_path, sheet_name="Process Time", index_col=[0])
    Pt_machine_tmp = pd.read_excel(read_path, sheet_name='Pt_on_machine', index_col=[0])
    process_time = [sum(map(int, pt_tmp.iloc[i])) for i in range(job_num)]  # 读取总的工时数据,索引为job
    PT_on_machine = []  # 初始化每个机器对于每个工件的加工时间, 索引为stage, machine_num, job
    cnt_1 = 0  # 计数器
    for i in range(stage_num):
        tmp = []  # 用来临时存储工时信息
        for m in range(Machine_num[i]):
            tmp.append(list(map(int, Pt_machine_tmp.values[:, cnt_1])))
            cnt_1 += 1
        PT_on_machine.append(tmp)

    Power_mat = []
    TB_mat = []
    Power_tmp = pd.read_excel(read_path, sheet_name='Power&TB', index_col=[0])
    cnt_2 = 0  # 计数器
    for i in range(stage_num):
        tmp_1 = []  # 用来临时存储功率信息
        tmp_2 = []  # 用来临时存储盈亏平衡时间信息
        for m in range(Machine_num[i]):
            tmp_1.append(list(map(int, Power_tmp.iloc[cnt_2]))[:-1])
            tmp_2.append(list(map(int, Power_tmp.iloc[cnt_2]))[-1])
            cnt_2 += 1
        Power_mat.append(tmp_1)
        TB_mat.append(tmp_2)
    TOU_mat = Q.TOU(TOU_mode)

    return EA, Machine_num, duedate, process_time, PT_on_machine, Power_mat, TB_mat, TOU_mat


def weight_initialization():
    lambda_ = list()  # lambda_记录所有权重
    m = population_size
    k = 0
    count = 0
    for i in range(m):
        for j in range(m):
            count += 1
            if i + j <= m:
                k = m - i - j  # i,j,k的总和是100
                try:
                    weight_scalars = [float(i) / m, float(j) / m, float(k) / m]
                    lambda_.append(weight_scalars)
                except Exception as e:
                    print("Error creating weight with 3 objectives at:")
                    print("count", count)
                    print(i, j, k)
                    raise e

    # Trim number of weights to fit population size
    lambda_ = sorted((x for x in lambda_), key=lambda x: sum(x), reverse=True)
    lambda_ = lambda_[:population_size]
    return lambda_


def distVector(vector1, vector2):  # Calculate the Euclidian distance
    dim = len(vector1)
    sum_ = 0
    for n in range(dim):
        sum_ += ((vector1[n] - vector2[n]) * (vector1[n] - vector2[n]))
    return math.sqrt(sum_)


def neighbor_initialization(lambda_, T):
    x = [[] for _ in range(population_size)]  # Of type float, 记录每个权重向量和其他向量的距离
    idx = [[] for _ in range(population_size)]  # Of type int
    neighborhood_ = [[] for _ in range(population_size)]  # Of type int, 记录每个权重向量的邻居

    for i in range(population_size):
        for j in range(population_size):
            x[i].append(distVector(lambda_[i], lambda_[j]))
            idx[i].append(j)
            # 接下来根据x里的距离对idx进行排序，并取出最好的T_个
            Z = sorted(zip(x[i], idx[i]))  # 按x里的距离从小到大排列idx
            x_new, idx_new = zip(*Z)
            neighborhood_[i][0:T] = list(idx_new[0:T])  # 记录neighborhood

    return neighborhood_


def matingSelection(neighbourhood_, vector, cid, size):
    """
     vector : the set of indexes of selected mating parents
     cid    : the id of current subproblem
     size   : the number of selected mating parents"""
    ss = len(neighbourhood_[cid])  # 第i个子问题的权重向量对应的邻域大小

    while len(vector) < size:
        r = random.randint(0, ss - 1)  # random number to select obj_values
        p = neighbourhood_[cid][r]
        flag = True
        for i in range(len(vector)):
            if vector[i] == p:  # p is in the list
                flag = False
                break
        if flag:
            vector.append(p)
    return vector


def Tche_fitness(obj_values, lambda_, z_):
    # obj_values记录三个目标函数，idx为该解的index
    # lambda_为权重向量, z_为参考点
    fitness = float()
    maxFun = -1.0e+30  # 记录最大的fitness

    for n in range(n_objectives):
        diff = abs(obj_values[n] - z_[n])  # JMetal default
        feval = float()
        if lambda_[n] == 0:  # 即权重为0，用一个极小值代替
            feval = 0.0001 * diff
        else:
            feval = diff * lambda_[n]

        if feval > maxFun:
            maxFun = feval

    fitness = maxFun

    return fitness


def MOEA_D(Job_num, stage_num, machine_num, neighborhood_size, CT):
    # Neighbourhood size
    global best_front
    T = neighborhood_size

    # Z vector (ideal point)
    # List of size number of minimum objectives. Reference point
    z_ = {}  # 用来记录最好的目标函数，其实是obj_min

    # Lambda vectors (Weight vectors)
    lambda_ = weight_initialization()  # of type list of floats, i.e. [][], e.g. Vector of vectors of floats

    # Neighbourhood, record the neighbors of each lambda_i
    neighbourhood_ = neighbor_initialization(lambda_, T)  # of type int, i.e. [][], e.g. Vector of vectors of integers

    Q.EA, Q.Machine_num, Q.duedate, Q.process_time, Q.PT_on_machine, Q.Power_mat, Q.TB_mat, \
        TOU_mat = para_initialization(Job_num, stage_num, machine_num, TOU_mode)

    '''-----Initialize population and record the best obj value  -----'''
    population_list = Q.Initialization(population_size, TOU_mat)  # 初始化个体
    parent_obj_record = {}  # 记录目标函数值
    best_list = []  # 用来记录最好的个体
    best_record = {}  # 用来记录最好种群的目标函数

    time_start = time.time()
    for n in range(num_generations):
        # print(n)
        parent_list = copy.deepcopy(population_list)  # 生成父代

        if n == 0:  # 如果是第一代，计算一下目标函数和参考点—
            parent_obj_record = Q.fitness_calculation(parent_list, TOU_mat)  # 计算目标参数需要种群
            # Initialization
            lambda_ = weight_initialization()
            # 初始化权重向量
            neighbourhood_ = neighbor_initialization(lambda_, T)
            # 找到每个权重向量的邻居

            '''-----------Find the reference point-------------'''
            for i in range(3):  # 记录每个目标函数的最大值和最小值
                sort_temp = sorted(parent_obj_record.items(), key=lambda x: x[1][i])
                z_[i] = sort_temp[0][1][i]

        # STEP 2.1: Reproduction based on crossoevr rate
        offspring_list = []  # Used to store offspring solutions
        offspring_record = {}  # objective record

        for i in range(population_size):  # i即为子问题的index
            select_idx = list()  # Vector of type integer
            select_idx = matingSelection(neighbourhood_, select_idx, i, 2)  # Select parents
            children = Q.order_crossover(parent_list, select_idx, crossover_rate)
            child = random.choice(children)  # 从两个变异的个体中选一个出来

            # STEP 2.2 Local search
            child = Q.N1_critical_swap(child)
            offspring_list.append(child)

            # STEP 2.3 Evaluation and Update z_
            offspring_record[i] = Q.fitness_calculation([child], TOU_mat)[0]

            '''-----------Update the reference vector--------------'''
            for j in range(n_objectives):
                if offspring_record[i][j] < z_[j]:
                    z_[j] = offspring_record[i][j]  # 更新为新的最小值

            # STEP 2.4: Update the solutions
            size = len(neighbourhood_[i])  # 第idx个解的邻居
            for _ in range(size):
                k = neighbourhood_[i][_]  # the solution idx waited to be evaluated
                x0_obj = parent_obj_record[k]  # x0的目标函数值,等待评估
                f1 = Tche_fitness(x0_obj, lambda_[k], z_)
                f2 = Tche_fitness(offspring_record[i], lambda_[k], z_)
                # f1是原解的Tche值， f2是候选解的值
                if f2 >= f1:  # maximization assuming DEAP weights paired with fitness
                    population_list[k] = child
                    parent_obj_record[k] = offspring_record[i]

        time_end = time.time()
        time_c = time_end - time_start  # 运行所花时间
        if time_c > CPU_time:
            front, rank = Q.non_dominated_sorting(population_size, parent_obj_record)
            best_front = front[0]
            best_list = [population_list[i] for i in best_front]
            best_record = [parent_obj_record[i] for i in best_front]
            key = list(np.arange(len(best_list)))  # 目标函数的指针
            best_record = dict(zip(key, best_record))  # 转换成字典型
            break
        print('time cost', time_c, 's')

    return best_front, best_record, best_list


if __name__ == '__main__':
    save_path = r'C:\Users\Lipei\Desktop' + '\\Program\\Outcome_2023\\' + 'MOEAD' + '.xlsx'
    for Q.Job_num in JOB_LIST:
        for Q.stage_num in STAGE_LIST:
            for machine_num in MACHINE_LIST:
                CPU_time = CT * Q.Job_num * Q.stage_num
                output_sum = []
                scale = str(Q.Job_num) + '-' + str(Q.stage_num) + '-' + str(machine_num)  # 问题规模

                for j_ in range(5):
                    best_front, best_record, best_list = MOEA_D(Q.Job_num, Q.stage_num, machine_num, neighborhood_size,
                                                                CPU_time)
                    Pareto_front = best_front  # 记录最优解
                    best_no_repeat = []  # 记录无重复的个体
                    best_record_no_repeat = {}  # 记录无重复的个体
                    c_ = 0  # 用来计数
                    for i_ in range(len(best_list)):
                        if best_list[i_] not in best_no_repeat:
                            best_no_repeat.append(best_list[i_])
                            best_record_no_repeat[c_] = best_record[i_]
                            c_ += 1

                    '''__OUTPUT__'''
                    Name = ['TT', 'TEC', 'CTC', 'NND'] + [str(i) for i in list(range(Q.Job_num))]
                    output = pd.DataFrame(
                        np.zeros((len(best_no_repeat) + 1, len(Name))),  # output
                        columns=Name, dtype=np.float64)
                    for i in range(len(best_record_no_repeat)):
                        output['TT'][i] = best_record_no_repeat[i][0]  # 读取目标函数
                        output['TEC'][i] = best_record_no_repeat[i][1]  # 读取目标函数
                        output['CTC'][i] = best_record_no_repeat[i][2]  # 读取目标函数
                        output['NND'][i] = len(best_no_repeat)  # 非支配解的数量
                        for j in range(Q.Job_num):
                            output[str(j)][i] = best_list[i][j]  # output的第j个工件第i行等于最终结果的第i行第j列
                    if j_ == 0:
                        output_sum = output
                    else:
                        output_sum = output_sum.append(output)

                if scale == '10-3-3':  # 如果是第一个规模，那么先新建一个文件
                    sv_path = pd.ExcelWriter(save_path)
                else:
                    sv_path = pd.ExcelWriter(save_path, mode='a', engine='openpyxl', if_sheet_exists='new')
                output_sum.to_excel(sv_path, sheet_name=scale)
                sv_path.save()
                sv_path.close()
