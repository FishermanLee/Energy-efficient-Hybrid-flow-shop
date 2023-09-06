#  Creator: Payne_Lee
#  Institution: NUAA
#  Date: 2023/5/3
############################################################################
import copy

############################################################################

# Required Libraries
import numpy as np
import random
import QVNS_NSGA_II_no_power_down as Q
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

EPISILON = Q.EPISILON = 0.1  # greedy police
ALPHA = Q.ALPHA = 0.1  # learning rate
LAMBDA = Q.LAMBDA = 0.9  # discount factor
d_max = 8  # 经过实验以后d_max确定为8个
MAX_ITER = Q.MAX_ITER = 10  # MAX_ITER without improvement
ACTIONS_1 = list(np.arange(1, d_max + 1))  # DC的可能插入点数量  # 记录DC的节点动作
ACTIONS_2 = ['N1_GInsr_TT', 'N2_GInsr_TEC', 'N3_GInsr_CTC',
             'N4_Cswap', 'N5_CInsr', 'N6_CInv']  # Second Q-table
random.seed(42)

# raw_input is used in python 3
population_size = int(input('Please input the size of population: ') or 120)  # default value is 20
crossover_rate = float(input('Please input the size of Crossover Rate: ') or 0.8)  # default value is 0.8
mutation_rate = float(input('Please input the size of Mutation Rate: ') or 0.3)  # default value is 0.3
num_generations = int(input('Please input number of generations: ') or 100)  # default value is 1000
max_episodes = int(input('Please input number of max iterations: ') or 3)  # default value is 1
# it is 10 when training Q-table
TOU_mode = int(input('Please input TOU_mode: ') or 4)  # default value is 4, spring and fall`
CT = float(input('Please input CT: ') or 0.8)  # default value is 0.8`


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


############################################################################
def evolve(CHS, best_solution, worst_solution):
    new_solution = copy.deepcopy(CHS)
    for i in range(len(CHS)):
        if new_solution[i] == worst_solution[i]:  # Same job in the same position
            new_solution[i] = best_solution[i]
    # drop duplicates and reinsert
    missing_job = [item for item in best_solution if item not in new_solution]
    if missing_job:  # is miss_job is not null, the solution has duplicated jobs
        new_solution = list(set(new_solution))
        for job in missing_job:
            idx = random.randint(0, len(new_solution))  # random_insert
            new_solution.insert(idx, job)
    return new_solution


############################################################################

if __name__ == '__main__':
    save_path = r'C:\Users\Lipei\Desktop' + '\\Program\\Outcome_2023\\' + 'Jaya' + '.xlsx'
    global best_list, best_front, best_record
    for Q.Job_num in JOB_LIST:
        for Q.stage_num in STAGE_LIST:
            for machine_num in MACHINE_LIST:
                Q.EA, Q.Machine_num, Q.duedate, Q.process_time, Q.PT_on_machine, Q.Power_mat, Q.TB_mat, \
                    TOU_mat = para_initialization(Q.Job_num, Q.stage_num, machine_num, TOU_mode)
                # time_start = time.time()  # 记录开始时间
                scale = str(Q.Job_num) + '-' + str(Q.stage_num) + '-' + str(machine_num)  # 问题规模
                CPU_Time = CT * Q.Job_num * Q.stage_num  # 运行时间限制
                output_sum = []  # 记录所有的记录

                for j_ in range(5):  # 五次实验
                    Q.EA, Q.Machine_num, Q.duedate, Q.process_time, Q.PT_on_machine, Q.Power_mat, Q.TB_mat, \
                        TOU_mat = para_initialization(Q.Job_num, Q.stage_num, machine_num, TOU_mode)
                    population_list = Q.Initialization(population_size, TOU_mat)  # Initial population

                    time_start = time.time()
                    for n in range(num_generations):
                        print('This is ', n, 'time')
                        '''---------------------------Calculate the objective values---------------------------'''
                        parent_obj_record = Q.fitness_calculation(population_list, TOU_mat)  # Objctive values calculation
                        front, rank = Q.non_dominated_sorting(population_size, parent_obj_record)  # non dominated sorting
                        obj_min_max = {}  # Used to store the objective values of best and worst individuals

                        '''---------------------------Find the best solution and worst solution---------------------'''
                        distance_first = Q.calculate_crowding_distance(front[0], parent_obj_record)  # 计算
                        sorted_cdf = sorted(distance_first, key=distance_first.get)
                        sorted_cdf.reverse()
                        best_solution = population_list[sorted_cdf[0]]

                        distance_last = Q.calculate_crowding_distance(front[len(front) - 1], parent_obj_record)
                        sorted_cdf = sorted(distance_last, key=distance_last.get)  # select the solution with minimal
                        # crowding distance
                        worst_solution = population_list[sorted_cdf[0]]

                        '''---------------------------Update the solution---------------------------'''
                        for i in range(population_size):
                            compare_record = {}  # Used to compare two solution
                            x0 = population_list[i]  # select the solution
                            x_ = evolve(x0, best_solution, worst_solution)
                            compare_record['x0'] = parent_obj_record[i]
                            compare_record['x_'] = Q.fitness_calculation([x_], TOU_mat)[0]
                            if Q.dominate('x_', 'x0', compare_record):
                                population_list[i] = copy.deepcopy(x_)
                                parent_obj_record[i] = compare_record['x_']

                        time_end = time.time()
                        time_c = time_end - time_start
                        if time_c > CPU_Time:
                            front, rank = Q.non_dominated_sorting(population_size, parent_obj_record)
                            best_front = front[0]  # Pareto front
                            best_list = [population_list[i] for i in best_front]
                            best_record = [parent_obj_record[i] for i in best_front]
                            key = list(np.arange(len(best_list)))  # 目标函数的指针
                            best_record = dict(zip(key, best_record))  # 转换成字典型
                            break  # 进入下一循环
                        print('Time cost', time_c, 's')

                        # if n == num_generations:
                        #     front, rank = Q.non_dominated_sorting(population_size, parent_obj_record)
                        #     best_front = front[0]  # Pareto front
                        #     best_list = [population_list[i] for i in best_front]
                        #
                        #     best_record = [parent_obj_record[i] for i in best_front]
                        #     key = list(np.arange(len(best_list)))  # 目标函数的指针
                        #     best_record = dict(zip(key, best_record))  # 转换成字典型
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

