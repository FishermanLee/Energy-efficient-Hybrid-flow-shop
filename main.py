import pandas as pd
import QVNS_NSGA_II as Q
import numpy as np
import time
import random
import copy

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
MAX_ITER = Q.MAX_ITER = 5  # MAX_ITER without improvement
ACTIONS_1 = list(np.arange(1, d_max + 1))  # DC的可能插入点数量  # 记录DC的节点动作
ACTIONS_2 = ['N1_Cswap', 'N2_CInsr', 'N3_CInv', 'N4_threepoint_heuristic', 'N5_semitwo_crossover']  # Second Q-table
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
    q_table_tmp_1 = pd.read_excel(read_path, sheet_name='Q_table_1', index_col=[0], header=0)
    q_table_tmp_2 = pd.read_excel(read_path, sheet_name='Q_table_2', index_col=[0], header=0)
    Q_table_1 = pd.DataFrame(q_table_tmp_1)  # q_table initialization
    Q_table_2 = pd.DataFrame(q_table_tmp_2)  # q_table initialization

    return EA, Machine_num, duedate, process_time, PT_on_machine, Power_mat, TB_mat, TOU_mat, Q_table_1, Q_table_2


if __name__ == "__main__":
    save_path = r'C:\Users\Lipei\Desktop' + '\\Program\\Outcome_2023\\' + 'QVNS-NSGA-II' + '.xlsx'
    for Q.Job_num in JOB_LIST:
        for Q.stage_num in STAGE_LIST:
            for machine_num in MACHINE_LIST:
                Q.EA, Q.Machine_num, Q.duedate, Q.process_time, Q.PT_on_machine, Q.Power_mat, Q.TB_mat, \
                    TOU_mat, Q_table_1, Q_table_2 = para_initialization(Q.Job_num, Q.stage_num, machine_num, TOU_mode)
                # time_start = time.time()  # 记录开始时间
                scale = str(Q.Job_num) + '-' + str(Q.stage_num) + '-' + str(machine_num)  # 问题规模
                CPU_Time = CT * Q.Job_num * Q.stage_num  # 运行时间限制
                output_sum = []  # 记录所有的记录

                for j_ in range(5):  # 五次实验
                    print('This is', j_, 'experiment of', scale)
                    population_list = Q.Initialization(population_size, TOU_mat)  # 初始化个体
                    best_list = []  # 用来记录最好的个体
                    best_record = {}  # 用来记录最好种群的目标函数
                    obj_min = {}  # 用来记录最好的目标函数
                    obj_max = {}  # 用来记录最差的目标函数
                    front, rank, parent_obj_record = {}, {}, {}  # 初始化

                    time_start = time.time()
                    for n in range(num_generations):
                        print(n)

                        parent_list = copy.deepcopy(population_list)  # 生成父代
                        '''-----non_dominated_rank and crowding_distance_calculation  -----'''
                        if n == 0:  # 如果是第一代，计算一下目标函数和
                            parent_obj_record = Q.fitness_calculation(population_list, TOU_mat)  # 计算目标参数需要种群
                            front, rank = Q.non_dominated_sorting(population_size, parent_obj_record)  # 父代的目标函数

                        '''----- selection  -----'''
                        select_pop_idx = Q.tournament_selection(parent_list, population_size, front, rank,
                                                                parent_obj_record)  # 选中的个体和个体编号
                        # 选择过后的种群，作为mating pool， select_idx记录个体的位置

                        '''----- crossover and mutation  -----'''
                        offspring_list = Q.order_crossover(parent_list, select_pop_idx, crossover_rate)
                        # 形成子代, 使用选择的工件进行交叉变异操作
                        offspring_list = Q.Mutation(offspring_list, mutation_rate)  # 变异操作

                        '''------Elite prservation-------'''
                        total_chromosome = copy.deepcopy(parent_list) + copy.deepcopy(offspring_list)  # 父子代合并
                        offspring_obj_record = Q.fitness_calculation(offspring_list, TOU_mat)  # 子代的目标函数
                        total_obj_record = copy.deepcopy(parent_obj_record)
                        for k in range(population_size, 2 * population_size):
                            total_obj_record[k] = offspring_obj_record[k - population_size]  # 传入子代个体的目标函数
                        total_front, total_rank = Q.non_dominated_sorting(population_size * 2,
                                                                          total_obj_record)
                        # 这里的目标函数是offspring——list的，和父代不一样
                        population_list, new_pop_idx = Q.elite_preservation(population_size, total_front,
                                                                            total_obj_record,
                                                                            total_chromosome)  # 父子代总计200——
                        parent_obj_value = [total_obj_record[k] for k in new_pop_idx]  # 下一代的目标函数值
                        key = list(np.arange(population_size))  # 目标函数的指针
                        parent_obj_record = dict(zip(key, parent_obj_value))

                        '''------GVNS based on Q-learning-------'''
                        Q_list = population_list[:2:]  # 取出三个变异
                        Q_obj = [parent_obj_record[k] for k in range(2)]  # 取出目标函数
                        for i in range(3):  # 记录每个目标函数的最大值和最小值
                            sort_temp = sorted(parent_obj_record.items(), key=lambda x: x[1][i])
                            obj_min[i], obj_max[i] = sort_temp[0][1][i], sort_temp[-1][1][i]

                        for i in range(len(Q_list)):  # 对offspring_front里最好的个体进行局部搜索'''
                            improvement = 0  # 每一轮都初始化improvement
                            x = copy.deepcopy(Q_list[i])
                            x_obj_record = Q_obj[i]  # objectives of solution x
                            not_improvement = 0
                            state_1 = random.choice(ACTIONS_1)  # 记录Destruction-construction的初始状态
                            state_2 = random.choice(ACTIONS_2)  # 记录VND的local search的初始状态
                            available_states = copy.deepcopy(ACTIONS_2)  # 记录VND所有的可行状态

                            '''-----原始解和初始邻域搜索的对比 -----'''
                            while not_improvement <= max_episodes:
                                '''------------------Destruction and Construction-----------------'''
                                action_1 = Q.choose_action(state_1, Q_table_1, ACTIONS_1)  # 记录初始动作a, 每个episode更新一次
                                next_state_1 = action_1
                                x_, x2_obj_record = Q.destruction_construction(x, action_1, TOU_mat)  # D-C, x_ 是新解
                                x_weighted_obj = Q.weighted_obj(x_obj_record, obj_min, obj_max)  # 原解x的加权函数
                                x2_weighted_obj = Q.weighted_obj(x2_obj_record, obj_min, obj_max)  # 新解x_的加权函数

                                if x2_weighted_obj < x_weighted_obj:  # 解提升了
                                    improvement += 1
                                    x = copy.deepcopy(x_)  # x_ is better than initial solution x，更换x为x_
                                    x_obj_record = copy.deepcopy(x2_obj_record)  # x的目标值更新为x_
                                    # print('DC improve')

                                else:
                                    not_improvement += 1
                                    if np.random.random() < EPISILON:  # print(i, 'not improved'),被不良取代
                                        x = copy.deepcopy(x_)  # x is better than inital x_，更换x_为x
                                        x_obj_record = copy.deepcopy(x2_obj_record)  # x_的目标值更新为x
                                state_1 = next_state_1  # 更新状态

                                '''-----------------------Variable neighborhood search------------------------'''

                                while available_states:
                                    action_2 = Q.choose_action(state_2, Q_table_2, available_states)
                                    next_state_2 = action_2  # s', 下一个状态
                                    x_ = Q.VNS(action_2, x)

                                    if type(x_[0]) is list:  # 如果解的个数大于1，则取出最好的解
                                        searched_obj_record = Q.fitness_calculation(x_, TOU_mat)
                                        # 计算目标参数需要种群，加工时间交期，功率和盈亏平衡点
                                        weighted_obj = []  # 记录加权目标函数
                                        for _ in range(len(searched_obj_record)):  # 对于每一个个体都计算加权目标
                                            temp = Q.weighted_obj(searched_obj_record[_], obj_min, obj_max)
                                            weighted_obj.append(temp)
                                        x_ = x_[np.argmin(weighted_obj)]  # 把具有最小加权目标函数的解取出来
                                        x2_weighted_obj = min(weighted_obj)  # 把最小的目标函数放进来
                                    else:
                                        x2_obj_record = Q.fitness_calculation([x_], TOU_mat)[0]
                                        x2_weighted_obj = Q.weighted_obj(x2_obj_record, obj_min, obj_max)  # 新解x_的加权函数

                                    x_weighted_obj = Q.weighted_obj(x_obj_record, obj_min, obj_max)  # 原解x的加权函数

                                    if x2_weighted_obj < x_weighted_obj:  # 解提升了
                                        improvement += 1
                                        x = copy.deepcopy(x_)  # x_ is better than initial solution x，更换x为x_
                                        x_obj_record = copy.deepcopy(x2_obj_record)  # x的目标值更新为x_
                                        # print(action_2, 'improve')
                                        break  # 跳出循环

                                    else:
                                        not_improvement += 1
                                        available_states.remove(action_2)
                                        state_2 = action_2

                            Q_list[i] = copy.deepcopy(x)  # 更新Q_list里的个体
                            Q_obj[i] = copy.deepcopy(x_obj_record)
                        population_list[0:2] = Q_list  # 更新个体
                        for k in range(2):
                            parent_obj_record[k] = Q_obj[k]  # 更新目标函数值

                        '''------Comparison with the recorded best population-------'''
                        if n == 0:  # 和历史最优解进行比对，如果第一次循环，则全局最优解就是当前的population——list
                            best_list = copy.deepcopy(population_list)
                            best_record = copy.deepcopy(parent_obj_record)
                        else:  # 如果不是第一次，则进行对比
                            total_list = copy.deepcopy(population_list) + copy.deepcopy(best_list)
                            total_obj = copy.deepcopy(parent_obj_record)
                            for k in range(population_size, 2 * population_size):
                                total_obj[k] = offspring_obj_record[k - population_size]  # 传入子代个体的目标函数
                            best_front, best_rank = Q.non_dominated_sorting(population_size,
                                                                            total_obj)  # 把局部最优和全局最优比对
                            best_list, best_pop_idx = Q.elite_preservation(population_size, best_front, total_obj,
                                                                           total_list)
                            best_record_value = [total_obj[k] for k in best_pop_idx]  # 新的最好的种群
                            key = list(np.arange(population_size))  # 目标函数的指针
                            best_record = dict(zip(key, parent_obj_value))  # 转换成字典型
                            if n % 10 == 0:
                                population_list = copy.deepcopy(best_list)  # 每经过十代就copy一次最好的种群
                                parent_obj_record = best_record  # 每经过十代就copy一次最好的种群

                        time_end = time.time()
                        time_c = time_end - time_start  # 运行所花时间
                        if time_c > CPU_Time:
                            break  # 停止运行
                        print('time cost', time_c, 's')

                    '''-----------------------Q-table 1 and Q-table 2 Output--------------------------'''
                    # path = r'C:\Users\Lipei\Desktop' + '\\Program\\Data\\' + str(Q.Job_num) + \
                    #        '-' + str(Q.stage_num) + '-' + str(machine_num) + '.xlsx'
                    # print('This is', path)
                    # writer = pd.ExcelWriter(path, mode='a', engine='openpyxl',
                    #                         if_sheet_exists='replace')
                    # Q_table_1.to_excel(writer, sheet_name='Q_table_1')
                    # Q_table_2.to_excel(writer, sheet_name='Q_table_2')
                    # writer.save()
                    # writer.close()

                    Pareto_front = best_front[0]  # 记录最优解
                    best_no_repeat = []  # 记录无重复的个体
                    best_record_no_repeat = {}  # 记录无重复的个体
                    i_ = 0  # 用来计数
                    for idx in Pareto_front:
                        if best_list[idx] not in best_no_repeat:
                            best_no_repeat.append(best_list[idx])
                            best_record_no_repeat[i_] = best_record[idx]
                            i_ += 1

                    '''__OUTPUT__'''
                    Name = ['TT', 'TEC', 'CTC', 'Runtime', 'NND'] + [str(i) for i in list(range(Q.Job_num))]
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
