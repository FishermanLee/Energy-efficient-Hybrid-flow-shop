#  Creator: Payne_Lee
#  Institution: NUAA
#  Date: 2023/4/19
import Scheduling
import numpy as np
import random
import copy
import pandas as pd
import itertools

random.seed(42)

global Job_num, Machine_num, stage_num, PT_on_machine, process_time, duedate
global TB_mat, Power_mat, carbon_rate, EA, carbon_price, MAX_ITER, TOU_mode, EPISILON
global ALPHA, LAMBDA
w = [0.4, 0.4, 0.2]  # weights of objectives
'''--------Variable neighborhood search-------------'''


def sorting(Job_sequence):  # 对工件序列按照生产时间进行排序
    Job_sequence = np.array(Job_sequence)
    Job_process_time = np.array([])  # 储存工件的对应生产时间
    for i in Job_sequence:  # 将所有工件对应的时间
        Job_process_time = np.append(Job_process_time, process_time[i])  # 索引
    sorted_job = Job_sequence[(-Job_process_time).argsort()]  # 按照生产时间进行降序排序
    return sorted_job


# Duedate也是numpy数组
def NEH(Job_sequence, obj_idx: int, TOU_mat):  # 该obj_idx可以记录不同的目标函数, TT为1，TEC为2，CTC为3
    sorted_job = sorting(Job_sequence)
    '''----Initialization and calculate the best record----'''
    best_seq = np.array([sorted_job[0]])  # 记录初始的工件，从排序后第一个开始生产
    '''----Greedy insertion----'''
    for job in sorted_job[1:]:  # 在序列完整之前,排列所有可能工件
        best_seq = greedy_insertion(best_seq, job, obj_idx, TOU_mat)
    return best_seq


def greedy_insertion(Job_sequence, insert_job, obj_idx, TOU_mat):
    insert_CHS_list = [np.insert(Job_sequence, i, insert_job) for i in range(len(Job_sequence) + 1)]
    # best_record = {}  # 记录最优的目标函数
    TT_best_record, TEC_best_record, CTC_best_record = 0, 0, 0
    best_seq = []  # 用于记录最优个体
    if obj_idx == 1:  # 目标函数为TT
        for _ in range(len(insert_CHS_list)):
            schedule = Scheduling.Scheduling(Job_num, Machine_num, stage_num, PT_on_machine)  # 用schedule定义该实例
            schedule.Decode(insert_CHS_list[_])  # 进行解码
            TT_temp = TT_calculation(schedule, insert_CHS_list[_])  # 当前
            if _ == 0:  # 如果是第一次评估，将global best初始化为该目标函数值
                TT_best_record = TT_temp
                best_seq = insert_CHS_list[_]
                continue  # 继续下一次循环
            if TT_temp < TT_best_record:
                best_seq = insert_CHS_list[_]
                TT_best_record = TT_temp
    else:  # 目标函数为TEC或CTC
        for _ in range(len(insert_CHS_list)):
            schedule = Scheduling.Scheduling(Job_num, Machine_num, stage_num, PT_on_machine)  # 用schedule定义该实例
            schedule.Decode(insert_CHS_list[_])  # 进行解码
            TEC_temp, CTC_temp = TEC_CTC_calculation(schedule, TOU_mat)
            if _ == 0:  # 如果是第一次评估，将global best初始化为该目标函数值
                TEC_best_record, CTC_best_record = TEC_temp, CTC_temp
                best_seq = insert_CHS_list[_]  # 第一次global best初始化为该个体
                continue  # 继续下一次循环
            # 当前记录
            if obj_idx == 2 and TEC_temp < TEC_best_record:
                best_seq = insert_CHS_list[_]  # 更新最优个体
                TEC_best_record = TEC_temp  # 更新最优记录
                continue  # 省略第三步判断
            elif obj_idx == 3 and CTC_temp < CTC_best_record:
                best_seq = insert_CHS_list[_]  # 更新最优个体
                CTC_best_record = CTC_temp  # 更新最优记录
    # best_record[0] = TT_best_record, best_record[1] = TEC_best_record, best_record[2] = CTC_best_record
    return best_seq


def greedy_insertion_with_limit(Job_sequence, insert_job, obj_idx, TOU_mat):
    insert_CHS_list = [np.insert(Job_sequence, i, insert_job) for i in range(len(Job_sequence) + 1)]
    # best_record = {}  # 记录最优的目标函数
    TT_best_record, TEC_best_record, CTC_best_record = 0, 0, 0
    best_seq = []  # 用于记录最优个体
    no_improvement = 0  # 用于限制搜索次数
    if obj_idx == 1:  # 目标函数为TT
        for _ in range(len(insert_CHS_list)):
            schedule = Scheduling.Scheduling(Job_num, Machine_num, stage_num, PT_on_machine)  # 用schedule定义该实例
            schedule.Decode(insert_CHS_list[_])  # 进行解码
            TT_temp = TT_calculation(schedule, insert_CHS_list[_])  # 当前
            if _ == 0:  # 如果是第一次评估，将global best初始化为该目标函数值
                TT_best_record = TT_temp
                best_seq = insert_CHS_list[_]
                continue  # 继续下一次循环
            if TT_temp < TT_best_record:
                best_seq = insert_CHS_list[_]
                TT_best_record = TT_temp
            else:
                no_improvement += 1
            if no_improvement > MAX_ITER:
                break
    else:  # 目标函数为TEC或CTC
        for _ in range(len(insert_CHS_list)):
            schedule = Scheduling.Scheduling(Job_num, Machine_num, stage_num, PT_on_machine)  # 用schedule定义该实例
            schedule.Decode(insert_CHS_list[_])  # 进行解码
            TEC_temp, CTC_temp = TEC_CTC_calculation(schedule, TOU_mat)
            if _ == 0:  # 如果是第一次评估，将global best初始化为该目标函数值
                TEC_best_record, CTC_best_record = TEC_temp, CTC_temp
                best_seq = insert_CHS_list[_]  # 第一次global best初始化为该个体
                continue  # 继续下一次循环
            # 当前记录
            if obj_idx == 2 and TEC_temp < TEC_best_record:
                best_seq = insert_CHS_list[_]  # 更新最优个体
                TEC_best_record = TEC_temp  # 更新最优记录
                continue  # 省略第三步判断
            elif obj_idx == 2 and TEC_temp >= TEC_best_record:
                no_improvement += 1
                continue
            elif obj_idx == 3 and CTC_temp < CTC_best_record:
                best_seq = insert_CHS_list[_]  # 更新最优个体
                CTC_best_record = CTC_temp  # 更新最优记录
            elif obj_idx == 3 and CTC_temp >= CTC_best_record:
                no_improvement += 1
            if no_improvement > MAX_ITER:
                break
    # best_record[0] = TT_best_record, best_record[1] = TEC_best_record, best_record[2] = CTC_best_record
    return best_seq


# def local_search(CHS, obj_idx, TOU_mat):  # 如果搜索次数太多，也许需要限制
#     pi = copy.deepcopy(CHS)
#     job = random.choice(CHS)
#     pi.remove(job)
#     best_CHS = greedy_insertion_with_limit(pi, job, obj_idx, TOU_mat)
#     return best_CHS.tolist()


def critical_path(CHS):  # 识别关键路径, 每次解码的方式不同，关键工序不同
    critical_job = []  # 记录critical_job
    schedule = Scheduling.Scheduling(Job_num, Machine_num, stage_num, PT_on_machine)  # 用schedule定义该实例
    schedule.Decode(CHS)  # 进行解码
    Job_end = np.array([schedule.Jobs[job].C_max for job in range(Job_num)], dtype=int)  # 每个工件的完工时间
    last_job = int(Job_end.argmax())  # 选出最后一个工件
    critical_job.append(last_job)
    stage = stage_num - 1  # 从最后一个阶段往前倒数
    while stage >= 0:  # 倒序循环阶段
        '''-------读取关键路径上的工件，和加工该工件的机器信息'''
        machine_idx = int(schedule.Jobs[last_job].job_on_machine[stage])  # 加工该工件的机器序号
        machine = schedule.Machine_list[stage][machine_idx]  # 记录了机器所有的信息
        job_on_machine = machine.job_on_machine  # 该机器上加工的工件
        idx = np.where(job_on_machine == last_job)[0][0]  # 找到该工件在机器上的顺序，如第4个
        job_start, job_end = machine.start_time, machine.completion_time
        while idx > 0:  # 当该工件不是第一个工件时
            if job_start[idx] - job_end[idx - 1] == 0:  # 上一工件是紧接着产的，说明是关键工序
                last_job = job_on_machine[idx - 1]
                critical_job.append(last_job)  # 将上一工件加入关键路径
                idx -= 1  # 往前跳一个工件
            elif job_start[idx] - job_end[idx - 1] > 0:
                stage -= 1  # 跳入下一阶段
                break
        if idx == 0 and stage >= 0:  # 如果idx等于机器上第一个工件，转入上一个阶段
            stage -= 1
    return critical_job


def N1_critical_swap(parent):
    critical_job = critical_path(parent)  # 得到要交换的关键工作
    non_critical_job = [x for x in parent if x not in critical_job]
    offspring = copy.deepcopy(parent)  # 可以进行多点交叉
    idx_1 = offspring.index(np.random.choice(critical_job))
    idx_2 = offspring.index(np.random.choice(non_critical_job))
    offspring[idx_1], offspring[idx_2] = offspring[idx_2], offspring[idx_1]  # 交换基因
    return offspring


def N2_critical_insertion(parent):
    critical_job = critical_path(parent)  # 得到要交换的关键工作
    non_critical_job = [x for x in parent if x not in critical_job]
    offspring = copy.deepcopy(parent)
    idx_1 = offspring.index(np.random.choice(critical_job))
    idx_2 = offspring.index(np.random.choice(non_critical_job))
    temp = copy.deepcopy(offspring)
    del (offspring[idx_1])  # 移动该位点插入第二个
    offspring.insert(idx_2, temp[idx_1])
    return offspring


def N3_critical_reversion(parent):
    critical_job = critical_path(parent)  # 得到要交换的关键工作
    non_critical_job = [x for x in parent if x not in critical_job]
    offspring = copy.deepcopy(parent)
    idx_1 = offspring.index(np.random.choice(critical_job))
    idx_2 = offspring.index(np.random.choice(non_critical_job))
    offspring[min(idx_1, idx_2):max(idx_1, idx_2)] = offspring[min(idx_1, idx_2):max(idx_1, idx_2)][::-1]  # 倒换
    return offspring


def destruction_construction(CHS: list, k, TOU_mat):  # k指代从抽取的数量
    pi = copy.deepcopy(CHS)
    best_record = {}  # 用于存储最优的目标函数
    pi_repair = []  # used to store reinserted jobs
    for i in range(k):  # 形成
        job = random.choice(pi)
        pi.remove(job)
        pi_repair.append(job)
    # 是否要对pi_repair里的工序进行排序
    pi_repair = sorting(pi_repair)
    for job in pi_repair:
        no_improvement = 0
        improve = False
        best_record = {}  # 用于存储最优的目标函数
        insert_CHS_list = [np.insert(pi, i, job) for i in range(len(pi) + 1)]  # 所有可能的插入
        for _ in range(len(insert_CHS_list)):  # 没有改进的次数过多以及没有改进的时候才允许继续搜索
            temp_record = fitness_calculation([insert_CHS_list[_]], TOU_mat)  # 记录个体的目标函数,{0：TT，TEC, CTC}
            compare_record = {}  # 用于存储best和temp两个个体的目标函数
            if _ == 0:  # 如果是第一次评估，将global best初始化为该目标函数值
                best_record = temp_record
                pi = insert_CHS_list[_]  # 更新pi为插入后的个体
                continue  # 继续下一次循环
            else:
                compare_record['best'] = best_record[0]
                compare_record['temp'] = temp_record[0]
                if dominate('temp', 'best', compare_record):  # 多目标对比
                    best_record = temp_record
                    pi = insert_CHS_list[_]
                    improve = True
                    break  # 获得提升，跳出for循环
                else:
                    no_improvement += 1
                    if no_improvement > MAX_ITER:
                        break

    return pi.tolist(), best_record[0]


def N4_threepoint_heuristic(parent):  # 假设有三个位点全排列
    of_list = []
    for i in range(6):
        of_list.append(copy.deepcopy(parent))  # deepcopy防止同地址引用
    rand1 = random.randint(0, len(parent) - 3)
    segment = (of_list[0][rand1: rand1 + 3])
    permutation = list(itertools.permutations(segment))
    for i in range(0, 6):
        for j in range(rand1, rand1 + 3):
            of_list[i][j] = permutation[i][j - rand1]
    return of_list


def N5_semitwo_crossover(parent):
    of_list = []
    cutting_point = sorted(random.sample(range(1, Job_num), 2))  # 产生不重复的三个随机数作为切点
    seg_1 = parent[:cutting_point[0]]  # 被切成三段
    seg_2 = parent[cutting_point[0]:cutting_point[1]]
    seg_3 = parent[cutting_point[1]:]
    permutation = list(itertools.permutations([seg_1, seg_2, seg_3]))  # 对三段进行全排列
    for i in range(len(permutation)):  # 产生后代列表
        of_list.append(permutation[i][0] + permutation[i][1] + permutation[i][2])
    return of_list


def VNS(state, CHS):
    searched_solution = []
    if state == 'N1_Cswap':
        searched_solution = N1_critical_swap(CHS)
    elif state == 'N2_CInsr':
        searched_solution = N2_critical_insertion(CHS)  # [[],[],[]] list型
    elif state == 'N3_CInv':
        searched_solution = N3_critical_reversion(CHS)  # [[],[],[]] list型
    elif state == 'N4_threepoint_heuristic':
        searched_solution = N4_threepoint_heuristic(CHS)  # [[],[],[]] list型
    elif state == 'N5_semitwo_crossover':
        searched_solution = N5_semitwo_crossover(CHS)  # [[],[],[]] list型

    return searched_solution


def weighted_obj(obj_record: list, obj_min: dict, obj_max: dict):
    weighted = 0
    for j in range(3):
        stand_obj = (obj_record[j] - obj_min[j]) / (
                obj_max[j] - obj_min[j])  # 标准化过程
        weighted += w[j] * stand_obj
    return weighted


'''--------NSGA-II-------------'''


def Initialization(Population_size: int, TOU_mat):  # 解的初始化
    population_list = []
    for obj_idx in range(3):
        job_sequence = np.arange(Job_num).tolist()
        CHS = NEH(job_sequence, obj_idx, TOU_mat)
        population_list.append(CHS.tolist())

    for i in range(3, Population_size):
        CHS = []
        for job in range(Job_num):
            CHS.append(job)
        random.shuffle(CHS)
        population_list.append(CHS)

    return population_list


'''--------fitness value(calculate TWT，TEC AND CTC-------------'''


def TT_calculation(schedule: Scheduling.Scheduling, job_sequence):  # 传入已经解码的染色体
    Job_end: np.ndarray = np.array([schedule.Jobs[job].C_max for job in job_sequence])
    # 每个工件的完工时间，due date和Job_end对应，都是0-job_num的大小
    sorted_duedate = np.array([duedate[i] for i in job_sequence])
    tardiness_record: np.ndarray = Job_end - sorted_duedate  # Tardiness的定义是完工时间-交货期
    tardiness_record = np.where(tardiness_record < 0, 0, tardiness_record)  # 将小于0的元素替换为0
    TT = tardiness_record.sum()  # 总拖期的定义
    return TT


def TOU(TOU_mode: int):  # 生产TOU_mat
    TOU_mat = np.zeros(9999)  # 分时电价矩阵,9999代指无穷大的数
    if TOU_mode == 1:
        '''------1.NO TOU------'''
        for i in range(9999):  # 对分时电价进行设定
            TOU_mat[i] = 600  # 无分时电价
    elif TOU_mode == 2:
        '''------2.Low off-peak------'''
        for i in range(9999):  # 对分时电价进行设定
            k = i % 24  # 周期计数器
            if 0 <= k < 10:  # 开始时间设为8点
                TOU_mat[i] = 200
            elif 10 <= k < 11:
                TOU_mat[i] = 1000
            elif 11 <= k < 17:
                TOU_mat[i] = 600
            elif 17 <= k < 22:
                TOU_mat[i] = 1000
            elif 22 <= k < 24:
                TOU_mat[i] = 600
    elif TOU_mode == 3:
        '''------3.Low on-peak------'''
        for i in range(9999):  # 对分时电价进行设定
            k = i % 24  # 周期计数器
            if 0 <= k < 8:  # 开始时间设为8点
                TOU_mat[i] = 250
            elif 8 <= k < 11:
                TOU_mat[i] = 750
            elif 11 <= k < 17:
                TOU_mat[i] = 600
            elif 17 <= k < 22:
                TOU_mat[i] = 750
            elif 22 <= k < 24:
                TOU_mat[i] = 600
    elif TOU_mode == 4:
        '''------4.Spring and fall------'''
        for i in range(9999):  # 对分时电价进行设定
            k = i % 24  # 周期计数器
            if 0 <= k < 8:  # 开始时间设为8点
                TOU_mat[i] = 250
            elif 8 <= k < 11:
                TOU_mat[i] = 1000
            elif 11 <= k < 17:
                TOU_mat[i] = 600
            elif 17 <= k < 22:
                TOU_mat[i] = 1000
            elif 22 <= k < 24:
                TOU_mat[i] = 600
    elif TOU_mode == 5:
        '''------5.Summer------'''
        for i in range(9999):  # 对分时电价进行设定
            k = i % 24  # 周期计数器
            if 0 <= k < 8:  # 开始时间设为8点
                TOU_mat[i] = 250
            elif 8 <= k < 10:
                TOU_mat[i] = 1000
            elif 10 <= k < 11:
                TOU_mat[i] = 1200
            elif 11 <= k < 14:
                TOU_mat[i] = 600
            elif 14 <= k < 15:
                TOU_mat[i] = 1200
            elif 15 <= k < 18:
                TOU_mat[i] = 600
            elif 18 <= k < 22:
                TOU_mat[i] = 1000
            elif 22 <= k < 24:
                TOU_mat[i] = 600
    elif TOU_mode == 6:
        '''------6.Winter------'''
        for i in range(9999):  # 对分时电价进行设定
            k = i % 24  # 周期计数器
            if 0 <= k < 8:  # 开始时间设为8点
                TOU_mat[i] = 250
            elif 8 <= k < 9:
                TOU_mat[i] = 1000
            elif 9 <= k < 11:
                TOU_mat[i] = 1200
            elif 11 <= k < 17:
                TOU_mat[i] = 600
            elif 17 <= k < 18:
                TOU_mat[i] = 1000
            elif 18 <= k < 20:
                TOU_mat[i] = 1200
            elif 20 <= k < 22:
                TOU_mat[i] = 1000
            elif 22 <= k < 24:
                TOU_mat[i] = 600
    return TOU_mat


def TEC_CTC_calculation(schedule: Scheduling.Scheduling, TOU_mat):
    energy_sum = 0  # 总能耗
    TEC = 0  # 总电力成本
    for S_k in range(stage_num):  # 对每一个阶段
        for M_i in range(Machine_num[S_k]):  # 每一阶段的每一台机器
            Machine = schedule.Machine_list[S_k][M_i]
            x_list = np.ones(schedule.makespan)  # x =1, y=0指待机
            y_list = np.zeros(schedule.makespan)  # X,Y用来记录机器的状态,一开始均设为idle
            for t in range(len(Machine.start_time)):  # 加工的机器y设为1,代表开工
                y_list[int(Machine.start_time[t]):int(Machine.completion_time[t])] = 1
            last_oper = np.insert(Machine.completion_time, 0, 0)
            # 加上dummy job端点0，上一操作的完成时间，统一使用左端点计数,如[0,9,13,16]
            next_oper = np.append(Machine.start_time, schedule.makespan)
            # 加上dummy job最后的完工时间，下一操作的开始时间，如[0,12,14,16]
            inter_arrival = next_oper - last_oper  # 两个操作之间的间隔时间
            '''================没有重启机器的时间=========================='''
            # for i in range(len(inter_arrival)):  # Power_down operation
            #     if inter_arrival[i] >= max(TB_mat[S_k][M_i], 2):  # 即两个操作之间的时间间隔大于盈亏平衡点和开关机时间，则关机
            #         x_list[last_oper[i].astype(int):(next_oper[i].astype(int))] = 0  # 在上一操作完成后关掉机器
            #         y_list[next_oper[i].astype(int) - 1] = 1  # 重启机器
            energy_per_time = x_list * y_list * Power_mat[S_k][M_i][0] + x_list * (1 - y_list) * \
                              Power_mat[S_k][M_i][1] + y_list * (1 - x_list) * Power_mat[S_k][M_i][2]
            # 记录该机器每个时间点的能耗
            TEC += np.dot(energy_per_time / 10, TOU_mat[:len(energy_per_time)])  # TEC加上每个机器的消耗费用
            energy_sum += np.sum(energy_per_time)  # 总能耗
    CTC = (carbon_rate * (energy_sum / 10) - EA) * carbon_price  # 碳排放权交易成本
    return TEC, CTC


def fitness_calculation(Population_list, TOU_mat):  # 计算某一染色体的目标函数
    chroms_obj_record = {}  # record each chromosome objective values as chromosome
    cnt = 0  # 个体计数器
    for C_i in Population_list:  # 对种群的每一个个体进行解码，有重复的C_i
        schedule = Scheduling.Scheduling(Job_num, Machine_num, stage_num, PT_on_machine)  # 用schedule定义该实例
        schedule.Decode(C_i)  # 进行解码
        '''------TT CALCULATION------'''
        TT = TT_calculation(schedule, C_i)
        '''------TEC and CTC CALCULATION------'''
        TEC, CTC = TEC_CTC_calculation(schedule, TOU_mat)
        chroms_obj_record[cnt] = [TT, TEC, CTC]  # 三个目标函数代入
        cnt += 1
    return chroms_obj_record


def tournament_selection(population_list, population_size, front, rank,
                         chroms_obj_record):  # tournament selection is divised,
    new_pop_idx = []  # 记录解的编号
    p_1 = np.random.permutation(population_size)
    p_2 = np.random.permutation(population_size)
    p_3 = np.random.permutation(population_size)  # the size of tournament is 3, generate 3 permutations
    for i in range(population_size):  # 三元锦标赛选择
        a = distance_comparison_operator(p_1[i], p_2[i], rank, front, chroms_obj_record)  # p_1[i]和p_2[i]比较
        b = distance_comparison_operator(a, p_3[i], rank, front, chroms_obj_record)  # 返回的是个体的序号
        new_pop_idx.append(b)

    return new_pop_idx


'''--------Genetic operation-------------'''


def Mutation(parent_list, mutation_rate):  # 传入整个群
    pt_list = copy.deepcopy(parent_list)
    of_list = []
    for _ in pt_list:
        if random.random() <= mutation_rate:
            cutting_point = sorted(random.sample(range(1, len(_)), 2))  # 产生不重复的两个随机数作为切点
            seg_1 = _[:cutting_point[0]]  # 被切成三段
            seg_2 = _[cutting_point[0]:cutting_point[1]]
            seg_3 = _[cutting_point[1]:]
            permutation = [seg_1 + seg_3 + seg_2, seg_3 + seg_2 + seg_1, seg_2 + seg_1 + seg_3]
            of_list.append(random.choice(permutation))  # choose randomly
        else:
            of_list.append(_)
    return of_list


def order_crossover(population_list, select_pop_idx, crossover_rate):
    pt_list = copy.deepcopy(population_list)
    of_list = []
    ox_idx = []  # 记录发生变异的个体编号
    for m in range(int((len(select_pop_idx)) / 2)):  # 迭代次数为种群的一般
        parent_1 = pt_list[select_pop_idx[2 * m]][:]  # 遍历所有父代
        parent_2 = pt_list[select_pop_idx[2 * m + 1]][:]
        if random.random() < crossover_rate:
            len_seg = np.random.randint(3, Job_num - 1)
            i = random.randint(0, len(parent_1) - len_seg)  # parent1的cutting point,假设交叉长度为3
            j = random.randint(0, len(parent_2) - len_seg)  # parent2的cutting point
            seg_lst1 = [item for item in parent_1 if item not in parent_2[j:j + len_seg]]
            seg_lst2 = [item for item in parent_2 if item not in parent_1[i:i + len_seg]]
            child_1 = seg_lst2[0:i] + parent_1[i:i + len_seg] + seg_lst2[i:]
            child_2 = seg_lst1[0:j] + parent_2[j:j + len_seg] + seg_lst1[j:]
            of_list.extend((child_1, child_2))  # append child chromosome to offspring list
        else:  # 如果不发生突变，则原样保留
            of_list.extend((parent_1, parent_2))
    return of_list  # 生成了100个后代


def update_obj_record(population_list, chroms_obj_record, pop_idx, TOU_mat):  # 更新改变的个体的目标函数
    for i in pop_idx:
        chroms_obj_record[i] = fitness_calculation(population_list[i], TOU_mat)
    return chroms_obj_record  # 更新目标函数


def dominate(p, q, chroms_obj_record):  # p，q是个体的编号
    if (chroms_obj_record[p][0] <= chroms_obj_record[q][0] and chroms_obj_record[p][1] <= chroms_obj_record[q][1]
        and chroms_obj_record[p][2] <= chroms_obj_record[q][2]) and (
            chroms_obj_record[p][0] != chroms_obj_record[q][0] and chroms_obj_record[p][1] != chroms_obj_record[q][1]
            and chroms_obj_record[p][2] != chroms_obj_record[q][2]):
        return 1
    else:
        return 0


def non_dominated_sorting(population_size, chroms_obj_record):
    s, n = {}, {}
    front, rank = {}, {}
    front[0] = []
    iter_range = population_size  # 如果不是亲子代混合，就是100+x个需要比较
    for p in range(iter_range):
        s[p] = []
        n[p] = 0
        for q in range(iter_range):
            if dominate(p, q, chroms_obj_record):  # if p dominates q for three objectives
                if q not in s[p]:
                    s[p].append(q)  # s[p] is the set of solutions dominated by p
            elif dominate(q, p, chroms_obj_record):  # if q dominates p for three objectives
                n[p] = n[p] + 1  # n[p] is the set of solutions dominating p, 3 obj
        if n[p] == 0:
            rank[p] = 0  # p belongs to front 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []  # Used to store the members of the next front
        for p in front[i]:
            for q in s[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front[i] = Q

    del front[len(front) - 1]
    return front, rank


def calculate_crowding_distance(front, chroms_obj_record):
    distance = {m: 0 for m in front}
    for o in range(3):  # we have three obj
        obj = {m: chroms_obj_record[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)  # 用目标函数进行排列
        distance[sorted_keys[0]] = distance[sorted_keys[len(front) - 1]] = 999999999999
        for i in range(1, len(front) - 1):  # 非边界个体
            if len(set(obj.values())) == 1:  # 相同个体的数目
                distance[sorted_keys[i]] = distance[sorted_keys[i]]
            else:  # 对每个目标都要加总
                distance[sorted_keys[i]] += distance[sorted_keys[i]] + (
                        obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) / (
                                                    obj[sorted_keys[len(front) - 1]] - obj[sorted_keys[0]])
    # The overall crowding distance is the sum of distance corresponding to each objective
    return distance


def distance_comparison_operator(chroms_1, chroms_2, rank, front, chroms_obj_record):  # 用在最后亲子代合并比较
    res = 0
    if rank[chroms_1] < rank[chroms_2]:
        res = chroms_1
    elif rank[chroms_1] > rank[chroms_2]:
        res = chroms_2
    elif rank[chroms_1] == rank[chroms_2]:
        distance = calculate_crowding_distance(front[rank[chroms_1]], chroms_obj_record)
        if distance[chroms_1] > distance[chroms_2]:  # 拥挤度距离越大越好
            res = chroms_1
        else:
            res = chroms_2
    return res


def elite_preservation(population_size, total_front, total_obj_record,
                       total_chromosome):  # tournament selection is divised,
    N = 0  # 记录下一代中的数量
    new_pop_idx = []  # new——pop记录原种群里的index
    while N < population_size:
        for i in range(len(total_front)):  # i是rank的层数
            N += len(total_front[i])
            if N > population_size:  # i是rank的层数
                distance = calculate_crowding_distance(total_front[i], total_obj_record)
                sorted_cdf = sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop_idx) == population_size:
                        break
                    new_pop_idx.append(j)
                break
            else:
                new_pop_idx.extend(total_front[i])
    pop_list = []
    for n in new_pop_idx:
        pop_list.append(total_chromosome[n])

    return pop_list, new_pop_idx


'''--------Q-learning-------------'''


def build_q_table(ACTIONS):
    q_table = pd.DataFrame(
        np.zeros((len(ACTIONS), len(ACTIONS))),  # q_table initial values
        index=ACTIONS, columns=ACTIONS, dtype=np.float64)
    return q_table


def choose_action(state, q_table, available_states):
    # action selection, observation is a state
    if np.random.uniform() > EPISILON:
        # choose best action
        state_action = q_table.loc[state, :]  # actions of the state s
        available_q = state_action[available_states]
        # some actions may have the same value, randomly choose on in these actions
        action_name = random.choice(available_q[available_q == np.max(available_q)].index)  # 返回策略的名字
    else:
        # choose random action
        action_name = random.choice(available_states)  # 返回策略的名字
    return action_name


def learn(q_table, s, a, r, s_):  # state s, action a, reward r, next stage s_
    q_predict = q_table.loc[s, a]
    q_target = r + LAMBDA * q_table.loc[s_, :].max()  # next state is not terminal
    return ALPHA * (q_target - q_predict)  # update


def Q_learning(x0, gbest, x0_obj_record, gbest_obj_record, state,
               available_actions, Q_table, TOU_mat, obj_min, obj_max, mode):
    compare_record = {}  # 用于存储best和temp两个个体的目标函数
    action = choose_action(state, Q_table, available_actions)  # 记录初始动作a, 每个episode更新一次
    next_state = action  # 更新index，s'
    if mode == 'destruction_construction':
        x_, x_obj_record = destruction_construction(x0, action, TOU_mat)  # D-C
    else:
        x_, x_obj_record = VNS(x0, action, TOU_mat)
    compare_record['x0'] = x0_obj_record  # list
    compare_record['x_'] = x_obj_record[0]  # dict
    compare_record['gbest'] = gbest_obj_record  # list
    if dominate('x_', 'x0', compare_record):  # if x_ dominates x0 and
        # this is not the first loop
        x0 = x_
        x0_obj_record = x_obj_record[0]  # Update the x
        reward = 0
        for i in range(3):  # we have 3 objectives
            reward += obj_min[i] / compare_record['x0'][i] * \
                      (compare_record['x0'][i] - compare_record['x_'][i]) / (obj_max[i] - obj_min[i])
        Q_table.loc[state, action] = learn(Q_table, state, action,
                                           reward, next_state)
    if dominate('x_', 'gbest', compare_record):
        gbest = x_
        gbest_obj_record = x_obj_record[0]
    return x0, x0_obj_record, gbest, gbest_obj_record
