#  Creator: Payne_Lee
#  Institution: NUAA
#  Date: 2023/5/12

import pandas as pd
import numpy as np
import math
import QVNS_NSGA_II as Q


def distance_calculation(data):  # data是一个包含所有点目标标准化函数的表
    distance_mat = []
    for i in range(len(data)):  # 对于每一个点i
        distance_temp = []  # 记录每个点i与其他店的距离
        for j in range(len(data)):
            if i == j:
                distance_temp.append(99999)  # 相同点不可选
            else:
                distance_temp.append(sum(abs(data.iloc[i, 0:3] - data.iloc[j, 0:3])))  # 记录点j和其他点的距离
        distance_mat.append(min(distance_temp))  # 最小的距离为i解与其他解的距离
    return distance_mat


def Coverage_metric(dominate_A, dominated_B):  # 输入两个记录
    C_A_B = pd.DataFrame(
        np.zeros((5, 5)), dtype=np.float64
    )  # 记录覆盖比例

    for num_Q in range(5):
        for num_N in range(5):
            A = dominate_A[num_Q]  # A 是支配B的
            B = dominated_B[num_N]  # B 是被A支配的
            num_dominated_B = 0
            for i in range(len(B)):
                for j in range(len(A)):
                    if ((A.loc[j] <= B.loc[i])['TT']) and ((A.loc[j] <= B.loc[i])['TEC']) and (
                            (A.loc[j] <= B.loc[i])['CTC']):
                        num_dominated_B += 1
                        print(str(num_dominated_B), 'is dominated')
                        break

            C_A_B.iloc[num_Q, num_N] = num_dominated_B / len(B)  # 被支配的比例
            print('num_dominated_B', str(len(B)))

    return C_A_B


def cut(Data: pd.DataFrame):
    cut_data = [pd.DataFrame() for _ in range(5)]  # 记录五个组的数据
    zero_loc = []
    cnt = 0

    for m in Data.index.tolist():  # Data[i]是某个实例，而index就是所有轮数的序号
        if m == 0:
            zero_loc.append(cnt)
        cnt += 1
    zero_loc.append(len(Data))  # 最后一个点

    for i_ in range(5):
        cut_data[i_] = Data.iloc[zero_loc[i_]:zero_loc[i_ + 1]]  # df3每次根据0出现的点进行截断

    return cut_data


if __name__ == '__main__':
    read_path_1 = r'C:\\Users\\Lipei\\Desktop\\Source Code\\Output\\QVNS-NSGA-II.xlsx'
    read_path_2 = r'C:\\Users\\Lipei\\Desktop\\Source Code\\Output\\NSGA-II.xlsx'
    read_path_3 = r'C:\\Users\\Lipei\\Desktop\\Source Code\\Output\\Jaya.xlsx'
    read_path_4 = r'C:\\Users\\Lipei\\Desktop\\Source Code\\Output\\MOEAD.xlsx'
    df = 2
    Instance_name = {}  # 记录所有算例的名字，{1:10-3-3， 2:10-3-4...}
    i = 0
    for Job in [10, 20, 50]:
        for State in [3, 5]:
            for Num_machine in [3, 4, 6]:
                scale = str(Job) + '-' + str(State) + '-' + str(Num_machine)
                Instance_name[i] = (scale)  # 实例的名字
                i += 1

    obj_min = {}  # 记录最小值
    obj_max = {}
    size = 27  # 实例的个数
    Data = []  # 存储每个实例的库

    wrt_path = r'C:\\Users\\Lipei\\Desktop\\Program\\Data\\Orthogonal Array.xlsx'
    Array = pd.DataFrame(pd.read_excel(wrt_path, index_col=[0]))

    index_1 = ['QVNS-NSGA-II'] * len(Instance_name)
    index_2 = ['NSGA-II'] * len(Instance_name)
    index_3 = ['Jaya'] * len(Instance_name)
    index_4 = ['MOEAD'] * len(Instance_name)
    index_algorithm = index_1 + index_2 + index_3 + index_4
    index_instance = list(Instance_name.values()) * 4
    '''----------------------------------记录所有SM的数据----------------------------------------'''
    Col_name_1 = ['SM_1', 'SM_2', 'SM_3', 'SM_4', 'SM_5', 'SM_AVG', 'SM_VAR']
    SM_record = pd.DataFrame(np.zeros((len(index_instance), len(Col_name_1))), index=[index_algorithm, index_instance],
                             columns=Col_name_1)  # 记录SM的数值

    '''----------------------------------记录所有Coverage metric的数据----------------------------------------'''
    index_name = list(Instance_name.values()) * 2
    index_AVG = ['CM_AVG'] * len(Instance_name) + ['CM_STD'] * len(Instance_name)
    Col_name_2 = ['C_Q_N', 'C_N_Q', 'C_Q_J', 'C_J_Q', 'C_Q_M', 'C_M_Q']
    CM_record = pd.DataFrame(np.zeros((len(index_name), len(Col_name_2))), index=[index_AVG, index_name],
                             columns=Col_name_2)
    # 记录SM的数值

    '''----------------------------------记录所有Number of Pareto solutions的数据------------------------------------'''
    Col_name_3 = ['NND_1', 'NND_2', 'NND_3', 'NND_4', 'NND_5', 'NND_AVG', 'NND_VAR']
    NND_Record = pd.DataFrame(np.zeros((len(index_instance), len(Col_name_3))), index=[index_algorithm, index_instance],
                              columns=Col_name_3)  # 记录SM的数值

    for _ in range(5):
        Array['SM_' + str(_)] = 0  # 增加新列记录SM值

    for i in range(len(Instance_name)):
        sheet_name = Instance_name[i]  # 读的页码
        OBJ_MIN = pd.DataFrame()
        OBJ_MAX = pd.DataFrame()

        QVNS_NSGA2_DATA = pd.read_excel(read_path_1, sheet_name=sheet_name, index_col=[0])
        QVNS_NSGA2_DATA = QVNS_NSGA2_DATA.loc[~(QVNS_NSGA2_DATA == 0).all(axis=1)]
        QVNS_NSGA2_DATA = QVNS_NSGA2_DATA.loc[:, ['TT', 'TEC', 'CTC', 'NND']]
        OBJ_MIN[0] = QVNS_NSGA2_DATA.min()
        OBJ_MAX[0] = QVNS_NSGA2_DATA.max()

        NSGA2_DATA = pd.read_excel(read_path_2, sheet_name=sheet_name, index_col=[0])
        NSGA2_DATA = NSGA2_DATA.loc[~(NSGA2_DATA == 0).all(axis=1)]
        NSGA2_DATA = NSGA2_DATA.loc[:, ['TT', 'TEC', 'CTC', 'NND']]
        OBJ_MIN[1] = NSGA2_DATA.min()
        OBJ_MAX[1] = NSGA2_DATA.max()

        Jaya_DATA = pd.read_excel(read_path_3, sheet_name=sheet_name, index_col=[0])
        Jaya_DATA = Jaya_DATA.loc[~(Jaya_DATA == 0).all(axis=1)]
        Jaya_DATA = Jaya_DATA.loc[:, ['TT', 'TEC', 'CTC', 'NND']]
        OBJ_MIN[2] = Jaya_DATA.min()
        OBJ_MAX[2] = Jaya_DATA.max()

        MOEAD_DATA = pd.read_excel(read_path_4, sheet_name=sheet_name, index_col=[0])
        MOEAD_DATA = MOEAD_DATA.loc[~(MOEAD_DATA == 0).all(axis=1)]
        MOEAD_DATA = MOEAD_DATA.loc[:, ['TT', 'TEC', 'CTC', 'NND']]
        OBJ_MIN[3] = MOEAD_DATA.min()
        OBJ_MAX[3] = MOEAD_DATA.max()

        '''-----------按实验次数分组---------------'''
        QVNS_data = cut(QVNS_NSGA2_DATA)
        NSGA_data = cut(NSGA2_DATA)
        Jaya_data = cut(Jaya_DATA)  # 记录五个组的数据
        MOEA_data = cut(MOEAD_DATA)  # 记录五个组的数据
        Data_base = [QVNS_data, NSGA_data, Jaya_data, MOEA_data]  # 便于迭代
        Algorithm = ['QVNS-NSGA-II', 'NSGA-II', 'Jaya', 'MOEAD']

        '''-----------Normalization-------------'''
        obj_min = OBJ_MIN.min(axis=1)[:3]
        obj_max = OBJ_MAX.min(axis=1)[:3]  # 求出最大值和最小值
        diff = (obj_max - obj_min).to_numpy()
        # obj_min = np.array([obj_min[i] for i in range(3)])  # [1,2,3] 1*3数组
        # obj_max = np.array([obj_max[i] for i in range(3)])
        # diff = (obj_max - obj_min).to_numpy()[:3]  # 差值

        for i_ in range(4):
            for j_ in range(5):  # 共有5轮
                Data_base[i_][j_].iloc[:, 0:3] = (Data_base[i_][j_].iloc[:, 0:3] - obj_min) / diff  # 对应位相减再除以差值,标准化
                df3 = Data_base[i_][j_]  # 第j_个表
                df3_dist = distance_calculation(df3)  # 每一个解的最小距离
                df3.loc[:, 'd_i'] = df3_dist
                mean_distance = df3['d_i'].mean()  # 平均空间距离
                SM_tmp = math.sqrt(((df3['d_i'] - mean_distance) ** 2).mean())  # SM值
                SM_record.loc[Algorithm[i_], sheet_name]['SM_' + str(j_ + 1)] = SM_tmp  # SM赋值
                NND_Record.loc[Algorithm[i_], sheet_name]['NND_' + str(j_ + 1)] = df3['NND'].mean()  # NND赋值

            SM_record.loc[Algorithm[i_], sheet_name]['SM_AVG'] = \
                SM_record.loc[Algorithm[i_], sheet_name][0:5].mean()
            SM_record.loc[Algorithm[i_], sheet_name]['SM_VAR'] = \
                SM_record.loc[Algorithm[i_], sheet_name][0:5].var()

        C_Q_N = Coverage_metric(QVNS_data, NSGA_data)  # 记录覆盖比例, 5*5矩阵
        C_N_Q = Coverage_metric(NSGA_data, QVNS_data)
        C_Q_J = Coverage_metric(QVNS_data, Jaya_data)
        C_J_Q = Coverage_metric(Jaya_data, QVNS_data)
        C_Q_M = Coverage_metric(QVNS_data, MOEA_data)
        C_M_Q = Coverage_metric(MOEA_data, QVNS_data)

        cnt = 0
        CM_data = [C_Q_N, C_N_Q, C_Q_J, C_J_Q, C_Q_M, C_M_Q]
        CM_name = ['C_Q_N', 'C_N_Q', 'C_Q_J', 'C_J_Q', 'C_Q_M', 'C_M_Q']
        for cnt in range(len(CM_data)):  # 共有6个位置
            CM_record.loc['CM_AVG', sheet_name][CM_name[cnt]] = CM_data[cnt].mean().mean()
            CM_record.loc['CM_STD', sheet_name][CM_name[cnt]] = CM_data[cnt].std().mean()  # 四个位置的信息

    wrt_path_2 = r'C:\\Users\\Lipei\\Desktop\\Program\\Outcome_2023\\Comparison result.xlsx'
    sv_path = pd.ExcelWriter(wrt_path_2, mode='a', engine='openpyxl', if_sheet_exists='new')
    SM_record.to_excel(sv_path, sheet_name='SM')
    CM_record.to_excel(sv_path, sheet_name='CM')
    NND_Record.to_excel(sv_path, sheet_name='NND')
    sv_path.save()
    sv_path.close()
