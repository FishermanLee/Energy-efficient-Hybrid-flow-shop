#  Creator: Payne_Lee
#  Institution: NUAA
#  Date: 2023/4/18
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Item:  # 定义工件和机器的属性，包括每个阶段开始时间，结束时间，加工的机器
    def __init__(self):
        self.start_time = np.array([], dtype=int)  # 定义工件在每个阶段的开始时间
        self.completion_time = np.array([], dtype=int)  # 定义工件在每个阶段的结束时间
        self.job_on_machine = np.array([], dtype=int)  # 定义加工工件的机器,和机器加工的工件
        self.time_list = np.array([], dtype=int)  # 定义加工时间序列
        self.C_max = 0  # 记录工件每阶段的完工时间

    def update(self, start, end, on, t):  # 类中的函数，更新状态
        self.start_time = np.append(self.start_time, start)  # 开始，在类里引用外部传来的参数，用self代替
        self.completion_time = np.append(self.completion_time, end)  # 结束
        self.job_on_machine = np.append(self.job_on_machine, on)  # 更新机器上加工的工件
        self.time_list = np.append(self.time_list, t)  # 更新机器上加工的时间
        self.C_max = end  # 最后的时间为end

    @property  # 属性修饰，使方法可以像属性一样访问。
    def on(self):
        return self.job_on_machine


class Scheduling:
    def __init__(self, Job_num: int, Machine_num: list, stage_num: int, Processing_Time_on_machine: np.ndarray):
        # 初始化排程,
        self.Machine_list = None  # 机器集合，包括i阶段k机器
        self.Jobs = None  # 工件集合，包含每个工件的加工信息
        self.Machine_num = Machine_num  # 机器数量集合，列表，包括i阶段k机器,如[3,3,4]
        self.Job_num = Job_num  # 工件数量
        self.Stage = stage_num  # 阶段数量
        self.PT_on_machine = Processing_Time_on_machine  # process time on machine
        self.TOU_mat = np.array([])
        self.makespan = 0  # 记录makespan
        self.critical_job = []  # 记录critical_path上的job
        self.Create_Job()
        self.Create_Machine()

    def Create_Job(self):  # 创建工件列表的函数
        self.Jobs = []  # 工件集合，包含每个工件的加工信息
        for i in range(self.Job_num):
            Job = Item()  # 定义每个工件的属性
            self.Jobs.append(Job)  # 工作列表，有每个工件的属性

    def Create_Machine(self):
        self.Machine_list = []  # 创建机器列表的函数
        for stage in range(len(self.Machine_num)):  # 突出机器的阶段性，即各阶段有哪些机器
            Machine_stage = []  # 每个阶段的机器集合
            for j in range(self.Machine_num[stage]):  # 对于每个阶段的机器i
                Machine = Item()  # 每台机器也定义开始，完成时间等属性
                Machine_stage.append(Machine)
            self.Machine_list.append(Machine_stage)  # 机器的列表

    def Stage_Decode(self, Job_sequence: list, Stage: int):  # 每个阶段的解码,每个阶段的解码
        for job in Job_sequence:
            last_job_end = self.Jobs[job].C_max  # 工件上一阶段的完工时间，论文中为C_i-1_j
            last_machine_end = np.array([self.Machine_list[Stage][M_i].C_max for M_i
                                         in range(self.Machine_num[Stage])])  # 机器上一工件的完成时间，C_i_j-1
            oper_process_time = [self.PT_on_machine[Stage][M_i][job] for M_i in range(self.Machine_num[Stage])]  #
            # 机器i对当前工序的加工时间，有所不同，长度为机器数量
            next_end_time = last_machine_end + oper_process_time  # 数组相加,每台机器加工该工件的可能完工时间
            # 接下来选定一台机器进行加工
            if Counter(next_end_time)[next_end_time.min()] > 1:
                available_machine = np.arange(len(next_end_time))[next_end_time == next_end_time.min()]
                Machine = np.random.choice(available_machine)
                # 如果完工时间最短的机器多于一台，在其中随机选择一台机器
            else:
                Machine = next_end_time.argmin()  # 选择加工时间最短的机器
            start = max(last_job_end, last_machine_end[Machine])  # C_i-1_j和C_i_j-1比大小
            t = oper_process_time[Machine]  # 在选定机器上的加工时间
            end = start + t
            self.Jobs[job].update(start, end, Machine, t)  # 工作记录该信息
            self.Machine_list[Stage][Machine].update(start, end, job, t)
            if end > self.makespan:
                self.makespan = end  # 更新makespan

    def Decode(self, CHS):  # 对一个染色体解码
        for i in range(self.Stage):  # 遍历每个阶段
            self.Stage_Decode(CHS, i)
            Job_end = np.array([self.Jobs[job].C_max for job in range(self.Job_num)])
            # 机器的完工时间
            CHS = np.argsort(Job_end)  # 按完工时间的增序对工件进行排列



    def Gantt(self):
        fig = plt.figure()
        Color = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle',
             'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
             'navajowhite', 'navy', 'sandybrown']
        M_num = 0
        for i in range(len(self.Machine_num)):
            for j in range(self.Machine_num[i]):
                for k in range(len(self.Machine_list[i][j].start_time)):  # 阶段i机器j的工件k
                    Start_time = self.Machine_list[i][j].start_time[k]
                    End_time = self.Machine_list[i][j].completion_time[k]
                    Job = self.Machine_list[i][j].job_on_machine[k]
                    plt.barh(M_num, width=End_time - Start_time, height=0.8, left=Start_time,
                             color=Color[Job], edgecolor='black')
                    plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=M_num - 0.2,
                             s=Job + 1, size=15, fontproperties='Times New Roman')
                M_num += 1
        plt.yticks(np.arange(M_num + 1), np.arange(1, M_num + 2), size=20, fontproperties='Times New Roman')

        plt.ylabel("Machine", size=20, fontproperties='Times New Roman')
        plt.xlabel("Time", size=20, fontproperties='Times New Roman')
        plt.tick_params(labelsize=20)
        plt.tick_params(direction='in')
        plt.show()
