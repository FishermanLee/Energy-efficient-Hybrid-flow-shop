import pandas as pd
import numpy as np

for Job in [10, 20, 50]:
    for Stage in [3, 5]:
        for Num_machine in [3, 4, 6]:
            '''Job = np.random.choice([10,20,50])  # 可以改成自己的name
            Stage = np.random.choice([3,5])
            Num_machine= np.random.choice([3,4,6])'''
            tao = np.random.choice([0.2, 0.4])
            R = np.random.choice([0.6, 1])

            Para = pd.DataFrame(np.zeros((6, 1)))
            Para.index = ['Job', 'Stage', 'Machine_stage', 'Total_machine', 'tao', 'R']
            Para.columns = ['value']
            Para.iloc[0][0] = int(Job)
            Para.iloc[1][0] = int(Stage)
            Para.iloc[2][0] = int(Num_machine)
            Para.iloc[3][0] = int(Stage * Num_machine)
            Para.iloc[4][0] = tao
            Para.iloc[5][0] = R

            Operation_time = pd.DataFrame(np.random.randint(5, 10, (Job, Stage)))  # 每个操作的时间
            row_name = ['J' + str(i + 1) for i in range(Job)]
            col_name = ['Stage' + str(i + 1) for i in range(Stage)]  # 起名字
            Operation_time.index = row_name
            Operation_time.columns = col_name

            Power_stage = pd.DataFrame(np.random.randint(5, 10, (1, Stage)))  # 每个阶段机器的基础功率
            Machine_ps = np.zeros((Num_machine * Stage, 2))  # 机器的速度和转换系数
            m_name = ['M' + str(i + 1) for i in range(Stage * Num_machine)]  # 起名字
            for i in range(Stage * Num_machine):  # 第一列是速度，第二列是功率系数转换比
                Machine_ps[i][0] = np.random.choice([1.2, 1, 0.8])
                if Machine_ps[i][0] == 1.2:
                    Machine_ps[i][1] = 1.5
                elif Machine_ps[i][0] == 1:
                    Machine_ps[i][1] = 1
                elif Machine_ps[i][0] == 0.8:
                    Machine_ps[i][1] = 0.6
            Machine_ps = pd.DataFrame(Machine_ps)
            Machine_ps.index = m_name  # 行重命名
            Machine_ps.columns = ['Speed', 'Conversion factor']  # 更改列名

            Machine_power = np.zeros((Num_machine * Stage, 4))  # 第一列开动功率，第二列待机，第三列重启，第四列盈亏平衡
            cnt = 0  # 计数器
            for i in range(Stage):
                for j in range(Num_machine):
                    Machine_power[cnt][0] = Machine_ps.iloc[cnt][1] * Power_stage[i]  # 功率的系转换数×基础功率
                    Machine_power[cnt][1] = 2  # 待机功率为2
                    Machine_power[cnt][2] = 4  # 重启功率为4
                    Machine_power[cnt][3] = 2  # 盈亏平衡点为2
                    cnt += 1
            Machine_power = pd.DataFrame(Machine_power)
            Machine_power.columns = ['on', 'idle', 'reset', 'break_even point']
            Machine_power.index = m_name

            Pt_on_machine = np.zeros((Job, Num_machine * Stage))
            for i in range(Job):
                cnt_1 = 0  # 对于每一个工件都清零
                for j in range(Stage):
                    for k in range(Num_machine):
                        Pt_on_machine[i][cnt_1] = round(
                            Operation_time.iloc[i][j] / Machine_ps.iloc[cnt_1][0])  # 加工时间除以机器速度
                        cnt_1 += 1
            Pt_on_machine = pd.DataFrame(Pt_on_machine)
            name = ['M' + str(i + 1) for i in range(len(Pt_on_machine.iloc[0]))]  # 起名字
            Pt_on_machine.columns = name
            Pt_on_machine.index = row_name

            Duedate = pd.DataFrame(np.zeros((Job, 1)))
            Duedate.columns = ['Duedate']
            Duedate.index = row_name

            scale = r'C:\Users\Lipei\Desktop\\' + str(Job) + '-' + str(Stage) + '-' + str(
                Num_machine) + '.xlsx'  # 这里是存储位置，请修改这里进行存储
            # This is saved path, revise this to save the file.

            writer = pd.ExcelWriter(scale)
            Para.to_excel(writer, sheet_name='Parameter')
            Operation_time.to_excel(writer, sheet_name='Process Time')
            Duedate.to_excel(writer, sheet_name='Duedate')
            Pt_on_machine.to_excel(writer, sheet_name='Pt_on_machine')
            Machine_power.to_excel(writer, sheet_name='Power&TB')
            Machine_ps.to_excel(writer, sheet_name='Speed&Rate')

            writer.close()
