#  Multi-objective energy-efficient hybrid flow shop scheduling using Q-learning and GVNS driven NSGA-II

This is source code of paper published on Computers and Operations research, you can download it on https://www.sciencedirect.com/science/article/pii/S0305054823002241. If you have any problem, contact me at lpz2001@foxmail.com.

The details of each file is given as  follows:

- **Data** contains original scheduling data of hybrid flow shop, including

  - the number of jobs/stages/parallel machines at each stage, $R,\tau$ (which are defined in (28) to determine due date). 

  - Process time of each operation at each stage.
  - The processing speed of each machine and the corresponding conversion rate (See **Table 5**).
  - The power of each machine at processing, standby, reset and shutdown state.
  - The due date $d_j$ of each job $j$.
  - The learned Q-tables Q1 and Q2, using to instruct how to select neighborhood structures.

- **Data_generation.py** is used to generate **Data** based on the above format.

- **Scheduling.py** is the main module of program, in which we define the class **Item** and class **Scheduling** .

  - **class** **Item** define *the basic property of each machine/job and update function*. For example,  Each job $j$ consists of multiple stages, which means that job $j$ should take down  the start time and completion time of  each operation, the machine it is processed on, process time on each machine and so on. Conversely, each machine $m_{ik}$ also records the jobs processed on it, and details of processing. Each time the schedule updates, we update the record of jobs/machines. 
  - **class** **Scheduling** define the initialization of jobs/machines, decoding of chromosomes and Gantt chart painting.  When we invoke **class** **Scheduling**, first we initialize the sets of jobs and machines. The initialization part adopts **class** **Item**. Second, we can decode a chromosome by the earliest completion time (ECT) rule. After decoding, the schedule is specified and we can draw Gantt chart of this schedule using Gantt.

- **QVNS-NSGA-II.py** contains all the functions in the main loop, such as NEH initialization, 5 local search operators and destruction-construction.  This module can be divided into three main parts: NSGA-II, local search operators and Q-learning.

- **QVNS-NSGA-II-no-power-down.py**: The objectives *TEC* and *CTC* do not adopt power down strategy to save energy.

- **Output**: The final outcome of each algorithm and parameter tuning. The xlsx covers objective values, and the job sequence of each Pareto solution.

## QVNS-NSGA-II

As mentioned in the paper, the QVNS-NSGA-II adopts *offline training*. So we need to train the Q-table at first, then we can use Q-table to select local search.

- **Q-table-training.py** trains Q-table 1 and Q-table 2 defined in **Section 5.4**. The detailed procedure is given in **Algorithm 8**. Now we briefly summarize the functions.
  - **para_initialization**: Read the original data in **Data**. Note that *read_path* should be modified when running it on your computer.
  - **main loop**: First, initialize Q-table 1 and Q-table 2. For each instance, we train 3 times and output the final outcome. Then, execute general NSGA-II programs including non-dominated rank, selection, crossover and mutation, elite preservation. Third, we take 5 best individuals from the population, and apply local search(VNS) based on Q-table. During the local search, Q-tables are updated based on the performance of each operator. Finally, compare the improved solutions with the global best solutions, then update the global best solutions.
  - **Output**: Note that *path* should be modified to save trained Q-table 1and Q-table 2.
- **main.py** leverages trained Q-tables to select neighborhood structures. **It outputs the final outcome of QVNS-NSGA-II.**

## NSGA2, MOEAD, Jaya

- **NSGA2.py, MOEAD.py**, **Jaya.py** are the code of NSGA2,MOEA/D and Jaya.
- NOTE THAT *read_path* in para_initialization and *save_path* in \__main__ need to be modified. 

## Parameter tuning

- **Parameter tuning.py** is a slightly modified version of **main.py**. The main difference is that it change algorithm parameters based on **Orthogonal Array.xlsx** in **Data**.  
- NOTE THAT *read_path* and *sv_path* in \__main__  need to be modified. 

## Estimation

- **Estimation.py** evaluates the performance of each algorithm. It calculates the coverage matrices(CM), space matrix(SM) and the number of Pareto solutions. The standard deviation and mean value are also given.
- NOTE THAT *read_path*, *wrt_path* and *wrt_path_2*  in \__main__  need to be modified. 

## How to run

1. Generate Data based on **Data_generation.py** 
2. Train Q-tables based on **Q-table_training.py**
3. Tune the parameters based on **Parameter tuning.py**.
4. Run **main.py** to get the outcome of QVNS-NSGA-II.
5. Run **NSGA2.py**, **MOEAD.py** and **Jaya.py**.
6. Compare the algorithms and output comparison result based on **Estimation.py**.

Again, please note that you have modified the read_path, wrt_path of each file unless program will not work. Also, make sure you have installed all the necessary packages. 
Step 1,2,3 can be skipped since **Data** contains all the needed data.