from ALB_instance_tools import *
import pandas as pd
import pulp as plp

class MMALBP_LP_Problem:
    def __init__(self, problem_instance, equipment_instance, sequence_length) -> None:
        self.problem_instance = problem_instance
        self.equipment_instance = equipment_instance
        self.sequence_length = sequence_length
        self.workers = list(range(0, problem_instance.no_workers+1))
        self.stations = list(range(problem_instance.no_stations))
        self.c_se = equipment_instance.c_se
        self.R_oe = equipment_instance.r_oe
        self.equipment = list(range(equipment_instance.no_equipment))
        self.prob = None
        self.u_se = None
        self.b_wtsl = None
        self.x_wsoj = None
        self.Y_w = None
        self.make_lp_variables()
        #Results info
        self.obj_value = None
        self.solver_status = None
        

    def get_obj_val_dict_old(self):
        '''generate a dictionary resluts from the solved problem'''
        obj_val_dict = {}
        for i, input_instance in enumerate(self.problem_instance.model_dicts):
            obj_val_dict['file_'+ str(i)] = input_instance['fp']
            model = list(self.problem_instance.data.keys())[i]
            for key, value in self.problem_instance.data[model].items():
                obj_val_dict[model+'_'+key] = value
        obj_val_dict['name'] = self.problem_instance.name
        obj_val_dict['obj_value'] = self.obj_value
        obj_val_dict['solver_status'] = self.solver_status
        return obj_val_dict
    
    def get_obj_val_dict(self):
        '''generate a dictionary resluts from the solved problem'''
        obj_val_dict = {}
        for i, input_instance in enumerate(self.problem_instance.model_mixtures):
            obj_val_dict['file_'+ str(i)] = self.problem_instance.instance_config_yaml
            model = list(self.problem_instance.data.keys())[i]
            for key, value in self.problem_instance.data[model].items():
                obj_val_dict[model+'_'+key] = value
        obj_val_dict['name'] = self.problem_instance.name
        obj_val_dict['obj_value'] = self.obj_value
        obj_val_dict['solver_status'] = self.solver_status
        return obj_val_dict

    def solve(self, solver=None, generate_report =True, file_name = ''):
        self.make_lp_problem()
        self.prob.solve(solver=solver)
        self.obj_value = self.prob.objective.value()
        self.solver_status = self.prob.status
        #only generate report if the problem is solved
        if generate_report and self.solver_status == 1:
            task_assignment_df, labor_assignments_df = self.generate_report(file_name)
            return self.prob, task_assignment_df, labor_assignments_df
        
        warnings.warn('Solver did not find a solution')

        return self.prob, None, None
    
    def generate_report(self, file_name):
        return

# class dynamic_problem_linear_labor(MMALBP_LP_Problem):
#     def __init__(self, problem_instance, equipment_instance, sequence_length, prod_sequences) -> None:
#         self.problem_instance = problem_instance
#         self.equipment_instance = equipment_instance
#         self.sequence_length = sequence_length
#         self.prod_sequences = prod_sequences
#         self.workers = list(range(0, problem_instance.max_workers+1))
#         self.stations = list(range(problem_instance.no_stations))
#         self.c_se = equipment_instance.c_se
#         self.R_oe = equipment_instance.r_oe
#         self.equipment = list( range(self.R_oe.shape[1]))
#         self.takts = list(range(sequence_length+problem_instance.no_stations-1))
#         self.prob = None
#         self.u_se = None
#         self.b_wtsl = None
#         self.x_wsoj = None
#         self.Y_w = None
#         self.make_lp_variables()
#         #Results info
#         self.obj_value = None
#         self.solver_status = None
    
#     def make_lp_variables(self):
#         self.u_se = plp.LpVariable.dicts('u_se', (self.stations, self.equipment), lowBound=0, cat='Binary')
#         self.b_wtsl = plp.LpVariable.dicts('b_wtsl', (self.prod_sequences.keys(), self.takts, self.stations, self.workers), lowBound=0, cat='Binary')
#         self.x_wsoj = plp.LpVariable.dicts('x_wsoj', (self.prod_sequences.keys(), self.stations, range(self.problem_instance.no_tasks), range(self.sequence_length) ), lowBound=0, cat='Binary')
#         self.Y_w = plp.LpVariable.dicts('Y_w', (self.prod_sequences.keys()), lowBound=0, cat='Integer')


#     def set_variables(self, u_se =None, b_wtsl = None, x_wsoj= None, x_soi=None, Y_w = None, fixed=False):
#         '''Sets variables before solving the problem. If fixed is true, then the variables are fixed to their initial values. Input is a dictionary of LpVariable'''
#         if u_se is not None:
#             for s, equipment_dict in u_se.items():
#                 for equip_id in equipment_dict:
#                     e = int(equip_id)
#                     self.u_se[s][e].setInitialValue(round(u_se[s][equip_id].value()))
#                 if fixed:
#                     self.u_se[s][e].fixValue()
#         if b_wtsl is not None:
#             for w in b_wtsl.keys():
#                 for t in b_wtsl[w].keys():
#                     for s in b_wtsl[w][t].keys():
#                         for l, labor_var in b_wtsl[w][t][s].items():
#                             labor_val = labor_var.value()
#                             if labor_val > 0:
#                                 self.b_wtsl[w][t][s][l].setInitialValue(round(labor_val))
#                                 if fixed:
#                                     self.b_wtsl[w][t][s][l].fixValue()
#         #Assignes tasks to dynamic model from list of model depedent task assignments
#         if x_soi is not None:
#             for w in self.prod_sequences.keys():
#                 for s in x_soi.keys():
#                     for o in x_soi[s].keys():
#                         for model_md, task_var in x_soi[s][o].values():
#                             task_assigned = task_var.value()
#                             #setting x_wsoj values from the start solution
#                             for j, model in enumerate(self.prod_sequences[w]['sequence']):
#                                 if model_md == model:
#                                     x_wsoj[w][s][o][j].setInitialValue(round(task_assigned))
#                                     if fixed:
#                                         x_wsoj[w][s][o][j].fixValue()
#         if x_wsoj is not None:
#             for w in x_wsoj.keys():
#                 for s in x_wsoj[w].keys():
#                     for o in x_wsoj[w][s].keys():
#                         for j, task_var in x_wsoj[w][s][o].items():
#                             task_assigned = task_var.value()
#                             self.x_wsoj[w][s][o][j].setInitialValue(round(task_assigned))
#                             if fixed:
#                                 self.x_wsoj[w][s][o][j].fixValue()
#         if Y_w is not None:
#             for w in Y_w.keys():
#                 self.Y_w[w].setInitialValue(round(Y_w[w].value()))
#                 if fixed:
#                     self.Y_w[w].fixValue()

#     def add_non_anticipation(self, w, w_prime , prod_sequences, problem_instance, x_wsoj, sequence_length, num_stations):
#         '''adds the non anticipation constraint for two scenarios w and w_prime'''
#         #We go backwards in time, look for first time t where the sequences are the same
#         for t in reversed(range(sequence_length)):
#             if check_scenarios(prod_sequences[w]['sequence'], prod_sequences[w_prime]['sequence'], t):
#                 for j in range(t+1):
#                     model = prod_sequences[w]['sequence'][j]
#                     max_station = min(t-j+1, num_stations)    
#                     for s in range(max_station):
#                         for o in range(problem_instance.data[model]['num_tasks']):
#                             self.prob += (x_wsoj[w][s][o][j] == x_wsoj[w_prime][s][o][j], 
#                                     f'anti_ww*soj_{w}_{w_prime}_{s}_{o}_{j}')
#                 return

#     def make_lp_problem(self):
#         #Defining LP problem
#         self.prob = plp.LpProblem("stochastic_problem", plp.LpMinimize)
#         #Objective function
#         self.prob += (plp.lpSum([self.c_se[s][e]*self.u_se[s][e]
#                             for s in self.stations
#                             for e in self.equipment]
#                         +
#                         [self.prod_sequences[w]['probability']*self.Y_w[w]* self.problem_instance.worker_cost
#                             for w in self.prod_sequences.keys()
#                             ]),
#                     "Total cost")
#         #Constraints
#         #Constraint 1 -- Must hire Y workers if we use Y workers in a given takt
#         for w in self.prod_sequences.keys():
#             for t in self.takts:
#                 self.prob += (plp.lpSum([l*self.b_wtsl[w][t][s][l] for s in self.stations for l in self.workers]) <= self.Y_w[w], f'Y_w_{w}_{t}')
#         #Constraint 2 -- can only assign l number of workers to a station for a given scenario and stage
#         for w in self.prod_sequences.keys():
#             for t in self.takts:
#                 for s in self.stations:
#                     self.prob += (plp.lpSum([self.b_wtsl[w][t][s][l] for l in self.workers]) == 1, f'b_wtsl_{w}_{t}_{s}')
#             #Constraint 3 all tasks must be assigned to a station
#         for w in self.prod_sequences.keys():
#             for j, model in enumerate(self.prod_sequences[w]['sequence']):
#                 for o in range(self.problem_instance.no_tasks): 
#                     self.prob += (plp.lpSum([self.x_wsoj[w][s][o][j] for s in self.stations]) == 1, f'x_wsoj_{w}_s_{o}_{j}')
#             #Constraint 4 -- sum of task times for assigned tasks must be less than takt time times the number of workers for a given station
#         for w in self.prod_sequences.keys():
#             for t in self.takts:
#                 for s in self.stations:
#                     #Get the model at the current scenario, stage, and station
#                     if 0<= t-s < self.sequence_length:
#                         j = t-s
#                         model = self.prod_sequences[w]['sequence'][j]
#                         task_times = self.problem_instance.data[model]['task_times']
#                         self.prob += (plp.lpSum([task_times[o]*self.x_wsoj[w][s][int(o)-1][j] 
#                                             for o in task_times]) 
#                                             <= 
#                                             self.problem_instance.takt_time*plp.lpSum([l * self.b_wtsl[w][t][s][l] for l in self.workers]), f'task_time_wts_{w}_{t}_{s}')

#         #Constraint 5 -- tasks can only be assigned to a station with the correct equipment
#         for w in self.prod_sequences.keys():
#             for j, model in enumerate(self.prod_sequences[w]['sequence']):
#                 for s in self.stations:
#                     for o in range(self.problem_instance.no_tasks):
#                         self.prob += self.x_wsoj[w][s][o][j] <= plp.lpSum([self.R_oe[o][e]*self.u_se[s][e] for  e in self.equipment]), f'equipment_wsoj_{w}_{s}_{o}_{j}'
#             #Constraint 6 -- precedence constraints
#         for w in self.prod_sequences.keys():
#             for j, model in enumerate(self.prod_sequences[w]['sequence']):
#                 for (pred, suc) in self.problem_instance.data[model]['precedence_relations']:
#                     self.prob += (plp.lpSum([ (s+1)  * self.x_wsoj[w][s][int(pred)-1][j] for s in self.stations])
#                             <=  
#                             plp.lpSum([ (s+1)  * self.x_wsoj[w][s][int(suc)-1][j] for s in self.stations]), 
#                             f'task{pred} before task{suc} for model{model}, item {j} seq {w}' )
#             #Constraint 7 -- non-anticipativity constraints
#         for w in self.prod_sequences.keys():
#             for w_prime in self.prod_sequences.keys():
#                 if w_prime > w:
#                     self.add_non_anticipation(w, w_prime , self.prod_sequences, self.problem_instance, self.x_wsoj, self.sequence_length,self.problem_instance.no_stations)             
#         return 
    
#     def generate_report(self, file_name):
#         '''Shows task assignments for fixed and model dependent task assignment'''
#         task_assignments = []
#         labor_assignments = []
#         labor_hire_assignments = []
#         for v in self.prob.variables():
#             if round(v.varValue) > 0:
#                 if 'x_wsoj' in v.name:
#                     sequence = int(v.name.split('_')[2])
#                     item = int(v.name.split('_')[5])
#                     model = self.prod_sequences[sequence]['sequence'][item]
#                     #change the task number to match with the instances
#                     task = str(int(v.name.split('_')[4])+1)
#                     task_time = self.problem_instance.data[model]['task_times'][task]
#                     assignment = {'scenario':v.name.split('_')[2], 'station': v.name.split('_')[3],'sequence_loc':str(item), 'model':model  , 'task': task, 'task_times': task_time}
#                     task_assignments.append(assignment)
#                 elif 'b_wtsl' in v.name:
#                     workers = int(v.name.split('_')[5])
#                     if workers > 0:
#                         stage = int(v.name.split('_')[3])
#                         station = int(v.name.split('_')[4])
#                         sequence_loc =stage-station
#                         model = self.prod_sequences[int(v.name.split('_')[2])]['sequence'][sequence_loc]
#                         labor = {'scenario':v.name.split('_')[2], 'stage':str(stage), 
#                                 'station':str(station), 'model':model, 'sequence_loc':str(sequence_loc), 'workers': int(v.name.split('_')[5]) }
#                         print('labor', labor)
#                         labor_assignments.append(labor)
#                 elif 'Y_w' in v.name:
#                     labor_hire = {'scenario':v.name.split('_')[2], 'scenario_workers': int(v.value()) }
#                     labor_hire_assignments.append(labor_hire)

#         #turns task_assignments into a dataframe
#         task_assignments_df = pd.DataFrame(task_assignments)
#         labor_assignments_df = pd.DataFrame(labor_assignments)
#         labor_hire_df = pd.DataFrame(labor_hire_assignments)
#         #concatenates the 'task' column in task_assignments_df if the 'station' and 'model' columns are the same
#         task_assignments_df = task_assignments_df.groupby(['scenario','station', 'sequence_loc','model'])['task', 'task_times'].agg({'task':lambda x: ','.join(x.astype(str)), 'task_times': sum }).reset_index()
#         #labor_assignments_df['sequence_loc'] = labor_assignments_df['stage'].astype(int) - labor_assignments_df['station'].astype(int)
#         #labor_assignments_df = labor_assignments_df[labor_assignments_df['sequence_loc'] >= 0]
#         task_seq = task_assignments_df[['scenario','station', 'task','task_times', 'sequence_loc']]
#         #merging labor and task sequence dataframes
#         labor_task_seq = pd.merge(labor_assignments_df, task_seq, on=['scenario','station', 'sequence_loc'], how='left')
#         labor_task_seq = pd.merge(labor_task_seq, labor_hire_df, on=['scenario'], how='left')
#         task_assignments_df.to_csv(file_name + f'task_assignment.csv', sep=' ')
#         labor_task_seq.to_csv(file_name + f'labor_assignment.csv', sep=' ')
#         return task_assignments_df, labor_assignments_df
    
class dynamic_problem_linear_labor_recourse(MMALBP_LP_Problem):
    def __init__(self, problem_instance, equipment_instance, sequence_length, prod_sequences) -> None:
        self.problem_instance = problem_instance
        self.equipment_instance = equipment_instance
        self.sequence_length = sequence_length
        self.prod_sequences = prod_sequences
        self.workers = list(range(0, problem_instance.max_workers+1))
        self.stations = list(range(problem_instance.no_stations))
        self.c_se = equipment_instance.c_se
        self.R_oe = equipment_instance.r_oe
        self.equipment = list( range(self.equipment_instance.no_equipment))
        self.takts = list(range(sequence_length+problem_instance.no_stations-1))
        self.prob = None
        self.u_se = None
        self.b_wtsl = None
        self.x_wsoj = None
        self.Y_w = None
        self.Y = None
        self.make_lp_variables()
        #Results info
        self.obj_value = None
        self.solver_status = None
    
    def make_lp_variables(self):
        self.u_se = plp.LpVariable.dicts('u_se', (self.stations, self.equipment), lowBound=0, cat='Binary')
        self.l_wts = plp.LpVariable.dicts('l_wts', (self.prod_sequences.keys(), self.takts, self.stations), lowBound=0, cat='Integer')
        self.x_wsoj = plp.LpVariable.dicts('x_wsoj', (self.prod_sequences.keys(), self.stations, range(self.problem_instance.no_tasks), range(self.sequence_length) ), lowBound=0, cat='Binary')
        self.Y_w = plp.LpVariable.dicts('Y_w', (self.prod_sequences.keys()), lowBound=0, cat='Integer')
        self.Y = plp.LpVariable('Y', lowBound=0, cat='Integer')

    def make_lp_problem(self):
        #Defining LP problem
        self.prob = plp.LpProblem("stochastic_problem", plp.LpMinimize)
        #Objective function
        self.prob += (plp.lpSum([self.c_se[s][e]*self.u_se[s][e]
                            for s in self.stations
                            for e in self.equipment]
                        +
                        self.Y*self.problem_instance.worker_cost 
                        +
                        [self.prod_sequences[w]['probability']*self.Y_w[w]* self.problem_instance.recourse_cost
                            for w in self.prod_sequences.keys()
                            ]),
                    "Total cost")
        #Constraints
        #Constraint 1 -- Must hire Y workers if we use Y workers in a given takt
        for w in self.prod_sequences.keys():
            for t in self.takts:
                self.prob += (plp.lpSum([self.l_wts[w][t][s] for s in self.stations ]) <= plp.lpSum([self.Y_w[w] + self.Y]) , f'Y_w_{w}_{t}')
        #Constraint 2 -- can only assign l number of workers to a station for a given scenario and stage
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    self.prob += (self.l_wts[w][t][s]  <= self.problem_instance.max_workers , f'b_wtsl_{w}_{t}_{s}')
            #Constraint 3 all tasks must be assigned to a station
        for w in self.prod_sequences.keys():
            for j, model in enumerate(self.prod_sequences[w]['sequence']):
                for o in range(self.problem_instance.no_tasks): 
                    self.prob += (plp.lpSum([self.x_wsoj[w][s][o][j] for s in self.stations]) == 1, f'x_wsoj_{w}_s_{o}_{j}')
            #Constraint 4 -- sum of task times for assigned tasks must be less than takt time times the number of workers for a given station
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    #Get the model at the current scenario, stage, and station
                    if 0<= t-s < self.sequence_length:
                        j = t-s
                        model = self.prod_sequences[w]['sequence'][j]
                        task_times = self.problem_instance.data[model]['task_times']
                        self.prob += (plp.lpSum([task_times[o]*self.x_wsoj[w][s][int(o)-1][j] 
                                            for o in task_times]) 
                                            <= 
                                            self.problem_instance.takt_time* self.l_wts[w][t][s], f'task_time_wts_{w}_{t}_{s}')

        #Constraint 5 -- tasks can only be assigned to a station with the correct equipment
        for w in self.prod_sequences.keys():
            for j, model in enumerate(self.prod_sequences[w]['sequence']):
                for s in self.stations:
                    for o in range(self.problem_instance.no_tasks):
                        self.prob += self.x_wsoj[w][s][o][j] <= plp.lpSum([self.R_oe[o][e]*self.u_se[s][e] for  e in self.equipment]), f'equipment_wsoj_{w}_{s}_{o}_{j}'
            #Constraint 6 -- precedence constraints
        for w in self.prod_sequences.keys():
            for j, model in enumerate(self.prod_sequences[w]['sequence']):
                for (pred, suc) in self.problem_instance.data[model]['precedence_relations']:
                    self.prob += (plp.lpSum([ (s+1)  * self.x_wsoj[w][s][int(pred)-1][j] for s in self.stations])
                            <=  
                            plp.lpSum([ (s+1)  * self.x_wsoj[w][s][int(suc)-1][j] for s in self.stations]), 
                            f'task{pred} before task{suc} for model{model}, item {j} seq {w}' )
            #Constraint 7 -- non-anticipativity constraints
        for w in self.prod_sequences.keys():
            for w_prime in self.prod_sequences.keys():
                if w_prime > w:
                    self.add_non_anticipation(w, w_prime , self.prod_sequences, self.problem_instance, self.x_wsoj, self.sequence_length,self.problem_instance.no_stations)

        print('prob', self.prob)             
        return 
    
    def add_non_anticipation(self, w, w_prime , prod_sequences, problem_instance, x_wsoj, sequence_length, num_stations):
        '''adds the non anticipation constraint for two scenarios w and w_prime'''
        #We go backwards in time, look for first time t where the sequences are the same
        for t in reversed(range(sequence_length)):
            if check_scenarios(prod_sequences[w]['sequence'], prod_sequences[w_prime]['sequence'], t):
                for j in range(t+1):
                    model = prod_sequences[w]['sequence'][j]
                    max_station = min(t-j+1, num_stations)    
                    for s in range(max_station):
                        for o in range(problem_instance.data[model]['num_tasks']):
                            self.prob += (x_wsoj[w][s][o][j] == x_wsoj[w_prime][s][o][j], 
                                    f'anti_ww*soj_{w}_{w_prime}_{s}_{o}_{j}')
                return
            

    def generate_report(self, file_name):
        '''Shows task assignments for fixed and model dependent task assignment'''
        task_assignments = []
        labor_assignments = []
        labor_hire_assignments = []
        fixed_labor = 0
        for v in self.prob.variables():
            if round(v.varValue) > 0:
                print('variable values', v.name, v.value())
                if 'x_wsoj' in v.name:
                    sequence = int(v.name.split('_')[2])
                    item = int(v.name.split('_')[5])
                    model = self.prod_sequences[sequence]['sequence'][item]
                    #change the task number to match with the instances
                    task = str(int(v.name.split('_')[4])+1)
                    task_time = self.problem_instance.data[model]['task_times'][task]
                    assignment = {'scenario':v.name.split('_')[2], 'station': v.name.split('_')[3],'sequence_loc':str(item), 'model':model  , 'task': task, 'task_times': task_time}
                    task_assignments.append(assignment)
                elif 'l_wts' in v.name:
                    workers = v.value()
                    if workers > 0:
                        stage = int(v.name.split('_')[3])
                        station = int(v.name.split('_')[4])
                        sequence_loc =stage-station
                        model = self.prod_sequences[int(v.name.split('_')[2])]['sequence'][sequence_loc]
                        labor = {'scenario':v.name.split('_')[2], 'stage':str(stage), 
                                'station':str(station), 'model':model, 'sequence_loc':str(sequence_loc), 'workers': workers }
                        labor_assignments.append(labor)
                elif 'Y_w' in v.name:
                    pass
                elif 'Y' in v.name:
                    fixed_labor = int(v.value())
            if 'Y_w' in v.name:
                    labor_hire = {'scenario':v.name.split('_')[2], 'scenario_workers': int(v.value()) }
                    labor_hire_assignments.append(labor_hire)

        #turns task_assignments into a dataframe
        task_assignments_df = pd.DataFrame(task_assignments)
        labor_assignments_df = pd.DataFrame(labor_assignments)
        labor_hire_df = pd.DataFrame(labor_hire_assignments)
        labor_hire_df['fixed_workers'] = fixed_labor
        #concatenates the 'task' column in task_assignments_df if the 'station' and 'model' columns are the same
        task_assignments_df = task_assignments_df.groupby(['scenario','station', 'sequence_loc','model'])['task', 'task_times'].agg({'task':lambda x: ','.join(x.astype(str)), 'task_times': sum }).reset_index()
        #labor_assignments_df['sequence_loc'] = labor_assignments_df['stage'].astype(int) - labor_assignments_df['station'].astype(int)
        #labor_assignments_df = labor_assignments_df[labor_assignments_df['sequence_loc'] >= 0]
        task_seq = task_assignments_df[['scenario','station', 'task','task_times', 'sequence_loc']]
        #merging labor and task sequence dataframes
        labor_task_seq = pd.merge(labor_assignments_df, task_seq, on=['scenario','station', 'sequence_loc'], how='left')
        labor_task_seq = pd.merge(labor_task_seq, labor_hire_df, on=['scenario'], how='left')
        task_assignments_df.to_csv(file_name + f'task_assignment.csv', sep=' ')
        labor_task_seq.to_csv(file_name + f'labor_assignment.csv', sep=' ')
        return task_assignments_df, labor_assignments_df


class dynamic_problem_multi_labor_recourse(MMALBP_LP_Problem):
    def __init__(self, problem_instance, equipment_instance, sequence_length, prod_sequences) -> None:
        self.name = 'dynamic_problem_multi_labor_recourse'
        self.problem_instance = problem_instance
        self.equipment_instance = equipment_instance
        self.sequence_length = sequence_length
        self.prod_sequences = prod_sequences
        self.workers = list(range(1, problem_instance.max_workers+1))
        self.stations = list(range(problem_instance.no_stations))
        self.c_se = equipment_instance.c_se
        self.R_oe = equipment_instance.r_oe
        self.equipment = list( range(self.equipment_instance.no_equipment))
        self.takts = list(range(sequence_length+problem_instance.no_stations-1))
        self.prob = None
        self.u_se = None
        self.b_wtsl = None
        self.x_wlsoj = None
        self.Y_w = None
        self.Y = None
        self.make_lp_variables()
        #Results info
        self.obj_value = None
        self.solver_status = None
    
    def make_lp_variables(self):
        self.u_se = plp.LpVariable.dicts('u_se', (self.stations, self.equipment), lowBound=0, cat='Binary')
        self.l_wts = plp.LpVariable.dicts('l_wts', (self.prod_sequences.keys(), self.takts, self.stations), lowBound=0, cat='Integer')
        self.x_wlsoj = plp.LpVariable.dicts('x_wlsoj', (self.prod_sequences.keys(), self.workers, self.stations, range(self.problem_instance.no_tasks), range(self.sequence_length) ), lowBound=0, cat='Binary')
        self.Y_w = plp.LpVariable.dicts('Y_w', (self.prod_sequences.keys()), lowBound=0, cat='Integer')
        self.Y = plp.LpVariable('Y', lowBound=0, cat='Integer')

    def make_lp_problem(self):
        #Defining LP problem
        self.prob = plp.LpProblem("stochastic_problem", plp.LpMinimize)
        #Objective function
        self.prob += (plp.lpSum([self.c_se[s][e]*self.u_se[s][e]
                            for s in self.stations
                            for e in self.equipment]
                        +
                        self.Y*self.problem_instance.worker_cost 
                        +
                        [self.prod_sequences[w]['probability']*self.Y_w[w]* self.problem_instance.recourse_cost
                            for w in self.prod_sequences.keys()
                            ]),
                    "Total cost")
        #Constraints
        #Constraint 1 -- Must hire Y workers if we use Y workers in a given takt
        for w in self.prod_sequences.keys():
            for t in self.takts:
                self.prob += (plp.lpSum([self.l_wts[w][t][s] for s in self.stations ]) <= plp.lpSum([self.Y_w[w] + self.Y]) , f'Y_w_{w}_{t}')
        #Constraint 2 -- can only assign l number of workers to a station for a given scenario and stage
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    self.prob += (self.l_wts[w][t][s]  <= self.problem_instance.max_workers , f'b_wtsl_{w}_{t}_{s}')
        #Constraint 3 all tasks must be assigned to a station
        for w in self.prod_sequences.keys():
            for j, model in enumerate(self.prod_sequences[w]['sequence']):
                for o in range(self.problem_instance.no_tasks): 
                    self.prob += (plp.lpSum([self.x_wlsoj[w][l][s][o][j] for l in self.workers for s in self.stations]) == 1, f'x_wlsoj_{w}_l_s_{o}_{j}')
        #Constraint 4 -- cannot assign more workers to task than there are workers assigned to station
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    #Get the model at the current scenario, stage, and station
                    for o in range(self.problem_instance.no_tasks):
                        if 0<= t-s < self.sequence_length:
                            j = t-s
                            self.prob += (plp.lpSum([ l * self.x_wlsoj[w][l][s][o][j] for l in self.workers]) <= self.l_wts[w][t][s], f'worker_assign_wlsoj_{w}_l_{s}_{o}_{j}')
        #Constraint 5 -- sum of task times for assigned tasks must be less than takt time times the number of workers for a given station
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    #Get the model at the current scenario, stage, and station
                    if 0<= t-s < self.sequence_length:
                        j = t-s
                        model = self.prod_sequences[w]['sequence'][j]
                        task_times = self.problem_instance.data[model]['task_times']
                        self.prob += (plp.lpSum([task_times[l][o]*self.x_wlsoj[w][l][s][int(o)-1][j] 
                                            for l in self.workers for o in task_times[l]]) 
                                            <= 
                                            self.problem_instance.takt_time, f'task_time_wts_{w}_{t}_{s}')

        #Constraint 6 -- tasks can only be assigned to a station with the correct equipment
        for w in self.prod_sequences.keys():
            for j, model in enumerate(self.prod_sequences[w]['sequence']):
                for s in self.stations:
                    for o in range(self.problem_instance.no_tasks):
                        self.prob += plp.lpSum([self.x_wlsoj[w][l][s][o][j] for l in self.workers])  <= plp.lpSum([self.R_oe[o][e]*self.u_se[s][e] for  e in self.equipment]), f'equipment_wsoj_{w}_{s}_{o}_{j}'
            #Constraint 7 -- precedence constraints
        for w in self.prod_sequences.keys():
            for j, model in enumerate(self.prod_sequences[w]['sequence']):
                for (pred, suc) in self.problem_instance.data[model]['precedence_relations']:
                    self.prob += (plp.lpSum([ (s+1)  * self.x_wlsoj[w][l][s][int(pred)-1][j] for s in self.stations for l in self.workers])
                            <=  
                            plp.lpSum([ (s+1)  * self.x_wlsoj[w][l][s][int(suc)-1][j] for s in self.stations for l in self.workers]), 
                            f'task{pred} before task{suc} for model{model}, item {j} seq {w}' )
            #Constraint 8 -- non-anticipativity constraints
        for w in self.prod_sequences.keys():
            for w_prime in self.prod_sequences.keys():
                if w_prime > w:
                    self.add_non_anticipation(w, w_prime , self.prod_sequences, self.problem_instance, self.x_wlsoj, self.sequence_length,self.problem_instance.no_stations)
         
        return 
    
    def add_non_anticipation(self, w, w_prime , prod_sequences, problem_instance, x_wlsoj, sequence_length, num_stations):
        '''adds the non anticipation constraint for two scenarios w and w_prime'''
        #We go backwards in time, look for first time t where the sequences are the same
        for t in reversed(range(sequence_length)):
            if check_scenarios(prod_sequences[w]['sequence'], prod_sequences[w_prime]['sequence'], t):
                for j in range(t+1):
                    model = prod_sequences[w]['sequence'][j]
                    max_station = min(t-j+1, num_stations)    
                    for s in range(max_station):
                        for o in range(problem_instance.data[model]['num_tasks']):
                            for l in self.workers:
                                self.prob += (x_wlsoj[w][l][s][o][j] == x_wlsoj[w_prime][l][s][o][j], 
                                        f'anti_ww*lsoj_{w}_{w_prime}_{l}_{s}_{o}_{j}')
                return
            

    def generate_report(self, file_name):
        '''Shows task assignments for fixed and model dependent task assignment'''
        task_assignments = []
        labor_assignments = []
        labor_hire_assignments = []
        fixed_labor = 0
        for v in self.prob.variables():
            if round(v.varValue) > 0:
                if 'x_wlsoj' in v.name:
                    sequence = int(v.name.split('_')[2])
                    item = int(v.name.split('_')[6])
                    workers = int(v.name.split('_')[3])
                    model = self.prod_sequences[sequence]['sequence'][item]
                    #change the task number to match with the instances
                    task = str(int(v.name.split('_')[5])+1)
                    task_time = self.problem_instance.data[model]['task_times'][workers][task]
                    assignment = {'scenario':v.name.split('_')[2], 'station': v.name.split('_')[4],'sequence_loc':str(item), 'model':model  , 'task': task, 'task_times': task_time}
                    task_assignments.append(assignment)
                elif 'l_wts' in v.name:
                    workers = v.value()
                    if workers > 0:
                        stage = int(v.name.split('_')[3])
                        station = int(v.name.split('_')[4])
                        sequence_loc =stage-station
                        model = self.prod_sequences[int(v.name.split('_')[2])]['sequence'][sequence_loc]
                        labor = {'scenario':v.name.split('_')[2], 'stage':str(stage), 
                                'station':str(station), 'model':model, 'sequence_loc':str(sequence_loc), 'workers': workers }
                        labor_assignments.append(labor)
                elif 'Y_w' in v.name:
                    pass
                elif 'Y' in v.name:
                    fixed_labor = int(v.value())
            if 'Y_w' in v.name:
                    labor_hire = {'scenario':v.name.split('_')[2], 'scenario_workers': int(v.value()) }
                    labor_hire_assignments.append(labor_hire)

        #turns task_assignments into a dataframe
        task_assignments_df = pd.DataFrame(task_assignments)
        labor_assignments_df = pd.DataFrame(labor_assignments)
        labor_hire_df = pd.DataFrame(labor_hire_assignments)
        labor_hire_df['fixed_workers'] = fixed_labor
        #concatenates the 'task' column in task_assignments_df if the 'station' and 'model' columns are the same
        task_assignments_df = task_assignments_df.groupby(['scenario','station', 'sequence_loc','model'])['task', 'task_times'].agg({'task':lambda x: ','.join(x.astype(str)), 'task_times': sum }).reset_index()
        #labor_assignments_df['sequence_loc'] = labor_assignments_df['stage'].astype(int) - labor_assignments_df['station'].astype(int)
        #labor_assignments_df = labor_assignments_df[labor_assignments_df['sequence_loc'] >= 0]
        task_seq = task_assignments_df[['scenario','station', 'task','task_times', 'sequence_loc']]
        #merging labor and task sequence dataframes
        labor_task_seq = pd.merge(labor_assignments_df, task_seq, on=['scenario','station', 'sequence_loc'], how='left')
        labor_task_seq = pd.merge(labor_task_seq, labor_hire_df, on=['scenario'], how='left')
        task_assignments_df.to_csv(file_name + f'task_assignment.csv', sep=' ')
        labor_task_seq.to_csv(file_name + f'labor_assignment.csv', sep=' ')
        return task_assignments_df, labor_assignments_df    

    

# class model_dependent_problem_linear_labor(MMALBP_LP_Problem):
#     def __init__(self, problem_instance, equipment_instance, sequence_length, prod_sequences, fixed_assignment = False) -> None:
#         self.problem_instance = problem_instance
#         self.equipment_instance = equipment_instance
#         self.sequence_length = sequence_length
#         self.prod_sequences = prod_sequences
#         self.workers = list(range(0, problem_instance.max_workers+1))
#         self.stations = list(range(problem_instance.no_stations))
#         self.c_se = equipment_instance.c_se
#         self.R_oe = equipment_instance.r_oe
#         self.equipment = list( range(self.R_oe.shape[1]))
#         self.takts = list(range(sequence_length+problem_instance.no_stations-1))
#         self.prob = None
#         self.u_se = None
#         self.b_wtsl = None
#         self.x_soi = None
#         self.Y_w = None
#         self.fixed_assignment = fixed_assignment
#         self.make_lp_variables()
#         #Results info
#         self.obj_value = None
#         self.solver_status = None
    
#     def make_lp_variables(self):
#         self.u_se = plp.LpVariable.dicts('u_se', (self.stations, self.equipment), lowBound=0, cat='Binary')
#         self.b_wtsl = plp.LpVariable.dicts('b_wtsl', (self.prod_sequences.keys(), self.takts, self.stations, self.workers), lowBound=0, cat='Binary')
#         self.x_soi = plp.LpVariable.dicts('x_soi', ( self.stations, range(self.problem_instance.no_tasks), self.problem_instance.data.keys() ), lowBound=0, cat='Binary')
#         self.Y_w = plp.LpVariable.dicts('Y_w', (self.prod_sequences.keys()), lowBound=0, cat='Integer')


#     def set_variables(self, u_se =None, b_wtsl = None, x_wsoj= None, x_soi=None, Y_w = None, fixed=False):
#         '''Sets variables before solving the problem. If fixed is true, then the variables are fixed to their initial values. Input is a dictionary of LpVariable'''
#         if u_se is not None:
#             for s, equipment_dict in u_se.items():
#                 for equip_id in equipment_dict:
#                     e = int(equip_id)
#                     self.u_se[s][e].setInitialValue(round(u_se[s][equip_id].value()))
#                 if fixed:
#                     self.u_se[s][e].fixValue()
#         if b_wtsl is not None:
#             for w in b_wtsl.keys():
#                 for t in b_wtsl[w].keys():
#                     for s in b_wtsl[w][t].keys():
#                         for l, labor_var in b_wtsl[w][t][s].items():
#                             labor_val = labor_var.value()
#                             if labor_val > 0:
#                                 self.b_wtsl[w][t][s][l].setInitialValue(round(labor_val))
#                                 if fixed:
#                                     self.b_wtsl[w][t][s][l].fixValue()
#         if x_soi is not None:
#             for s in x_soi.keys():
#                 for o in x_soi[s].keys():
#                     for i, model in x_soi[s][o].items():
#                         task_assigned = model.value()
#                         self.x_soi[s][o][i].setInitialValue(round(task_assigned))
#                         if fixed:
#                             self.x_soi[s][o][i].fixValue()

#         if Y_w is not None:
#             for w in Y_w.keys():
#                 self.Y_w[w].setInitialValue(round(Y_w[w].value()))
#                 if fixed:
#                     self.Y_w[w].fixValue()



#     def make_lp_problem(self):
#         #Defining LP problem
#         self.prob = plp.LpProblem("model_dependent_eq_problem", plp.LpMinimize)
#         #Objective function
#         self.prob += (plp.lpSum([self.c_se[s][e]*self.u_se[s][e]
#                             for s in self.stations
#                             for e in self.equipment]
#                         +
#                         [self.prod_sequences[w]['probability']*self.Y_w[w]* self.problem_instance.worker_cost
#                             for w in self.prod_sequences.keys()
#                             ]),
#                     "Total cost")
#         #Constraints
#         #Constraint 1 -- Must hire Y workers if we use Y workers in a given takt
#         for w in self.prod_sequences.keys():
#             for t in self.takts:
#                 self.prob += (plp.lpSum([l*self.b_wtsl[w][t][s][l] for s in self.stations for l in self.workers]) <= self.Y_w[w], f'Y_w_{w}_{t}')

#         #Constraint 2 -- can only assign l number of workers to a station for a given scenario and stage
#         for w in self.prod_sequences.keys():
#             for t in self.takts:
#                 for s in self.stations:
#                     self.prob += (plp.lpSum([self.b_wtsl[w][t][s][l] for l in self.workers]) == 1, f'b_wtsl_{w}_{t}_{s}')
#         #Constraint 3 all tasks must be assigned to a station
#         for i, model in enumerate(self.problem_instance.data.keys()):
#             for o in range(self.problem_instance.no_tasks): 
#                 self.prob += (plp.lpSum([self.x_soi[s][o][model] for s in self.stations]) == 1, f'x_soi_{s}_{o}_{model}')
#             #Constraint 4 -- sum of task times for assigned tasks must be less than takt time times the number of workers for a given station
#         for w in self.prod_sequences.keys():
#             for t in self.takts:
#                 for s in self.stations:
#                     #Get the model at the current scenario, stage, and station
#                     if 0<= t-s < self.sequence_length:
#                         j = t-s
#                         model = self.prod_sequences[w]['sequence'][j]
#                         task_times = self.problem_instance.data[model]['task_times']
#                         self.prob += (plp.lpSum([task_times[o]*self.x_soi[s][int(o)-1][model] 
#                                             for o in task_times]) 
#                                             <= 
#                                             self.problem_instance.takt_time*plp.lpSum([l * self.b_wtsl[w][t][s][l] for l in self.workers]), f'task_time_wts_{w}_{t}_{s}')

#         #Constraint 5 -- tasks can only be assigned to a station with the correct equipment
#         for i, model in enumerate(self.problem_instance.data.keys()):
#             for s in self.stations:
#                 for o in range(self.problem_instance.data[model]['num_tasks']):
#                     self.prob += self.x_soi[s][o][model] <= plp.lpSum([self.R_oe[o][e]*self.u_se[s][e] for  e in self.equipment]), f'equipment_soj_{s}_{o}_{model}'
#             #Constraint 6 -- precedence constraints
#         for i, model in enumerate(self.problem_instance.data.keys()):
#             for (pred, suc) in self.problem_instance.data[model]['precedence_relations']:
#                 self.prob += (plp.lpSum([ (s+1)  * self.x_soi[s][int(pred)-1][model] for s in self.stations])
#                             <=  
#                             plp.lpSum([ (s+1)  * self.x_soi[s][int(suc)-1][model] for s in self.stations]), 
#                             f'task{pred} before task{suc} for model{model} ' )
        
#         #Constraint 7 -- fixed task assignment (optional)
#         if self.fixed_assignment:
#             for i, model in enumerate(self.problem_instance.data.keys()):
#                 for i_2, model_2 in enumerate(self.problem_instance.data.keys()):
#                     if i != i_2:
#                         for s in self.stations:
#                             for o in range(self.problem_instance.data[model]['num_tasks']):
#                                 self.prob += (self.x_soi[s][o][model] == self.x_soi[s][o][model_2], f'fixed_task_{s}_{o}_{model}_{model_2}')
#         return 
        
#     def generate_report(self, file_name):
#         '''Shows task assignments for fixed and model dependent task assignment'''
#         task_assignments = []
#         labor_assignments = []
#         labor_hire_assignments = []
#         for v in self.prob.variables():
#             if round(v.varValue) > 0:
#                 if 'x_soi' in v.name:
#                     model = v.name.split('_')[4]
#                     #change the task number to match with the instances
#                     task = str(int(v.name.split('_')[3])+1)
#                     task_time = self.problem_instance.data[model]['task_times'][task]
#                     assignment = {'station': v.name.split('_')[2],'model':model  , 'task': task, 'task_times': task_time}
#                     task_assignments.append(assignment)
#                 elif 'b_wtsl' in v.name:
#                     model = self.prod_sequences[int(v.name.split('_')[2])]['sequence'][int(v.name.split('_')[4])]
#                     labor = {'scenario':v.name.split('_')[2], 'stage':v.name.split('_')[3], 'station': v.name.split('_')[4], 'model':model, 'workers': int(v.name.split('_')[5]) }
#                     labor_assignments.append(labor)
#                 elif 'Y_w' in v.name:
#                     labor_hire = {'scenario':v.name.split('_')[2], 'scenario_workers': int(v.value()) }
#                     labor_hire_assignments.append(labor_hire)

#         #turns task_assignments into a dataframe
#         task_assignments_df = pd.DataFrame(task_assignments)
#         labor_assignments_df = pd.DataFrame(labor_assignments)
#         labor_hire_df = pd.DataFrame(labor_hire_assignments)
#         #concatenates the 'task' column in task_assignments_df if the 'station' and 'model' columns are the same
#         task_assignments_df = task_assignments_df.groupby(['station', 'model'])['task', 'task_times'].agg({'task':lambda x: ','.join(x.astype(str)), 'task_times': sum }).reset_index()
#         labor_assignments_df['sequence_loc'] = labor_assignments_df['stage'].astype(int) - labor_assignments_df['station'].astype(int)
#         labor_assignments_df = labor_assignments_df[labor_assignments_df['sequence_loc'] >= 0]
#         task_seq = task_assignments_df[['station', 'model','task','task_times']]
#         #merging labor and task sequence dataframes
#         labor_task_seq = pd.merge(labor_assignments_df, task_seq, on=['model','station'], how='left')
#         labor_task_seq = pd.merge(labor_task_seq, labor_hire_df, on=['scenario'], how='left')
#         task_assignments_df.to_csv(file_name + f'task_assignment.csv', sep=' ')
#         labor_task_seq.to_csv(file_name + f'labor_assignment.csv', sep=' ')
#         return task_assignments_df, labor_assignments_df
    


class model_dependent_problem_linear_labor_recourse(MMALBP_LP_Problem):
    def __init__(self, problem_instance, equipment_instance, sequence_length, prod_sequences, fixed_assignment = False) -> None:
        self.name = 'model_dependent_problem_linear_labor_recourse'
        self.problem_instance = problem_instance
        self.equipment_instance = equipment_instance
        self.sequence_length = sequence_length
        self.prod_sequences = prod_sequences
        self.stations = list(range(problem_instance.no_stations))
        self.c_se = equipment_instance.c_se
        self.R_oe = equipment_instance.r_oe
        self.equipment = list( range(self.equipment_instance.no_equipment))
        
        self.takts = list(range(sequence_length+problem_instance.no_stations-1))
        self.prob = None
        self.u_se = None
        self.b_wtsl = None
        self.x_soi = None
        self.Y_w = None
        self.Y = None
        self.fixed_assignment = fixed_assignment
        self.make_lp_variables()
        #Results info
        self.obj_value = None
        self.solver_status = None
    
    def make_lp_variables(self):
        self.u_se = plp.LpVariable.dicts('u_se', (self.stations, self.equipment), lowBound=0, cat='Binary')
        self.l_wts = plp.LpVariable.dicts('l_wts', (self.prod_sequences.keys(), self.takts, self.stations), lowBound=0, cat='Integer')
        self.x_soi = plp.LpVariable.dicts('x_soi', ( self.stations, range(self.problem_instance.no_tasks), self.problem_instance.data.keys() ), lowBound=0, cat='Binary')
        self.Y_w = plp.LpVariable.dicts('Y_w', (self.prod_sequences.keys()), lowBound=0, cat='Integer')
        self.Y = plp.LpVariable('Y', lowBound=0, cat='Integer')

    def set_variables(self, u_se =None, l_wts = None, x_wsoj= None, x_soi=None, Y_w = None, fixed=False):
        '''Sets variables before solving the problem. If fixed is true, then the variables are fixed to their initial values. Input is a dictionary of LpVariable'''
        if u_se is not None:
            for s, equipment_dict in u_se.items():
                for equip_id in equipment_dict:
                    e = int(equip_id)
                    self.u_se[s][e].setInitialValue(round(u_se[s][equip_id].value()))
                if fixed:
                    self.u_se[s][e].fixValue()
        if l_wts is not None:
            for w in l_wts.keys():
                for t in l_wts[w].keys():
                    for s in l_wts[w][t].keys():
                        labor_val = l_wts[w][t][s].value()
                        if labor_val > 0:
                            self.l_wts[w][t][s].setInitialValue(round(labor_val))
                            if fixed:
                                self.l_wts[w][t][s].fixValue()
        if x_soi is not None:
            for s in x_soi.keys():
                for o in x_soi[s].keys():
                    for i, model in x_soi[s][o].items():
                        task_assigned = model.value()
                        self.x_soi[s][o][i].setInitialValue(round(task_assigned))
                        if fixed:
                            self.x_soi[s][o][i].fixValue()

        if Y_w is not None:
            for w in Y_w.keys():
                self.Y_w[w].setInitialValue(round(Y_w[w].value()))
                if fixed:
                    self.Y_w[w].fixValue()



    def make_lp_problem(self):
        
        #Defining LP problem
        self.prob = plp.LpProblem("model_dependent_eq_problem", plp.LpMinimize)
        #Objective function
        self.prob += (plp.lpSum([self.c_se[s][e]*self.u_se[s][e]
                            for s in self.stations
                            for e in self.equipment]
                        +
                        self.Y*self.problem_instance.worker_cost
                        +
                        [self.prod_sequences[w]['probability']*self.Y_w[w]* self.problem_instance.recourse_cost
                            for w in self.prod_sequences.keys()
                            ]),
                    "Total cost")
        #Constraints
        #Constraint 1 -- Must hire Y workers if we use Y workers in a given takt
        for w in self.prod_sequences.keys():
            for t in self.takts:
                self.prob += (plp.lpSum([self.l_wts[w][t][s] for s in self.stations]) <= plp.lpSum([self.Y_w[w] + self.Y]), f'Y_w_{w}_{t}')

        #Constraint 2 -- can only assign up to l_max number of workers to a station for a given scenario and stage
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    self.prob += (self.l_wts[w][t][s]  <= self.problem_instance.max_workers , f'worker_constraint_l_{w}_{t}_{s}')
        #Constraint 3 all tasks must be assigned to a station
        for i, model in enumerate(self.problem_instance.data.keys()):
            for o in range(self.problem_instance.no_tasks): 
                self.prob += (plp.lpSum([self.x_soi[s][o][model] for s in self.stations]) == 1, f'x_soi_{s}_{o}_{model}')
            #Constraint 4 -- sum of task times for assigned tasks must be less than takt time times the number of workers for a given station
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    #Get the model at the current scenario, stage, and station
                    if 0<= t-s < self.sequence_length:
                        j = t-s
                        model = self.prod_sequences[w]['sequence'][j]
                        task_times = self.problem_instance.data[model]['task_times']
                        self.prob += (plp.lpSum([task_times[o]*self.x_soi[s][int(o)-1][model] 
                                            for o in task_times]) 
                                            <= 
                                            self.problem_instance.takt_time* self.l_wts[w][t][s] , f'task_time_wts_{w}_{t}_{s}')

        #Constraint 5 -- tasks can only be assigned to a station with the correct equipment
        for i, model in enumerate(self.problem_instance.data.keys()):
            for s in self.stations:
                for o in range(self.problem_instance.data[model]['num_tasks']):
                    self.prob += self.x_soi[s][o][model] <= plp.lpSum([self.R_oe[o][e]*self.u_se[s][e] for  e in self.equipment]), f'equipment_soj_{s}_{o}_{model}'
            #Constraint 6 -- precedence constraints
        for i, model in enumerate(self.problem_instance.data.keys()):
            for (pred, suc) in self.problem_instance.data[model]['precedence_relations']:
                self.prob += (plp.lpSum([ (s+1)  * self.x_soi[s][int(pred)-1][model] for s in self.stations])
                            <=  
                            plp.lpSum([ (s+1)  * self.x_soi[s][int(suc)-1][model] for s in self.stations]), 
                            f'task{pred} before task{suc} for model{model} ' )
        
        #Constraint 7 -- fixed task assignment (optional)
        if self.fixed_assignment:
            for i, model in enumerate(self.problem_instance.data.keys()):
                for i_2, model_2 in enumerate(self.problem_instance.data.keys()):
                    if i != i_2:
                        for s in self.stations:
                            for o in range(self.problem_instance.data[model]['num_tasks']):
                                self.prob += (self.x_soi[s][o][model] == self.x_soi[s][o][model_2], f'fixed_task_{s}_{o}_{model}_{model_2}')
        return 
        
    def generate_report(self, file_name):
        '''Shows task assignments for fixed and model dependent task assignment'''
        task_assignments = []
        labor_assignments = []
        labor_hire_assignments = []
        fixed_labor = 0
        for v in self.prob.variables():
            if round(v.varValue) > 0:
                print(v.name, v.value())
                if 'x_soi' in v.name:
                    model = v.name.split('_')[4]
                    #change the task number to match with the instances
                    task = str(int(v.name.split('_')[3])+1)
                    task_time = self.problem_instance.data[model]['task_times'][task]
                    assignment = {'station': v.name.split('_')[2],'model':model  , 'task': task, 'task_times': task_time}
                    task_assignments.append(assignment)
                elif 'l_wts' in v.name:
                    model = self.prod_sequences[int(v.name.split('_')[2])]['sequence'][int(v.name.split('_')[4])]
                    labor = {'scenario':v.name.split('_')[2], 'stage':v.name.split('_')[3], 'station': v.name.split('_')[4], 'model':model, 'workers': int(v.value()) }
                    labor_assignments.append(labor)
                elif 'Y_w' in v.name: #otherwise 'Y" catches the scenario workers
                    pass
                elif 'Y' in v.name:
                    fixed_labor = int(v.value())
            if 'Y_w' in v.name:
                labor_hire = {'scenario':v.name.split('_')[2], 'scenario_workers': int(v.value()) }
                labor_hire_assignments.append(labor_hire)
    #turns task_assignments into a dataframe
        task_assignments_df = pd.DataFrame(task_assignments)
        labor_assignments_df = pd.DataFrame(labor_assignments)
        labor_hire_df = pd.DataFrame(labor_hire_assignments)
        labor_hire_df['fixed_workers'] = fixed_labor
        #concatenates the 'task' column in task_assignments_df if the 'station' and 'model' columns are the same
        task_assignments_df = task_assignments_df.groupby(['station', 'model'])['task', 'task_times'].agg({'task':lambda x: ','.join(x.astype(str)), 'task_times': sum }).reset_index()
        labor_assignments_df['sequence_loc'] = labor_assignments_df['stage'].astype(int) - labor_assignments_df['station'].astype(int)
        labor_assignments_df = labor_assignments_df[labor_assignments_df['sequence_loc'] >= 0]
        task_seq = task_assignments_df[['station', 'model','task','task_times']]
        #merging labor and task sequence dataframes
        labor_task_seq = pd.merge(labor_assignments_df, task_seq, on=['model','station'], how='left')
        labor_task_seq = pd.merge(labor_task_seq, labor_hire_df, on=['scenario'], how='left')
        task_assignments_df.to_csv(file_name + f'task_assignment.csv', sep=' ')
        labor_task_seq.to_csv(file_name + f'labor_assignment.csv', sep=' ')
        return task_assignments_df, labor_task_seq
    


class model_dependent_problem_multi_labor_recourse(MMALBP_LP_Problem):
    def __init__(self, problem_instance, equipment_instance, sequence_length, prod_sequences, fixed_assignment = False) -> None:
        self.name = 'model_dependent_problem_multi_labor_recourse'
        self.problem_instance = problem_instance
        self.equipment_instance = equipment_instance
        self.sequence_length = sequence_length
        self.prod_sequences = prod_sequences
        self.stations = list(range(problem_instance.no_stations))
        self.c_se = equipment_instance.c_se
        self.R_oe = equipment_instance.r_oe
        self.equipment = list( range(self.equipment_instance.no_equipment))
        self.workers = list(range(1, problem_instance.max_workers+1))
        self.takts = list(range(sequence_length+problem_instance.no_stations-1))
        self.prob = None
        self.u_se = None
        self.b_wtsl = None
        self.x_soi = None
        self.Y_w = None
        self.Y = None
        self.fixed_assignment = fixed_assignment
        self.make_lp_variables()
        #Results info
        self.obj_value = None
        self.solver_status = None
    
    def make_lp_variables(self):
        self.u_se = plp.LpVariable.dicts('u_se', (self.stations, self.equipment), lowBound=0, cat='Binary')
        self.l_wts = plp.LpVariable.dicts('l_wts', (self.prod_sequences.keys(), self.takts, self.stations), lowBound=0, cat='Integer')
        self.x_lsoi = plp.LpVariable.dicts('x_lsoi', (self.workers, self.stations, range(self.problem_instance.no_tasks), self.problem_instance.data.keys() ), lowBound=0, cat='Binary')
        self.Y_w = plp.LpVariable.dicts('Y_w', (self.prod_sequences.keys()), lowBound=0, cat='Integer')
        self.Y = plp.LpVariable('Y', lowBound=0, cat='Integer')

    #TODO fix this
    def set_variables(self, u_se =None, l_wts = None, x_lsoj= None, x_soi=None, Y_w = None, fixed=False):
        '''Sets variables before solving the problem. If fixed is true, then the variables are fixed to their initial values. Input is a dictionary of LpVariable'''
        if u_se is not None:
            for s, equipment_dict in u_se.items():
                for equip_id in equipment_dict:
                    e = int(equip_id)
                    self.u_se[s][e].setInitialValue(round(u_se[s][equip_id].value()))
                if fixed:
                    self.u_se[s][e].fixValue()
        if l_wts is not None:
            for w in l_wts.keys():
                for t in l_wts[w].keys():
                    for s in l_wts[w][t].keys():
                        labor_val = l_wts[w][t][s].value()
                        if labor_val > 0:
                            self.l_wts[w][t][s].setInitialValue(round(labor_val))
                            if fixed:
                                self.l_wts[w][t][s].fixValue()
        if x_soi is not None:
            for s in x_soi.keys():
                for o in x_soi[s].keys():
                    for i, model in x_soi[s][o].items():
                        task_assigned = model.value()
                        self.x_soi[s][o][i].setInitialValue(round(task_assigned))
                        if fixed:
                            self.x_soi[s][o][i].fixValue()

        if Y_w is not None:
            for w in Y_w.keys():
                self.Y_w[w].setInitialValue(round(Y_w[w].value()))
                if fixed:
                    self.Y_w[w].fixValue()



    def make_lp_problem(self):
        #Defining LP problem
        self.prob = plp.LpProblem("model_dependent_eq_problem", plp.LpMinimize)
        #Objective function
        self.prob += (plp.lpSum([self.c_se[s][e]*self.u_se[s][e]
                            for s in self.stations
                            for e in self.equipment]
                        +
                        self.Y*self.problem_instance.worker_cost
                        +
                        [self.prod_sequences[w]['probability']*self.Y_w[w]* self.problem_instance.recourse_cost
                            for w in self.prod_sequences.keys()
                            ]),
                    "Total cost")
        #Constraints
        #Constraint 1 -- Must hire Y workers if we use Y workers in a given takt
        for w in self.prod_sequences.keys():
            for t in self.takts:
                self.prob += (plp.lpSum([self.l_wts[w][t][s] for s in self.stations]) <= plp.lpSum([self.Y_w[w] + self.Y]), f'Y_w_{w}_{t}')

        #Constraint 2 -- can only assign up to l_max number of workers to a station for a given scenario and stage
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    self.prob += (self.l_wts[w][t][s]  <= self.problem_instance.max_workers , f'worker_constraint_l_{w}_{t}_{s}')
        #Constraint 3 all tasks must be assigned to a station
        for i, model in enumerate(self.problem_instance.data.keys()):
            for o in range(self.problem_instance.no_tasks): 
                self.prob += (plp.lpSum([self.x_lsoi[l][s][o][model] for l in self.workers for s in self.stations]) == 1, f'x_lsoi_{s}_{o}_{model}')
            #Constraint 4 -- sum of task times for assigned tasks adjusted by number of workers must be less than takt time
        for i, model in enumerate(self.problem_instance.data.keys()):
            for s in self.stations:
                #Get the model at the current scenario, stage, and station
                    task_times = self.problem_instance.data[model]['task_times']
                    self.prob += (plp.lpSum([task_times[l][o]*self.x_lsoi[l][s][int(o)-1][model] 
                                        for l in self.workers for o in task_times[l]]) 
                                        <= 
                                        self.problem_instance.takt_time , f'task_time_os_{model}_{s}')
        #Constraint 5 -- can only assign as many workers to task as there is in station
        for w in self.prod_sequences.keys():
            for t in self.takts:
                for s in self.stations:
                    #Get the model at the current scenario, stage, and station
                    for o in range(self.problem_instance.no_tasks):
                        if 0<= t-s < self.sequence_length:
                            j = t-s
                            model = self.prod_sequences[w]['sequence'][j]
                            task_times = self.problem_instance.data[model]['task_times']
                            self.prob += (plp.lpSum([l*self.x_lsoi[l][s][o][model] 
                                                for l in self.workers ]) 
                                                <= 
                                                 self.l_wts[w][t][s] , f'worker_assignment_wtso_{w}_{t}_{s}_{o}')


        #Constraint 6 -- tasks can only be assigned to a station with the correct equipment
        for i, model in enumerate(self.problem_instance.data.keys()):
            for s in self.stations:
                for o in range(self.problem_instance.data[model]['num_tasks']):
                    self.prob += plp.lpSum([self.x_lsoi[l][s][o][model] for l in self.workers])<= plp.lpSum([self.R_oe[o][e]*self.u_se[s][e] for  e in self.equipment]), f'equipment_soj_{s}_{o}_{model}'
            #Constraint 7 -- precedence constraints
        for i, model in enumerate(self.problem_instance.data.keys()):
            for (pred, suc) in self.problem_instance.data[model]['precedence_relations']:
                self.prob += (plp.lpSum([ (s+1)  * self.x_lsoi[l][s][int(pred)-1][model] for s in self.stations for l in self.workers])
                            <=  
                            plp.lpSum([ (s+1)  * self.x_lsoi[l][s][int(suc)-1][model] for s in self.stations for l in self.workers]), 
                            f'task{pred} before task{suc} for model{model} ' )
        
        #Constraint 8 -- fixed task assignment (optional)
        if self.fixed_assignment:
            for i, model in enumerate(self.problem_instance.data.keys()):
                for i_2, model_2 in enumerate(self.problem_instance.data.keys()):
                    if i != i_2:
                        for s in self.stations:
                            for l in self.workers:
                                for o in range(self.problem_instance.data[model]['num_tasks']):
                                    self.prob += (self.x_lsoi[l][s][o][model] == self.x_soi[l][s][o][model_2], f'fixed_task_{s}_{o}_{model}_{model_2}')
        return 
    
    #TODO: fix for non-linear task time reduction
    def generate_report(self, file_name):
        '''Shows task assignments for fixed and model dependent task assignment'''
        task_assignments = []
        labor_assignments = []
        labor_hire_assignments = []
        fixed_labor = 0
        for v in self.prob.variables():
            if round(v.varValue) > 0:
                print(v.name, v.value())
                if 'x_lsoi' in v.name:
                    workers = int(v.name.split('_')[2])
                    model = v.name.split('_')[5]
                    #change the task number to match with the instances
                    task = str(int(v.name.split('_')[4])+1)
                    task_time = self.problem_instance.data[model]['task_times'][workers][task]
                    assignment = {'station': v.name.split('_')[3],'workers': workers, 'model':model  , 'task': task, 'task_times': task_time}
                    task_assignments.append(assignment)
                elif 'l_wts' in v.name:
                    model = self.prod_sequences[int(v.name.split('_')[2])]['sequence'][int(v.name.split('_')[4])]
                    labor = {'scenario':v.name.split('_')[2], 'stage':v.name.split('_')[3], 'station': v.name.split('_')[4], 'model':model, 'workers': int(v.value()) }
                    labor_assignments.append(labor)
                elif 'Y_w' in v.name: #otherwise 'Y" catches the scenario workers
                    pass
                elif 'Y' in v.name:
                    fixed_labor = int(v.value())
            if 'Y_w' in v.name:
                labor_hire = {'scenario':v.name.split('_')[2], 'scenario_workers': int(v.value()) }
                labor_hire_assignments.append(labor_hire)
    #turns task_assignments into a dataframe
        task_assignments_df = pd.DataFrame(task_assignments)
        labor_assignments_df = pd.DataFrame(labor_assignments)
        labor_hire_df = pd.DataFrame(labor_hire_assignments)
        labor_hire_df['fixed_workers'] = fixed_labor
        #concatenates the 'task' column in task_assignments_df if the 'station' and 'model' columns are the same
        task_assignments_df = task_assignments_df.groupby(['station', 'model'])['task', 'task_times'].agg({'task':lambda x: ','.join(x.astype(str)), 'task_times': sum }).reset_index()
        labor_assignments_df['sequence_loc'] = labor_assignments_df['stage'].astype(int) - labor_assignments_df['station'].astype(int)
        labor_assignments_df = labor_assignments_df[labor_assignments_df['sequence_loc'] >= 0]
        task_seq = task_assignments_df[['station', 'model','task','task_times']]
        #merging labor and task sequence dataframes
        labor_task_seq = pd.merge(labor_assignments_df, task_seq, on=['model','station'], how='left')
        labor_task_seq = pd.merge(labor_task_seq, labor_hire_df, on=['scenario'], how='left')
        task_assignments_df.to_csv(file_name + f'task_assignment.csv', sep=' ')
        labor_task_seq.to_csv(file_name + f'labor_assignment.csv', sep=' ')
        return task_assignments_df, labor_task_seq