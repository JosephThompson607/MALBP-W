import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pulp as plp
import networkx as nx
from ALB_instance_tools import *
from report_functions import *
from milp_models import *
from scenario_trees import *
from collections import namedtuple
import glob
import os
import string
import time
import shutil
from datetime import datetime
from timeit import default_timer as timer




def random_station_deconstructor(vars , station_block_size = 2):
    '''Randomly chooses a block of stations to be not fixed'''
    x_wsoj, u_se, l_wts, y_w, y_def = vars
    no_scenarios = len(x_wsoj)
    no_stations = len(x_wsoj[0])
    no_tasks = len(x_wsoj[0][0])    

    #Randomly chooses a block of station_block_size stations to be not fixed
    start_station = np.random.randint(0, no_stations - station_block_size)
    end_station = start_station + station_block_size
    loose_stations = list(range(start_station, end_station))
    #filters the x_wsoj dict to only include the loose stations
    not_fixed_x_wsoj = {}
    fixed_x_wsoj = {}
    for w in range(no_scenarios):
        not_fixed_x_wsoj[w] = {}   
        fixed_x_wsoj[w] = {}
        for s in range(no_stations):
            if s in loose_stations:
                not_fixed_x_wsoj[w][s] = x_wsoj[w][s]
            else:
                fixed_x_wsoj[w][s] = x_wsoj[w][s]
    fixed_vars = {'x_wsoj': fixed_x_wsoj, 'u_se': None, 'l_wts': None, 'y_w': None, 'y': None}
    not_fixed_vars = {'x_wsoj': not_fixed_x_wsoj, 'u_se': u_se, 'l_wts': l_wts, 'y_w': y_w, 'y': y_def}
    return fixed_vars, not_fixed_vars



def fix_and_optimize_dl( deconstructor,md_results_folder, problem_instance, equipment_instance,prod_sequences, fp, group_counter=0, n_iter = 3, run_time = 600, **kwargs):
    '''fixes the variables in the problem instance and optimizes'''
    #creates the dynamic problem
    sequence_length = len(prod_sequences[0]['sequence'])
    print('defining dynamic problem')
    dynamic_problem = dynamic_problem_linear_labor_recourse(problem_instance, equipment_instance,sequence_length, prod_sequences)
    print('importing results from model dependent problem')
    #loads the results from the model dependent problem
    start_time = timer()
    dynamic_problem.set_up_from_model_dependent(md_results_folder)
    prev_vars = dynamic_problem.get_variables()
    print('x_wsoj', prev_vars[0])
    results_batch = []
    solver = plp.GUROBI_CMD(options=[ ('TimeLimit', run_time)])
    for i in range(n_iter):
        print('solving dynamic problem')
        fix_vars, not_fix_vars = deconstructor(prev_vars, **kwargs)
        print('fix_vars', fix_vars)
        print('not_fix_vars', not_fix_vars)
        dynamic_problem.set_variables(**fix_vars, fixed=True)
        dynamic_problem.set_variables(**not_fix_vars, fixed=False)
        #prints the lp problem
        dynamic_problem.solve(solver=solver, file_name=fp + str(i))
        #Gets the previous results for use in the next iteration
        prev_vars = dynamic_problem.get_variables() 
        end_time = timer()
        result = dynamic_problem.get_obj_val_dict()
        result['run_time'] = end_time - start_time
        result['iteration'] = i
        result['group_counter'] = group_counter
        results_batch.append(result)
        #Resetting dynamic problem to original state
        dynamic_problem = dynamic_problem_linear_labor_recourse(problem_instance, equipment_instance,sequence_length, prod_sequences)
    result_df = pd.DataFrame(results_batch)
    return result_df

def run_fix_and_optimize(param_dict_list, base_xp_yaml, base_file_name, deconstructor, run_time = 600, **kwargs):
    '''runs the fix and optimize function with the given parameters'''
    print('Opening config file', base_xp_yaml)
    print('base_file_name', base_file_name)
    #Removes file extension from config file name
    conf_name = base_xp_yaml.split('.')[0].split('/')[-1]
    with open(base_xp_yaml) as f:
        xp_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #configuring problem
    SEQUENCE_LENGTH = xp_yaml['sequence_length']
    NO_WORKERS = xp_yaml['max_workers']
    NO_STATIONS = xp_yaml['no_stations']
    WORKER_COST = xp_yaml['worker_cost']
    RECOURSE_COST = xp_yaml['recourse_cost']
    shutil.copyfile(base_xp_yaml, base_file_name +'/'+ conf_name + '_config.yaml')
    out_folder = base_file_name + '/dynamic_problem_linear_labor_recourse_fo/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    #loads the model file
    group_counter = 0
    for param_dict in param_dict_list:
        test_instance = MultiModelTaskTimesInstance( init_type='model_data_from_yaml',
                                                    model_yaml=param_dict['model_file'], 
                                                    sequence_length=SEQUENCE_LENGTH, 
                                                    max_workers=NO_WORKERS, 
                                                    no_stations=NO_STATIONS, 
                                                    worker_cost=WORKER_COST, 
                                                    recourse_cost=RECOURSE_COST)
        #loads the equipment file
        equipment_instance = Equipment(generation_method='import_yaml', equipment_file=param_dict['equipment_instance'])
        #makes the scenario tree
        scenario_tree, final_sequences = make_scenario_tree(SEQUENCE_LENGTH, test_instance.model_mixtures)
        #makes directory if it does not exist
        if not os.path.exists(base_file_name):
            os.makedirs(base_file_name)
        #runs the fix and optimize function
        result_df = fix_and_optimize_dl(deconstructor, 
                                        param_dict['start_var_folder'],
                                        test_instance, 
                                        equipment_instance, 
                                        final_sequences,                             
                                        out_folder, 
                                        run_time=run_time,
                                        group_counter=group_counter,
                                        **kwargs)
        if group_counter == 0:
            results_df = result_df.copy()
        else:
            results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
        #writes the results to a csv file
        results_df.to_csv(out_folder + 'results.csv')
        group_counter += 1
    return results_df


#main function
if __name__ == "__main__":
    param_dict_list = [{'start_var_folder': 'model_runs/xp_23_12_07_ws_15_16_17_seq5/model_dependent_linear/small_instance_hard0_variables/',
                    'model_file': 'SALBP_benchmark/MM_instances/model_data/small_instances/3_models/n=20_14_n=20_15_n=20_16.yaml',
                    'equipment_instance': 'SALBP_benchmark/MM_instances/equipment_data/random_O20_E4_S4_seed42.yaml',},
                    {'start_var_folder': 'model_runs/xp_23_12_07_ws_15_16_17_seq5/model_dependent_linear/small_instance_hard1_variables/',
                     'model_file': 'SALBP_benchmark/MM_instances/model_data/small_instances/3_models/n=20_15_n=20_16_n=20_17.yaml',
                     'equipment_instance': 'SALBP_benchmark/MM_instances/equipment_data/random_O20_E4_S4_seed42.yaml',},]
    today = datetime.today().strftime('%y_%m_%d')
    xp_name = f"xp_{today}_fo"
    base_file_name = 'model_runs/'+xp_name
    if not os.path.exists(base_file_name):
        os.makedirs(base_file_name)
    run_fix_and_optimize(param_dict_list, 'SALBP_benchmark/MM_instances/small_instance_hard.yaml', base_file_name, random_station_deconstructor, run_time = 600)