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
import argparse


def run_from_config(config_file, seed = None, base_file_name = 'test'):
   test_instances = []
   with open(config_file) as f:
      print('Opening config file', config_file)
      print('base_file_name', base_file_name)
      #Removes file extension from config file name
      conf_name = config_file.split('.')[0].split('/')[-1]
      print('conf_name', conf_name)
      xp_yaml = yaml.load(f, Loader=yaml.FullLoader)
      #configuring problem
      SEQUENCE_LENGTH = xp_yaml['sequence_length']
      NO_WORKERS = xp_yaml['max_workers']
      NO_STATIONS = xp_yaml['no_stations']
      WORKER_COST = xp_yaml['worker_cost']
      RECOURSE_COST = xp_yaml['recourse_cost']
      #configuring scenario tree generator
      tree_kwargs = {}
      if xp_yaml['scenario_generator']== 'monte_carlo_tree':
         scenario_generator = monte_carlo_tree
         tree_kwargs['n_samples'] = xp_yaml['scenario_generator']['n_samples']
         tree_kwargs['enum_depth'] = xp_yaml['scenario_generator']['enum_depth']
      else:
         scenario_generator = make_scenario_tree
      
      #copying config file to results folder
      print('copying config file to results folder',  config_file, base_file_name +'/'+ config_file)
      shutil.copyfile(config_file, base_file_name +'/'+ conf_name + '_config.yaml')
      for milp_model in xp_yaml['milp_models']:
         if milp_model == 'model_dependent_problem_multi_labor_recourse':
            milp_model = model_dependent_problem_multi_labor_recourse
            file_name = base_file_name + '/model_dependent/'
            #if model_dependent directory does not exist, make it
            if not os.path.exists(file_name):
               os.makedirs(file_name)
            file_name = file_name + 'md_'
         elif milp_model == 'dynamic_problem_multi_labor_recourse':
            milp_model = dynamic_problem_multi_labor_recourse
            file_name = base_file_name + '/dynamic/'
            #if model_dependent directory does not exist, make it
            if not os.path.exists(file_name):
               os.makedirs(file_name)
            file_name = file_name + 'd_'
         elif milp_model == 'model_dependent_problem_linear_labor_recourse':
            milp_model = model_dependent_problem_linear_labor_recourse
            file_name = base_file_name + '/model_dependent_linear/'
            #if model_dependent directory does not exist, make it
            if not os.path.exists(file_name):
               os.makedirs(file_name)
            file_name = file_name + 'lmd_'
         elif milp_model == 'dynamic_problem_linear_labor_recourse':
            milp_model = dynamic_problem_linear_labor_recourse
            file_name = base_file_name + '/dynamic_linear/'
            #if model_dependent directory does not exist, make it
            if not os.path.exists(file_name):
               os.makedirs(file_name)
            file_name = file_name + 'ld_'
         else:
            raise ValueError('milp_model not recognized')
         #Keeps track of time
         start_time = time.time()
         group_counter = 0
         for model_file in xp_yaml['model_files']:
            print('\n\n')
            print('running milp_model', milp_model)
            test_instance = MultiModelTaskTimesInstance( init_type='model_data_from_yaml',
                                             model_yaml=model_file, 
                                             sequence_length=SEQUENCE_LENGTH, 
                                             max_workers=NO_WORKERS, 
                                             no_stations=NO_STATIONS, 
                                             worker_cost=WORKER_COST, 
                                             recourse_cost=RECOURSE_COST)
            print('Running instance', test_instance.name)
            test_instances.append(test_instance)
            #create equipment
            if xp_yaml['equipment_files']:
               print('loading equipment from', xp_yaml['equipment_files'][0])
               equipment = Equipment(generation_method='import_yaml', equipment_file=xp_yaml['equipment_files'][0])
               if equipment.no_tasks != test_instance.no_tasks:
                  print('equipmen no tasks', equipment.no_tasks)
                  print('instance no tasks', test_instance.no_tasks)
                  #raises an error if the equipment and instance have different number of tasks
                  raise ValueError('Equipment and instance have different number of tasks')
            else:
               print('creating equipment')
               NO_EQUIPMENT = xp_yaml['no_equipment']
               equipment = Equipment(test_instance.no_tasks, 
                                     NO_EQUIPMENT, 
                                     NO_STATIONS, 
                                     generate_equipment, 
                                     seed)
            #create scenario tree
            print('generating scenario tree')
            model_mixtures = test_instance.model_mixtures
            scenario_tree_graph, final_sequences = scenario_generator(SEQUENCE_LENGTH, model_mixtures, **tree_kwargs)
            print('defining problem')
            milp_prob = milp_model(problem_instance = test_instance, 
                                   equipment_instance = equipment, 
                                   sequence_length=SEQUENCE_LENGTH, 
                                   prod_sequences=final_sequences)
            start = timer()
            solver = plp.GUROBI_CMD(options=[ ('TimeLimit', 600), ('LogFile', file_name+conf_name + str(group_counter) + ".log")])#
            milp_prob.solve(solver=solver, 
                            file_name=file_name + conf_name+ str(group_counter))
            end = timer()
            result = milp_prob.get_obj_val_dict()
            result['run_time'] = end - start
            result_df = pd.DataFrame([result], index=[0])
            if group_counter == 0:
               results_df = result_df.copy()
            else:
               results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
            output_path=file_name + conf_name +  '_results.csv'
            results_df.to_csv(output_path)
            group_counter += 1
         #deletes results df
         del results_df
         end_time = time.time()
         print('time for', milp_model, end_time - start_time)
   return 1

# today = datetime.today().strftime('%Y_%m_%d')
# xp_name = f"xp_{today}_mc_debug"
# if not os.path.exists('model_runs/'+ xp_name):
#    os.makedirs('model_runs/'+xp_name)
   
# file_name = 'model_runs/'+xp_name
# run_from_config('SALBP_benchmark/MM_instances/small_instance.yaml',
#            base_file_name=file_name)



#main function that runs if this file is run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs ALB model from config file')
    parser.add_argument('config_file', type=str, help='config file to run')
    parser.add_argument('--seed', type=int, help='seed for random number generator')
    parser.add_argument('--xp_name', type=str, help='directory to save results')
    args = parser.parse_args()
    today = datetime.today().strftime('%Y_%m_%d')
    if args.xp_name:
        print("Writing output to model_runs/"+str(today)+args.xp_name)
        if not os.path.exists('model_runs/'+str(today)+args.xp_name):
            os.makedirs('model_runs/'+str(today)+args.xp_name)
        file_name = 'model_runs/'+str(today)+args.xp_name
        run_from_config(args.config_file, seed=args.seed, base_file_name=file_name)
    else:
        if not os.path.exists('model_runs/test'):
            os.makedirs('model_runs/test')
        file_name = 'model_runs/'+str(today)+ 'test'
        run_from_config(args.config_file, seed=args.seed, file_name=file_name)