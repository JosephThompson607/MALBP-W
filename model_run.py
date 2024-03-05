import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pulp as plp
import networkx as nx
from ALB_instance_tools import *
#from report_functions import *
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


def dict_list_to_csv(dict_list, file_name):
    '''writes a list of dictionaries to a csv file'''
    keys = dict_list[0].keys()
    with open(file_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)


def warmstart_dynamic_problem_linear_labor_recourse( problem_instance, equipment_instance, production_sequences, md_results_folder,file_name, group_counter, save_variables=False, run_time =600):
    '''warmstarts the dynamic problem with the results from the model dependent problem'''
    sequence_length = len(production_sequences[0]['sequence'])
    #creates the dynamic problem
    print('defining dynamic problem')
    dynamic_problem = dynamic_problem_linear_labor_recourse(problem_instance, equipment_instance,sequence_length, production_sequences)
    print('importing results from model dependent problem')
    #loads the results from the model dependent problem
    start_time = timer()
    dynamic_problem.set_up_from_model_dependent(md_results_folder)
    print('solving dynamic problem: ', problem_instance.name, flush=True)
    solver = plp.GUROBI_CMD(warmStart=True,options=[ ('TimeLimit', run_time), ('LogFile', f"{file_name}{group_counter}.log")])
    
    dynamic_problem.solve(solver=solver, file_name=file_name + str(group_counter))
    folder_name = None
    if save_variables:
         folder_name = file_name + str(group_counter) + '_variables/'
         if not os.path.exists(folder_name):
                  os.makedirs(folder_name)
         dynamic_problem.save_variables(folder_name)
    end_time = timer()
    result = dynamic_problem.get_obj_val_dict()
    result['run_time'] = end_time - start_time
    #result['model_type'] = 'dynamic_problem_linear_labor_recourse'
    result_df = pd.DataFrame([result], index=[0])
    return result_df, folder_name

def warmstart_dynamic_from_results(results_file, base_file_name = 'test', run_time = 600, seed=None, save_variables=False):
   ''' warm starts dynamic linear labor problem from the results file of a model dependent run'''
   res_dict_list = dict_list_from_csv(results_file)
   #,model_name,instance_name,obj_value,solver_status,
   # equipment_instance,xp_config_file,model_file,variables_folder,run_time,group_counter
   print('using results from ', results_file)
   if res_dict_list[0]['model_name'] != 'model dependent problem linear labor recourse':
      raise ValueError('Results file is not from a model dependent problem')
   group_counter = 0
   start_time = time.time()

   for start_results in res_dict_list:
      print(start_results)
      config_file = start_results['xp_config_file']

      print('config_file', config_file)
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

      equipment_instance = Equipment(generation_method='import_yaml', equipment_file=start_results['equipment_instance'])
      test_instance = MultiModelTaskTimesInstance( init_type='model_data_from_yaml',
                                          model_yaml=start_results['model_file'], 
                                          sequence_length=SEQUENCE_LENGTH, 
                                          max_workers=NO_WORKERS, 
                                          no_stations=NO_STATIONS, 
                                          worker_cost=WORKER_COST, 
                                          recourse_cost=RECOURSE_COST)
      #create scenario tree
      tree_kwargs, scenario_generator = get_scenario_generator(xp_yaml, seed)
      print('generating scenario tree')
      model_mixtures = test_instance.model_mixtures
      _, final_sequences = scenario_generator(SEQUENCE_LENGTH, model_mixtures, **tree_kwargs)
      #Warm starts the dynamic problem with the results from the model dependent problem
      warm_start_fp = base_file_name + '/dynamic_problem_linear_labor_recourse/warmstart/'
      if not os.path.exists(warm_start_fp):
         os.makedirs(warm_start_fp)
      ws_res_df, var_folder = warmstart_dynamic_problem_linear_labor_recourse(test_instance, 
                                                                                    equipment_instance, 
                                                                                    final_sequences,
                                                                                    start_results['variables_folder'],
                                                                                    warm_start_fp,
                                                                                    group_counter,
                                                                                    run_time = run_time,
                                                                                    save_variables=save_variables)
      ws_res_df['equipment_instance'] = xp_yaml['equipment_files'][0]
      ws_res_df['xp_config_file'] = config_file
      ws_res_df['model_file'] = start_results['model_file']
      ws_res_df['variables_folder'] = var_folder
      ws_res_df['original_run_time'] = start_results['run_time']
      ws_res_df['original_obj_value'] = start_results['obj_value']
      if group_counter == 0:
            results_df = ws_res_df.copy()
      else:
         results_df = pd.concat([results_df, ws_res_df], axis=0, ignore_index=True)
      group_counter += 1
      output_path=base_file_name + '/results.csv'
      results_df.to_csv(output_path)   
   end_time = time.time()
   print('time for', 'dynamic_problem_linear_labor_recourse', end_time - start_time)

      



# def warm_start_from_config(config_file, seed = None, base_file_name = 'test', run_time = 600):
#    test_instances = []
#    with open(config_file) as f:
#       print('Opening config file', config_file)
#       print('base_file_name', base_file_name)
#       #Removes file extension from config file name
#       conf_name = config_file.split('.')[0].split('/')[-1]
#       print('conf_name', conf_name)
#       xp_yaml = yaml.load(f, Loader=yaml.FullLoader)
#       #configuring problem
#       SEQUENCE_LENGTH = xp_yaml['sequence_length']
#       NO_WORKERS = xp_yaml['max_workers']
#       NO_STATIONS = xp_yaml['no_stations']
#       WORKER_COST = xp_yaml['worker_cost']
#       RECOURSE_COST = xp_yaml['recourse_cost']
#       #configuring scenario tree generator
#       tree_kwargs = {}
#       if xp_yaml['scenario_generator']== 'monte_carlo_tree':
#          scenario_generator = monte_carlo_tree
#          tree_kwargs['n_samples'] = xp_yaml['scenario_generator']['n_samples']
#          tree_kwargs['enum_depth'] = xp_yaml['scenario_generator']['enum_depth']
#       else:
#          scenario_generator = make_scenario_tree
      
#       #copying config file to results folder
#       print('copying config file to results folder',  config_file, base_file_name +'/'+ config_file)
#       shutil.copyfile(config_file, base_file_name +'/'+ conf_name + '_config.yaml')
#       #TODO make this work for other milp models
#       milp_model = model_dependent_problem_linear_labor_recourse
#       file_name = base_file_name + '/model_dependent_linear/'
#       #if model_dependent directory does not exist, make it
#       if not os.path.exists(file_name):
#          os.makedirs(file_name)
         
#       start_time = time.time()
#       group_counter = 0
#       for model_file in xp_yaml['model_files']:
#          print('\n\n')
#          print('running milp_model', milp_model)
#          test_instance = MultiModelTaskTimesInstance( init_type='model_data_from_yaml',
#                                           model_yaml=model_file, 
#                                           sequence_length=SEQUENCE_LENGTH, 
#                                           max_workers=NO_WORKERS, 
#                                           no_stations=NO_STATIONS, 
#                                           worker_cost=WORKER_COST, 
#                                           recourse_cost=RECOURSE_COST)
#          print('Running instance', test_instance.name)
#          test_instances.append(test_instance)
#          #create equipment
#          if xp_yaml['equipment_files']:
#             print('loading equipment from', xp_yaml['equipment_files'][0])
#             equipment = Equipment(generation_method='import_yaml', equipment_file=xp_yaml['equipment_files'][0])
#             if equipment.no_tasks != test_instance.no_tasks:
#                print('equipmen no tasks', equipment.no_tasks)
#                print('instance no tasks', test_instance.no_tasks)
#                #raises an error if the equipment and instance have different number of tasks
#                raise ValueError('Equipment and instance have different number of tasks')
#          else:
#             print('creating equipment')
#             NO_EQUIPMENT = xp_yaml['no_equipment']
#             equipment = Equipment(test_instance.no_tasks, 
#                                     NO_EQUIPMENT, 
#                                     NO_STATIONS, 
#                                     generate_equipment, 
#                                     seed)
#          #create scenario tree
#          print('generating scenario tree')
#          model_mixtures = test_instance.model_mixtures
#          scenario_tree_graph, final_sequences = scenario_generator(SEQUENCE_LENGTH, model_mixtures, **tree_kwargs)
#          print('defining problem')
#          milp_prob = milp_model(problem_instance = test_instance, 
#                                  equipment_instance = equipment, 
#                                  sequence_length=SEQUENCE_LENGTH, 
#                                  prod_sequences=final_sequences)
#          start = timer()
#          solver = plp.GUROBI_CMD(options=[ ('TimeLimit', run_time), ('LogFile', file_name+conf_name + str(group_counter) + ".log")])#
#          milp_prob.solve(solver=solver, 
#                            file_name=file_name + conf_name+ str(group_counter))
#          #Saves lp variables to be used by the dynamic problem
#          folder_name = file_name + conf_name + str(group_counter) + test_instance.name+ '_variables/'
#          if not os.path.exists(folder_name):
#             os.makedirs(folder_name)
#          milp_prob.save_variables(folder_name)
#          end = timer()
#          result = milp_prob.get_obj_val_dict()
#          result['run_time'] = end - start
#          #var_folder,model_file,equipment_instance
#          result['equipment_instance'] = xp_yaml['equipment_files'][0]
#          result['xp_config_file'] = config_file
#          result['model_file'] = model_file
#          result['variables_folder'] = folder_name
#          #result['model_type'] = 'model_dependent'
#          result_df = pd.DataFrame([result], index=[0])
#          if group_counter == 0:
#             results_df = result_df.copy()
#          else:
#             results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
         
         
#          #Warm starts the dynamic problem with the results from the model dependent problem
#          warm_start_fp = base_file_name + '/dynamic_problem_linear_labor_recourse/warmstart/'
#          if not os.path.exists(warm_start_fp):
#             os.makedirs(warm_start_fp)
#          ws_res_df = warmstart_dynamic_problem_linear_labor_recourse(test_instance, 
#                                                                                     equipment, 
#                                                                                     final_sequences,
#                                                                                     folder_name,
#                                                                                     warm_start_fp,
#                                                                                     group_counter,
#                                                                                     run_time = run_time)
#          results_df = pd.concat([results_df, ws_res_df], axis=0, ignore_index=True)
#          group_counter += 1
#          output_path=file_name + conf_name +  '_results.csv'
#          results_df.to_csv(output_path)   
#          end_time = time.time()
#          print('time for', milp_model, end_time - start_time)
#    return 1



def run_from_config(config_file, save_variables=False, run_time = 600, seed = None, base_file_name = 'test'):
   '''runs the model from a config file'''
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
      tree_kwargs, scenario_generator = get_scenario_generator(xp_yaml, seed)
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
            solver = plp.GUROBI_CMD(options=[ ('TimeLimit', run_time), ('LogFile', file_name+conf_name + str(group_counter) + ".log")])#
            milp_prob.solve(solver=solver, 
                            file_name=file_name + conf_name+ str(group_counter))
            end = timer()
            folder_name = None
            if save_variables:
               folder_name = file_name + conf_name + str(group_counter) + test_instance.name+ '_variables/'
               if not os.path.exists(folder_name):
                  os.makedirs(folder_name)
               milp_prob.save_variables(folder_name)
            result = milp_prob.get_obj_val_dict()
            result['equipment_instance'] = xp_yaml['equipment_files'][0]
            result['xp_config_file'] = config_file
            result['model_file'] = model_file
            result['variables_folder'] = folder_name
            result['run_time'] = end - start
            result['group_counter'] = group_counter
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

def arg_parse():
   '''parses arguments from command line'''
   parser = argparse.ArgumentParser(description='Runs ALB model from config file')
   parser.add_argument('--xp_type', type=str, help='type of experiment to run')
   parser.add_argument('--config_file', type=str, help='config file to run')
   parser.add_argument('--seed', type=int, required=False, default=None, help='seed for random number generator')
   parser.add_argument('--xp_name', type=str, help='directory to save results')
   parser.add_argument('--run_time', type=int, help='time limit for solver')
   parser.add_argument('--save_variables', action=argparse.BooleanOptionalAction, help='whether to save the lp variables')
   args = parser.parse_args()
   return args


def main_run():
   '''runs the model from the command line'''
   args = arg_parse()
   today = datetime.today().strftime('%Y_%m_%d')
   if args.xp_name:
      print("Writing output to model_runs/"+str(today)+args.xp_name)
      if not os.path.exists('model_runs/'+str(today)+args.xp_name):
         os.makedirs('model_runs/'+str(today)+args.xp_name)
      file_name = 'model_runs/'+str(today)+args.xp_name
   else:
      if not os.path.exists('model_runs/test'):
         os.makedirs('model_runs/test')
      file_name = 'model_runs/'+str(today)+ 'test'
   if args.xp_type == 'warm_start':
      print('running warm start')
      warmstart_dynamic_from_results(args.config_file, seed=args.seed, base_file_name=file_name, run_time=args.run_time, save_variables=args.save_variables)
   else:
      print('running from config file')
      run_from_config(args.config_file, seed=args.seed, run_time=args.run_time, base_file_name=file_name, save_variables=args.save_variables)


#main function that runs if this file is run
if __name__ == "__main__":
   main_run()