import numpy as np
import pandas as pd
import pulp as plp
import networkx as nx
from ALB_instance_tools import *
from milp_models import *
from scenario_trees import *
import os
import shutil
from datetime import datetime
from timeit import default_timer as timer
import argparse


class deconstructor():
    '''class for deconstructing the variables'''
    def __init__(self, block_size = 2, depth = 1, no_improvement_limit=2):
        self.original_depth = depth
        self.depth = depth
        self.original_block_size = block_size
        self.block_size = block_size
        self.original_deconstructor_method = self.random_model_deconstructor
        self.deconstructor_method = self.random_model_deconstructor
        self.deconstructor_modifier = self.no_change
        self.no_improvement_limit = no_improvement_limit
        self.no_improvement_counter = 0
        self.deconstructor_name = None
        self.y_decrement = 0

    def reset(self):
        self.no_improvement_counter = 0
        self.block_size = self.original_block_size
        self.depth = self.original_depth
        self.deconstructor_method = self.original_deconstructor_method
        self.deconstructor_name = None


    def no_change(self, result):
        '''does nothing'''
        return None
    
    def increase_block_size(self, result ):
        '''increases the block size if no improvement has been made'''
        if self.no_improvement_counter >= self.no_improvement_limit:
            print('increasing block size')
            self.block_size += 1
            self.no_improvement_counter = 0

    def decrease_depth_increase_block(self, result):
        '''decreases the depth and increases the block size if no improvement has been made'''
        self.depth = self.depth - 1
        if self.no_improvement_counter >= self.no_improvement_limit:
            print(' increasing block size, resetting starting depth')
            self.block_size += 1
            self.no_improvement_counter = 0
            self.depth = self.original_depth

    def fix_y(self, result):
        '''fixes the y variables'''
        if self.no_improvement_counter >= self.no_improvement_limit :
            self.y_decrement = 1
            if self.deconstructor_name != 'fix_y_deconstructor':
                print('fixing y')
                def new_deconstructor_method(vars, dynamic_problem):
                    return self.fix_y_deconstructor(*self.original_deconstructor_method(vars, dynamic_problem))
                self.deconstructor_method = new_deconstructor_method
                self.no_improvement_counter = 0
            else:
                print('going back to original deconstructor')
                self.deconstructor_method = self.original_deconstructor_method
                self.deconstructor_name = None
                self.no_improvement_counter = 0
        else:
            self.y_decrement = 0

    def update_deconstructor(self, result):
        '''updates the deconstructor'''
        self.deconstructor_modifier(result)


    def set_deconstructor(self, deconstructor_method):
        self.deconstructor_method = deconstructor_method
        self.original_deconstructor_method = deconstructor_method
        
    def deconstruct(self, vars, dynamic_problem):
        return self.deconstructor_method(vars, dynamic_problem)
    
    def random_subtree_deconstructor(self, vars, dynamic_problem):
        '''Randomly chooses a subtree to not be fixed'''
        x_wsoj, u_se, l_wts, y_w, y_def = vars
        no_scenarios = len(x_wsoj)
        model_dict = dynamic_problem.problem_instance.model_mixtures
        model_keys = list(model_dict.keys())
        test_seq = []
        depth = max(self.depth, 1)
        block_size = min(self.block_size, len(model_keys)**depth)
        #TODO fix it so that no two subtrees are not the same
        for j in range(block_size):
            models = list(np.random.choice(model_keys, depth, replace=True))
            test_seq.append(models)
        not_fixed_x_wsoj = {}
        fixed_x_wsoj = {}
        for w in range(no_scenarios):
            for j in range(block_size):
                #If the first depth number of elements of the sequence match the test sequence, then the sequence is not fixed
                print('test_seq[j]', test_seq[j])
                print('dynamic_problem.prod_sequences[w][sequence]', dynamic_problem.prod_sequences[w]['sequence'])
                if dynamic_problem.prod_sequences[w]['sequence'][:depth] == test_seq[j]:
                    not_fixed_x_wsoj[w] = x_wsoj[w]
                    print('not fixed', not_fixed_x_wsoj[w])
                else:
                    fixed_x_wsoj[w] = x_wsoj[w]
        fixed_vars = {'x_wsoj': fixed_x_wsoj, 'u_se': None, 'l_wts': None, 'y_w': None, 'y': None}
        not_fixed_vars = {'x_wsoj': not_fixed_x_wsoj, 'u_se': u_se, 'l_wts': l_wts, 'y_w': y_w, 'y': y_def}
        return fixed_vars, not_fixed_vars

    def fix_y_deconstructor(self, fixed, not_fixed):
        '''decreases by 1 and fixes the y variables'''
        self.deconstructor_name = 'fix_y_deconstructor'
        #Only fixes the first stage y value if it is not already fixed
        not_fixed_vars = {'x_wsoj': not_fixed['x_wsoj'], 'u_se': not_fixed['u_se'], 'l_wts': not_fixed['l_wts'], 'y_w': not_fixed['y_w'], 'y': None}
        fixed_vars = {'x_wsoj': fixed['x_wsoj'], 'u_se': fixed['u_se'], 'l_wts': fixed['l_wts'], 'y_w': fixed['y_w'], 'y': not_fixed['y'] -self.y_decrement}
        return fixed_vars, not_fixed_vars
    
    def random_sequence_deconstructor(self, vars, dynamic_problem):
        '''Randomly chooses a set of sequences to not be fixed'''
        print('using random_sequence_deconstructor')
        x_wsoj, u_se, l_wts, y_w, y_def = vars
        no_scenarios = len(x_wsoj)
        scenario_list = list(range(no_scenarios))
        block_size = min(self.block_size, no_scenarios)
        scenarios = np.random.choice(scenario_list, block_size, replace=False)
        not_fixed_x_wsoj = {}
        fixed_x_wsoj = {}

        for w in range(no_scenarios):
            if w in scenarios:
                not_fixed_x_wsoj[w] = x_wsoj[w]
            else:
                fixed_x_wsoj[w] = x_wsoj[w]
        fixed_vars = {'x_wsoj': fixed_x_wsoj, 'u_se': None, 'l_wts': None, 'y_w': None, 'y': None}
        not_fixed_vars = {'x_wsoj': not_fixed_x_wsoj, 'u_se': u_se, 'l_wts': l_wts, 'y_w': y_w, 'y': y_def}
        return fixed_vars, not_fixed_vars


    def random_model_deconstructor(self, vars, dynamic_problem):
        '''Randomly chooses a block of models to be not fixed'''
        print('using random_model_deconstructor')
        model_dict = dynamic_problem.problem_instance.model_mixtures
        x_wsoj, u_se, l_wts, y_w, y_def = vars
        no_scenarios = len(x_wsoj)
        no_stations = len(x_wsoj[0])
        no_tasks = len(x_wsoj[0][0])
        seq_length = len(x_wsoj[0][0][0])
        block_size = min(self.block_size, len(model_dict))
        #Randomly chooses a model block of model_block_size models to be not fixed from model dict keys
        model_keys = list(model_dict.keys())
        models = np.random.choice(model_keys, block_size, replace=False)
        #filters the x_wsoj dict into two dicts, one with the fixed variables and one with the not fixed variables
        not_fixed_x_wsoj = {}
        fixed_x_wsoj = {}
        
        for w in range(no_scenarios):
            not_fixed_x_wsoj[w] = {}   
            fixed_x_wsoj[w] = {}
            for s in range(no_stations):
                not_fixed_x_wsoj[w][s] = {}
                fixed_x_wsoj[w][s] = {}
                for o in range(no_tasks):
                    not_fixed_x_wsoj[w][s][o] = {}
                    fixed_x_wsoj[w][s][o] = {}
                    for j in range(seq_length):
                        model = dynamic_problem.prod_sequences[w]['sequence'][j]
                        if model in models:
                            not_fixed_x_wsoj[w][s][o][j] = x_wsoj[w][s][o][j]
                        else:
                            fixed_x_wsoj[w][s][o][j] = x_wsoj[w][s][o][j]
        fixed_vars = {'x_wsoj': fixed_x_wsoj, 'u_se': None, 'l_wts': None, 'y_w': None, 'y': None}
        not_fixed_vars = {'x_wsoj': not_fixed_x_wsoj, 'u_se': u_se, 'l_wts': l_wts, 'y_w': y_w, 'y': y_def}
        return fixed_vars, not_fixed_vars




    def random_station_deconstructor(self, vars , dynamic_problem):
        '''Randomly chooses a block of stations to be not fixed'''
        x_wsoj, u_se, l_wts, y_w, y_def = vars
        no_scenarios = len(x_wsoj)
        no_stations = len(x_wsoj[0])
        no_tasks = len(x_wsoj[0][0])    
        #TODO: fix l_wts and u_se for stations that are fixed
        block_size = min(self.block_size, no_stations)
        #Randomly chooses a block of station_block_size stations to be not fixed
        start_station = np.random.randint(0, no_stations - block_size+1)
        end_station = start_station + block_size
        loose_stations = list(range(start_station, end_station))
        #filters the x_wsoj dict into two dicts, one with the fixed variables and one with the not fixed variables
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
        #filters the u_se into two dicts, one with the fixed variables and one with the not fixed variables
        not_fixed_u_se = {}
        fixed_u_se = {}
        for s in range(no_stations):
            if s in loose_stations:
                not_fixed_u_se[s] = u_se[s]
            else:
                fixed_u_se[s] = u_se[s]
        fixed_vars = {'x_wsoj': fixed_x_wsoj, 'u_se': fixed_u_se, 'l_wts': None, 'y_w': None, 'y': None}
        not_fixed_vars = {'x_wsoj': not_fixed_x_wsoj, 'u_se': not_fixed_u_se, 'l_wts': l_wts, 'y_w': y_w, 'y': y_def}
        return fixed_vars, not_fixed_vars



def fix_and_optimize_dl( deconstructor,md_results_folder,start_obj_value, problem_instance, equipment_instance,prod_sequences, fp, group_counter=0, n_iter = 3, run_time = 600, total_run_time=None, no_improvement=2, **kwargs):
    '''fixes the variables in the problem instance and optimizes'''
    #creates the dynamic problem
    sequence_length = len(prod_sequences[0]['sequence'])
    print('defining dynamic problem')
    dynamic_problem = dynamic_problem_linear_labor_recourse(problem_instance, equipment_instance,sequence_length, prod_sequences)
    print('importing results from model dependent problem')
    #loads the results from the model dependent problem
    start_time = timer()
    dynamic_problem.set_up_from_model_dependent(md_results_folder)
    prev_best = start_obj_value
    prev_vars = dynamic_problem.get_variables()
    results_batch = []
    solver = plp.GUROBI_CMD(warmStart=True,options=[ ('TimeLimit', run_time)])
    for i in range(n_iter):
        print('solving dynamic problem: ', problem_instance.name, 'iteration: ', n_iter, flush=True)
        fix_vars, not_fix_vars = deconstructor.deconstruct(vars=prev_vars,dynamic_problem=dynamic_problem)
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
        #Checks if the total run time has been exceeded
        if total_run_time is not None:
            if end_time - start_time > total_run_time:
                print('total run time exceeded')
                print('ending at iteration counter', i)
                break
        #Checks if the objective value has improved
        if result['obj_value'] < prev_best:
            prev_best = result['obj_value']
            deconstructor.no_improvement_counter = 0
        else:
            deconstructor.no_improvement_counter += 1
        #Updating the deconstructor
        deconstructor.update_deconstructor(result)
        #Resetting dynamic problem to original state
        dynamic_problem = dynamic_problem_linear_labor_recourse(problem_instance, equipment_instance,sequence_length, prod_sequences)
    result_df = pd.DataFrame(results_batch)
    return result_df

def run_fix_and_optimize(param_dict_list, base_file_name, deconstructor, run_time = 600, total_run_time=None, **kwargs):
    '''runs the fix and optimize function with the given parameters'''
    
    #loads the model file
    group_counter = 0
    for param_dict in param_dict_list:
        base_xp_yaml = param_dict['xp_config_file']
        print('Opening config file', base_xp_yaml)
        print('base_file_name', base_file_name)
        print('model_file', param_dict['model_file'])
        print('equipment_instance', param_dict['equipment_instance'])
        #Removes file extension from config file name
        with open(base_xp_yaml) as f:
            xp_yaml = yaml.load(f, Loader=yaml.FullLoader)
        #configuring problem
        SEQUENCE_LENGTH = xp_yaml['sequence_length']
        NO_WORKERS = xp_yaml['max_workers']
        NO_STATIONS = xp_yaml['no_stations']
        WORKER_COST = xp_yaml['worker_cost']
        RECOURSE_COST = xp_yaml['recourse_cost']
        out_folder = base_file_name + 'dynamic_problem_linear_labor_recourse_fo/'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
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
        #TODO fix seed for mc tree generator
        tree_kwargs, scenario_tree_generator = get_scenario_generator(xp_yaml)
        scenario_tree, final_sequences = scenario_tree_generator(SEQUENCE_LENGTH, test_instance.model_mixtures, **tree_kwargs)
        #makes directory if it does not exist
        if not os.path.exists(base_file_name):
            os.makedirs(base_file_name)
        #runs the fix and optimize function
        result_df = fix_and_optimize_dl(deconstructor, 
                                        param_dict['variables_folder'],
                                        float(param_dict['obj_value']),
                                        test_instance, 
                                        equipment_instance, 
                                        final_sequences,                             
                                        out_folder, 
                                        run_time=run_time,
                                        total_run_time=total_run_time,
                                        group_counter=group_counter,
                                        **kwargs)
        result_df['base_xp_config'] = base_xp_yaml
        result_df['original_obj_value'] = param_dict['obj_value']
        result_df['original_run_time'] = param_dict['run_time']
        result_df['model_file'] = param_dict['model_file']
        result_df['equipment_instance'] = param_dict['equipment_instance']
        if group_counter == 0:
            results_df = result_df.copy()
        else:
            results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
        #writes the results to a csv file
        
        #resets the deconstructor
        deconstructor.reset()
        results_df.to_csv(out_folder + 'results.csv')
        group_counter += 1
    return results_df





def arg_parse_lns():
    '''parses the arguments for the lns function'''
    parser = argparse.ArgumentParser(description='Runs the fix and optimize function')
    #parser.add_argument('--base_xp_yaml', '-bxy', type=str, required=True, help='The base yaml file for the experiment')
    parser.add_argument('--setup_file', '-sf', type=str, required=True, 
                        help='The file that contains the setup information: model instance file, equipment instance file, start variables folder, base file name, run time, block size, and number of iterations')
    parser.add_argument('--xp_name', '-xp', type=str, required=True, help='The base file name for the results')
    parser.add_argument('--results_folder', '-rf', type=str, nargs='?', const='model_runs/', default='model_runs/', help='The folder to save the results in')
    parser.add_argument('--milp_run_time', '-rt', type=int, nargs='?', const=300, default=300, help='The maximum run time for the optimization problems, default is 300')
    parser.add_argument('--run_time', '-r', type=int, nargs='?', const=600, default=600, help='The maximum run time for the fix and optimize lns, default is 600')
    parser.add_argument('--update_method', '-u', type=str, nargs='?', const='increase_block_size', default='increase_block_size', help='The method for updating the deconstructor, default is increase_block_size')
    parser.add_argument('--block_size', '-bs', type=int, nargs='?', const=2, default=2, help='The size of the block to not be fixed, default is 2')
    parser.add_argument('--depth', '-dt', type=int, nargs='?', const=1, default=1, help='The depth of the subtree to not be fixed, default is 1. Only works for random_subtree_deconstructor')
    parser.add_argument('--n_iter', '-ni', type=int, nargs='?', const=3, default=3, help='The number of iterations to be run, default is 3')
    parser.add_argument('--n_iter_no_imp', '-nimp', type=int, nargs='?', const=2, default=2, help='The number of iterations to be run before we modify the desconstructor, default is 2')
    parser.add_argument('--deconstructor', '-d', type=str, required=True, help='The deconstructor to be used')
    args = parser.parse_args()
    return args

def get_deconstructor(deconstructor_method, update_method, block_size = 2, depth = 1, n_iter_no_imp=2):
    '''gets the deconstructor function'''
    decon = deconstructor(block_size, depth, no_improvement_limit=n_iter_no_imp)
    if deconstructor_method == 'random_model_deconstructor':
        print('using random_model_deconstructor')
        decon.set_deconstructor(decon.random_model_deconstructor)
    elif deconstructor_method == 'random_station_deconstructor':
        print('using random_station_deconstructor')
        decon.set_deconstructor(decon.random_station_deconstructor)
    elif deconstructor_method == 'random_subtree_deconstructor':
        print('using random_subtree_deconstructor')
        decon.set_deconstructor(decon.random_subtree_deconstructor)
    elif deconstructor_method == 'random_sequence_deconstructor':
        print('using random_sequence_deconstructor')
        decon.set_deconstructor(decon.random_sequence_deconstructor)
    else:
        raise ValueError("invalid deconstructor. The options are: \n    random_model_deconstructor, \n\
                                                                            random_station_deconstructor,\n\
                                                                             random_subtree_deconstructor,\n\
                                                                             random_sequence_deconstructor")
    if update_method == 'increase_block_size':
        print('using increase_block_size')
        decon.deconstructor_modifier = decon.increase_block_size
    elif update_method == 'decrease_depth_increase_block':
        print('using decrease_depth_increase_block')
        decon.deconstructor_modifier = decon.decrease_depth_increase_block
    elif update_method == 'fix_y':
        print('using fix_y')
        decon.deconstructor_modifier = decon.fix_y
    elif update_method == 'no_change':
        print('using no_change')
        decon.deconstructor_modifier = decon.no_change
    else:
        raise ValueError("invalid update method. The options are: \n    increase_block_size, \n\
                                                                            no_change")
    return decon

def main_lns():
    '''runs the lns function'''
    args = arg_parse_lns()
    setup_file = args.setup_file
    today = datetime.today().strftime('%y_%m_%d')
    base_file_name = args.results_folder +'xp_lns_'+  today+ '_' + args.xp_name + '/' + args.deconstructor + '/'
    if not os.path.exists(base_file_name):
        os.makedirs(base_file_name)
    #base_xp_yaml = args.base_xp_yaml
    run_time = args.milp_run_time
    total_run_time = args.run_time
    block_size = args.block_size
    n_iter = args.n_iter
    deconstructor = get_deconstructor(args.deconstructor, args.update_method, block_size, depth=args.depth,  n_iter_no_imp=args.n_iter_no_imp)
    #reads the setup file
    setup_dict_list = dict_list_from_csv(setup_file)
    #runs the fix and optimize function
    results_df = run_fix_and_optimize(setup_dict_list, base_file_name, deconstructor, run_time = run_time, total_run_time=total_run_time, n_iter = n_iter)
    return results_df


#main function
if __name__ == "__main__":
    
    results_df = main_lns()
    print(results_df.head())

    # param_dict_list = [{'start_var_folder': 'model_runs/xp_23_12_07_ws_15_16_17_seq5/model_dependent_linear/small_instance_hard0_variables/',
    #                 'model_file': 'SALBP_benchmark/MM_instances/model_data/small_instances/3_models/n=20_14_n=20_15_n=20_16.yaml',
    #                 'equipment_instance': 'SALBP_benchmark/MM_instances/equipment_data/random_O20_E4_S4_seed42.yaml',},
    #                 {'start_var_folder': 'model_runs/xp_23_12_07_ws_15_16_17_seq5/model_dependent_linear/small_instance_hard1_variables/',
    #                  'model_file': 'SALBP_benchmark/MM_instances/model_data/small_instances/3_models/n=20_15_n=20_16_n=20_17.yaml',
    #                  'equipment_instance': 'SALBP_benchmark/MM_instances/equipment_data/random_O20_E4_S4_seed42.yaml',},]
    # today = datetime.today().strftime('%y_%m_%d')
    # xp_name = f"xp_{today}_fo"
    # base_file_name = 'model_runs/'+xp_name
    # if not os.path.exists(base_file_name):
    #     os.makedirs(base_file_name)
    # run_fix_and_optimize(param_dict_list, 'SALBP_benchmark/MM_instances/small_instance_hard.yaml', base_file_name, random_model_deconstructor, run_time = 600,block_size=2 )