import numpy as np
import pandas as pd
import pulp as plp
from ALB_instance_tools import *
from report_functions import *
from milp_models import *
from scenario_trees import *
from timeit import default_timer as timer



### Deconstructor heurisitcs for the fix and opt LNS
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

    
### Running the fix and opt LNS
def fix_and_optimize_dl( deconstructor,md_results_folder, problem_instance, equipment_instance,prod_sequences, fp, n_iter = 3, run_time = 600, **kwargs):
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
        results_batch.append(result)
        #Resetting dynamic problem to original state
        dynamic_problem = dynamic_problem_linear_labor_recourse(problem_instance, equipment_instance,sequence_length, prod_sequences)
    result_df = pd.DataFrame(results_batch)
    return result_df   
