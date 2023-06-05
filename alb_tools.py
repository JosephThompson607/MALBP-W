import numpy as np
import re
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import glob
import copy
#READING OF INSTANCES, MODIFICATION
def parse_alb(alb_file_name):
    '''Reads assembly line balancing instance .alb file, returns dictionary with the information'''
    parse_dict = {}
    alb_file = open(alb_file_name).read()
    #Get number of tasks
    num_tasks = re.search('<number of tasks>\n(\d*)', alb_file)
    parse_dict['num_tasks'] = int(num_tasks.group(1))
    
    #Get cycle time
    cycle_time = re.search('<cycle time>\n(\d*)', alb_file)
    parse_dict['cycle_time'] = int(cycle_time.group(1))
    
    #Order Strength
    order_strength= re.search('<order strength>\n(\d*,\d*)', alb_file)
    parse_dict['order_strength'] = float(order_strength.group(1).replace(',','.'))
    
    #Task_times 
    task_times = re.search('<task times>(.|\n)+?<', alb_file)
    
    #Get lines in this regex ignoring the first and last 2
    task_times = task_times.group(0).split('\n')[1:-2]
    task_times = {task.split()[0]:int(task.split()[1]) for task in task_times}
    parse_dict['task_times'] = task_times
   
    #Precedence relations
    precedence_relations = re.search('<precedence relations>(.|\n)+?<', alb_file)
    precedence_relations = precedence_relations.group(0).split('\n')[1:-2]
    precedence_relations = [task.split(',') for task in precedence_relations]
    parse_dict['precedence_relations'] = precedence_relations
    return parse_dict

def rand_pert_precedence(p_graph_orig, seed= None):
    #randomly change at least 1 edge in the precedence graph
    #Seed random number generators
    while(True):
        p_graph = p_graph_orig.copy()
        random.seed(seed)
        rng = np.random.default_rng(seed = seed)
        #calculate number of edges to change
        num_edges = 1 +  rng.poisson(lam=4)
       # nx.swap.directed_edge_swap( p_graph, nswap=num_edges, seed=seed)
        edges_to_remove = random.sample(list(p_graph.edges()), num_edges)
        edges_to_add = random.sample(list(nx.non_edges(p_graph)), num_edges)
        for index, edge in enumerate(edges_to_remove):
            p_graph.remove_edge(edge[0], edge[1])
            p_graph.add_edge(edges_to_add[index][0], edges_to_add[index][1])
        pos = nx.spring_layout(p_graph_orig, k=1)
        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
        #nx.draw(p_graph_orig, ax=ax1, pos= pos, with_labels=True)
        #nx.draw(p_graph, ax=ax2, pos=pos, with_labels=True)
        simple_cycles =  list(nx.simple_cycles(p_graph))
        if not simple_cycles:
            return(list(p_graph.edges()))

#this function actually removes tasks from the precedence graph and task times
def eliminate_tasks(old_instance, elim_interval=(.6, .8), seed=None):
    instance = old_instance.copy()
    rng = np.random.default_rng(seed = seed)
    interval_choice = rng.uniform(low=elim_interval[0], high=elim_interval[1])
    #Instance must have same number of tasks and task numbering
    to_remove = rng.choice(list(instance[0]['task_times'].keys()),size=(int(instance[0]['num_tasks']*(interval_choice))), replace=False)
    #nx.draw_planar(p_graph, with_labels=True)
    for index, model in enumerate(instance):
        entries_to_remove(to_remove, instance[index]['task_times']) #Remove node from task times list
        instance[index]['precedence_relations'] = reconstruct_precedence_constraints(instance[index]['precedence_relations'], to_remove) #change precedence graph
        instance[index]['num_tasks'] = len(instance[index]['task_times']) #Correct number of tasks
    return(instance)

#This function removes tasks by setting their task times to 0
# def eliminate_tasks2(old_instance, elim_interval=(.6, .8), seed=None):
#     instance = old_instance.copy()
#     rng = np.random.default_rng(seed = seed)
#     interval_choice = rng.uniform(low=elim_interval[0], high=elim_interval[1])
#     #nx.draw_planar(p_graph, with_labels=True)
#     for index, model in enumerate(instance):
#         #Instances have different number of tasks and task numbering (task times with zero are ignored)
#         to_remove = rng.choice(list(instance[0]['task_times'].keys()),size=(int(instance[0]['num_tasks']*(interval_choice))), replace=False)
#         for key, value in  instance[index]['task_times'].items() #Remove node from task times list
#             if key in to_remove:
#                 instance[index]['task_times'][key] = 0
#     return(instance)

def reconstruct_precedence_constraints(precedence_relations, to_remove):
    '''Removes given tasks from precedence constraint, relinks preceding and succeeding tasks in the precedence  contraints'''
    p_graph = nx.DiGraph()
    p_graph.add_edges_from(precedence_relations)
    for node in to_remove:
        if node in p_graph.nodes():
            for parent in p_graph.predecessors(node):
                for child in p_graph.successors(node):
                    p_graph.add_edge(parent, child)
            p_graph.remove_node(node)
    return list(p_graph.edges())


def entries_to_remove(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]

def change_task_times(instance, perc_reduct_interval = (.40,.60), seed=None):
    #this function creates new task times based on the original task times takes original task times and how much they need to be reduced
    new_task_times = instance['task_times']
    print('old task times', new_task_times)
    rng = np.random.default_rng(seed = seed)
    for key in new_task_times:
        new_task_times[key] = int(new_task_times[key] * rng.uniform(low=perc_reduct_interval[0], high=perc_reduct_interval[1]))
    print(new_task_times)
    return(new_task_times)

def create_instance_pairs(instance_names, size_pair = 2):
    instance_pairs = []
    instance_groups = [instance_names[i:i+size_pair] for i in range(0,len(instance_names),2)]
    for instances in instance_groups:
        parsed_instances = []
        for index, instance in enumerate(instances):
            parsed_instance= parse_alb(instance)
            parsed_instance['model_no'] = index
            parsed_instances.append(parsed_instance)
        instance_pairs.append(parsed_instances)
    return instance_pairs

def list_all_tasks(instance):
    '''Generates the set O of all tasks from a list of models'''
    tasks = []
    for index, model in enumerate(instance):
        tasks += model['task_times'].keys()
    return list(set(tasks))


#EQUIPMENT SECTION
def generate_equipment(number_of_pieces, number_of_stations, all_tasks, cost_range=[100,300], seed=None):
    '''Generates equipment cost and r_oe matrices. Returns equipment cost and roe matrix'''
    rng = np.random.default_rng(seed=seed)
    equipment_cost_matrix = generate_equipment_cost(number_of_pieces, number_of_stations,rng,  cost_range)
    r_oe_matrix = generate_r_oe(equipment_cost_matrix, all_tasks, number_of_pieces, number_of_stations,rng)
    return equipment_cost_matrix, r_oe_matrix

def generate_r_oe(equipment_cost_matrix,all_tasks, number_of_pieces, number_of_stations, rng):
    mean_cost = equipment_cost_matrix.mean()
    equip_avg = equipment_cost_matrix.mean(0)
    equip_val = equip_avg/mean_cost
    #Equipment works for task with probability based on its relative cost
    #TODO make sure that this calculation is correct, equipment that are
    #when equipment is above average cost, it works for all stations
    #In case paper was wrong, this compares the sum of the equipment cost, 
    #not the average, has problem that it sometimes generates infeasible
    #r_oes
    #equip_val = equipment_cost_matrix.sum(0)/equipment_cost_matrix.sum()
    random_matrix = rng.random((len(all_tasks),number_of_pieces))
    r_oe = random_matrix < equip_val
    return r_oe

def generate_equipment_cost(number_of_pieces, number_of_stations,rng,  cost_range=[200,300]):
    equipment_cost_matrix = rng.integers(low= cost_range[0], high=cost_range[1], size=(number_of_stations,number_of_pieces))
    return equipment_cost_matrix



#MDP section
#Generate all possible snapshots
def enumerate_universe(omega, omega_list, number_of_models, number_of_stations):
    '''Recursively generates omega. Modifies in place omega_list, returns nothing'''
    if len(omega) >= number_of_stations:
        omega_list.append(list(omega))
        return 
    for i in range(number_of_models):
        enumerate_universe(omega+str(i), omega_list, number_of_models, number_of_stations)


def task_to_stations(task , station, old_partition):
    partition = copy.deepcopy(old_partition)
    partition[station].append(task)
    return partition

#Creates all possible partitions of tasks to stations
def task_partitioning( partitions, old_partition,  old_tasks, no_stations, instance):
    if not old_tasks:
        #Below are some checks that reduce the universe of possible partitions
        #checks if all lists in partition are non-empty
        if all(partition for partition in old_partition):
            #checks if partition respects precedence constraints
            if check_precedence_constraints( old_partition, instance['precedence_relations'], no_stations):
                    if check_takt_time_constraints(old_partition, instance['task_times'], instance['cycle_time'], instance['max_workers']):
                        partitions.append(old_partition)
        return
    tasks = old_tasks.copy()
    task_to_assign = tasks.pop(0)
    for station in range(no_stations):
        new_partition = task_to_stations(task_to_assign, station, old_partition)
        task_partitioning(partitions,new_partition, tasks, no_stations,instance)

#Function that makes sure that the takt time constraints are respected. This function does this by summing the total task time of tasks at each station
#and divides it by the maximum number of workers at that station. If the result is greater than the takt time, then the partition is not feasible
def check_takt_time_constraints(partition, task_times, cycle_time, max_workers):
    for index, station in enumerate(partition):
        if sum([task_times[task] for task in partition[index]]) / max_workers > cycle_time:
            return False
    return True


def check_precedence_constraints(partition, precedence_constraints,  NO_S):
    '''Checks to see if a model's precedence constraints are violated by a given partition'''
    for (task1, task2) in precedence_constraints:
        #if index of list containing task1 is greater than index of list containing task2 return False
        for station in range(NO_S):
            if task1 in partition[station]:
                index_task1 = station
            if task2 in partition[station]:
                index_task2 = station
        if index_task1 > index_task2:
            return False
    return True

# Function that generates all possible actions
def generate_actions(old_action, actions_list, all_tasks, number_of_stations, number_of_models):
    action = old_action.copy()
    '''Generates all possible actions for a given instance. Returns a list of actions'''
    print('remaining tasks', all_tasks)
    if not all_tasks:
        actions_list.append(list(action))
        print('action list', actions_list)
        return
    for station in range(number_of_stations):
        for task in all_tasks:
            #remove task from copy of all_tasks
            all_tasks_copy = all_tasks.copy()
            all_tasks_copy.remove(task)
            print('action before', action)
            action[station] += [ task]
            print('action after', action)
            generate_actions(action, actions_list, all_tasks_copy,  number_of_stations, number_of_models)


def linear_reduction(old_task_times, number_of_workers):
    '''Divides time of task by number of workers.
        INPUT: task times dictionary, number_of_workers int
        OUTPUT: new task_times dictonary with reduced times
    '''
    task_times=old_task_times.copy()
    for key, values in task_times.items():
        task_times[key] = values / number_of_workers
    return task_times