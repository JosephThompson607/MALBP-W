import numpy as np
import re
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import glob
import copy
import yaml
import warnings

class MultiModelInstance:
    def __init__(self,  init_type='from_model_dicts',  instance_config_yaml = None, model_yaml = None, model_dicts=None, 
                 name = None, takt_time=None, max_workers = None, no_stations=None, 
                 sequence_length=None, worker_cost=None,recourse_cost = None, no_equipment=None, instance_type = 'alb'):
        self.instances = {}
        self.takt_time = takt_time
        self.no_models = None
        self.max_workers = max_workers
        self.no_stations = no_stations
        self.model_dicts = model_dicts
        self.model_mixtures = {}
        self.worker_cost = worker_cost
        self.recourse_cost = recourse_cost
        self.no_equipment = no_equipment
        self.instance_config_yaml = instance_config_yaml

        if init_type == 'from_model_dicts':
            self.data = create_instance_pair_stochastic(model_dicts)
            self.no_models = len(model_dicts)
            if not name:
                self.name = self.generate_name()
        elif init_type == 'instance_from_yaml':
            #Reads config data from a yaml then model data from the model file 
            self.instance_config_from_yaml(instance_config_yaml)
            self.model_data_from_yaml(model_yaml)
        elif init_type == 'model_data_from_yaml':
            #Only reads data on models from a yaml file
            self.model_data_from_yaml(model_yaml)
        else:
            raise ValueError('init_type must be either from_model_dicts or from yaml files')
        #Make sure that config data is set correctly
        self.config_check()
        self.scenario_tree = None
        self.sequence_length = sequence_length
        self.all_tasks = get_task_union(self.data, *list(self.data.keys()) )
        self.no_tasks = len(self.all_tasks)
        
    def instance_config_from_yaml(self, malbpw_yaml_file):
         '''Gets all instance data from a yaml file'''
         with open(malbpw_yaml_file) as file:
             malbpw_yaml = yaml.load(file, Loader=yaml.FullLoader)
             self.max_workers = malbpw_yaml['max_workers']
             self.no_stations = malbpw_yaml['no_stations']
             self.worker_cost = malbpw_yaml['worker_cost']
             self.recourse_cost = malbpw_yaml['recourse_cost']
             self.equipment_file = malbpw_yaml['equipment_file']
             self.sequence_length = malbpw_yaml['sequence_length']
             

    def config_check(self):
        #Check that the total probability of entering is equal to 1
        total_probability = 0
        for model in self.data:
            total_probability += self.data[model]['probability']
        if total_probability != 1:
            warnings.warn('Model probabilities do not sum to 1')
        #Check that the total task time is less than the takt time * number of stations * max workers
        for model in self.data:
            total_task_time = 0
            for task in self.data[model]['task_times']:
                total_task_time += self.data[model]['task_times'][task]
            if total_task_time > self.takt_time * self.no_stations * self.max_workers:
                warnings.warn(f"Total task time for model {model} exceeds the takt time * number of stations * max workers,\n \
                               consider increasing the number of stations")

    def model_data_from_yaml(self, model_yaml_file):
        '''This function reads a model_yaml_file and uses it to fill in the model 
        data for the MultiModelInstance object'''
        with open(model_yaml_file) as file:
            mm_yaml = yaml.load(file, Loader=yaml.FullLoader)
            self.takt_time = mm_yaml['takt_time']
            self.data = mm_yaml['model_data']
            self.no_models = len(mm_yaml['model_data'])
            self.name = mm_yaml['name']
            self.model_mixtures = {}
            for model in self.data:
                self.model_mixtures[model] = self.data[model]['probability']
                

    def model_data_to_yaml(self, fp=None):
        '''This function writes a MultiModelInstance object to a yaml file'''
        mmi_dict = {'name':self.name, 'model_data':self.data, 'takt_time':self.takt_time}
        my_yaml = open(fp + f'{self.name}.yaml', 'w')
        yaml.dump(mmi_dict, my_yaml)

    def generate_name(self):
        '''generates name from the filename of the .albp problem. The name is from the two numbers before the .albp extension'''
        name = ''
        for model in self.model_dicts:

            name += model['fp'].split('/')[-1].split('.')[0].split('_')[-2]+ '_'+ model['fp'].split('/')[-1].split('.')[0].split('_')[-1] + '_'
        #remove last character of name string
        name = name[:-1]
        return name

        
    # def create_instance_pair_stochastic(self, model_dicts):
    #     '''read .alb files, create a dictionary for each model, and include model name and probability
    #      input: list of dictionaries with keys 'name' 'location' and probability '''
    #     parsed_instances = {}
    #     print('model dicts', model_dicts)
    #     for instance_name, instance  in model_dicts.items():
    #         print('instance', instance)
    #         print('instance_name', instance_name)
    #         parsed_instances[instance_name] = {'instance':SALBP_Instance(instance_name, instance['fp'], takt_time=self.takt_time)}
    #         parsed_instances[instance_name]['probability'] = instance['probability']
    #     return parsed_instances
    
    def generate_scenario_tree(self,scenario_tree_generator, **kwargs):
        '''Generates a scenario tree for the instance, based on the model mixtures'''
        self.scenario_tree = scenario_tree_generator(self.n_takts, self.model_mixtures, **kwargs)

        

class SALBP_Instance:
    def __init__(self, instance_name, instance_location, takt_time=None, instance_type = 'alb'):
        self.name = instance_name
        self.location = instance_location
        self.task_times = {}
        self.precedence_relations = []
        self.num_tasks = 0
        self.cycle_time = takt_time
        self.order_strength = 0
        self.model_no = 0
        if instance_type == 'alb':
            self.parse_alb()

    def parse_alb(self):
        """Reads assembly line balancing instance .alb file, returns dictionary with the information"""
        alb_file = open(self.location).read()
        # Get number of tasks
        num_tasks = re.search("<number of tasks>\n(\d*)", alb_file)
        self.num_tasks = int(num_tasks.group(1))

        # Get cycle time if not already set
        if self.cycle_time == None:
            cycle_time = re.search("<cycle time>\n(\d*)", alb_file)
            self.cycle_time = int(cycle_time.group(1))

        # Order Strength
        order_strength = re.search("<order strength>\n(\d*,\d*)", alb_file)
        if order_strength:
            self.order_strength = float(order_strength.group(1).replace(",", "."))
        else:
            order_strength = re.search("<order strength>\n(\d*.\d*)", alb_file)
            self.order_strength = float(order_strength.group(1))

        # Task_times
        task_times = re.search("<task times>(.|\n)+?<", alb_file)

        # Get lines in this regex ignoring the first and last 2
        task_times = task_times.group(0).split("\n")[1:-2]
        task_times = {task.split()[0]: int(task.split()[1]) for task in task_times}
        self.task_times = task_times

        # Precedence relations
        precedence_relations = re.search("<precedence relations>(.|\n)+?<", alb_file)
        precedence_relations = precedence_relations.group(0).split("\n")[1:-2]
        precedence_relations = [task.split(",") for task in precedence_relations]
        self.precedence_relations = precedence_relations



## Scenario Tree Generation
def sum_prob(sequences):
    '''function for sanity checking that the probabilities sum to 1'''
    total = 0
    for seq in sequences:
        total += sequences[seq]['probability']
    return total

def make_consecutive_luxury_models_restricted_scenario_tree(n_takts, entry_probabilities, max_consecutive=3, luxury_models=['B']):
    """
    Creates a scenario tree for the given number of takts, instance and entry probabilities. 
    This scenario tree restricts the number of consective times a set of  model, the "luxury" models, can enter the line
    """
    # Create a directed graph
    G = nx.DiGraph()
    # Add the root node
    G.add_node('R', stage=0, scenario=0)
    #Create a list of final sequences
    final_sequences = {}
    def add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability=1, parent=0, sequence = [], current_stage=0, counter=[0], consecutive=0):
        if current_stage == n_takts:
            final_sequences[counter[0]] = {'sequence':sequence, 'probability': probability}
            counter[0] += 1
            return
        else:
            #Handle case where the model is the first one or is not the luxury model
            if len(sequence) == 0 or sequence[-1] not in  luxury_models:
                for model, prob in entry_probabilities.items():
                    new_sequence = sequence.copy()
                    new_sequence.append(model)
                    node_name = str(parent)+ str(model) 
                    graph.add_node(node_name, stage = current_stage, scenario = new_sequence)
                    graph.add_edge(parent, node_name, probability = probability)
                    add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability* prob,node_name, new_sequence, current_stage+1, consecutive=1)
            else:
                #Create a dictionary of models that excludes the previous model
                entry_probabilities_excluding_previous = entry_probabilities.copy()
                entry_probabilities_excluding_previous.pop(sequence[-1], None)
                #Handle case where the model is the same as the previous one
                if consecutive < max_consecutive:
                    new_sequence = sequence.copy()
                    model = new_sequence[-1]
                    prob = entry_probabilities[model]
                    new_sequence.append(model)
                    node_name = str(parent)+ str(model) 
                    graph.add_node(node_name, stage = current_stage, scenario = new_sequence)
                    graph.add_edge(parent, node_name, probability = probability)
                    add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability* prob,node_name, new_sequence, current_stage+1, consecutive=consecutive+1)
                else:
                    #If there are too many of this model, ignore it Change the entry probabilities of the other models proportional to their probability
                    total_prob = sum(entry_probabilities_excluding_previous.values())
                    for model, prob in entry_probabilities_excluding_previous.items():
                        entry_probabilities_excluding_previous[model] = prob/total_prob
                for model, prob in entry_probabilities_excluding_previous.items():
                    new_sequence = sequence.copy()
                    new_sequence.append(model)
                    node_name = str(parent)+ str(model) 
                    graph.add_node(node_name, stage = current_stage, scenario = new_sequence)
                    graph.add_edge(parent, node_name, probability = probability)
                    add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability* prob,node_name, new_sequence, current_stage+1, consecutive=1)
    add_nodes(n_takts, entry_probabilities, G, final_sequences, parent='R')
    return G, final_sequences

def make_consecutive_model_restricted_scenario_tree(n_takts, entry_probabilities, max_consecutive=3):
    """
    Creates a scenario tree for the given number of takts, instance and entry probabilities. 
    This scenario tree restricts the number of consective times a model can enter the line
    """
    # Create a directed graph
    G = nx.DiGraph()
    # Add the root node
    G.add_node('R', stage=0, scenario=0)
    #Create a list of final sequences
    final_sequences = {}
    def add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability=1, parent=0, sequence = [], current_stage=0, counter=[0], consecutive=0):
        if current_stage == n_takts:
            final_sequences[counter[0]] = {'sequence':sequence, 'probability': probability}
            counter[0] += 1
            return
        else:
            #Handle case where the model is the first one
            if len(sequence) == 0:
                for model, prob in entry_probabilities.items():
                    new_sequence = sequence.copy()
                    new_sequence.append(model)
                    node_name = str(parent)+ str(model) 
                    graph.add_node(node_name, stage = current_stage, scenario = new_sequence)
                    graph.add_edge(parent, node_name, probability = probability)
                    add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability* prob,node_name, new_sequence, current_stage+1, consecutive=1)
            else:
                #Create a dictionary of models that excludes the previous model
                entry_probabilities_excluding_previous = entry_probabilities.copy()
                entry_probabilities_excluding_previous.pop(sequence[-1], None)
                #Handle case where the model is the same as the previous one
                if consecutive < max_consecutive:
                    new_sequence = sequence.copy()
                    model = new_sequence[-1]
                    prob = entry_probabilities[model]
                    new_sequence.append(model)
                    node_name = str(parent)+ str(model) 
                    graph.add_node(node_name, stage = current_stage, scenario = new_sequence)
                    graph.add_edge(parent, node_name, probability = probability)
                    add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability* prob,node_name, new_sequence, current_stage+1, consecutive=consecutive+1)
                else:
                    #If there are too many of this model, ignore it Change the entry probabilities of the other models proportional to their probability
                    total_prob = sum(entry_probabilities_excluding_previous.values())
                    for model, prob in entry_probabilities_excluding_previous.items():
                        entry_probabilities_excluding_previous[model] = prob/total_prob
                for model, prob in entry_probabilities_excluding_previous.items():
                    new_sequence = sequence.copy()
                    new_sequence.append(model)
                    node_name = str(parent)+ str(model) 
                    graph.add_node(node_name, stage = current_stage, scenario = new_sequence)
                    graph.add_edge(parent, node_name, probability = probability)
                    add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability* prob,node_name, new_sequence, current_stage+1, consecutive=1)
    add_nodes(n_takts, entry_probabilities, G, final_sequences, parent='R')
    return G, final_sequences


def check_scenarios(prod_sequence1,prod_sequence2,t):
    '''compares two production sequences up to time t and returns true if they are the same'''
    if prod_sequence1[:t+1] == prod_sequence2[:t+1]:
        return True
    else:
        return False

def make_scenario_tree(n_takts, entry_probabilities):
    """
    Creates a scenario tree for the given number of takts, instance and entry probabilities.
    """
    # Create a directed graph
    G = nx.DiGraph()
    # Add the root node
    G.add_node('R', stage=0, scenario=0)
    #Create a list of final sequences
    final_sequences = {}
    def add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability=1, parent=0, sequence = [], current_stage=0, counter=[0]):
        if current_stage == n_takts:
            final_sequences[counter[0]] = {'sequence':sequence, 'probability': probability}
            counter[0] += 1
            return
        else:
            for model, prob in entry_probabilities.items():
                new_sequence = sequence.copy()
                new_sequence.append(model)
                node_name = str(parent)+ str(model) 
                graph.add_node(node_name, stage = current_stage, scenario = new_sequence)
                graph.add_edge(parent, node_name, probability = probability)
                add_nodes(n_takts, entry_probabilities, graph, final_sequences, probability* prob,node_name, new_sequence, current_stage+1)
    add_nodes(n_takts, entry_probabilities, G, final_sequences, parent='R')
    return G, final_sequences
# READING OF INSTANCES, MODIFICATION
# def pair_instances(instance_list, MODEL_MIXTURES):
#    '''takes a list of .alb filenames and a set of model mixtures 
#    (dictionary containing model names and probabilities) and 
#    returns a list of dictionaries containing the filenames and probabilities of the instances'''
#    instance_groups = []
#    for i in range(len(instance_list)-len(MODEL_MIXTURES)):
#       instance_dict = {}
#       for j in range(i, i+ len(MODEL_MIXTURES)):
#          model_name = list(MODEL_MIXTURES.keys())[j-i]
#          instance_dict[model_name] = {'fp':instance_list[j], 'name':model_name, 'probability':MODEL_MIXTURES[model_name]}
         
#       instance_groups.append(instance_dict)
#    return instance_groups

def pair_instances(instance_list, MODEL_MIXTURES):
      '''returns a list of lists of multi-model instances, where each list of instances is a list of instances that will be run together'''
      instance_groups = []
      for i in range(len(instance_list)-len(MODEL_MIXTURES)+1):
         instance_group= []
         for j in range(i, i+ len(MODEL_MIXTURES)):
            model_name = list(MODEL_MIXTURES.keys())[j-i]
            instance_group.append({'fp':instance_list[j], 'name':model_name, 'probability':MODEL_MIXTURES[model_name]})
         instance_groups.append(instance_group)
      return instance_groups

def read_instance_folder(folder_loc):
   '''looks in folder_loc for all .alb files and returns a list of filepaths to the .alb files'''
   instance_list = []
   for file in glob.glob(f"{folder_loc}*.alb"):
      instance_list.append(file)
   instance_list.sort(key = lambda file: int(file.split("_")[-1].split(".")[0]))
   return instance_list

def parse_alb(alb_file_name):
    """Reads assembly line balancing instance .alb file, returns dictionary with the information"""
    parse_dict = {}
    alb_file = open(alb_file_name).read()
    # Get number of tasks
    num_tasks = re.search("<number of tasks>\n(\d*)", alb_file)
    parse_dict["num_tasks"] = int(num_tasks.group(1))

    # Get cycle time
    cycle_time = re.search("<cycle time>\n(\d*)", alb_file)
    parse_dict["cycle_time"] = int(cycle_time.group(1))

    # Order Strength
    order_strength = re.search("<order strength>\n(\d*,\d*)", alb_file)
    
    if order_strength:
        parse_dict["order_strength"] = float(order_strength.group(1).replace(",", "."))
    else:
        order_strength = re.search("<order strength>\n(\d*.\d*)", alb_file)
        parse_dict["order_strength"] = float(order_strength.group(1))

    # Task_times
    task_times = re.search("<task times>(.|\n)+?<", alb_file)

    # Get lines in this regex ignoring the first and last 2
    task_times = task_times.group(0).split("\n")[1:-2]
    task_times = {task.split()[0]: int(task.split()[1]) for task in task_times}
    parse_dict["task_times"] = task_times

    # Precedence relations
    precedence_relations = re.search("<precedence relations>(.|\n)+?<", alb_file)
    precedence_relations = precedence_relations.group(0).split("\n")[1:-2]
    precedence_relations = [task.split(",") for task in precedence_relations]
    parse_dict["precedence_relations"] = precedence_relations
    return parse_dict


#function that returns names of all files in a directory with a given extension
def get_instance_list(directory, keep_directory_location = True,  extension='.alb'):
    if keep_directory_location:
        return [ directory + '/' + f for f in os.listdir(directory) if f.endswith(extension)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(extension)]

def rand_pert_precedence(p_graph_orig, seed=None):
    # randomly change at least 1 edge in the precedence graph
    # Seed random number generators
    while True:
        p_graph = p_graph_orig.copy()
        random.seed(seed)
        rng = np.random.default_rng(seed=seed)
        # calculate number of edges to change
        num_edges = 1 + rng.poisson(lam=4)
        # nx.swap.directed_edge_swap( p_graph, nswap=num_edges, seed=seed)
        edges_to_remove = random.sample(list(p_graph.edges()), num_edges)
        edges_to_add = random.sample(list(nx.non_edges(p_graph)), num_edges)
        for index, edge in enumerate(edges_to_remove):
            p_graph.remove_edge(edge[0], edge[1])
            p_graph.add_edge(edges_to_add[index][0], edges_to_add[index][1])
        pos = nx.spring_layout(p_graph_orig, k=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # nx.draw(p_graph_orig, ax=ax1, pos= pos, with_labels=True)
        # nx.draw(p_graph, ax=ax2, pos=pos, with_labels=True)
        simple_cycles = list(nx.simple_cycles(p_graph))
        if not simple_cycles:
            return list(p_graph.edges())


# this function actually removes tasks from the precedence graph and task times
def eliminate_tasks(old_instance, elim_interval=(0.6, 0.8), seed=None):
    instance = old_instance.copy()
    rng = np.random.default_rng(seed=seed)
    interval_choice = rng.uniform(low=elim_interval[0], high=elim_interval[1])
    # Instance must have same number of tasks and task numbering
    to_remove = rng.choice(
        list(instance[0]["task_times"].keys()),
        size=(int(instance[0]["num_tasks"] * (interval_choice))),
        replace=False,
    )
    # nx.draw_planar(p_graph, with_labels=True)
    for index, model in enumerate(instance):
        # Remove node from task times list
        entries_to_remove(
            to_remove, instance[index]["task_times"]
        )  
        # change precedence graph
        instance[index]["precedence_relations"] = reconstruct_precedence_constraints(
            instance[index]["precedence_relations"], to_remove
        )  
        #update number of tasks
        instance[index]["num_tasks"] = len(
            instance[index]["task_times"]
        )  
        
    return instance


# This function removes tasks by setting their task times to 0
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
    """Removes given tasks from precedence constraint, relinks preceding and succeeding tasks in the precedence  contraints"""
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


def change_task_times(instance, perc_reduct_interval=(0.40, 0.60), seed=None):
    # this function creates new task times based on the original task times takes original task times and how much they need to be reduced
    new_task_times = instance["task_times"]
    print("old task times", new_task_times)
    rng = np.random.default_rng(seed=seed)
    for key in new_task_times:
        new_task_times[key] = int(
            new_task_times[key]
            * rng.uniform(low=perc_reduct_interval[0], high=perc_reduct_interval[1])
        )
    print(new_task_times)
    return new_task_times

def get_task_intersection(test_instance, model_1, model_2):
    '''Returns the intersection of tasks between two models'''
    return  set(test_instance[model_1]['task_times']).intersection(set(test_instance[model_2]['task_times']))

def get_task_union(test_instance, *args):
    '''Returns the union of tasks between all models, input is a series of models to check'''
    for index, model in enumerate(args):
        if index == 0:
            task_union = set(test_instance[model]['task_times'])
        else:
            task_union = task_union.union(set(test_instance[model]['task_times']))
    return  task_union
    
def construct_precedence_matrix(instance):
    '''constructs a precedence matrix representation of a model's precedence relations'''
    precedence_matrix = np.zeros((len(instance['task_times'].keys()), len(instance['task_times'].keys())))
    for precedence in instance['precedence_relations']:
        precedence_matrix[int(precedence[0]) - 1][int(precedence[1]) - 1] = 1
    return precedence_matrix

def create_instance_pairs(instance_names, size_pair=2):
    instance_pairs = []
    instance_groups = [
        instance_names[i : i + size_pair] for i in range(0, len(instance_names), 2)
    ]
    for instances in instance_groups:
        parsed_instances = []
        for index, instance in enumerate(instances):
            parsed_instance = parse_alb(instance)
            parsed_instance["model_no"] = index
            parsed_instances.append(parsed_instance)
        instance_pairs.append(parsed_instances)
    return instance_pairs

def create_instance_pair_stochastic(instance_dicts):
    '''read .alb files, create a dictionary for each model, and include model name and probability
     input: list of dictionaries with keys 'name' 'location' and probability '''
    parsed_instances = {}
    for instance in instance_dicts:
        parsed_instances[instance['name']] = {}
        parsed_instance = parse_alb(instance['fp'])
        #Cycle time will be set to the same for all models
        del parsed_instance['cycle_time']
        parsed_instances[instance['name']].update(parsed_instance)
        parsed_instances[instance['name']]['probability'] = instance['probability']
    return parsed_instances


def list_all_tasks(instance):
    """Generates the set O of all tasks from a list of models"""
    tasks = []
    for index, model in enumerate(instance):
        tasks += model["task_times"].keys()
    return list(set(tasks))


def linear_reduction(old_task_times, number_of_workers):
    """Divides time of task by number of workers.
    INPUT: task times dictionary, number_of_workers int
    OUTPUT: new task_times dictonary with reduced times
    """
    if number_of_workers == 0:
        return old_task_times
    task_times = old_task_times.copy()
    for key, values in task_times.items():
        task_times[key] = values / number_of_workers
    return task_times


# EQUIPMENT SECTION
def generate_equipment(
    number_of_pieces, number_of_stations, no_tasks, cost_range=[100, 300], seed=None
):
    """Generates equipment cost and r_oe matrices. Returns equipment cost and roe matrix"""
    rng = np.random.default_rng(seed=seed)
    equipment_cost_matrix = generate_equipment_cost(
        number_of_pieces, number_of_stations, rng, cost_range
    )
    r_oe_matrix = generate_r_oe(
        equipment_cost_matrix, no_tasks, number_of_pieces, rng
    )
    return equipment_cost_matrix, r_oe_matrix


def generate_r_oe(
    equipment_cost_matrix, no_tasks, number_of_pieces, rng
):
    mean_cost = equipment_cost_matrix.mean()
    equip_avg = equipment_cost_matrix.mean(0)
    equip_val = equip_avg / mean_cost
    # Equipment works for task with probability based on its relative cost
    # TODO make sure that this calculation is correct, equipment that are
    # when equipment is above average cost, it works for all stations
    # In case paper was wrong, this compares the sum of the equipment cost,
    # not the average, has problem that it sometimes generates infeasible
    # r_oes
    # equip_val = equipment_cost_matrix.sum(0)/equipment_cost_matrix.sum()
    random_matrix = rng.random((no_tasks, number_of_pieces))
    r_oe = random_matrix < equip_val
    return r_oe


def generate_equipment_cost(
    number_of_pieces, number_of_stations, rng, cost_range=[200, 300]
):
    equipment_cost_matrix = rng.integers(
        low=cost_range[0],
        high=cost_range[1],
        size=(number_of_stations, number_of_pieces),
    )
    return equipment_cost_matrix


def generate_equipment_2(no_equipment, no_stations,no_tasks,mean=100, variance=15, seed = None):
    rng = np.random.default_rng(seed=seed)
    r_oe =rng.integers(low=0, high=2, size=( no_tasks, no_equipment))
    while np.sum(r_oe,axis=1).min() == 0:
        warnings.warn('infeasible equipment, trying random seed again')
        rng = np.random.default_rng()
        #makes sure we have feasible equipment
        r_oe =rng.integers(low=0, high=2, size=( no_tasks, no_equipment))
        print(np.sum(r_oe,axis=1))
    c_se = np.zeros((no_stations,no_equipment))
    for equipment in range(no_equipment):
        for station in range(no_stations):
            c_se[ station, equipment] =int((mean+np.random.randn()*variance ))* np.sum(r_oe,axis=0)[equipment]    
    return c_se,r_oe

class Equipment():
    def __init__(self,no_tasks = None, no_stations = None, no_equipment = None, generation_method = None, equipment_file = None, seed= 42, **kwargs):
        self.no_tasks = no_tasks
        self.no_stations = no_stations
        self.no_equipment = no_equipment
        if generation_method == 'import_yaml':
            self.from_yaml(equipment_file)
        elif generation_method == 'cost_based':
            print('doing cost based')
            self.name = f'cost_based_O{no_tasks}_E{no_equipment}_S{no_stations}_seed{seed}'
            self.c_se, self.r_oe = generate_equipment(no_equipment, no_stations, no_tasks, seed=seed, **kwargs)
        else:
            print('doing new method')
            self.name = f'random_O{no_tasks}_E{no_equipment}_S{no_stations}_seed{seed}'
            self.c_se, self.r_oe = generate_equipment_2(no_equipment, no_stations, no_tasks, seed=seed, **kwargs)

    def from_yaml(self, file_name):
        '''reads equipment from a yaml file'''
        with open(file_name) as file:
            equipment_yaml = yaml.load(file, Loader=yaml.FullLoader)
            self.no_equipment = equipment_yaml['no_equipment']
            self.no_stations = equipment_yaml['no_stations']
            self.no_tasks = equipment_yaml['no_tasks']
            self.c_se = np.array(equipment_yaml['c_se'])
            self.r_oe =np.array( equipment_yaml['r_oe'])

            self.name = equipment_yaml['name']
    
    def to_yaml(self, fp):
        '''writes equipment to a yaml file'''
        equipment_dict = { 'name':self.name, 'no_equipment': self.no_equipment, 'no_stations': self.no_stations, 'no_tasks':self.no_tasks, 'c_se':self.c_se.tolist(), 'r_oe':self.r_oe.tolist(),}
        my_yaml = open(fp + self.name + '.yaml', 'w')
        yaml.dump(equipment_dict, my_yaml)

def save_MALBPW_config(config_name, no_workers, no_stations, worker_cost, recourse_cost, sequence_length, file_name, equipment_files = None, instance_files = None):
   '''Saves config data to a yaml file'''
   config_dict = {'config_name': config_name,
                  'no_workers': no_workers, 
                  'no_stations': no_stations, 
                  'worker_cost': worker_cost, 
                  'recourse_cost': recourse_cost,  
                  'sequence_length': sequence_length,
                  'equipment_files': equipment_files, 
                  'model_files': instance_files}
   with open(file_name, 'w') as file:
      documents = yaml.dump(config_dict, file)
   return documents
