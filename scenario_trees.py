import sys
sys.path.insert(1, 'instance_generation/')
import os
import networkx as nx
import random
import pandas as pd
import yaml


def random_model_mixture(model_names, seed = None):
    '''Creates a dictionary where the model name are the keys and the probability is the values. The probabilities sum to 1'''
    model_mixture = {}
    random.seed(seed)
    for model_name in model_names:
        model_mixture[model_name] = random.random()
    total = sum(model_mixture.values())
    for model_name in model_mixture.keys():
        model_mixture[model_name] = model_mixture[model_name]/total
    return model_mixture


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


def monte_carlo_tree(n_takts, entry_probabilities, enum_depth=0, n_samples = 100, seed=None):
    '''Generates a scenario tree by sampling from the entry probabilities'''
    #set the seed
    print("seed at monte_carlo_tree", seed)
    #IF a seed is given, set it
    if seed is not None:
        print('setting seed')
        random.seed(seed)
    #else, set the seed to a number determined by the system
    else:
        random.seed()
    
    #enumerates the fist enum_depth takts
    _, final_sequences = make_scenario_tree(enum_depth, entry_probabilities)
    #randomly sample from the final sequences dictionary of dictionaries, based off of the probabilities
    sampled_sequences = {}
    for i in range(n_samples):
        seq = random.choices(list(final_sequences.values()),[x['probability'] for x in final_sequences.values()])[0]['sequence'].copy()
        for j in range(n_takts-enum_depth):
            seq.append(random.choices(list(entry_probabilities.keys()),[x for x in entry_probabilities.values()])[0])
        sampled_sequences[i] = {'sequence':seq, 'probability': 1/n_samples}
  
    return None, sampled_sequences

def monte_carlo_tree_limit(n_takts, entry_probabilities,  enum_depth=0, n_samples = 100, seed=None):
    '''Checks and see if the number of samples is greater than the number of possible sequences, if so, returns all possible sequences'''
    if n_samples > len(entry_probabilities)**n_takts:
        print("Too many samples, returning all possible sequences, solving for exact solution ",  flush=True)
        return make_scenario_tree(n_takts, entry_probabilities)
    else:
        print("Sampling ", n_samples, "with seed ", seed,   flush=True)
        return monte_carlo_tree(n_takts, entry_probabilities, enum_depth=enum_depth, n_samples=n_samples, seed=seed)


def get_scenario_generator(xp_yaml, seed = None):
    '''Reads the xp_yaml config file and seed and returns a scenario generator'''
    tree_kwargs = {}
    #old name for the scenario generator
    scenario_key = 'scenario_generator'
    if scenario_key not in xp_yaml:
        #new name for the scenario generator
        scenario_key = 'scenario'
    if isinstance(xp_yaml[scenario_key], dict):
        if xp_yaml[scenario_key]['generator']== 'monte_carlo_tree':
            scenario_generator = monte_carlo_tree
            tree_kwargs['n_samples'] = xp_yaml[scenario_key]['n_samples']
            tree_kwargs['enum_depth'] = xp_yaml[scenario_key]['enum_depth']
            tree_kwargs['seed'] = xp_yaml[scenario_key]['seed']
            if seed != None:
                tree_kwargs['seed'] = seed
        elif xp_yaml[scenario_key]['generator']== 'monte_carlo_tree_limit':
            scenario_generator = monte_carlo_tree_limit
            tree_kwargs['n_samples'] = xp_yaml[scenario_key]['n_samples']
            tree_kwargs['enum_depth'] = xp_yaml[scenario_key]['enum_depth']
            tree_kwargs['seed'] = xp_yaml[scenario_key]['seed']
            if seed != "None":
                tree_kwargs['seed'] = seed
            else:
                tree_kwargs['seed'] = None
        elif xp_yaml[scenario_key]['generator']== 'full':
            scenario_generator = make_scenario_tree     
        else:
            raise ValueError('scenario generator not recognized')
    else:
        scenario_generator = make_scenario_tree
    return tree_kwargs, scenario_generator

def generate_tree_csv(tree_generator, sequence_length, entry_probabilities, enum_depth=0, n_samples = 100, seed=None, filename='scenario_tree.csv'):
    '''Generates a scenario tree and writes it to a csv file'''
    tree, final_sequences = tree_generator(sequence_length, entry_probabilities, enum_depth=enum_depth, n_samples=n_samples, seed=seed)
    #convert the final sequences to a dataframe
    print("final sequences", final_sequences)
    frames = []
    for key in final_sequences.keys():
        row_frame = pd.Series(final_sequences[key]).to_frame().T
        frames.append(row_frame)

    
    df = pd.concat(frames, axis=0, ignore_index=True)
    df['entry_probabilities'] = entry_probabilities
    df['sequence_length'] = sequence_length
    df['num_samples'] = n_samples
    print("writing to file ", filename)
    df.to_csv(filename, index=False)

def save_tree_yaml( final_sequences, entry_probabilities, sequence_length, n_samples, seed, enum_depth, filename, tree_name):
    '''Saves a scenario tree to a yaml file'''
    scenario_dict = {'filename':filename, 'tree_name':tree_name, 'entry_probabilities':entry_probabilities, 'sequence_length':sequence_length, 'n_samples' : n_samples, 'seed':seed, 'enum_depth':enum_depth, 'final_sequences':final_sequences}
    my_yaml = open(filename + '.yaml', 'w')
    yaml.dump(scenario_dict, my_yaml)

def generate_tree_yaml(tree_generator, sequence_length, entry_probabilities, fp, enum_depth=0, n_samples = 100, seed=None,  tree_name='tree', filename='st'):
    '''Generates a scenario tree and writes it to a yaml file'''
    tree, final_sequences = tree_generator(sequence_length, entry_probabilities, enum_depth=enum_depth, n_samples=n_samples, seed=seed)
    #convert the final sequences to a dataframe
    print("final sequences", final_sequences)
    scenario_dict = {'filename':filename, 'tree_name':tree_name, 'entry_probabilities':entry_probabilities, 'sequence_length':sequence_length, 'n_samples' : n_samples, 'seed':seed, 'enum_depth':enum_depth, 'final_sequences':final_sequences}

    my_yaml = open(fp + f'{filename}.yaml', 'w')
    yaml.dump(scenario_dict, my_yaml)

def make_scenarios(n_s_trees, sequence_length, model_names, n_samples, fp, enum_depth=0, filename='s_tree_', no_model_mixture_repeats=1):
    #if the fp does not exist yet, make it
    if not os.path.exists(fp):
        os.makedirs(fp)
    for i in range(n_s_trees):
        model_mixtures = random_model_mixture(model_names=model_names)
        #if the fp does not exist yet, make it
        if not os.path.exists(fp + filename + str(i+1) + '/'):
            os.makedirs(fp + filename + str(i+1) + '/')
        for j in range(no_model_mixture_repeats):
            generate_tree_yaml(monte_carlo_tree_limit, sequence_length, model_mixtures, enum_depth=enum_depth , n_samples=n_samples, fp = fp, tree_name=i+1, filename=filename + str(i+1) + '/'+ 'mc_sample_' + str(j+1))

def read_tree_csv(filename):
    '''Reads a scenario tree from a csv file'''
    with open(filename) as f:
        lines = f.readlines()
    final_sequences = {}
    for line in lines:
        key, value = line.split(',')
        final_sequences[key] = value
    return final_sequences