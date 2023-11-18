import networkx as nx
import random


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


def monte_carlo_tree(n_takts, entry_probabilities, enum_depth=0, n_samples = 100):
    '''Generates a scenario tree by sampling from the entry probabilities'''
    #enumerates the fist enum_depth takts
    _, final_sequences = make_scenario_tree(enum_depth, entry_probabilities)
    #randomly sample from the final sequences dictionary of dictionaries, based off of the probabilities
    sampled_sequences = {}
    for i in range(n_samples):
        seq = random.choices(list(final_sequences.values()),[x['probability'] for x in final_sequences.values()])[0]['sequence'].copy()
        print("seq", seq)
        print("i", i)
        for j in range(n_takts-enum_depth):
            print("j", j)
            seq.append(random.choices(list(entry_probabilities.keys()),[x for x in entry_probabilities.values()])[0])
        sampled_sequences[i] = {'sequence':seq, 'probability': 1/n_samples}
  
    return None, sampled_sequences

