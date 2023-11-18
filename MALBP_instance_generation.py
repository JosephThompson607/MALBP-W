from ALB_instance_tools import *
import random

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

def make_instances(filepath,SALBP_instance_list,model_names,takt_time, sequence_length, max_workers, no_stations, worker_cost, recourse_cost,task_time_adjuster=change_task_times_linear, seed = None):
    for i in range(len(SALBP_instance_list)-len(model_names)+1):
        instances = SALBP_instance_list[i:i+len(model_names)]
        model_mixture = random_model_mixture(model_names, seed)
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MultiModelTaskTimesInstance(model_dicts=model_dicts, takt_time=takt_time, sequence_length=sequence_length, max_workers=max_workers, no_stations=no_stations, worker_cost=worker_cost, recourse_cost=recourse_cost)
        mm_instance.genererate_task_times(change_func=task_time_adjuster)

        mm_instance.model_data_to_yaml(filepath)