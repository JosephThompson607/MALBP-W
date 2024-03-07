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

def make_reduced_instances(filepath, SALBP_instance_list, model_names, takt_time, sequence_length, max_workers, no_stations, worker_cost, recourse_cost, task_time_adjuster=change_task_times_linear, interval_choice = (0.7,0.8), seed = None):
    for i in range(len(SALBP_instance_list)-len(model_names)+1):
        instances = SALBP_instance_list[i:i+len(model_names)]
        model_mixture = random_model_mixture(model_names, seed)
        model_dicts = make_instance_pair(instances, model_mixture)
        print("These are the model_dicts")
        print(model_dicts)
        mm_instance = MultiModelTaskTimesInstance(model_dicts=model_dicts, takt_time=takt_time, sequence_length=sequence_length, max_workers=max_workers, no_stations=no_stations, worker_cost=worker_cost, recourse_cost=recourse_cost)
        mm_instance.genererate_task_times(change_func=task_time_adjuster)
        print("This is the data")
        new_instance = eliminate_tasks(mm_instance, interval_choice, seed=seed)
        new_instance.generate_name()
        new_instance.model_data_to_yaml(filepath)

def make_instances(filepath,SALBP_instance_list,model_names,takt_time, sequence_length, max_workers, no_stations, worker_cost, recourse_cost,task_time_adjuster=change_task_times_linear, seed = None):
    for i in range(len(SALBP_instance_list)-len(model_names)+1):
        instances = SALBP_instance_list[i:i+len(model_names)]
        model_mixture = random_model_mixture(model_names, seed)
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MultiModelTaskTimesInstance(model_dicts=model_dicts, takt_time=takt_time, sequence_length=sequence_length, max_workers=max_workers, no_stations=no_stations, worker_cost=worker_cost, recourse_cost=recourse_cost)
        mm_instance.genererate_task_times(change_func=task_time_adjuster)

        mm_instance.model_data_to_yaml(filepath)




if __name__ == "__main__":
    # instance_list = read_instance_folder("SALBP_benchmark/small data set_n=20/")[:100]
    # #NO_EQUIPMENT = 4
    # seed = 42
    # NO_WORKERS =1
    # NO_STATIONS = 2
    # WORKER_COST = 500
    # RECOURSE_COST = WORKER_COST * 2
    # TAKT_TIME = 500
    # SEQUENCE_LENGTH = 20
    # model_names = ['A', 'B','C','D','E']



    # fp = 'SALBP_benchmark/MM_instances/model_data/small_instances/5_models/'
    # #if filepath does not exist, create it
    # if not os.path.exists(fp):
    #     os.makedirs(fp)
    # make_instances(fp,instance_list, model_names, takt_time=TAKT_TIME, sequence_length=SEQUENCE_LENGTH, max_workers=NO_WORKERS, no_stations=NO_STATIONS, worker_cost=WORKER_COST, recourse_cost=RECOURSE_COST, seed=seed)

    model_files = get_model_files_list('../SALBP_benchmark/MM_instances/small_instance_longSeq.yaml')
    print('model_files list', model_files)
    equipment_files = ["SALBP_benchmark/MM_instances/equipment_data/random_O20_E8_S4_seed42.yaml"]
    scenario_trees = ['None']
    config_files = ['SALBP_benchmark/MM_instances/small_instance_longSeq.yaml']
    output_folder = '../SALBP_benchmark/MM_instances/csv_config'
    make_slurm_csv(model_files, equipment_files, scenario_trees, config_files, output_folder, name= 'no_tree', model_repeats=10)