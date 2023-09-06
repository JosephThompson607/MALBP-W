
import pulp as plp
from ALB_instance_tools import parse_alb
def define_ALBP_1_problem(instance, max_stations = 20):
    prob = plp.LpProblem("ALPB_1", plp.LpMinimize)
    #creating decision variables
    tasks = plp.LpVariable.dicts("task_o_s", (instance['task_times'].keys(), range(1,max_stations + 1)), cat='Binary')
    #objective function
    prob += plp.lpSum([ station * tasks[task][station] for station in range(1,max_stations + 1) for task in instance['task_times'].keys()])
    #definining constraints
    #constraint 1 only choose 1 station for each task
    for task in instance['task_times'].keys():
        prob += plp.lpSum([tasks[task][station] for station in range(1,max_stations + 1)]) == 1
    #constraint 2 task and station assignment must respect takt time
    for station in range(1,max_stations + 1):
        prob += plp.lpSum([instance['task_times'][task] * tasks[task][station] for task in instance['task_times'].keys()]) <= instance['cycle_time']
    #constraint 3 tasks must respect precedence constraints
    for precedence in instance['precedence_relations']:
        prob += plp.lpSum([station * tasks[precedence[0]][station] for station in range(1,max_stations + 1)]) <= plp.lpSum([station * tasks[precedence[1]][station] for station in range(1,max_stations + 1)])
    return prob

def solve_ALBP_1(instance, max_stations = 20):
    prob = define_ALBP_1_problem(instance, max_stations)
    prob.solve(solver=plp.XPRESS_PY( msg=False))
    max_station = -10
    task_assignment = {}
    for variable in prob.variables():
        if variable.varValue > 0:
            task = variable.name.split("_")[3]
            station = variable.name.split("_")[4]
            #Adds dictionary where key is the task and value is the station
            task_assignment[task] = station
            #Find the largest station number that is used
            if int(station) > max_station:
                max_station = int(station)
    return max_station, task_assignment

def get_ALBP_solutions(problems_list, ALBP_solver = solve_ALBP_1, max_stations = 20, **kwargs):
    solutions = []
    for problem in problems_list:
        instance = parse_alb(problem['location'])
        print('solving problem', problem['name'])
        no_stations, task_assignment = ALBP_solver(instance, max_stations = max_stations,**kwargs)
        #creates a new dictionary entry that contains the data on the instances
        
        entry = {'name':problem['name']}
        entry['no_tasks'] = len(instance['task_times'].keys())
        entry['order_strength'] = instance['order_strength']
        entry['cycle_time'] = instance['cycle_time']
        entry['no_stations'] = no_stations
        entry['task_assignment'] = task_assignment
        solutions.append(entry)
    return solutions

#Scoring functions
def task_time_weight(tasks_dict, instance):
    for task in tasks_dict.keys():
        tasks_dict[task]['score'] = instance['task_times'][task]

def backwards_recursive_positional_weight(tasks_dict, instance):
    #calculates the positional weight of each task of the instance
    p_graph = nx.DiGraph()
    p_graph.add_nodes_from([(key, {'task_time':value}) for key, value in instance["task_times"].items()])
    p_graph.add_edges_from(instance["precedence_relations"], color="r")
    weights = {}
    while len(p_graph.nodes) > 0:
        leaves = []
        for n in p_graph.nodes():
            if not list(p_graph.successors(n)):
                weights[n] = p_graph.nodes[n]['task_time']
                leaves.append(n)
                for p in p_graph.predecessors(n):
                    p_graph.nodes[p]['task_time'] += p_graph.nodes[n]['task_time']
        p_graph.remove_nodes_from(leaves)
    for weight in weights.keys():
        tasks_dict[weight]['score'] = weights[weight]


def collect_parents(p_graph):
    '''Collects the parents of each node in a graph and returns a dictionary with the node as key and the set of parents as value'''
    weight_set = {}
    while len(p_graph.nodes) > 0:
        leaves = []
        for n in p_graph.nodes():
            if not list(p_graph.successors(n)):
                weight_set[n] = p_graph.nodes[n]['task_set']
                leaves.append(n)
                for p in p_graph.predecessors(n):
                    p_graph.nodes[p]['task_set'] =  p_graph.nodes[p]['task_set'].union(p_graph.nodes[n]['task_set'])
        p_graph.remove_nodes_from(leaves)
    return weight_set

def reverse_positional_weight(tasks_dict, instance):
    #calculates the reverse positional weight of each task of the instance
    p_graph = nx.DiGraph()
    p_graph.add_nodes_from([(key, {'task_time':value, 'task_set':set([key])}) for key, value in instance["task_times"].items()])
    p_graph.add_edges_from(instance["precedence_relations"], color="r")
    p_graph = p_graph.reverse()
    weight_set = collect_parents(p_graph)
    for weight in weight_set.keys():
        tasks_dict[weight]['score'] = sum([instance['task_times'][task] for task in weight_set[weight]])

def positional_weight(tasks_dict, instance):
    #calculates the positional weight of each task of the instance
    p_graph = nx.DiGraph()
    p_graph.add_nodes_from([(key, {'task_time':value, 'task_set':set([key])}) for key, value in instance["task_times"].items()])
    p_graph.add_edges_from(instance["precedence_relations"], color="r")
    weight_set = collect_parents(p_graph)
    for weight in weight_set.keys():
        tasks_dict[weight]['score'] = sum([instance['task_times'][task] for task in weight_set[weight]])


def rank_and_assign_initialization(instance,score_function, max_stations = 20):
    station_capacities = [instance['cycle_time'] for i in range(0, max_stations)]
    tasks_dict = {}
    for task in instance['task_times'].keys():
        task_dict = {}
        task_dict['predecessors'] = [precedence[0] for precedence in instance['precedence_relations'] if precedence[1] == task]
        tasks_dict[task] = task_dict
    score_function(tasks_dict, instance)
    #sorts tasks_dict by score
    tasks_dict = {k: v for k, v in sorted(tasks_dict.items(), key=lambda item: item[1]['score'])}
     
    return tasks_dict, station_capacities

#fills up each station with available tasks in order of score
def insert_task(instance, tasks_dict, station_capacities, assignment_dict):
    for index, station in enumerate(station_capacities):
            for task in tasks_dict.keys():
                if instance['task_times'][task] <= station and all(predecessor not in tasks_dict.keys() for predecessor in tasks_dict[task]['predecessors']):
                    station_capacities[index] -= instance['task_times'][task]
                    assignment_dict[task] = index + 1
                    tasks_dict.pop(task)
                    return
    raise ValueError(f'no task can be assigned to any station (try adding more stations)')

# RA heuristic as described in "A comparative Evaluation of Heuristics for the Assembly Line Balancing Problem" by Ponnanbalam et. al              
def rank_and_assign( instance,score_function, max_stations = 20):
    task_assignment = {}
    tasks_dict, station_capacities = rank_and_assign_initialization(instance, score_function, max_stations)
    #Inserts tasks into stations until there are no more tasks
    while len(tasks_dict.keys()) > 0:
        insert_task(instance, tasks_dict, station_capacities, task_assignment)
    return  sum([1 for station in station_capacities if station < instance['cycle_time']]),task_assignment


#inserts tasks into first available station
def insert_task_iuff(instance, task, tasks_dict,assignment_dict, station_capacities):
    for index, station_cap in enumerate(station_capacities):
            if instance['task_times'][task] <= station_cap:
                if tasks_dict[task]['available_after'] <= index + 1:
                    station_capacities[index] -= instance['task_times'][task]
                    assignment_dict[task] = index + 1
                    tasks_dict.pop(task)
                    return


def update_tasks(tasks_dict, assignment_dict):
    available_tasks = []
    for task in tasks_dict.keys():
            #if all predecessors are assigned to a station, mark task as available
            if tasks_dict[task]['available_after'] == 'unavailable':
                if all([predecessor in  assignment_dict.keys() for predecessor in tasks_dict[task]['predecessors']]):
                    tasks_dict[task]['available_after'] = max([assignment_dict[predecessor] for predecessor in tasks_dict[task]['predecessors']])
                    return task
            else:
                return task


def iuff_initialization(instance, score_function, max_stations = 20, **kwargs):
    station_capacities = [instance['cycle_time'] for i in range(0, max_stations)]
    tasks_dict = {}
    available_tasks = []
    for task in instance['task_times'].keys():
        task_dict = {}
        task_dict['predecessors'] = [precedence[0] for precedence in instance['precedence_relations'] if precedence[1] == task]
        #adds tasks with no predecessors to available tasks
        if not task_dict['predecessors']:
            task_dict['available_after'] = 0
            available_tasks.append(task)
        else:
            task_dict['available_after'] = 'unavailable'
        tasks_dict[task] = task_dict
    #sorts tasks_dict by score
    score_function(tasks_dict, instance, **kwargs)
    tasks_dict = {k: v for k, v in sorted(tasks_dict.items(), key=lambda item: item[1]['score'], reverse=True)}
    return tasks_dict, available_tasks[0], station_capacities
    
     
# IUFF heuristic as described in "A comparative Evaluation of Heuristics for the Assembly Line Balancing Problem" by Ponnanbalam et. al              
def immediate_update_first_fit( instance,score_function = None, max_stations = 20, **kwargs):
    tasks_dict, available_task, station_capacities = iuff_initialization(instance, score_function, max_stations, **kwargs)
    assignment_dict = {}
    while tasks_dict:
        insert_task_iuff(instance, available_task, tasks_dict, assignment_dict, station_capacities)
        available_task = update_tasks(tasks_dict, assignment_dict)
    return  sum([1 for station in station_capacities if station < instance['cycle_time']]), assignment_dict