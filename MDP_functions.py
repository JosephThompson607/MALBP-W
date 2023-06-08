import copy

# MDP section
# Generate all possible snapshots
def enumerate_universe(omega, omega_list, number_of_models, number_of_stations):
    """Recursively generates omega. Modifies in place omega_list, returns nothing"""
    if len(omega) >= number_of_stations:
        omega_list.append(list(omega))
        return
    for i in range(number_of_models):
        enumerate_universe(
            omega + str(i), omega_list, number_of_models, number_of_stations
        )


def task_to_stations(task, station, old_partition):
    partition = copy.deepcopy(old_partition)
    partition[station].append(task)
    return partition


# Creates all possible partitions of tasks to stations for a given model
def task_partitioning(partitions, old_partition, old_tasks, no_stations, instance):
    if not old_tasks:
        # Below are some checks that reduce the universe of possible partitions
        # checks if all lists in partition are non-empty
        if all(partition for partition in old_partition):
            # checks if partition respects precedence constraints
            if check_precedence_constraints(
                old_partition, instance["precedence_relations"], no_stations
            ):
                if check_takt_time_constraints(
                    old_partition,
                    instance["task_times"],
                    instance["cycle_time"],
                    instance["max_workers"],
                ):
                    partitions.append(old_partition)
        return
    tasks = old_tasks.copy()
    task_to_assign = tasks.pop(0)
    for station in range(no_stations):
        new_partition = task_to_stations(task_to_assign, station, old_partition)
        task_partitioning(partitions, new_partition, tasks, no_stations, instance)


# Function that generates all possible actions for all models
def get_feasible_partitions(models, no_s, all_tasks, test_instance):
    feasible_partitions = {}
    for model in models:
        partitions = []
        task_partitioning(
            partitions,
            [[] for _ in range(no_s)],
            list(all_tasks),
            no_s,
            test_instance[model],
        )
        feasible_partitions[f"model_{model}"] = partitions
    return feasible_partitions


# Function that makes sure that the takt time constraints are respected. This function does this by summing the total task time of tasks at each station
# and divides it by the maximum number of workers at that station. If the result is greater than the takt time, then the partition is not feasible
def check_takt_time_constraints(partition, task_times, cycle_time, max_workers):
    for index, station in enumerate(partition):
        if (
            sum([task_times[task] for task in partition[index]]) / max_workers
            > cycle_time
        ):
            return False
    return True


def check_precedence_constraints(partition, precedence_constraints, NO_S):
    """Checks to see if a model's precedence constraints are violated by a given partition"""
    for (task1, task2) in precedence_constraints:
        # if index of list containing task1 is greater than index of list containing task2 return False
        for station in range(NO_S):
            if task1 in partition[station]:
                index_task1 = station
            if task2 in partition[station]:
                index_task2 = station
        if index_task1 > index_task2:
            return False
    return True


# Function that generates all possible actions
# def generate_actions(old_action, actions_list, all_tasks, number_of_stations, number_of_models):
#     action = old_action.copy()
#     '''Generates all possible actions for a given instance. Returns a list of actions'''
#     print('remaining tasks', all_tasks)
#     if not all_tasks:
#         actions_list.append(list(action))
#         print('action list', actions_list)
#         return
#     for station in range(number_of_stations):
#         for task in all_tasks:
#             #remove task from copy of all_tasks
#             all_tasks_copy = all_tasks.copy()
#             all_tasks_copy.remove(task)
#             print('action before', action)
#             action[station] += [ task]
#             print('action after', action)
#             generate_actions(action, actions_list, all_tasks_copy,  number_of_stations, number_of_models)
# def calculate_time_and_workers(tasks, instance, model, takt_time, max_workers):
#     total_time = 0
#     for task in tasks:
#         total_time += instance[model]['task_times'][task]
#     workers = 1

#     while total_time / workers > takt_time:
#         workers += 1
#         if workers > max_workers:
#              return total_time, None, None
#     reduced_time = total_time / workers
#     return total_time, reduced_time, workers
# Function that generates all possible actions
def create_A(D, model_histories, instances, all_tasks, TAKT_TIME, MAX_L, NO_S):
    def create_action(
        d,
        A,
        a,
        a_count,
        instance,
        takt_time,
        max_workers,
        model_histories,
        current_station,
        no_s,
    ):

        if current_station == no_s:
            a_copy = copy.deepcopy(a)
            a_copy["state_index"] = d["index"]
            a_copy["action_index"] = str(d["index"]) + "_" + str(a_count)
            a_copy["total_workers"] = sum(
                a_copy[f"workers_at_{station}"] for station in range(no_s)
            )
            A.append(a_copy)
            return a_count + 1
        # Stop exploring branch if the previous statement was infeasible
        if current_station > 0 and not a[f"reduced_time_at_{current_station-1}"]:
            return a_count
        for history in model_histories[f'model_{d[f"model_at_{current_station}"]}']:
            if (
                d[f"history_at_{current_station}"]
                == history[f"station_{current_station}"]
            ):
                a_copy = copy.deepcopy(a)
                model_at_station = d[f"model_at_{current_station}"]
                a_copy[f"model_at_{current_station}"] = model_at_station
                if current_station < no_s - 1:
                    a_copy[f"history_at_{current_station}"] = d[
                        f"history_at_{current_station}"
                    ]
                    a_copy[f"action_at_{current_station}"] = list(
                        set(history[f"station_{current_station + 1}"])
                        - set(d[f"history_at_{current_station}"])
                    )
                else:
                    a_copy[f"history_at_{current_station}"] = d[
                        f"history_at_{current_station}"
                    ]
                    a_copy[f"action_at_{current_station}"] = list(
                        set(all_tasks) - set(d[f"history_at_{current_station}"])
                    )
                (
                    a_copy[f"total_task_time_at_{current_station}"],
                    a_copy[f"reduced_time_at_{current_station}"],
                    a_copy[f"workers_at_{current_station}"],
                ) = calculate_time_and_workers(
                    a_copy[f"action_at_{current_station}"],
                    instance,
                    int(model_at_station),
                    takt_time,
                    max_workers,
                )

                a_count = create_action(
                    d,
                    A,
                    a_copy,
                    a_count,
                    instance,
                    takt_time,
                    max_workers,
                    model_histories,
                    current_station + 1,
                    no_s,
                )
        return a_count

    A = []
    a_count = 0
    # create action(s) for each state
    for d in D:
        action = []
        a_count = create_action(
            d,
            action,
            {},
            a_count,
            instances,
            TAKT_TIME,
            MAX_L,
            model_histories,
            0,
            NO_S,
        )
        A = A + action
    return A


def state_transistion_probability(
    action, current_state, next_state, NO_S, entry_probs=(0.4, 0.6)
):
    """Given an action, gives the probability of going from current_state to next_state"""
    # Can only go to states who share models and history between station n and n+1
    for station in range(NO_S - 1):
        if (
            current_state["history_at_" + str(station)]
            + action["action_at_" + str(station)]
            != next_state["history_at_" + str(station + 1)]
        ):
            return 0
        elif (
            current_state["model_at_" + str(station)]
            != next_state["model_at_" + str(station + 1)]
        ):
            return 0
    # TODO modify the entry probabilities if we have constraints on the number of models of a given product in a line
    # returns the likelihood of going from current_state to next_state given action. This just depends on what model enters the first station
    return entry_probs[int(next_state["model_at_0"])]


def create_D(omega, model_histories, NO_S):
    def create_picture_hist(
        picture, D, d, d_count, model_histories, current_station, no_s
    ):
        if current_station >= no_s:
            d["index"] = d_count
            D.append(d)
            return d_count + 1
        if current_station == 0:
            d_copy = copy.deepcopy(d)
            d_copy[f"model_at_{current_station}"] = picture[current_station]
            d_copy[f"history_at_{current_station}"] = []
            d_count = create_picture_hist(
                picture, D, d_copy, d_count, model_histories, current_station + 1, no_s
            )
        else:
            possible_histories = (
                list(x)
                for x in set(
                    tuple(sorted(history[f"station_{current_station}"]))
                    for history in model_histories[f"model_{picture[current_station]}"]
                )
            )
            for history in possible_histories:
                d_copy = copy.deepcopy(d)
                d_copy[f"model_at_{current_station}"] = picture[current_station]
                d_copy[f"history_at_{current_station}"] = history
                d_count = create_picture_hist(
                    picture,
                    D,
                    d_copy,
                    d_count,
                    model_histories,
                    current_station + 1,
                    no_s,
                )
        return d_count

    D = []
    d_count = 0
    for picture in omega:
        picture_hist = []
        d_count = create_picture_hist(
            picture, picture_hist, {}, d_count, model_histories, 0, NO_S
        )
        # print('picture_hist', picture_hist)
        D = D + picture_hist
    return D


# Function that generates all possible actions that have already been performed all models
def create_model_histories(feasible_partitions, models, NO_S):
    model_histories = {}
    for model in models:
        feasible_histories = []
        for index, partition in enumerate(feasible_partitions[f"model_{model}"]):
            feasible_history = {}
            feasible_history["index"] = index
            feasible_history["station_0"] = []
            completed_tasks = partition[0]
            for station in range(1, NO_S):
                feasible_history[f"station_{station}"] = completed_tasks.copy()
                completed_tasks = completed_tasks + partition[station]
            feasible_histories.append(feasible_history)
        model_histories[f"model_{model}"] = feasible_histories
    return model_histories
