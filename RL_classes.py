import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from ALB_heuristics import immediate_update_first_fit
def import_assignment( tasks_dict,  **kwargs): 
    '''Copies over priority list from another priority list'''
    assignments = kwargs['assignments_dict']
    for task in tasks_dict:
        tasks_dict[task]['score'] = len(tasks_dict) - assignments.index(int(task)-1)



class ALB_instance_graph():
    def __init__(self, instance, device) -> None:
        precedence_relations = [[int(precedence[0]), int(precedence[1])] for precedence in instance['precedence_relations']]
        self.edge_index = torch.tensor(precedence_relations, dtype=torch.long).to(device)
        self.x = torch.tensor([[instance['task_times'][str(task)],1] for task in instance['task_times'].keys()], dtype=torch.float).to(device)
        self.data = Data(x=self.x, edge_index=self.edge_index.t().contiguous()).to(device)
        self.action_mask = torch.ones((self.x.shape[0])).to(device)
        self.priority_list = []
        self.instance = instance
        
    def update_state(self, task_index):
        '''Updates the state of the priority list and graph, 
        and returns reward if at the end of the episode'''
        #update the mask that bans certain actions
        self.action_mask[task_index] = 0
        #Update the network to reflect that a task has been listed
        self.x[task_index, 1] = 0
        self.data = Data(x=x, edge_index=edge_index.t().contiguous())
        self.priority_list.append(int(task_index))

        if torch.sum(self.action_mask)== 0:
            no_stations, task_assignment = immediate_update_first_fit(self.instance, import_assignment, max_stations=30, assignments_dict = self.priority_list)
            return -no_stations, True
        return 0, False
    
    def reset(self):
        self.action_mask = torch.ones((x.shape[0]))
        self.priority_list = []
        self.x = torch.tensor([[self.instance['task_times'][str(task)],1] for task in self.instance['task_times'].keys()], dtype=torch.float)
        self.data = Data(x=x, edge_index=self.edge_index.t().contiguous())



#enivorment for reinforcement learning of priority rules
class Env():
    def __init__(self, instance, device) -> None:
        precedence_relations = [[int(precedence[0]), int(precedence[1])] for precedence in instance['precedence_relations']]
        self.edge_index = torch.tensor(precedence_relations, dtype=torch.long).to(device)
        self.x = torch.tensor([[instance['task_times'][str(task)],1] for task in instance['task_times'].keys()], dtype=torch.float).to(device)
        self.data = Data(x=self.x, edge_index=self.edge_index.t().contiguous()).to(device)
        self.action_mask = torch.ones((x.shape[0])).to(device)
        self.priority_list = []
        self.instance = instance
        

    
    def update_state(self, task_index):
        '''Updates the state of the priority list and graph, 
        and returns reward if at the end of the episode'''
        #update the mask that bans certain actions
        self.action_mask[task_index] = 0
        #Update the network to reflect that a task has been listed
        self.x[task_index, 1] = 0
        self.data = Data(x=x, edge_index=edge_index.t().contiguous())
        self.priority_list.append(int(task_index))

        if torch.sum(self.action_mask)== 0:
            no_stations, task_assignment = immediate_update_first_fit(self.instance, import_assignment, max_stations=30, assignments_dict = self.priority_list)
            return -no_stations, True
        return 0, False
    
    def reset(self):
        self.action_mask = torch.ones((x.shape[0]))
        self.priority_list = []
        self.x = torch.tensor([[self.instance['task_times'][str(task)],1] for task in self.instance['task_times'].keys()], dtype=torch.float)
        self.data = Data(x=self.x, edge_index=self.edge_index.t().contiguous())


def get_action(env, model, device = 'cpu'):
    '''Gets the action from the model and returns the action and the log probability of the action'''
    with torch.no_grad():
        action_probs = model(env.data.to(device)).squeeze()
        action_probs = action_probs * env.action_mask.to(device)
        if action_probs.sum() == 0:
            action_probs = torch.ones(action_probs.shape).to(device)
        action = torch.multinomial(action_probs, 1)
    return action, action_probs