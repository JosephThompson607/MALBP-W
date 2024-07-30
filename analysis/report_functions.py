import gurobi_logtools as glt
import plotly.graph_objects as go
import pandas as pd

def plot_incumbent_vs_bound(fp, log_file, show=True, save_output=True):
    '''Plots incumbent vs bound for Gurobi log file'''
    summary = glt.parse(fp+ log_file)
    nl = summary.progress("nodelog")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nl["Time"], y=nl["Incumbent"], name="Primal Bound"))
    fig.add_trace(go.Scatter(x=nl["Time"], y=nl["BestBd"], fill="tonexty", name="Dual Bound"))
    fig.update_xaxes(title_text="Runtime")
    fig.update_yaxes(title_text="Objective function value")
    if show:
        fig.show()
    if save_output:
        fig.write_image(fp + log_file.split(".")[0] + ".png")
    return fig

def get_integer_gap(intance_log):
    results = glt.parse([intance_log])
    summary = results.summary()
    #gets the value of the cell for summary['MIPGap']
    return summary.iloc[0]['MIPGap']

def read_csv_add_gap(df, gapname= 'gap'):
    df[gapname] = df['log'].apply(get_integer_gap)*100
    return df

def read_csv_add_log(log_file_location, file_name_stuff, df):
    df['log'] = df['group_counter'].apply(lambda x: log_file_location+ f'{file_name_stuff}{x}.log')
    return df

def get_log(log_file_location):
    #look for .log files in all folders in the log_file_location
    log_files = []
    for root, dirs, files in os.walk(log_file_location):
        for file in files:
            if file.endswith(".log"):
                print("Found log file: ", os.path.join(root, file))
                log_files.append(os.path.join(root, file))
    return log_files

def log_to_summary(log_file_location, csv_config_file):
    log_files = get_log(log_file_location)
    summary = summarize_logs(log_files, csv_config_file)
    return summary

def get_row_from_file_path(file_path, row_location = 8):
    #gets the row of the csv run file from the file path. 
    file_path = file_path.split("/")
    file_path = file_path[row_location]
    return file_path


def summarize_logs(log_files, csv_config_file):
    results = glt.parse(log_files)
    summary = results.summary(results)
    #Only keeps objVal, MIPGap, LogFilePath
    summary = summary[['ObjVal', 'MIPGap', 'LogFilePath']]
    summary['row'] = summary['LogFilePath'].apply(lambda x: int(get_row_from_file_path(x)))
    summary = summary.sort_values(by='row')
    summary.reset_index(drop=True, inplace=True)
    #groups the summary table by every 10 rows, takes the average and standard deviation of the ObjVal and MIPGap
    config_df = pd.read_csv(csv_config_file)
    #merges the summary table with the config file, using the index of the config file and the row of the summary table
    summary = pd.merge( config_df,summary, left_index=True, right_on='row')
    #Groups by the model yaml
    #summary = summary.groupby('model_yaml').agg({'ObjVal': ['mean', 'std'], 'MIPGap': ['mean', 'std']})
    # summary['group'] = summary.index // instance_repeat
    # summary = summary.groupby('group').agg({'ObjVal': ['mean', 'std'], 'MIPGap': ['mean', 'std']})
    #creates the name from the model_yaml column. It is the text before the .yaml and after the last backslash
    summary['instance_name'] = summary['model_yaml'].apply( lambda x: x.split("/")[-1].split(".")[0])
    return summary

def calculate_station_entropy(task_assignments, model_prob, model_column = 'item'):
    '''This function calculates the task assignment entropy at each station.
         The inputs are the x_soi or x_wsoj csv files from an output of the model. 
         parameters:
            task_assignments: dataframe, the output of the model. It should have columns 'station', 'task', 'item' (or what is set with model column), 'value'
            model_prob: the demand ratio or probability of a model entering the line
             model_column: the column name of the model in the task_assignments dataframe'''
    task_assignments = task_assignments[task_assignments['value']>0]
    task_prob_df = task_assignments.groupby([model_column, 'task'])['station'].value_counts(normalize=True).reset_index(name='task_prob')
    #groups the task_prob_df  by station and task and creates a lists the items that have been assigned to that station and task
    shared_tasks = task_prob_df.groupby(['task', 'station'])[model_column].apply(list).reset_index(name='shared_items')
    #merges the shared_tasks with the task_prob_df to get the probability of each task being assigned to each station
    task_prob_df = task_prob_df.merge(shared_tasks, on=['task', 'station'])
    #multiplies the task probality by the sum of the model_prob that have a model in the shared_items
    task_prob_df['total_prob'] = task_prob_df['task_prob'] * task_prob_df['shared_items'].apply(lambda x: sum([model_prob[item] for item in x]))
    task_prob_df['entropy'] = -task_prob_df['total_prob'] * task_prob_df['total_prob'].apply(lambda x: np.log2(x))
    task_entropy = task_prob_df.groupby('station')['entropy'].sum()
    return task_entropy