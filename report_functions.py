import grblogtools as glt
import plotly.graph_objects as go

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