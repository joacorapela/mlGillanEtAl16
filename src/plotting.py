
import plotly.graph_objects as go

def getBoxPlots(data, ylab, ylim=None, xlab="", showLegend=False):
    # data is a dictionary with key equal to series name and value
    # equal to series values
    fig = go.Figure()
    for key in data.keys():
        fig.add_trace(go.Box(y=data[key], name=key, boxpoints="all"))
    fig.update_xaxes(title_text=xlab)
    if ylim is not None:
        fig.update_yaxes(title_text=ylab, range=ylim)
    else:
        fig.update_yaxes(title_text=ylab)
    fig.update_layout(showlegend=showLegend)
    return fig

