import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
from tqdm import tqdm
from morphogenesis.hyphae import Network
from morphogenesis.domains import domain_lookup_2d, domain_lookup_3d

def plot_network(network):
    fig = go.Figure()

    fig.add_trace(go.Scattergl(x=network.vein_nodes[:, 0],
                             y=network.vein_nodes[:, 1],
                             mode="markers",
                             marker=dict(size=2, color="blue")))
    for edge in network.edges:
        node_idx1, node_idx2 = edge.source, edge.target
        node1 = network.vein_nodes[node_idx1]
        node2 = network.vein_nodes[node_idx2]

        fig.add_trace(go.Scattergl(x=[node1[0], node2[0]],
                                y=[node1[1], node2[1]],
                                mode="lines",
                                line=dict(width=1, color="blue")))
        
    fig.update_layout(showlegend=False)

    return fig

def plot_network_3d(network):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=network.vein_nodes[:, 0],
                                y=network.vein_nodes[:, 1],
                                z=network.vein_nodes[:, 2],
                                mode="markers",
                                marker=dict(size=2, color="blue")))
    for edge in network.edges:
        node_idx1, node_idx2 = edge.source, edge.target
        node1 = network.vein_nodes[node_idx1]
        node2 = network.vein_nodes[node_idx2]

        fig.add_trace(go.Scatter3d(x=[node1[0], node2[0]],
                                   y=[node1[1], node2[1]],
                                   z=[node1[2], node2[2]],
                                   mode="lines",
                                   line=dict(width=1, color="blue")))
        
    fig.update_layout(showlegend=False)

    return fig

app = dash.Dash(__name__, external_stylesheets=['assets/dashstyle.css'])
start_domain = domain_lookup_2d["circle2d"]["function"](4)
NET = Network(start_domain,
              root_node=start_domain.generate(),
              dim = 2)

params = {"new_vein_distance" : 0.05,
          "birth_distance_v" : 0.05,
          "birth_distance_a" : 0.05,
          "max_birth_distance_a" : 1.0,
          "kill_distance" : 0.1,
          "darts_per_step" : 100,
          "start_size" : 4,
          "start_width" : 2.7,
          "width_decay" : 0.98,
          "distance_decay" : 0.99,
          "kill_decay" : 0.995,
          "max_auxin_sources" : 300,
          "source_drift" : 0.04}

param_divs = []
for name, value in params.items():
    param_divs.append(html.Div([
        html.Label(name),
        dcc.Input(
            id=f"{name}-input",
            type="number",
            value=value,
            className="input"
        )],
        className="control-container"))


app.layout = html.Div([
    html.Div([
        html.Button("Reset",
                    id="reset-button",
                    n_clicks=0,
                    className="button"),
        html.Br(),
        html.Button("Run",
                    id="run-button",
                    n_clicks=0,
                    className="button"),
        html.Div([
            html.Label("Number of Steps"),
            dcc.Input(
                id="numsteps",
                type="number",
                className="input",
                value=30
            )],
            className="control-container"),
        html.Br(),
        html.Button("Step",
                    id="step-button",
                    n_clicks=0,
                    className="button"),
        html.Div([
            html.Label("Dimension"),
            dcc.Dropdown(
                id="dimension-dropdown",
                options=[
                    {"label" : "2D", "value" : 2},
                    {"label" : "3D", "value" : 3}
                ],
                value=2,
                className="dropdown"
            )],
            className="control-container"),
        *param_divs,
        html.Div([
            html.Label("Decay Type"),
            dcc.Dropdown(
                id="decay-type-dropdown",
                options=[
                    {"label" : "Scale Free", "value" : "scale_free",
                     "label" : "Exponential", "value" : "exponential"},
                ],
                value="scale_free",
                className="dropdown"
            )],
            className="control-container"),
        html.Div([
            html.Label("Domain Type"),
            dcc.Dropdown(
                id="domain-type-dropdown",
                options=[
                ],
                className="dropdown"
            )],
            className="control-container"),
        html.Div([
            html.Label("Domain Parameters"),
            html.Div(children = [], id="domain-parameters")],
            className="control-container"),
    ], className="sidebar"),

    html.Div([
        html.Div([
            html.H1("Dashboard"),
        ], className="header"),

        html.Div([
            html.Div([
                html.Div("0", id="iterations", className="stat-value"),
                html.Div("Number of Iterations", className="stat-label")
            ], className="stat-card"),
            html.Div([
                html.Div("0", id="nodes", className="stat-value"),
                html.Div("Number of Nodes", className="stat-label")
            ], className="stat-card"),
            html.Div([
                html.Div("0", id="edges", className="stat-value"),
                html.Div("Number of Edges", className="stat-label")
            ], className="stat-card"),
            html.Div([
                html.Div("0", id="polygons", className="stat-value"),
                html.Div("Estimated Polygons", className="stat-label")
            ], className="stat-card")
        ], className="stats-row"),


        html.Div([
            html.H2("Representation"),
            dcc.Graph(id='main-chart')
        ], className="card"),],
        className="main-content"),

], className="dashboard-container")

@callback(
    Output("domain-parameters", "children"),
    [Input("domain-type-dropdown", "value")]
)
def update_domain_parameters(domain_type):
    if domain_type in domain_lookup_2d:
        domain_function = domain_lookup_2d[domain_type]["function"]
        domain_parameters = domain_lookup_2d[domain_type]["parameters"]

    elif domain_type in domain_lookup_3d:
        domain_function = domain_lookup_3d[domain_type]["function"]
        domain_parameters = domain_lookup_3d[domain_type]["parameters"]

    else:
        raise dash.exceptions.PreventUpdate

    dropdowns = []
    reset_args = {}
    for name, details in domain_parameters.items():
        reset_args[name] = details["default"]
        if details["type"] == "float":
            dropdowns.append(dcc.Input(
                id=f"{name}-input",
                type="number",
                value=details["default"],
                className="input"
            ))
        elif details["type"] == "bool":
            dropdowns.append(dcc.Dropdown(
                id=f"{name}-input",
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False}
                ],
                value=details["default"],
                className="dropdown"
            ))

    NET.reset(domain = domain_function(**reset_args))

    return dropdowns

@callback(
    Output("domain-type-dropdown", "options"),
    [Input("dimension-dropdown", "value")]
)
def update_domain_type_options(dimension):
    if dimension != NET.dim:
        NET.reset(domain = new_domain, dim = dimension)

    values = []
    if dimension == 2:
        for name in domain_lookup_2d.keys():
            values.append({"label" : name, "value" : name})
    elif dimension == 3:
        for name in domain_lookup_3d.keys():
            values.append({"label" : name, "value" : name})

    return values


@callback(
     Output("reset-button", "n_clicks"),
    [Input("reset-button", "n_clicks")],
    [State("new_vein_distance-input", "value"),
     State("birth_distance_v-input", "value"),
     State("birth_distance_a-input", "value"),
     State("max_birth_distance_a-input", "value"),
     State("kill_distance-input", "value"),
     State("darts_per_step-input", "value"),
     State("start_size-input", "value"),
     State("start_width-input", "value"),
     State("width_decay-input", "value"),
     State("distance_decay-input", "value"),
     State("kill_decay-input", "value"),
     State("max_auxin_sources-input", "value"),
     State("source_drift-input", "value")]
)
def reset_network(reset_button, new_vein_distance, birth_distance_v, birth_distance_a, max_birth_distance_a, kill_distance, darts_per_step, start_size, start_width, width_decay, distance_decay, kill_decay, max_auxin_sources, source_drift):
    if reset_button:

        NET.reset(new_vein_distance = new_vein_distance,
                  birth_distance_v = birth_distance_v,
                  birth_distance_a = birth_distance_a,
                  max_birth_distance_a = max_birth_distance_a,
                  kill_distance = kill_distance,
                  darts_per_step = darts_per_step,
                  start_size = start_size,
                  start_width = start_width,
                  width_decay = width_decay,
                  distance_decay = distance_decay,
                  kill_decay = kill_decay,
                  max_auxin_sources = max_auxin_sources,
                  source_drift = source_drift)

        return 0
    raise dash.exceptions.PreventUpdate

@callback(
    [Output("iterations", "children"),
     Output("nodes", "children"),
     Output("edges", "children"),
     Output("polygons", "children"),
     Output("main-chart", "figure"),
     Output("run-button", "n_clicks")],
    [Input("run-button", "n_clicks"),
     Input("numsteps", "value"),
     Input("iterations", "children"),
     Input("nodes", "children"),
     Input("edges", "children"),
     Input("polygons", "children"),
     Input("dimension-dropdown", "value")],
)
def update_dashboard(run_button, numsteps, iterations, nodes, edges, polygons, dim):
    if run_button:
        for _ in tqdm(range(numsteps)):
            NET.step()
        # iterations += numsteps

        if dim == 2:
            fig = plot_network(NET)
        elif dim == 3:
            fig = plot_network_3d(NET)

        fig.update_layout(height = 800,
                          autosize = True)
    
        return iterations, nodes, edges, polygons, fig, 0
    
    raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)
