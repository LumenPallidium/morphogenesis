import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
from tqdm import tqdm
from morphogenesis.hyphae import Network
from morphogenesis.domains import domain_lookup_2d, domain_lookup_3d

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


    return fig

app = dash.Dash(__name__, external_stylesheets=['assets/dashstyle.css'])
start_domain = domain_lookup_3d["thick_cut_ellipsoid"]["function"](4, 1.5, 2.5, 0.3)
NET = Network(start_domain,
              root_node=start_domain.generate(),
              dim = 3)

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
                value=150
            )],
            className="control-container"),
        html.Br(),
        html.Button("Step",
                    id="step-button",
                    n_clicks=0,
                    className="button"),
        html.Div([
            html.Label("Step Size"),
            dcc.Slider(
                id="step-size-slider",
                min=0,
                max=10,
                step=0.1,
                value=1,
                marks={i: str(i) for i in range(11)},
                className="slider"
            )],
            className="control-container"),
        html.Div([
            html.Label("Domain Type"),
            dcc.Dropdown(
                id="domain-type-dropdown",
                options=[
                    {"label" : "Ellipsoid", "value" : "thick_cut_ellipsoid"},
                ],
                value="thick_cut_ellipsoid",
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
     Input("polygons", "children")],
)
def update_dashboard(run_button, numsteps, iterations, nodes, edges, polygons):
    if run_button:
        for _ in tqdm(range(numsteps)):
            NET.step()
        # iterations += numsteps

        fig = plot_network_3d(NET)
    
        return iterations, nodes, edges, polygons, fig, 0
    
    raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)
