# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display':'grid', 'grid-auto-flow': 'row'}
row_style = {'display':'grid', 'grid-auto-flow': 'column'}

import plotly.express as px
import pandas as pd

import requests

app = Dash(__name__)

df = pd.read_csv("iris_extended_encoded.csv",sep=',')
df_csv = df.to_csv(index=False)

app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Tabs([
    dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

    html.Div([
        html.Div([
            html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
            dcc.Input(id='file-for-train', type='text', style={'width':'100px'}),
            html.Div([
                html.Button('Load', id='load-val', style={"width":"60px", "height":"30px"}),
                html.Div(id='load-response', children='Click to load')
            ], style=col_style)
        ], style=col_style),

        html.Div([
            html.Button('Upload', id='upload-val', style={"width":"60px", "height":"30px"}),
            html.Div(id='upload-response', children='Click to upload')
        ], style=col_style| {'margin-top':'20px'})

    ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),


html.Div([
    html.Div([
        html.Div([
            html.Label(['Feature'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                df.columns, # dropdown values for histogram
                df.columns[0], # default value for dropdown
                id='hist-column'
            )
            ], style=col_style ),
        dcc.Graph( id='selected_hist' )
    ], style=col_style | {'height':'400px', 'width':'400px'}),

    html.Div([

    html.Div([

    html.Div([
        html.Label(['X-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns, # dropdown values for scatter plot x-axis
            df.columns[0], # default value for dropdown
            id='xaxis-column'
            )
        ]),

    html.Div([
        html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns, # dropdown values for scatter plot y-axis
            df.columns[1], # default value for dropdown
            id='yaxis-column'
            )
        ])
    ], style=row_style | {'margin-left':'50px', 'margin-right': '50px'}),

    dcc.Graph(id='indicator-graphic')
    ], style=col_style)
], style=row_style),

    html.Div(id='tablecontainer', children=[
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
            id='datatable' )
        ])
    ]),
    dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Button('New model', id='build-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='build-response', children='Click to build new model and train')
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Re-Train', id='train-val', style={"width":"90px", "height":"30px"})
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-train', children='')
    ]),
    dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='row-for-score', type='text', style={'width':'300px'}))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-score', type='text'))
            ], style=col_style | {'margin-top':'20px'}),            
            html.Div([
                html.Button('Score', id='score-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='score-response', children='Click to score')
            ], style=col_style | {'margin-top':'20px'})
        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),
        
        html.Div(id='container-button-score', children='')
    ]),

    dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Test', id='test-val'),
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-test', children='')
    ])

    ])
])

# callbacks for Explore data tab

@app.callback(
    Output('load-response', 'children'),
    Input('load-val', 'n_clicks'),
    State('file-for-train', 'value')
)
def update_output_load(nclicks, filename):
    global df, df_csv

    if nclicks != None and filename:
        # load local data given input filename
        try:
            df = pd.read_csv(filename, sep=',')
            df_csv = df.to_csv(index=False)
            return f'Loaded {filename} successfully.'
        except Exception as e:
            return f'Error loading file: {str(e)}'
    else:
        return 'Click to load'


@app.callback(
    Output('build-response', 'children'),
    Input('build-val', 'n_clicks'),
    State('dataset-for-train', 'value')
)
def update_output_build(nclicks, dataset_id):
    if nclicks != None and dataset_id:
        # invoke new model endpoint to build and train model given data set ID
        try:
            # Set proper headers for form data
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            
            # Send dataset_id as form data
            r = requests.post(
                'http://localhost:4000/iris/model',
                headers=headers,
                data={'dataset': dataset_id}
            )
            
            if r.status_code == 201:
                model_id = r.json().get('model_id', 'unknown')
                return f'Model built and trained with ID: {model_id}'
            else:
                return f'Error: {r.text}'
        except Exception as e:
            return f'Error building model: {str(e)}'
    else:
        return 'Click to build new model and train'


@app.callback(
    Output('upload-response', 'children'),
    Input('upload-val', 'n_clicks'),
    State('file-for-train', 'value')
)
def update_output_upload(nclicks, filename):
    if nclicks != None:
        # Use the provided filename or default if none provided
        file_to_use = filename if filename else 'iris_extended_encoded.csv'
        
        # invoke the upload API endpoint with form-data
        try:
            # Create a form-data with 'train' as the key and filename as the value
            form_data = {'train': file_to_use}
            r = requests.post('http://localhost:4000/iris/datasets', data=form_data)
            if r.status_code == 201:
                dataset_id = r.json().get('dataset_id', 'unknown')
                return f'Dataset uploaded with ID: {dataset_id}'
            else:
                return f'Error: {r.text}'
        except Exception as e:
            return f'Error uploading dataset: {str(e)}'
    else:
        return 'Click to upload'

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('load-response', 'children')]
)
def update_graph(xaxis_column_name, yaxis_column_name, load_response):
    fig = px.scatter(df, 
                     x=xaxis_column_name,
                     y=yaxis_column_name,
                     color='species')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=xaxis_column_name)
    fig.update_yaxes(title=yaxis_column_name)

    return fig


@app.callback(
    Output('selected_hist', 'figure'),
    [Input('hist-column', 'value'),
     Input('load-response', 'children')]
)
def update_hist(hist_column_name, load_response):
    fig = px.histogram(df, x=hist_column_name, color='species')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=hist_column_name)

    return fig

@app.callback(
    Output('datatable', 'data'),
    Output('datatable', 'columns'),
    Input('load-response', 'children')
)
def update_table(load_response):
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]


# callbacks for Training tab
@app.callback(
    Output('container-button-train', 'children'),
    Input('train-val', 'n_clicks'),
    State('model-for-train', 'value'),
    State('dataset-for-train', 'value')
)
def update_output_train(nclicks, model_id, dataset_id):
    if nclicks is None or not model_id or not dataset_id:
        return "Please provide both model ID and dataset ID."
    try:
        model_id = int(model_id)
        dataset_id = int(dataset_id)
    except ValueError:
        return "Error: Model ID and Dataset ID must be valid integers."
    if dataset_id < 0:
        return "Error: Dataset ID must be a non-negative integer."
    try:
        r = requests.put(
            f'http://localhost:4000/iris/model/{model_id}?dataset={dataset_id}'
        )

        if r.status_code == 200:
            train_df = pd.DataFrame(r.json()['training_history'])
            train_fig = px.line(train_df)
            return dcc.Graph(figure=train_fig)
        else:
            return f"Error: {r.text}"
    except Exception as e:
        return f"Error during training: {str(e)}"

# callbacks for Scoring tab

@app.callback(
    Output('score-response', 'children'),
    Input('score-val', 'n_clicks'),
    State('model-for-score', 'value'),
    State('row-for-score', 'value')
)
def update_output_score(nclicks, model_id, row_text):
    if nclicks != None and model_id and row_text:
        # add API endpoint request for scoring here with constructed input row
        try:
            print(model_id,row_text)
            url = f'http://localhost:4000/iris/model/{model_id}/score?fields={row_text}'
            r = requests.get(url)
            if r.status_code == 200:
                score_result = r.text
                return f"Score result: {score_result}"
            else:
                return f"Error: {r.text}"
        except Exception as e:
            return f"Error during scoring: {str(e)}"
    else:
        return "Click to score"
    

# callbacks for Testing tab

@app.callback(
    Output('container-button-test', 'children'),
    Input('test-val', 'n_clicks'),
    State('model-for-test', 'value'),
    State('dataset-for-test', 'value')
)
def update_output_test(nclicks, model_id, dataset_id):
    if nclicks != None and model_id and dataset_id:
        # add API endpoint request for testing with given dataset ID
        try:
            r = requests.get(f'http://localhost:4000/iris/model/{model_id}/test?dataset={dataset_id}')
            if r.status_code == 200:
                test_df = pd.DataFrame(r.json())
                test_fig = px.line(test_df)
                return dcc.Graph(figure=test_fig)
            else:
                return f"Error: {r.text}"
        except Exception as e:
            return f"Error during testing: {str(e)}"
    else:
        return ""


if __name__ == '__main__':
    app.run(debug=True)