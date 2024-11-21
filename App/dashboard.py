#### Import Modules 
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
from datetime import datetime
from google.cloud import bigquery
import networkx as nx
import plotly.graph_objects as go
import polars as pl
import igraph as ig
import leidenalg
from scipy import stats 
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import community as community_louvain
from networkx.algorithms.community import label_propagation_communities, modularity
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
import shap
import matplotlib.pyplot as plt 
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg') 

#### Load data 
# Load yearly and monthly metrics data 
yearly_stats = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Yearly_stats.csv')
monthly_stats = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Monthly_stats.csv')

# First year trend data 
first_year_trend = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/first_year_trend.csv')

# ETH in-degree and out-degree data 
in_deg_dist = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Ethereum/Month by month/In-degree distribution.csv')
out_deg_dist = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Ethereum/Month by month/Out-degree distribution.csv')
in_deg_dist['Date'] = pd.to_datetime(in_deg_dist['Date'], format = '%Y-%m')
out_deg_dist['Date'] = pd.to_datetime(out_deg_dist['Date'], format = '%Y-%m')

# BTC in-degree and out-degree data 
in_deg1 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/in_deg_dist_from2019.csv')
in_deg2 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/in_deg_dist_till2018.csv')
out_deg1 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/out_deg_dist_till2018.csv')
out_deg2 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/out_deg_dist_from2019.csv')
btc_in_deg_dist = pd.concat([in_deg1, in_deg2])
btc_out_deg_dist = pd.concat([out_deg1, out_deg2])

# BTC influential sender and recipient 
inf_sender_till2019 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/Most influential sender till 2019.csv')
inf_sender_from2020 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/Most influential sender from 2020.csv')
inf_recipient_till2019 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/Most influential recipient till 2019.csv')
inf_recipient_from2020till2022 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/Most influential recipient from 2020 to 2022.csv')
inf_recipient_till2023 = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Bitcoin/Month by month/Most influential recipient 2023.csv')
btc_influential_sender = pd.concat([inf_sender_from2020, inf_sender_till2019])
btc_influential_recipient = pd.concat([inf_recipient_from2020till2022, inf_recipient_till2019, inf_recipient_till2023])
btc_in_deg_dist['Date'] = pd.to_datetime(btc_in_deg_dist['Date'], format = '%Y-%m')
btc_out_deg_dist['Date'] = pd.to_datetime(btc_out_deg_dist['Date'], format = '%Y-%m')
btc_influential_sender['Date'] = pd.to_datetime(btc_influential_sender['Date'], format = '%Y-%m')
btc_influential_recipient['Date'] = pd.to_datetime(btc_influential_recipient['Date'], format = '%Y-%m')

# Iotex in-degree and out-degree data 
iotex_in_deg_dist = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Iotex/Month by month/In-degree distribution.csv')
iotex_out_deg_dist = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Iotex/Month by month/Out-degree distribution.csv')
iotex_in_deg_dist['Date'] = pd.to_datetime(iotex_in_deg_dist['Date'], format = '%Y-%m')
iotex_out_deg_dist['Date'] = pd.to_datetime(iotex_out_deg_dist['Date'], format = '%Y-%m')

# Iotex data for transaction subgraph 
iotex_2022_transaction_pd = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Iotex/Month by month/2022 transaction network.csv')
iotex_2022_transaction = pl.from_pandas(iotex_2022_transaction_pd)

# Tezos in-degree and out-degree data 
tezos_in_deg_dist = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Tezos/Month by month/In degree distribution.csv')
tezos_out_deg_dist = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Tezos/Month by month/Out degree distribution.csv')
tezos_in_deg_dist['Date'] = pd.to_datetime(tezos_in_deg_dist['Date'], format = '%Y-%m')
tezos_out_deg_dist['Date'] = pd.to_datetime(tezos_out_deg_dist['Date'], format = '%Y-%m')

# Tezos data for transaction subgraph 
tezos_2022_transaction_pd = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/Tezos/Month by month/2022JunetoDec_transaction.csv')
tezos_2022_transaction = pl.from_pandas(tezos_2022_transaction_pd)

# Anomaly detction data 
ethereum_anomaly = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/ethereum_anomalies.csv')
iotex_anomaly = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/iotex_anomalies.csv')
tezos_anomaly = pd.read_csv('/Users/kelvinfoo/Desktop/Crypto Research/Bitcoin and Ethereum Basic Statistics/tezos_anomalies.csv')

#### Initialize the Dash app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
server = app.server

app.layout = html.Div([
    html.H1("Cryptocurrency Dashboard", style={'text-align': 'center'}),
    html.Hr(),
    html.Br(),
    html.H2("Common Cryptocurrency Metrics", style={'text-align': 'center'}),
    html.Div(id='crypto-select'),
    html.H4("Select the comparison type:", style={'text-align': 'left'}),
    dcc.RadioItems(
        id='comparison-type',
        options=[ {'label': 'Year by year comparison', 'value': 'yearly'}, {'label': 'Month by month comparison', 'value': 'monthly'}], 
        value='yearly', labelStyle={'display': 'inline-block', 'margin-right': '20px'}),
    html.H4("Select the cryptocurrency: ", style={'text-align': 'left'}),  
    dcc.Checklist(
        id='crypto-selection',
        options=[
            {'label': 'Bitcoin', 'value': 'BTC'},
            {'label': 'Ethereum', 'value': 'ETH'},
            {'label': 'Iotex', 'value': 'Iotex'}, 
            {'label': 'Tezos', 'value': 'Tezos'}
        ],
        value=['ETH'],
        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
    ), 
    html.Div(id='comparison-content'), 
    html.Div(id='crypto-content'), 
    html.Hr(), 
    html.Br(), 
    html.H2("Network Analysis", style={'text-align': 'center'}),
    html.H4("Select the cryptocurrency for network analysis:", style={'text-align': 'left'}),  
    dcc.RadioItems(
        id='network-crypto-selection',
        options=[{'label': 'Bitcoin', 'value': 'BTC'}, {'label': 'Ethereum', 'value': 'ETH'}, {'label': 'Iotex', 'value': 'Iotex'}, {'label': 'Tezos', 'value': 'Tezos'}],
        value='ETH', labelStyle={'display': 'inline-block', 'margin-right': '20px'}), 
    html.Div(id='network-content')])

@app.callback(
    [Output('comparison-content', 'children'), 
     Output('crypto-content', 'children')],
    [Input('comparison-type', 'value'), 
     Input('crypto-selection', 'value')])

def update_comparison_content(selected_type, selected_crypto):
    if selected_type == 'yearly':
        return (
            html.Div([
                html.Br(),
                html.H4("Select the range of years that you are interested in", style={'text-align': 'left'}),
                dcc.RangeSlider(
                    id='year-slider', min=2009, max=2023, step=1, marks={year: str(year) for year in range(2009, 2024)}, value=[2009, 2024]),
                html.Br(),
                html.Div([
                    dcc.Graph(id='transactions-with-contract-plot'),
                    dcc.Graph(id='transactions-contract-plot')
                ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),
                html.Br(), 
                html.Div([
                    dcc.Graph(id='wallet-address'), 
                    dcc.Graph(id = 'transaction-value')
                ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}), 
                html.H4("The transaction value were based on the following conversion rates: 1 BTC = 23.60 ETH, 1 IOTX = 0.000014 ETH and 1 Tezos = 0.000258 ETH.")]),
            html.Div())
    
    elif selected_type == 'monthly':
        return (
            html.Div([
                html.Br(),
                html.H4("Select the range of dates that you are interested in", style={'text-align': 'left'}),
                html.Div([
                    html.Div([
                        html.Label("Start date", style={'font-weight': 'bold'}),
                        html.Div([
                            html.Label("Year", style={'margin-right': '10px'}),
                            dcc.Dropdown(
                                id='start-year',
                                options=[{'label': str(year), 'value': str(year)} for year in range(2015, 2025)],value='2020',
                                style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),
                            html.Label("Month", style={'margin-right': '10px'}),
                            dcc.Dropdown(
                                id='start-month',
                                options=[{'label': str(month), 'value': str(month).zfill(2)} for month in range(1, 13)],value='01',
                                style={'width': '45%', 'display': 'inline-block'})], 
                                style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'})
                    ], style={'display': 'inline-block', 'width': '45%'}),
                    html.Div([
                        html.Label("End date", style={'font-weight': 'bold'}),
                        html.Div([
                            html.Label("Year", style={'margin-right': '10px'}),
                            dcc.Dropdown(
                                id='end-year',
                                options=[{'label': str(year), 'value': str(year)} for year in range(2015, 2025)],
                                value='2022',
                                style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}
                            ),
                            html.Label("Month", style={'margin-right': '10px'}),
                            dcc.Dropdown(
                                id='end-month',
                                options=[{'label': str(month), 'value': str(month).zfill(2)} for month in range(1, 13)],
                                value='12',
                                style={'width': '45%', 'display': 'inline-block'})
                        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'})
                    ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '10%'})], style={'text-align': 'center'}),
                html.Br(),
                html.Div([
                    dcc.Graph(id='month-transactions-with-contract-plot'),
                    dcc.Graph(id='month-transactions-contract-plot')
                ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),
                html.Br(), 
                html.Div([
                    dcc.Graph(id='month-wallet-address'), 
                    dcc.Graph(id = 'month-transaction-value')
                ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}), 
                html.H4("The transaction value were based on the following conversion rates: 1 BTC = 23.60 ETH, 1 IOTX = 0.000014 ETH and 1 Tezos = 0.000258 ETH."), 
                html.Br(), 
                html.H2('Common Cryptocurrency Metrics Trend During First Year of Launch', style={'text-align': 'center'}), 
                html.H4("Select the cryptocurrency: ", style={'text-align': 'left'}),  
                dcc.Checklist(
        id='first-year-crypto-selection',
        options=[{'label': 'Bitcoin', 'value': 'BTC'},{'label': 'Ethereum', 'value': 'ETH'},{'label': 'Dogecoin', 'value': 'Dogecoin'}],
        value=['ETH'],labelStyle={'display': 'inline-block', 'margin-right': '20px'}), 
    html.H4('Spearman correlation was used to analyze how number of transactions and transaction value correlate between different currencies. Number of transactions for BTC and Dogecoin has a moderate tendency to increase or decrease together in the same direction while number of transactions for BTC and ETH and number of transactions for ETH and Dogecoin have a moderate tendency to move in opposite directions.'),
    html.H4('Transaction value for BTC and Dogecoin has a moderate tendency to move together in the same direction while there exists no correlation between transaction value for BTC and ETH and between ETH and Dogecoin.'), 
                html.Div([
                    dcc.Graph(id='first-year-num-transaction'),
                    dcc.Graph(id='first-year-value')
                ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),
    html.H4('The total value for each currency was converted to USD using the average exchange rate in the first year of launch of the respective cryptocurrency.')]),
    html.Div())

@app.callback(
    [Output('transactions-with-contract-plot', 'figure'),
     Output('transactions-contract-plot', 'figure'),
     Output('wallet-address', 'figure'), 
     Output('transaction-value', 'figure')],
    [Input('year-slider', 'value'), 
     Input('crypto-selection', 'value')])

def update_yearly_comparison_content(year_range, selected_cryptos):
    start_year, end_year = year_range
    filtered_data = yearly_stats[(yearly_stats['Year'] >= start_year) & (yearly_stats['Year'] <= end_year) & (yearly_stats['Currency'].isin(selected_cryptos))]
    ethereum_contract = filtered_data[filtered_data['Currency'] == 'ETH']
    fig_with_contract = px.line(
        filtered_data, x='Year', y='Num_transaction', color='Currency',
        title='<b>Total number of transactions</b>',labels={'Year': 'Year', 'Num_transaction': 'Number of transactions'})
    fig_contract = px.line(
        ethereum_contract, x='Year', y='Num_contract_creation', 
        title='<b>Number of contract creations for Ethereum</b>',labels={'Year': 'Year', 'Num_contract_creation': 'Number of contract creations'})
    fig_wallet = px.line(
        filtered_data, x='Year', y='New_wallet_address', color='Currency',
        title='<b>Number of new wallet addresses</b>',labels={'Year': 'Year', 'New_wallet_address': 'Number of new wallet addresses'})
    fig_value = px.line(
        filtered_data,x='Year',y='Total_value (in ETH)',color='Currency',
        title='<b>Transaction Value (in USD)</b>',labels={'Year': 'Year', 'Total_value (in ETH)': 'Transaction Value in USD'})

    tickvals = list(range(start_year, end_year + 1))
    fig_with_contract.update_traces(mode='lines+markers')
    fig_contract.update_traces(mode='lines+markers')
    fig_wallet.update_traces(mode='lines+markers')
    fig_value.update_traces(mode='lines+markers')
    fig_with_contract.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=900)
    fig_contract.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=900)
    fig_wallet.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=900)
    fig_value.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=900)
    return fig_with_contract, fig_contract, fig_wallet, fig_value

@app.callback(
    [ 
     Output('month-transactions-with-contract-plot', 'figure'),
     Output('month-transactions-contract-plot', 'figure'),
     Output('month-wallet-address', 'figure'), 
     Output('month-transaction-value', 'figure')], 
    [Input('start-year', 'value'), 
     Input('start-month', 'value'), 
     Input('end-year', 'value'), 
     Input('end-month', 'value'), 
     Input('crypto-selection', 'value')])

def update_monthly_comparison_content(start_year, start_month, end_year, end_month, selected_crypto): 
    start_date = pd.to_datetime(f"{start_year}-{start_month}", format='%Y-%m')
    end_date = pd.to_datetime(f"{end_year}-{end_month}", format='%Y-%m')
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    date_range_df = pd.DataFrame(date_range, columns=['Date'])
    monthly_stats['Date'] = pd.to_datetime(monthly_stats['Date'])
    filtered_data = pd.merge(date_range_df, monthly_stats, on='Date', how='left')
    filtered_data = filtered_data[filtered_data['Currency'].isin(selected_crypto)]
    ethereum_contract = filtered_data[filtered_data['Currency'] == 'ETH']
    filtered_data['total_amount'] = filtered_data['total_amount'].astype('float64')

    fig_with_contract = px.line(
        filtered_data, x='Date', y='num_transactions', color='Currency',
        title='<b>Total number of transactions</b>', labels={'Year': 'Year', 'Num_transaction': 'Number of transactions'})
    fig_contract = px.line(
        ethereum_contract, x='Date', y='contract_creation', 
        title='<b>Number of contract creations for Ethereum</b>',labels={'Year': 'Year', 'Num_contract_creation': 'Number of contract creations'})
    fig_wallet = px.line(
        filtered_data, x='Date', y='new_wallet_address', color='Currency',
        title='<b>Number of new wallet addresses</b>',labels={'Year': 'Year', 'New_wallet_address': 'Number of new wallet addresses'})
    fig_value = px.line(
        filtered_data,x='Date',y='total_amount',color='Currency',
        title='<b>Transaction Value (in USD)</b>',labels={'Year': 'Year', 'Total_value (in ETH)': 'Transaction Value in USD'})
    date_format = "%Y-%m"
    fig_with_contract.update_traces(mode='lines+markers')
    fig_contract.update_traces(mode='lines+markers')
    fig_wallet.update_traces(mode='lines+markers')
    fig_value.update_traces(mode='lines+markers')
    fig_with_contract.update_layout(xaxis=dict(tickformat=date_format,showgrid=False),yaxis=dict(showgrid=False),width=900)
    fig_contract.update_layout(xaxis=dict(tickformat=date_format,showgrid=False),yaxis=dict(showgrid=False),width=900)
    fig_wallet.update_layout(xaxis=dict(tickformat=date_format,showgrid=False),yaxis=dict(showgrid=False),width=900)
    fig_value.update_layout(xaxis=dict(tickformat=date_format,showgrid=False),yaxis=dict(showgrid=False),width=900)  
    return fig_with_contract, fig_contract, fig_wallet, fig_value

@app.callback(
       [Output('first-year-num-transaction', 'figure'), 
        Output('first-year-value', 'figure')], 
        Input('first-year-crypto-selection', 'value'))

def first_year_plots(first_year_crypto_selection): 
    filtered_data = first_year_trend[first_year_trend['currency'].isin(first_year_crypto_selection)]
    filtered_data['total_value_usd'] = filtered_data['total_value_usd'].astype('float64')
    first_year_num = px.line(
        filtered_data, x='Date', y='num_transactions', color='currency',
        title='<b>Total number of transactions</b>',labels={'Date': 'Day Index', 'num_transactions': 'Number of transactions'})
    first_year_value = px.line(
        filtered_data, x='Date', y='total_value_usd', color='currency',
        title='<b>Transaction value (in USD)</b>',labels={'Date': 'Day Index', 'total_value_usd': 'Transaction value in USD'})
    first_year_num.update_traces(mode='lines+markers')
    first_year_value.update_traces(mode='lines+markers')
    first_year_num.update_layout(xaxis=dict(showgrid=False),yaxis=dict(showgrid=False),width=900)
    first_year_value.update_layout(xaxis=dict(showgrid=False),yaxis=dict(showgrid=False),width=900)
    return first_year_num, first_year_value

@app.callback(
    Output('network-content', 'children'),
    [Input('network-crypto-selection', 'value')]
)

def update_network_content(selected_network_crypto):
    if selected_network_crypto == 'ETH': 
        return html.Div([
        html.H4("Degree Distribution of Wallet Addresses", style={'text-align': 'center'}),
        html.H4("Select the year and month that you are interested in", style={'text-align': 'left'}),
        html.Div([
            html.Div([
                html.Label("Year", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='dist-year',
                    options=[{'label': str(year), 'value': str(year)} for year in range(2015, 2025)], value='2020',
                    style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),
                html.Label("Month", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='dist-month',
                    options=[{'label': str(month), 'value': str(month).zfill(2)} for month in range(1, 13)],value='01',
                    style={'width': '45%', 'display': 'inline-block'})
                    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'})
        ], style={'display': 'inline-block', 'width': '45%'}),
        html.H4("Select the range of in-degrees that you are interested in", style={'text-align': 'left'}),
        dcc.RangeSlider(
            id='indeg-slider',min=1,max=100,step=5,value=[1, 50],marks={i: f'{i}' for i in range(0, 101, 5)},tooltip={"placement": "bottom", "always_visible": True}),
        html.H4("Note: When the maximum selected value is 100, frequencies for in-degree of 100 and greater will be summed up and represented as 'greater than 100'."), 
        dcc.Graph(id='in-deg-dist'),
        html.H4("Select the range of out-degrees that you are interested in", style={'text-align': 'left'}),
        dcc.RangeSlider(
            id='outdeg-slider',min=1,max=100,step=5,value=[1, 50],marks={i: f'{i}' for i in range(0, 101, 5)},tooltip={"placement": "bottom", "always_visible": True}),
        html.H4("Note: When the maximum selected value is 100, frequencies for out-degree of 100 and greater will be summed up and represented as 'greater than 100'."), 
        dcc.Graph(id='out-deg-dist'),
        html.Br(),
        html.H2("Anomaly Detection", style={'text-align': 'center'}), 
        html.H4("Isolation forest and local outlier factor were used to perform anomaly detection given selected features and their performance were compared using silhouette score. The model with the higher silhouette score was used for the eventual classification."),
        html.Img(src='assets/ethereum_anomaly.png'),  
        html.H4('The Kruskal-Wallis test indicates that there is a statistically significant difference in the medians of out-degree, in-degree, total number of transactions, total amount, minimum amount and maximum amount between the anomaly and non-anomaly groups.'), 
        html.H4('The table contains addresses whose total number of transactions is greater than 100,000 from 2015 to 2023. Isolation forest was used for the eventual classification of anomalous addresses.'), 
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='eth-influential-sender-table',
                    columns=[{'name': 'Address', 'id': 'sender'}, {'name': 'Total number of transactions', 'id': 'tot_transactions'}, {'name': 'Out-Degree', 'id': 'out_degree'},
                             {'name': 'In-Degree', 'id': 'in_degree'}, {'name': 'Total amount transacted', 'id': 'total_amount'}, {'name': 'Maximum amount transacted', 'id': 'max_amount'}, 
                             {'name': 'Minimum amount transacted', 'id': 'min_amount'}, {'name': 'Type of Address', 'id': 'type_of_address'}],
                    data=ethereum_anomaly.to_dict('records'), page_size=20, 
                    style_table={'width': '100%', 'margin': 'auto'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}, style_cell={'textAlign': 'center'})
            ], style={'width': '100%', 'margin': 'auto', 'marginBottom': '20px'}),]),])

    elif selected_network_crypto == 'BTC':
        return html.Div([
            html.H2("Degree Distribution of Wallet Addresses", style={'text-align': 'center'}),
            html.H4("Select the year and month that you are interested in", style={'text-align': 'left'}),
            html.Div([
                html.Div([
                    html.Label("Year", style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='btc-dist-year', options=[{'label': str(year), 'value': str(year)} for year in range(2009, 2025)],
                        value='2020', style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),
                    html.Label("Month", style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='btc-dist-month', options=[{'label': str(month), 'value': str(month).zfill(2)} for month in range(1, 13)],
                        value='01', style={'width': '45%', 'display': 'inline-block'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'})
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.H4("Select the range of in-degrees that you are interested in", style={'text-align': 'left'}),
            dcc.Slider(id='btc-indeg-slider', min=1,max=500,step=20,value=100),
            dcc.Graph(id='btc-in-deg-dist'),
            html.H4("Select the range of out-degrees that you are interested in", style={'text-align': 'left'}),
            dcc.Slider(id='btc-outdeg-slider',min=1,max=500,step=20,value=100),
            dcc.Graph(id='btc-out-deg-dist')])
    
    elif selected_network_crypto == 'Iotex':
        return html.Div([
        html.H2("Degree Distribution of Wallet Addresses", style={'text-align': 'center'}),
        html.H4("Select the year and month that you are interested in", style={'text-align': 'left'}),
        html.Div([
            html.Div([
                html.Label("Year", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='iotex-dist-year', options=[{'label': str(year), 'value': str(year)} for year in range(2019, 2023)],
                    value='2021',style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),
                html.Label("Month", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='iotex-dist-month',options=[{'label': str(month).zfill(2), 'value': str(month).zfill(2)} for month in range(1, 13)],
                    value='05',style={'width': '45%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'})
        ], style={'display': 'inline-block', 'width': '45%'}),
        html.Div([
            html.H4("Select the range of in-degrees that you are interested in", style={'text-align': 'left'}),
            dcc.RangeSlider(id='iotex-indeg-slider',min=0,max=100,step=5,value=[1,50],marks={i: f'{i}' for i in range(0, 101, 5)},tooltip={"placement": "bottom", "always_visible": True}),
            html.H4("Note: When the maximum selected value is 100, frequencies for in-degree of 100 and greater will be summed up and represented as 'greater than 100'."), 
            dcc.Graph(id='iotex-in-deg-dist')]),
        html.Div([
            html.H4("Select the range of out-degrees that you are interested in", style={'text-align': 'left'}),
            dcc.RangeSlider(id='iotex-outdeg-slider',min=0,max=100,step=5,value=[1,50],marks={i: f'{i}' for i in range(0, 101, 5)},tooltip={"placement": "bottom", "always_visible": True}),
            html.H4("Note: When the maximum selected value is 100, frequencies for out-degree of 100 and greater will be summed up and represented as 'greater than 100'."), 
            dcc.Graph(id='iotex-out-deg-dist')]),
        html.Br(), 
        html.H2("Anomaly Detection", style={'text-align': 'center'}), 
        html.H4("Isolation forest and local outlier factor were used to perform anomaly detection given selected features and their performance were compared using silhouette score. The model with the higher silhouette score was used for the eventual classification."),
        html.Img(src='assets/iotex_anomaly.png'),  
        html.H4('The Kruskal-Wallis test indicates that there is a statistically significant difference in the medians of out-degree, in-degree, total number of transactions, total amount, minimum amount and maximum amount between the anomaly and non-anomaly groups.'), 
        html.H4('The table contains addresses whose total number of transactions is greater than 10,000 from 2019 to 2022. Local outlier factor was used for the eventual classification of anomalous addresses.'), 
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='iotex-influential-sender-table',
                    columns=[{'name': 'Address', 'id': 'sender'},{'name': 'Total number of transactions', 'id': 'tot_transactions'}, 
                        {'name': 'Out-Degree', 'id': 'out_degree'},{'name': 'In-Degree', 'id': 'in_degree'},{'name': 'Total amount transacted', 'id': 'total_amount'}, 
                        {'name': 'Maximum amount transacted', 'id': 'max_amount'}, {'name': 'Minimum amount transacted', 'id': 'min_amount'}, {'name': 'Type of Address', 'id': 'type_of_address'}],
                    data=iotex_anomaly.to_dict('records'), page_size = 20, 
                    style_table={'width': '100%', 'margin': 'auto'},style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},style_cell={'textAlign': 'center'})
            ], style={'width': '100%', 'margin': 'auto', 'marginBottom': '20px'}),]), 
        html.Hr(), 
        html.Br(),
        html.H2("Subgraph Analysis", style={'text-align': 'center'}),
        html.Br(),
        html.Div([
            html.H4("Type the wallet address"),
            dcc.Input(id='iotex-wallet', type='text', placeholder='Type the wallet address', style={'width': '100%', 'margin': 'auto', 'display': 'block'}),
        ]),
        html.Div([
            html.H4("Select the diameter of the subgraph"),
            dcc.Slider(id='iotex-tracker-slider',min=1,max=5,step=1,value=2)]),
        html.H4("Take note that the wallet address io0000000000000000000000rewardingprotocol is excluded from the subgraph."),
        dcc.Graph(id='iotex-tracker-network'),
        html.Div(id='metrics-output'),
        dcc.Graph(id='iotex-in-degree-distribution-sub'),
        dcc.Graph(id='iotex-out-degree-distribution-sub'),
        html.H4("Top 10 most influential wallet addresses by in and out-degree centralities"),
        html.Div([
            dash_table.DataTable(
                id='iotex-influential-recipient-table-sub',
                columns=[{'name': 'Recipient', 'id': 'recipient'},{'name': 'In-Degree', 'id': 'in_degree'}],
                style_table={'width': '80%', 'margin': 'auto'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},style_cell={'textAlign': 'center'}),
            dash_table.DataTable(
                id='iotex-influential-sender-table-sub',
                columns=[{'name': 'Sender', 'id': 'sender'},{'name': 'Out-Degree', 'id': 'out_degree'}],
                style_table={'width': '80%', 'margin': 'auto'},style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},style_cell={'textAlign': 'center'})
        ], style={'display': 'flex', 'justify-content': 'space-around'}),
        html.H3("Community Detection", style={'text-align': 'center'}), 
        html.H4("Select Community Number"),
dcc.Dropdown(id='iotex-community-dropdown',options=[],  value=None,  placeholder="Select a community"), 
html.Div([
    html.H3("Subgraph for Selected Community", style={'text-align': 'center'}),
    dcc.Graph(id='iotex-community-subgraph')]), 
html.Div(id = 'iotex-community-metrics')])

    elif selected_network_crypto == 'Tezos': 
        return (
           html.Div([
           html.H2("Degree Distribution of Wallet Addresses", style={'text-align': 'center'}),
           html.H4("Select the year and month that you are interested in", style={'text-align': 'left'}),
           html.Div([
            html.Div([
                html.Label("Year", style={'margin-right': '10px'}),
                dcc.Dropdown(id='tezos-dist-year',options=[{'label': str(year), 'value': str(year)} for year in range(2019, 2023)],value='2021',style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),
                html.Label("Month", style={'margin-right': '10px'}),
                dcc.Dropdown(id='tezos-dist-month',options=[{'label': str(month).zfill(2), 'value': str(month).zfill(2)} for month in range(1, 13)],value='05',style={'width': '45%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'})
        ], style={'display': 'inline-block', 'width': '45%'}),
        html.Div([
            html.H4("Select the range of in-degrees that you are interested in", style={'text-align': 'left'}),
            dcc.RangeSlider(id='tezos-indeg-slider',min=0,max=100,step=5,value=[1,50], marks={i: f'{i}' for i in range(0, 101, 5)},tooltip={"placement": "bottom", "always_visible": True}),
            html.H4("Note: When the maximum selected value is 100, frequencies for in-degree of 100 and greater will be summed up and represented as 'greater than 100'."), 
            dcc.Graph(id='tezos-in-deg-dist')]),
        html.Div([
            html.H4("Select the range of out-degrees that you are interested in", style={'text-align': 'left'}),
            dcc.RangeSlider(id='tezos-outdeg-slider',min=0,max=100,step=5,value=[1,50], marks={i: f'{i}' for i in range(0, 101, 5)},tooltip={"placement": "bottom", "always_visible": True}),
            html.H4("Note: When the maximum selected value is 100, frequencies for out-degree of 100 and greater will be summed up and represented as 'greater than 100'."), 
            dcc.Graph(id='tezos-out-deg-dist')]),
        html.Br(), 
        html.H2("Anomaly Detection", style={'text-align': 'center'}), 
        html.H4("Isolation forest and local outlier factor were used to perform anomaly detection given selected features and their performance were compared using silhouette score. The model with the higher silhouette score was used for the eventual classification."),
        html.Img(src='assets/tezos_anomaly.png'),  
        html.H4('The Kruskal-Wallis test indicates that there is a statistically significant difference in the medians of out-degree, in-degree, total number of transactions, total amount, minimum amount and maximum amount between the anomaly and non-anomaly groups.'), 
        html.H4('The table contains addresses whose total number of transactions is greater than 30,000 from 2019 to 2022. Isolation forest was used for the eventual classification of anomalous addresses.'), 
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='tezos-influential-sender-table',
                    columns=[{'name': 'Address', 'id': 'sender'},{'name': 'Total number of transactions', 'id': 'tot_transactions'}, 
                        {'name': 'Out-Degree', 'id': 'out_degree'},{'name': 'In-Degree', 'id': 'in_degree'},
                        {'name': 'Total amount transacted', 'id': 'total_amount'}, {'name': 'Maximum amount transacted', 'id': 'max_amount'}, 
                        {'name': 'Minimum amount transacted', 'id': 'min_amount'}, {'name': 'Type of Address', 'id': 'type_of_address'}],
                    data=tezos_anomaly.to_dict('records'),page_size = 30, style_table={'width': '100%', 'margin': 'auto'},style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},style_cell={'textAlign': 'center'})
            ], style={'display': 'inline-block', 'width': '45%', 'verticalAlign': 'top'})]), 
        html.Hr(), 
        html.Br(), 
        html.H2("Subgraph Analysis", style={'text-align': 'center'}),
        html.Br(),
        html.Div([
            html.H4("Type the wallet address"), dcc.Input(id='tezos-wallet', type='text', placeholder='Type the wallet address', style={'width': '100%', 'margin': 'auto', 'display': 'block'}),]),
        html.Div([html.H4("Select the diameter of the subgraph"),
            dcc.Slider(id='tezos-tracker-slider',min=1,max=5,step=1,value=2)]),
        dcc.Graph(id='tezos-tracker-network'), 
        html.Div(id='tezos-metrics-output'), 
        dcc.Graph(id = 'tezos-in-degree-distribution-sub'), 
        dcc.Graph(id = 'tezos-out-degree-distribution-sub'), 
        html.H4("Top 10 most influential wallet addresses by in and out-degree centralities"), 
        html.Div([
            dash_table.DataTable(
                id='tezos-influential-recipient-table-sub',
                columns=[{'name': 'Recipient', 'id': 'recipient'},{'name': 'In-Degree', 'id': 'in_degree'}],
                style_table={'width': '80%', 'margin': 'auto'},style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},style_cell={'textAlign': 'center'}),
            dash_table.DataTable(
                id='tezos-influential-sender-table-sub',
                columns=[{'name': 'Sender', 'id': 'sender'},{'name': 'Out-Degree', 'id': 'out_degree'}],
                style_table={'width': '80%', 'margin': 'auto'},style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},style_cell={'textAlign': 'center'})
        ], style={'display': 'flex', 'justify-content': 'space-around'}),
    html.H3("Community Detection", style={'text-align': 'center'}), 
        html.H4("Select Community Number"),
dcc.Dropdown(id='tezos-community-dropdown',options=[],  value=None,  placeholder="Select a community"), 
html.Div([
    html.H3("Subgraph for Selected Community", style={'text-align': 'center'}),
    dcc.Graph(id='tezos-community-subgraph')]), 
html.Div(id = 'tezos-community-metrics')]), 
   html.Div()) 

@app.callback([
     Output('in-deg-dist', 'figure')], 
    [Input('dist-year', 'value'),
     Input('dist-month', 'value'), 
     Input('indeg-slider', 'value')])

### Ethereum in-degree distribution 
def in_degree_distribution(dist_year, dist_month, indeg_slider): 
    dist_date = pd.to_datetime(f"{dist_year}-{dist_month}", format='%Y-%m')
    eth_filtered_in_deg = in_deg_dist[in_deg_dist['Date'] == dist_date]
    min_in_degree, max_in_degree = indeg_slider
    if max_in_degree == 100:
        in_range_data = eth_filtered_in_deg[(eth_filtered_in_deg['in_degree'] >= min_in_degree) & (eth_filtered_in_deg['in_degree'] < 100)]
        greater_than_100_data = eth_filtered_in_deg[eth_filtered_in_deg['in_degree'] >= 100]
        greater_than_100_count = greater_than_100_data['frequency'].sum()
        if not greater_than_100_data.empty:
            new_row = pd.DataFrame({'in_degree': ['100 and greater'], 'frequency': [greater_than_100_count]})
            in_range_data = pd.concat([in_range_data, new_row], ignore_index=True)
        eth_filtered_in_deg = in_range_data
    else:
        eth_filtered_in_deg = eth_filtered_in_deg[eth_filtered_in_deg['in_degree'].between(min_in_degree, max_in_degree)]
    eth_filtered_in_deg['in_degree'] = eth_filtered_in_deg['in_degree'].astype(str)
    eth_in_deg_line = px.line(eth_filtered_in_deg, x='in_degree', y='frequency', title='<b>In-degree distribution</b>', labels={'Date': 'Date', 'in_degree': 'in-degree'})
    tickvals = [str(t) for t in range(min_in_degree, max_in_degree + 1, 5)]
    if max_in_degree == 100:
        tickvals.append('100 and greater')
    eth_in_deg_line.update_traces(mode='markers+lines')
    eth_in_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [eth_in_deg_line]

@app.callback([
     Output('out-deg-dist', 'figure')], 
    [Input('dist-year', 'value'),
     Input('dist-month', 'value'), 
     Input('outdeg-slider', 'value')])

### ETH out-degree distribution 
def out_degree_distribution(dist_year, dist_month, outdeg_slider): 
    dist_date = pd.to_datetime(f"{dist_year}-{dist_month}", format='%Y-%m')
    eth_filtered_out_deg = out_deg_dist[out_deg_dist['Date'] == dist_date]
    min_out_degree, max_out_degree = outdeg_slider
    if max_out_degree == 100:
        in_range_data = eth_filtered_out_deg[(eth_filtered_out_deg['out_degree'] >= min_out_degree) & (eth_filtered_out_deg['out_degree'] < 100)]
        greater_than_100_data = eth_filtered_out_deg[eth_filtered_out_deg['out_degree'] >= 100]
        greater_than_100_count = greater_than_100_data['frequency'].sum()
        if not greater_than_100_data.empty:
            new_row = pd.DataFrame({'out_degree': ['100 and greater'], 'frequency': [greater_than_100_count]})
            in_range_data = pd.concat([in_range_data, new_row], ignore_index=True)
        eth_filtered_out_deg = in_range_data
    else:
        eth_filtered_out_deg = eth_filtered_out_deg[eth_filtered_out_deg['out_degree'].between(min_out_degree, max_out_degree)]
    eth_filtered_out_deg['out_degree'] = eth_filtered_out_deg['out_degree'].astype(str)
    eth_out_deg_line = px.line(eth_filtered_out_deg, x='out_degree', y='frequency', title='<b>Out-degree distribution</b>', labels={'Date': 'Date', 'out_degree': 'out-degree'})
    tickvals = [str(t) for t in range(min_out_degree, max_out_degree + 1, 5)]
    if max_out_degree == 100:
        tickvals.append('100 and greater')
    eth_out_deg_line.update_traces(mode='markers+lines')
    eth_out_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [eth_out_deg_line]

@app.callback([Output('btc-in-deg-dist', 'figure')], 
    [Input('btc-dist-year', 'value'),
     Input('btc-dist-month', 'value'), 
     Input('btc-indeg-slider', 'value')])

### BTC in-degree distribution 
def btc_in_degree_distribution(btc_dist_year, btc_dist_month, btc_indeg_slider): 
    dist_date = pd.to_datetime(f"{btc_dist_year}-{btc_dist_month}", format='%Y-%m')
    btc_filtered_in_deg = btc_in_deg_dist[btc_in_deg_dist['Date'] == dist_date]
    btc_filtered_in_deg = btc_filtered_in_deg[btc_filtered_in_deg['in_degree'] <= btc_indeg_slider]
    btc_in_deg_line = px.line(btc_filtered_in_deg, x='in_degree', y='frequency', title='<b>In-degree distribution</b>', labels={'Date': 'Date', 'in_degree': 'in-degree'}) 
    tickvals = list(range(0, btc_indeg_slider + 1, 50))
    btc_in_deg_line.update_traces(mode='markers')
    btc_in_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [btc_in_deg_line]
    
@app.callback([
     Output('btc-out-deg-dist', 'figure')], 
    [Input('btc-dist-year', 'value'),
     Input('btc-dist-month', 'value'), 
     Input('btc-outdeg-slider', 'value')])

### BTC out-degree distribution
def btc_out_degree_distribution(btc_dist_year, btc_dist_month, btc_outdeg_slider): 
    dist_date = pd.to_datetime(f"{btc_dist_year}-{btc_dist_month}", format='%Y-%m')
    filtered_out_deg = btc_out_deg_dist[btc_out_deg_dist['Date'] == dist_date]
    filtered_out_deg = filtered_out_deg[filtered_out_deg['out_degree'] <= btc_outdeg_slider]
    out_deg_line = px.line(filtered_out_deg, x='out_degree', y='frequency', title='<b>Out-degree distribution</b>', labels={'Date': 'Date', 'out_degree': 'Out-degree'}) 
    tickvals = list(range(0, btc_outdeg_slider + 1, 50))
    out_deg_line.update_traces(mode='markers')
    out_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [out_deg_line]

@app.callback([Output('iotex-in-deg-dist', 'figure')], 
    [Input('iotex-dist-year', 'value'),
     Input('iotex-dist-month', 'value'), 
     Input('iotex-indeg-slider', 'value')])

def iotex_in_degree_distribution(iotex_dist_year, iotex_dist_month, iotex_indeg_slider): 
    dist_date = pd.to_datetime(f"{iotex_dist_year}-{iotex_dist_month}", format='%Y-%m')
    iotex_filtered_in_deg = iotex_in_deg_dist[iotex_in_deg_dist['Date'] == dist_date]
    min_in_degree, max_in_degree = iotex_indeg_slider
    if max_in_degree == 100:
        in_range_data = iotex_filtered_in_deg[(iotex_filtered_in_deg['in_degree'] >= min_in_degree) & (iotex_filtered_in_deg['in_degree'] < 100)]
        greater_than_100_data = iotex_filtered_in_deg[iotex_filtered_in_deg['in_degree'] >= 100]
        greater_than_100_count = greater_than_100_data['frequency'].sum()
        if not greater_than_100_data.empty:
            new_row = pd.DataFrame({'in_degree': ['100 and greater'], 'frequency': [greater_than_100_count]})
            in_range_data = pd.concat([in_range_data, new_row], ignore_index=True)
        iotex_filtered_in_deg = in_range_data
    else:
        iotex_filtered_in_deg = iotex_filtered_in_deg[iotex_filtered_in_deg['in_degree'].between(min_in_degree, max_in_degree)]
    iotex_filtered_in_deg['in_degree'] = iotex_filtered_in_deg['in_degree'].astype(str)
    iotex_in_deg_line = px.line(iotex_filtered_in_deg, x='in_degree', y='frequency', title='<b>In-degree distribution</b>', labels={'Date': 'Date', 'in_degree': 'in-degree'})
    tickvals = [str(t) for t in range(min_in_degree, max_in_degree + 1, 5)]
    if max_in_degree == 100:
        tickvals.append('100 and greater')
    iotex_in_deg_line.update_traces(mode='markers+lines')
    iotex_in_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [iotex_in_deg_line]
    
@app.callback([
     Output('iotex-out-deg-dist', 'figure')], 
    [Input('iotex-dist-year', 'value'),
     Input('iotex-dist-month', 'value'), 
     Input('iotex-outdeg-slider', 'value')])

### Iotex out-degree distribution 
def update_iotex_out_deg_distribution(iotex_dist_year, iotex_dist_month, iotex_outdeg_slider): 
    dist_date = pd.to_datetime(f"{iotex_dist_year}-{iotex_dist_month}", format='%Y-%m')
    iotex_filtered_out_deg = iotex_out_deg_dist[iotex_out_deg_dist['Date'] == dist_date]
    min_out_degree, max_out_degree = iotex_outdeg_slider
    if max_out_degree == 100:
        in_range_data = iotex_filtered_out_deg[(iotex_filtered_out_deg['out_degree'] >= min_out_degree) & (iotex_filtered_out_deg['out_degree'] < 100)]
        greater_than_100_data = iotex_filtered_out_deg[iotex_filtered_out_deg['out_degree'] >= 100]
        greater_than_100_count = greater_than_100_data['frequency'].sum()
        if not greater_than_100_data.empty:
            new_row = pd.DataFrame({'out_degree': ['100 and greater'], 'frequency': [greater_than_100_count]})
            in_range_data = pd.concat([in_range_data, new_row], ignore_index=True)
        iotex_filtered_out_deg = in_range_data
    else:
        iotex_filtered_out_deg = iotex_filtered_out_deg[iotex_filtered_out_deg['out_degree'].between(min_out_degree, max_out_degree)]
    iotex_filtered_out_deg['out_degree'] = iotex_filtered_out_deg['out_degree'].astype(str)
    iotex_out_deg_line = px.line(iotex_filtered_out_deg, x='out_degree', y='frequency', title='<b>Out-degree distribution</b>',labels={'Date': 'Date', 'out_degree': 'out-degree'})
    tickvals = [str(t) for t in range(min_out_degree, max_out_degree + 1, 5)]
    if max_out_degree == 100:
        tickvals.append('100 and greater')
    iotex_out_deg_line.update_traces(mode='markers+lines')
    iotex_out_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [iotex_out_deg_line]

@app.callback(
    [
        Output('iotex-tracker-network', 'figure'),
        Output('metrics-output', 'children'),
        Output('iotex-in-degree-distribution-sub', 'figure'),
        Output('iotex-out-degree-distribution-sub', 'figure'),
        Output('iotex-influential-recipient-table-sub', 'data'),
        Output('iotex-influential-sender-table-sub', 'data'),
        Output('iotex-community-dropdown', 'options'),
        Output('iotex-community-dropdown', 'value'),
        Output('iotex-community-subgraph', 'figure'), 
        Output('iotex-community-metrics', 'children')
    ],
    [
        Input('iotex-wallet', 'value'),
        Input('iotex-tracker-slider', 'value'), 
        Input('iotex-community-dropdown', 'value')
    ])

### Iotex subgraph extraction
def update_iotex_network(wallet_address, iotex_tracker_slider, selected_community):
    relevant_addresses = set([wallet_address])
    G = ig.Graph(directed=True)
    current_addresses = set([wallet_address]) 
    exclude_address = 'io0000000000000000000000rewardingprotocol'
    for depth in range(iotex_tracker_slider):  
        new_addresses = set()
        for addr in current_addresses:
            filtered_df = iotex_2022_transaction.filter(((pl.col('sender') == addr) | (pl.col('recipient') == addr)) &(pl.col('sender') != exclude_address) &(pl.col('recipient') != exclude_address))
            new_addresses.update(filtered_df['sender'].to_list())
            new_addresses.update(filtered_df['recipient'].to_list())
        relevant_addresses.update(new_addresses)
        current_addresses = new_addresses
        if not new_addresses:
            break
    G.add_vertices(list(relevant_addresses))
    for row in iotex_2022_transaction.iter_rows(named=True):
        from_address = row['sender']
        to_address = row['recipient']
        if from_address in relevant_addresses and to_address in relevant_addresses:
            G.add_edge(from_address, to_address, num_transactions=row['num_transactions'], total_amount=row['tot_amount'])
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(G.vs['name'])
    for edge in G.es:
        source = G.vs[edge.source]['name']
        target = G.vs[edge.target]['name']
        nx_graph.add_edge(source, target)
    community_relevant_addresses = set(relevant_addresses)  
    for addr in relevant_addresses:
        expanded_df = iotex_2022_transaction.filter(((pl.col('sender') == addr) | (pl.col('recipient') == addr)) &(pl.col('sender') != exclude_address) &(pl.col('recipient') != exclude_address))
        community_relevant_addresses.update(expanded_df['sender'].to_list())
        community_relevant_addresses.update(expanded_df['recipient'].to_list())
    community_subgraph = G.subgraph([v.index for v in G.vs if v['name'] in community_relevant_addresses])
    nx_community_subgraph = nx.Graph()
    nx_community_subgraph.add_nodes_from(community_subgraph.vs['name'])
    for edge in community_subgraph.es:
        source = community_subgraph.vs[edge.source]['name']
        target = community_subgraph.vs[edge.target]['name']
        nx_community_subgraph.add_edge(source, target)
    louvain_partition = community_louvain.best_partition(nx_community_subgraph)
    louvain_communities = [louvain_partition[node] for node in nx_community_subgraph.nodes()]
    louvain_modularity = community_louvain.modularity(louvain_partition, nx_community_subgraph)
    layout = G.layout('fr')  
    layout_sub = community_subgraph.layout('fr')
    traceRecode = []
    for edge in G.es:
        x0, y0 = layout[edge.source]
        x1, y1 = layout[edge.target]
        edge_trace = go.Scatter(x=[x0, x1, None],y=[y0, y1, None],line=dict(width=1, color='#888'),hoverinfo='text',mode='lines',text=f"Transactions: {edge['num_transactions']}<br>Value: {edge['total_amount']}")
        traceRecode.append(edge_trace)
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for vertex in G.vs:
        x, y = layout[vertex.index]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Address: {vertex['name']}")
        node_color.append(louvain_communities[vertex.index])
    node_trace = go.Scatter(x=node_x,y=node_y,text=node_text,mode='markers',hoverinfo='text',marker=dict(color=node_color,colorscale='Plotly3',size=10))
    traceRecode.append(node_trace)
    middle_x = []
    middle_y = []
    middle_hover_text = []
    for edge in G.es:
        x0, y0 = layout[edge.source]
        x1, y1 = layout[edge.target]
        middle_x.append((x0 + x1) / 2)
        middle_y.append((y0 + y1) / 2)
        middle_hover_text.append(f"Transactions: {edge['num_transactions']}<br>Value: {edge['total_amount']}")
    middle_trace = go.Scatter(x=middle_x,y=middle_y,text=middle_hover_text,mode='markers',hoverinfo='text',
        marker=dict(size=20,color='LightSkyBlue',opacity=0))
    traceRecode.append(middle_trace)
    figure_data = traceRecode
    figure_layout = {'hovermode': 'closest','margin': {'b': 20,'l': 5,'r': 5,'t': 40},'showlegend': False,
        'xaxis': {'showgrid': False,'showticklabels': False,'zeroline': False},
        'yaxis': {'showgrid': False,'showticklabels': False,'zeroline': False}}
    figure = {'data': figure_data, 'layout': figure_layout}
    community_sizes = [sum(1 for x in louvain_communities if x == community) for community in set(louvain_communities)]
    num_edges_in_community = []
    density_in_community = []
    for community in set(louvain_communities):
        community_nodes = [v for v in range(len(louvain_communities)) if louvain_communities[v] == community]
        subgraph = G.subgraph(community_nodes)
        num_edges_in_community.append(subgraph.ecount())
        density_in_community.append(subgraph.density())
    in_degrees = G.degree(mode="in")
    out_degrees = G.degree(mode="out")
    in_degree_counts = {}
    out_degree_counts = {}
    for degree in in_degrees:
        if degree in in_degree_counts:
            in_degree_counts[degree] += 1
        else:
            in_degree_counts[degree] = 1
    for degree in out_degrees:
        if degree in out_degree_counts:
            out_degree_counts[degree] += 1
        else:
            out_degree_counts[degree] = 1
    in_degree_figure = {
        'data': [go.Bar(x=[str(k) for k in in_degree_counts.keys() if k <= 100],y=[v for k, v in in_degree_counts.items() if k <= 100])],
        'layout': go.Layout(title='In-Degree Distribution (Nodes with in-degree of 100 and lower)',xaxis=dict(title='In-Degree', range=[0, 100], showgrid=False, zeroline=False),yaxis=dict(title='Number of Nodes', showgrid=False, zeroline=False))}
    out_degree_figure = {
        'data': [go.Bar(x=[str(k) for k in out_degree_counts.keys() if k <= 100],y=[v for k, v in out_degree_counts.items() if k <= 100])],
        'layout': go.Layout(title='Out-Degree Distribution (Nodes with out-degree of 100 and lower)',xaxis=dict(title='Out-Degree', range=[0, 100], showgrid=False, zeroline=False),yaxis=dict(title='Number of Nodes', showgrid=False, zeroline=False))}
    in_degree_sorted = sorted([(v, in_degrees[v]) for v in range(G.vcount())], key=lambda x: x[1], reverse=True)[:10]
    out_degree_sorted = sorted([(v, out_degrees[v]) for v in range(G.vcount())], key=lambda x: x[1], reverse=True)[:10]
    influential_recipient_data = [{'recipient': G.vs[v]['name'],'in_degree': deg,'type_of_address': 'Recipient'} for v, deg in in_degree_sorted]
    influential_sender_data = [{'sender': G.vs[v]['name'],'out_degree': deg,'type_of_address': 'Sender'} for v, deg in out_degree_sorted]

    num_nodes = G.vcount()
    num_edges = G.ecount()
    avg_in_degree = round(sum(G.degree(mode="in")) / num_nodes, 2) if num_nodes > 0 else 0
    avg_out_degree = round(sum(G.degree(mode="out")) / num_nodes, 2) if num_nodes > 0 else 0
    density = round(G.density(), 5)
    num_connected_components = len(G.components(mode="weak"))
    num_communities = len(set(louvain_communities))

    metrics_output = html.Div([
        html.Div(f"Number of nodes: {num_nodes}"),
        html.Div(f"Number of edges: {num_edges}"),
        html.Div(f"Average in-degree / out-degree: {avg_in_degree}"), 
        html.Div(f"Density: {density}"),
        html.Div(f"Number of connected components: {num_connected_components}"), 
        html.Div(f"Number of communities: {num_communities}")])
    community_options = [{'label': f'Community {i}', 'value': i} for i in sorted(set(louvain_communities))]
    if selected_community is None:
        if community_options:
            selected_community = community_options[0]['value']
        else:
            selected_community = None  
    community_metrics_output = html.Div([html.Div(f"No metrics available for the selected community.")])
    if selected_community is None:
        community_subgraph_figure = {'data': [],'layout': {'title': 'No Data Available','xaxis': {'visible': False},'yaxis': {'visible': False},'annotations': [{'text': 'No subgraph to display','xref': 'paper','yref': 'paper','showarrow': False,'font': {'size': 28}}]}}
    else:
        community_nodes = [v for v in range(len(louvain_communities)) if louvain_communities[v] == selected_community]
        if not community_nodes:
            community_subgraph_figure = {'data': [],'layout': {'title': f'No nodes available for Community {selected_community}','xaxis': {'visible': False},'yaxis': {'visible': False},'annotations': [{'text': f'No data for Community {selected_community}','xref': 'paper','yref': 'paper','showarrow': False,'font': {'size': 28}}]}}
        else:
            subgraph = G.subgraph(community_nodes)
            layout_sub = subgraph.layout('fr') 
            subgraph_traceRecode = []
            for edge in subgraph.es:
                x0, y0 = layout_sub[edge.source]
                x1, y1 = layout_sub[edge.target]
                subgraph_edge_trace = go.Scatter(x=[x0, x1, None],y=[y0, y1, None],line=dict(width=1, color='#888'),hoverinfo='text',mode='lines',showlegend=False, text=f"Transactions: {edge['num_transactions']}<br>Value: {edge['total_amount']}")
                subgraph_traceRecode.append(subgraph_edge_trace)
            node_x_sub = []
            node_y_sub = []
            node_text_sub = []
            node_color_sub = []
            for vertex in subgraph.vs:
                x, y = layout_sub[vertex.index]
                node_x_sub.append(x)
                node_y_sub.append(y)
                node_text_sub.append(f"Address: {vertex['name']}")
                node_color_sub.append(louvain_communities[vertex.index])
            node_trace_sub = go.Scatter(x=node_x_sub,y=node_y_sub,text=node_text_sub,mode='markers',hoverinfo='text',marker=dict(color='blue', size=10))
            subgraph_traceRecode.append(node_trace_sub)
            community_subgraph_figure = {'data': subgraph_traceRecode,
                'layout': {'title': f'Subgraph for Community {selected_community}','xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},'hovermode': 'closest','margin': {'l': 40, 'r': 40, 'b': 40, 't': 40}}}
            num_nodes = subgraph.vcount()
            num_edges = subgraph.ecount()
            avg_in_degree = round(sum(subgraph.degree(mode="in")) / num_nodes, 2) if num_nodes > 0 else 0
            avg_out_degree = round(sum(subgraph.degree(mode="out")) / num_nodes, 2) if num_nodes > 0 else 0
            density = round(subgraph.density(), 5)
            num_connected_components = len(subgraph.components(mode="weak"))
            community_metrics_output = html.Div([
                html.Div(f"Number of nodes: {num_nodes}"),
                html.Div(f"Number of edges: {num_edges}"),
                html.Div(f"Average in-degree: {avg_in_degree}"),
                html.Div(f"Average out-degree: {avg_out_degree}"),
                html.Div(f"Density: {density}"),
                html.Div(f"Number of connected components: {num_connected_components}")])
    return (figure, metrics_output, in_degree_figure, 
            out_degree_figure, influential_recipient_data, influential_sender_data, 
            community_options, selected_community, community_subgraph_figure, community_metrics_output)

@app.callback([Output('tezos-in-deg-dist', 'figure')], 
    [Input('tezos-dist-year', 'value'),
     Input('tezos-dist-month', 'value'), 
     Input('tezos-indeg-slider', 'value')])

### Tezos in-degree distribution 
def tezos_in_degree_distribution(tezos_dist_year, tezos_dist_month, tezos_indeg_slider): 
    dist_date = pd.to_datetime(f"{tezos_dist_year}-{tezos_dist_month}", format='%Y-%m')
    tezos_filtered_in_deg = tezos_in_deg_dist[tezos_in_deg_dist['Date'] == dist_date]
    min_in_degree, max_in_degree = tezos_indeg_slider
    if max_in_degree == 100:
        in_range_data = tezos_filtered_in_deg[(tezos_filtered_in_deg['in_degree'] >= min_in_degree) & (tezos_filtered_in_deg['in_degree'] < 100)]
        greater_than_100_data = tezos_filtered_in_deg[tezos_filtered_in_deg['in_degree'] >= 100]
        greater_than_100_count = greater_than_100_data['frequency'].sum()
        if not greater_than_100_data.empty:
            new_row = pd.DataFrame({'in_degree': ['greater than 100'], 'frequency': [greater_than_100_count]})
            in_range_data = pd.concat([in_range_data, new_row], ignore_index=True)
        tezos_filtered_in_deg = in_range_data
    else:
        tezos_filtered_in_deg = tezos_filtered_in_deg[tezos_filtered_in_deg['in_degree'].between(min_in_degree, max_in_degree)]
    tezos_filtered_in_deg['in_degree'] = tezos_filtered_in_deg['in_degree'].astype(str)
    tezos_in_deg_line = px.line(tezos_filtered_in_deg, x='in_degree', y='frequency', title='<b>In-degree distribution</b>', labels={'Date': 'Date', 'in_degree': 'in-degree'})
    tickvals = [str(t) for t in range(min_in_degree, max_in_degree + 1, 5)]
    if max_in_degree == 100:
        tickvals.append('greater than 100')
    tezos_in_deg_line.update_traces(mode='markers+lines')
    tezos_in_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [tezos_in_deg_line]

@app.callback(
    [Output('tezos-out-deg-dist', 'figure')],
    [Input('tezos-dist-year', 'value'),
     Input('tezos-dist-month', 'value'),
     Input('tezos-outdeg-slider', 'value')])

### Tezos out-degree distribution
def tezos_out_degree_distribution(tezos_dist_year, tezos_dist_month, tezos_outdeg_slider): 
    dist_date = pd.to_datetime(f"{tezos_dist_year}-{tezos_dist_month}", format='%Y-%m')
    tezos_filtered_out_deg = tezos_out_deg_dist[tezos_out_deg_dist['Date'] == dist_date]
    min_out_degree, max_out_degree = tezos_outdeg_slider
    if max_out_degree == 100:
        in_range_data = tezos_filtered_out_deg[(tezos_filtered_out_deg['out_degree'] >= min_out_degree) & (tezos_filtered_out_deg['out_degree'] < 100)]
        greater_than_100_data = tezos_filtered_out_deg[tezos_filtered_out_deg['out_degree'] >= 100]
        greater_than_100_count = greater_than_100_data['frequency'].sum()
        if not greater_than_100_data.empty:
            new_row = pd.DataFrame({'out_degree': ['greater than 100'], 'frequency': [greater_than_100_count]})
            in_range_data = pd.concat([in_range_data, new_row], ignore_index=True)
        tezos_filtered_out_deg = in_range_data
    else:
        tezos_filtered_out_deg = tezos_filtered_out_deg[tezos_filtered_out_deg['out_degree'].between(min_out_degree, max_out_degree)]
    tezos_filtered_out_deg['out_degree'] = tezos_filtered_out_deg['out_degree'].astype(str)
    tezos_out_deg_line = px.line(tezos_filtered_out_deg, x='out_degree', y='frequency', title='<b>Out-degree distribution</b>', labels={'Date': 'Date', 'out_degree': 'out-degree'})
    tickvals = [str(t) for t in range(min_out_degree, max_out_degree + 1, 5)]
    if max_out_degree == 100:
        tickvals.append('greater than 100')
    tezos_out_deg_line.update_traces(mode='markers+lines')
    tezos_out_deg_line.update_layout(xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=tickvals, showgrid=False),yaxis=dict(showgrid=False),width=1800)
    return [tezos_out_deg_line]

@app.callback(
    [
        Output('tezos-tracker-network', 'figure'),
        Output('tezos-metrics-output', 'children'),
        Output('tezos-in-degree-distribution-sub', 'figure'),
        Output('tezos-out-degree-distribution-sub', 'figure'),
        Output('tezos-influential-recipient-table-sub', 'data'),
        Output('tezos-influential-sender-table-sub', 'data'),
        Output('tezos-community-dropdown', 'options'),
        Output('tezos-community-dropdown', 'value'),
        Output('tezos-community-subgraph', 'figure'),
        Output('tezos-community-metrics', 'children')],
    [
        Input('tezos-wallet', 'value'),
        Input('tezos-tracker-slider', 'value'),
        Input('tezos-community-dropdown', 'value')])

### Tezos subgraph extraction 
def update_tezos_network(wallet_address, tezos_tracker_slider, selected_community):
    relevant_addresses = set([wallet_address])
    G = ig.Graph(directed=True)
    current_addresses = set([wallet_address])
    exclude_address = 'tz000000000000000000000000000000000000'  
    for depth in range(tezos_tracker_slider):
        new_addresses = set()
        for addr in current_addresses:
            filtered_df = tezos_2022_transaction.filter(((pl.col('source') == addr) | (pl.col('destination') == addr)) &(pl.col('source') != exclude_address) &(pl.col('destination') != exclude_address))
            new_addresses.update(filtered_df['source'].to_list())
            new_addresses.update(filtered_df['destination'].to_list())
        relevant_addresses.update(new_addresses)
        current_addresses = new_addresses
        if not new_addresses:
            break
    G.add_vertices(list(relevant_addresses))
    for row in tezos_2022_transaction.iter_rows(named=True):
        from_address = row['source']
        to_address = row['destination']
        if from_address in relevant_addresses and to_address in relevant_addresses:
            G.add_edge(from_address, to_address, num_transactions=row['num_transaction'])
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(G.vs['name'])
    for edge in G.es:
        source = G.vs[edge.source]['name']
        target = G.vs[edge.target]['name']
        nx_graph.add_edge(source, target)
    louvain_partition = community_louvain.best_partition(nx_graph)
    louvain_communities = [louvain_partition[node] for node in nx_graph.nodes()]
    louvain_modularity = community_louvain.modularity(louvain_partition, nx_graph)
    community_relevant_addresses = set(relevant_addresses)  
    for addr in relevant_addresses:
        expanded_df = tezos_2022_transaction.filter(((pl.col('source') == addr) | (pl.col('destination') == addr)) &(pl.col('source') != exclude_address) &(pl.col('destination') != exclude_address))
        community_relevant_addresses.update(expanded_df['source'].to_list())
        community_relevant_addresses.update(expanded_df['destination'].to_list())
    community_subgraph = G.subgraph([v.index for v in G.vs if v['name'] in community_relevant_addresses])
    nx_community_subgraph = nx.Graph()
    nx_community_subgraph.add_nodes_from(community_subgraph.vs['name'])
    for edge in community_subgraph.es:
        source = community_subgraph.vs[edge.source]['name']
        target = community_subgraph.vs[edge.target]['name']
        nx_community_subgraph.add_edge(source, target)
    louvain_partition = community_louvain.best_partition(nx_community_subgraph)
    louvain_communities = [louvain_partition[node] for node in nx_community_subgraph.nodes()]
    layout = G.layout('fr')  
    layout_sub = community_subgraph.layout('fr')
    traceRecode = []
    for edge in G.es:
        x0, y0 = layout[edge.source]
        x1, y1 = layout[edge.target]
        edge_trace = go.Scatter(x=[x0, x1, None],y=[y0, y1, None],line=dict(width=1, color='#888'),hoverinfo='text',mode='lines',text=f"Transactions: {edge['num_transactions']}")
        traceRecode.append(edge_trace)
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for vertex in G.vs:
        x, y = layout[vertex.index]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Address: {vertex['name']}")
        node_color.append(louvain_communities[vertex.index])
    node_trace = go.Scatter(x=node_x,y=node_y,text=node_text,mode='markers',hoverinfo='text',marker=dict(color=node_color,colorscale='Plotly3',size=10))
    traceRecode.append(node_trace)
    middle_x = []
    middle_y = []
    middle_hover_text = []
    for edge in G.es:
        x0, y0 = layout[edge.source]
        x1, y1 = layout[edge.target]
        middle_x.append((x0 + x1) / 2)
        middle_y.append((y0 + y1) / 2)
        middle_hover_text.append(f"Transactions: {edge['num_transactions']}")
    middle_trace = go.Scatter(x=middle_x,y=middle_y,text=middle_hover_text,mode='markers',hoverinfo='text',marker=dict(size=20,color='LightSkyBlue',opacity=0))
    traceRecode.append(middle_trace)
    figure_data = traceRecode
    figure_layout = {'hovermode': 'closest','margin': {'b': 20,'l': 5,'r': 5,'t': 40},'showlegend': False,'xaxis': {'showgrid': False,'showticklabels': False,'zeroline': False},
        'yaxis': {'showgrid': False,'showticklabels': False,'zeroline': False}}
    figure = {'data': figure_data, 'layout': figure_layout}
    community_sizes = [sum(1 for x in louvain_communities if x == community) for community in set(louvain_communities)]
    num_edges_in_community = []
    density_in_community = []
    for community in set(louvain_communities):
        community_nodes = [v for v in range(len(louvain_communities)) if louvain_communities[v] == community]
        subgraph = G.subgraph(community_nodes)
        num_edges_in_community.append(subgraph.ecount())
        density_in_community.append(subgraph.density())
    in_degrees = G.degree(mode="in")
    out_degrees = G.degree(mode="out")
    in_degree_counts = {}
    out_degree_counts = {}
    for degree in in_degrees:
        in_degree_counts[degree] = in_degree_counts.get(degree, 0) + 1
    for degree in out_degrees:
        out_degree_counts[degree] = out_degree_counts.get(degree, 0) + 1
    in_degree_sorted = sorted([(v, in_degrees[v]) for v in range(G.vcount())], key=lambda x: x[1], reverse=True)[:10]
    out_degree_sorted = sorted([(v, out_degrees[v]) for v in range(G.vcount())], key=lambda x: x[1], reverse=True)[:10]
    influential_recipient_data = [{'recipient': G.vs[v]['name'],'in_degree': deg, 'type_of_address': 'Recipient'} for v, deg in in_degree_sorted]
    influential_sender_data = [{'sender': G.vs[v]['name'],'out_degree': deg,'type_of_address': 'Sender'} for v, deg in out_degree_sorted]
    in_degree_figure = {'data': [go.Bar(x=[str(k) for k in in_degree_counts.keys() if k <= 100],y=[v for k, v in in_degree_counts.items() if k <= 100])],
        'layout': go.Layout(title='In-Degree Distribution (Nodes with in-degree of 100 and lower)',xaxis=dict(title='In-Degree', range=[0, 100], showgrid=False, zeroline=False),yaxis=dict(title='Number of Nodes', showgrid=False, zeroline=False))}
    out_degree_figure = {'data': [go.Bar(x=[str(k) for k in out_degree_counts.keys() if k <= 100],y=[v for k, v in out_degree_counts.items() if k <= 100])],
        'layout': go.Layout(title='Out-Degree Distribution (Nodes with out-degree of 100 and lower)',xaxis=dict(title='Out-Degree', range=[0, 100], showgrid=False, zeroline=False),yaxis=dict(title='Number of Nodes', showgrid=False, zeroline=False))}
    if selected_community is None:
        community_subgraph_figure = {'data': [],'layout': {'title': 'No Data Available','xaxis': {'visible': False},'yaxis': {'visible': False},
                'annotations': [{'text': 'No subgraph to display','xref': 'paper','yref': 'paper','showarrow': False,'font': {'size': 28}}]}}
        community_metrics_output = html.Div([html.Div(f"No community selected")])
    else:
        community_nodes = [v for v in range(len(louvain_communities)) if louvain_communities[v] == selected_community]
        if not community_nodes:
            community_subgraph_figure = {'data': [],'layout': {'title': f'No nodes available for Community {selected_community}','xaxis': {'visible': False},'yaxis': {'visible': False},
                    'annotations': [{'text': f'No data for Community {selected_community}','xref': 'paper','yref': 'paper','showarrow': False,'font': {'size': 28}}]}}
        else:
            subgraph = G.subgraph(community_nodes)
            layout_sub = subgraph.layout('fr')  
            subgraph_traceRecode = []
            for edge in subgraph.es:
                x0, y0 = layout_sub[edge.source]
                x1, y1 = layout_sub[edge.target]
                subgraph_edge_trace = go.Scatter(x=[x0, x1, None],y=[y0, y1, None],line=dict(width=1, color='#888'),hoverinfo='text',mode='lines',showlegend=False,text=f"Transactions: {edge['num_transactions']}")
                subgraph_traceRecode.append(subgraph_edge_trace)
            node_x_sub = []
            node_y_sub = []
            node_text_sub = []
            for vertex in subgraph.vs:
                x, y = layout_sub[vertex.index]
                node_x_sub.append(x)
                node_y_sub.append(y)
                node_text_sub.append(f"Address: {vertex['name']}")
            node_trace_sub = go.Scatter(x=node_x_sub,y=node_y_sub,text=node_text_sub,mode='markers',hoverinfo='text',marker=dict(color='blue',size=10))
            subgraph_traceRecode.append(node_trace_sub)
            community_subgraph_figure = {'data': subgraph_traceRecode,
                'layout': {'title': f'Subgraph for Community {selected_community}','xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},'hovermode': 'closest','margin': {'l': 40, 'r': 40, 'b': 40, 't': 40}}}
            num_nodes = subgraph.vcount()
            num_edges = subgraph.ecount()
            avg_in_degree = round(sum(subgraph.degree(mode="in")) / num_nodes, 2) if num_nodes > 0 else 0
            avg_out_degree = round(sum(subgraph.degree(mode="out")) / num_nodes, 2) if num_nodes > 0 else 0
            density = round(subgraph.density(), 5)
            num_connected_components = len(subgraph.components(mode="weak"))
            community_metrics_output = html.Div([
                html.Div(f"Number of nodes: {num_nodes}"),
                html.Div(f"Number of edges: {num_edges}"),
                html.Div(f"Average in-degree: {avg_in_degree}"),
                html.Div(f"Average out-degree: {avg_out_degree}"),
                html.Div(f"Density: {density}"),
                html.Div(f"Number of connected components: {num_connected_components}")])
    num_nodes = G.vcount()
    num_edges = G.ecount()
    avg_in_degree = round(sum(G.degree(mode="in")) / num_nodes, 2) if num_nodes > 0 else 0
    avg_out_degree = round(sum(G.degree(mode="out")) / num_nodes, 2) if num_nodes > 0 else 0
    density = round(G.density(), 5)
    num_connected_components = len(G.components(mode="weak"))
    num_communities = len(set(louvain_communities))
    metrics_output = html.Div([
        html.Div(f"Number of nodes: {num_nodes}"),
        html.Div(f"Number of edges: {num_edges}"),
        html.Div(f"Average in-degree / out-degree: {avg_in_degree}"), 
        html.Div(f"Density: {density}"),
        html.Div(f"Number of connected components: {num_connected_components}"), 
        html.Div(f"Number of communities: {num_communities}")])
    community_options = [{'label': f'Community {i}', 'value': i} for i in sorted(set(louvain_communities))]
    if selected_community is None and community_options:
        selected_community = community_options[0]['value']
    elif not community_options:
        selected_community = None
    return (figure, metrics_output, in_degree_figure, 
            out_degree_figure, influential_recipient_data, influential_sender_data, 
            community_options, selected_community, community_subgraph_figure, community_metrics_output)
if __name__ == '__main__':
    app.run_server(debug = True, port = 8071)
