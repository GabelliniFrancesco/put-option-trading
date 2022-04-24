#Import libraries
import pandas as pd
import numpy as np
import investpy
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import *
from optionprice import Option
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import math
import plotly.graph_objects as go
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Downloading data
today_datetime = datetime.today()
today = today_datetime.strftime("%d/%m/%Y")
today_v2 = today_datetime.strftime("%Y-%m-%d")
delta = relativedelta(days=5)
option_close = (today_datetime + delta).strftime("%Y-%m-%d")

s_p500 = investpy.get_index_historical_data(index='S&P 500',
                                          country='United States',
                                          from_date='01/01/2020',
                                          to_date=today)
s_p_vix = investpy.get_index_historical_data(index='S&P 500 VIX',
                                          country='United States',
                                          from_date='01/01/2020',
                                          to_date=today)
s_p_vvix = investpy.get_index_historical_data(index='CBOE Vix Volatility',
                                          country='United States',
                                          from_date='01/01/2020',
                                          to_date=today)


price_data = s_p500


def CloseToClose_estimator(price_data, window=5, trading_periods=252, clean=True):
    
    log_return = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)

    result = log_return.rolling(
        window=window,
        center=False
    ).std() * math.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result
    
    
def Parkinson_estimator(price_data, window=5, trading_periods=252, clean=True):

    rs = (1.0 / (4.0 * math.log(2.0))) * ((price_data['High'] / price_data['Low']).apply(np.log))**2.0

    def f(v):
        return trading_periods * v.mean()**0.5
    
    result = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result
    

def GarmanKlass_estimator(price_data, window=5, trading_periods=252, clean=True):

    log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
    
    def f(v):
        return (trading_periods * v.mean())**0.5
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result


    
def RogerSatchell_estimator(price_data, window=5, trading_periods=252, clean=True):
    
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return trading_periods * v.mean()**0.5
    
    result = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result
    
    
    
def YangZhang_estimator(price_data, window=5, trading_periods=252, clean=True):

    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result
    

    

    
volatilities = pd.DataFrame({
        'Close_Close':pd.DataFrame(CloseToClose_estimator(price_data)).iloc[:,0]
        ,'Parkinson':pd.DataFrame(Parkinson_estimator(price_data)/10).iloc[:,0]
        ,'GalmanKlass':pd.DataFrame(GarmanKlass_estimator(price_data)).iloc[:,0]
        ,'RogerSatchell':pd.DataFrame(RogerSatchell_estimator(price_data)/10).iloc[:,0]
        , 'YangZhang':pd.DataFrame(YangZhang_estimator(price_data)).iloc[:,0]
        
    
    })
volatilities['AverageVolatility'] = volatilities.mean(axis=1)


vol_reindex=volatilities["AverageVolatility"].copy()
start_date = datetime(2021,3,18)
vol_reindex = vol_reindex[start_date:today_v2]
vol_reindex = vol_reindex.reset_index(level=0)
vol_reindex.Date = pd.to_datetime(vol_reindex.Date)
s_p_reindex=s_p_vix.copy()
start_date = datetime(2021,3,18)
s_p_reindex = s_p_reindex[start_date:today_v2]
s_p_reindex = s_p_reindex.reset_index(level=0)
s_p_reindex.Date = pd.to_datetime(s_p_reindex.Date)
start_date = datetime(2021,3,18)
#s_p_reindex["Weekly"] = s_p_reindex.Close.rolling(window=5,center=False).mean()
merged=s_p_reindex.merge(vol_reindex,on="Date")
merged.AverageVolatility=merged.AverageVolatility*100
merged["VIX"]=merged.Close

fig1 = px.line(merged, x='Date', y=["VIX","AverageVolatility"],title="Vix & Realized")

fig1.update_traces(mode='markers+lines')
fig1.update_yaxes(title_text='Index')

fig1.update_layout()


s_p_vreindex=s_p_vvix.copy()
start_date = datetime(2021,3,18)
s_p_vreindex = s_p_vreindex[start_date:today_v2]
s_p_vreindex = s_p_vreindex.reset_index(level=0)
s_p_vreindex.Date = pd.to_datetime(s_p_vreindex.Date)
start_date = datetime(2021,3,18)

fig2 = px.line(s_p_vreindex, x='Date', y="Close",title="VVIX")
fig2.update_traces(mode='markers+lines')
fig2.update_yaxes(title_text='VVIX')

fig2.update_layout()


fig3 = px.line(s_p500, x=s_p500.index, y="Close",title="S&P500 close")
fig3.update_traces(mode='markers+lines')
fig3.update_yaxes(title_text='S&P500')

fig3.update_layout()


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([

#Header
html.Div(
    children=[
        html.H1("Volatility analysis for put options"),
        html.H5("Visualization of S&P 500 daily close, VIX and realized volatility")
]),

#Row of close & VVIX
html.Div([
        html.Div([
            dcc.Graph( id='fig3',figure=fig3),  
        ])
]),

#Row of Volatility
html.Div([
        html.Div([
            dcc.Graph( id='fig1',figure=fig1),  
        ])
]),
#Row of Calculator
html.Div(
    children=[
        html.H5("Calculator of put option pricing edge")
]),

html.Div(
    children=[
        html.Label('Enter Put Option current price'),
        dcc.Input(id="input_price", type="number", placeholder="Option Price", value=10),
        html.Label('Enter the Strike value'),
        dcc.Input(id="input_strike", type="number", placeholder="Option Strike", value=s_p500['Close'][-1:][0]
)
    ]
),
#Row of kpis
html.Div([
        html.Div([
            dcc.Graph(
                id='indic1'),  
        ], className='four columns'),
        html.Div([
            dcc.Graph(
                id='indic2'),  
        ], className='four columns'),
        html.Div([
            dcc.Graph(
                id='indic3'),  
        ], className='four columns'),
])
])

@app.callback(
    [dash.dependencies.Output('indic1', 'figure'),
    dash.dependencies.Output('indic2', 'figure'),
    dash.dependencies.Output('indic3', 'figure')
     ],
    [dash.dependencies.Input('input_price', 'value'),
     dash.dependencies.Input('input_strike', 'value')
    ])
def option_calculator(input_price, input_strike):

    weekly_volatility_forecast = (volatilities['AverageVolatility'][-1:][0])

    ## VARIANCE PREMIUM ##
    index_current =s_p500['Close'][-1:][0]

    put_option_prop = Option(european=False,
                        kind='put',
                        s0=index_current, ## ATM_price,
                        k=input_strike,##round(ATM_price,0),
                        sigma=round(weekly_volatility_forecast,4),
                        start = today_v2,
                        end = option_close,
                        r=0.00037,
                        dv=0.0302)

    # derive the prices -- PUT
    prop_price_put = put_option_prop.getPrice(method = 'BT', iteration = 10000)
    market_price_put = input_price

    #KPIS
    prop_price_put=round(prop_price_put,2)
    market_price_put= round(market_price_put,2)

    ## print mis-pricing edge
    PricingEdge=round((market_price_put/prop_price_put - 1)*100,4)

    indic1 = go.Figure()
    indic1.add_trace(go.Indicator(
        title="Market Price",
        mode = "number",
        number= { "suffix": "$" },
        value = market_price_put,
        domain = {'row': 1, 'column': 1}))

    indic2 = go.Figure()
    indic2.add_trace(go.Indicator(
        title="PUT Proprietary Price",
        mode = "number",
        number= { "suffix": "$" },
        value = prop_price_put,
        domain = {'row': 1, 'column': 1}))

    indic3 = go.Figure()
    indic3.add_trace(go.Indicator(
        title="Pricing Edge",
        mode = "number",
        number= { "suffix": "%" },
        value = PricingEdge,
        domain = {'row': 1, 'column': 1}))

    return indic1,indic2,indic3


if __name__ == "__main__":
    app.run_server(debug=True)