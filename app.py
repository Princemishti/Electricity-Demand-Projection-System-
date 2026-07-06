import os
import secrets
import io
import base64
import warnings
import json
import calendar
from datetime import datetime
from pathlib import Path
from urllib.parse import quote as url_quote  # Replace Werkzeug's url_quote with urllib.parse.quote

import matplotlib  # Ensure matplotlib is installed
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session  # Ensure Flask is installed
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user  # Ensure Flask-Login is installed
from werkzeug.security import generate_password_hash, check_password_hash  # Ensure Werkzeug is installed
from werkzeug.utils import secure_filename

import psycopg2  # Ensure psycopg2-binary is installed
from psycopg2 import sql
import pandas as pd  # Ensure pandas is installed
import numpy as np  # Ensure numpy is installed
import seaborn as sns  # Ensure seaborn is installed
import matplotlib.pyplot as plt  # Ensure matplotlib is installed
from matplotlib.dates import DateFormatter

from sklearn.model_selection import train_test_split  # Ensure scikit-learn is installed
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.seasonal import seasonal_decompose  # Ensure statsmodels is installed
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
from statsmodels.tsa.api import Holt
try:
    from prophet import Prophet
except ImportError:
    Prophet = None


# Plotly imports with error handling
try:
    import plotly.graph_objects as go  # Ensure plotly is installed
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will not work.")

# Generate or load secret key
def get_or_create_secret_key():
    secret_file = Path('instance/secret_key.txt')
    secret_file.parent.mkdir(exist_ok=True)
    
    if secret_file.exists():
        return secret_file.read_text()
    else:
        new_key = secrets.token_hex(32)
        secret_file.write_text(new_key)
        return new_key

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or get_or_create_secret_key()


# Configuration
app.config['STATIC_FOLDER'] = 'static'
app.config['USER_GRAPHS_FOLDER'] = 'user_graphs'

# Database configuration
DB_CONFIG = {
    'dbname': 'power_analytics_db',
    'user': 'prince',
    'password': '9999',
    'host': 'localhost'
}

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Import database and model functions
from models.user import User
import utils.db

from routes.prediction import prediction_bp
app.register_blueprint(prediction_bp)

@login_manager.user_loader
def load_user(user_id):
    return utils.db.load_user(user_id)

def init_db():
    utils.db.init_db()

# Initialize database
init_db()


# ----------------------------
# DATA LOADING AND PREPROCESSING
# ----------------------------


df1 = pd.read_csv("Daily_Power_Gen_Source_march_23.csv")
df1_source = df1.drop(df1[df1['source'] == 'Total'].index)
df1_source['date'] = pd.to_datetime(df1['date'])
df1_source['day'] = df1_source['date'].dt.day
df1_source['month'] = df1_source['date'].dt.month
df1_source['year'] = df1_source['date'].dt.year
df1_source['month_name'] = df1_source['date'].dt.month_name()

df2 = pd.read_csv("Daily_Power_Gen_States_march_23.csv")
df2['Shortage during maximum Demand(MW)'] = df2['Shortage during maximum Demand(MW)'].fillna(df2['Shortage during maximum Demand(MW)'].mean())
df2['Energy Met (MU)'] = df2['Energy Met (MU)'].fillna(0)
df2['date'] = pd.to_datetime(df2['date'])
df2['day'] = df2['date'].dt.day
df2['month'] = df2['date'].dt.month
df2['year'] = df2['date'].dt.year
df2['month_name'] = df2['date'].dt.month_name()
df2.rename(columns={'Energy Met (MU)': 'Energy Met (MW)'}, inplace=True)
df2['Energy Met (MW)'] = df2['Energy Met (MW)'] * 1000

# Ensure month names are in correct order
month_order = list(calendar.month_name)[1:]
df1_source['month_name'] = pd.Categorical(df1_source['month_name'], categories=month_order, ordered=True)
df2['month_name'] = pd.Categorical(df2['month_name'], categories=month_order, ordered=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def cleanup_plots():
    plt.close('all')

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics with enhanced accuracy calculation"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'mse': mse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    relative_error = np.mean(np.abs(y_true - y_pred) / (np.max(y_true) - np.min(y_true)))
    metrics['accuracy'] = max(0, min(100, (1 - relative_error) * 100))
    
    # Add rating based on accuracy
    if metrics['accuracy'] >= 90:
        metrics['rating'] = 'Excellent'
        metrics['rating_color'] = 'success'
    elif metrics['accuracy'] >= 75:
        metrics['rating'] = 'Good'
        metrics['rating_color'] = 'primary'
    elif metrics['accuracy'] >= 50:
        metrics['rating'] = 'Fair'
        metrics['rating_color'] = 'warning'
    else:
        metrics['rating'] = 'Poor'
        metrics['rating_color'] = 'danger'
    
    return metrics

def save_user_history(user_id, graph_type, filters, graph_path=None):
    """Save user's graph generation history to database"""
    utils.db.save_user_history(user_id, graph_type, filters, graph_path)

def get_user_history(user_id, limit=10):
    """Retrieve user's graph generation history"""
    return utils.db.get_user_history(user_id, limit)


def generate_plotly_figure(graph_type, filtered_df1, filtered_df2, source, region, state, year, month):
    """Generate interactive Plotly figure with enhanced explanations and metrics"""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from statsmodels.tsa.holtwinters import Holt
    import warnings
    
    explanation = ""
    model_metrics = None
    fig = None
    
    try:
        # Create copies of dataframes to avoid modifying originals
        df1_all = filtered_df1.copy()
        df2_all = filtered_df2.copy()
        
        # Apply year filter only if specific year is selected
        if year != 'All':
            filtered_df1 = df1_all[df1_all['year'] == int(year)]
            filtered_df2 = df2_all[df2_all['year'] == int(year)]
        else:
            filtered_df1 = df1_all
            filtered_df2 = df2_all
            
        if graph_type == 'time_series':
            if source and source != 'All':
                if region and region != 'All':
                    if year != 'All':
                        fig = px.line(filtered_df1, x='date', y=region, 
                                    title=f'{source} Power Generation in {region} ({year})',
                                    labels={'date': 'Date', region: 'Power Generation (MW)'})
                    else:
                        # For all years, aggregate by year
                        yearly_data = filtered_df1.groupby('year')[region].mean().reset_index()
                        fig = px.line(yearly_data, x='year', y=region,
                                    title=f'{source} Power Generation in {region} (All Years)',
                                    labels={'year': 'Year', region: 'Power Generation (MW)'})
                else:
                    if year != 'All':
                        fig = px.line(filtered_df1, x='date', y='All India',
                                    title=f'{source} Power Generation ({year})',
                                    labels={'date': 'Date', 'All India': 'Power Generation (MW)'})
                    else:
                        yearly_data = filtered_df1.groupby('year')['All India'].mean().reset_index()
                        fig = px.line(yearly_data, x='year', y='All India',
                                    title=f'{source} Power Generation (All Years)',
                                    labels={'year': 'Year', 'All India': 'Power Generation (MW)'})
            else:
                if region and region != 'All':
                    if year != 'All':
                        fig = px.line(filtered_df1, x='date', y=region, color='source',
                                    title=f'Power Generation by Source in {region} ({year})',
                                    labels={'date': 'Date', region: 'Power Generation (MW)', 'source': 'Power Source'})
                    else:
                        yearly_data = filtered_df1.groupby(['year', 'source'])[region].mean().reset_index()
                        fig = px.line(yearly_data, x='year', y=region, color='source',
                                    title=f'Power Generation by Source in {region} (All Years)',
                                    labels={'year': 'Year', region: 'Power Generation (MW)', 'source': 'Power Source'})
                else:
                    if year != 'All':
                        fig = px.line(filtered_df1, x='date', y='All India', color='source',
                                    title=f'Power Generation by Source ({year})',
                                    labels={'date': 'Date', 'All India': 'Power Generation (MW)', 'source': 'Power Source'})
                    else:
                        yearly_data = filtered_df1.groupby(['year', 'source'])['All India'].mean().reset_index()
                        fig = px.line(yearly_data, x='year', y='All India', color='source',
                                    title=f'Power Generation by Source (All Years)',
                                    labels={'year': 'Year', 'All India': 'Power Generation (MW)', 'source': 'Power Source'})

            if year == 'All':
                fig.update_layout(
                    xaxis=dict(
                        title='Year',
                        type='category'
                    )
                )
            else:
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
        
        elif graph_type == 'bar_chart':
            if source and source != 'All':
                if state and state != 'All':
                    state_data = filtered_df2[filtered_df2['States'] == state]
                    if year != 'All':
                        # Monthly view for specific year
                        agg_df = state_data.groupby('month_name').agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        }).reset_index()
                        x_axis = 'month_name'
                        title = f'{source} Power Analysis - {state} ({year})'
                    else:
                        # Yearly view for all years
                        agg_df = state_data.groupby('year').agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        }).reset_index()
                        x_axis = 'year'
                        title = f'{source} Power Analysis - {state} (All Years)'
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Demand Met',
                        x=agg_df[x_axis],
                        y=agg_df['Max.Demand Met during the day(MW)'],
                        text=agg_df['Max.Demand Met during the day(MW)'].round(0),
                        textposition='auto',
                    ))
                    fig.add_trace(go.Bar(
                        name='Shortage',
                        x=agg_df[x_axis],
                        y=agg_df['Shortage during maximum Demand(MW)'],
                        text=agg_df['Shortage during maximum Demand(MW)'].round(0),
                        textposition='auto',
                    ))
                    
                    if year == 'All':
                        fig.add_trace(go.Scatter(
                            x=agg_df[x_axis],
                            y=agg_df['Max.Demand Met during the day(MW)'].rolling(window=2).mean(),
                            name='Trend',
                            line=dict(color='red', dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title='Time Period',
                        yaxis_title='Power (MW)',
                        barmode='group'
                    )
                else:
                    if year != 'All':
                        state_metrics = filtered_df2.groupby('States').agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        }).reset_index()
                    else:
                        state_metrics = filtered_df2.groupby(['States', 'year']).agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        }).groupby('States').mean().reset_index()
                    
                    state_metrics['Total'] = state_metrics['Max.Demand Met during the day(MW)'] + \
                                           state_metrics['Shortage during maximum Demand(MW)']
                    state_metrics = state_metrics.nlargest(10, 'Total')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Demand Met',
                        x=state_metrics['States'],
                        y=state_metrics['Max.Demand Met during the day(MW)'],
                        text=state_metrics['Max.Demand Met during the day(MW)'].round(0),
                        textposition='auto',
                    ))
                    fig.add_trace(go.Bar(
                        name='Shortage',
                        x=state_metrics['States'],
                        y=state_metrics['Shortage during maximum Demand(MW)'],
                        text=state_metrics['Shortage during maximum Demand(MW)'].round(0),
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title=f'Top 10 States - {source} Power Analysis ({year if year != "All" else "All Years"})',
                        xaxis_title='State',
                        yaxis_title='Power (MW)',
                        barmode='group'
                    )
            else:
                if state and state != 'All':
                    state_data = filtered_df2[filtered_df2['States'] == state]
                    if year != 'All':
                        agg_df = state_data.groupby('month_name')['Max.Demand Met during the day(MW)'].mean().reset_index()
                        x_axis = 'month_name'
                        title = f'Average Monthly Power Demand in {state} ({year})'
                    else:
                        agg_df = state_data.groupby('year')['Max.Demand Met during the day(MW)'].mean().reset_index()
                        x_axis = 'year'
                        title = f'Average Yearly Power Demand in {state} (All Years)'
                    
                    fig = px.bar(agg_df, x=x_axis, y='Max.Demand Met during the day(MW)', title=title)
                    
                    if year == 'All':
                        fig.add_trace(go.Scatter(
                            x=agg_df[x_axis],
                            y=agg_df['Max.Demand Met during the day(MW)'].rolling(window=2).mean(),
                            name='Trend',
                            line=dict(color='red', dash='dash')
                        ))
                else:
                    if year != 'All':
                        agg_df = filtered_df1.groupby(['month_name', 'source'])['All India'].mean().reset_index()
                        x_axis = 'month_name'
                        title = f'Average Monthly Power Generation by Source ({year})'
                    else:
                        agg_df = filtered_df1.groupby(['year', 'source'])['All India'].mean().reset_index()
                        x_axis = 'year'
                        title = 'Average Yearly Power Generation by Source (All Years)'
                    
                    fig = px.bar(agg_df, x=x_axis, y='All India', color='source', title=title)
                    
                    if year == 'All':
                        yearly_total = agg_df.groupby('year')['All India'].sum().reset_index()
                        fig.add_trace(go.Scatter(
                            x=yearly_total['year'],
                            y=yearly_total['All India'].rolling(window=2).mean(),
                            name='Overall Trend',
                            line=dict(color='red', dash='dash')
                        ))
        
        elif graph_type == 'pie_chart':
            if source and source != 'All':
                if state and state != 'All':
                    state_data = filtered_df2[filtered_df2['States'] == state]
                    total_demand = state_data['Max.Demand Met during the day(MW)'].sum()
                    total_shortage = state_data['Shortage during maximum Demand(MW)'].sum()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Demand Met', 'Shortage'],
                        values=[total_demand, total_shortage],
                        hole=.3,
                        text=[f'{(total_demand/(total_demand+total_shortage)*100):.1f}%',
                              f'{(total_shortage/(total_demand+total_shortage)*100):.1f}%']
                    )])
                    
                    fig.update_layout(
                        title=f'Power Demand Distribution in {state}<br>{source} ({year if year != "All" else "All Years"})'
                    )
                else:
                    state_data = filtered_df2.copy()
                    if year != 'All':
                        state_data = state_data[state_data['year'] == int(year)]
                    
                    state_totals = state_data.groupby('States').agg({
                        'Max.Demand Met during the day(MW)': 'sum'
                    }).reset_index()
                    
                    top_states = state_totals.nlargest(8, 'Max.Demand Met during the day(MW)')
                    others_total = state_totals[~state_totals['States'].isin(top_states['States'])]['Max.Demand Met during the day(MW)'].sum()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=list(top_states['States']) + ['Others'],
                        values=list(top_states['Max.Demand Met during the day(MW)']) + [others_total],
                        hole=.3
                    )])
                    
                    fig.update_layout(
                        title=f'State-wise Power Distribution - {source}<br>({year if year != "All" else "All Years"})'
                    )
            else:
                agg_df = filtered_df1.groupby('source')['All India'].sum().reset_index()
                fig = px.pie(agg_df, values='All India', names='source',
                           title='Power Generation Source Distribution', hole=0.3)
        
        elif graph_type == 'box_plot':
            if source and source != 'All':
                if state and state != 'All':
                    state_data = filtered_df2[filtered_df2['States'] == state]
                    plot_data = pd.melt(state_data, 
                                      id_vars=['month_name'],
                                      value_vars=['Max.Demand Met during the day(MW)', 
                                                'Shortage during maximum Demand(MW)'],
                                      var_name='Metric', value_name='Power')
                    
                    fig = px.box(plot_data, x='month_name', y='Power', color='Metric',
                               title=f'Monthly Power Distribution - {source} in {state}')
                else:
                    state_data = filtered_df2.copy()
                    state_avgs = state_data.groupby('States').agg({
                        'Max.Demand Met during the day(MW)': 'mean',
                        'Shortage during maximum Demand(MW)': 'mean'
                    })
                    state_avgs['Total'] = state_avgs['Max.Demand Met during the day(MW)'] + \
                                        state_avgs['Shortage during maximum Demand(MW)']
                    top_states = state_avgs.nlargest(10, 'Total').index
                    
                    plot_data = state_data[state_data['States'].isin(top_states)]
                    plot_data = pd.melt(plot_data,
                                      id_vars=['States'],
                                      value_vars=['Max.Demand Met during the day(MW)',
                                                'Shortage during maximum Demand(MW)'],
                                      var_name='Metric', value_name='Power')
                    
                    fig = px.box(plot_data, x='States', y='Power', color='Metric',
                               title=f'Power Distribution by State - {source}')
            else:
                if state and state != 'All':
                    state_data = filtered_df2[filtered_df2['States'] == state]
                    fig = px.box(state_data, x='month_name', y='Max.Demand Met during the day(MW)',
                               title=f'Distribution of Power Demand in {state}')
                else:
                    if region and region != 'All':
                        fig = px.box(filtered_df1, x='source', y=region,
                                   title=f'Distribution of Power Generation by Source in {region}')
                    else:
                        fig = px.box(filtered_df1, x='source', y='All India',
                                   title='Distribution of Power Generation by Source')
            
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
        
        elif graph_type == 'heatmap':
            if source and source != 'All':
                if state and state != 'All':
                    state_data = filtered_df2[filtered_df2['States'] == state]
                    
                    pivot_demand = state_data.pivot_table(
                        index='month_name',
                        columns='year',
                        values='Max.Demand Met during the day(MW)',
                        aggfunc='mean'
                    )
                    
                    pivot_shortage = state_data.pivot_table(
                        index='month_name',
                        columns='year',
                        values='Shortage during maximum Demand(MW)',
                        aggfunc='mean'
                    )
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Demand Met (MW)', 'Shortage (MW)'),
                        horizontal_spacing=0.15
                    )
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=pivot_demand.values,
                            x=pivot_demand.columns,
                            y=pivot_demand.index,
                            colorscale='YlOrRd',
                            showscale=True,
                            name='Demand Met'
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=pivot_shortage.values,
                            x=pivot_shortage.columns,
                            y=pivot_shortage.index,
                            colorscale='YlOrRd',
                            showscale=True,
                            name='Shortage'
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title=f'Power Analysis Heatmap - {source} in {state}',
                        height=600
                    )
                else:
                    state_data = filtered_df2.copy()
                    
                    state_avgs = state_data.groupby('States').agg({
                        'Max.Demand Met during the day(MW)': 'mean',
                        'Shortage during maximum Demand(MW)': 'mean'
                    })
                    state_avgs['Total'] = state_avgs['Max.Demand Met during the day(MW)'] + \
                                        state_avgs['Shortage during maximum Demand(MW)']
                    top_states = state_avgs.nlargest(15, 'Total').index
                    
                    plot_data = state_data[state_data['States'].isin(top_states)]
                    
                    pivot_demand = plot_data.pivot_table(
                        index='States',
                        columns='month_name',
                        values='Max.Demand Met during the day(MW)',
                        aggfunc='mean'
                    )
                    
                    pivot_shortage = plot_data.pivot_table(
                        index='States',
                        columns='month_name',
                        values='Shortage during maximum Demand(MW)',
                        aggfunc='mean'
                    )
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Average Demand Met (MW)', 'Average Shortage (MW)'),
                        horizontal_spacing=0.15
                    )
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=pivot_demand.values,
                            x=pivot_demand.columns,
                            y=pivot_demand.index,
                            colorscale='YlOrRd',
                            showscale=True,
                            name='Demand Met'
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=pivot_shortage.values,
                            x=pivot_shortage.columns,
                            y=pivot_shortage.index,
                            colorscale='YlOrRd',
                            showscale=True,
                            name='Shortage'
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title=f'State-wise Monthly Power Analysis - {source}',
                        height=800
                    )
            else:
                if state and state != 'All':
                    state_data = filtered_df2[filtered_df2['States'] == state]
                    pivot_df = state_data.pivot_table(
                        index='month_name',
                        columns='year',
                        values='Max.Demand Met during the day(MW)',
                        aggfunc='mean'
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='YlOrRd'
                    ))
                    
                    fig.update_layout(
                        title=f'Monthly Power Demand Heatmap for {state}',
                        xaxis_title='Year',
                        yaxis_title='Month'
                    )
                else:
                    pivot_df = filtered_df1.pivot_table(
                        index='month_name',
                        columns='year',
                        values='All India',
                        aggfunc='mean'
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='YlOrRd'
                    ))
                    
                    fig.update_layout(
                        title='Monthly Power Generation Heatmap for All India',
                        xaxis_title='Year',
                        yaxis_title='Month'
                    )
        
        elif graph_type == 'prediction':
            if state and state != 'All':
                y = filtered_df2['Max.Demand Met during the day(MW)']
                X = filtered_df2[['Energy Met (MW)', 'Shortage during maximum Demand(MW)']]
                title = f"Energy Demand Prediction for {state}"
                data_description = f"state {state}"
            elif region and region != 'All':
                y = filtered_df1[region]
                X = filtered_df1[['day', 'month', 'year']]
                title = f"Energy Demand Prediction for {region} Region"
                data_description = f"{region} region"
            else:
                y = filtered_df1['All India']
                X = filtered_df1[['day', 'month', 'year']]
                title = "All India Energy Demand Prediction"
                data_description = "all India"
            
            train_size = int(len(y) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
                'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
                'Holt-Winters': Holt(y_train)
            }
            
            predictions = {}
            model_metrics = {}
            
            for name, model in models.items():
                try:
                    if name == 'Holt-Winters':
                        fit = model.fit(smoothing_level=0.8, smoothing_trend=0.02)
                        pred = fit.forecast(len(y_test))
                    else:
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                    
                    predictions[name] = pred
                    model_metrics[name] = calculate_metrics(y_test, pred)
                except Exception as e:
                    warnings.warn(f"Error fitting {name}: {str(e)}")
                    continue
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(y_train))),
                y=y_train,
                name='Training Data',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(y_train), len(y))),
                y=y_test,
                name='Test Data',
                line=dict(color='pink')
            ))
            
            colors = ['green', 'red', 'purple', 'orange']
            for i, (name, pred) in enumerate(predictions.items()):
                fig.add_trace(go.Scatter(
                    x=list(range(len(y_train), len(y))),
                    y=pred,
                    name=f'{name} ({model_metrics[name]["rating"]})',
                    line=dict(color=colors[i], dash='dash')
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Time Index',
                yaxis_title='Power Demand (MW)',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            
            best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
            explanation = f"""
            <div class="alert alert-info">
                <h5><i class="bi bi-lightbulb"></i> Prediction Analysis for {data_description}</h5>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-body">
                                <h6 class="card-title"><i class="bi bi-trophy-fill text-warning"></i> Best Model</h6>
                                <div class="d-flex align-items-center">
                                    <h4 class="mb-0">{best_model[0]}</h4>
                                    <span class="badge bg-{best_model[1]['rating_color']} ms-2">{best_model[1]['rating']}</span>
                                </div>
                                <p class="mb-0 small text-muted">Accuracy: {best_model[1]['accuracy']:.1f}%</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-body">
                                <h6 class="card-title"><i class="bi bi-info-circle"></i> Model Guide</h6>
                                <ul class="small mb-0">
                                    <li>90-100%: Excellent</li>
                                    <li>75-89%: Good</li>
                                    <li>50-74%: Fair</li>
                                    <li>Below 50%: Poor</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="mt-2">
                    <p class="mb-1"><strong>Analysis Notes:</strong></p>
                    <ul class="mb-0">
                        <li>Training period: {len(y_train)} days</li>
                        <li>Test period: {len(y_test)} days</li>
                        <li>Dashed lines show model predictions vs actual (pink)</li>
                    </ul>
                </div>
            </div>
            """
        
        # Common layout updates for all plots
        fig.update_layout(
            template='plotly_white',
            margin=dict(t=100, l=50, r=50, b=50),
            hovermode='closest',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
        
        # For year-based x-axis, ensure proper ordering
        if year == 'All' and 'xaxis' in fig.layout:
            fig.update_xaxes(type='category', categoryorder='array', 
                           categoryarray=sorted(filtered_df1['year'].unique()))
        
        return fig, model_metrics, explanation
        
    except Exception as e:
        print(f"Error in generate_plotly_figure: {str(e)}")
        return None, None, f"Error generating plot: {str(e)}"

# ----------------------------
# AUTHENTICATION ROUTES
# ----------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            user_data = utils.db.get_user_by_username(username)
            if user_data and check_password_hash(user_data['password'], password):
                user = User(user_data['id'], user_data['username'])
                login_user(user)
                flash('Logged in successfully!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password', 'danger')
        except Exception as e:
            flash('Database error occurred', 'danger')
            app.logger.error(f"Login error: {str(e)}")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        
        # Enhanced validation
        if len(username) < 4:
            flash('Username must be at least 4 characters', 'danger')
            return redirect(url_for('register'))
            
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'danger')
            return redirect(url_for('register'))
        
        try:
            # Check if username exists
            existing_user = utils.db.get_user_by_username(username)
            if existing_user:
                flash('Username already exists', 'danger')
                return redirect(url_for('register'))
            
            # Create new user
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            user_id = utils.db.create_user(username, hashed_password)
            
            if not user_id:
                flash('Registration failed. Please try again.', 'danger')
                return redirect(url_for('register'))
            
            # Create user folder
            user_folder = os.path.join(app.config['STATIC_FOLDER'], 
                                     app.config['USER_GRAPHS_FOLDER'], 
                                     str(user_id))
            os.makedirs(user_folder, exist_ok=True)
            
            user = User(user_id, username)
            login_user(user)
            flash('Account created successfully!', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            app.logger.error(f"Registration error: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')
            
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# ----------------------------
# MAIN APPLICATION ROUTES
# ----------------------------

def check_plotly_available():
    try:
        import plotly
        return True
    except ImportError:
        return False

@app.route('/')
@login_required
def index():
    # Get only sources that have data
    sources = sorted(df1_source['source'].unique().tolist())
    
    # Regions available in the dataset
    regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']
    
    # Get years with data
    years = sorted(df1_source['year'].unique().tolist())
    
    # Get months with data
    months = sorted(df1_source['month'].unique().tolist())
    
    # Get states with data (excluding NaN values)
    states = sorted(df2['States'].dropna().unique().tolist())
    
    stats = {
        'years_count': df1_source['year'].nunique(),
        'sources_count': df1_source['source'].nunique(),
        'states_count': df2['States'].nunique(),
        'total_generation': "{:,.0f}".format(df1_source['All India'].sum()/1000) + "K",
        'renewable_growth': "{:.1f}x".format(
            df1_source[df1_source['source'].str.contains('Solar')]['All India'].sum() / 
            df1_source[df1_source['source'].str.contains('Solar') & 
                      (df1_source['year'] == 2015)]['All India'].sum()
        ),
        'western_region_share': "{:.0f}%".format(
            df1_source['WR'].sum() / df1_source['All India'].sum() * 100
        ),
        'plotly_available': check_plotly_available()
    }
    
    # Get user history
    history = get_user_history(current_user.id)
    
    return render_template('index.html', 
                         sources=sources,
                         regions=regions,
                         years=years,
                         months=months,
                         states=states,
                         stats=stats,
                         history=history,
                         current_user=current_user)

@app.route('/generate_graph', methods=['POST'])
@login_required
def generate_graph():
    try:
        # Get interactive parameter from form data
        is_interactive = request.form.get('interactive', 'false').lower() == 'true'
        
        if is_interactive and not check_plotly_available():
            return jsonify({
                'error': 'Interactive mode is not available. Please install plotly using: pip install plotly'
            }), 400
            
        cleanup_plots()
        
        # Get form data
        graph_type = request.form.get('graph_type')
        source = request.form.get('source', 'All')
        region = request.form.get('region', 'All')
        year = request.form.get('year', 'All')
        month = request.form.get('month', 'All')
        state = request.form.get('state', 'All')

        # Create copies of the dataframes
        filtered_df1 = df1_source.copy()
        filtered_df2 = df2.copy()

        # Apply filters
        if year != 'All':
            year = int(year)
            filtered_df1 = filtered_df1[filtered_df1['year'] == year]
            filtered_df2 = filtered_df2[filtered_df2['year'] == year]

        if source != 'All':
            filtered_df1 = filtered_df1[filtered_df1['source'] == source]
            source_dates = set(filtered_df1['date'])
            
            if state and state != 'All':
                state_dates = set(filtered_df2[filtered_df2['States'] == state]['date'])
                if not source_dates.intersection(state_dates):
                    return jsonify({
                        'error': f'No data available for source "{source}" in state "{state}". This combination might not exist in the dataset.'
                    }), 400
            
            # Filter df2 to only include dates where we have data for this source
            filtered_df2 = filtered_df2[filtered_df2['date'].isin(source_dates)]

        # Apply state filter
        if state and state != 'All':
            filtered_df2 = filtered_df2[filtered_df2['States'] == state]
            if filtered_df2.empty:
                return jsonify({
                    'error': f'No data available for state "{state}" with the selected filters.'
                }), 400

        # Apply month filter
        if month and month != 'All':
            month = int(month)
            filtered_df1 = filtered_df1[filtered_df1['month'] == month]
            filtered_df2 = filtered_df2[filtered_df2['month'] == month]

        # Validate data availability
        if filtered_df1.empty and filtered_df2.empty:
            error_msg = "No data available for the selected combination. "
            if source != 'All':
                error_msg += f'Source "{source}" '
                if state != 'All':
                    error_msg += f'is not available in state "{state}" '
                if year != 'All':
                    error_msg += f'for year {year} '
                if month != 'All':
                    error_msg += f'in month {month} '
            error_msg += "Please try different filters."
            
            return jsonify({'error': error_msg}), 400

        # Create user-specific folder
        user_folder = os.path.join(
            app.config['STATIC_FOLDER'],
            app.config['USER_GRAPHS_FOLDER'],
            str(current_user.id)
        )
        os.makedirs(user_folder, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{graph_type}_{source}_{region}_{timestamp}"
        png_filename = f"{filename_base}.png"
        html_filename = f"{filename_base}.html"
        png_path = os.path.join(user_folder, png_filename)
        html_path = os.path.join(user_folder, html_filename)
        relative_png_path = os.path.join(app.config['USER_GRAPHS_FOLDER'], str(current_user.id), png_filename)
        relative_html_path = os.path.join(app.config['USER_GRAPHS_FOLDER'], str(current_user.id), html_filename)

        if is_interactive and check_plotly_available():
            fig, model_metrics, explanation = generate_plotly_figure(
                graph_type, filtered_df1, filtered_df2, 
                source, region, state, year, month
            )
            
            # Convert the Plotly figure to JSON
            fig_json = fig.to_json()
            
            return jsonify({
                'plotly_json': fig_json,
                'model_metrics': model_metrics,
                'explanation': explanation,
                'interactive': True
            })
        else:
            # Generate static plot
            fig, ax = plt.subplots(figsize=(12, 7))
            model_metrics = {}
            explanation = ""
            
            if graph_type == 'time_series':
                if source and source != 'All':
                    if region and region != 'All':
                        ax.plot(filtered_df1['date'], filtered_df1[region])
                        ax.set_title(f'{source} Power Generation in {region} Over Time')
                    else:
                        ax.plot(filtered_df1['date'], filtered_df1['All India'])
                        ax.set_title(f'{source} Power Generation Over Time')
                else:
                    if region and region != 'All':
                        sns.lineplot(data=filtered_df1, x='date', y=region, hue='source', ax=ax)
                        ax.set_title(f'Power Generation by Source in {region}')
                    else:
                        sns.lineplot(data=filtered_df1, x='date', y='All India', hue='source', ax=ax)
                        ax.set_title('Power Generation by Source Over Time')
                ax.set_xlabel('Date')
                ax.set_ylabel('Power Generation (MW)')
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                fig.autofmt_xdate()
                
                explanation = f"""
                <div class="alert alert-info">
                    <h5><i class="bi bi-info-circle"></i> Time Series Analysis</h5>
                    <ul>
                        <li>Showing data from {filtered_df1['date'].min().strftime('%Y-%m-%d')} to {filtered_df1['date'].max().strftime('%Y-%m-%d')}</li>
                        <li>Vertical axis shows power generation in Megawatts (MW)</li>
                        <li>Multiple sources shown when "All Sources" selected</li>
                        <li>Seasonal patterns may be visible in the data</li>
                    </ul>
                </div>
                """
                
            elif graph_type == 'bar_chart':
                # Modify bar chart to show state-wise or source-wise data based on selection
                if source and source != 'All':
                    if state and state != 'All':
                        # Single state and source view - show monthly pattern
                        state_data = filtered_df2[filtered_df2['States'] == state]
                        if year != 'All':
                            state_data = state_data[state_data['year'] == int(year)]
                        
                        # Calculate monthly averages for the specific state
                        agg_df = state_data.groupby('month_name').agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        }).reset_index()
                        
                        # Create a grouped bar chart
                        bar_width = 0.35
                        index = np.arange(len(agg_df['month_name']))
                        
                        ax.bar(index - bar_width/2, agg_df['Max.Demand Met during the day(MW)'], 
                              bar_width, label='Demand Met', color='skyblue')
                        ax.bar(index + bar_width/2, agg_df['Shortage during maximum Demand(MW)'], 
                              bar_width, label='Shortage', color='lightcoral')
                        
                        ax.set_xticks(index)
                        ax.set_xticklabels(agg_df['month_name'], rotation=45)
                        ax.set_title(f'{source} Power Analysis - {state}\n({year if year != "All" else "All Years"})')
                        ax.set_ylabel('Power (MW)')
                        ax.legend()
                        
                        # Add value labels on bars
                        for i, v in enumerate(agg_df['Max.Demand Met during the day(MW)']):
                            ax.text(i - bar_width/2, v, f'{v:,.0f}', ha='center', va='bottom')
                        for i, v in enumerate(agg_df['Shortage during maximum Demand(MW)']):
                            ax.text(i + bar_width/2, v, f'{v:,.0f}', ha='center', va='bottom')
                    else:
                        # Compare all states for selected source
                        states_data = filtered_df2.copy()  # Use filtered data instead of original
                        
                        # Calculate metrics for each state
                        state_metrics = states_data.groupby('States').agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        }).reset_index()
                        
                        # Sort by total power (demand + shortage)
                        state_metrics['Total'] = state_metrics['Max.Demand Met during the day(MW)'] + \
                                               state_metrics['Shortage during maximum Demand(MW)']
                        state_metrics = state_metrics.nlargest(10, 'Total')
                        
                        # Create grouped bar chart
                        bar_width = 0.35
                        index = np.arange(len(state_metrics))
                        
                        ax.bar(index - bar_width/2, state_metrics['Max.Demand Met during the day(MW)'],
                              bar_width, label='Demand Met', color='skyblue')
                        ax.bar(index + bar_width/2, state_metrics['Shortage during maximum Demand(MW)'],
                              bar_width, label='Shortage', color='lightcoral')
                        
                        ax.set_xticks(index)
                        ax.set_xticklabels(state_metrics['States'], rotation=45, ha='right')
                        ax.set_title(f'Top 10 States - {source} Power Analysis\n({year if year != "All" else "All Years"})')
                        ax.set_ylabel('Power (MW)')
                        ax.legend()
                        
                        # Add value labels
                        for i, v in enumerate(state_metrics['Max.Demand Met during the day(MW)']):
                            ax.text(i - bar_width/2, v, f'{v:,.0f}', ha='center', va='bottom')
                        for i, v in enumerate(state_metrics['Shortage during maximum Demand(MW)']):
                            ax.text(i + bar_width/2, v, f'{v:,.0f}', ha='center', va='bottom')
                else:
                    # Original bar chart logic for when no specific source is selected
                    if state and state != 'All':
                        state_data = filtered_df2[filtered_df2['States'] == state]
                        if year != 'All':
                            agg_df = state_data.groupby('month_name')['Max.Demand Met during the day(MW)'].mean().reset_index()
                            sns.barplot(data=agg_df, x='month_name', y='Max.Demand Met during the day(MW)', ax=ax)
                            ax.set_title(f'Average Monthly Power Demand in {state} ({year})')
                        else:
                            agg_df = state_data.groupby('year')['Max.Demand Met during the day(MW)'].mean().reset_index()
                            sns.barplot(data=agg_df, x='year', y='Max.Demand Met during the day(MW)', ax=ax)
                            ax.set_title(f'Average Yearly Power Demand in {state}')
                        ax.set_ylabel('Power Demand (MW)')
                    else:
                        if year != 'All':
                            agg_df = filtered_df1.groupby(['month_name', 'source'])['All India'].mean().reset_index()
                            sns.barplot(data=agg_df, x='month_name', y='All India', hue='source', ax=ax)
                            ax.set_title(f'Average Monthly Power Generation by Source ({year})')
                        else:
                            agg_df = filtered_df1.groupby(['year', 'source'])['All India'].mean().reset_index()
                            sns.barplot(data=agg_df, x='year', y='All India', hue='source', ax=ax)
                            ax.set_title('Average Yearly Power Generation by Source')
                        ax.set_ylabel('Power Generation (MW)')
                
                ax.set_xlabel('Time Period')
                if ax.get_legend():
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                
                explanation = f"""
                <div class="alert alert-info">
                    <h5><i class="bi bi-info-circle"></i>Average Analysis</h5>
                    <ul>
                        <li>Shows average power generation by month</li>
                        <li>Helps identify seasonal patterns</li>
                        <li>Grouped bars show comparison between sources</li>
                    </ul>
                </div>
                """
                
            elif graph_type == 'pie_chart':
                if source and source != 'All':
                    if state and state != 'All':
                        # Show demand vs shortage distribution for selected state
                        state_data = filtered_df2[filtered_df2['States'] == state]
                        total_demand = state_data['Max.Demand Met during the day(MW)'].sum()
                        total_shortage = state_data['Shortage during maximum Demand(MW)'].sum()
                        
                        # Calculate percentages
                        total = total_demand + total_shortage
                        demand_pct = (total_demand / total) * 100
                        shortage_pct = (total_shortage / total) * 100
                        
                        data = [total_demand, total_shortage]
                        labels = [f'Demand Met\n({demand_pct:.1f}%)', f'Shortage\n({shortage_pct:.1f}%)']
                        colors = ['skyblue', 'lightcoral']
                        
                        ax.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', 
                              wedgeprops=dict(width=0.3))
                        ax.set_title(f'Power Demand Distribution in {state}\n{source} ({year if year != "All" else "All Years"})')
                    else:
                        state_data = filtered_df2.copy()
                        if year != 'All':
                            state_data = state_data[state_data['year'] == int(year)]
                        
                        # Calculate total power (demand + shortage) for each state
                        state_totals = state_data.groupby('States').agg({
                            'Max.Demand Met during the day(MW)': 'sum',
                            'Shortage during maximum Demand(MW)': 'sum'
                        })
                        state_totals['Total'] = state_totals['Max.Demand Met during the day(MW)'] + \
                                              state_totals['Shortage during maximum Demand(MW)']
                        
                        # Get top states
                        top_states = state_totals.nlargest(8, 'Total')
                        others_total = state_totals[~state_totals.index.isin(top_states.index)]['Total'].sum()
                        
                        # Prepare data for pie chart
                        plot_data = pd.concat([
                            top_states['Total'],
                            pd.Series({'Others': others_total})
                        ])
                        
                        ax.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', 
                              wedgeprops=dict(width=0.3))
                        ax.set_title(f'State-wise Power Distribution - {source}\n({year if year != "All" else "All Years"})')
                else:
                    agg_df = filtered_df1.groupby('source')['All India'].sum().reset_index()
                    ax.pie(agg_df['All India'], labels=agg_df['source'], autopct='%1.1f%%', 
                          wedgeprops=dict(width=0.3))
                    ax.set_title(f'Power Generation Source Distribution ({year})')
            
            elif graph_type == 'box_plot':
                if source and source != 'All':
                    if state and state != 'All':
                        # Show monthly distribution for selected state
                        state_data = filtered_df2[filtered_df2['States'] == state]
                        
                        # Create side-by-side box plots for demand and shortage
                        plot_data = pd.melt(state_data, 
                                          id_vars=['month_name'],
                                          value_vars=['Max.Demand Met during the day(MW)', 
                                                    'Shortage during maximum Demand(MW)'],
                                          var_name='Metric', value_name='Power')
                        
                        sns.boxplot(data=plot_data, x='month_name', y='Power', hue='Metric', ax=ax)
                        ax.set_title(f'Monthly Power Distribution - {source} in {state}')
                        ax.set_xlabel('Month')
                        ax.set_ylabel('Power (MW)')
                        plt.xticks(rotation=45)
                    else:
                        # Compare distributions across states
                        state_data = df2.copy()
                        if year != 'All':
                            state_data = state_data[state_data['year'] == int(year)]
                        
                        # Get top 10 states by average total power
                        state_avgs = state_data.groupby('States').agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        })
                        state_avgs['Total'] = state_avgs['Max.Demand Met during the day(MW)'] + \
                                            state_avgs['Shortage during maximum Demand(MW)']
                        top_states = state_avgs.nlargest(10, 'Total').index
                        
                        # Filter data for top states
                        plot_data = state_data[state_data['States'].isin(top_states)]
                        plot_data = pd.melt(plot_data,
                                          id_vars=['States'],
                                          value_vars=['Max.Demand Met during the day(MW)',
                                                    'Shortage during maximum Demand(MW)'],
                                          var_name='Metric', value_name='Power')
                        
                        sns.boxplot(data=plot_data, x='States', y='Power', hue='Metric', ax=ax)
                        ax.set_title(f'Power Distribution by State - {source}')
                        ax.set_xlabel('State')
                        ax.set_ylabel('Power (MW)')
                        plt.xticks(rotation=45, ha='right')
                else:
                    # Original box plot logic
                    if state and state != 'All':
                        state_data = filtered_df2[filtered_df2['States'] == state]
                        sns.boxplot(data=state_data, x='month_name', y='Max.Demand Met during the day(MW)', ax=ax)
                        ax.set_title(f'Distribution of Power Demand in {state}')
                        ax.set_ylabel('Power Demand (MW)')
                    else:
                        if region and region != 'All':
                            sns.boxplot(data=filtered_df1, x='source', y=region, ax=ax)
                            ax.set_title(f'Distribution of Power Generation by Source in {region}')
                        else:
                            sns.boxplot(data=filtered_df1, x='source', y='All India', ax=ax)
                            ax.set_title('Distribution of Power Generation by Source')
                
                ax.set_xlabel('Time Period' if state != 'All' else 'Source')
                if ax.get_legend():
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
            
            elif graph_type == 'heatmap':
                if source and source != 'All':
                    if state and state != 'All':
                        # Show monthly-yearly heatmap for selected state
                        state_data = filtered_df2[filtered_df2['States'] == state]
                        
                        # Create two heatmaps side by side
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Demand Met heatmap
                        pivot_demand = state_data.pivot_table(
                            index='month_name',
                            columns='year',
                            values='Max.Demand Met during the day(MW)',
                            aggfunc='mean'
                        )
                        sns.heatmap(pivot_demand, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
                        ax1.set_title('Demand Met (MW)')
                        
                        # Shortage heatmap
                        pivot_shortage = state_data.pivot_table(
                            index='month_name',
                            columns='year',
                            values='Shortage during maximum Demand(MW)',
                            aggfunc='mean'
                        )
                        sns.heatmap(pivot_shortage, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2)
                        ax2.set_title('Shortage (MW)')
                        
                        plt.suptitle(f'Power Analysis Heatmap - {source} in {state}\n({year if year != "All" else "All Years"})')
                    else:
                        # Create state-wise monthly heatmap
                        state_data = df2.copy()
                        if year != 'All':
                            state_data = state_data[state_data['year'] == int(year)]
                        
                        # Get top 15 states by average total power
                        state_avgs = state_data.groupby('States').agg({
                            'Max.Demand Met during the day(MW)': 'mean',
                            'Shortage during maximum Demand(MW)': 'mean'
                        })
                        state_avgs['Total'] = state_avgs['Max.Demand Met during the day(MW)'] + \
                                            state_avgs['Shortage during maximum Demand(MW)']
                        top_states = state_avgs.nlargest(15, 'Total').index
                        
                        # Create two heatmaps side by side
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
                        
                        # Demand Met heatmap
                        pivot_demand = state_data[state_data['States'].isin(top_states)].pivot_table(
                            index='States',
                            columns='month_name',
                            values='Max.Demand Met during the day(MW)',
                            aggfunc='mean'
                        )
                        sns.heatmap(pivot_demand, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
                        ax1.set_title('Average Demand Met (MW)')
                        
                        # Shortage heatmap
                        pivot_shortage = state_data[state_data['States'].isin(top_states)].pivot_table(
                            index='States',
                            columns='month_name',
                            values='Shortage during maximum Demand(MW)',
                            aggfunc='mean'
                        )
                        sns.heatmap(pivot_shortage, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2)
                        ax2.set_title('Average Shortage (MW)')
                        
                        plt.suptitle(f'State-wise Monthly Power Analysis - {source}\n({year if year != "All" else "All Years"})')
                else:
                    # Original heatmap logic
                    if state and state != 'All':
                        state_data = filtered_df2[filtered_df2['States'] == state]
                        pivot_df = state_data.pivot_table(
                            index='month_name',
                            columns='year',
                            values='Max.Demand Met during the day(MW)',
                            aggfunc='mean'
                        )
                        sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
                        ax.set_title(f'Monthly Power Demand Heatmap for {state}')
                    else:
                        pivot_df = filtered_df1.pivot_table(
                            index='month_name',
                            columns='year',
                            values='All India',
                            aggfunc='mean'
                        )
                        sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
                        ax.set_title('Monthly Power Generation Heatmap for All India')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            elif graph_type == 'prediction':
                # Add prediction generation
                if state and state != 'All':
                    y = filtered_df2['Max.Demand Met during the day(MW)']
                    X = filtered_df2[['Energy Met (MW)', 'Shortage during maximum Demand(MW)']]
                    title = f"Energy Demand Prediction for {state}"
                    data_description = f"state {state}"
                elif region and region != 'All':
                    y = filtered_df1[region]
                    X = filtered_df1[['day', 'month', 'year']]
                    title = f"Energy Demand Prediction for {region} Region"
                    data_description = f"{region} region"
                else:
                    y = filtered_df1['All India']
                    X = filtered_df1[['day', 'month', 'year']]
                    title = "All India Energy Demand Prediction"
                    data_description = "all India"
                
                train_size = int(len(y) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
                    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
                    'Holt-Winters': Holt(y_train)
                }
                
                predictions = {}
                model_metrics = {}
                
                for name, model in models.items():
                    try:
                        if name == 'Holt-Winters':
                            fit = model.fit(smoothing_level=0.8, smoothing_trend=0.02)
                            pred = fit.forecast(len(y_test))
                        else:
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)
                        
                        predictions[name] = pred
                        model_metrics[name] = calculate_metrics(y_test, pred)
                    except Exception as e:
                        warnings.warn(f"Error fitting {name}: {str(e)}")
                        continue
                
                # Plot results
                y_train.plot(color="blue", label='Train', ax=ax)
                y_test.plot(color="pink", label='Test', ax=ax)
                
                colors = ['green', 'red', 'purple', 'orange']
                for i, (name, pred) in enumerate(predictions.items()):
                    ax.plot(y_test.index, pred, color=colors[i], linestyle='--', 
                           label=f'{name} ({model_metrics[name]["rating"]})')
                
                ax.set_title(title)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_ylabel('Power Demand (MW)')
                ax.set_xlabel('Time Index')
                ax.set_ylim(bottom=0, top=max(y.max(), max(pred.max() for pred in predictions.values())) * 1.2)
                
                best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
                explanation = f"""
                <div class="alert alert-info">
                    <h5><i class="bi bi-lightbulb"></i> Prediction Analysis for {data_description}</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h6 class="card-title"><i class="bi bi-trophy-fill text-warning"></i> Best Model</h6>
                                    <div class="d-flex align-items-center">
                                        <h4 class="mb-0">{best_model[0]}</h4>
                                        <span class="badge bg-{best_model[1]['rating_color']} ms-2">{best_model[1]['rating']}</span>
                                    </div>
                                    <p class="mb-0 small text-muted">Accuracy: {best_model[1]['accuracy']:.1f}%</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h6 class="card-title"><i class="bi bi-info-circle"></i> Model Guide</h6>
                                    <ul class="small mb-0">
                                        <li>90-100%: Excellent</li>
                                        <li>75-89%: Good</li>
                                        <li>50-74%: Fair</li>
                                        <li>Below 50%: Poor</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="mt-2">
                        <p class="mb-1"><strong>Analysis Notes:</strong></p>
                        <ul class="mb-0">
                            <li>Training period: {len(y_train)} days</li>
                            <li>Test period: {len(y_test)} days</li>
                            <li>Dashed lines show model predictions vs actual (pink)</li>
                        </ul>
                    </div>
                </div>
                """
            
            # Save the static plot
            fig.savefig(png_path, format='png', bbox_inches='tight', dpi=100)
            
            # Generate base64 for immediate display
            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close(fig)
            
            # Save to user history
            save_user_history(
                current_user.id, 
                graph_type, 
                {
                    'source': source,
                    'region': region,
                    'year': year,
                    'month': month,
                    'state': state,
                    'interactive': False
                },
                relative_png_path.replace('\\', '/')  # Ensure correct path format
            )
            
            return jsonify({
                'image': plot_url,
                'model_metrics': model_metrics,
                'explanation': explanation,
                'interactive': False,
                'saved_path': url_for('static', filename=relative_png_path.replace('\\', '/'))  # Ensure correct path format
            })
    
    except Exception as e:
        app.logger.error(f"Error generating graph: {str(e)}")
        return jsonify({
            'error': f"""
            <div class="alert alert-danger">
                <h5><i class="bi bi-exclamation-triangle-fill"></i> Error Generating Visualization</h5>
                <p><strong>Error Details:</strong> {str(e)}</p>
                <p class="mb-0"><strong>Suggested Action:</strong> Try adjusting your filters or select a different visualization type.</p>
            </div>
            """
        }), 400

@app.route('/check_plotly')
def check_plotly():
    return jsonify({'plotly_available': check_plotly_available()})

@app.route('/generate_india_graphs', methods=['GET'])
@login_required
def generate_india_graphs():
    """Generate and save graphs for India-wide analysis."""
    try:
        # Line plot: Power Generation by Source over Time
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df1_source, x='date', y='All India', hue='source')
        plt.title('Power Generation by Source over Time')
        plt.xlabel('Date')
        plt.ylabel('Power Generation (MW)')
        line_plot_path = os.path.join(app.config['STATIC_FOLDER'], 'line_plot.png')
        plt.savefig(line_plot_path)
        plt.close()

        # Bar plot: Power Generation by Source
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df1_source, y='source', x='All India')
        plt.title('Power Generation by Source')
        plt.ylabel('Source')
        plt.xlabel('Power Generation (MW)')
        bar_plot_path = os.path.join(app.config['STATIC_FOLDER'], 'bar_plot.png')
        plt.savefig(bar_plot_path)
        plt.close()

        # Pie chart: Total Energy Production by Source
        df1_source_grouped = df1_source.groupby('source')['All India'].sum()
        plt.figure(figsize=(8, 8))
        plt.pie(df1_source_grouped, labels=df1_source_grouped.index, autopct='%1.1f%%')
        plt.title('Total Energy Production by Source (2013-2023)')
        pie_chart_path = os.path.join(app.config['STATIC_FOLDER'], 'pie_chart.png')
        plt.savefig(pie_chart_path)
        plt.close()

        # Box plot: Energy Demand by Source and Region (e.g., NR)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='source', y='NR', data=df1_source)
        plt.xlabel('Source')
        plt.ylabel('Energy Demand (MW)')
        plt.title('Energy Demand by Source and Region NR')
        box_plot_path = os.path.join(app.config['STATIC_FOLDER'], 'box_plot.png')
        plt.savefig(box_plot_path)
        plt.close()

        # Time-series decomposition: Seasonal Decompose for 'NR'
        series = df1['NR']
        result = seasonal_decompose(series, model='additive', period=12)
        result.plot()
        ts_decompose_path = os.path.join(app.config['STATIC_FOLDER'], 'ts_decompose.png')
        plt.savefig(ts_decompose_path)
        plt.close()

        return jsonify({
            'line_plot': url_for('static', filename='line_plot.png'),
            'bar_plot': url_for('static', filename='bar_plot.png'),
            'pie_chart': url_for('static', filename='pie_chart.png'),
            'box_plot': url_for('static', filename='box_plot.png'),
            'ts_decompose': url_for('static', filename='ts_decompose.png')
        })
    except Exception as e:
        app.logger.error(f"Error generating India graphs: {str(e)}")
        return jsonify({'error': 'Failed to generate graphs. Please try again later.'}), 500

# Add this function after the data loading section
def get_states_for_source(source):
    """Get list of states that have data for the selected source"""
    try:
        # Return all states if source is 'All'
        if source == 'All':
            states = df2['States'].dropna().unique().tolist()
            app.logger.info(f"Returning all {len(states)} states for 'All' sources")
            return sorted(states)
        
        # Ensure source exists in the data
        if source not in df1_source['source'].unique():
            app.logger.warning(f"Source '{source}' not found in data")
            return []
        
        # Get states that have data for the selected source
        source_data = df1_source[df1_source['source'] == source]
        if source_data.empty:
            app.logger.warning(f"No data found for source: {source}")
            return []
        
        # Get all dates for this source
        source_dates = pd.to_datetime(source_data['date'].unique())
        
        # Filter df2 for these dates and get unique states
        mask = df2['date'].isin(source_dates)
        states_with_data = df2[mask]['States'].dropna().unique().tolist()
        
        app.logger.info(f"Found {len(states_with_data)} states with data for source {source}")
        return sorted(states_with_data)
    
    except Exception as e:
        app.logger.error(f"Error in get_states_for_source: {str(e)}")
        return []

@app.route('/get_states', methods=['POST'])
def get_states():
    """AJAX endpoint to get states for selected source"""
    try:
        source = request.form.get('source')
        if not source:
            app.logger.error("No source parameter provided")
            return jsonify({'error': 'Source parameter is required'}), 400
        
        app.logger.info(f"Fetching states for source: {source}")
        states = get_states_for_source(source)
        
        if not states:
            app.logger.warning(f"No states found for source: {source}")
            return jsonify({'states': [], 'message': f'No states available for {source}'}), 200
        
        app.logger.info(f"Successfully found {len(states)} states for source {source}")
        return jsonify({'states': states})
    
    except Exception as e:
        app.logger.error(f"Error processing get_states request: {str(e)}")
        return jsonify({'error': f'Failed to fetch states: {str(e)}'}), 500

if __name__ == '__main__':
    # Create necessary folders on startup
    os.makedirs(os.path.join(app.config['STATIC_FOLDER'], app.config['USER_GRAPHS_FOLDER']), exist_ok=True)
    app.run(debug=True)