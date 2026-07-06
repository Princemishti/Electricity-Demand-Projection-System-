import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

def get_transparent_layout():
    """Returns layout settings to blend transparently with dark/light background stylesheets."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif', color='#6c757d'),
        xaxis=dict(
            gridcolor='rgba(108, 117, 125, 0.1)',
            linecolor='rgba(108, 117, 125, 0.2)',
            tickfont=dict(color='#6c757d')
        ),
        yaxis=dict(
            gridcolor='rgba(108, 117, 125, 0.1)',
            linecolor='rgba(108, 117, 125, 0.2)',
            tickfont=dict(color='#6c757d')
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )

def create_timeline_chart(hist_df, forecast_dates, forecast_values, lower_bounds, upper_bounds, state, source):
    """
    Creates a detailed timeline chart with historical demand, forecasted demand,
    and a shaded 95% confidence interval area.
    """
    fig = go.Figure()
    
    # 1. Shaded confidence interval (Lower bound to Upper bound)
    # Plot upper bound first (transparent line)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=upper_bounds,
        mode='lines',
        line=dict(width=0),
        name='Upper Confidence (95%)',
        showlegend=False
    ))
    
    # Plot lower bound and fill to upper bound
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=lower_bounds,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.15)',
        name='95% Confidence Interval'
    ))
    
    # 2. Historical demand trace (last 36 months to avoid visual clutter)
    hist_subset = hist_df.tail(36)
    fig.add_trace(go.Scatter(
        x=hist_subset['date'].dt.strftime('%Y-%m-%d').tolist(),
        y=hist_subset['demand'].tolist(),
        mode='lines+markers',
        line=dict(color='#2c3e50', width=2.5),
        marker=dict(size=4),
        name='Historical (Last 3Y)'
    ))
    
    # 3. Forecasted demand trace
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines+markers',
        line=dict(color='#3498db', width=2.5, dash='dash'),
        marker=dict(size=4),
        name='Predicted Forecast'
    ))
    
    fig.update_layout(
        title=f"Demand Forecasting Timeline - {state} ({source})",
        xaxis_title="Timeline",
        yaxis_title="Demand (MU)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **get_transparent_layout()
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_model_comparison_chart(metrics_dict, best_model_name):
    """
    Creates a bar chart comparing test-set accuracies (100 - MAPE)
    for each trained model, highlighting the best performer.
    """
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    
    # Colors: blue for other models, gold/emerald for the best model
    colors = ['#2ecc71' if m == best_model_name else '#3498db' for m in models]
    
    fig = go.Figure(go.Bar(
        x=models,
        y=accuracies,
        marker_color=colors,
        text=[f"{acc:.1f}%" for acc in accuracies],
        textposition='auto',
        hovertemplate="Model: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Model Accuracy Comparison (100 - MAPE)",
        xaxis_title="Model Type",
        yaxis_title="Validation Accuracy (%)",
        **get_transparent_layout()
    )
    
    fig.update_layout(yaxis=dict(range=[0, 105]))

    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_yearly_comparison_chart(hist_df, forecast_dates, forecast_values):
    """
    Creates a bar chart comparing annual total consumption,
    including historical years and the future projected year.
    """
    # Group history by year
    hist_yearly = hist_df.copy()
    hist_yearly['year'] = hist_yearly['date'].dt.year
    hist_y_grouped = hist_yearly.groupby('year')['demand'].sum().reset_index()
    
    # Group forecast by year
    fore_df = pd.DataFrame({
        'date': pd.to_datetime(forecast_dates),
        'demand': forecast_values
    })
    fore_df['year'] = fore_df['date'].dt.year
    fore_y_grouped = fore_df.groupby('year')['demand'].sum().reset_index()
    
    # Merge and build list
    years = []
    values = []
    types = []
    
    for _, row in hist_y_grouped.iterrows():
        years.append(str(int(row['year'])))
        values.append(row['demand'])
        types.append('Historical')
        
    for _, row in fore_y_grouped.iterrows():
        # If year already exists, we combine it (e.g. 2023 has both historical and predicted)
        yr_str = str(int(row['year']))
        if yr_str in years:
            idx = years.index(yr_str)
            values[idx] += row['demand']
            types[idx] = 'Historical + Forecast'
        else:
            years.append(yr_str)
            values.append(row['demand'])
            types.append('Forecasted')
            
    df_chart = pd.DataFrame({
        'Year': years,
        'Demand': values,
        'Type': types
    })
    
    fig = px.bar(
        df_chart,
        x='Year',
        y='Demand',
        color='Type',
        color_discrete_map={
            'Historical': '#2c3e50',
            'Historical + Forecast': '#16a085',
            'Forecasted': '#3498db'
        },
        title="Year-over-Year Energy Demand Comparison"
    )
    
    fig.update_layout(
        yaxis_title="Total Energy (MU)",
        **get_transparent_layout()
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_monthly_trend_chart(hist_df):
    """
    Creates a line chart illustrating seasonal trends (demand by month).
    """
    hist_df = hist_df.copy()
    hist_df['month'] = hist_df['date'].dt.month_name()
    
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    grouped = hist_df.groupby('month')['demand'].mean().reindex(month_order).reset_index()
    
    fig = go.Figure(go.Scatter(
        x=grouped['month'],
        y=grouped['demand'],
        mode='lines+markers',
        line=dict(color='#e67e22', width=3),
        marker=dict(size=6, color='#d35400'),
        hovertemplate="Month: %{x}<br>Average Demand: %{y:.1f} MU<extra></extra>"
    ))
    
    fig.update_layout(
        title="Average Monthly Seasonality Pattern",
        xaxis_title="Month",
        yaxis_title="Average Demand (MU)",
        **get_transparent_layout()
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_state_comparison_chart(selected_state, selected_demand_mean, df2_data):
    """
    Creates a comparison chart of the selected state's average demand
    compared with top consuming states in India.
    """
    # Get top 5 states by average monthly demand
    state_avgs = df2_data.groupby('States')['Energy Met (MU)'].mean().reset_index()
    state_avgs.rename(columns={'Energy Met (MU)': 'demand'}, inplace=True)
    state_avgs = state_avgs.sort_values('demand', ascending=False)
    
    top_5 = state_avgs.head(5).copy()
    
    # Check if selected state is already in top 5
    if selected_state not in top_5['States'].values and selected_state != 'All India' and selected_state != 'All':
        # Append selected state
        state_row = pd.DataFrame({
            'States': [selected_state],
            'demand': [selected_demand_mean]
        })
        top_5 = pd.concat([top_5, state_row], ignore_index=True)
        
    colors = ['#e74c3c' if s == selected_state else '#95a5a6' for s in top_5['States']]
    
    fig = go.Figure(go.Bar(
        x=top_5['States'],
        y=top_5['demand'],
        marker_color=colors,
        hovertemplate="State: %{x}<br>Average demand: %{y:.1f} MU<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"State Consumption Ranking Comparison",
        xaxis_title="States",
        yaxis_title="Average Monthly demand (MU)",
        **get_transparent_layout()
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_source_contribution_chart(df1_data):
    """
    Creates a donut pie chart showing contribution of different power sources
    nationally on the latest historical date.
    """
    # Filter for latest date
    latest_date = df1_data['date'].max()
    latest_df = df1_data[df1_data['date'] == latest_date].copy()
    
    # Sum All India by source
    grouped = latest_df.groupby('source')['All India'].sum().reset_index()
    grouped.rename(columns={'All India': 'generation'}, inplace=True)
    
    # Filter out total or zero generation
    grouped = grouped[grouped['generation'] > 0]
    
    fig = go.Figure(go.Pie(
        labels=grouped['source'],
        values=grouped['generation'],
        hole=.4,
        hovertemplate="Source: %{label}<br>Gen Share: %{percent}<br>Value: %{value:.1f} MU<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Power Grid Generation Source Share",
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        **get_transparent_layout()
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
