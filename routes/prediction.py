import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, send_file
from flask_login import login_required, current_user

# Service Imports
from services.data_service import get_monthly_series, states_list, sources_list, df2, df1_source
from services.ml_service import train_and_evaluate, forecast_future
from services.insights_service import generate_dynamic_insights
from services.report_service import generate_pdf_report

# DB Imports
from utils.db import save_prediction_history, get_prediction_history

# Chart Imports
from utils.charts import (
    create_timeline_chart,
    create_model_comparison_chart,
    create_yearly_comparison_chart,
    create_monthly_trend_chart,
    create_state_comparison_chart,
    create_source_contribution_chart
)

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/prediction')
@login_required
def prediction_page():
    """Renders the Future Demand Prediction Module Page."""
    # Get available states and sources
    states = states_list
    sources = sources_list
    
    # Selected years and months for custom selection (project objective range 2013-2023)
    # But predictions project forward, e.g., 2023, 2024, 2025, 2026, 2027
    years = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
    months = list(range(1, 13))
    
    # Get recent predictions history to display
    history = get_prediction_history(current_user.id, limit=10)
    
    return render_template(
        'prediction.html',
        states=states,
        sources=sources,
        years=years,
        months=months,
        history=history,
        current_user=current_user
    )

@prediction_bp.route('/model-performance')
@login_required
def model_performance_page():
    """Renders the Model Performance Page."""
    return render_template(
        'model_performance.html',
        states=states_list,
        sources=sources_list,
        current_user=current_user
    )

@prediction_bp.route('/api/model-performance', methods=['GET'])
@login_required
def api_model_performance():
    """Endpoint for retrieving model performance metrics on-the-fly for selected state and source."""
    state = request.args.get('state', 'All India')
    source = request.args.get('power_source', 'All Sources')
    
    # Get series
    df_monthly = get_monthly_series(state, source)
    
    # Run evaluation
    perf, best_model, info, df_out, acc = train_and_evaluate(df_monthly, state, source)
    
    # Generate model accuracy comparison chart
    chart_json = create_model_comparison_chart(perf, best_model)
    
    # Format times and floats
    formatted_perf = {}
    for name, stats in perf.items():
        formatted_perf[name] = {
            'accuracy': f"{stats['accuracy']:.2f}%",
            'mae': f"{stats['mae']:,.2f} MU",
            'mse': f"{stats['mse']:,.2f} MU²",
            'rmse': f"{stats['rmse']:,.2f} MU",
            'mape': f"{stats['mape']:.2f}%",
            'r2': f"{stats['r2']:.4f}",
            'training_time': f"{stats['training_time'] * 1000:.1f} ms",
            'prediction_time': f"{stats['prediction_time'] * 1000:.1f} ms"
        }
        
    return jsonify({
        'metrics': formatted_perf,
        'best_model': best_model,
        'chart': chart_json
    })

@prediction_bp.route('/api/train', methods=['POST'])
@login_required
def api_train():
    """Direct model training trigger."""
    state = request.form.get('state', 'All India')
    source = request.form.get('power_source', 'All Sources')
    
    df_monthly = get_monthly_series(state, source)
    perf, best_model, info, df_out, acc = train_and_evaluate(df_monthly, state, source)
    
    return jsonify({
        'status': 'success',
        'message': f"Successfully trained models for {state} - {source}!",
        'best_model': best_model,
        'accuracy': f"{acc:.2f}%"
    })

@prediction_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """Performs electricity demand predictions based on selected filters, stores the history, and returns charts."""
    state = request.form.get('state', 'All India')
    power_source = request.form.get('power_source', 'All Sources')
    forecast_period = request.form.get('forecast_period', 'Next Month')
    
    custom_year = request.form.get('year')
    custom_month = request.form.get('month')
    
    # Load monthly time-series
    df_monthly = get_monthly_series(state, power_source)
    if df_monthly.empty or len(df_monthly) < 3:
        return jsonify({'error': 'Insufficient historical data for selected state and source combination.'}), 400
        
    # Last historical date
    last_hist_row = df_monthly.iloc[-1]
    last_hist_date = last_hist_row['date']
    last_hist_val = last_hist_row['demand']
    
    # Calculate forecast steps
    if forecast_period == 'Next Month':
        steps = 1
        target_year = (last_hist_date.year if last_hist_date.month < 12 else last_hist_date.year + 1)
        target_month = (last_hist_date.month + 1 if last_hist_date.month < 12 else 1)
    elif forecast_period == 'Next 3 Months':
        steps = 3
        future_date = last_hist_date + pd.DateOffset(months=3)
        target_year = future_date.year
        target_month = future_date.month
    elif forecast_period == 'Next 6 Months':
        steps = 6
        future_date = last_hist_date + pd.DateOffset(months=6)
        target_year = future_date.year
        target_month = future_date.month
    elif forecast_period == 'Next Year':
        steps = 12
        future_date = last_hist_date + pd.DateOffset(months=12)
        target_year = future_date.year
        target_month = future_date.month
    elif forecast_period == 'Custom':
        try:
            target_year = int(custom_year)
            target_month = int(custom_month)
            
            # calculate steps chronologically from last_hist_date (March 2023)
            steps = (target_year - last_hist_date.year) * 12 + (target_month - last_hist_date.month)
            if steps <= 0:
                return jsonify({'error': 'Target date must be in the future (after March 2023).'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid target year or month values.'}), 400
    else:
        steps = 1
        target_year = 2023
        target_month = 4
        
    # Generate future dates
    forecast_dates = pd.date_range(
        start=last_hist_date + pd.DateOffset(months=1),
        periods=steps,
        freq='ME'
    ).strftime('%Y-%m-%d').tolist()
    
    # Train models and find best one
    perf, best_model, info, df_out, acc = train_and_evaluate(df_monthly, state, power_source)
    
    # Generate future forecast
    forecast_vals, lower_bounds, upper_bounds = forecast_future(info, steps)
    
    # Predicted demand at target month
    predicted_val = forecast_vals[-1]
    confidence_val = acc
    
    # Calculate growth from last historical value
    if last_hist_val > 0:
        growth = (predicted_val - last_hist_val) / last_hist_val * 100
    else:
        growth = 0.0
        
    growth_direction = "+" if growth >= 0 else ""
    
    # Calculate trend text
    if growth > 2.0:
        trend = "Increasing"
    elif growth < -2.0:
        trend = "Decreasing"
    else:
        trend = "Stable"
        
    # Identify Peak and Lowest month in forecasted timeline
    month_names = [datetime.strptime(d, "%Y-%m-%d").strftime("%B") for d in forecast_dates]
    peak_idx = np.argmax(forecast_vals)
    lowest_idx = np.argmin(forecast_vals)
    
    peak_month = month_names[peak_idx]
    lowest_month = month_names[lowest_idx]
    
    # Save prediction history to database
    new_id = save_prediction_history(
        user_id=current_user.id,
        state=state,
        year=target_year,
        month=target_month,
        power_source=power_source,
        forecast_period=forecast_period,
        predicted_demand=predicted_val,
        confidence=confidence_val,
        expected_growth=growth,
        trend=trend,
        best_model_used=best_model,
        accuracy=acc,
        metrics=perf
    )
    
    # Generate Plotly charts
    timeline_chart = create_timeline_chart(
        df_monthly, forecast_dates, forecast_vals, lower_bounds, upper_bounds, state, power_source
    )
    model_comparison_chart = create_model_comparison_chart(perf, best_model)
    yearly_comparison_chart = create_yearly_comparison_chart(df_monthly, forecast_dates, forecast_vals)
    monthly_trend_chart = create_monthly_trend_chart(df_monthly)
    
    # State rank comparison: Selected vs Top Consuming
    selected_mean = df_monthly['demand'].mean()
    state_comparison_chart = create_state_comparison_chart(state, selected_mean, df2)
    
    # Source share contribution
    source_contribution_chart = create_source_contribution_chart(df1_source)
    
    # Generate dynamic insights
    insights = generate_dynamic_insights(state, power_source, predicted_val, growth, forecast_period)
    
    # Format and pack result
    return jsonify({
        'id': new_id,
        'predicted_demand': f"{predicted_val:,.0f} MU",
        'confidence': f"{confidence_val:.0f}%",
        'expected_growth': f"{growth_direction}{growth:.1f}%",
        'trend': trend,
        'peak_month': peak_month,
        'lowest_month': lowest_month,
        'explanation': "This prediction is based on historical electricity generation patterns between 2013 and 2023.",
        'insights': insights,
        'metrics': perf,
        'best_model': best_model,
        'charts': {
            'timeline': timeline_chart,
            'model_comparison': model_comparison_chart,
            'yearly_comparison': yearly_comparison_chart,
            'monthly_trend': monthly_trend_chart,
            'state_comparison': state_comparison_chart,
            'source_contribution': source_contribution_chart
        }
    })

@prediction_bp.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """Retrieve user's forecast prediction history."""
    history = get_prediction_history(current_user.id, limit=20)
    # Format times for JSON delivery
    for item in history:
        if isinstance(item['prediction_time'], datetime):
            item['prediction_time'] = item['prediction_time'].strftime('%Y-%m-%d %H:%M')
        elif isinstance(item['prediction_time'], str):
            # Already a string
            pass
            
    return jsonify({'history': history})

@prediction_bp.route('/api/download-report')
@login_required
def download_report():
    """Generates and downloads the PDF forecast report."""
    prediction_id = request.args.get('id')
    
    if not prediction_id:
        return "Prediction ID is required.", 400
        
    # Get prediction parameters by ID from database
    history_list = get_prediction_history(current_user.id, limit=100)
    prediction = None
    for h in history_list:
        if str(h['id']) == str(prediction_id):
            prediction = h
            break
            
    if not prediction:
        return "Prediction record not found or access denied.", 404
        
    # Extract prediction filters
    state = prediction['state']
    source = prediction['power_source']
    forecast_period = prediction['forecast_period']
    target_year = prediction['year']
    target_month = prediction['month']
    best_model = prediction['best_model_used']
    accuracy = prediction['accuracy']
    metrics = prediction['metrics']
    predicted_val = prediction['predicted_demand']
    growth = prediction['expected_growth']
    trend = prediction['trend']
    
    # Re-run aggregation and prediction to reconstruct plotting dates and forecast curves
    df_monthly = get_monthly_series(state, source)
    last_hist_row = df_monthly.iloc[-1]
    last_hist_date = last_hist_row['date']
    
    # Calculate steps
    steps = (target_year - last_hist_date.year) * 12 + (target_month - last_hist_date.month)
    steps = max(1, steps)
    
    # Generate dates
    forecast_dates = pd.date_range(
        start=last_hist_date + pd.DateOffset(months=1),
        periods=steps,
        freq='ME'
    ).strftime('%Y-%m-%d').tolist()
    
    # Run ML forecast
    perf, best_model_ref, info, df_out, acc_ref = train_and_evaluate(df_monthly, state, source)
    forecast_vals, lower_bounds, upper_bounds = forecast_future(info, steps)
    
    # Re-evaluate peak/lowest
    month_names = [datetime.strptime(d, "%Y-%m-%d").strftime("%B") for d in forecast_dates]
    peak_month = month_names[np.argmax(forecast_vals)]
    lowest_month = month_names[np.argmin(forecast_vals)]
    
    # Re-calculate insights
    insights = generate_dynamic_insights(state, source, predicted_val, growth, forecast_period)
    
    # Generate PDF report
    pdf_buffer = generate_pdf_report(
        state=state,
        source=source,
        forecast_period=forecast_period,
        target_year=target_year,
        target_month=target_month,
        best_model=best_model,
        accuracy=accuracy,
        metrics=metrics,
        predicted_value=predicted_val,
        growth=growth,
        trend=trend,
        peak_month=peak_month,
        lowest_month=lowest_month,
        insights=insights,
        history_dates=df_monthly['date'].dt.strftime('%Y-%m-%d').tolist(),
        history_values=df_monthly['demand'].tolist(),
        forecast_dates=forecast_dates,
        forecast_values=forecast_vals,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds
    )
    
    filename = f"Electricity_Forecast_Report_{state.replace(' ', '_')}_{source.replace(' ', '_')}.pdf"
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )
