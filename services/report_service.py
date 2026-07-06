import os
import io
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_pdf_report(state, source, forecast_period, target_year, target_month, best_model, accuracy, metrics, predicted_value, growth, trend, peak_month, lowest_month, insights, history_dates, history_values, forecast_dates, forecast_values, lower_bounds, upper_bounds):
    """
    Generates a professional PDF report containing prediction results, model metrics, charts and AI insights.
    Returns: BytesIO object containing the PDF data.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#2c3e50")
    secondary_color = colors.HexColor("#3498db")
    accent_color = colors.HexColor("#e74c3c")
    light_bg = colors.HexColor("#f8f9fa")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=5
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=15
    )
    
    heading2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=8,
        borderColor=secondary_color,
        borderWidth=0.5,
        borderPadding=(0, 0, 2, 0)
    )
    
    normal_style = styles['Normal']
    
    # Add Header
    story.append(Paragraph("India Power Analytics - Forecast Report", title_style))
    story.append(Paragraph(f"Electricity Demand Projection System &bull; Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 5))
    
    # Metadata Table
    metadata_data = [
        [Paragraph("<b>Target State:</b>", normal_style), Paragraph(state, normal_style),
         Paragraph("<b>Power Source:</b>", normal_style), Paragraph(source, normal_style)],
        [Paragraph("<b>Forecast Period:</b>", normal_style), Paragraph(forecast_period, normal_style),
         Paragraph("<b>Target Horizon:</b>", normal_style), Paragraph(f"{target_month}/{target_year}", normal_style)]
    ]
    meta_table = Table(metadata_data, colWidths=[1.5*inch, 1.8*inch, 1.5*inch, 1.8*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ecf0f1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))
    
    # Predictions Summary Grid
    story.append(Paragraph("Future Projection Summary", heading2_style))
    growth_str = f"+{growth:.1f}%" if growth >= 0 else f"{growth:.1f}%"
    summary_data = [
        [Paragraph("<b>Predicted Demand</b>", normal_style), Paragraph("<b>Expected Growth</b>", normal_style), Paragraph("<b>Trend</b>", normal_style)],
        [Paragraph(f"<font size=14 color='{secondary_color.hexval()}'><b>{predicted_value:,.1f} MU</b></font>", normal_style),
         Paragraph(f"<font size=12 color='{accent_color.hexval() if growth < 0 else '#27ae60'}'><b>{growth_str}</b></font>", normal_style),
         Paragraph(f"<b>{trend}</b>", normal_style)],
        [Paragraph("<b>Best Model Used</b>", normal_style), Paragraph("<b>Model Accuracy</b>", normal_style), Paragraph("<b>Peak / Lowest Month</b>", normal_style)],
        [Paragraph(best_model, normal_style), Paragraph(f"{accuracy:.1f}%", normal_style), Paragraph(f"{peak_month} / {lowest_month}", normal_style)]
    ]
    summary_table = Table(summary_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
    summary_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ecf0f1")),
        ('BACKGROUND', (0,0), (-1,0), light_bg),
        ('BACKGROUND', (0,2), (-1,2), light_bg),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))
    
    # Draw chart and insert it
    plt.figure(figsize=(6.5, 3))
    
    # Parse dates
    hist_dates = [pd.to_datetime(d) for d in history_dates]
    fore_dates = [pd.to_datetime(d) for d in forecast_dates]
    
    # Plot last 3 years of history to avoid cluttering the PDF
    hist_len = min(36, len(hist_dates))
    plt.plot(hist_dates[-hist_len:], history_values[-hist_len:], color="#2c3e50", label="Historical")
    plt.plot(fore_dates, forecast_values, color="#3498db", linestyle="--", label="Forecast")
    plt.fill_between(fore_dates, lower_bounds, upper_bounds, color="#3498db", alpha=0.15, label="95% Confidence")
    plt.title(f"Demand Projection: {state} ({source})", fontsize=10)
    plt.xlabel("Timeline", fontsize=8)
    plt.ylabel("Demand (MU)", fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close()
    
    chart_image = Image(img_buf, width=6.5*inch, height=3.0*inch)
    story.append(Paragraph("Forecast Visualisation", heading2_style))
    story.append(chart_image)
    story.append(Spacer(1, 10))
    
    # Model Comparison Table
    story.append(Paragraph("Forecasting Model Comparison", heading2_style))
    comparison_data = [
        ["Model Name", "Accuracy", "MAE", "RMSE", "MAPE"]
    ]
    for m_name, m_stats in metrics.items():
        is_best = m_name == best_model
        acc_str = f"<b>{m_stats['accuracy']:.1f}%</b>" if is_best else f"{m_stats['accuracy']:.1f}%"
        mae_val = f"{m_stats['mae']:.1f}"
        rmse_val = f"{m_stats['rmse']:.1f}"
        mape_val = f"{m_stats['mape']:.1f}%"
        
        comparison_data.append([
            f"<b>{m_name} (Best)</b>" if is_best else m_name,
            Paragraph(acc_str, normal_style),
            mae_val,
            rmse_val,
            mape_val
        ])
    
    comp_table = Table(comparison_data, colWidths=[2.2*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ecf0f1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 10))
    
    # AI Insights Section
    story.append(Paragraph("AI-Generated Forecast Insights", heading2_style))
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=normal_style,
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=5
    )
    for insight in insights:
        story.append(Paragraph(f"&bull; {insight}", bullet_style))
        
    doc.build(story)
    buffer.seek(0)
    return buffer
