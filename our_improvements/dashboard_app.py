"""
Face Mask Detection Web Dashboard
Real-time compliance monitoring using Flask

Run: python our_improvements/dashboard_app.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, send_file, make_response
import pandas as pd
import json
from datetime import datetime
import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path to access templates
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(parent_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Configuration
LOG_FILE = os.path.join(parent_dir, 'logs', 'compliance_log.csv')
EXPORT_DIR = os.path.join(parent_dir, 'logs')


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/stats')
def get_stats():
    """Get current statistics from log file"""
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            
            if len(df) == 0:
                return jsonify({'error': 'No data available'})
            
            # Get latest entry
            latest = df.tail(1).to_dict('records')[0]
            
            # Calculate totals - handle different column naming
            with_mask_col = 'With_Mask' if 'With_Mask' in df.columns else 'With Mask'
            without_mask_col = 'Without_Mask' if 'Without_Mask' in df.columns else 'Without Mask'
            
            total_with_mask = int(df[with_mask_col].sum()) if with_mask_col in df.columns else 0
            total_without_mask = int(df[without_mask_col].sum()) if without_mask_col in df.columns else 0
            total = total_with_mask + total_without_mask
            
            compliance = round((total_with_mask / total * 100), 1) if total > 0 else 0
            
            # Get hourly data for chart
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.hour
            hourly = df.groupby('Hour').agg({
                with_mask_col: 'sum',
                without_mask_col: 'sum'
            }).reset_index()
            hourly.columns = ['Hour', 'With_Mask', 'Without_Mask']
            
            # Get recent entries (last 100)
            recent = df.tail(100).to_dict('records')
            
            return jsonify({
                'latest': latest,
                'totals': {
                    'with_mask': total_with_mask,
                    'without_mask': total_without_mask,
                    'total': total,
                    'compliance': compliance
                },
                'hourly': hourly.to_dict('records'),
                'recent_count': len(df),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            return jsonify({
                'error': 'No data file found',
                'totals': {'with_mask': 0, 'without_mask': 0, 'total': 0, 'compliance': 0}
            })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/export')
def export_data():
    """Export data to CSV file"""
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = os.path.join(EXPORT_DIR, f'export_{timestamp}.csv')
            df.to_csv(export_file, index=False)
            return jsonify({'success': True, 'file': export_file, 'rows': len(df)})
        else:
            return jsonify({'error': 'No data to export'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            with_mask_col = 'With_Mask' if 'With_Mask' in df.columns else 'With Mask'
            without_mask_col = 'Without_Mask' if 'Without_Mask' in df.columns else 'Without Mask'
            
            summary = {
                'total_entries': len(df),
                'date_range': {
                    'start': df['Timestamp'].min().strftime("%Y-%m-%d %H:%M"),
                    'end': df['Timestamp'].max().strftime("%Y-%m-%d %H:%M")
                },
                'totals': {
                    'with_mask': int(df[with_mask_col].sum()),
                    'without_mask': int(df[without_mask_col].sum())
                },
                'averages': {
                    'per_frame_mask': round(df[with_mask_col].mean(), 2),
                    'per_frame_no_mask': round(df[without_mask_col].mean(), 2)
                }
            }
            
            return jsonify(summary)
        else:
            return jsonify({'error': 'No data available'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/export_pdf')
def export_pdf():
    """Generate and download PDF report"""
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({'error': 'No data available'})
        
        df = pd.read_csv(LOG_FILE)
        if len(df) == 0:
            return jsonify({'error': 'No data available'})
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                              leftMargin=0.75*inch, rightMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("SECURE GUARD PRO", title_style))
        elements.append(Paragraph("Face Mask Detection Compliance Report", styles['Heading2']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Report metadata
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        report_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        with_mask_col = 'With_Mask' if 'With_Mask' in df.columns else 'With Mask'
        without_mask_col = 'Without_Mask' if 'Without_Mask' in df.columns else 'Without Mask'
        
        total_with_mask = int(df[with_mask_col].sum())
        total_without_mask = int(df[without_mask_col].sum())
        total = total_with_mask + total_without_mask
        compliance = round((total_with_mask / total * 100), 1) if total > 0 else 0
        
        # Summary info
        meta_data = [
            ['Report Generated:', report_time],
            ['Data Period:', f"{df['Timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['Timestamp'].max().strftime('%Y-%m-%d %H:%M')}"],
            ['Total Entries:', f"{len(df):,}"],
            ['', '']
        ]
        
        meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(meta_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Summary statistics
        elements.append(Paragraph("Overall Statistics", heading_style))
        
        summary_data = [
            ['Metric', 'Count', 'Percentage'],
            ['Faces with Mask', f"{total_with_mask:,}", f"{compliance:.1f}%"],
            ['Faces without Mask', f"{total_without_mask:,}", f"{100-compliance:.1f}%"],
            ['Total Detections', f"{total:,}", '100.0%'],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')]),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Compliance status
        elements.append(Paragraph("Compliance Status", heading_style))
        
        if compliance >= 80:
            status = "EXCELLENT"
            status_color = colors.green
        elif compliance >= 60:
            status = "GOOD"
            status_color = colors.orange
        else:
            status = "NEEDS IMPROVEMENT"
            status_color = colors.red
        
        status_data = [
            ['Overall Compliance Rate', f"{compliance:.1f}%", status]
        ]
        
        status_table = Table(status_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        status_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#f3f4f6')),
            ('BACKGROUND', (2, 0), (2, 0), status_color),
            ('TEXTCOLOR', (2, 0), (2, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 2, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(status_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Generate compliance chart
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            
            categories = ['With Mask', 'Without Mask']
            values = [total_with_mask, total_without_mask]
            colors_chart = ['#10b981', '#ef4444']
            
            ax.bar(categories, values, color=colors_chart, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Number of Detections', fontsize=12, fontweight='bold')
            ax.set_title('Face Mask Detection Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(i, v + max(values)*0.02, f'{v:,}', ha='center', fontweight='bold')
            
            # Save chart to buffer
            chart_buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            chart_buffer.seek(0)
            
            # Add chart to PDF
            elements.append(Paragraph("Visual Distribution", heading_style))
            chart_img = Image(chart_buffer, width=6*inch, height=3.5*inch)
            elements.append(chart_img)
            elements.append(Spacer(1, 0.2*inch))
            
        except Exception as e:
            print(f"Chart generation error: {e}")
        
        # Hourly breakdown
        elements.append(Paragraph("Hourly Compliance Breakdown", heading_style))
        
        df['Hour'] = df['Timestamp'].dt.hour
        hourly = df.groupby('Hour').agg({
            with_mask_col: 'sum',
            without_mask_col: 'sum'
        }).reset_index()
        
        # Take last 12 hours or available data
        hourly_display = hourly.tail(12)
        
        hourly_data = [['Hour', 'With Mask', 'Without Mask', 'Compliance %']]
        for _, row in hourly_display.iterrows():
            hour_total = row[with_mask_col] + row[without_mask_col]
            hour_compliance = round((row[with_mask_col] / hour_total * 100), 1) if hour_total > 0 else 0
            hourly_data.append([
                f"{int(row['Hour']):02d}:00",
                f"{int(row[with_mask_col]):,}",
                f"{int(row[without_mask_col]):,}",
                f"{hour_compliance}%"
            ])
        
        hourly_table = Table(hourly_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        hourly_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')]),
        ]))
        
        elements.append(hourly_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER
        )
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Generated by SECURE GUARD PRO - Face Mask Detection System", footer_style))
        elements.append(Paragraph(f"Report Date: {report_time}", footer_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Send file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mask_compliance_report_{timestamp}.pdf"
        
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        
        return response
        
    except Exception as e:
        import traceback
        print(f"PDF generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/api/clear')
def clear_data():
    """Clear log file (for testing)"""
    try:
        if os.path.exists(LOG_FILE):
            # Backup before clearing
            backup_file = LOG_FILE.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            os.rename(LOG_FILE, backup_file)
            return jsonify({'success': True, 'backup': backup_file})
        return jsonify({'success': True, 'message': 'No file to clear'})
    except Exception as e:
        return jsonify({'error': str(e)})


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Face Mask Detection Dashboard")
    print("=" * 50)
    print(f"Log file: {LOG_FILE}")
    print(f"Template directory: {template_dir}")
    print("Starting server at http://localhost:9000")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=9000, debug=False)
