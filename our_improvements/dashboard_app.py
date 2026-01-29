"""
Face Mask Detection Web Dashboard
Real-time compliance monitoring using Flask

Run: python our_improvements/dashboard_app.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import json
from datetime import datetime
import os

app = Flask(__name__)

# Configuration
LOG_FILE = 'logs/compliance_log.csv'
EXPORT_DIR = 'logs'


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
    os.makedirs('our_improvements/templates', exist_ok=True)
    
    print("=" * 50)
    print("Face Mask Detection Dashboard")
    print("=" * 50)
    print(f"Log file: {LOG_FILE}")
    print("Starting server at http://localhost:9000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=9000, debug=True)
