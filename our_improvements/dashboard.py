"""
Compliance Dashboard Module
Tracks and reports face mask detection statistics

Features:
- CSV logging with timestamps
- Daily/hourly compliance reports
- Real-time statistics tracking
- Export functionality for reports
- Matplotlib visualizations
"""

import csv
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque
import json


class ComplianceDashboard:
    """
    Dashboard for tracking and reporting mask compliance statistics
    
    Features:
    - Real-time logging of detection results
    - Compliance percentage tracking
    - Daily/hourly aggregated reports
    - CSV export and visualization
    """
    
    def __init__(self, log_file: str = 'logs/compliance_log.csv', 
                 max_memory_entries: int = 1000):
        """
        Initialize dashboard
        
        Args:
            log_file: Path to CSV log file
            max_memory_entries: Maximum entries to keep in memory for real-time stats
        """
        self.log_file = log_file
        self.max_memory_entries = max_memory_entries
        
        # In-memory buffer for real-time stats
        self.recent_entries: deque = deque(maxlen=max_memory_entries)
        
        # Session statistics
        self.session_start = datetime.now()
        self.session_mask_count = 0
        self.session_no_mask_count = 0
        self.session_detections = 0
        
        # Initialize log file
        self._init_log()
    
    def _init_log(self):
        """Initialize log file with headers if it doesn't exist"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        try:
            with open(self.log_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'With_Mask',
                    'Without_Mask',
                    'Total',
                    'Compliance_Percent',
                    'FPS',
                    'Processing_Mode'
                ])
                print(f"[INFO] Created log file: {self.log_file}")
        except FileExistsError:
            print(f"[INFO] Using existing log file: {self.log_file}")
    
    def log_detection(self, mask_count: int, no_mask_count: int, 
                     fps: float = 0.0, mode: str = 'CPU') -> float:
        """
        Log a detection result
        
        Args:
            mask_count: Number of people with masks
            no_mask_count: Number of people without masks
            fps: Current FPS
            mode: Processing mode ('CPU', 'FPGA', etc.)
            
        Returns:
            Compliance percentage
        """
        timestamp = datetime.now()
        total = mask_count + no_mask_count
        compliance = (mask_count / total * 100) if total > 0 else 0.0
        
        # Update session stats
        self.session_mask_count += mask_count
        self.session_no_mask_count += no_mask_count
        self.session_detections += 1
        
        # Add to memory buffer
        entry = {
            'timestamp': timestamp,
            'mask_count': mask_count,
            'no_mask_count': no_mask_count,
            'total': total,
            'compliance': compliance,
            'fps': fps,
            'mode': mode
        }
        self.recent_entries.append(entry)
        
        # Write to CSV
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    mask_count,
                    no_mask_count,
                    total,
                    f"{compliance:.1f}",
                    f"{fps:.1f}",
                    mode
                ])
        except Exception as e:
            print(f"[WARN] Could not write to log: {e}")
        
        return compliance
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        total = self.session_mask_count + self.session_no_mask_count
        compliance = (self.session_mask_count / total * 100) if total > 0 else 0.0
        duration = datetime.now() - self.session_start
        
        return {
            'session_start': self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
            'duration_minutes': duration.total_seconds() / 60,
            'total_detections': self.session_detections,
            'with_mask': self.session_mask_count,
            'without_mask': self.session_no_mask_count,
            'compliance_percent': compliance
        }
    
    def get_recent_stats(self, minutes: int = 5) -> Dict:
        """Get statistics for the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        recent = [e for e in self.recent_entries if e['timestamp'] > cutoff]
        
        if not recent:
            return {
                'period_minutes': minutes,
                'entries': 0,
                'with_mask': 0,
                'without_mask': 0,
                'avg_compliance': 0.0,
                'avg_fps': 0.0
            }
        
        total_mask = sum(e['mask_count'] for e in recent)
        total_no_mask = sum(e['no_mask_count'] for e in recent)
        avg_compliance = sum(e['compliance'] for e in recent) / len(recent)
        avg_fps = sum(e['fps'] for e in recent) / len(recent)
        
        return {
            'period_minutes': minutes,
            'entries': len(recent),
            'with_mask': total_mask,
            'without_mask': total_no_mask,
            'avg_compliance': avg_compliance,
            'avg_fps': avg_fps
        }
    
    def generate_hourly_report(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """Generate hourly aggregated report from log file"""
        try:
            df = pd.read_csv(self.log_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.floor('H')
            
            hourly = df.groupby('Hour').agg({
                'With_Mask': 'sum',
                'Without_Mask': 'sum',
                'Total': 'sum',
                'Compliance_Percent': 'mean',
                'FPS': 'mean'
            }).reset_index()
            
            hourly.columns = ['Hour', 'With_Mask', 'Without_Mask', 'Total', 
                            'Avg_Compliance', 'Avg_FPS']
            
            if output_file:
                hourly.to_csv(output_file, index=False)
                print(f"[INFO] Hourly report saved to {output_file}")
            
            return hourly
            
        except Exception as e:
            print(f"[ERROR] Could not generate hourly report: {e}")
            return pd.DataFrame()
    
    def generate_daily_report(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """Generate daily aggregated report from log file"""
        try:
            df = pd.read_csv(self.log_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Date'] = df['Timestamp'].dt.date
            
            daily = df.groupby('Date').agg({
                'With_Mask': 'sum',
                'Without_Mask': 'sum',
                'Total': 'sum',
                'Compliance_Percent': 'mean',
                'FPS': 'mean'
            }).reset_index()
            
            daily.columns = ['Date', 'With_Mask', 'Without_Mask', 'Total',
                           'Avg_Compliance', 'Avg_FPS']
            
            # Add compliance rate
            daily['Compliance_Rate'] = (daily['With_Mask'] / daily['Total'] * 100).round(1)
            
            if output_file:
                daily.to_csv(output_file, index=False)
                print(f"[INFO] Daily report saved to {output_file}")
            
            return daily
            
        except Exception as e:
            print(f"[ERROR] Could not generate daily report: {e}")
            return pd.DataFrame()
    
    def export_summary(self, output_file: str = 'logs/summary_report.json') -> Dict:
        """Export comprehensive summary report"""
        try:
            df = pd.read_csv(self.log_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Convert Compliance_Percent to numeric, handling potential string issues
            df['Compliance_Percent'] = pd.to_numeric(df['Compliance_Percent'], errors='coerce')
            
            total_mask = int(df['With_Mask'].sum())
            total_no_mask = int(df['Without_Mask'].sum())
            total = total_mask + total_no_mask
            
            summary = {
                'report_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_start': df['Timestamp'].min().strftime("%Y-%m-%d %H:%M:%S"),
                'data_end': df['Timestamp'].max().strftime("%Y-%m-%d %H:%M:%S"),
                'total_entries': len(df),
                'total_detections': total,
                'with_mask': total_mask,
                'without_mask': total_no_mask,
                'overall_compliance': round(total_mask / total * 100, 2) if total > 0 else 0,
                'avg_compliance_per_frame': round(df['Compliance_Percent'].mean(), 2),
                'min_compliance': round(df['Compliance_Percent'].min(), 2),
                'max_compliance': round(df['Compliance_Percent'].max(), 2),
                'avg_fps': round(df['FPS'].mean(), 2)
            }
            
            # Save to JSON
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"[INFO] Summary exported to {output_file}")
            return summary
            
        except Exception as e:
            print(f"[ERROR] Could not export summary: {e}")
            return {}
    
    def plot_compliance_trend(self, save_path: str = 'logs/compliance_trend.png',
                             hours: int = 24) -> None:
        """Plot compliance trend over time"""
        try:
            df = pd.read_csv(self.log_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Compliance_Percent'] = pd.to_numeric(df['Compliance_Percent'], errors='coerce')
            
            # Filter to recent hours
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df[df['Timestamp'] > cutoff]
            
            if df.empty:
                print("[WARN] No data for plotting")
                return
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Face Mask Compliance Dashboard - Last {hours} Hours', 
                        fontsize=14, fontweight='bold')
            
            # Plot 1: Compliance over time
            ax1 = axes[0, 0]
            ax1.plot(df['Timestamp'], df['Compliance_Percent'], 
                    color='blue', alpha=0.7, linewidth=1)
            ax1.axhline(y=80, color='green', linestyle='--', label='80% Target')
            ax1.axhline(y=df['Compliance_Percent'].mean(), color='orange', 
                       linestyle=':', label=f"Avg: {df['Compliance_Percent'].mean():.1f}%")
            ax1.fill_between(df['Timestamp'], df['Compliance_Percent'], 
                            alpha=0.3, color='blue')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Compliance %')
            ax1.set_title('Compliance Over Time')
            ax1.legend(loc='lower right')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.set_ylim(0, 105)
            
            # Plot 2: Detections distribution
            ax2 = axes[0, 1]
            total_mask = df['With_Mask'].sum()
            total_no_mask = df['Without_Mask'].sum()
            colors = ['#2ecc71', '#e74c3c']
            ax2.pie([total_mask, total_no_mask], 
                   labels=['With Mask', 'Without Mask'],
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title(f'Detection Distribution (n={total_mask + total_no_mask})')
            
            # Plot 3: Hourly aggregation
            ax3 = axes[1, 0]
            df['Hour'] = df['Timestamp'].dt.hour
            hourly = df.groupby('Hour').agg({
                'With_Mask': 'sum',
                'Without_Mask': 'sum'
            })
            
            x = hourly.index
            ax3.bar(x - 0.2, hourly['With_Mask'], 0.4, label='With Mask', color='#2ecc71')
            ax3.bar(x + 0.2, hourly['Without_Mask'], 0.4, label='Without Mask', color='#e74c3c')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Count')
            ax3.set_title('Detections by Hour')
            ax3.legend()
            ax3.set_xticks(range(0, 24, 2))
            
            # Plot 4: FPS over time
            ax4 = axes[1, 1]
            ax4.plot(df['Timestamp'], df['FPS'], color='purple', alpha=0.7)
            ax4.axhline(y=df['FPS'].mean(), color='orange', linestyle='--',
                       label=f"Avg: {df['FPS'].mean():.1f} FPS")
            ax4.set_xlabel('Time')
            ax4.set_ylabel('FPS')
            ax4.set_title('Performance (FPS)')
            ax4.legend(loc='lower right')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Trend plot saved to {save_path}")
            
        except Exception as e:
            print(f"[ERROR] Could not generate plot: {e}")
    
    def print_session_summary(self):
        """Print session summary to console"""
        stats = self.get_session_stats()
        recent = self.get_recent_stats(5)
        
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Started: {stats['session_start']}")
        print(f"Duration: {stats['duration_minutes']:.1f} minutes")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"With Mask: {stats['with_mask']}")
        print(f"Without Mask: {stats['without_mask']}")
        print(f"Overall Compliance: {stats['compliance_percent']:.1f}%")
        print("-" * 50)
        print(f"Last 5 Minutes:")
        print(f"  Entries: {recent['entries']}")
        print(f"  Avg Compliance: {recent['avg_compliance']:.1f}%")
        print(f"  Avg FPS: {recent['avg_fps']:.1f}")
        print("=" * 50 + "\n")


class AlertSystem:
    """Alert system for low compliance detection"""
    
    def __init__(self, threshold: float = 70.0, 
                 consecutive_alerts: int = 5):
        """
        Initialize alert system
        
        Args:
            threshold: Compliance threshold for alerts (%)
            consecutive_alerts: Number of consecutive low readings before alert
        """
        self.threshold = threshold
        self.consecutive_alerts = consecutive_alerts
        self.low_count = 0
        self.alert_active = False
    
    def check(self, compliance: float) -> Optional[str]:
        """
        Check compliance and return alert if needed
        
        Returns:
            Alert message or None
        """
        if compliance < self.threshold:
            self.low_count += 1
            
            if self.low_count >= self.consecutive_alerts and not self.alert_active:
                self.alert_active = True
                return f"⚠️ ALERT: Low compliance detected! ({compliance:.1f}%)"
        else:
            if self.low_count > 0:
                self.low_count -= 1
            
            if self.low_count == 0:
                self.alert_active = False
        
        return None


# ============== CLI INTERFACE ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compliance Dashboard')
    parser.add_argument('--export', action='store_true', help='Export reports')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--hours', type=int, default=24, help='Hours for analysis')
    parser.add_argument('--log', type=str, default='logs/compliance_log.csv', 
                       help='Log file path')
    args = parser.parse_args()
    
    dashboard = ComplianceDashboard(log_file=args.log)
    
    # Generate sample data if log is empty
    if not os.path.exists(args.log) or os.path.getsize(args.log) < 100:
        print("[INFO] Generating sample data for demonstration...")
        import random
        for i in range(100):
            mask = random.randint(1, 5)
            no_mask = random.randint(0, 2)
            fps = random.uniform(20, 35)
            dashboard.log_detection(mask, no_mask, fps, 'CPU')
    
    if args.export:
        print("\n[INFO] Generating reports...")
        dashboard.generate_hourly_report('logs/hourly_report.csv')
        dashboard.generate_daily_report('logs/daily_report.csv')
        summary = dashboard.export_summary()
        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    
    if args.plot:
        print("\n[INFO] Generating plots...")
        dashboard.plot_compliance_trend(hours=args.hours)
    
    dashboard.print_session_summary()
