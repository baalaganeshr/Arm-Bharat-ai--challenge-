"""
Create sample compliance log for dashboard demo
"""

import pandas as pd
from datetime import datetime, timedelta
import os

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Create sample compliance data for the past 24 hours
print("Creating sample compliance log for dashboard...")

timestamps = []
with_mask = []
without_mask = []
compliance = []

# Generate hourly data for the past 24 hours
for i in range(24, 0, -1):
    timestamp = datetime.now() - timedelta(hours=i)
    timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Simulate increasing compliance over time
    mask_count = 50 + i * 2 + (i % 3) * 5
    no_mask_count = max(5, 30 - i)
    
    with_mask.append(mask_count)
    without_mask.append(no_mask_count)
    
    total = mask_count + no_mask_count
    compliance_pct = round((mask_count / total) * 100, 1)
    compliance.append(compliance_pct)

# Create DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'With_Mask': with_mask,
    'Without_Mask': without_mask,
    'Compliance_Percent': compliance
})

# Save to CSV
log_file = "logs/compliance_log.csv"
df.to_csv(log_file, index=False)

print(f"✓ Sample log created: {log_file}")
print(f"  Total entries: {len(df)}")
print(f"  Latest compliance: {compliance[-1]}%")
print(f"  Total detections: {sum(with_mask) + sum(without_mask)}")
print(f"    With mask: {sum(with_mask)}")
print(f"    Without mask: {sum(without_mask)}")
