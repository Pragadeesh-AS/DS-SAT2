import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_charts():
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv("cleaned_data/staynest_cleaned_data.csv")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    # 1. Channel Cancellation Rate
    ch = df.groupby('booking_channel')['is_canceled'].agg(['count', 'mean']).reset_index()
    ch['pct'] = ch['mean'] * 100
    ch = ch.sort_values('pct', ascending=False)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=ch, x='booking_channel', y='pct', hue='booking_channel', palette='Reds_r', legend=False)
    plt.title('Cancellation Rate by Booking Channel', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Cancellation Rate (%)', fontsize=11)
    plt.xlabel('Booking Channel', fontsize=11)
    plt.ylim(0, 38)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/1_channel_cancellation.png", dpi=300)
    plt.close()
    
    # 2. Lead Time vs Cancellation
    order = ['0-14 days (Short)', '15-30 days (Moderate)', '31-90 days (Quarterly)', '91-180 days (Long)', '180+ days (Very Long)']
    lt = df.groupby('lead_time_group')['is_canceled'].agg(['count', 'mean']).reindex(order).reset_index()
    lt['pct'] = lt['mean'] * 100
    
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=lt, x='lead_time_group', y='pct', hue='lead_time_group', palette='Blues_r', legend=False)
    plt.title('Cancellation Rate Across Lead-Time Brackets', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Cancellation Rate (%)', fontsize=11)
    plt.xlabel('Lead Time Bracket', fontsize=11)
    plt.ylim(0, 48)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/2_lead_time_cancellation.png", dpi=300)
    plt.close()

    # 3. Seasonal Occupancy & Revenue
    season_order = ['Summer', 'Spring', 'Autumn', 'Winter']
    season = df.groupby('season').agg(
        completed=('is_canceled', lambda x: (x == 0).sum()),
        revenue=('revenue', lambda x: x[df.loc[x.index, 'is_canceled'] == 0].sum() / 1e6)
    ).reindex(season_order).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = '#1f77b4'
    ax1.set_title('Seasonal Completed Stays and Realized Revenue', fontsize=13, fontweight='bold', pad=12)
    ax1.bar(season['season'], season['completed'], color=color, alpha=0.75, width=0.4, label='Completed Stays')
    ax1.set_ylabel('Completed Stays', color=color, fontsize=11)
    ax1.set_xlabel('Season', fontsize=11)
    
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.plot(season['season'], season['revenue'], color=color2, marker='s', linewidth=2.5, label='Revenue ($M)')
    ax2.set_ylabel('Realized Room Revenue ($ Millions)', color=color2, fontsize=11)
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/3_seasonal_occupancy_revenue.png", dpi=300)
    plt.close()

    # 4. Repeat Guests vs First-Time
    rep = df.groupby('repeat_status')['is_canceled'].agg(['count', 'mean']).reset_index()
    rep['pct'] = rep['mean'] * 100
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=rep, x='repeat_status', y='pct', hue='repeat_status', palette=['#3498db', '#2ecc71'], legend=False)
    plt.title('Cancellation Rate: Repeat Guests vs First-Time Guests', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Cancellation Rate (%)', fontsize=11)
    plt.xlabel('Guest Loyalty Status', fontsize=11)
    plt.ylim(0, 35)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/4_repeat_guests_cancellation.png", dpi=300)
    plt.close()

    # 5. Room Performance
    rooms = df.groupby('room_type').agg(
        revenue=('revenue', lambda x: x[df.loc[x.index, 'is_canceled'] == 0].sum() / 1e6),
        adr=('adr', 'mean')
    ).sort_values('revenue', ascending=False).head(5).reset_index()
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=rooms, x='room_type', y='revenue', hue='room_type', palette='viridis', legend=False)
    plt.title('Top Room Types by Realized Revenue ($ Millions)', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Revenue ($ Millions)', fontsize=11)
    plt.xlabel('Room Type', fontsize=11)
    for p in ax.patches:
        ax.annotate(f"${p.get_height():.2f}M", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/5_room_performance.png", dpi=300)
    plt.close()

    # 6. Event Impact
    events = df.groupby('has_event').agg(
        adr=('adr', 'mean'),
        rate=('is_canceled', lambda x: x.mean() * 100)
    ).reset_index()
    events['status'] = events['has_event'].map({0: 'Regular Non-Event Days', 1: 'Major Event Days'})
    
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=events, x='status', y='adr', hue='status', palette=['#7f8c8d', '#e67e22'], legend=False)
    plt.title('Average Daily Rate (ADR) Comparison: Event vs Normal Days', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Average Daily Rate ($)', fontsize=11)
    plt.xlabel('Calendar Day Type', fontsize=11)
    plt.ylim(0, 150)
    for p in ax.patches:
        ax.annotate(f"${p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/6_event_impact.png", dpi=300)
    plt.close()
    
    print("All 6 publication-quality figures successfully saved in reports/figures/")

if __name__ == "__main__":
    generate_charts()
