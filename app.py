import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="StayNest Hospitality Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B365D;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #566573;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8F9FA 0%, #EDF2F7 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1B365D;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-alert {
        background-color: #FED7D7;
        color: #9B2C2C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-success {
        background-color: #C6F6D5;
        color: #22543D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data/staynest_cleaned_data.csv")
    df['stay_date_dt'] = pd.to_datetime(df['stay_date'])
    df['property_type'] = df['hotel']
    return df

@st.cache_resource
def load_model():
    model_path = "models/staynest_cancellation_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
            return artifact
    return None

df_raw = load_data()
model_artifact = load_model()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1566073771259-6a8506099945?w=500&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("🏨 StayNest Control Panel")
st.sidebar.markdown("**DS Sprint - Skill Based Assessment 2**  \n*Use Case 10: Hotel Booking Insight*")

# Filter 1: Property Type
hotel_options = ["All Properties"] + list(df_raw['hotel_name'].unique())
selected_hotel = st.sidebar.selectbox("Filter Property:", hotel_options)

# Filter 2: Season
season_options = ["All Seasons", "Summer", "Spring", "Autumn", "Winter"]
selected_season = st.sidebar.selectbox("Filter Season:", season_options)

# Filter 3: Channel
channel_options = ["All Channels"] + list(df_raw['booking_channel'].unique())
selected_channel = st.sidebar.selectbox("Filter Acquisition Channel:", channel_options)

# Filter 4: Guest Loyalty
loyalty_options = ["All Guests", "Repeat Guests Only", "First-Time Guests Only"]
selected_loyalty = st.sidebar.selectbox("Guest Loyalty Status:", loyalty_options)

# Apply filters
df = df_raw.copy()
if selected_hotel != "All Properties":
    df = df[df['hotel_name'] == selected_hotel]
if selected_season != "All Seasons":
    df = df[df['season'] == selected_season]
if selected_channel != "All Channels":
    df = df[df['booking_channel'] == selected_channel]
if selected_loyalty == "Repeat Guests Only":
    df = df[df['repeat_status'] == 'Repeat Guest']
elif selected_loyalty == "First-Time Guests Only":
    df = df[df['repeat_status'] == 'New Guest']

# Header
st.markdown('<p class="main-title">StayNest Hospitality Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Understand Booking Dynamics • Mitigate Cancellations • Maximize Seasonal Occupancy & Room Yield</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP EXECUTIVE KPI CARDS
# -----------------------------------------------------------------------------
total_bookings = len(df)
cancel_rate = (df['is_canceled'].mean() * 100) if total_bookings > 0 else 0
completed_rev = df[df['is_canceled'] == 0]['revenue'].sum() if total_bookings > 0 else 0
avg_adr = df['adr'].mean() if total_bookings > 0 else 0
avg_lead = df['lead_time'].mean() if total_bookings > 0 else 0
repeat_pct = (df['is_repeated_guest'].mean() * 100) if total_bookings > 0 else 0

kpi_cols = st.columns(6)
with kpi_cols[0]:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{total_bookings:,}</div><div class="metric-lbl">Total Bookings</div></div>', unsafe_allow_html=True)
with kpi_cols[1]:
    badge = "badge-alert" if cancel_rate > 25 else "badge-success"
    st.markdown(f'<div class="metric-card"><div class="metric-val">{cancel_rate:.1f}%</div><div class="metric-lbl">Cancel Rate</div></div>', unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f'<div class="metric-card"><div class="metric-val">${completed_rev/1e6:.2f}M</div><div class="metric-lbl">Realized Revenue</div></div>', unsafe_allow_html=True)
with kpi_cols[3]:
    st.markdown(f'<div class="metric-card"><div class="metric-val">${avg_adr:.2f}</div><div class="metric-lbl">Average ADR</div></div>', unsafe_allow_html=True)
with kpi_cols[4]:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_lead:.1f}d</div><div class="metric-lbl">Avg Lead Time</div></div>', unsafe_allow_html=True)
with kpi_cols[5]:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{repeat_pct:.1f}%</div><div class="metric-lbl">Repeat Loyalty</div></div>', unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# MAIN DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🔍 6 Core Analytical Questions",
    "🤖 Live ML Cancellation Predictor",
    "📋 Strategic Action Plan & Data"
])

# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tab1:
    col_t1, col_t2 = st.columns([3, 2])
    
    with col_t1:
        st.subheader("📅 Monthly Booking & Cancellation Trajectory")
        monthly_trend = df.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_month_num']).agg(
            Bookings=('is_canceled', 'count'),
            Completed=('is_canceled', lambda x: (x == 0).sum()),
            Canceled=('is_canceled', 'sum')
        ).reset_index().sort_values(['arrival_date_year', 'arrival_month_num'])
        monthly_trend['YearMonth'] = monthly_trend['arrival_date_month'] + " " + monthly_trend['arrival_date_year'].astype(str)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=monthly_trend['YearMonth'], y=monthly_trend['Completed'], name='Completed Stays', marker_color='#2ECC71'))
        fig_trend.add_trace(go.Bar(x=monthly_trend['YearMonth'], y=monthly_trend['Canceled'], name='Cancellations', marker_color='#E74C3C'))
        fig_trend.update_layout(barmode='stack', height=360, margin=dict(l=20, r=20, t=30, b=50), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_t2:
        st.subheader("🏨 Revenue by Property Type")
        prop_rev = df.groupby('property_type').agg(
            Realized=('revenue', lambda x: x[df.loc[x.index, 'is_canceled'] == 0].sum() / 1e6),
            Lost=('revenue', lambda x: x[df.loc[x.index, 'is_canceled'] == 1].sum() / 1e6)
        ).reset_index()
        
        fig_prop = go.Figure()
        fig_prop.add_trace(go.Bar(x=prop_rev['property_type'], y=prop_rev['Realized'], name='Realized Revenue ($M)', marker_color='#1B365D'))
        fig_prop.add_trace(go.Bar(x=prop_rev['property_type'], y=prop_rev['Lost'], name='Lost Revenue ($M)', marker_color='#E67E22'))
        fig_prop.update_layout(barmode='group', height=360, margin=dict(l=20, r=20, t=30, b=50), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_prop, use_container_width=True)

    st.markdown("---")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.subheader("Market Segment Volume")
        seg_df = df['market_segment'].value_counts().reset_index()
        seg_df.columns = ['Segment', 'Bookings']
        fig_seg = px.pie(seg_df, names='Segment', values='Bookings', hole=0.45, color_discrete_sequence=px.colors.qualitative.Safe)
        fig_seg.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig_seg, use_container_width=True)
        
    with col_d2:
        st.subheader("Guest Origin (Top 7)")
        origin_df = df[df['origin'] != 'Unknown']['origin'].value_counts().head(7).reset_index()
        origin_df.columns = ['Country', 'Bookings']
        fig_org = px.bar(origin_df, x='Country', y='Bookings', color='Bookings', color_continuous_scale='Blues')
        fig_org.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_org, use_container_width=True)

    with col_d3:
        st.subheader("Room Rate Tier Mix")
        tier_df = df['rate_tier'].value_counts().reset_index()
        tier_df.columns = ['Tier', 'Count']
        fig_tier = px.bar(tier_df, x='Tier', y='Count', color='Tier', color_discrete_sequence=px.colors.sequential.Teal)
        fig_tier.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_tier, use_container_width=True)

# =============================================================================
# TAB 2: THE 6 CORE ASSESSMENT QUESTIONS
# =============================================================================
with tab2:
    st.subheader("Detailed Solutions to Assessment Rubric Analytical Questions")
    
    # Row 1: Questions 1 & 2
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.markdown("#### 1️⃣ Which channels have higher cancellations?")
        ch_stat = df.groupby('booking_channel')['is_canceled'].agg(['count', 'sum', 'mean']).reset_index()
        ch_stat['rate'] = (ch_stat['mean'] * 100).round(2)
        fig_q1 = px.bar(ch_stat, x='booking_channel', y='rate', text='rate', color='rate',
                        color_continuous_scale='Reds', labels={'rate': 'Cancellation Rate (%)', 'booking_channel': 'Channel'})
        fig_q1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_q1.update_layout(height=320, coloraxis_showscale=False, yaxis_range=[0, 42])
        st.plotly_chart(fig_q1, use_container_width=True)
        st.info("💡 **Insight:** TA/TO intermediary channel has the highest cancellation rate (31.2%). Direct (15.0%) and Corporate (12.8%) channels show over 2x stronger commitment.")

    with r1_col2:
        st.markdown("#### 2️⃣ How does lead time relate to cancellation?")
        order_lt = ['0-14 days (Short)', '15-30 days (Moderate)', '31-90 days (Quarterly)', '91-180 days (Long)', '180+ days (Very Long)']
        lt_stat = df.groupby('lead_time_group')['is_canceled'].agg(['count', 'mean']).reindex(order_lt).reset_index()
        lt_stat['rate'] = (lt_stat['mean'] * 100).round(2)
        fig_q2 = px.line(lt_stat, x='lead_time_group', y='rate', markers=True, text='rate',
                         labels={'rate': 'Cancellation Rate (%)', 'lead_time_group': 'Lead Time Bracket'})
        fig_q2.update_traces(line_color='#2980B9', line_width=3, marker_size=9, texttemplate='%{text:.1f}%', textposition='top center')
        fig_q2.update_layout(height=320, yaxis_range=[0, 48])
        st.plotly_chart(fig_q2, use_container_width=True)
        st.info("💡 **Insight:** Strong positive correlation (+0.1835). Bookings placed >180 days in advance cancel at 39.8%—nearly 3.4x higher than immediate bookings (11.6%).")

    st.markdown("---")
    # Row 2: Questions 3 & 4
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.markdown("#### 3️⃣ Which seasons have higher occupancy & revenue?")
        s_stat = df.groupby('season').agg(
            Completed=('is_canceled', lambda x: (x == 0).sum()),
            Revenue=('revenue', lambda x: x[df.loc[x.index, 'is_canceled'] == 0].sum() / 1e6),
            ADR=('adr', 'mean')
        ).reindex(['Summer', 'Spring', 'Autumn', 'Winter']).reset_index()
        
        fig_q3 = go.Figure()
        fig_q3.add_trace(go.Bar(x=s_stat['season'], y=s_stat['Completed'], name='Completed Stays', marker_color='#2980B9', yaxis='y1'))
        fig_q3.add_trace(go.Scatter(x=s_stat['season'], y=s_stat['Revenue'], name='Revenue ($M)', mode='lines+markers', marker_color='#E67E22', line_width=3, yaxis='y2'))
        fig_q3.update_layout(
            height=320,
            yaxis=dict(title='Completed Stays'),
            yaxis2=dict(title='Revenue ($M)', overlaying='y', side='right'),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_q3, use_container_width=True)
        st.info("💡 **Insight:** Summer is the peak revenue engine ($16.60M revenue, 19,735 stays, $138 ADR). Winter occupancy plunges to $3.96M (4.2x seasonal gap).")

    with r2_col2:
        st.markdown("#### 4️⃣ Do repeat guests cancel less?")
        rep_stat = df.groupby('repeat_status')['is_canceled'].agg(['count', 'mean']).reset_index()
        rep_stat['rate'] = (rep_stat['mean'] * 100).round(2)
        fig_q4 = px.bar(rep_stat, x='repeat_status', y='rate', text='rate', color='repeat_status',
                        color_discrete_map={'Repeat Guest': '#2ECC71', 'New Guest': '#E74C3C'},
                        labels={'rate': 'Cancellation Rate (%)', 'repeat_status': 'Loyalty Status'})
        fig_q4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_q4.update_layout(height=320, showlegend=False, yaxis_range=[0, 38])
        st.plotly_chart(fig_q4, use_container_width=True)
        st.info("💡 **Insight:** Yes, decisively! Repeat guests cancel at only 8.2% vs 28.4% for first-time guests (3.45x less likely to cancel).")

    st.markdown("---")
    # Row 3: Questions 5 & 6
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        st.markdown("#### 5️⃣ Which room types perform best?")
        rm_stat = df.groupby('room_type').agg(
            Bookings=('is_canceled', 'count'),
            Revenue=('revenue', lambda x: x[df.loc[x.index, 'is_canceled'] == 0].sum() / 1e6),
            ADR=('adr', 'mean')
        ).sort_values('Revenue', ascending=False).head(5).reset_index()
        fig_q5 = px.bar(rm_stat, x='room_type', y='Revenue', text='Revenue', color='ADR',
                        color_continuous_scale='Viridis', labels={'Revenue': 'Realized Revenue ($M)', 'room_type': 'Room Category'})
        fig_q5.update_traces(texttemplate='$%{text:.2f}M', textposition='outside')
        fig_q5.update_layout(height=320)
        st.plotly_chart(fig_q5, use_container_width=True)
        st.info("💡 **Insight:** Room Type A anchors volume (64.6% of stays, $11.98M revenue). Room Type D is the premium yield driver ($5.69M revenue, $122.80 ADR).")

    with r3_col2:
        st.markdown("#### 6️⃣ How do local events affect bookings & rates?")
        evt_stat = df.groupby('has_event').agg(
            ADR=('adr', 'mean'),
            Count=('is_canceled', 'count')
        ).reset_index()
        evt_stat['Day_Type'] = evt_stat['has_event'].map({0: 'Regular Days', 1: 'Major Event Days'})
        fig_q6 = px.bar(evt_stat, x='Day_Type', y='ADR', text='ADR', color='Day_Type',
                        color_discrete_map={'Regular Days': '#95A5A6', 'Major Event Days': '#F39C12'})
        fig_q6.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_q6.update_layout(height=320, showlegend=False, yaxis_range=[0, 155])
        st.plotly_chart(fig_q6, use_container_width=True)
        st.info("💡 **Insight:** Major festivals and conferences command a +22.1% rate premium ($125.59 vs $102.83). Algarve Regatta leads at $167.33 ADR.")

# =============================================================================
# TAB 3: LIVE MACHINE LEARNING CANCELLATION PREDICTOR
# =============================================================================
with tab3:
    st.subheader("🤖 Live Booking Cancellation Risk Estimator (CO3 & CO4)")
    st.markdown("Simulate reservation attributes below to test the trained **Random Forest Classifier (Accuracy: 80.61%, ROC-AUC: 0.848)** in real-time.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        inp_lead = st.slider("Lead Time (Days):", 0, 365, 45)
        inp_adr = st.slider("Room Rate / ADR ($):", 30, 450, 115)
        inp_nights = st.slider("Total Stay Nights:", 1, 21, 3)
        inp_guests = st.slider("Total Guests:", 1, 6, 2)
        
    with col_p2:
        inp_channel = st.selectbox("Booking Acquisition Channel:", ["TA/TO", "Direct", "Corporate", "GDS"])
        inp_deposit = st.selectbox("Deposit Type:", ["No Deposit", "Non Refund", "Refundable"])
        inp_repeat = st.selectbox("Is Repeat Guest?", ["No (First-Time)", "Yes (Loyalty)"])
        inp_requests = st.slider("Special Requests:", 0, 5, 1)

    with col_p3:
        inp_parking = st.selectbox("Requires Car Parking?", ["No", "Yes"])
        inp_changes = st.slider("Prior Booking Changes:", 0, 5, 0)
        inp_mismatch = st.selectbox("Room Reassignment / Upgrade?", ["No", "Yes"])
        inp_season = st.selectbox("Stay Season:", ["Summer", "Spring", "Autumn", "Winter"])
        inp_event = st.selectbox("During Major Event / Festival?", ["No", "Yes"])

    st.write("")
    if st.button("🚀 Calculate Cancellation Risk Score", type="primary", use_container_width=True):
        if model_artifact:
            model = model_artifact['model']
            features = model_artifact['features']
            encoders = model_artifact['cat_encoders']
            
            # Format input row
            row_dict = {
                'lead_time': inp_lead,
                'total_stay_nights': inp_nights,
                'rate': inp_adr,
                'total_guests': inp_guests,
                'total_of_special_requests': inp_requests,
                'required_car_parking_spaces': 1 if inp_parking == "Yes" else 0,
                'room_type_mismatch': 1 if inp_mismatch == "Yes" else 0,
                'is_repeated_guest': 1 if inp_repeat.startswith("Yes") else 0,
                'previous_cancellations': 0,
                'booking_changes': inp_changes,
                'deposit_type': inp_deposit,
                'booking_channel': inp_channel,
                'guest_type': 'Transient',
                'market_segment': 'Online TA' if inp_channel == 'TA/TO' else 'Direct',
                'season': inp_season,
                'has_event': 1 if inp_event == "Yes" else 0
            }
            
            input_df = pd.DataFrame([row_dict])[features]
            for col in encoders:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
                
            pred_prob = model.predict_proba(input_df)[0][1]
            risk_pct = pred_prob * 100
            
            st.write("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_pct,
                    number = {'suffix': "%"},
                    title = {'text': "Cancellation Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1B365D"},
                        'steps': [
                            {'range': [0, 30], 'color': "#C6F6D5"},
                            {'range': [30, 60], 'color': "#FEFCBF"},
                            {'range': [60, 100], 'color': "#FED7D7"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with res_col2:
                if risk_pct >= 50:
                    st.error(f"⚠️ **HIGH RISK ALERT ({risk_pct:.1f}% Probability of Cancellation)**")
                    st.markdown("""
                    **Recommended Automated Management Interventions:**
                    - 💳 **Deposit Policy:** Enforce a 25% non-refundable advance deposit.
                    - 📩 **Engagement:** Trigger automated confirmation email requesting guest arrival time.
                    - 🔄 **Overbooking Protection:** Release room back into live inventory if unconfirmed 48h prior.
                    """)
                else:
                    st.success(f"✅ **LOW RISK BOOKING ({risk_pct:.1f}% Probability of Cancellation)**")
                    st.markdown("""
                    **Recommended Guest Experience Actions:**
                    - 🌟 **Upsell Opportunity:** Offer a pre-arrival paid upgrade to Room Type D or Suite.
                    - 🎁 **Loyalty Recruitment:** Invite guest to join the *StayNest Circle* rewards program.
                    - 🍽️ **Ancillary Revenue:** Promote dining packages or spa vouchers.
                    """)
        else:
            st.warning("Trained model artifact not found. Please run scripts/train_model.py first.")

# =============================================================================
# TAB 4: STRATEGIC ACTION ROADMAP & DATA EXPORT
# =============================================================================
with tab4:
    st.subheader("📋 Executive Action Roadmap (Pitch Focus)")
    st.markdown("""
    | Phase | Timeline | Core Focus | Specific Operational Interventions |
    | :--- | :--- | :--- | :--- |
    | **Phase 1** | **Immediate (30 Days)** | **Channel De-Risking** | Impose 20% non-refundable deposits on OTA intermediaries; launch best-rate guarantee on StayNest website. |
    | **Phase 2** | **Mid-Term (90 Days)** | **Yield & Capacity Optimization** | Restructure lead-time cancellation tiers; launch automated pre-arrival room upgrade nudges. |
    | **Phase 3** | **Strategic (180 Days)** | **Loyalty & Retention** | Roll out 'StayNest Circle' loyalty perks; integrate ML cancellation scoring into front-desk reservation queue. |
    """)
    
    st.markdown("---")
    st.subheader("📥 Export Cleaned Dataset")
    st.dataframe(df.head(100), use_container_width=True)
    
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Cleaned Dataset (CSV)",
        data=csv_bytes,
        file_name="staynest_cleaned_export.csv",
        mime="text/csv",
        type="secondary"
    )

st.markdown("---")
st.caption("StayNest Hospitality Intelligence Dashboard | U23AD491 Data Science Assessment | Candidate: Pragadeesh")
