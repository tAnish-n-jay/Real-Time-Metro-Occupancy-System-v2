import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="Pattanagere Coach Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 2rem 1rem 2rem; }

    /* Typography */
    body { font-family: 'Inter', sans-serif; }

    h1 { font-size: 1.1rem !important; font-weight: 600 !important;
         color: #111 !important; letter-spacing: -0.01em; }

    /* Metric overrides */
    [data-testid="metric-container"] {
        background: #f8f8f8;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label {
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #888 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #111 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e5e5e5; margin: 1.2rem 0; }

    /* Section label */
    .section-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #888;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH  = 'data/live.db'
FULL_G   = 1500.0
STALE_S  = 30

def load_data(limit=200):
    con  = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT ts, grams, percent FROM readings ORDER BY ts DESC LIMIT {limit}",
        con
    )
    con.close()
    if df.empty:
        return df
    df['ts']       = pd.to_datetime(df['ts'], format='mixed')
    df['load_pct'] = (df['percent'] * 100).round(1)
    return df.sort_values('ts')

def level(pct: float) -> str:
    if pct < 25:  return "Low"
    if pct < 50:  return "Moderate"
    if pct < 75:  return "Crowded"
    return "Very Crowded"

def level_color(pct: float) -> str:
    if pct < 25:  return "#22c55e"
    if pct < 50:  return "#f59e0b"
    if pct < 75:  return "#f97316"
    return "#ef4444"

# ── Header ──
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("**PATTANAGERE COACH MONITOR**")
with col_time:
    st.markdown(
        f"<div style='text-align:right;color:#888;font-size:0.8rem;padding-top:4px'>"
        f"{datetime.now().strftime('%d %b %Y, %H:%M:%S')}</div>",
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

df = load_data()

if df.empty:
    st.warning("No readings in database yet.")
    st.stop()

latest   = df.iloc[-1]
prev     = df.iloc[-2] if len(df) > 1 else latest
curr_pct = latest['load_pct']
delta    = round(curr_pct - prev['load_pct'], 1)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current Load",
    f"{curr_pct:.0f}%",
    delta=f"{delta:+.1f}% vs prev"
)
c2.metric(
    "Status",
    level(curr_pct),
)
c3.metric(
    "Weight on Platform",
    f"{latest['grams']:.0f} g",
)
c4.metric(
    "Total Readings",
    f"{len(df)}",
)

st.markdown("<hr>", unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="section-label">Load history — last 200 readings</div>', unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['ts'], y=df['load_pct'],
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.08)',
        line=dict(color='#3b82f6', width=1.5),
        hovertemplate='%{x|%H:%M:%S}<br>Load: %{y:.1f}%<extra></extra>'
    ))

    fig.add_hline(
        y=100, line_dash='dot',
        line_color='#ef4444', line_width=1,
        annotation_text="Full", annotation_position="right"
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=240,
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#888')),
        yaxis=dict(
            showgrid=True, gridcolor='#f0f0f0',
            ticksuffix='%', range=[0, 115],
            tickfont=dict(size=11, color='#888')
        ),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-label">OCCUPANCY</div>', unsafe_allow_html=True)

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=curr_pct,
        number=dict(suffix="%", font=dict(size=28, color='#111')),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor='#e5e5e5',
                tickfont=dict(size=10, color='#aaa')
            ),
            bar=dict(color=level_color(curr_pct), thickness=0.25),
            bgcolor='white',
            borderwidth=0,
            steps=[
                dict(range=[0, 25],  color='#f0fdf4'),
                dict(range=[25, 50], color='#fefce8'),
                dict(range=[50, 75], color='#fff7ed'),
                dict(range=[75, 100],color='#fef2f2'),
            ],
            threshold=dict(
                line=dict(color='#111', width=2),
                thickness=0.75,
                value=curr_pct
            )
        )
    ))
    fig2.update_layout(
        margin=dict(l=20, r=20, t=20, b=10),
        height=240,
        paper_bgcolor='white'
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown('<div class="section-label">Average load by day type</div>', unsafe_allow_html=True)

df['dow'] = df['ts'].dt.dayofweek
df['day_type'] = df['dow'].apply(
    lambda d: 'Weekday' if d < 5 else ('Saturday' if d == 5 else 'Sunday')
)
summary = df.groupby('day_type')['load_pct'].agg(['mean', 'max', 'count']).reset_index()
summary.columns = ['Day Type', 'Avg Load %', 'Peak Load %', 'Readings']
summary['Avg Load %']  = summary['Avg Load %'].round(1)
summary['Peak Load %'] = summary['Peak Load %'].round(1)

b1, b2 = st.columns([1, 2])

with b1:
    st.dataframe(
        summary.set_index('Day Type'),
        use_container_width=True,
        height=140
    )

with b2:
    fig3 = go.Figure()
    colors = {'Weekday': '#3b82f6', 'Saturday': '#f59e0b', 'Sunday': '#8b5cf6'}
    for _, row in summary.iterrows():
        fig3.add_trace(go.Bar(
            x=[row['Day Type']],
            y=[row['Avg Load %']],
            name=row['Day Type'],
            marker_color=colors.get(row['Day Type'], '#888'),
            text=f"{row['Avg Load %']:.1f}%",
            textposition='outside',
            width=0.4
        ))
    fig3.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=160,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#888')),
        yaxis=dict(
            showgrid=True, gridcolor='#f0f0f0',
            ticksuffix='%', range=[0, 115],
            tickfont=dict(size=11, color='#888')
        ),
        bargap=0.4
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown('<div class="section-label">Recent readings</div>', unsafe_allow_html=True)

recent = df.tail(10).copy().sort_values('ts', ascending=False)
recent['ts']      = recent['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
recent['Status']  = recent['load_pct'].apply(level)
recent = recent[['ts', 'grams', 'load_pct', 'Status']].rename(columns={
    'ts': 'Timestamp', 'grams': 'Grams', 'load_pct': 'Load %'
})
st.dataframe(recent.set_index('Timestamp'), use_container_width=True, height=220)

time.sleep(10)
st.rerun()