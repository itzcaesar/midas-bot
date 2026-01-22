"""
MT5Bot Performance Dashboard
Real-time monitoring dashboard using Streamlit.

Run with: streamlit run src/dashboard/app.py
"""

from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try importing streamlit
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit and/or Plotly not installed. Run: pip install streamlit plotly")

import pandas as pd
from datetime import datetime, timedelta


def create_dashboard():
    """Main dashboard application."""
    
    if not STREAMLIT_AVAILABLE:
        print("Please install streamlit: pip install streamlit plotly")
        return
    
    # Page config
    st.set_page_config(
        page_title="MT5 Trading Bot Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .profit { color: #4CAF50; }
        .loss { color: #F44336; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🏆 MT5 Trading Bot Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Date range
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        st.divider()
        
        # Bot status
        st.header("🤖 Bot Status")
        status = st.selectbox("Status", ["Running", "Stopped", "Paused"])
        
        if status == "Running":
            st.success("● Bot is running")
        else:
            st.error(f"● Bot is {status.lower()}")
        
        st.divider()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.button("📊 Export Report"):
            st.info("Report exported!")
    
    # Main content
    # Load data (placeholder - would connect to database in real implementation)
    balance_data = generate_sample_data()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="💰 Balance",
            value=f"${balance_data['current_balance']:,.2f}",
            delta=f"{balance_data['daily_pnl']:+.2f}"
        )
    
    with col2:
        st.metric(
            label="📊 Total Trades",
            value=balance_data['total_trades'],
            delta=f"+{balance_data['today_trades']} today"
        )
    
    with col3:
        st.metric(
            label="✅ Win Rate",
            value=f"{balance_data['win_rate']:.1%}",
            delta=f"{balance_data['win_rate_change']:+.1%}"
        )
    
    with col4:
        st.metric(
            label="📈 Profit Factor",
            value=f"{balance_data['profit_factor']:.2f}",
            delta=f"{balance_data['pf_change']:+.2f}"
        )
    
    with col5:
        st.metric(
            label="📉 Max Drawdown",
            value=f"{balance_data['max_drawdown']:.1%}",
            delta=None
        )
    
    st.divider()
    
    # Charts row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 Equity Curve")
        
        # Generate sample equity curve
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        equity = generate_equity_curve(balance_data['current_balance'], 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='#1E88E5', width=2),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.1)'
        ))
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📊 Trade Distribution")
        
        # Pie chart
        labels = ['Winning Trades', 'Losing Trades']
        values = [balance_data['winning_trades'], balance_data['losing_trades']]
        colors = ['#4CAF50', '#F44336']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=colors
        )])
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recent trades table
    st.subheader("📋 Recent Trades")
    
    trades_df = generate_sample_trades()
    
    # Apply styling
    def color_profit(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}'
    
    styled_df = trades_df.style.applymap(color_profit, subset=['Profit'])
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    st.divider()
    
    # Bottom row - Additional charts
    bottom_col1, bottom_col2, bottom_col3 = st.columns(3)
    
    with bottom_col1:
        st.subheader("📅 Daily P/L")
        daily_pnl = generate_daily_pnl()
        
        colors = ['green' if x > 0 else 'red' for x in daily_pnl['pnl']]
        fig = go.Figure(data=[go.Bar(
            x=daily_pnl['date'],
            y=daily_pnl['pnl'],
            marker_color=colors
        )])
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with bottom_col2:
        st.subheader("⏰ Trades by Hour")
        hours = list(range(24))
        trade_count = [5, 3, 1, 0, 0, 2, 8, 12, 15, 18, 20, 22, 
                       25, 28, 30, 28, 22, 18, 15, 12, 10, 8, 6, 5]
        
        fig = go.Figure(data=[go.Bar(x=hours, y=trade_count)])
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Hour (UTC)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with bottom_col3:
        st.subheader("🎯 Signal Confidence")
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        win_rates = [45, 52, 61, 72, 85]
        
        fig = go.Figure(data=[go.Scatter(
            x=confidences,
            y=win_rates,
            mode='lines+markers',
            line=dict(color='#1E88E5')
        )])
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Confidence",
            yaxis_title="Win Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | MT5Bot Dashboard v1.0")


def generate_sample_data() -> dict:
    """Generate sample dashboard data."""
    return {
        'current_balance': 12543.67,
        'daily_pnl': 234.50,
        'total_trades': 156,
        'today_trades': 3,
        'win_rate': 0.62,
        'win_rate_change': 0.02,
        'profit_factor': 1.85,
        'pf_change': 0.12,
        'max_drawdown': 0.08,
        'winning_trades': 97,
        'losing_trades': 59
    }


def generate_equity_curve(final_balance: float, periods: int) -> list:
    """Generate sample equity curve."""
    import numpy as np
    
    start = final_balance * 0.85
    returns = np.random.normal(0.002, 0.01, periods)
    equity = [start]
    
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    
    # Scale to end at final balance
    scale = final_balance / equity[-1]
    return [e * scale for e in equity]


def generate_sample_trades() -> pd.DataFrame:
    """Generate sample trades data."""
    import numpy as np
    
    n_trades = 10
    directions = np.random.choice(['BUY', 'SELL'], n_trades)
    profits = np.random.normal(50, 100, n_trades)
    
    return pd.DataFrame({
        'Ticket': range(10001, 10001 + n_trades),
        'Time': pd.date_range(end=datetime.now(), periods=n_trades, freq='4H'),
        'Symbol': ['XAUUSD'] * n_trades,
        'Direction': directions,
        'Lot': [0.1] * n_trades,
        'Entry': np.random.uniform(1950, 2050, n_trades).round(2),
        'Exit': np.random.uniform(1950, 2050, n_trades).round(2),
        'Profit': profits.round(2)
    })


def generate_daily_pnl() -> pd.DataFrame:
    """Generate sample daily P/L data."""
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=14, freq='D')
    pnl = np.random.normal(50, 150, 14)
    
    return pd.DataFrame({
        'date': dates,
        'pnl': pnl.round(2)
    })


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        create_dashboard()
    else:
        print("Please install streamlit: pip install streamlit plotly")
