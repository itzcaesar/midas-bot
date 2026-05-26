"""
MIDAS Performance Dashboard (REQ-P1-10).
Real-time monitoring backed by the SQLite trade database.

Run with: streamlit run src/dashboard/app.py
"""
import sys
from pathlib import Path

# Add project root to path so config/ and src/ are importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from datetime import datetime, timedelta

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    print("Install dashboard deps: pip install streamlit plotly")
    raise SystemExit(1)

from config import settings
from core.database import Database, Trade, Signal


# ---------------------------------------------------------------------------
# Data layer — reads from the real SQLite DB
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db() -> Database:
    """Singleton DB connection for the dashboard."""
    return Database(db_path=settings.database_url.replace("sqlite:///", ""))


def load_trades(db: Database, days: int = 30) -> pd.DataFrame:
    """Load closed trades from the last N days into a DataFrame."""
    with db.get_session() as session:
        cutoff = datetime.utcnow() - timedelta(days=days)
        trades = (
            session.query(Trade)
            .filter(Trade.status == "CLOSED", Trade.exit_time >= cutoff)
            .order_by(Trade.exit_time.desc())
            .all()
        )
        if not trades:
            return pd.DataFrame()
        rows = [t.to_dict() for t in trades]
    df = pd.DataFrame(rows)
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_open_trades(db: Database) -> pd.DataFrame:
    with db.get_session() as session:
        trades = session.query(Trade).filter(Trade.status == "OPEN").all()
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in trades])


def load_signals(db: Database, days: int = 7) -> pd.DataFrame:
    with db.get_session() as session:
        cutoff = datetime.utcnow() - timedelta(days=days)
        signals = (
            session.query(Signal)
            .filter(Signal.timestamp >= cutoff)
            .order_by(Signal.timestamp.desc())
            .limit(200)
            .all()
        )
        if not signals:
            return pd.DataFrame()
        rows = [
            {
                "timestamp": s.timestamp,
                "symbol": s.symbol,
                "timeframe": s.timeframe,
                "direction": s.direction,
                "confidence": s.confidence,
                "executed": s.executed,
            }
            for s in signals
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="MIDAS Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # REQ-P2-08: Simple password-based auth gate.
    # For production, use streamlit-authenticator or reverse-proxy auth.
    import os
    dashboard_password = os.getenv("DASHBOARD_PASSWORD", "")
    if dashboard_password:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if not st.session_state.authenticated:
            st.title("MIDAS Dashboard - Login")
            pwd = st.text_input("Password", type="password")
            if st.button("Login"):
                if pwd == dashboard_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")
            return

    st.title("MIDAS Trading Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        days = st.slider("History (days)", 1, 90, 30)
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption(f"DB: {settings.database_url}")
        st.caption(f"Symbol: {settings.symbol} | TF: {settings.timeframe}")
        st.caption(f"Dry run: {'Yes' if settings.dry_run else 'No'}")

    db = get_db()
    trades_df = load_trades(db, days=days)
    open_df = load_open_trades(db)
    signals_df = load_signals(db, days=days)

    # --- Top metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)

    if trades_df.empty:
        total_pnl = 0.0
        win_rate = 0.0
        total_trades = 0
        profit_factor = 0.0
        max_dd = 0.0
    else:
        total_pnl = trades_df["profit"].sum()
        wins = trades_df[trades_df["profit"] > 0]
        losses = trades_df[trades_df["profit"] <= 0]
        total_trades = len(trades_df)
        win_rate = len(wins) / total_trades if total_trades else 0.0
        gross_profit = wins["profit"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["profit"].sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        # Max drawdown from cumulative PnL
        cum_pnl = trades_df.sort_values("exit_time")["profit"].cumsum()
        peak = cum_pnl.cummax()
        dd = (peak - cum_pnl) / (peak.replace(0, 1))
        max_dd = dd.max() if not dd.empty else 0.0

    col1.metric("Total P/L", f"${total_pnl:+,.2f}")
    col2.metric("Trades", str(total_trades))
    col3.metric("Win Rate", f"{win_rate:.1%}")
    col4.metric("Profit Factor", f"{profit_factor:.2f}")
    col5.metric("Max DD", f"{max_dd:.1%}")

    st.divider()

    # --- Equity curve ---
    chart_col, info_col = st.columns([2, 1])

    with chart_col:
        st.subheader("Equity Curve")
        if not trades_df.empty:
            sorted_trades = trades_df.sort_values("exit_time")
            cum = sorted_trades["profit"].cumsum()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sorted_trades["exit_time"],
                    y=cum,
                    mode="lines",
                    name="Cumulative P/L",
                    line=dict(color="#1E88E5", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(30, 136, 229, 0.1)",
                )
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Cumulative P/L ($)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet. Run the bot to generate data.")

    with info_col:
        st.subheader("Open Positions")
        if not open_df.empty:
            st.dataframe(
                open_df[["ticket", "symbol", "direction", "lot_size", "entry_price", "stop_loss", "take_profit"]],
                use_container_width=True,
                height=300,
            )
        else:
            st.info("No open positions.")

    st.divider()

    # --- Recent trades table ---
    st.subheader("Recent Trades")
    if not trades_df.empty:
        display_cols = [
            "ticket", "symbol", "direction", "lot_size",
            "entry_price", "exit_price", "profit", "exit_time",
        ]
        display_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(
            trades_df[display_cols].head(50),
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No trade history.")

    st.divider()

    # --- Signals log ---
    st.subheader("Recent Signals")
    if not signals_df.empty:
        st.dataframe(signals_df.head(50), use_container_width=True, height=250)
    else:
        st.info("No signals recorded yet.")

    # --- Daily P/L bar chart ---
    if not trades_df.empty and "exit_time" in trades_df.columns:
        st.subheader("Daily P/L")
        daily = trades_df.copy()
        daily["date"] = daily["exit_time"].dt.date
        daily_pnl = daily.groupby("date")["profit"].sum().reset_index()
        colors = ["green" if x > 0 else "red" for x in daily_pnl["profit"]]
        fig = go.Figure(
            data=[go.Bar(x=daily_pnl["date"], y=daily_pnl["profit"], marker_color=colors)]
        )
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.divider()
    st.caption(
        f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"MIDAS Dashboard v2.0"
    )


if __name__ == "__main__":
    main()
