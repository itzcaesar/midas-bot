# VPS Deployment Guide (REQ-P3-06)

## Requirements

- Windows VPS with MetaTrader 5 terminal installed
- Python 3.10+ 
- Latency budget: < 50ms to broker datacenter

## Recommended Setup

1. **VPS Provider**: Choose one co-located with your broker's datacenter
   - For ICMarkets: Equinix NY4/NY5 or LD4
   - For Pepperstone: Equinix NY4
   - Budget option: any Windows VPS with < 10ms ping to broker

2. **Installation**:
   ```bash
   git clone <repo>
   cd midas-bot
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   pip install pydantic-settings lightgbm optuna
   ```

3. **Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with real MT5 credentials
   # Set DRY_RUN=false only after shadow mode validation
   ```

4. **MT5 Terminal**:
   - Install MetaTrader 5 from your broker
   - Login with your trading account
   - Enable "Allow Algo Trading" in Tools > Options > Expert Advisors
   - Keep the terminal running (the bot connects via IPC)

5. **Running**:
   ```bash
   # Start the bot (use a process manager like nssm or PM2 for persistence)
   python src/main.py

   # Start the dashboard (separate terminal)
   streamlit run src/dashboard/app.py --server.port 8501
   ```

6. **Process Management** (recommended):
   ```bash
   # Install NSSM (Non-Sucking Service Manager)
   nssm install MidasBot "C:\path\to\.venv\Scripts\python.exe" "C:\path\to\src\main.py"
   nssm set MidasBot AppDirectory "C:\path\to\midas-bot"
   nssm start MidasBot
   ```

7. **Monitoring**:
   - Dashboard at http://localhost:8501 (set DASHBOARD_PASSWORD in .env)
   - Audit logs in logs/audit/
   - Trade logs in logs/trades.csv
   - Set up Discord/Telegram alerts for errors

## Latency Optimization

- Use `ORDER_FILLING_IOC` (already configured)
- Keep MT5 terminal on the same machine as the bot
- Minimize other processes on the VPS
- Use SSD storage for the database
- Consider running the bot with `--optimize` flag for PyPy compatibility

## Security Checklist

- [ ] .env file has restrictive permissions (icacls .env /grant:r %USERNAME%:R)
- [ ] Dashboard behind password or VPN
- [ ] MT5 terminal set to "read-only" mode during maintenance
- [ ] Regular credential rotation
- [ ] Audit logs backed up to separate storage
