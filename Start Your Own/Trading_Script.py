"""Utilities for maintaining the ChatGPT micro cap portfolio.

The script processes portfolio positions, logs trades, and prints daily
results. It is intentionally lightweight and avoids changing existing
logic or behaviour.
"""

from datetime import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf
from typing import cast
import schedule
import time

# Shared file locations
DATA_DIR = "Start Your Own"
PORTFOLIO_CSV = f"{DATA_DIR}/chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = f"{DATA_DIR}/chatgpt_trade_log.csv"

# Today's date reused across logs
today = datetime.today().strftime("%Y-%m-%d")
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
market_summary = f"<p><b>Report generated:</b> {now}</p>"


def process_portfolio(portfolio: pd.DataFrame, starting_cash: float) -> pd.DataFrame:
    """Update daily price information, log stop-loss sells, and prompt for trades.

    The function iterates through each position, retrieves the latest close
    price and appends a summary row. Before processing, the user may record
    one or more manual buys or sells which are then applied to the portfolio.
    Results are appended to ``PORTFOLIO_CSV``.
    """
    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0
    cash = starting_cash

    while True:
        action = input(
            "Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: "
        ).strip().lower()
        if action == "b":
            try:
                ticker = input("Enter ticker symbol: ").strip().upper()
                shares = float(input("Enter number of shares: "))
                buy_price = float(input("Enter buy price: "))
                stop_loss = float(input("Enter stop loss: "))
                if shares <= 0 or buy_price <= 0 or stop_loss <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid input. Manual buy cancelled.")
            else:
                cash, portfolio = log_manual_buy(
                    buy_price, shares, ticker, stop_loss, cash, portfolio
                )
            continue
        if action == "s":
            try:
                ticker = input("Enter ticker symbol: ").strip().upper()
                shares = float(input("Enter number of shares to sell: "))
                sell_price = float(input("Enter sell price: "))
                if shares <= 0 or sell_price <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid input. Manual sell cancelled.")
            else:
                cash, portfolio = log_manual_sell(
                    sell_price, shares, ticker, cash, portfolio
                )
            continue
        break

    for _, stock in portfolio.iterrows():
        ticker = stock["ticker"]
        shares = int(stock["shares"])
        cost = stock["buy_price"]
        stop = stock["stop_loss"]
        data = yf.Ticker(ticker).history(period="1d")

        if data.empty:
            print(f"No data for {ticker}")
            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Cost Basis": cost,
                "Stop Loss": stop,
                "Current Price": "",
                "Total Value": "",
                "PnL": "",
                "Action": "NO DATA",
                "Cash Balance": "",
                "Total Equity": "",
            }
        else:
            price = round(data["Close"].iloc[-1], 2)
            value = round(price * shares, 2)
            pnl = round((price - cost) * shares, 2)

            if price <= stop:
                action = "SELL - Stop Loss Triggered"
                cash += value
                portfolio = log_sell(ticker, shares, price, cost, pnl, portfolio)
            else:
                action = "HOLD"
                total_value += value
                total_pnl += pnl

            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Cost Basis": cost,
                "Stop Loss": stop,
                "Current Price": price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": "",
            }

        results.append(row)

    # Append TOTAL summary row
    total_row = {
        "Date": today,
        "Ticker": "TOTAL",
        "Shares": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2),
        "Action": "",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    df = pd.DataFrame(results)
    if os.path.exists(PORTFOLIO_CSV):
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != today]
        print("rows for today already logged, not saving results to CSV...")
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(PORTFOLIO_CSV, index=False)
    return portfolio


def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Record a stop-loss sale in ``TRADE_LOG_CSV`` and remove the ticker."""
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }

    portfolio = portfolio[portfolio["ticker"] != ticker]

    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    return portfolio


def log_manual_buy(
    buy_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Log a manual purchase and append to the portfolio."""
    check = input(
        f"You are currently trying to buy {ticker}."
        " If this a mistake enter 1."
    )
    if check == "1":
        raise SystemExit("Please remove this function call.")

    data = yf.download(ticker, period="1d")
    data = cast(pd.DataFrame, data)
    if data.empty:
        SystemExit(f"error, could not find ticker {ticker}")
    if buy_price * shares > cash:
        SystemExit(
            f"error, you have {cash} but are trying to spend {buy_price * shares}. Are you sure you can do this?"
        )
    pnl = 0.0

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": buy_price,
        "Cost Basis": buy_price * shares,
        "PnL": pnl,
        "Reason": "MANUAL BUY - New position",
    }

    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    new_trade = {
        "ticker": ticker,
        "shares": shares,
        "stop_loss": stoploss,
        "buy_price": buy_price,
        "cost_basis": buy_price * shares,
    }
    chatgpt_portfolio = pd.concat(
        [chatgpt_portfolio, pd.DataFrame([new_trade])], ignore_index=True
    )
    cash = cash - shares * buy_price
    print(f"Manual buy for {ticker} complete!")
    return cash, chatgpt_portfolio


def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Log a manual sale and update the portfolio."""
    reason = input(
        f"You are currently trying to sell {ticker}.\nIf this is a mistake, enter 1. "
    )

    if reason == "1":
        raise SystemExit("Delete this function call from the program.")

    if isinstance(chatgpt_portfolio, list):
        chatgpt_portfolio = pd.DataFrame(chatgpt_portfolio)
    if ticker not in chatgpt_portfolio["ticker"].values:
        raise KeyError(f"error, could not find {ticker} in portfolio")
    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]

    total_shares = int(ticker_row["shares"].item())
    print(total_shares)
    if shares_sold > total_shares:
        raise ValueError(
            f"You are trying to sell {shares_sold} but only own {total_shares}."
        )
    buy_price = float(ticker_row["buy_price"].item())

    cost_basis = buy_price * shares_sold
    pnl = sell_price * shares_sold - cost_basis
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": "",
        "Buy Price": "",
        "Cost Basis": cost_basis,
        "PnL": pnl,
        "Reason": f"MANUAL SELL - {reason}",
        "Shares Sold": shares_sold,
        "Sell Price": sell_price,
    }
    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        ticker_row["shares"] = total_shares - shares_sold
        ticker_row["cost_basis"] = ticker_row["shares"] * ticker_row["buy_price"]

    cash = cash + shares_sold * sell_price
    print(f"manual sell for {ticker} complete!")
    return cash, chatgpt_portfolio


def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics."""
    if isinstance(chatgpt_portfolio, pd.DataFrame):
        portfolio_dict = chatgpt_portfolio.to_dict(orient="records")
    print(f"prices and updates for {today}")
    for stock in portfolio_dict + [{"ticker": "^RUT"}] + [{"ticker": "IWO"}] + [{"ticker": "XBI"}]:
        ticker = stock["ticker"]
        try:
            data = yf.download(ticker, period="2d", progress=False)
            data = cast(pd.DataFrame, data)
            if data.empty or len(data) < 2:
                print(f"Data for {ticker} was empty or incomplete.")
                continue
            price = float(data["Close"].iloc[-1].item())
            last_price = float(data["Close"].iloc[-2].item())

            percent_change = ((price - last_price) / last_price) * 100
            volume = float(data["Volume"].iloc[-1].item())
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e} Try checking internet connection.")
        print(f"{ticker} closing price: {price:.2f}")
        print(f"{ticker} volume for today: ${volume:,}")
        print(f"percent change from the day before: {percent_change:.2f}%")
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)

    # Filter TOTAL rows and get latest equity
    chatgpt_totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    chatgpt_totals["Date"] = pd.to_datetime(chatgpt_totals["Date"])
    final_date = chatgpt_totals["Date"].max()
    final_value = chatgpt_totals[chatgpt_totals["Date"] == final_date]
    final_equity = float(final_value["Total Equity"].values[0])
    equity_series = chatgpt_totals["Total Equity"].astype(float).reset_index(drop=True)

    # Daily returns
    daily_pct = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]

    # Number of total trading days
    n_days = len(chatgpt_totals)

    # Risk-free return over total trading period (assuming 4.5% risk-free rate)
    rf_annual = 0.045
    rf_period = (1 + rf_annual) ** (n_days / 252) - 1

    # Standard deviation of daily returns
    std_daily = daily_pct.std()
    negative_pct = daily_pct[daily_pct < 0]
    negative_std = negative_pct.std()
    # Sharpe Ratio
    sharpe_total = (total_return - rf_period) / (std_daily * np.sqrt(n_days))
    # Sortino Ratio
    sortino_total = (total_return - rf_period) / (negative_std * np.sqrt(n_days))

    # Output
    print(f"Total Sharpe Ratio over {n_days} days: {sharpe_total:.4f}")
    print(f"Total Sortino Ratio over {n_days} days: {sortino_total:.4f}")
    print(f"Latest ChatGPT Equity: ${final_equity:.2f}")
    # Get S&P 500 data
    spx = yf.download("^SPX", start="2025-06-27", end=final_date + pd.Timedelta(days=1), progress=False)
    spx = cast(pd.DataFrame, spx)
    spx = spx.reset_index()

    # Normalize to $100
    initial_price = spx["Close"].iloc[0].item()
    price_now = spx["Close"].iloc[-1].item()
    scaling_factor = 100 / initial_price
    spx_value = price_now * scaling_factor
    print(f"$100 Invested in the S&P 500: ${spx_value:.2f}")
    print(f"today's portfolio: {chatgpt_portfolio}")
    print(f"cash balance: {cash}")

    print(
        "Here are is your update for today. You can make any changes you see fit (if necessary),\n"
        "but you may not use deep research. You do have to ask premissons for any changes, as you have full control.\n"
        "You can however use the Internet and check current prices for potenial buys."
    )


def send_email(subject, html_body, to_email, from_email, from_password):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    part = MIMEText(html_body, "html")
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())


def dataframe_to_html(df):
    """Convert a pandas DataFrame to an HTML table with some basic styling."""
    return df.to_html(index=False, border=1, justify="center", classes="portfolio-table", escape=False)


def portfolio_to_custom_html(portfolio):
    total_cost_basis = 0
    total_risk = 0
    total_market_value = 0
    total_pnl = 0
    rows = ""
    for stock in portfolio:
        ticker = stock["ticker"]
        shares = stock["shares"]
        buy_price = stock["buy_price"]
        stop_loss = stock["stop_loss"]
        cost_basis = stock["cost_basis"]
        # Fetch live price and daily % change
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty and len(data) > 1:
                latest_price = round(data["Close"].iloc[-1], 2)
                prev_price = round(data["Close"].iloc[-2], 2)
                daily_pct = round(100 * (latest_price - prev_price) / prev_price, 2)
            elif not data.empty:
                latest_price = round(data["Close"].iloc[-1], 2)
                daily_pct = 0.0
            else:
                latest_price = "N/A"
                daily_pct = "N/A"
        except Exception:
            latest_price = "N/A"
            daily_pct = "N/A"
        # Calculate market value and PnL
        if isinstance(latest_price, float):
            market_value = round(shares * latest_price, 2)
            pnl = round((latest_price - buy_price) * shares, 2)
            total_market_value += market_value
            total_pnl += pnl
        else:
            market_value = "N/A"
            pnl = "N/A"
        risk_per_share = round(buy_price - stop_loss, 2)
        total_risk_stock = round(risk_per_share * shares, 2)
        total_cost_basis += cost_basis
        total_risk += total_risk_stock
        # Highlight if stop-loss triggered
        row_style = ""
        if isinstance(latest_price, float) and latest_price <= stop_loss:
            row_style = ' style="background-color:#ffcccc;"'
        pnl_style = 'color:red;' if isinstance(pnl, float) and pnl < 0 else ''
        rows += f"""
        <tr{row_style}>
            <td>{ticker}</td>
            <td>{shares}</td>
            <td>${buy_price:,.2f}</td>
            <td>${stop_loss:,.2f}</td>
            <td>💰 ${cost_basis:,.2f}</td>
            <td>📈 ${latest_price if latest_price == 'N/A' else f"{latest_price:,.2f}"}</td>
            <td>{daily_pct if daily_pct == 'N/A' else f"{daily_pct:.2f}%"}</td>
            <td>💵 ${market_value if market_value == 'N/A' else f"{market_value:,.2f}"}</td>
            <td><b style="{'color:red;' if pnl < 0 else ''}">📊 ${pnl if pnl == 'N/A' else f"{pnl:,.2f}"}</b></td>
            <td>${risk_per_share:,.2f}</td>
            <td>${total_risk_stock:,.2f}</td>
        </tr>
        """

    html = f"""
    <table border="1" style="border-collapse:collapse;text-align:center;">
        <tr>
            <th>Ticker</th>
            <th>Shares</th>
            <th>Buy Price</th>
            <th>Stop Loss</th>
            <th>💰 Cost Basis</th>
            <th>📈 Last Price</th>
            <th>📊 Daily % Change</th>
            <th>💵 Market Value</th>
            <th>📊 PnL</th>
            <th>📉 Risk (per share)</th>
            <th>⚠️ Total Risk</th>
        </tr>
        {rows}
        <tr>
            <td><b>Total</b></td>
            <td></td>
            <td></td>
            <td></td>
            <td><b>${total_cost_basis:,.2f}</b></td>
            <td></td>
            <td></td>
            <td><b>${total_market_value:,.2f}</b></td>
            <td><b style="{'color:red;' if total_pnl < 0 else ''}">${total_pnl:,.2f}</b></td>
            <td></td>
            <td><b>${total_risk:,.2f}</b></td>
        </tr>
    </table>
    """
    return html, total_market_value, total_pnl


def get_sp500_daily_change():
    try:
        data = yf.Ticker("^GSPC").history(period="2d")
        if not data.empty and len(data) > 1:
            latest = data["Close"].iloc[-1]
            prev = data["Close"].iloc[-2]
            pct = round(100 * (latest - prev) / prev, 2)
            return pct
    except Exception:
        pass
    return "N/A"


def suggest_microcap_momentum_stocks():
    # Example: S&P 500 tickers (for demo, use a small subset or load from a file)
    tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "AMD", "INTC", "IBM", "ORCL", "QCOM", "CSCO", "ADBE"]  # Add more as desired
    results = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="21d", interval="1d", progress=False, auto_adjust=False)
            if data.empty or len(data) < 15:
                continue
            price_now = data["Close"].iloc[-1]
            price_5d_ago = data["Close"].iloc[-6]
            pct_change = ((price_now - price_5d_ago) / price_5d_ago) * 100
            avg_volume = data["Volume"].tail(5).mean()
            rsi = get_rsi(data["Close"]).iloc[-1]
            results.append({
                "Ticker": ticker,
                "5d % Change": round(pct_change, 2),
                "Avg Vol (5d)": int(avg_volume),
                "RSI": round(rsi, 2)
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        momentum_html = "<p>No suggestions today.</p>"
    else:
        # Example filter: RSI between 30 and 70, volume above median
        median_vol = df["Avg Vol (5d)"].median()
        filtered = df[(df["RSI"] > 30) & (df["RSI"] < 70) & (df["Avg Vol (5d)"] > median_vol)]
        top = filtered.sort_values("5d % Change", ascending=False).head(3)
        momentum_html = "<h3>Momentum Screener: Top 3 Stocks (5-Day % Change, RSI, Volume)</h3>"
        momentum_html += top.to_html(index=False, border=1, justify="center")

    # Compose and send email
    send_email(
        subject="Daily Stock Suggestions - Momentum Screener",
        html_body=f"""
            <html>
            <body>
                <h2>Top 3 Momentum Stocks (Last 5 Days)</h2>
                {momentum_html}
                <p style="font-size:12px;color:gray;">This is an automated suggestion based on 5-day price momentum.</p>
            </body>
            </html>
        """,
        to_email="cristianursan81@gmail.com",
        from_email="cristianursan81@gmail.com",
        from_password="qvtv pebu bajp uoqp"
    )


def main() -> None:
    # Suggest stocks and email the suggestions
    suggestions = suggest_microcap_momentum_stocks()
    # Email credentials (replace with your own)
    to_email = "cristianursan81@gmail.com"
    from_email = "cristianursan81@gmail.com"
    from_password = "qvtv pebu bajp uoqp"  # Use an app password, not your main password!
    send_email(
        subject="Your Microcap Stock Suggestions",
        html_body=suggestions,  # <-- changed from body= to html_body=
        to_email=to_email,
        from_email=from_email,
        from_password=from_password
    )

    starting_capital = 100
    cash = starting_capital
    chatgpt_portfolio = [
        {"ticker": "AAPL", "shares": 5, "stop_loss": 150, "buy_price": 180, "cost_basis": 900},
        {"ticker": "TSLA", "shares": 5, "stop_loss": 500, "buy_price": 700, "cost_basis": 3500},
        {"ticker": "NVDA", "shares": 5, "stop_loss": 350, "buy_price": 450, "cost_basis": 2250},
        {"ticker": "DT",   "shares": 5, "stop_loss": 35,  "buy_price": 45,  "cost_basis": 225},
    ]
    chatgpt_portfolio = pd.DataFrame(chatgpt_portfolio)

    process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)


# Add at the bottom of your script
def job():
    # Place your main email-sending code here
    # For example:
    # main()
    pass

schedule.every().day.at("09:00").do(job)  # Set your desired time (24h format)

while True:
    schedule.run_pending()
    time.sleep(60)


if __name__ == "__main__":
    chatgpt_portfolio = [
        {"ticker": "AAPL", "shares": 5, "stop_loss": 150, "buy_price": 180, "cost_basis": 900},
        {"ticker": "TSLA", "shares": 5, "stop_loss": 500, "buy_price": 700, "cost_basis": 3500},
        {"ticker": "NVDA", "shares": 5, "stop_loss": 350, "buy_price": 450, "cost_basis": 2250},
        {"ticker": "DT",   "shares": 5, "stop_loss": 35,  "buy_price": 45,  "cost_basis": 225},
    ]
    portfolio_html, total_market_value, total_pnl = portfolio_to_custom_html(chatgpt_portfolio)
    sp500_change = get_sp500_daily_change()
    portfolio_descriptions = """
    <ul style="font-size:13px;">
      <li><b>Ticker</b>: Stock symbol</li>
      <li><b>Shares</b>: Number of shares held</li>
      <li><b>Buy Price</b>: Price per share at purchase</li>
      <li><b>Stop Loss</b>: Price at which to sell to limit loss</li>
      <li><b>💰 Cost Basis</b>: Total amount invested (Shares × Buy Price)</li>
      <li><b>📈 Last Price</b>: Most recent closing price</li>
      <li><b>📊 Daily % Change</b>: Change from previous close</li>
      <li><b>💵 Market Value</b>: Shares × Last Price</li>
      <li><b>📊 PnL</b>: (Last Price - Buy Price) × Shares</li>
      <li><b>📉 Risk (per share)</b>: Buy Price minus Stop Loss</li>
      <li><b>⚠️ Total Risk</b>: Risk per share × Shares</li>
    </ul>
    """
    analytics_html = f"""
    <h3>Advanced Analytics</h3>
    <ul style="font-size:13px;">
      <li><b>Portfolio Market Value:</b> ${total_market_value:,.2f}</li>
      <li><b>Portfolio Total PnL:</b> ${total_pnl:,.2f}</li>
      <li><b>S&amp;P 500 Daily % Change:</b> {sp500_change}%</li>
    </ul>
    """

    # --- Momentum Screener Section ---
    tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "AMD", "INTC", "IBM", "ORCL", "QCOM", "CSCO", "ADBE"]  # Add more as desired
    results = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="21d", interval="1d", progress=False, auto_adjust=False)
            if data.empty or len(data) < 15:
                continue
            price_now = data["Close"].iloc[-1]
            price_5d_ago = data["Close"].iloc[-6]
            pct_change = ((price_now - price_5d_ago) / price_5d_ago) * 100
            avg_volume = data["Volume"].tail(5).mean()
            rsi = get_rsi(data["Close"]).iloc[-1]
            results.append({
                "Ticker": ticker,
                "5d % Change": round(pct_change, 2),
                "Avg Vol (5d)": int(avg_volume),
                "RSI": round(rsi, 2)
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        momentum_html = "<p>No suggestions today.</p>"
    else:
        # Example filter: RSI between 30 and 70, volume above median
        median_vol = df["Avg Vol (5d)"].median()
        filtered = df[(df["RSI"] > 30) & (df["RSI"] < 70) & (df["Avg Vol (5d)"] > median_vol)]
        top = filtered.sort_values("5d % Change", ascending=False).head(3)
        momentum_html = "<h3>Momentum Screener: Top 3 Stocks (5-Day % Change, RSI, Volume)</h3>"
        momentum_html += top.to_html(index=False, border=1, justify="center")

    # --- Send the combined email ---
    send_email(
        subject="Your Portfolio & Momentum Screener",
        html_body=f"""
            <html>
            <body>
                {market_summary}
                <h2>Your Portfolio - {today}</h2>
                {portfolio_html}
                {portfolio_descriptions}
                {analytics_html}
                {momentum_html}
                <p style="font-size:12px;color:gray;">This is an automated message.</p>
            </body>
            </html>
        """,
        to_email="cristianursan81@gmail.com",
        from_email="cristianursan81@gmail.com",
        from_password="qvtv pebu bajp uoqp"
    )

