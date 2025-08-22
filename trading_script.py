"""Utilities for maintaining the ChatGPT micro cap portfolio.

The script processes portfolio positions, logs trades, and prints daily
results. It is intentionally lightweight and avoids changing existing
logic or behaviour.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any, cast
import os
import time
import requests
import json
import logging
from functools import wraps
from typing import Dict, Any
import argparse
import sys
import psutil  # You may need to install this: pip install psutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Shared file locations
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # Save files in the same folder as this script
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"


def set_data_dir(data_dir: Path) -> None:
    """Update global paths for portfolio and trade logs.

    Parameters
    ----------
    data_dir:
        Directory where ``chatgpt_portfolio_update.csv`` and
        ``chatgpt_trade_log.csv`` are stored.
    """

    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    DATA_DIR = Path(data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"

# Today's date reused across logs
today = datetime.today().strftime("%Y-%m-%d")
now = datetime.now()
day = now.weekday()



def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Update daily price information, log stop-loss sells, and prompt for trades.

    Parameters
    ----------
    portfolio:
        Current holdings provided as a DataFrame, mapping of column names to
        lists, or a list of row dictionaries. The input is normalised to a
        ``DataFrame`` before any processing so that downstream code only deals
        with a single type.
    cash:
        Cash balance available for trading.
    interactive:
        When ``True`` (default) the function prompts for manual trades via
        ``input``. Set to ``False`` to skip all interactive prompts – useful
        when the function is driven by a user interface or automated tests.

    Returns
    -------
    tuple[pd.DataFrame, float]
        Updated portfolio and cash balance.
    """
    print(portfolio)
    if isinstance(portfolio, pd.DataFrame):
        portfolio_df = portfolio.copy()
    elif isinstance(portfolio, (dict, list)):
        portfolio_df = pd.DataFrame(portfolio)
    else:  # pragma: no cover - defensive type check
        raise TypeError("portfolio must be a DataFrame, dict, or list of dicts")

    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    if day == 6 or day == 5 and interactive:
        check = input(
            """Today is currently a weekend, so markets were never open.
This will cause the program to calculate data from the last day (usually Friday), and save it as today.
Are you sure you want to do this? To exit, enter 1. """
        )
        if check == "1":
            raise SystemError("Exitting program...")

    if interactive:
        while True:
            action = input(
                f""" You have {cash} in cash.
Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: """
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
                    cash, portfolio_df = log_manual_buy(
                        buy_price,
                        shares,
                        ticker,
                        stop_loss,
                        cash,
                        portfolio_df,
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
                    cash, portfolio_df = log_manual_sell(
                        sell_price,
                        shares,
                        ticker,
                        cash,
                        portfolio_df,
                    )
                continue
            break
    print(portfolio_df)
    for _, stock in portfolio_df.iterrows():
        ticker = stock["ticker"]
        shares = int(stock["shares"])
        cost = stock["buy_price"]
        cost_basis = stock["cost_basis"]
        stop = stock["stop_loss"]
        data = yf.Ticker(ticker).history(period="1d")

        if data.empty:
            print(f"No data for {ticker}")
            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": cost,
                "Cost Basis": cost_basis,
                "Stop Loss": stop,
                "Current Price": "",
                "Total Value": "",
                "PnL": "",
                "Action": "NO DATA",
                "Cash Balance": "",
                "Total Equity": "",
            }
        else:
            low_price = round(float(data["Low"].iloc[-1]), 2)
            close_price = round(float(data["Close"].iloc[-1]), 2)

            if low_price <= stop:
                price = stop
                value = round(price * shares, 2)
                pnl = round((price - cost) * shares, 2)
                action = "SELL - Stop Loss Triggered"
                cash += value
                portfolio_df = log_sell(ticker, shares, price, cost, pnl, portfolio_df)
            else:
                price = close_price
                value = round(price * shares, 2)
                pnl = round((price - cost) * shares, 2)
                action = "HOLD"
                total_value += value
                total_pnl += pnl

            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": cost,
                "Cost Basis": cost_basis,
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
        "Buy Price": "",
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
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != today]
        print("Saving results to CSV...")
        time.sleep(1)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(PORTFOLIO_CSV, index=False)
    return portfolio_df, cash


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
    print(f"{ticker} stop loss was met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
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
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Log a manual purchase and append to the portfolio."""

    if interactive:
        check = input(
            f"""You are currently trying to buy {shares} shares of {ticker} with a price of {buy_price} and a stoploss of {stoploss}.
        If this a mistake, type "1". """
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    # Ensure DataFrame exists with required columns
    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])

    # Download current market data
    data = yf.download(ticker, period="1d", auto_adjust=False, progress=False)
    data = cast(pd.DataFrame, data)

    if data.empty:
        print(f"Manual buy for {ticker} failed: no market data available.")
        return cash, chatgpt_portfolio

    day_high = float(data["High"].iloc[-1].item())
    day_low = float(data["Low"].iloc[-1].item())

    if not (day_low <= buy_price <= day_high):
        print(
            f"Manual buy for {ticker} at {buy_price} failed: price outside today's range {round(day_low, 2)}-{round(day_high, 2)}."
        )
        return cash, chatgpt_portfolio

    if buy_price * shares > cash:
        print(
            f"Manual buy for {ticker} failed: cost {buy_price * shares} exceeds cash balance {cash}."
        )
        return cash, chatgpt_portfolio

    # Log trade to trade log CSV
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
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    # === Update portfolio DataFrame ===
    rows = chatgpt_portfolio.loc[
        chatgpt_portfolio["ticker"].astype(str).str.upper() == ticker.upper()
    ]

    if rows.empty:
        # New position
        new_trade = {
            "ticker": ticker,
            "shares": float(shares),
            "stop_loss": float(stoploss),
            "buy_price": float(buy_price),
            "cost_basis": float(buy_price * shares),
        }
        if chatgpt_portfolio.empty:
            chatgpt_portfolio = pd.DataFrame([new_trade])
        else:
            chatgpt_portfolio = pd.concat([chatgpt_portfolio, pd.DataFrame([new_trade])], ignore_index=True)
    else:
        # Add to existing position — recompute weighted avg price
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])

        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(buy_price * shares)
        avg_price = new_cost / new_shares if new_shares else 0.0

        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = avg_price
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss)

    # Deduct cash
    cash -= shares * buy_price
    print(f"Manual buy for {ticker} complete!")
    return cash, chatgpt_portfolio



def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    reason: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Log a manual sale and update the portfolio.

    Parameters
    ----------
    reason:
        Description of why the position is being sold. Ignored when
        ``interactive`` is ``True``.
    interactive:
        When ``False`` no interactive confirmation is requested.
    """
    if interactive:
        reason = input(
            f"""You are currently trying to sell {shares_sold} shares of {ticker} at a price of {sell_price}.
If this is a mistake, enter 1. """
        )

        if reason == "1":
            print("Returning...")
            return cash, chatgpt_portfolio
    elif reason is None:
        reason = ""
    if ticker not in chatgpt_portfolio["ticker"].values:
        print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
        return cash, chatgpt_portfolio
    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]

    total_shares = int(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(
            f"Manual sell for {ticker} failed: trying to sell {shares_sold} shares but only own {total_shares}."
        )
        return cash, chatgpt_portfolio
    data = yf.download(ticker, period="1d")
    data = cast(pd.DataFrame, data)
    if data.empty:
        print(f"Manual sell for {ticker} failed: no market data available.")
        return cash, chatgpt_portfolio
    day_high = float(data["High"].iloc[-1])
    day_low = float(data["Low"].iloc[-1])
    if not (day_low <= sell_price <= day_high):
        print(
            f"Manual sell for {ticker} at {sell_price} failed: price outside today's range {round(day_low, 2)}-{round(day_high, 2)}."
        )
        return cash, chatgpt_portfolio
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
        row_index = ticker_row.index[0]
        chatgpt_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        chatgpt_portfolio.at[row_index, "cost_basis"] = (
            chatgpt_portfolio.at[row_index, "shares"]
            * chatgpt_portfolio.at[row_index, "buy_price"]
        )

    cash = cash + shares_sold * sell_price
    print(f"manual sell for {ticker} complete!")
    return cash, chatgpt_portfolio


def get_ollama_analysis(portfolio_data: str, market_summary: str, model: str = "llama3.2:3b") -> str:
    """Get AI analysis from local Ollama instance."""
    try:
        prompt = f"""
        Analyze this trading portfolio performance:
        
        {portfolio_data}
        
        Market Context: {market_summary}
        
        Provide a brief analysis (3-4 sentences) focusing on:
        1. Portfolio risk assessment
        2. Performance vs market
        3. One specific actionable recommendation
        
        Be concise and professional.
        """
        
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9
                }
            })
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return "AI analysis unavailable (connection error)"
    except Exception as e:
        return f"AI analysis unavailable: {str(e)[:50]}..."


def ai_stock_screener(portfolio_tickers: list[str], model: str = "phi3:mini") -> str:
    """Use Ollama to suggest potential trades based on current portfolio."""
    try:
        prompt = f"""
        Current portfolio holdings: {', '.join(portfolio_tickers)}
        
        Based on current market conditions and momentum trading principles:
        1. Suggest 1-2 stocks to consider for momentum trading
        2. Identify any sector concentration risks
        3. Recommend position sizing
        
        Keep response under 80 words. Focus on actionable insights.
        """
        
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,  # Lower temperature for more focused suggestions
                }
            })
            
        return response.json()['response'].strip() if response.status_code == 200 else "Stock screener unavailable"
    except:
        return "AI stock screener unavailable"


# Modify your daily_results function to include AI analysis
def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics."""
    portfolio_dict: list[dict[str, object]] = chatgpt_portfolio.to_dict(orient="records")

    print(f"prices and updates for {today}")
    
    # ... existing price display code ...
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
    
    # ... existing performance calculation code ...
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)

    # Use only TOTAL rows, sorted by date
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    totals["Date"] = pd.to_datetime(totals["Date"])
    totals = totals.sort_values("Date")
    final_equity = float(totals.iloc[-1]["Total Equity"])
    equity = totals["Total Equity"].astype(float).reset_index(drop=True)

    # Daily simple returns
    r = equity.pct_change().dropna()
    n_days = len(r)

    # Config
    rf_annual = 0.045

    # Risk-free aligned to frequency and window
    rf_daily  = (1 + rf_annual)**(1 / 252) - 1
    rf_period = (1 + rf_daily)**n_days - 1

    # Stats
    mean_daily = r.mean()
    std_daily  = r.std(ddof=1)

    # Downside deviation vs MAR = rf_daily
    downside = (r - rf_daily).clip(upper=0)
    downside_std = (downside.pow(2).mean())**0.5

    # total return over the window
    period_return = (1 + r).prod() - 1

    # --- Sharpe ---
    sharpe_period = (period_return - rf_period) / (std_daily * np.sqrt(n_days))
    sharpe_annual = ((mean_daily - rf_daily) / std_daily) * np.sqrt(252)

    # --- Sortino ---
    sortino_period = (period_return - rf_period) / (downside_std * np.sqrt(n_days))
    sortino_annual = ((mean_daily - rf_daily) / downside_std) * np.sqrt(252)

    # Output existing metrics
    print(f"Total Sharpe Ratio over {n_days} days: {sharpe_period:.4f}")
    print(f"Total Sortino Ratio over {n_days} days: {sortino_period:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe_annual:.4f}")
    print(f"Annualized Sortino Ratio: {sortino_annual:.4f}")
    print(f"Latest ChatGPT Equity: ${final_equity:.2f}")
    
    # Get S&P 500 data
    final_date = totals.loc[totals.index[-1], "Date"]
    spx = yf.download("^SPX", start="2025-06-27", end=final_date + pd.Timedelta(days=1), progress=False)
    spx = cast(pd.DataFrame, spx)
    spx = spx.reset_index()

    # Normalize to $100
    initial_price = spx["Close"].iloc[0].item()
    price_now = spx["Close"].iloc[-1].item()
    scaling_factor = 100 / initial_price
    spx_value = price_now * scaling_factor
    print(f"$100 Invested in the S&P 500: ${spx_value:.2f}")
    
    # ===================
    # NEW: AI ANALYSIS SECTION
    # ===================
    print("\n" + "="*60)
    print("🤖 LOCAL AI ANALYSIS (Ollama)")
    print("="*60)
    
    # Prepare portfolio summary for AI
    portfolio_summary = f"""
Portfolio Value: ${final_equity:.2f}
Cash: ${cash:.2f} 
Period Return: {period_return*100:.2f}%
Sharpe Ratio: {sharpe_annual:.3f}
Holdings: {[stock['ticker'] for stock in portfolio_dict]}
"""
    
    market_context = f"S&P 500 equivalent: ${spx_value:.2f} (vs $100 baseline)"
    
    # Get AI analysis
    print("📊 Portfolio Analysis (llama3.2:3b):")
    ai_analysis = get_ollama_analysis(portfolio_summary, market_context)
    print(ai_analysis)
    
    print("\n💡 Trading Suggestions (phi3:mini):")
    current_tickers = [stock['ticker'] for stock in portfolio_dict]
    if current_tickers:  # Only if we have positions
        ai_suggestions = ai_stock_screener(current_tickers)
        print(ai_suggestions)
    else:
        # For empty portfolio, provide cash deployment suggestions
        print("Portfolio is 100% cash. Consider:")
        print("• SPY (S&P 500 ETF) for broad market exposure")
        print("• Start with 60-80% allocation to reduce cash drag")
        print("• Use dollar-cost averaging for initial deployment")
    
    print("="*60)
    # ===================
    # END AI SECTION
    # ===================
    
    print("today's portfolio:")
    print(chatgpt_portfolio)
    print(f"cash balance: {cash}")
    print(
        "Today's update:\n"
        "You have complete control over every decision. Make any changes you believe are beneficial—no approval required.\n"
        "Deep research is not permitted.\n"
        "Act at your discretion to achieve the best outcome.\n"
        "If you do not make a clear indiction you want to change position IMMEDIATELY after this message, portfolio will remain unchanged for tommorow.\n"
        "You are encouraged to use the internet to check current prices (and related up-to-date info) for potential buys.\n"
        "*Paste everything above into ChatGPT*"
    )

def main(file: str, data_dir: Path | None = None) -> None:
    """Run the trading script.

    Parameters
    ----------
    file:
        CSV file containing historical portfolio records.
    data_dir:
        Directory where trade and portfolio CSVs will be stored.
    """
    chatgpt_portfolio, cash = load_latest_portfolio_state(file)
    if data_dir is not None:
        set_data_dir(data_dir)

    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)

def load_latest_portfolio_state(
    file: str,
) -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """Load the most recent portfolio snapshot and cash balance.

    Parameters
    ----------
    file:
        CSV file containing historical portfolio records.

    Returns
    -------
    tuple[pd.DataFrame | list[dict[str, Any]], float]
        A representation of the latest holdings (either an empty DataFrame or a
        list of row dictionaries) and the associated cash balance.
    """

    df = pd.read_csv(file)
    if df.empty:
        portfolio = pd.DataFrame([])
        print(
            "Portfolio CSV is empty. Returning set amount of cash for creating portfolio."
        )
        try:
            cash = float(input("What would you like your starting cash amount to be? "))
        except ValueError:
            raise ValueError(
                "Cash could not be converted to float datatype. Please enter a valid number."
            )
        return portfolio, cash
    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"])

    latest_date = non_total["Date"].max()
    # Get all tickers from the latest date
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    sold_mask = latest_tickers["Action"].astype(str).str.startswith("SELL")
    latest_tickers = latest_tickers[~sold_mask].copy()
    latest_tickers.drop(columns=["Date", "Cash Balance", "Total Equity", "Action", "Current Price", "PnL", "Total Value"], inplace=True)
    latest_tickers.rename(columns={"Cost Basis": "cost_basis", "Buy Price": "buy_price", "Shares": "shares", "Ticker": "ticker", "Stop Loss": "stop_loss"}, inplace=True)
    latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient='records')
    df = df[df["Ticker"] == "TOTAL"]  # Only the total summary rows
    df["Date"] = pd.to_datetime(df["Date"])
    latest = df.sort_values("Date").iloc[-1]
    cash = float(latest["Cash Balance"])
    return latest_tickers, cash

# Set up proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'trading.log'),
        logging.StreamHandler()
    ]
)

def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=2)
def get_stock_data(ticker: str, period: str = "1d"):
    """Robust stock data fetching with retries"""
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        raise ValueError(f"No data available for {ticker}")
    return data

def validate_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean portfolio data"""
    required_columns = ['ticker', 'shares', 'buy_price', 'stop_loss', 'cost_basis']
    
    # Check required columns
    missing_cols = set(required_columns) - set(portfolio.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Data type validation
    portfolio['shares'] = pd.to_numeric(portfolio['shares'], errors='coerce')
    portfolio['buy_price'] = pd.to_numeric(portfolio['buy_price'], errors='coerce')
    portfolio['stop_loss'] = pd.to_numeric(portfolio['stop_loss'], errors='coerce')
    portfolio['cost_basis'] = pd.to_numeric(portfolio['cost_basis'], errors='coerce')
    
    # Remove invalid rows
    invalid_mask = portfolio[['shares', 'buy_price', 'stop_loss', 'cost_basis']].isnull().any(axis=1)
    if invalid_mask.any():
        logging.warning(f"Removing {invalid_mask.sum()} invalid portfolio rows")
        portfolio = portfolio[~invalid_mask]
    
    # Validate business logic
    portfolio['ticker'] = portfolio['ticker'].str.upper()
    
    # Check for negative values
    if (portfolio[['shares', 'buy_price', 'stop_loss']] <= 0).any().any():
        logging.warning("Found negative or zero values in portfolio")
    
    return portfolio.reset_index(drop=True)

def calculate_advanced_metrics(returns: pd.Series) -> dict:
    """Calculate comprehensive portfolio metrics"""
    if len(returns) < 2:
        return {}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk metrics
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    max_drawdown = (returns.cumsum().expanding().max() - returns.cumsum()).max()
    
    # Performance ratios
    calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    # Win/loss ratios
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    win_rate = winning_days / len(returns) if len(returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_volatility': volatility,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'winning_days': winning_days,
        'losing_days': losing_days
    }

def get_market_context() -> dict:
    """Get broader market context for analysis"""
    benchmarks = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ',
        'IWM': 'Russell 2000',
        'VIX': 'Volatility Index'
    }
    
    context = {}
    for ticker, name in benchmarks.items():
        try:
            data = yf.download(ticker, period="5d", progress=False)
            if not data.empty:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2] if len(data) > 1 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                context[name] = {'price': current, 'daily_change': change}
        except Exception as e:
            logging.warning(f"Failed to get data for {ticker}: {e}")
    
    return context

def get_enhanced_ai_analysis(portfolio_data: str, market_context: dict, performance_metrics: dict, model: str = "llama3.2:3b") -> str:
    """Enhanced AI analysis with richer context"""
    
    # Format market context
    market_summary = "\n".join([
        f"{name}: {data['price']:.2f} ({data['daily_change']:+.2f}%)" 
        for name, data in market_context.items()
    ])
    
    # Format performance metrics
    metrics_summary = f"""
Win Rate: {performance_metrics.get('win_rate', 0):.1%}
Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
Volatility: {performance_metrics.get('annualized_volatility', 0):.1%}
VaR (95%): {performance_metrics.get('var_95', 0):.2%}"""

    prompt = f"""
    PORTFOLIO ANALYSIS REQUEST
    
    Current Holdings: {portfolio_data}
    
    Market Context:
    {market_summary}
    
    Performance Metrics:
    {metrics_summary}
    
    As a quantitative analyst, provide:
    1. Risk assessment (High/Medium/Low) with key concerns
    2. Portfolio diversification analysis
    3. Market timing considerations based on current conditions
    4. One specific, actionable recommendation for the next trading day
    
    Format: Risk level first, then 3-4 concise sentences. Focus on actionable insights.
    """
    
    return get_ollama_response(prompt, model)

def get_ollama_response(prompt: str, model: str, timeout: int = 30) -> str:
    """Robust Ollama API call with timeout and error handling"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 200  # Limit response length
                }
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            logging.error(f"Ollama API error: {response.status_code}")
            return f"AI analysis unavailable (HTTP {response.status_code})"
    
    except requests.exceptions.Timeout:
        return "AI analysis timeout - Ollama may be busy"
    except requests.exceptions.ConnectionError:
        return "AI analysis unavailable - check Ollama connection"
    except Exception as e:
        logging.error(f"Ollama error: {e}")
        return f"AI analysis error: {str(e)[:50]}..."

def display_portfolio_summary(portfolio: pd.DataFrame, cash: float, metrics: dict):
    """Pretty print portfolio with colors and formatting"""
    print("\n" + "="*70)
    print("📊 PORTFOLIO SUMMARY")
    print("="*70)
    
    if portfolio.empty:
        print("No positions currently held.")
        print(f"💰 Cash: ${cash:,.2f}")
        return
    
    # Calculate totals
    total_value = (portfolio['shares'] * portfolio['current_price']).sum() if 'current_price' in portfolio.columns else 0
    total_equity = total_value + cash
    
    print(f"💰 Cash Balance: ${cash:,.2f}")
    print(f"📈 Position Value: ${total_value:,.2f}")
    print(f"💎 Total Equity: ${total_equity:,.2f}")
    print(f"📊 Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"⚠️  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print()
    
    # Position details
    for _, position in portfolio.iterrows():
        pnl = (position.get('current_price', position['buy_price']) - position['buy_price']) * position['shares']
        pnl_pct = (pnl / (position['buy_price'] * position['shares'])) * 100
        
        status = "🟢" if pnl >= 0 else "🔴"
        print(f"{status} {position['ticker']}: {position['shares']} shares @ ${position['buy_price']:.2f}")
        print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | Stop: ${position['stop_loss']:.2f}")

class TradingConfig:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = Path(config_file)  # Convert to Path object
        self.default_config = {
            "starting_cash": 10000.0,
            "default_stop_loss_pct": 0.10,
            "min_cash_reserve": 1000.0,
            "max_position_size_pct": 0.20,
            "ai_timeout_seconds": 120,
            # Email settings
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",  # Use app password for Gmail
                "recipient_email": "",
                "send_daily_report": True,
                "send_trade_alerts": True,
                "send_ai_analysis": True
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                return {**self.default_config, **config}
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logging.warning(f"Error loading config file {self.config_file}: {e}")
                logging.info("Using default configuration")
                self.save_config(self.default_config)
                return self.default_config.copy()
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save current configuration to file"""
        config_to_save = config or self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config to {self.config_file}: {e}")
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'ai.enabled')"""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

# Usage
config = TradingConfig()

def backup_data(backup_dir: Path = None):
    """Create timestamped backups of important data"""
    backup_dir = backup_dir or DATA_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files_to_backup = [PORTFOLIO_CSV, TRADE_LOG_CSV]
    
    for file_path in files_to_backup:
        if file_path.exists():
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name
            backup_path.write_text(file_path.read_text())
            logging.info(f"Backed up {file_path.name} to {backup_name}")

def auto_backup_decorator(func):
    """Decorator to automatically backup data before risky operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        backup_data()
        return func(*args, **kwargs)
    return wrapper

def initialize_portfolio(starting_cash: float = None, force: bool = False):
    """Initialize a new portfolio with starting cash"""
    
    # Check if files already exist
    if not force and (PORTFOLIO_CSV.exists() or TRADE_LOG_CSV.exists()):
        existing_files = []
        if PORTFOLIO_CSV.exists():
            existing_files.append(str(PORTFOLIO_CSV))
        if TRADE_LOG_CSV.exists():
            existing_files.append(str(TRADE_LOG_CSV))
        
        print(f"⚠️  Existing files found: {', '.join(existing_files)}")
        response = input("Do you want to overwrite? This will DELETE all existing data! (type 'YES' to confirm): ")
        if response != 'YES':
            print("❌ Initialization cancelled.")
            return False
    
    # Get starting cash if not provided
    if starting_cash is None:
        while True:
            try:
                starting_cash = float(input("Enter starting cash amount: $"))
                if starting_cash <= 0:
                    print("❌ Starting cash must be greater than 0")
                    continue
                break
            except ValueError:
                print("❌ Please enter a valid number")
    
    # Create backup if files exist
    if PORTFOLIO_CSV.exists() or TRADE_LOG_CSV.exists():
        print("📦 Creating backup of existing files...")
        backup_data()
    
    # Create empty portfolio CSV with required structure
    portfolio_df = pd.DataFrame(columns=[
        'Date', 'Ticker', 'Shares', 'Buy Price', 'Cost Basis', 
        'Stop Loss', 'Current Price', 'Total Value', 'PnL', 
        'Action', 'Cash Balance', 'Total Equity'
    ])
    
    # Add initial TOTAL row with starting cash
    initial_total = {
        'Date': today,
        'Ticker': 'TOTAL',
        'Shares': '',
        'Buy Price': '',
        'Cost Basis': '',
        'Stop Loss': '',
        'Current Price': '',
        'Total Value': 0.00,
        'PnL': 0.00,
        'Action': 'INITIALIZE',
        'Cash Balance': starting_cash,
        'Total Equity': starting_cash
    }
    
    portfolio_df = pd.concat([portfolio_df, pd.DataFrame([initial_total])], ignore_index=True)
    portfolio_df.to_csv(PORTFOLIO_CSV, index=False)
    
    # Create empty trade log CSV
    trade_log_df = pd.DataFrame(columns=[
        'Date', 'Ticker', 'Shares Bought', 'Buy Price', 'Cost Basis', 
        'Shares Sold', 'Sell Price', 'PnL', 'Reason'
    ])
    trade_log_df.to_csv(TRADE_LOG_CSV, index=False)
    
    # Create default config file
    config = TradingConfig()
    
    print("✅ Portfolio initialized successfully!")
    print(f"📁 Portfolio file: {PORTFOLIO_CSV}")
    print(f"📁 Trade log file: {TRADE_LOG_CSV}")
    print(f"💰 Starting cash: ${starting_cash:,.2f}")
    print(f"⚙️  Config file: {config.config_file}")
    
    return True

def stop_trading_operations():
    """Stop any running trading operations and show status"""
    
    print("🛑 STOP TRADING OPERATIONS")
    print("=" * 50)
    
    # Check if any processes might be running (this is basic detection)
    import psutil
    current_pid = os.getpid()
    python_processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('trading_script' in arg for arg in cmdline):
                    if proc.info['pid'] != current_pid:  # Don't include current process
                        python_processes.append(proc.info)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    if python_processes:
        print(f"⚠️  Found {len(python_processes)} other trading script processes:")
        for proc in python_processes:
            print(f"   PID {proc['pid']}: {' '.join(proc['cmdline'])}")
        
        response = input("Do you want to terminate these processes? (y/N): ")
        if response.lower() in ['y', 'yes']:
            terminated = 0
            for proc in python_processes:
                try:
                    process = psutil.Process(proc['pid'])
                    process.terminate()
                    process.wait(timeout=5)  # Wait up to 5 seconds
                    terminated += 1
                    print(f"✅ Terminated PID {proc['pid']}")
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    print(f"❌ Failed to terminate PID {proc['pid']}")
            print(f"🛑 Terminated {terminated} processes")
        else:
            print("ℹ️  No processes terminated")
    else:
        print("✅ No other trading script processes found")
    
    # Show current portfolio status
    print("\n📊 CURRENT PORTFOLIO STATUS")
    print("=" * 50)
    
    if not PORTFOLIO_CSV.exists():
        print("❌ No portfolio file found")
        return
    
    try:
        df = pd.read_csv(PORTFOLIO_CSV)
        if df.empty:
            print("📭 Portfolio is empty")
            return
        
        # Get latest total row
        total_rows = df[df['Ticker'] == 'TOTAL']
        if total_rows.empty:
            print("⚠️  No portfolio summary found")
            return
        
        latest_total = total_rows.iloc[-1]
        
        print(f"💰 Cash Balance: ${latest_total['Cash Balance']:,.2f}")
        print(f"📈 Total Value: ${latest_total['Total Value']:,.2f}")
        print(f"💎 Total Equity: ${latest_total['Total Equity']:,.2f}")
        print(f"📅 Last Update: {latest_total['Date']}")
        
        # Show current positions
        latest_date = pd.to_datetime(latest_total['Date'])
        positions = df[
            (df['Ticker'] != 'TOTAL') & 
            (pd.to_datetime(df['Date']) == latest_date) &
            (~df['Action'].astype(str).str.contains('SELL', na=False))
        ]
        
        if not positions.empty:
            print(f"\n🎯 Current Positions ({len(positions)}):")
            for _, pos in positions.iterrows():
                pnl_str = f"${pos['PnL']:+,.2f}" if pd.notna(pos['PnL']) else "N/A"
                print(f"   {pos['Ticker']}: {pos['Shares']} shares @ ${pos['Buy Price']:.2f} (P&L: {pnl_str})")
        else:
            print("\n📭 No open positions")
            
    except Exception as e:
        print(f"❌ Error reading portfolio: {e}")
    
    print("\n🛑 Trading operations stopped")

def show_status():
    """Show comprehensive status of trading system"""
    
    print("📊 TRADING SYSTEM STATUS")
    print("=" * 60)
    
    # File status
    print("📁 FILES:")
    files_status = [
        (PORTFOLIO_CSV, "Portfolio CSV"),
        (TRADE_LOG_CSV, "Trade Log CSV"),
        (Path("trading_config.json"), "Configuration"),
        (DATA_DIR / "trading.log", "Log File")
    ]
    
    for file_path, description in files_status:
        if file_path.exists():
            size = file_path.stat().st_size
            modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"   ✅ {description}: {file_path} ({size} bytes, modified {modified.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"   ❌ {description}: Not found")
    
    # AI status
    print("\n🤖 AI SERVICES:")
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   ✅ Ollama: Running ({len(models)} models available)")
            for model in models:
                print(f"      - {model['name']} ({model['size']} bytes)")
        else:
            print("   ⚠️  Ollama: Connected but response error")
    except requests.exceptions.ConnectionError:
        print("   ❌ Ollama: Not running or not accessible")
    except Exception as e:
        print(f"   ⚠️  Ollama: Error checking status ({e})")
    
    # Market status (basic check)
    print("\n📈 MARKET DATA:")
    try:
        test_data = yf.download("AAPL", period="1d", progress=False)
        if not test_data.empty:
            print("   ✅ Yahoo Finance: Accessible")
        else:
            print("   ⚠️  Yahoo Finance: No data returned")
    except Exception as e:
        print(f"   ❌ Yahoo Finance: Error ({e})")
    
    # Recent activity
    print("\n📊 RECENT ACTIVITY:")
    if PORTFOLIO_CSV.exists():
        try:
            df = pd.read_csv(PORTFOLIO_CSV)
            recent_trades = df.tail(5)
            if not recent_trades.empty:
                print("   Last 5 entries:")
                for _, trade in recent_trades.iterrows():
                    action_str = f"({trade['Action']})" if pd.notna(trade['Action']) and trade['Action'] else ""
                    print(f"      {trade['Date']}: {trade['Ticker']} {action_str}")
            else:
                print("   No recent activity")
        except Exception as e:
            print(f"   ❌ Error reading recent activity: {e}")
    else:
        print("   No portfolio file found")

# Add this email function
def send_email_report(portfolio_analysis: str, trading_suggestions: str, portfolio_summary: dict, config: TradingConfig):
    """Send email report with AI analysis and portfolio summary"""
    
    if not config.config["email"]["enabled"]:
        return
    
    email_config = config.config["email"]
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config["sender_email"]
        msg['To'] = email_config["recipient_email"]
        msg['Subject'] = f"Daily Trading Report - {datetime.today().strftime('%Y-%m-%d')}"
        
        # Create HTML email body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .portfolio-summary {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
                .ai-analysis {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
                .suggestions {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
                pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Daily Trading Report</h1>
                <p><strong>Date:</strong> {datetime.today().strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="section portfolio-summary">
                <h2>💰 Portfolio Summary</h2>
                <ul>
                    <li><strong>Total Equity:</strong> ${portfolio_summary.get('total_equity', 0):,.2f}</li>
                    <li><strong>Cash Balance:</strong> ${portfolio_summary.get('cash_balance', 0):,.2f}</li>
                    <li><strong>Total Value:</strong> ${portfolio_summary.get('total_value', 0):,.2f}</li>
                    <li><strong>Today's P&L:</strong> ${portfolio_summary.get('pnl', 0):+,.2f}</li>
                </ul>
            </div>
            
            <div class="section ai-analysis">
                <h2>🤖 AI Portfolio Analysis</h2>
                <pre>{portfolio_analysis}</pre>
            </div>
            
            <div class="section suggestions">
                <h2>💡 Trading Suggestions</h2>
                <pre>{trading_suggestions}</pre>
            </div>
            
            <div class="section">
                <h2>📈 Current Positions</h2>
                <p>See attached CSV file for detailed portfolio data.</p>
            </div>
            
            <hr>
            <p><em>Generated by ChatGPT Micro Cap Trading Script</em></p>
        </body>
        </html>
        """
        
        # Attach HTML body
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach CSV file
        try:
            with open(PORTFOLIO_CSV, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= portfolio_{datetime.today().strftime("%Y%m%d")}.csv',
                )
                msg.attach(part)
        except Exception as e:
            logging.warning(f"Could not attach CSV file: {e}")
        
        # Send email
        server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
        server.starttls()
        server.login(email_config["sender_email"], email_config["sender_password"])
        text = msg.as_string()
        server.sendmail(email_config["sender_email"], email_config["recipient_email"], text)
        server.quit()
        
        print("✅ Email report sent successfully!")
        logging.info(f"Email report sent to {email_config['recipient_email']}")
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        logging.error(f"Email sending failed: {e}")

# Add this function to extract portfolio summary
def get_portfolio_summary(portfolio_df: pd.DataFrame) -> dict:
    """Extract portfolio summary for email"""
    if portfolio_df.empty:
        return {"total_equity": 0, "cash_balance": 0, "total_value": 0, "pnl": 0}
    
    total_row = portfolio_df[portfolio_df['Ticker'] == 'TOTAL'].iloc[-1]
    
    return {
        "total_equity": float(total_row.get('Total Equity', 0)),
        "cash_balance": float(total_row.get('Cash Balance', 0)),
        "total_value": float(total_row.get('Total Value', 0)),
        "pnl": float(total_row.get('PnL', 0))
    }

# Add this at the end of the daily_results function, incorporating AI analysis:

def daily_results(file_path: str, data_dir: Path = None):
    # ... existing code ...
    
    # After the AI analysis section, add:

    # Ensure ai_suggestions is defined
    if 'ai_suggestions' not in locals():
        ai_suggestions = "No trading suggestions available."

    # Send email report if enabled
    if config.config["email"]["enabled"]:
        portfolio_df = pd.read_csv(PORTFOLIO_CSV)
        portfolio_summary = get_portfolio_summary(portfolio_df)
        # Ensure ai_analysis is defined
        if 'ai_analysis' not in locals():
            ai_analysis = "No AI analysis available."

        send_email_report(
            portfolio_analysis=ai_analysis,
            trading_suggestions=ai_suggestions,
            portfolio_summary=portfolio_summary,
            config=config
        )

# Remove these lines from the bottom of your script (around line 1080+):
# config = {
#     "email": {
#         "enabled": True,
#         ...
#     }
# }
# 
# # Update your existing config file
# with open("trading_config.json", "r") as f:
#     existing_config = json.load(f)
# 
# existing_config.update(config)
# 
# with open("trading_config.json", "w") as f:
#     json.dump(existing_config, f, indent=2)
# 
# print("Email configuration added!")

# Also remove the global config line (around line 940):
# config = TradingConfig()

# Update the create_cli function to include new commands:
def create_cli():
    """Create command line interface"""
    parser = argparse.ArgumentParser(description='Trading Script with AI Analysis')
    parser.add_argument('--file', type=str, 
                       default='chatgpt_portfolio_update.csv',
                       help='Portfolio CSV file (default: chatgpt_portfolio_update.csv)')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    parser.add_argument('--backup', action='store_true', help='Create backup before running')
    parser.add_argument('--config', type=str, default='trading_config.json', help='Config file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')
    
    # New commands
    parser.add_argument('--init', action='store_true', help='Initialize new portfolio')
    parser.add_argument('--starting-cash', type=float, help='Starting cash for initialization')
    parser.add_argument('--force-init', action='store_true', help='Force initialization (overwrite existing files)')
    parser.add_argument('--stop', action='store_true', help='Stop trading operations and show status')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--send-email', action='store_true', help='Send email report')
    
    return parser

# Add the main execution block at the very end:
if __name__ == "__main__":
    parser = create_cli()
    args = parser.parse_args()
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Load configuration AFTER parsing arguments
    config = TradingConfig(args.config)
    
    # Set data directory if specified
    if args.data_dir:
        set_data_dir(Path(args.data_dir))
    
    # Handle special commands
    if args.init:
        if initialize_portfolio(args.starting_cash, args.force_init):
            print("\n💡 You can now run the script normally:")
            print(f"   python {sys.argv[0]} --file {args.file}")
        exit(0)
    
    if args.stop:
        stop_trading_operations()
        exit(0)
    
    if args.status:
        show_status()
        exit(0)
    
    if args.send_email:
        try:
            if not PORTFOLIO_CSV.exists():
                print("❌ No portfolio file found for email report")
                exit(1)
                
            df = pd.read_csv(PORTFOLIO_CSV)
            portfolio_summary = get_portfolio_summary(df)
            send_email_report(
                portfolio_analysis="Test email - Portfolio analysis would go here",
                trading_suggestions="Test email - Trading suggestions would go here", 
                portfolio_summary=portfolio_summary,
                config=config
            )
            print("📧 Test email sent!")
        except Exception as e:
            print(f"❌ Email test failed: {e}")
        exit(0)
    
    # Create backup if requested
    if args.backup:
        backup_data()
    
    # Check if portfolio file exists
    if not Path(args.file).exists():
        print(f"❌ Portfolio file '{args.file}' not found!")
        print("💡 Initialize a new portfolio with:")
        print(f"   python {sys.argv[0]} --init")
        exit(1)
    
    # Run main logic
    chatgpt_portfolio, cash = load_latest_portfolio_state(args.file)
    if args.data_dir:
        set_data_dir(Path(args.data_dir))

    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash)
    
    # Only run AI analysis if not disabled
    if not args.no_ai:
        daily_results(chatgpt_portfolio, cash)
    else:
        print("AI analysis disabled. Portfolio processing complete.")

