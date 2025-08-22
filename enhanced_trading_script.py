"""Enhanced Trading Script with AI Analysis and Comprehensive Features

The script processes portfolio positions, logs trades, provides AI analysis,
and offers advanced analytics with robust error handling and monitoring.
"""

from datetime import datetime
from pathlib import Path
import argparse
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==============================================
# CONFIGURATION MANAGEMENT
# ==============================================

class TradingConfig:
    """Centralized configuration management"""
    
    DEFAULT_CONFIG = {
        'ai': {
            'enabled': True,
            'primary_model': 'llama3.2:3b',
            'fallback_model': 'phi3:mini',
            'timeout': 30,
            'ollama_url': 'http://localhost:11434/api/generate'
        },
        'trading': {
            'max_position_size': 0.15,  # 15% of portfolio
            'default_stop_loss_pct': 0.15,  # 15% stop loss
            'cash_reserve_pct': 0.05,  # Keep 5% cash
            'risk_free_rate': 0.045  # 4.5% annual
        },
        'data': {
            'retry_attempts': 3,
            'retry_delay': 2,
            'backup_enabled': True,
            'max_backups': 10
        },
        'notifications': {
            'log_level': 'INFO',
            'alert_near_stop_loss': 0.05  # Alert when within 5% of stop loss
        },
        'market': {
            'trading_hours_only': False,
            'weekend_warning': True
        }
    }
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                merged = self.DEFAULT_CONFIG.copy()
                self._deep_update(merged, config)
                return merged
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Error loading config: {e}. Using defaults.")
        
        self.save_config(self.DEFAULT_CONFIG)
        return self.DEFAULT_CONFIG.copy()
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save current configuration to file"""
        config_to_save = config or self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except IOError as e:
            logging.error(f"Failed to save config: {e}")
    
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


# ==============================================
# LOGGING AND ERROR HANDLING
# ==============================================

def setup_logging(config: TradingConfig, log_dir: Path = None):
    """Set up comprehensive logging"""
    log_dir = log_dir or Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_level = getattr(logging, config.get('notifications.log_level', 'INFO').upper())
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_dir / 'trading.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / 'errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_handler)


def retry_on_failure(max_retries: int = 3, delay: float = 1, backoff: float = 2):
    """Decorator to retry failed operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


# ==============================================
# DATA MANAGEMENT AND VALIDATION
# ==============================================

class DataManager:
    """Handle all data operations with validation and backup"""
    
    def __init__(self, config: TradingConfig, script_dir: Path = None):
        self.config = config
        self.script_dir = script_dir or Path(__file__).resolve().parent
        self.data_dir = self.script_dir
        self.portfolio_csv = self.data_dir / "chatgpt_portfolio_update.csv"
        self.trade_log_csv = self.data_dir / "chatgpt_trade_log.csv"
        self.backup_dir = self.data_dir / "backups"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        if config.get('data.backup_enabled'):
            self.backup_dir.mkdir(exist_ok=True)
    
    def set_data_dir(self, data_dir: Path):
        """Update data directory paths"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.portfolio_csv = self.data_dir / "chatgpt_portfolio_update.csv"
        self.trade_log_csv = self.data_dir / "chatgpt_trade_log.csv"
        self.backup_dir = self.data_dir / "backups"
        if self.config.get('data.backup_enabled'):
            self.backup_dir.mkdir(exist_ok=True)
    
    def validate_portfolio(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean portfolio data"""
        if portfolio.empty:
            return portfolio
        
        required_columns = ['ticker', 'shares', 'buy_price', 'stop_loss', 'cost_basis']
        
        # Check required columns
        missing_cols = set(required_columns) - set(portfolio.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Data type validation
        numeric_cols = ['shares', 'buy_price', 'stop_loss', 'cost_basis']
        for col in numeric_cols:
            portfolio[col] = pd.to_numeric(portfolio[col], errors='coerce')
        
        # Remove invalid rows
        invalid_mask = portfolio[numeric_cols].isnull().any(axis=1)
        if invalid_mask.any():
            logging.warning(f"Removing {invalid_mask.sum()} invalid portfolio rows")
            portfolio = portfolio[~invalid_mask]
        
        # Validate business logic
        portfolio['ticker'] = portfolio['ticker'].str.upper()
        
        # Check for negative values
        negative_mask = (portfolio[numeric_cols] <= 0).any(axis=1)
        if negative_mask.any():
            logging.warning(f"Found {negative_mask.sum()} rows with negative/zero values")
            portfolio = portfolio[~negative_mask]
        
        return portfolio.reset_index(drop=True)
    
    def backup_data(self):
        """Create timestamped backups of important data"""
        if not self.config.get('data.backup_enabled'):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_to_backup = [self.portfolio_csv, self.trade_log_csv]
        
        for file_path in files_to_backup:
            if file_path.exists():
                backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                backup_path = self.backup_dir / backup_name
                backup_path.write_text(file_path.read_text())
                logging.info(f"Backed up {file_path.name} to {backup_name}")
        
        # Clean old backups
        self._cleanup_old_backups()
    
    def _cleanup_old_backups(self):
        """Remove old backup files beyond the configured limit"""
        max_backups = self.config.get('data.max_backups', 10)
        
        for pattern in ['chatgpt_portfolio_update_*.csv', 'chatgpt_trade_log_*.csv']:
            backup_files = sorted(self.backup_dir.glob(pattern), reverse=True)
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                logging.debug(f"Removed old backup: {old_backup.name}")
    
    def auto_backup(self, func):
        """Decorator to automatically backup data before risky operations"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.backup_data()
            return func(*args, **kwargs)
        return wrapper


# ==============================================
# MARKET DATA AND ANALYSIS
# ==============================================

class MarketDataProvider:
    """Enhanced market data provider with error handling and caching"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
    
    @retry_on_failure(max_retries=3, delay=1)
    def get_stock_data(self, ticker: str, period: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        """Get stock data with retry logic and caching"""
        cache_key = f"{ticker}_{period}"
        current_time = time.time()
        
        # Check cache
        if use_cache and cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if current_time - timestamp < self._cache_timeout:
                return data
        
        # Fetch new data
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, auto_adjust=False)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Cache the result
            if use_cache:
                self._cache[cache_key] = (data, current_time)
            
            logging.debug(f"Fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logging.error(f"Failed to fetch data for {ticker}: {e}")
            raise
    
    def get_current_price(self, ticker: str) -> float:
        """Get current/last closing price for a ticker"""
        data = self.get_stock_data(ticker, "1d")
        return float(data["Close"].iloc[-1])
    
    def get_market_context(self) -> Dict[str, Dict[str, float]]:
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
                data = self.get_stock_data(ticker, "5d")
                if len(data) >= 2:
                    current = float(data['Close'].iloc[-1])
                    prev = float(data['Close'].iloc[-2])
                    change = ((current - prev) / prev * 100) if prev != 0 else 0
                    context[name] = {'price': current, 'daily_change': change}
            except Exception as e:
                logging.warning(f"Failed to get market context for {ticker}: {e}")
        
        return context
    
    def validate_trade_price(self, ticker: str, price: float) -> bool:
        """Validate if a trade price is within today's range"""
        try:
            data = self.get_stock_data(ticker, "1d")
            day_high = float(data["High"].iloc[-1])
            day_low = float(data["Low"].iloc[-1])
            return day_low <= price <= day_high
        except Exception as e:
            logging.warning(f"Could not validate price for {ticker}: {e}")
            return True  # Allow trade if we can't validate


# ==============================================
# ADVANCED ANALYTICS
# ==============================================

class PortfolioAnalytics:
    """Advanced portfolio analytics and performance metrics"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.risk_free_rate = config.get('trading.risk_free_rate', 0.045)
    
    def calculate_comprehensive_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        if len(returns) < 2:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Performance ratios
        rf_daily = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = returns - rf_daily
        
        sharpe_ratio = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = returns[returns < rf_daily]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std != 0 else 0
        
        # Calmar ratio
        calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else np.inf
        
        # Win/loss statistics
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_days > 0 else 0
        profit_factor = abs(avg_win * winning_days / (avg_loss * losing_days)) if avg_loss != 0 and losing_days > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'winning_days': int(winning_days),
            'losing_days': int(losing_days),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def check_portfolio_alerts(self, portfolio: pd.DataFrame, market_data_provider: MarketDataProvider) -> List[str]:
        """Check for various portfolio alerts"""
        alerts = []
        
        if portfolio.empty:
            return alerts
        
        max_position = self.config.get('trading.max_position_size', 0.15)
        stop_loss_threshold = self.config.get('notifications.alert_near_stop_loss', 0.05)
        
        # Calculate total portfolio value
        total_value = 0
        for _, position in portfolio.iterrows():
            try:
                current_price = market_data_provider.get_current_price(position['ticker'])
                position_value = position['shares'] * current_price
                total_value += position_value
            except Exception as e:
                logging.warning(f"Could not get price for {position['ticker']}: {e}")
        
        # Position size alerts
        for _, position in portfolio.iterrows():
            try:
                current_price = market_data_provider.get_current_price(position['ticker'])
                position_value = position['shares'] * current_price
                position_pct = position_value / total_value if total_value > 0 else 0
                
                if position_pct > max_position:
                    alerts.append(f"⚠️ {position['ticker']} is {position_pct:.1%} of portfolio (max: {max_position:.1%})")
                
                # Stop loss proximity alerts
                stop_distance = (current_price - position['stop_loss']) / current_price
                if stop_distance < stop_loss_threshold:
                    alerts.append(f"🚨 {position['ticker']} is near stop loss: ${current_price:.2f} vs ${position['stop_loss']:.2f}")
                    
            except Exception as e:
                logging.warning(f"Could not analyze position {position['ticker']}: {e}")
        
        return alerts


# ==============================================
# AI ANALYSIS INTEGRATION
# ==============================================

class AIAnalyst:
    """Enhanced AI analysis with multiple models and robust error handling"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = config.get('ai.enabled', True)
        self.primary_model = config.get('ai.primary_model', 'llama3.2:3b')
        self.fallback_model = config.get('ai.fallback_model', 'phi3:mini')
        self.timeout = config.get('ai.timeout', 30)
        self.ollama_url = config.get('ai.ollama_url', 'http://localhost:11434/api/generate')
    
    def _make_ollama_request(self, prompt: str, model: str, max_tokens: int = 200) -> str:
        """Make request to Ollama API with error handling"""
        if not self.enabled:
            return "AI analysis disabled in configuration"
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': max_tokens
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                logging.error(f"Ollama API error: {response.status_code}")
                return f"AI analysis unavailable (HTTP {response.status_code})"
                
        except requests.exceptions.Timeout:
            return f"AI analysis timeout after {self.timeout}s - Ollama may be busy"
        except requests.exceptions.ConnectionError:
            return "AI analysis unavailable - check Ollama connection"
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            return f"AI analysis error: {str(e)[:50]}..."
    
    def get_portfolio_analysis(self, portfolio_summary: str, market_context: Dict, performance_metrics: Dict) -> str:
        """Get comprehensive portfolio analysis from AI"""
        market_summary = "\n".join([
            f"{name}: ${data['price']:.2f} ({data['daily_change']:+.2f}%)"
            for name, data in market_context.items()
        ])
        
        metrics_summary = f"""
Performance Metrics:
- Win Rate: {performance_metrics.get('win_rate', 0):.1%}
- Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
- Volatility: {performance_metrics.get('annualized_volatility', 0):.1%}
- VaR (95%): {performance_metrics.get('var_95', 0):.2%}"""

        prompt = f"""
PORTFOLIO ANALYSIS REQUEST

Portfolio Summary:
{portfolio_summary}

Market Context:
{market_summary}

{metrics_summary}

As a quantitative analyst, provide a comprehensive analysis:

1. RISK ASSESSMENT (High/Medium/Low): Key risk factors and concerns
2. PERFORMANCE EVALUATION: How the portfolio is performing relative to market and metrics
3. DIVERSIFICATION ANALYSIS: Sector concentration and balance issues
4. MARKET TIMING: Current market conditions and their impact
5. ACTIONABLE RECOMMENDATION: One specific trade or adjustment for tomorrow

Format: Start with risk level, then provide analysis in 4-5 concise sentences. Focus on actionable insights based on the data provided.
"""
        
        # Try primary model first, fallback to secondary if needed
        result = self._make_ollama_request(prompt, self.primary_model, 250)
        
        if "unavailable" in result or "error" in result:
            logging.info(f"Primary AI model failed, trying fallback: {self.fallback_model}")
            result = self._make_ollama_request(prompt, self.fallback_model, 200)
        
        return result
    
    def get_stock_screening_suggestions(self, current_tickers: List[str]) -> str:
        """Get AI-powered stock screening suggestions"""
        prompt = f"""
STOCK SCREENING REQUEST

Current Portfolio Holdings: {', '.join(current_tickers)}
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

Based on current market conditions and momentum trading principles, provide:

1. SECTOR ANALYSIS: Are we overconcentrated in any sectors?
2. MOMENTUM OPPORTUNITIES: 2-3 specific stocks showing strong momentum (provide tickers)
3. POSITION SIZING: Recommended allocation percentages
4. RISK MANAGEMENT: Any positions that should be reduced or hedged?

Keep response under 120 words. Focus on specific, actionable recommendations with ticker symbols where applicable.
"""
        
        return self._make_ollama_request(prompt, self.fallback_model, 150)
    
    def get_risk_assessment(self, portfolio_data: str, alerts: List[str]) -> str:
        """Get AI risk assessment based on current portfolio and alerts"""
        alerts_text = "\n".join(alerts) if alerts else "No current alerts"
        
        prompt = f"""
RISK ASSESSMENT REQUEST

Portfolio Data:
{portfolio_data}

Current Alerts:
{alerts_text}

Provide a focused risk assessment:
1. Immediate risks that need attention
2. Position sizing recommendations
3. Stop-loss adjustments if needed
4. Overall risk level (Low/Medium/High/Critical)

Maximum 80 words. Be specific and actionable.
"""
        
        return self._make_ollama_request(prompt, self.fallback_model, 100)


# ==============================================
# ENHANCED TRADING OPERATIONS
# ==============================================

class TradingEngine:
    """Enhanced trading engine with validation and logging"""
    
    def __init__(self, config: TradingConfig, data_manager: DataManager, market_data: MarketDataProvider):
        self.config = config
        self.data_manager = data_manager
        self.market_data = market_data
        self.today = datetime.today().strftime("%Y-%m-%d")
        self.analytics = PortfolioAnalytics(config)
    
    def process_portfolio(
        self,
        portfolio: Union[pd.DataFrame, Dict, List],
        cash: float,
        interactive: bool = True
    ) -> Tuple[pd.DataFrame, float]:
        """Enhanced portfolio processing with comprehensive validation and logging"""
        
        # Normalize input to DataFrame
        if isinstance(portfolio, pd.DataFrame):
            portfolio_df = portfolio.copy()
        elif isinstance(portfolio, (dict, list)):
            portfolio_df = pd.DataFrame(portfolio)
        else:
            raise TypeError("portfolio must be a DataFrame, dict, or list of dicts")
        
        # Validate portfolio data
        portfolio_df = self.data_manager.validate_portfolio(portfolio_df)
        
        logging.info(f"Processing portfolio with {len(portfolio_df)} positions and ${cash:.2f} cash")
        
        # Weekend check
        if self._is_weekend() and interactive and self.config.get('market.weekend_warning'):
            if not self._confirm_weekend_processing():
                raise SystemExit("Exiting due to weekend processing cancellation")
        
        # Interactive trading
        if interactive:
            cash, portfolio_df = self._handle_manual_trades(cash, portfolio_df)
        
        # Process existing positions
        results = []
        total_value = 0.0
        total_pnl = 0.0
        
        for _, position in portfolio_df.iterrows():
            result_row = self._process_position(position, cash)
            
            # Handle stop loss triggers
            if result_row['Action'] == 'SELL - Stop Loss Triggered':
                cash += result_row['Total Value']
                portfolio_df = self._remove_position(portfolio_df, position['ticker'])
            else:
                total_value += result_row.get('Total Value', 0)
                total_pnl += result_row.get('PnL', 0)
            
            results.append(result_row)
        
        # Add summary row
        total_row = self._create_summary_row(total_value, total_pnl, cash)
        results.append(total_row)
        
        # Save results
        self._save_portfolio_results(results)
        
        logging.info(f"Portfolio processing complete. Total value: ${total_value:.2f}, Cash: ${cash:.2f}")
        return portfolio_df, cash
    
    def _is_weekend(self) -> bool:
        """Check if today is weekend"""
        return datetime.now().weekday() in [5, 6]  # Saturday, Sunday
    
    def _confirm_weekend_processing(self) -> bool:
        """Confirm weekend processing with user"""
        response = input(
            "Today is a weekend - markets are closed. This will use the last trading day's data. "
            "Continue? (y/N): "
        ).strip().lower()
        return response in ['y', 'yes']
    
    def _handle_manual_trades(self, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Handle interactive manual trading"""
        while True:
            action = input(
                f"\n💰 Cash available: ${cash:,.2f}\n"
                "Manual trade options:\n"
                "  'b' - Buy position\n"
                "  's' - Sell position\n"
                "  'v' - View portfolio\n"
                "  Enter - Continue\n"
                "Choice: "
            ).strip().lower()
            
            if action == 'b':
                cash, portfolio = self._handle_manual_buy(cash, portfolio)
            elif action == 's':
                cash, portfolio = self._handle_manual_sell(cash, portfolio)
            elif action == 'v':
                self._display_portfolio_summary(portfolio, cash)
            elif action == '':
                break
            else:
                print("Invalid option. Please try again.")
        
        return cash, portfolio
    
    def _handle_manual_buy(self, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Handle manual buy orders with comprehensive validation"""
        try:
            ticker = input("Enter ticker symbol: ").strip().upper()
            shares = float(input("Enter number of shares: "))
            buy_price = float(input("Enter buy price: "))
            stop_loss = float(input("Enter stop loss price: "))
            
            # Validation
            if shares <= 0 or buy_price <= 0 or stop_loss <= 0:
                raise ValueError("All values must be positive")
            
            if stop_loss >= buy_price:
                print("Warning: Stop loss is above buy price")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            # Cost check
            total_cost = buy_price * shares
            if total_cost > cash:
                print(f"❌ Insufficient cash: ${total_cost:.2f} required, ${cash:.2f} available")
                return cash, portfolio
            
            # Price validation
            if not self.market_data.validate_trade_price(ticker, buy_price):
                print(f"⚠️ Price ${buy_price:.2f} may be outside today's trading range")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            # Position size check
            max_position = self.config.get('trading.max_position_size', 0.15)
            portfolio_value = cash + self._calculate_portfolio_value(portfolio)
            position_pct = total_cost / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > max_position:
                print(f"⚠️ This position would be {position_pct:.1%} of portfolio (max recommended: {max_position:.1%})")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            # Execute buy
            cash, portfolio = self._execute_buy(ticker, shares, buy_price, stop_loss, cash, portfolio)
            print(f"✅ Buy order executed: {shares} shares of {ticker} at ${buy_price:.2f}")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except Exception as e:
            print(f"❌ Buy order failed: {e}")
            logging.error(f"Manual buy failed: {e}")
        
        return cash, portfolio
    
    def _handle_manual_sell(self, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Handle manual sell orders with validation"""
        if portfolio.empty:
            print("❌ No positions to sell")
            return cash, portfolio
        
        print("\nCurrent positions:")
        for i, (_, pos) in enumerate(portfolio.iterrows()):
            print(f"  {i+1}. {pos['ticker']}: {pos['shares']} shares @ ${pos['buy_price']:.2f}")
        
        try:
            ticker = input("Enter ticker to sell: ").strip().upper()
            
            if ticker not in portfolio['ticker'].values:
                print(f"❌ {ticker} not found in portfolio")
                return cash, portfolio
            
            position = portfolio[portfolio['ticker'] == ticker].iloc[0]
            max_shares = position['shares']
            
            shares = float(input(f"Enter shares to sell (max {max_shares}): "))
            sell_price = float(input("Enter sell price: "))
            
            if shares <= 0 or sell_price <= 0:
                raise ValueError("Values must be positive")
            
            if shares > max_shares:
                raise ValueError(f"Cannot sell {shares} shares, only {max_shares} available")
            
            # Price validation
            if not self.market_data.validate_trade_price(ticker, sell_price):
                print(f"⚠️ Price ${sell_price:.2f} may be outside today's trading range")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            reason = input("Reason for sale (optional): ").strip() or "Manual sell"
            
            # Execute sell
            cash, portfolio = self._execute_sell(ticker, shares, sell_price, reason, cash, portfolio)
            print(f"✅ Sell order executed: {shares} shares of {ticker} at ${sell_price:.2f}")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except Exception as e:
            print(f"❌ Sell order failed: {e}")
            logging.error(f"Manual sell failed: {e}")
        
        return cash, portfolio
    
    def _display_portfolio_summary(self, portfolio: pd.DataFrame, cash: float):
        """Display formatted portfolio summary"""
        print("\n" + "="*60)
        print("📊 CURRENT PORTFOLIO SUMMARY")
        print("="*60)
        
        if portfolio.empty:
            print("No positions currently held.")
            print(f"💰 Cash: ${cash:,.2f}")
            return
        
        total_value = self._calculate_portfolio_value(portfolio)
        total_equity = total_value + cash
        
        print(f"💰 Cash Balance: ${cash:,.2f}")
        print(f"📈 Position Value: ${total_value:,.2f}")
        print(f"💎 Total Equity: ${total_equity:,.2f}")
        print()
        
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                pnl = (current_price - pos['buy_price']) * pos['shares']
                pnl_pct = (pnl / (pos['buy_price'] * pos['shares'])) * 100
                
                status = "🟢" if pnl >= 0 else "🔴"
                risk_per_share = pos['buy_price'] - pos['stop_loss']
                total_risk = risk_per_share * pos['shares']
                
                print(f"{status} {pos['ticker']}: {pos['shares']} shares @ ${pos['buy_price']:.2f}")
                print(f"   Current: ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                print(f"   Stop: ${pos['stop_loss']:.2f} | Risk: ${total_risk:.2f}")
                print()
            except Exception as e:
                print(f"❌ {pos['ticker']}: Error getting current data - {e}")
        
        print("="*60)

    # Continue with remaining methods...
    def _calculate_portfolio_value(self, portfolio: pd.DataFrame) -> float:
        """Calculate total current portfolio value"""
        if portfolio.empty:
            return 0.0
        
        total_value = 0.0
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                total_value += current_price * pos['shares']
            except Exception as e:
                logging.warning(f"Could not get price for {pos['ticker']}: {e}")
                # Use buy price as fallback
                total_value += pos['buy_price'] * pos['shares']
        
        return total_value
    
    def _process_position(self, position: pd.Series, cash: float) -> Dict[str, Any]:
        """Process individual position for stop losses and current value"""
        ticker = position['ticker']
        shares = position['shares']
        buy_price = position['buy_price']
        cost_basis = position['cost_basis']
        stop_loss = position['stop_loss']
        
        try:
            data = self.market_data.get_stock_data(ticker, "1d")
            low_price = float(data["Low"].iloc[-1])
            close_price = float(data["Close"].iloc[-1])
            
            # Check for stop loss trigger
            if low_price <= stop_loss:
                price = stop_loss
                value = price * shares
                pnl = (price - buy_price) * shares
                action = "SELL - Stop Loss Triggered"
                
                # Log the stop loss sale
                self._log_stop_loss_sale(ticker, shares, price, buy_price, pnl)
                logging.warning(f"Stop loss triggered for {ticker}: ${price:.2f}")
            else:
                price = close_price
                value = price * shares
                pnl = (price - buy_price) * shares
                action = "HOLD"
            
            return {
                "Date": self.today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": buy_price,
                "Cost Basis": cost_basis,
                "Stop Loss": stop_loss,
                "Current Price": price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": ""
            }
            
        except Exception as e:
            logging.error(f"Failed to process position {ticker}: {e}")
            return {
                "Date": self.today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": buy_price,
                "Cost Basis": cost_basis,
                "Stop Loss": stop_loss,
                "Current Price": "ERROR",
                "Total Value": 0,
                "PnL": 0,
                "Action": f"ERROR: {str(e)[:30]}",
                "Cash Balance": "",
                "Total Equity": ""
            }
    
    def _execute_buy(self, ticker: str, shares: float, buy_price: float, stop_loss: float, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Execute buy order and update portfolio"""
        total_cost = buy_price * shares
        
        # Log the trade
        self._log_trade({
            "Date": self.today,
            "Ticker": ticker,
            "Shares Bought": shares,
            "Buy Price": buy_price,
            "Cost Basis": total_cost,
            "PnL": 0.0,
            "Reason": "MANUAL BUY - New/Additional position"
        })
        
        # Update portfolio
        existing_position = portfolio[portfolio['ticker'] == ticker]
        
        if existing_position.empty:
            # New position
            new_position = {
                'ticker': ticker,
                'shares': shares,
                'buy_price': buy_price,
                'stop_loss': stop_loss,
                'cost_basis': total_cost
            }
            portfolio = pd.concat([portfolio, pd.DataFrame([new_position])], ignore_index=True)
        else:
            # Add to existing position
            idx = existing_position.index[0]
            current_shares = portfolio.at[idx, 'shares']
            current_cost = portfolio.at[idx, 'cost_basis']
            
            new_shares = current_shares + shares
            new_cost = current_cost + total_cost
            avg_price = new_cost / new_shares
            
            portfolio.at[idx, 'shares'] = new_shares
            portfolio.at[idx, 'cost_basis'] = new_cost
            portfolio.at[idx, 'buy_price'] = avg_price
            portfolio.at[idx, 'stop_loss'] = stop_loss
        
        cash -= total_cost
        return cash, portfolio
    
    def _execute_sell(self, ticker: str, shares_sold: float, sell_price: float, reason: str, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Execute sell order and update portfolio"""
        position_idx = portfolio[portfolio['ticker'] == ticker].index[0]
        position = portfolio.iloc[position_idx]
        
        buy_price = position['buy_price']
        cost_basis = buy_price * shares_sold
        pnl = (sell_price - buy_price) * shares_sold
        total_proceeds = sell_price * shares_sold
        
        # Log the trade
        self._log_trade({
            "Date": self.today,
            "Ticker": ticker,
            "Shares Sold": shares_sold,
            "Sell Price": sell_price,
            "Cost Basis": cost_basis,
            "PnL": pnl,
            "Reason": f"MANUAL SELL - {reason}"
        })
        
        # Update portfolio
        remaining_shares = position['shares'] - shares_sold
        
        if remaining_shares <= 0:
            # Remove position entirely
            portfolio = portfolio.drop(position_idx).reset_index(drop=True)
        else:
            # Reduce position
            portfolio.at[position_idx, 'shares'] = remaining_shares
            portfolio.at[position_idx, 'cost_basis'] = remaining_shares * buy_price
        
        cash += total_proceeds
        return cash, portfolio

    def _log_trade(self, trade_data: Dict[str, Any]):
        """Log trade to trade log CSV"""
        trade_df = pd.DataFrame([trade_data])
        
        if self.data_manager.trade_log_csv.exists():
            existing_df = pd.read_csv(self.data_manager.trade_log_csv)
            trade_df = pd.concat([existing_df, trade_df], ignore_index=True)
        
        trade_df.to_csv(self.data_manager.trade_log_csv, index=False)
        logging.info(f"Trade logged: {trade_data}")

    def _log_stop_loss_sale(self, ticker: str, shares: float, price: float, cost: float, pnl: float):
        """Log stop loss triggered sale"""
        self._log_trade({
            "Date": self.today,
            "Ticker": ticker,
            "Shares Sold": shares,
            "Sell Price": price,
            "Cost Basis": cost * shares,
            "PnL": pnl,
            "Reason": "AUTOMATED SELL - STOP LOSS TRIGGERED"
        })

    def _remove_position(self, portfolio: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove position from portfolio"""
        return portfolio[portfolio['ticker'] != ticker].reset_index(drop=True)

    def _create_summary_row(self, total_value: float, total_pnl: float, cash: float) -> Dict[str, Any]:
        """Create summary row for portfolio results"""
        return {
            "Date": self.today,
            "Ticker": "TOTAL",
            "Shares": "",
            "Buy Price": "",
            "Cost Basis": "",
            "Stop Loss": "",
            "Current Price": "",
            "Total Value": total_value,
            "PnL": total_pnl,
            "Action": "",
            "Cash Balance": cash,
            "Total Equity": total_value + cash
        }

    def _save_portfolio_results(self, results: List[Dict[str, Any]]):
        """Save portfolio results to CSV"""
        new_df = pd.DataFrame(results)
        
        if self.data_manager.portfolio_csv.exists():
            existing_df = pd.read_csv(self.data_manager.portfolio_csv)
            # Remove today's data if it exists
            existing_df = existing_df[existing_df['Date'] != self.today]
            # Append new results
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        combined_df.to_csv(self.data_manager.portfolio_csv, index=False)
        logging.info("Portfolio results saved to CSV")


# ==============================================
# ENHANCED REPORTING AND RESULTS
# ==============================================

class EnhancedReporting:
    """Enhanced reporting with comprehensive analytics and AI insights"""
    
    def __init__(self, config: TradingConfig, data_manager: DataManager, market_data: MarketDataProvider, ai_analyst: AIAnalyst):
        self.config = config
        self.data_manager = data_manager
        self.market_data = market_data
        self.ai_analyst = ai_analyst
        self.analytics = PortfolioAnalytics(config)
        self.today = datetime.today().strftime("%Y-%m-%d")
    
    def generate_daily_report(self, portfolio: pd.DataFrame, cash: float):
        """Generate comprehensive daily report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DAILY TRADING REPORT")
        print(f"📅 {self.today}")
        print("="*80)
        
        # Market overview
        self._display_market_overview()
        
        # Portfolio summary
        self._display_portfolio_summary(portfolio, cash)
        
        # Performance metrics
        self._display_performance_metrics()
        
        # Risk analysis and alerts
        self._display_risk_analysis(portfolio)
        
        # AI insights
        if self.config.get('ai.enabled'):
            self._display_ai_insights(portfolio, cash)
        
        # Trading recommendations
        self._display_trading_recommendations()
        
        print("="*80)
    
    def _display_market_overview(self):
        """Display current market conditions"""
        print("\n📈 MARKET OVERVIEW")
        print("-" * 40)
        
        try:
            market_context = self.market_data.get_market_context()
            
            if market_context:
                for name, data in market_context.items():
                    change_color = "🟢" if data['daily_change'] >= 0 else "🔴"
                    print(f"{change_color} {name}: ${data['price']:.2f} ({data['daily_change']:+.2f}%)")
            else:
                print("Market data unavailable")
                
        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            logging.error(f"Market overview error: {e}")
    
    def _display_portfolio_summary(self, portfolio: pd.DataFrame, cash: float):
        """Display detailed portfolio summary"""
        print(f"\n💼 PORTFOLIO SUMMARY")
        print("-" * 40)
        
        if portfolio.empty:
            print("No positions currently held")
            print(f"💰 Cash: ${cash:,.2f}")
            return
        
        total_market_value = 0.0
        total_pnl = 0.0
        
        print(f"{'Ticker':<8} {'Shares':<8} {'Buy Price':<10} {'Current':<10} {'P&L':<12} {'P&L %':<8} {'Risk':<10}")
        print("-" * 75)
        
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                market_value = current_price * pos['shares']
                pnl = (current_price - pos['buy_price']) * pos['shares']
                pnl_pct = (pnl / (pos['buy_price'] * pos['shares'])) * 100
                risk_per_position = (pos['buy_price'] - pos['stop_loss']) * pos['shares']
                
                total_market_value += market_value
                total_pnl += pnl
                
                status = "🟢" if pnl >= 0 else "🔴"
                
                print(f"{status} {pos['ticker']:<6} {pos['shares']:<8.0f} ${pos['buy_price']:<9.2f} ${current_price:<9.2f} "
                      f"${pnl:<11.2f} {pnl_pct:<7.1f}% ${risk_per_position:<9.2f}")
                      
            except Exception as e:
                print(f"❌ {pos['ticker']:<6} - Error: {str(e)[:30]}")
                logging.warning(f"Could not display position {pos['ticker']}: {e}")
        
        print("-" * 75)
        total_equity = total_market_value + cash
        
        print(f"💰 Cash Balance: ${cash:,.2f}")
        print(f"📈 Market Value: ${total_market_value:,.2f}")
        print(f"📊 Total P&L: ${total_pnl:,.2f}")
        print(f"💎 Total Equity: ${total_equity:,.2f}")
    
    def _display_performance_metrics(self):
        """Display comprehensive performance metrics"""
        print(f"\n📊 PERFORMANCE METRICS")
        print("-" * 40)
        
        try:
            if not self.data_manager.portfolio_csv.exists():
                print("No historical data available")
                return
            
            # Load historical data
            df = pd.read_csv(self.data_manager.portfolio_csv)
            totals = df[df['Ticker'] == 'TOTAL'].copy()
            
            if len(totals) < 2:
                print("Insufficient data for performance calculation")
                return
            
            totals['Date'] = pd.to_datetime(totals['Date'])
            totals = totals.sort_values('Date')
            
            # Calculate returns
            equity = totals['Total Equity'].astype(float)
            returns = equity.pct_change().dropna()
            
            if len(returns) == 0:
                print("No returns data available")
                return
            
            # Get comprehensive metrics
            metrics = self.analytics.calculate_comprehensive_metrics(returns)
            
            # Display key metrics
            print(f"📈 Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"📊 Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"⚡ Volatility: {metrics.get('annualized_volatility', 0):.2%}")
            print(f"🎯 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"🎯 Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
            print(f"🎯 Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
            print(f"🎲 Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"💪 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            # Compare with S&P 500
            self._display_benchmark_comparison(totals)
            
        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            logging.error(f"Performance metrics error: {e}")
    
    def _display_benchmark_comparison(self, portfolio_data: pd.DataFrame):
        """Compare portfolio performance with S&P 500"""
        try:
            start_date = portfolio_data['Date'].min()
            end_date = portfolio_data['Date'].max()
            
            # Get S&P 500 data
            spx_data = self.market_data.get_stock_data('^SPX', f"{(end_date - start_date).days + 5}d")
            
            if not spx_data.empty and len(spx_data) >= 2:
                spx_start = float(spx_data['Close'].iloc[0])
                spx_end = float(spx_data['Close'].iloc[-1])
                spx_return = (spx_end - spx_start) / spx_start
                
                portfolio_start = float(portfolio_data['Total Equity'].iloc[0])
                portfolio_end = float(portfolio_data['Total Equity'].iloc[-1])
                portfolio_return = (portfolio_end - portfolio_start) / portfolio_start
                
                outperformance = portfolio_return - spx_return
                
                print(f"\n📊 BENCHMARK COMPARISON")
                print(f"Portfolio Return: {portfolio_return:.2%}")
                print(f"S&P 500 Return: {spx_return:.2%}")
                
                if outperformance > 0:
                    print(f"🟢 Outperformance: +{outperformance:.2%}")
                else:
                    print(f"🔴 Underperformance: {outperformance:.2%}")
                    
        except Exception as e:
            logging.warning(f"Benchmark comparison error: {e}")
    
    def _display_risk_analysis(self, portfolio: pd.DataFrame):
        """Display risk analysis and alerts"""
        print(f"\n⚠️ RISK ANALYSIS")
        print("-" * 40)
        
        try:
            alerts = self.analytics.check_portfolio_alerts(portfolio, self.market_data)
            
            if alerts:
                for alert in alerts:
                    print(alert)
            else:
                print("✅ No current risk alerts")
                
            # Additional risk metrics
            if not portfolio.empty:
                portfolio_value = 0
                max_risk = 0
                
                for _, pos in portfolio.iterrows():
                    try:
                        current_price = self.market_data.get_current_price(pos['ticker'])
                        position_value = current_price * pos['shares']
                        position_risk = (pos['buy_price'] - pos['stop_loss']) * pos['shares']
                        
                        portfolio_value += position_value
                        max_risk += position_risk
                        
                    except Exception:
                        continue
                
                if portfolio_value > 0:
                    risk_pct = (max_risk / portfolio_value) * 100
                    print(f"📉 Maximum Portfolio Risk: ${max_risk:.2f} ({risk_pct:.1f}%)")
                    
        except Exception as e:
            print(f"❌ Error in risk analysis: {e}")
            logging.error(f"Risk analysis error: {e}")
    
    def _display_ai_insights(self, portfolio: pd.DataFrame, cash: float):
        """Display AI-powered insights and analysis"""
        print(f"\n🤖 AI INSIGHTS")
        print("-" * 40)
        
        try:
            # Prepare data for AI analysis
            portfolio_summary = self._prepare_portfolio_summary(portfolio, cash)
            market_context = self.market_data.get_market_context()
            
            # Get performance metrics for AI context
            metrics = {}
            if self.data_manager.portfolio_csv.exists():
                df = pd.read_csv(self.data_manager.portfolio_csv)
                totals = df[df['Ticker'] == 'TOTAL']
                if len(totals) >= 2:
                    equity = totals['Total Equity'].astype(float)
                    returns = equity.pct_change().dropna()
                    if len(returns) > 0:
                        metrics = self.analytics.calculate_comprehensive_metrics(returns)
            
            # Get AI portfolio analysis
            print("📊 Portfolio Analysis:")
            ai_analysis = self.ai_analyst.get_portfolio_analysis(
                portfolio_summary, market_context, metrics
            )
            print(f"   {ai_analysis}")
            
            # Get stock screening suggestions
            print("\n💡 Stock Screening:")
            current_tickers = portfolio['ticker'].tolist() if not portfolio.empty else []
            screening_suggestions = self.ai_analyst.get_stock_screening_suggestions(current_tickers)
            print(f"   {screening_suggestions}")
            
            # Get risk assessment if there are alerts
            alerts = self.analytics.check_portfolio_alerts(portfolio, self.market_data)
            if alerts:
                print("\n⚠️ Risk Assessment:")
                risk_assessment = self.ai_analyst.get_risk_assessment(
                    portfolio_summary, alerts
                )
                print(f"   {risk_assessment}")
                
        except Exception as e:
            print(f"❌ AI insights unavailable: {e}")
            logging.error(f"AI insights error: {e}")
    
    def _prepare_portfolio_summary(self, portfolio: pd.DataFrame, cash: float) -> str:
        """Prepare portfolio summary for AI analysis"""
        if portfolio.empty:
            return f"Empty portfolio with ${cash:.2f} cash"
        
        summary_parts = [f"Cash: ${cash:.2f}"]
        
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                pnl = (current_price - pos['buy_price']) * pos['shares']
                pnl_pct = (pnl / (pos['buy_price'] * pos['shares'])) * 100
                
                summary_parts.append(
                    f"{pos['ticker']}: {pos['shares']} shares, "
                    f"P&L: ${pnl:.2f} ({pnl_pct:+.1f}%), "
                    f"Stop: ${pos['stop_loss']:.2f}"
                )
            except Exception:
                summary_parts.append(f"{pos['ticker']}: {pos['shares']} shares (price unavailable)")
        
        return "\n".join(summary_parts)
    
    def _display_trading_recommendations(self):
        """Display final trading recommendations and instructions"""
        print(f"\n🎯 TRADING RECOMMENDATIONS")
        print("-" * 40)
        print("📋 Today's Action Items:")
        print("• Review AI analysis and risk alerts above")
        print("• Consider position sizing recommendations")
        print("• Monitor positions near stop losses")
        print("• Evaluate new opportunities from screening")
        print("• Update stop losses based on market conditions")
        print()
        print("💭 Decision Framework:")
        print("• You have complete control over all decisions")
        print("• No approval required for any trades")
        print("• Act based on analysis and your judgment")
        print("• Research current prices before making changes")
        print()
        print("⏰ Next Steps:")
        print("• If no immediate action needed, portfolio remains unchanged")
        print("• Use provided analysis for informed decision making")
        print("• Consider copying analysis to external tools if needed")


# ==============================================
# MAIN APPLICATION AND CLI
# ==============================================

class TradingApplication:
    """Main application orchestrating all components"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config = TradingConfig(config_file)
        
        # Set up logging
        setup_logging(self.config)
        
        # Initialize components
        self.data_manager = DataManager(self.config)
        self.market_data = MarketDataProvider(self.config)
        self.ai_analyst = AIAnalyst(self.config)
        self.trading_engine = TradingEngine(self.config, self.data_manager, self.market_data)
        self.reporter = EnhancedReporting(self.config, self.data_manager, self.market_data, self.ai_analyst)
        
        logging.info("Trading application initialized")
    
    def run(self, file: str, data_dir: Path = None, interactive: bool = True, backup: bool = True):
        """Run the main trading application"""
        try:
            logging.info(f"Starting trading session with file: {file}")
            
            # Set data directory if provided
            if data_dir:
                self.data_manager.set_data_dir(data_dir)
            
            # Create backup if requested
            if backup:
                self.data_manager.backup_data()
            
            # Load portfolio state
            portfolio, cash = self.load_latest_portfolio_state(file)
            logging.info(f"Loaded portfolio: {len(portfolio) if isinstance(portfolio, pd.DataFrame) else 'empty'} positions, ${cash:.2f} cash")
            
            # Process portfolio
            portfolio, cash = self.trading# filepath: c:\Users\crist\OneDrive\Documents\GitHub\ChatGPT-Micro-Cap-Experiment\enhanced_trading_script.py
"""Enhanced Trading Script with AI Analysis and Comprehensive Features

The script processes portfolio positions, logs trades, provides AI analysis,
and offers advanced analytics with robust error handling and monitoring.
"""

from datetime import datetime
from pathlib import Path
import argparse
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==============================================
# CONFIGURATION MANAGEMENT
# ==============================================

class TradingConfig:
    """Centralized configuration management"""
    
    DEFAULT_CONFIG = {
        'ai': {
            'enabled': True,
            'primary_model': 'llama3.2:3b',
            'fallback_model': 'phi3:mini',
            'timeout': 30,
            'ollama_url': 'http://localhost:11434/api/generate'
        },
        'trading': {
            'max_position_size': 0.15,  # 15% of portfolio
            'default_stop_loss_pct': 0.15,  # 15% stop loss
            'cash_reserve_pct': 0.05,  # Keep 5% cash
            'risk_free_rate': 0.045  # 4.5% annual
        },
        'data': {
            'retry_attempts': 3,
            'retry_delay': 2,
            'backup_enabled': True,
            'max_backups': 10
        },
        'notifications': {
            'log_level': 'INFO',
            'alert_near_stop_loss': 0.05  # Alert when within 5% of stop loss
        },
        'market': {
            'trading_hours_only': False,
            'weekend_warning': True
        }
    }
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                merged = self.DEFAULT_CONFIG.copy()
                self._deep_update(merged, config)
                return merged
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Error loading config: {e}. Using defaults.")
        
        self.save_config(self.DEFAULT_CONFIG)
        return self.DEFAULT_CONFIG.copy()
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save current configuration to file"""
        config_to_save = config or self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except IOError as e:
            logging.error(f"Failed to save config: {e}")
    
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


# ==============================================
# LOGGING AND ERROR HANDLING
# ==============================================

def setup_logging(config: TradingConfig, log_dir: Path = None):
    """Set up comprehensive logging"""
    log_dir = log_dir or Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_level = getattr(logging, config.get('notifications.log_level', 'INFO').upper())
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_dir / 'trading.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / 'errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_handler)


def retry_on_failure(max_retries: int = 3, delay: float = 1, backoff: float = 2):
    """Decorator to retry failed operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


# ==============================================
# DATA MANAGEMENT AND VALIDATION
# ==============================================

class DataManager:
    """Handle all data operations with validation and backup"""
    
    def __init__(self, config: TradingConfig, script_dir: Path = None):
        self.config = config
        self.script_dir = script_dir or Path(__file__).resolve().parent
        self.data_dir = self.script_dir
        self.portfolio_csv = self.data_dir / "chatgpt_portfolio_update.csv"
        self.trade_log_csv = self.data_dir / "chatgpt_trade_log.csv"
        self.backup_dir = self.data_dir / "backups"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        if config.get('data.backup_enabled'):
            self.backup_dir.mkdir(exist_ok=True)
    
    def set_data_dir(self, data_dir: Path):
        """Update data directory paths"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.portfolio_csv = self.data_dir / "chatgpt_portfolio_update.csv"
        self.trade_log_csv = self.data_dir / "chatgpt_trade_log.csv"
        self.backup_dir = self.data_dir / "backups"
        if self.config.get('data.backup_enabled'):
            self.backup_dir.mkdir(exist_ok=True)
    
    def validate_portfolio(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean portfolio data"""
        if portfolio.empty:
            return portfolio
        
        required_columns = ['ticker', 'shares', 'buy_price', 'stop_loss', 'cost_basis']
        
        # Check required columns
        missing_cols = set(required_columns) - set(portfolio.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Data type validation
        numeric_cols = ['shares', 'buy_price', 'stop_loss', 'cost_basis']
        for col in numeric_cols:
            portfolio[col] = pd.to_numeric(portfolio[col], errors='coerce')
        
        # Remove invalid rows
        invalid_mask = portfolio[numeric_cols].isnull().any(axis=1)
        if invalid_mask.any():
            logging.warning(f"Removing {invalid_mask.sum()} invalid portfolio rows")
            portfolio = portfolio[~invalid_mask]
        
        # Validate business logic
        portfolio['ticker'] = portfolio['ticker'].str.upper()
        
        # Check for negative values
        negative_mask = (portfolio[numeric_cols] <= 0).any(axis=1)
        if negative_mask.any():
            logging.warning(f"Found {negative_mask.sum()} rows with negative/zero values")
            portfolio = portfolio[~negative_mask]
        
        return portfolio.reset_index(drop=True)
    
    def backup_data(self):
        """Create timestamped backups of important data"""
        if not self.config.get('data.backup_enabled'):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_to_backup = [self.portfolio_csv, self.trade_log_csv]
        
        for file_path in files_to_backup:
            if file_path.exists():
                backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                backup_path = self.backup_dir / backup_name
                backup_path.write_text(file_path.read_text())
                logging.info(f"Backed up {file_path.name} to {backup_name}")
        
        # Clean old backups
        self._cleanup_old_backups()
    
    def _cleanup_old_backups(self):
        """Remove old backup files beyond the configured limit"""
        max_backups = self.config.get('data.max_backups', 10)
        
        for pattern in ['chatgpt_portfolio_update_*.csv', 'chatgpt_trade_log_*.csv']:
            backup_files = sorted(self.backup_dir.glob(pattern), reverse=True)
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                logging.debug(f"Removed old backup: {old_backup.name}")
    
    def auto_backup(self, func):
        """Decorator to automatically backup data before risky operations"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.backup_data()
            return func(*args, **kwargs)
        return wrapper


# ==============================================
# MARKET DATA AND ANALYSIS
# ==============================================

class MarketDataProvider:
    """Enhanced market data provider with error handling and caching"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
    
    @retry_on_failure(max_retries=3, delay=1)
    def get_stock_data(self, ticker: str, period: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        """Get stock data with retry logic and caching"""
        cache_key = f"{ticker}_{period}"
        current_time = time.time()
        
        # Check cache
        if use_cache and cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if current_time - timestamp < self._cache_timeout:
                return data
        
        # Fetch new data
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, auto_adjust=False)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Cache the result
            if use_cache:
                self._cache[cache_key] = (data, current_time)
            
            logging.debug(f"Fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logging.error(f"Failed to fetch data for {ticker}: {e}")
            raise
    
    def get_current_price(self, ticker: str) -> float:
        """Get current/last closing price for a ticker"""
        data = self.get_stock_data(ticker, "1d")
        return float(data["Close"].iloc[-1])
    
    def get_market_context(self) -> Dict[str, Dict[str, float]]:
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
                data = self.get_stock_data(ticker, "5d")
                if len(data) >= 2:
                    current = float(data['Close'].iloc[-1])
                    prev = float(data['Close'].iloc[-2])
                    change = ((current - prev) / prev * 100) if prev != 0 else 0
                    context[name] = {'price': current, 'daily_change': change}
            except Exception as e:
                logging.warning(f"Failed to get market context for {ticker}: {e}")
        
        return context
    
    def validate_trade_price(self, ticker: str, price: float) -> bool:
        """Validate if a trade price is within today's range"""
        try:
            data = self.get_stock_data(ticker, "1d")
            day_high = float(data["High"].iloc[-1])
            day_low = float(data["Low"].iloc[-1])
            return day_low <= price <= day_high
        except Exception as e:
            logging.warning(f"Could not validate price for {ticker}: {e}")
            return True  # Allow trade if we can't validate


# ==============================================
# ADVANCED ANALYTICS
# ==============================================

class PortfolioAnalytics:
    """Advanced portfolio analytics and performance metrics"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.risk_free_rate = config.get('trading.risk_free_rate', 0.045)
    
    def calculate_comprehensive_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        if len(returns) < 2:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Performance ratios
        rf_daily = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = returns - rf_daily
        
        sharpe_ratio = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = returns[returns < rf_daily]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std != 0 else 0
        
        # Calmar ratio
        calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else np.inf
        
        # Win/loss statistics
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_days > 0 else 0
        profit_factor = abs(avg_win * winning_days / (avg_loss * losing_days)) if avg_loss != 0 and losing_days > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'winning_days': int(winning_days),
            'losing_days': int(losing_days),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def check_portfolio_alerts(self, portfolio: pd.DataFrame, market_data_provider: MarketDataProvider) -> List[str]:
        """Check for various portfolio alerts"""
        alerts = []
        
        if portfolio.empty:
            return alerts
        
        max_position = self.config.get('trading.max_position_size', 0.15)
        stop_loss_threshold = self.config.get('notifications.alert_near_stop_loss', 0.05)
        
        # Calculate total portfolio value
        total_value = 0
        for _, position in portfolio.iterrows():
            try:
                current_price = market_data_provider.get_current_price(position['ticker'])
                position_value = position['shares'] * current_price
                total_value += position_value
            except Exception as e:
                logging.warning(f"Could not get price for {position['ticker']}: {e}")
        
        # Position size alerts
        for _, position in portfolio.iterrows():
            try:
                current_price = market_data_provider.get_current_price(position['ticker'])
                position_value = position['shares'] * current_price
                position_pct = position_value / total_value if total_value > 0 else 0
                
                if position_pct > max_position:
                    alerts.append(f"⚠️ {position['ticker']} is {position_pct:.1%} of portfolio (max: {max_position:.1%})")
                
                # Stop loss proximity alerts
                stop_distance = (current_price - position['stop_loss']) / current_price
                if stop_distance < stop_loss_threshold:
                    alerts.append(f"🚨 {position['ticker']} is near stop loss: ${current_price:.2f} vs ${position['stop_loss']:.2f}")
                    
            except Exception as e:
                logging.warning(f"Could not analyze position {position['ticker']}: {e}")
        
        return alerts


# ==============================================
# AI ANALYSIS INTEGRATION
# ==============================================

class AIAnalyst:
    """Enhanced AI analysis with multiple models and robust error handling"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = config.get('ai.enabled', True)
        self.primary_model = config.get('ai.primary_model', 'llama3.2:3b')
        self.fallback_model = config.get('ai.fallback_model', 'phi3:mini')
        self.timeout = config.get('ai.timeout', 30)
        self.ollama_url = config.get('ai.ollama_url', 'http://localhost:11434/api/generate')
    
    def _make_ollama_request(self, prompt: str, model: str, max_tokens: int = 200) -> str:
        """Make request to Ollama API with error handling"""
        if not self.enabled:
            return "AI analysis disabled in configuration"
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': max_tokens
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                logging.error(f"Ollama API error: {response.status_code}")
                return f"AI analysis unavailable (HTTP {response.status_code})"
                
        except requests.exceptions.Timeout:
            return f"AI analysis timeout after {self.timeout}s - Ollama may be busy"
        except requests.exceptions.ConnectionError:
            return "AI analysis unavailable - check Ollama connection"
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            return f"AI analysis error: {str(e)[:50]}..."
    
    def get_portfolio_analysis(self, portfolio_summary: str, market_context: Dict, performance_metrics: Dict) -> str:
        """Get comprehensive portfolio analysis from AI"""
        market_summary = "\n".join([
            f"{name}: ${data['price']:.2f} ({data['daily_change']:+.2f}%)"
            for name, data in market_context.items()
        ])
        
        metrics_summary = f"""
Performance Metrics:
- Win Rate: {performance_metrics.get('win_rate', 0):.1%}
- Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
- Volatility: {performance_metrics.get('annualized_volatility', 0):.1%}
- VaR (95%): {performance_metrics.get('var_95', 0):.2%}"""

        prompt = f"""
PORTFOLIO ANALYSIS REQUEST

Portfolio Summary:
{portfolio_summary}

Market Context:
{market_summary}

{metrics_summary}

As a quantitative analyst, provide a comprehensive analysis:

1. RISK ASSESSMENT (High/Medium/Low): Key risk factors and concerns
2. PERFORMANCE EVALUATION: How the portfolio is performing relative to market and metrics
3. DIVERSIFICATION ANALYSIS: Sector concentration and balance issues
4. MARKET TIMING: Current market conditions and their impact
5. ACTIONABLE RECOMMENDATION: One specific trade or adjustment for tomorrow

Format: Start with risk level, then provide analysis in 4-5 concise sentences. Focus on actionable insights based on the data provided.
"""
        
        # Try primary model first, fallback to secondary if needed
        result = self._make_ollama_request(prompt, self.primary_model, 250)
        
        if "unavailable" in result or "error" in result:
            logging.info(f"Primary AI model failed, trying fallback: {self.fallback_model}")
            result = self._make_ollama_request(prompt, self.fallback_model, 200)
        
        return result
    
    def get_stock_screening_suggestions(self, current_tickers: List[str]) -> str:
        """Get AI-powered stock screening suggestions"""
        prompt = f"""
STOCK SCREENING REQUEST

Current Portfolio Holdings: {', '.join(current_tickers)}
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

Based on current market conditions and momentum trading principles, provide:

1. SECTOR ANALYSIS: Are we overconcentrated in any sectors?
2. MOMENTUM OPPORTUNITIES: 2-3 specific stocks showing strong momentum (provide tickers)
3. POSITION SIZING: Recommended allocation percentages
4. RISK MANAGEMENT: Any positions that should be reduced or hedged?

Keep response under 120 words. Focus on specific, actionable recommendations with ticker symbols where applicable.
"""
        
        return self._make_ollama_request(prompt, self.fallback_model, 150)
    
    def get_risk_assessment(self, portfolio_data: str, alerts: List[str]) -> str:
        """Get AI risk assessment based on current portfolio and alerts"""
        alerts_text = "\n".join(alerts) if alerts else "No current alerts"
        
        prompt = f"""
RISK ASSESSMENT REQUEST

Portfolio Data:
{portfolio_data}

Current Alerts:
{alerts_text}

Provide a focused risk assessment:
1. Immediate risks that need attention
2. Position sizing recommendations
3. Stop-loss adjustments if needed
4. Overall risk level (Low/Medium/High/Critical)

Maximum 80 words. Be specific and actionable.
"""
        
        return self._make_ollama_request(prompt, self.fallback_model, 100)


# ==============================================
# ENHANCED TRADING OPERATIONS
# ==============================================

class TradingEngine:
    """Enhanced trading engine with validation and logging"""
    
    def __init__(self, config: TradingConfig, data_manager: DataManager, market_data: MarketDataProvider):
        self.config = config
        self.data_manager = data_manager
        self.market_data = market_data
        self.today = datetime.today().strftime("%Y-%m-%d")
        self.analytics = PortfolioAnalytics(config)
    
    def process_portfolio(
        self,
        portfolio: Union[pd.DataFrame, Dict, List],
        cash: float,
        interactive: bool = True
    ) -> Tuple[pd.DataFrame, float]:
        """Enhanced portfolio processing with comprehensive validation and logging"""
        
        # Normalize input to DataFrame
        if isinstance(portfolio, pd.DataFrame):
            portfolio_df = portfolio.copy()
        elif isinstance(portfolio, (dict, list)):
            portfolio_df = pd.DataFrame(portfolio)
        else:
            raise TypeError("portfolio must be a DataFrame, dict, or list of dicts")
        
        # Validate portfolio data
        portfolio_df = self.data_manager.validate_portfolio(portfolio_df)
        
        logging.info(f"Processing portfolio with {len(portfolio_df)} positions and ${cash:.2f} cash")
        
        # Weekend check
        if self._is_weekend() and interactive and self.config.get('market.weekend_warning'):
            if not self._confirm_weekend_processing():
                raise SystemExit("Exiting due to weekend processing cancellation")
        
        # Interactive trading
        if interactive:
            cash, portfolio_df = self._handle_manual_trades(cash, portfolio_df)
        
        # Process existing positions
        results = []
        total_value = 0.0
        total_pnl = 0.0
        
        for _, position in portfolio_df.iterrows():
            result_row = self._process_position(position, cash)
            
            # Handle stop loss triggers
            if result_row['Action'] == 'SELL - Stop Loss Triggered':
                cash += result_row['Total Value']
                portfolio_df = self._remove_position(portfolio_df, position['ticker'])
            else:
                total_value += result_row.get('Total Value', 0)
                total_pnl += result_row.get('PnL', 0)
            
            results.append(result_row)
        
        # Add summary row
        total_row = self._create_summary_row(total_value, total_pnl, cash)
        results.append(total_row)
        
        # Save results
        self._save_portfolio_results(results)
        
        logging.info(f"Portfolio processing complete. Total value: ${total_value:.2f}, Cash: ${cash:.2f}")
        return portfolio_df, cash
    
    def _is_weekend(self) -> bool:
        """Check if today is weekend"""
        return datetime.now().weekday() in [5, 6]  # Saturday, Sunday
    
    def _confirm_weekend_processing(self) -> bool:
        """Confirm weekend processing with user"""
        response = input(
            "Today is a weekend - markets are closed. This will use the last trading day's data. "
            "Continue? (y/N): "
        ).strip().lower()
        return response in ['y', 'yes']
    
    def _handle_manual_trades(self, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Handle interactive manual trading"""
        while True:
            action = input(
                f"\n💰 Cash available: ${cash:,.2f}\n"
                "Manual trade options:\n"
                "  'b' - Buy position\n"
                "  's' - Sell position\n"
                "  'v' - View portfolio\n"
                "  Enter - Continue\n"
                "Choice: "
            ).strip().lower()
            
            if action == 'b':
                cash, portfolio = self._handle_manual_buy(cash, portfolio)
            elif action == 's':
                cash, portfolio = self._handle_manual_sell(cash, portfolio)
            elif action == 'v':
                self._display_portfolio_summary(portfolio, cash)
            elif action == '':
                break
            else:
                print("Invalid option. Please try again.")
        
        return cash, portfolio
    
    def _handle_manual_buy(self, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Handle manual buy orders with comprehensive validation"""
        try:
            ticker = input("Enter ticker symbol: ").strip().upper()
            shares = float(input("Enter number of shares: "))
            buy_price = float(input("Enter buy price: "))
            stop_loss = float(input("Enter stop loss price: "))
            
            # Validation
            if shares <= 0 or buy_price <= 0 or stop_loss <= 0:
                raise ValueError("All values must be positive")
            
            if stop_loss >= buy_price:
                print("Warning: Stop loss is above buy price")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            # Cost check
            total_cost = buy_price * shares
            if total_cost > cash:
                print(f"❌ Insufficient cash: ${total_cost:.2f} required, ${cash:.2f} available")
                return cash, portfolio
            
            # Price validation
            if not self.market_data.validate_trade_price(ticker, buy_price):
                print(f"⚠️ Price ${buy_price:.2f} may be outside today's trading range")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            # Position size check
            max_position = self.config.get('trading.max_position_size', 0.15)
            portfolio_value = cash + self._calculate_portfolio_value(portfolio)
            position_pct = total_cost / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > max_position:
                print(f"⚠️ This position would be {position_pct:.1%} of portfolio (max recommended: {max_position:.1%})")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            # Execute buy
            cash, portfolio = self._execute_buy(ticker, shares, buy_price, stop_loss, cash, portfolio)
            print(f"✅ Buy order executed: {shares} shares of {ticker} at ${buy_price:.2f}")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except Exception as e:
            print(f"❌ Buy order failed: {e}")
            logging.error(f"Manual buy failed: {e}")
        
        return cash, portfolio
    
    def _handle_manual_sell(self, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Handle manual sell orders with validation"""
        if portfolio.empty:
            print("❌ No positions to sell")
            return cash, portfolio
        
        print("\nCurrent positions:")
        for i, (_, pos) in enumerate(portfolio.iterrows()):
            print(f"  {i+1}. {pos['ticker']}: {pos['shares']} shares @ ${pos['buy_price']:.2f}")
        
        try:
            ticker = input("Enter ticker to sell: ").strip().upper()
            
            if ticker not in portfolio['ticker'].values:
                print(f"❌ {ticker} not found in portfolio")
                return cash, portfolio
            
            position = portfolio[portfolio['ticker'] == ticker].iloc[0]
            max_shares = position['shares']
            
            shares = float(input(f"Enter shares to sell (max {max_shares}): "))
            sell_price = float(input("Enter sell price: "))
            
            if shares <= 0 or sell_price <= 0:
                raise ValueError("Values must be positive")
            
            if shares > max_shares:
                raise ValueError(f"Cannot sell {shares} shares, only {max_shares} available")
            
            # Price validation
            if not self.market_data.validate_trade_price(ticker, sell_price):
                print(f"⚠️ Price ${sell_price:.2f} may be outside today's trading range")
                if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
                    return cash, portfolio
            
            reason = input("Reason for sale (optional): ").strip() or "Manual sell"
            
            # Execute sell
            cash, portfolio = self._execute_sell(ticker, shares, sell_price, reason, cash, portfolio)
            print(f"✅ Sell order executed: {shares} shares of {ticker} at ${sell_price:.2f}")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except Exception as e:
            print(f"❌ Sell order failed: {e}")
            logging.error(f"Manual sell failed: {e}")
        
        return cash, portfolio
    
    def _display_portfolio_summary(self, portfolio: pd.DataFrame, cash: float):
        """Display formatted portfolio summary"""
        print("\n" + "="*60)
        print("📊 CURRENT PORTFOLIO SUMMARY")
        print("="*60)
        
        if portfolio.empty:
            print("No positions currently held.")
            print(f"💰 Cash: ${cash:,.2f}")
            return
        
        total_value = self._calculate_portfolio_value(portfolio)
        total_equity = total_value + cash
        
        print(f"💰 Cash Balance: ${cash:,.2f}")
        print(f"📈 Position Value: ${total_value:,.2f}")
        print(f"💎 Total Equity: ${total_equity:,.2f}")
        print()
        
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                pnl = (current_price - pos['buy_price']) * pos['shares']
                pnl_pct = (pnl / (pos['buy_price'] * pos['shares'])) * 100
                
                status = "🟢" if pnl >= 0 else "🔴"
                risk_per_share = pos['buy_price'] - pos['stop_loss']
                total_risk = risk_per_share * pos['shares']
                
                print(f"{status} {pos['ticker']}: {pos['shares']} shares @ ${pos['buy_price']:.2f}")
                print(f"   Current: ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                print(f"   Stop: ${pos['stop_loss']:.2f} | Risk: ${total_risk:.2f}")
                print()
            except Exception as e:
                print(f"❌ {pos['ticker']}: Error getting current data - {e}")
        
        print("="*60)

    # Continue with remaining methods...
    def _calculate_portfolio_value(self, portfolio: pd.DataFrame) -> float:
        """Calculate total current portfolio value"""
        if portfolio.empty:
            return 0.0
        
        total_value = 0.0
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                total_value += current_price * pos['shares']
            except Exception as e:
                logging.warning(f"Could not get price for {pos['ticker']}: {e}")
                # Use buy price as fallback
                total_value += pos['buy_price'] * pos['shares']
        
        return total_value
    
    def _process_position(self, position: pd.Series, cash: float) -> Dict[str, Any]:
        """Process individual position for stop losses and current value"""
        ticker = position['ticker']
        shares = position['shares']
        buy_price = position['buy_price']
        cost_basis = position['cost_basis']
        stop_loss = position['stop_loss']
        
        try:
            data = self.market_data.get_stock_data(ticker, "1d")
            low_price = float(data["Low"].iloc[-1])
            close_price = float(data["Close"].iloc[-1])
            
            # Check for stop loss trigger
            if low_price <= stop_loss:
                price = stop_loss
                value = price * shares
                pnl = (price - buy_price) * shares
                action = "SELL - Stop Loss Triggered"
                
                # Log the stop loss sale
                self._log_stop_loss_sale(ticker, shares, price, buy_price, pnl)
                logging.warning(f"Stop loss triggered for {ticker}: ${price:.2f}")
            else:
                price = close_price
                value = price * shares
                pnl = (price - buy_price) * shares
                action = "HOLD"
            
            return {
                "Date": self.today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": buy_price,
                "Cost Basis": cost_basis,
                "Stop Loss": stop_loss,
                "Current Price": price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": ""
            }
            
        except Exception as e:
            logging.error(f"Failed to process position {ticker}: {e}")
            return {
                "Date": self.today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": buy_price,
                "Cost Basis": cost_basis,
                "Stop Loss": stop_loss,
                "Current Price": "ERROR",
                "Total Value": 0,
                "PnL": 0,
                "Action": f"ERROR: {str(e)[:30]}",
                "Cash Balance": "",
                "Total Equity": ""
            }
    
    def _execute_buy(self, ticker: str, shares: float, buy_price: float, stop_loss: float, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Execute buy order and update portfolio"""
        total_cost = buy_price * shares
        
        # Log the trade
        self._log_trade({
            "Date": self.today,
            "Ticker": ticker,
            "Shares Bought": shares,
            "Buy Price": buy_price,
            "Cost Basis": total_cost,
            "PnL": 0.0,
            "Reason": "MANUAL BUY - New/Additional position"
        })
        
        # Update portfolio
        existing_position = portfolio[portfolio['ticker'] == ticker]
        
        if existing_position.empty:
            # New position
            new_position = {
                'ticker': ticker,
                'shares': shares,
                'buy_price': buy_price,
                'stop_loss': stop_loss,
                'cost_basis': total_cost
            }
            portfolio = pd.concat([portfolio, pd.DataFrame([new_position])], ignore_index=True)
        else:
            # Add to existing position
            idx = existing_position.index[0]
            current_shares = portfolio.at[idx, 'shares']
            current_cost = portfolio.at[idx, 'cost_basis']
            
            new_shares = current_shares + shares
            new_cost = current_cost + total_cost
            avg_price = new_cost / new_shares
            
            portfolio.at[idx, 'shares'] = new_shares
            portfolio.at[idx, 'cost_basis'] = new_cost
            portfolio.at[idx, 'buy_price'] = avg_price
            portfolio.at[idx, 'stop_loss'] = stop_loss
        
        cash -= total_cost
        return cash, portfolio
    
    def _execute_sell(self, ticker: str, shares_sold: float, sell_price: float, reason: str, cash: float, portfolio: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Execute sell order and update portfolio"""
        position_idx = portfolio[portfolio['ticker'] == ticker].index[0]
        position = portfolio.iloc[position_idx]
        
        buy_price = position['buy_price']
        cost_basis = buy_price * shares_sold
        pnl = (sell_price - buy_price) * shares_sold
        total_proceeds = sell_price * shares_sold
        
        # Log the trade
        self._log_trade({
            "Date": self.today,
            "Ticker": ticker,
            "Shares Sold": shares_sold,
            "Sell Price": sell_price,
            "Cost Basis": cost_basis,
            "PnL": pnl,
            "Reason": f"MANUAL SELL - {reason}"
        })
        
        # Update portfolio
        remaining_shares = position['shares'] - shares_sold
        
        if remaining_shares <= 0:
            # Remove position entirely
            portfolio = portfolio.drop(position_idx).reset_index(drop=True)
        else:
            # Reduce position
            portfolio.at[position_idx, 'shares'] = remaining_shares
            portfolio.at[position_idx, 'cost_basis'] = remaining_shares * buy_price
        
        cash += total_proceeds
        return cash, portfolio

    def _log_trade(self, trade_data: Dict[str, Any]):
        """Log trade to trade log CSV"""
        trade_df = pd.DataFrame([trade_data])
        
        if self.data_manager.trade_log_csv.exists():
            existing_df = pd.read_csv(self.data_manager.trade_log_csv)
            trade_df = pd.concat([existing_df, trade_df], ignore_index=True)
        
        trade_df.to_csv(self.data_manager.trade_log_csv, index=False)
        logging.info(f"Trade logged: {trade_data}")

    def _log_stop_loss_sale(self, ticker: str, shares: float, price: float, cost: float, pnl: float):
        """Log stop loss triggered sale"""
        self._log_trade({
            "Date": self.today,
            "Ticker": ticker,
            "Shares Sold": shares,
            "Sell Price": price,
            "Cost Basis": cost * shares,
            "PnL": pnl,
            "Reason": "AUTOMATED SELL - STOP LOSS TRIGGERED"
        })

    def _remove_position(self, portfolio: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove position from portfolio"""
        return portfolio[portfolio['ticker'] != ticker].reset_index(drop=True)

    def _create_summary_row(self, total_value: float, total_pnl: float, cash: float) -> Dict[str, Any]:
        """Create summary row for portfolio results"""
        return {
            "Date": self.today,
            "Ticker": "TOTAL",
            "Shares": "",
            "Buy Price": "",
            "Cost Basis": "",
            "Stop Loss": "",
            "Current Price": "",
            "Total Value": total_value,
            "PnL": total_pnl,
            "Action": "",
            "Cash Balance": cash,
            "Total Equity": total_value + cash
        }

    def _save_portfolio_results(self, results: List[Dict[str, Any]]):
        """Save portfolio results to CSV"""
        new_df = pd.DataFrame(results)
        
        if self.data_manager.portfolio_csv.exists():
            existing_df = pd.read_csv(self.data_manager.portfolio_csv)
            # Remove today's data if it exists
            existing_df = existing_df[existing_df['Date'] != self.today]
            # Append new results
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        combined_df.to_csv(self.data_manager.portfolio_csv, index=False)
        logging.info("Portfolio results saved to CSV")


# ==============================================
# ENHANCED REPORTING AND RESULTS
# ==============================================

class EnhancedReporting:
    """Enhanced reporting with comprehensive analytics and AI insights"""
    
    def __init__(self, config: TradingConfig, data_manager: DataManager, market_data: MarketDataProvider, ai_analyst: AIAnalyst):
        self.config = config
        self.data_manager = data_manager
        self.market_data = market_data
        self.ai_analyst = ai_analyst
        self.analytics = PortfolioAnalytics(config)
        self.today = datetime.today().strftime("%Y-%m-%d")
    
    def generate_daily_report(self, portfolio: pd.DataFrame, cash: float):
        """Generate comprehensive daily report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DAILY TRADING REPORT")
        print(f"📅 {self.today}")
        print("="*80)
        
        # Market overview
        self._display_market_overview()
        
        # Portfolio summary
        self._display_portfolio_summary(portfolio, cash)
        
        # Performance metrics
        self._display_performance_metrics()
        
        # Risk analysis and alerts
        self._display_risk_analysis(portfolio)
        
        # AI insights
        if self.config.get('ai.enabled'):
            self._display_ai_insights(portfolio, cash)
        
        # Trading recommendations
        self._display_trading_recommendations()
        
        print("="*80)
    
    def _display_market_overview(self):
        """Display current market conditions"""
        print("\n📈 MARKET OVERVIEW")
        print("-" * 40)
        
        try:
            market_context = self.market_data.get_market_context()
            
            if market_context:
                for name, data in market_context.items():
                    change_color = "🟢" if data['daily_change'] >= 0 else "🔴"
                    print(f"{change_color} {name}: ${data['price']:.2f} ({data['daily_change']:+.2f}%)")
            else:
                print("Market data unavailable")
                
        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            logging.error(f"Market overview error: {e}")
    
    def _display_portfolio_summary(self, portfolio: pd.DataFrame, cash: float):
        """Display detailed portfolio summary"""
        print(f"\n💼 PORTFOLIO SUMMARY")
        print("-" * 40)
        
        if portfolio.empty:
            print("No positions currently held")
            print(f"💰 Cash: ${cash:,.2f}")
            return
        
        total_market_value = 0.0
        total_pnl = 0.0
        
        print(f"{'Ticker':<8} {'Shares':<8} {'Buy Price':<10} {'Current':<10} {'P&L':<12} {'P&L %':<8} {'Risk':<10}")
        print("-" * 75)
        
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                market_value = current_price * pos['shares']
                pnl = (current_price - pos['buy_price']) * pos['shares']
                pnl_pct = (pnl / (pos['buy_price'] * pos['shares'])) * 100
                risk_per_position = (pos['buy_price'] - pos['stop_loss']) * pos['shares']
                
                total_market_value += market_value
                total_pnl += pnl
                
                status = "🟢" if pnl >= 0 else "🔴"
                
                print(f"{status} {pos['ticker']:<6} {pos['shares']:<8.0f} ${pos['buy_price']:<9.2f} ${current_price:<9.2f} "
                      f"${pnl:<11.2f} {pnl_pct:<7.1f}% ${risk_per_position:<9.2f}")
                      
            except Exception as e:
                print(f"❌ {pos['ticker']:<6} - Error: {str(e)[:30]}")
                logging.warning(f"Could not display position {pos['ticker']}: {e}")
        
        print("-" * 75)
        total_equity = total_market_value + cash
        
        print(f"💰 Cash Balance: ${cash:,.2f}")
        print(f"📈 Market Value: ${total_market_value:,.2f}")
        print(f"📊 Total P&L: ${total_pnl:,.2f}")
        print(f"💎 Total Equity: ${total_equity:,.2f}")
    
    def _display_performance_metrics(self):
        """Display comprehensive performance metrics"""
        print(f"\n📊 PERFORMANCE METRICS")
        print("-" * 40)
        
        try:
            if not self.data_manager.portfolio_csv.exists():
                print("No historical data available")
                return
            
            # Load historical data
            df = pd.read_csv(self.data_manager.portfolio_csv)
            totals = df[df['Ticker'] == 'TOTAL'].copy()
            
            if len(totals) < 2:
                print("Insufficient data for performance calculation")
                return
            
            totals['Date'] = pd.to_datetime(totals['Date'])
            totals = totals.sort_values('Date')
            
            # Calculate returns
            equity = totals['Total Equity'].astype(float)
            returns = equity.pct_change().dropna()
            
            if len(returns) == 0:
                print("No returns data available")
                return
            
            # Get comprehensive metrics
            metrics = self.analytics.calculate_comprehensive_metrics(returns)
            
            # Display key metrics
            print(f"📈 Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"📊 Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"⚡ Volatility: {metrics.get('annualized_volatility', 0):.2%}")
            print(f"🎯 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"🎯 Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
            print(f"🎯 Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
            print(f"🎲 Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"💪 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            # Compare with S&P 500
            self._display_benchmark_comparison(totals)
            
        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            logging.error(f"Performance metrics error: {e}")
    
    def _display_benchmark_comparison(self, portfolio_data: pd.DataFrame):
        """Compare portfolio performance with S&P 500"""
        try:
            start_date = portfolio_data['Date'].min()
            end_date = portfolio_data['Date'].max()
            
            # Get S&P 500 data
            spx_data = self.market_data.get_stock_data('^SPX', f"{(end_date - start_date).days + 5}d")
            
            if not spx_data.empty and len(spx_data) >= 2:
                spx_start = float(spx_data['Close'].iloc[0])
                spx_end = float(spx_data['Close'].iloc[-1])
                spx_return = (spx_end - spx_start) / spx_start
                
                portfolio_start = float(portfolio_data['Total Equity'].iloc[0])
                portfolio_end = float(portfolio_data['Total Equity'].iloc[-1])
                portfolio_return = (portfolio_end - portfolio_start) / portfolio_start
                
                outperformance = portfolio_return - spx_return
                
                print(f"\n📊 BENCHMARK COMPARISON")
                print(f"Portfolio Return: {portfolio_return:.2%}")
                print(f"S&P 500 Return: {spx_return:.2%}")
                
                if outperformance > 0:
                    print(f"🟢 Outperformance: +{outperformance:.2%}")
                else:
                    print(f"🔴 Underperformance: {outperformance:.2%}")
                    
        except Exception as e:
            logging.warning(f"Benchmark comparison error: {e}")
    
    def _display_risk_analysis(self, portfolio: pd.DataFrame):
        """Display risk analysis and alerts"""
        print(f"\n⚠️ RISK ANALYSIS")
        print("-" * 40)
        
        try:
            alerts = self.analytics.check_portfolio_alerts(portfolio, self.market_data)
            
            if alerts:
                for alert in alerts:
                    print(alert)
            else:
                print("✅ No current risk alerts")
                
            # Additional risk metrics
            if not portfolio.empty:
                portfolio_value = 0
                max_risk = 0
                
                for _, pos in portfolio.iterrows():
                    try:
                        current_price = self.market_data.get_current_price(pos['ticker'])
                        position_value = current_price * pos['shares']
                        position_risk = (pos['buy_price'] - pos['stop_loss']) * pos['shares']
                        
                        portfolio_value += position_value
                        max_risk += position_risk
                        
                    except Exception:
                        continue
                
                if portfolio_value > 0:
                    risk_pct = (max_risk / portfolio_value) * 100
                    print(f"📉 Maximum Portfolio Risk: ${max_risk:.2f} ({risk_pct:.1f}%)")
                    
        except Exception as e:
            print(f"❌ Error in risk analysis: {e}")
            logging.error(f"Risk analysis error: {e}")
    
    def _display_ai_insights(self, portfolio: pd.DataFrame, cash: float):
        """Display AI-powered insights and analysis"""
        print(f"\n🤖 AI INSIGHTS")
        print("-" * 40)
        
        try:
            # Prepare data for AI analysis
            portfolio_summary = self._prepare_portfolio_summary(portfolio, cash)
            market_context = self.market_data.get_market_context()
            
            # Get performance metrics for AI context
            metrics = {}
            if self.data_manager.portfolio_csv.exists():
                df = pd.read_csv(self.data_manager.portfolio_csv)
                totals = df[df['Ticker'] == 'TOTAL']
                if len(totals) >= 2:
                    equity = totals['Total Equity'].astype(float)
                    returns = equity.pct_change().dropna()
                    if len(returns) > 0:
                        metrics = self.analytics.calculate_comprehensive_metrics(returns)
            
            # Get AI portfolio analysis
            print("📊 Portfolio Analysis:")
            ai_analysis = self.ai_analyst.get_portfolio_analysis(
                portfolio_summary, market_context, metrics
            )
            print(f"   {ai_analysis}")
            
            # Get stock screening suggestions
            print("\n💡 Stock Screening:")
            current_tickers = portfolio['ticker'].tolist() if not portfolio.empty else []
            screening_suggestions = self.ai_analyst.get_stock_screening_suggestions(current_tickers)
            print(f"   {screening_suggestions}")
            
            # Get risk assessment if there are alerts
            alerts = self.analytics.check_portfolio_alerts(portfolio, self.market_data)
            if alerts:
                print("\n⚠️ Risk Assessment:")
                risk_assessment = self.ai_analyst.get_risk_assessment(
                    portfolio_summary, alerts
                )
                print(f"   {risk_assessment}")
                
        except Exception as e:
            print(f"❌ AI insights unavailable: {e}")
            logging.error(f"AI insights error: {e}")
    
    def _prepare_portfolio_summary(self, portfolio: pd.DataFrame, cash: float) -> str:
        """Prepare portfolio summary for AI analysis"""
        if portfolio.empty:
            return f"Empty portfolio with ${cash:.2f} cash"
        
        summary_parts = [f"Cash: ${cash:.2f}"]
        
        for _, pos in portfolio.iterrows():
            try:
                current_price = self.market_data.get_current_price(pos['ticker'])
                pnl = (current_price - pos['buy_price']) * pos['shares']
                pnl_pct = (pnl / (pos['buy_price'] * pos['shares'])) * 100
                
                summary_parts.append(
                    f"{pos['ticker']}: {pos['shares']} shares, "
                    f"P&L: ${pnl:.2f} ({pnl_pct:+.1f}%), "
                    f"Stop: ${pos['stop_loss']:.2f}"
                )
            except Exception:
                summary_parts.append(f"{pos['ticker']}: {pos['shares']} shares (price unavailable)")
        
        return "\n".join(summary_parts)
    
    def _display_trading_recommendations(self):
        """Display final trading recommendations and instructions"""
        print(f"\n🎯 TRADING RECOMMENDATIONS")
        print("-" * 40)
        print("📋 Today's Action Items:")
        print("• Review AI analysis and risk alerts above")
        print("• Consider position sizing recommendations")
        print("• Monitor positions near stop losses")
        print("• Evaluate new opportunities from screening")
        print("• Update stop losses based on market conditions")
        print()
        print("💭 Decision Framework:")
        print("• You have complete control over all decisions")
        print("• No approval required for any trades")
        print("• Act based on analysis and your judgment")
        print("• Research current prices before making changes")
        print()
        print("⏰ Next Steps:")
        print("• If no immediate action needed, portfolio remains unchanged")
        print("• Use provided analysis for informed decision making")
        print("• Consider copying analysis to external tools if needed")


# ==============================================
# MAIN APPLICATION AND CLI
# ==============================================

class TradingApplication:
    """Main application orchestrating all components"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config = TradingConfig(config_file)
        
        # Set up logging
        setup_logging(self.config)
        
        # Initialize components
        self.data_manager = DataManager(self.config)
        self.market_data = MarketDataProvider(self.config)
        self.ai_analyst = AIAnalyst(self.config)
        self.trading_engine = TradingEngine(self.config, self.data_manager, self.market_data)
        self.reporter = EnhancedReporting(self.config, self.data_manager, self.market_data, self.ai_analyst)
        
        logging.info("Trading application initialized")
    
    def run(self, file: str, data_dir: Path = None, interactive: bool = True, backup: bool = True):
        """Run the main trading application"""
        try:
            logging.info(f"Starting trading session with file: {file}")
            
            # Set data directory if provided
            if data_dir:
                self.data_manager.set_data_dir(data_dir)
            
            # Create backup if requested
            if backup:
                self.data_manager.backup_data()
            
            # Load portfolio state
            portfolio, cash = self.load_latest_portfolio_state(file)
            logging.info(f"Loaded portfolio: {len(portfolio) if isinstance(portfolio, pd.DataFrame) else 'empty'} positions, ${cash:.2f} cash")
            
            # Process portfolio
            portfolio, cash = self.trading