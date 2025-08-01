# Start Your Own

This folder lets you run the trading experiment on your own computer. It contains two small scripts and the CSV files they produce.

Run the commands below from the repository root. The scripts automatically
save their CSV data inside this folder.

## Trading_Script.py

This script updates your portfolio and logs trades.

1. **Install Python packages**
   ```bash
   pip install pandas yfinance numpy matplotlib
   ```
2. **Set up your portfolio**
   - Open `Trading_Script.py`.
   - At the bottom of the file, change `starting_capital` and the `chatgpt_portfolio` list to match your tickers, share counts, stop losses, and buy prices.
3. **Run the script**
   ```bash
   python "Start Your Own/Trading_Script.py"
   ```
4. **Follow the prompts**
   - The script asks if you want to record manual buys or sells before it fetches prices.
   - Daily results are saved to `chatgpt_portfolio_update.csv` and any trades are added to `chatgpt_trade_log.csv`.

## Generate_Graph.py

This script draws a graph of your portfolio versus the S&P 500.

1. **Ensure you have portfolio data**
   - Run `Trading_Script.py` at least once so `chatgpt_portfolio_update.csv` has data.
2. **Run the graph script**
   ```bash
   python "Start Your Own/Generate_Graph.py" --baseline-equity 100
   ```
   - Optional flags `--start-date` and `--end-date` accept dates in `YYYY-MM-DD` format. For example:
   ```bash
   python "Start Your Own/Generate_Graph.py" --baseline-equity 100 --start-date 2023-01-01 --end-date 2023-12-31
   ```
3. **View the chart**
   - A window opens showing your portfolio value and a $100 investment in the S&P 500.

**Note: All prompting is manual currently.**

All of this is still VERY NEW, so there is likely bugs. Please reach out if you find an issue or have a question.

Both scripts are designed for beginners, feel free to experiment and modify them as you learn.
