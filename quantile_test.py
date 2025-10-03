import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_scores(prices, lookback=20, vol_window=30):
    """Compute risk-adjusted scores: momentum / (vol-of-vol * drawdown)."""
    momentum = prices.pct_change(lookback)
    vol = prices.pct_change().rolling(vol_window).std()
    vol_of_vol = vol.pct_change().rolling(vol_window).std()
    rolling_max = prices.rolling(vol_window).max()
    dd = (prices - rolling_max) / rolling_max
    dd30 = dd.rolling(vol_window).min().abs()
    score = momentum / ((1 + vol_of_vol) * (1 + dd30))
    return score

def backtest_with_signals(price_file="data_testing.csv", initial_capital=100000,
                          threshold=0.95, stop_loss=0.25, cash_rate=0.04,
                          lookback=20, vol_window=30):
    # Load prices
    df = pd.read_csv(price_file)
    price_cols = [c for c in df.columns if "_" not in c and c != "date"]
    prices = df[["date"] + price_cols].set_index("date").sort_index().astype(float)

    # --- FORWARD-FILL missing prices ---
    prices = prices.ffill().bfill()

    # Compute risk-adjusted scores
    scores = compute_scores(prices, lookback, vol_window)

    # Initialize portfolio
    cash = initial_capital
    positions = {}  # {symbol: {"entry_price": float, "shares": float}}
    equity_curve = []
    daily_pnl = []
    dates = prices.index

    for i, date in enumerate(dates):
        current_prices = prices.loc[date]
        current_scores = scores.loc[date]

        # --- Update positions + stop-loss ---
        to_close = []
        for sym, pos in positions.items():
            price = current_prices.get(sym, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            if price <= pos["entry_price"] * (1 - stop_loss):
                cash += pos["shares"] * price
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # --- Grow idle cash ---
        cash *= (1 + cash_rate / 252)

        # --- Determine buy signals ---
        valid_scores = current_scores.dropna()
        if len(valid_scores) > 0:
            threshold_val = np.quantile(valid_scores, threshold)
            buy_signals = valid_scores[valid_scores >= threshold_val].index.tolist()
        else:
            buy_signals = []

        # --- Execute buys (5% allocation each) ---
        portfolio_value = cash + sum(
            pos["shares"] * current_prices.get(sym, 0)
            for sym, pos in positions.items()
        )
        allocation_per_trade = portfolio_value * 0.05

        for sym in buy_signals:
            price = current_prices.get(sym, np.nan)
            if pd.isna(price) or price <= 0 or sym in positions:
                continue
            shares = allocation_per_trade / price
            cost = shares * price
            if cash >= cost:
                cash -= cost
                positions[sym] = {"entry_price": price, "shares": shares}

        # --- Update portfolio value ---
        position_value = sum(
            pos["shares"] * current_prices.get(sym, 0)
            for sym, pos in positions.items()
        )
        portfolio_value = cash + position_value

        equity_curve.append(portfolio_value)
        daily_pnl.append(portfolio_value - equity_curve[-2] if i > 0 else 0)

    # --- Compute stats ---
    equity_series = pd.Series(equity_curve, index=dates)
    pnl_series = pd.Series(daily_pnl, index=dates)
    total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100
    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 0 else 0
    max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100

    return {
        "equity_curve": equity_series,
        "daily_pnl": pnl_series,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(equity_series.iloc[-1], 2),
        "total_pnl": round(equity_series.iloc[-1] - initial_capital, 2),
    }

if __name__ == "__main__":
    results = backtest_with_signals("data_testing.csv")
    print(f"Total Return: {results['total_return_pct']}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total P/L: ${results['total_pnl']:,.2f}")

    results["equity_curve"].to_csv("signal_backtest_equity.csv")
    results["daily_pnl"].to_csv("signal_backtest_pnl.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(results["equity_curve"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.xticks(results["equity_curve"].index[::100])
    plt.ylabel("Portfolio Value")
    plt.show()
