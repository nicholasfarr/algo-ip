import pandas as pd
import numpy as np

def compute_scores(prices, lookback=20):
    """Momentum-based scores (N-day percentage returns)."""
    return prices.pct_change(lookback)

def backtest_with_signals(price_file="data_testing.csv", initial_capital=100000,
                          threshold=0.95, stop_loss=0.25, cash_rate=0.04, lookback=20):
    # Load prices
    df = pd.read_csv(price_file)
    price_cols = [c for c in df.columns if "_" not in c and c != "date"]
    prices = df[["date"] + price_cols].set_index("date").sort_index().astype(float)

    # Compute scores
    scores = compute_scores(prices, lookback)

    # Portfolio trackers
    cash = initial_capital
    positions = {}   # {symbol: {"entry_price": float, "shares": int}}
    equity_curve = []
    daily_pnl = []
    dates = prices.index

    for i, date in enumerate(dates):
        current_prices = prices.loc[date]
        current_scores = scores.loc[date]

        # 1. Update value of existing positions
        position_value = 0
        to_close = []
        for sym, pos in positions.items():
            price = current_prices.get(sym, np.nan)
            if pd.isna(price):
                continue
            value = pos["shares"] * price
            position_value += value

            # Stop loss check
            if price <= pos["entry_price"] * (1 - stop_loss):
                cash += value  # sell at current price
                to_close.append(sym)

        # Remove stopped out positions
        for sym in to_close:
            del positions[sym]

        # 2. Interest on idle cash
        cash *= (1 + cash_rate / 252)

        # 3. Generate buy signals (scores in top percentile)
        valid_scores = current_scores.dropna()
        if len(valid_scores) > 0:
            threshold_val = np.quantile(valid_scores, threshold)
            buy_signals = valid_scores[valid_scores >= threshold_val].index.tolist()
        else:
            buy_signals = []

        # 4. Execute buys (5% allocation each)
        portfolio_value = cash + sum(
            pos["shares"] * current_prices.get(sym, 0)
            for sym, pos in positions.items()
            if not pd.isna(current_prices.get(sym, np.nan))
        )
        allocation_per_trade = portfolio_value * 0.05

        for sym in buy_signals:
            price = current_prices.get(sym, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            if sym not in positions:  # only open if not already held
                shares = int(allocation_per_trade / price)
                if shares > 0:
                    cost = shares * price
                    if cash >= cost:
                        cash -= cost
                        positions[sym] = {"entry_price": price, "shares": shares}

        # 5. Update portfolio value
        position_value = sum(
            pos["shares"] * current_prices.get(sym, 0)
            for sym, pos in positions.items()
            if not pd.isna(current_prices.get(sym, np.nan))
        )
        portfolio_value = cash + position_value

        # Track equity and pnl
        equity_curve.append(portfolio_value)
        if i > 0:
            daily_pnl.append(portfolio_value - equity_curve[-2])
        else:
            daily_pnl.append(0)

    # Convert to series
    equity_series = pd.Series(equity_curve, index=dates)
    pnl_series = pd.Series(daily_pnl, index=dates)

    # Stats
    total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100
    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 else 0
    max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100

    return {
        "equity_curve": equity_series,
        "daily_pnl": pnl_series,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(equity_series.iloc[-1], 2),
        "total_pnl": round(equity_series.iloc[-1] - initial_capital, 2)
    }

if __name__ == "__main__":
    results = backtest_with_signals("data_testing.csv")
    print(f"Total Return: {results['total_return_pct']}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total P/L: ${results['total_pnl']:,.2f}\n")

    # Save results
    results['equity_curve'].to_csv("signal_backtest_equity.csv")
    results['daily_pnl'].to_csv("signal_backtest_pnl.csv")
    print("\nResults saved to signal_backtest_equity.csv and signal_backtest_pnl.csv")
