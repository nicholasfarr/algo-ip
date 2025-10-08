import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

def compute_scores(prices, lookback=100, vol_window=30):
    """Compute risk-adjusted scores: momentum / ((vol-of-vol + 1) * (drawdown + 1))."""
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
    raw_mom= (prices.pct_change(100)+prices.pct_change(50)+prices.pct_change(200))/3
    momentum = raw_mom.apply(
        lambda col: pd.Series(kf.filter(col.dropna().values.reshape(-1, 1))[0].flatten(),
                              index=col.dropna().index)
    )
    nan_padding = pd.DataFrame(np.nan, index=prices.index[:200], columns=prices.columns)
    momentum = pd.concat([nan_padding, momentum]).iloc[:len(prices)]
    daily_ret = prices.pct_change()
    downside = daily_ret.copy()
    downside[downside > 0] = 0  # only negative returns
    dd30 = downside.rolling(vol_window).std()  # downside deviation
    vol_of_vol = dd30.pct_change().rolling(vol_window).std()
    score = momentum / ((vol_of_vol) + (dd30))

    return score

def backtest_with_vol_adjusted_allocation(price_file="data_testing.csv",
                                          initial_capital=100000,
                                          threshold=0.9,
                                          stop_loss=1,
                                          cash_rate=0.04,
                                          lookback=100,
                                          vol_window=30,
                                          max_alloc=0.1,
                                          vol_scale_window=30,
                                          vol_threshold=0.01,
                                          trans_fee=1):
    """
    vol_scale_window: number of days to compute average portfolio volatility
    vol_threshold: above this daily vol, allocations are scaled down
    """
    df = pd.read_csv(price_file)
    price_cols = [c for c in df.columns if "_" not in c and c != "date"]
    prices = df[["date"] + price_cols].set_index("date").sort_index().astype(float)
    prices = prices.ffill().bfill()

    scores = compute_scores(prices, lookback, vol_window)

    cash = initial_capital
    positions = {}
    equity_curve = []
    daily_pnl = []
    dates = prices.index
    common_dates = prices.index.intersection(scores.index)
    print(common_dates)
    for i, date in enumerate(common_dates):
        current_prices = prices.loc[date]
        current_scores = scores.loc[date].dropna()

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
        # --- Compute market volatility proxy ---
        if i >= vol_scale_window:
            recent_prices = prices.iloc[i-vol_scale_window:i]
            daily_vol = recent_prices.pct_change().std().mean()  # average daily vol across assets
        else:
            daily_vol = 0.01  # default small value for first few days

        # Scale allocation down if vol above threshold
        if daily_vol <= vol_threshold:
            vol_multiplier = 1.0
        else:
            # Linear scaling: reduce allocation as vol increases
            vol_multiplier = max(0.1, vol_threshold / daily_vol)


        # --- Select buy candidates ---
        if len(current_scores) > 0:
            threshold_val = np.quantile(current_scores, threshold)
            buy_signals = current_scores[current_scores >= threshold_val]
        else:
            buy_signals = pd.Series(dtype=float)

        if len(buy_signals) > 0:
            rel_scores = buy_signals / buy_signals.max()
            target_allocations = rel_scores * max_alloc * vol_multiplier

            portfolio_value = cash + sum(
                pos["shares"] * current_prices.get(sym, 0) for sym, pos in positions.items()
            )

            for sym, alloc in target_allocations.items():
                price = current_prices.get(sym, np.nan)
                if pd.isna(price) or price <= 0:
                    continue
                target_value = portfolio_value * alloc
                current_value = positions.get(sym, {}).get("shares", 0) * price
                delta_value = target_value - current_value

                if delta_value > 0 and cash >= delta_value:  # Buy more
                    shares_to_buy = delta_value / price
                    cash -= (shares_to_buy * price + trans_fee)
                    if sym in positions:
                        positions[sym]["shares"] += shares_to_buy
                    else:
                        positions[sym] = {"entry_price": price, "shares": shares_to_buy}
                elif delta_value < 0:  # Sell excess
                    shares_to_sell = -delta_value / price
                    if sym in positions:
                        positions[sym]["shares"] -= shares_to_sell
                        cash += shares_to_sell * price
                        if positions[sym]["shares"] <= 0:
                            del positions[sym]

        # --- Update portfolio value ---
        position_value = sum(
            pos["shares"] * current_prices.get(sym, 0) for sym, pos in positions.items()
        )
        portfolio_value = cash + position_value
        equity_curve.append(portfolio_value)
        daily_pnl.append(portfolio_value - equity_curve[-2] if i > 0 else 0)

    # --- Compute stats ---
    equity_series = pd.Series(equity_curve, index=common_dates)
    pnl_series = pd.Series(daily_pnl, index=dates)
    total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100
    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 0 else 0
    max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
    downside_returns = daily_returns.copy()
    downside_returns[downside_returns > 0] = 0  # keep only negative returns
    downside_std = downside_returns.std()
    sortino = (daily_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    return {
        "equity_curve": equity_series,
        "daily_pnl": pnl_series,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(equity_series.iloc[-1], 2),
        "total_pnl": round(equity_series.iloc[-1] - initial_capital, 2),
    }

if __name__ == "__main__":
    results = backtest_with_vol_adjusted_allocation("data_testing.csv")
    print(f"Total Return: {results['total_return_pct']}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"Sortino Ratio: {results['sortino_ratio']}")
    print(f"Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total P/L: ${results['total_pnl']:,.2f}")

    results["equity_curve"].to_csv("signal_backtest_only_equity.csv")
    results["daily_pnl"].to_csv("signal_backtest_pnl.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(results["equity_curve"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.xticks(results["equity_curve"].index[::400])
    plt.ylabel("Portfolio Value")
    plt.show()
