import pandas as pd
import numpy as np
from bentesting import backtest

def backtest_inverted(allocations, price_file="data_testing.csv", initial_capital=100000, pct=0.25):
    """
    INVERTED: Buy BOTTOM 25% (contrarian strategy)
    """
    df = pd.read_csv(price_file)
    price_cols = [c for c in df.columns if "_" not in c and c != "date"]
    prices = df[["date"] + price_cols].set_index("date").sort_index()

    cash = initial_capital
    positions = {}
    equity_curve = []
    daily_pnl = []
    dates = sorted(allocations.keys())

    for i, date in enumerate(dates):
        if date not in prices.index:
            continue

        current_prices = prices.loc[date]

        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value

        target_allocs = allocations[date]
        sorted_assets = sorted(target_allocs.items(), key=lambda x: x[1], reverse=True)
        valid_assets = [(sym, weight) for sym, weight in sorted_assets
                       if sym in current_prices and pd.notna(current_prices[sym]) and current_prices[sym] > 0]

        n_assets = len(valid_assets)
        if n_assets == 0:
            equity_curve.append(portfolio_value)
            daily_pnl.append(0)
            continue

        n_select = max(1, int(n_assets * pct))
        bottom_assets = valid_assets[-n_select:]  # LOWEST weights (inverted)

        # Sell all existing positions
        for symbol, contracts in positions.items():
            if pd.notna(current_prices.get(symbol)) and current_prices[symbol] > 0:
                cash += contracts * current_prices[symbol]
        positions = {}

        # Buy bottom assets with equal weight
        allocation_per_asset = portfolio_value / n_select

        for symbol, _ in bottom_assets:
            price = current_prices[symbol]
            if pd.notna(price) and price > 0:
                num_contracts = int(allocation_per_asset / price)
                if num_contracts > 0:
                    cash -= num_contracts * price
                    positions[symbol] = num_contracts

        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value

        if i > 0:
            pnl = portfolio_value - equity_curve[-1]
            daily_pnl.append(pnl)
        else:
            daily_pnl.append(0)

        equity_curve.append(portfolio_value)

    equity_series = pd.Series(equity_curve, index=dates[:len(equity_curve)])
    pnl_series = pd.Series(daily_pnl, index=dates[:len(daily_pnl)])

    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
    max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100

    return {
        "equity_curve": equity_series,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(equity_curve[-1], 2),
    }

if __name__ == "__main__":
    allocations = backtest()
    results = backtest_inverted(allocations, pct=0.25)

    print(f"INVERTED (Buy Bottom 25%) Results:")
    print(f"Total Return: {results['total_return_pct']}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"Final Value: ${results['final_value']:,.2f}")
