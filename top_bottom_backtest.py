import pandas as pd
import numpy as np
from bentesting import backtest

def backtest_top_bottom(allocations, price_file="data_testing.csv", initial_capital=100000, top_pct=0.25, bottom_pct=0.25):
    """
    Buy top 25% of assets by score, ignore/short bottom 25%.

    allocations: dict from bentesting.backtest() - {date: {symbol: weight}}
    top_pct: fraction of top assets to buy (0.25 = top 25%)
    bottom_pct: fraction of bottom assets to skip/short (0.25 = bottom 25%)
    """
    df = pd.read_csv(price_file)

    # Get price columns (no underscore)
    price_cols = [c for c in df.columns if "_" not in c and c != "date"]
    prices = df[["date"] + price_cols].set_index("date").sort_index()

    # Track portfolio
    cash = initial_capital
    positions = {}  # {symbol: num_contracts}
    equity_curve = []
    daily_pnl = []

    dates = sorted(allocations.keys())

    for i, date in enumerate(dates):
        if date not in prices.index:
            continue

        current_prices = prices.loc[date]

        # Calculate current portfolio value
        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value

        # Get target allocations for today
        target_allocs = allocations[date]

        # Sort by allocation weight (proxy for score)
        sorted_assets = sorted(target_allocs.items(), key=lambda x: x[1], reverse=True)

        # Filter out assets with no price data
        valid_assets = [(sym, weight) for sym, weight in sorted_assets
                       if sym in current_prices and pd.notna(current_prices[sym]) and current_prices[sym] > 0]

        n_assets = len(valid_assets)
        if n_assets == 0:
            equity_curve.append(portfolio_value)
            daily_pnl.append(0)
            continue

        # Select top 25% and bottom 25%
        n_top = max(1, int(n_assets * top_pct))
        n_bottom = max(1, int(n_assets * bottom_pct))

        top_assets = valid_assets[:n_top]  # highest weights
        # bottom_assets = valid_assets[-n_bottom:]  # lowest weights (for shorting later)

        # Sell all existing positions
        for symbol, contracts in positions.items():
            if pd.notna(current_prices.get(symbol)) and current_prices[symbol] > 0:
                cash += contracts * current_prices[symbol]

        # Clear positions
        positions = {}

        # Buy top assets with equal weight
        allocation_per_asset = portfolio_value / n_top

        for symbol, _ in top_assets:
            price = current_prices[symbol]
            if pd.notna(price) and price > 0:
                num_contracts = int(allocation_per_asset / price)
                if num_contracts > 0:
                    cash -= num_contracts * price
                    positions[symbol] = num_contracts

        # Calculate final portfolio value after rebalance
        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value

        # Track daily P/L
        if i > 0:
            pnl = portfolio_value - equity_curve[-1]
            daily_pnl.append(pnl)
        else:
            daily_pnl.append(0)

        equity_curve.append(portfolio_value)

    # Convert to series
    equity_series = pd.Series(equity_curve, index=dates[:len(equity_curve)])
    pnl_series = pd.Series(daily_pnl, index=dates[:len(daily_pnl)])

    # Calculate stats
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
    max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100

    return {
        "equity_curve": equity_series,
        "daily_pnl": pnl_series,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(equity_curve[-1], 2),
        "total_pnl": round(equity_curve[-1] - initial_capital, 2)
    }


if __name__ == "__main__":
    # Run backtest
    allocations = backtest()
    results = backtest_top_bottom(allocations, top_pct=0.25)

    print(f"Top 25% Strategy Results:")
    print(f"Total Return: {results['total_return_pct']}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total P/L: ${results['total_pnl']:,.2f}")

    # Save results
    results['equity_curve'].to_csv("top_bottom_backtest_equity.csv")
    results['daily_pnl'].to_csv("top_bottom_backtest_pnl.csv")
    print("\nResults saved to top_bottom_backtest_equity.csv and top_bottom_backtest_pnl.csv")
