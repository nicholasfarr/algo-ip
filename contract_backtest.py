import pandas as pd
import numpy as np
from bentesting import backtest

def backtest_contracts(allocations, price_file="data_testing.csv", initial_capital=100000):
   
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
        )
        portfolio_value = cash + position_value

        # Get target allocations for today
        target_allocs = allocations[date]

        # Rebalance: sell all current positions, buy new ones
        # Sell existing positions
        for symbol, contracts in positions.items():
            if pd.notna(current_prices.get(symbol)):
                cash += contracts * current_prices[symbol]

        # Clear positions
        positions = {}

        # Buy new positions based on allocations
        for symbol, weight in target_allocs.items():
            if weight > 0 and symbol in current_prices and pd.notna(current_prices[symbol]):
                price = current_prices[symbol]
                if pd.notna(price) and price > 0 and not np.isnan(price):
                    # Allocate capital based on weight
                    allocated_capital = portfolio_value * weight
                    if pd.notna(allocated_capital) and not np.isnan(allocated_capital):
                        num_contracts = int(allocated_capital / price)
                        if num_contracts > 0:
                            cash -= num_contracts * price
                            positions[symbol] = num_contracts

        # Calculate final portfolio value after rebalance
        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
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
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 else 0
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
    results = backtest_contracts(allocations)

    print(f"Total Return: {results['total_return_pct']}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total P/L: ${results['total_pnl']:,.2f}")

    # Save results
    results['equity_curve'].to_csv("contract_backtest_equity.csv")
    results['daily_pnl'].to_csv("contract_backtest_pnl.csv")
    print("\nResults saved to contract_backtest_equity.csv and contract_backtest_pnl.csv")
