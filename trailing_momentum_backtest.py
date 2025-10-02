import pandas as pd
import numpy as np

def backtest_trailing_momentum(price_file="data_testing.csv", initial_capital=100000,
                                lookback=20, top_pct=0.25):
    """
    Calculate TRUE trailing momentum from price history and backtest

    lookback: number of days to look back for momentum calculation
    top_pct: top X% of assets by momentum to buy
    """
    df = pd.read_csv(price_file)

    # Get price columns only
    price_cols = [c for c in df.columns if "_" not in c and c != "date"]
    prices = df[["date"] + price_cols].copy()
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.set_index('date').sort_index()

    # Calculate trailing returns for each asset
    # Returns over last 'lookback' days
    trailing_returns = prices.pct_change(periods=lookback)

    # Calculate rolling volatility
    rolling_vol = prices.pct_change().rolling(window=lookback).std()

    # Portfolio tracking
    cash = initial_capital
    positions = {}
    equity_curve = []
    trade_dates = []

    # Start after we have enough history
    start_idx = lookback + 20
    dates = prices.index[start_idx:]

    for i, date in enumerate(dates):
        # Get current prices
        current_prices = prices.loc[date]

        # Calculate portfolio value
        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value

        # Get trailing momentum and vol for all assets
        momentum_scores = {}
        for symbol in price_cols:
            if pd.notna(current_prices[symbol]) and current_prices[symbol] > 0:
                mom = trailing_returns.loc[date, symbol]
                vol = rolling_vol.loc[date, symbol]

                if pd.notna(mom) and pd.notna(vol) and vol > 0:
                    # Score = momentum / volatility (risk-adjusted momentum)
                    score = mom / vol
                    momentum_scores[symbol] = score

        if len(momentum_scores) == 0:
            equity_curve.append(portfolio_value)
            trade_dates.append(date)
            continue

        # Sort by momentum score
        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top X%
        n_select = max(1, int(len(sorted_assets) * top_pct))
        top_assets = sorted_assets[:n_select]

        # Rebalance: sell all positions
        for symbol, contracts in positions.items():
            if pd.notna(current_prices.get(symbol)) and current_prices[symbol] > 0:
                cash += contracts * current_prices[symbol]
        positions = {}

        # Buy top momentum assets with equal weight
        allocation_per_asset = portfolio_value / n_select

        for symbol, score in top_assets:
            price = current_prices[symbol]
            if pd.notna(price) and price > 0:
                num_contracts = int(allocation_per_asset / price)
                if num_contracts > 0:
                    cash -= num_contracts * price
                    positions[symbol] = num_contracts

        # Recalculate portfolio value
        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value

        equity_curve.append(portfolio_value)
        trade_dates.append(date)

    # Calculate statistics
    equity_series = pd.Series(equity_curve, index=trade_dates)
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    daily_returns = equity_series.pct_change().dropna()

    sharpe = 0
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    max_dd = 0
    if len(equity_curve) > 0:
        max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100

    return {
        "equity_curve": equity_series,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(equity_curve[-1], 2),
    }


if __name__ == "__main__":
    print("Running TRAILING MOMENTUM backtest (calculated from actual price history)...")

    # Test with different lookback periods
    for lookback in [20, 50, 100]:
        print(f"\n{'='*60}")
        print(f"Lookback Period: {lookback} days")
        print('='*60)
        results = backtest_trailing_momentum(lookback=lookback, top_pct=0.25)

        print(f"Total Return: {results['total_return_pct']}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']}")
        print(f"Max Drawdown: {results['max_drawdown_pct']}%")
        print(f"Final Value: ${results['final_value']:,.2f}")
