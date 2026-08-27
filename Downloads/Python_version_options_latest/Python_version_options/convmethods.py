from AlgorithmImports import *          # resolved via local stub
import pandas as pd

# Your New Python File
DEFAULT_TRADE_QUANTITY = 1 # Number of contracts

class BuyAndHoldStrategy:
    def __init__(self, default_trade_quantity=DEFAULT_TRADE_QUANTITY):
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity

    def generate_signals(self, date, market_data, options_data_df, portfolio, backtester):
        signals = []
        current_underlying_price = market_data.get('SPY') if isinstance(market_data, (pd.Series, dict)) else None
        if current_underlying_price is None or options_data_df is None or options_data_df.empty:
            return signals

        call_options = options_data_df[options_data_df['option_type'] == 'call'].copy()
        if call_options.empty:
            return signals

        closest_call = call_options.iloc[
            (call_options['Model Strike'] - current_underlying_price).abs().argsort()[:1]
        ].iloc[0]

        # Skip if we already hold a call position (genuine hold — don't rebuy weekly)
        if hasattr(portfolio, '_positions'):
            held_calls = [s for s, p in portfolio._positions.items()
                          if p.quantity > 0 and 'C' in str(s).split()[-1]]
            if held_calls:
                return signals

        if closest_call['Actual Call Price'] and closest_call['Actual Call Price'] > 0:
            signals.append({
                'contract_symbol': closest_call['contract_symbol'],
                'type': 'buy',
                'quantity': self.DEFAULT_TRADE_QUANTITY,
                'price': closest_call['Actual Call Price'],
            })
        return signals

class MomentumStrategy:
    def __init__(self, default_trade_quantity=DEFAULT_TRADE_QUANTITY, lookback_period=5, momentum_threshold=0.02):
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity
        self.lookback_period = lookback_period
        self.MOMENTUM_THRESHOLD = momentum_threshold

    def generate_signals(self, date, market_data, options_data_df, portfolio, backtester):
        signals = []
        
        # Access the global prices DataFrame if it exists
        if options_data_df is None or options_data_df.empty:
            return signals
        # Use backtester's historical data if available
        historical_prices = backtester.history(backtester._underlying_symbol, self.lookback_period + 1, Resolution.DAILY)
        if historical_prices.empty:
            return signals
        historical_prices = historical_prices['close'].dropna()

        if len(historical_prices) > self.lookback_period:
            # Calculate simple momentum
            past_price = historical_prices.iloc[-self.lookback_period]
            current_price = historical_prices.iloc[-1]

            if past_price > 0:
                 momentum = (current_price - past_price) / past_price
            else:
                 momentum = 0

            if options_data_df is not None and not options_data_df.empty:
                 # Filter for ATM options to trade
                 current_underlying_price = current_price
                 atm_options = options_data_df[
                     (options_data_df['Model Strike'] - current_underlying_price).abs() / current_underlying_price < 0.02 # Within 2% of ATM
                 ].copy()

                 if not atm_options.empty:
                     # Trade calls if momentum is positive, puts if momentum is negative
                     if momentum > self.MOMENTUM_THRESHOLD:
                        # Buy ATM Call options
                        call_options = atm_options[atm_options['option_type'] == 'call']
                        if not call_options.empty:
                             option_to_trade = call_options.iloc[0]
                             signals.append({
                                 'contract_symbol': option_to_trade['contract_symbol'],
                                 'type': 'buy',
                                 'quantity': self.DEFAULT_TRADE_QUANTITY,
                                 'price': option_to_trade['Actual Call Price'],
                             })
                     elif momentum < -self.MOMENTUM_THRESHOLD:
                         # Buy ATM Put options
                         put_options = atm_options[atm_options['option_type'] == 'put']
                         if not put_options.empty:
                             option_to_trade = put_options.iloc[0]
                             signals.append({
                                 'contract_symbol': option_to_trade['contract_symbol'],
                                 'type': 'buy',
                                 'quantity': self.DEFAULT_TRADE_QUANTITY,
                                 'price': option_to_trade['Actual Put Price'],
                             })

        return signals

class MeanReversionStrategy:
    def __init__(self, default_trade_quantity=DEFAULT_TRADE_QUANTITY, ma_period=20, deviation_threshold=0.02):
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity
        self.ma_period = ma_period
        self.DEVIATION_THRESHOLD = deviation_threshold

    def generate_signals(self, date, market_data, options_data_df, portfolio, backtester):
        signals = []

        # Access the global prices DataFrame if it exists
        if options_data_df is None or options_data_df.empty:
            return signals
        historical_prices = backtester.history(backtester._underlying_symbol, self.ma_period + 1, Resolution.DAILY)
        if historical_prices.empty:
            return signals
        historical_prices = historical_prices['close'].dropna()

        if len(historical_prices) > self.ma_period:
            # Calculate the moving average and current price
            moving_average = historical_prices.rolling(window=self.ma_period).mean().iloc[-1]
            current_price = historical_prices.iloc[-1]

            if moving_average > 1e-9: # Avoid division by zero
                 deviation = (current_price - moving_average) / moving_average
            else:
                 deviation = 0

            if options_data_df is not None and not options_data_df.empty:
                 # Filter for OTM options to trade
                 current_underlying_price = current_price
                 # OTM definition: Strike significantly away from current price
                 otm_calls = options_data_df[
                     (options_data_df['option_type'] == 'call') &
                     (options_data_df['Model Strike'] > current_underlying_price * (1 + 0.02)) # More than 2% OTM
                 ].copy()
                 otm_puts = options_data_df[
                     (options_data_df['option_type'] == 'put') &
                     (options_data_df['Model Strike'] < current_underlying_price * (1 - 0.02)) # More than 2% OTM
                 ].copy()


                 # Price below MA: expect bounce — buy the nearest OTM call
                 if deviation < -self.DEVIATION_THRESHOLD:
                     if not otm_calls.empty:
                         option_to_trade = otm_calls.sort_values('Model Strike').iloc[0]
                         signals.append({
                             'contract_symbol': option_to_trade['contract_symbol'],
                             'type': 'buy',
                             'quantity': self.DEFAULT_TRADE_QUANTITY,
                             'price': option_to_trade['Actual Call Price'],
                         })
                 # Price above MA: expect pullback — buy the nearest OTM put
                 elif deviation > self.DEVIATION_THRESHOLD:
                     if not otm_puts.empty:
                         option_to_trade = otm_puts.sort_values('Model Strike', ascending=False).iloc[0]
                         signals.append({
                             'contract_symbol': option_to_trade['contract_symbol'],
                             'type': 'buy',
                             'quantity': self.DEFAULT_TRADE_QUANTITY,
                             'price': option_to_trade['Actual Put Price'],
                         })

        return signals