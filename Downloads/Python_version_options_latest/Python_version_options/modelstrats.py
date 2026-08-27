from AlgorithmImports import *          # resolved via local stub
from models import *
import pandas as pd

# Your New Python File
MISPRICING_THRESHOLD = 0.03  # 3% minimum edge required after spread costs
DEFAULT_TRADE_QUANTITY = 1 # Number of contracts

# Define classes for each model-based strategy

class MMARStrategy:
    def __init__(self, mispricing_threshold=MISPRICING_THRESHOLD, default_trade_quantity=DEFAULT_TRADE_QUANTITY):
        self.MISPRICING_THRESHOLD = mispricing_threshold
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity

    def generate_signals(self, date, market_data, options_data, portfolio, backtester):
        signals = []
        if options_data is not None:
            # Ensure 'options_data' is a DataFrame if it's not already
            if not isinstance(options_data, pd.DataFrame):
                options_data = pd.DataFrame(options_data)

            for index, option in options_data.iterrows():
                model_price = option.get('MMAR Call') if option['option_type'] == 'call' else option.get('MMAR Put')
                actual_price = option.get('Actual Call Price') if option['option_type'] == 'call' else option.get('Actual Put Price')
                spread_cost = option.get('spread_cost', 0.0) or 0.0

                if model_price is not None and actual_price is not None and actual_price > 0:
                    # Buy only if model edge exceeds the effective ask (mid + half-spread)
                    if model_price > actual_price + spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'buy',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
                    # Write (short) if model price is below the effective bid: option is overpriced
                    elif model_price < actual_price - spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'write',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
        return signals

class BlackScholesStrategy:
    def __init__(self, mispricing_threshold=MISPRICING_THRESHOLD, default_trade_quantity=DEFAULT_TRADE_QUANTITY):
        self.MISPRICING_THRESHOLD = mispricing_threshold
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity

    def generate_signals(self, date, market_data, options_data, portfolio, backtester):
        signals = []
        if options_data is not None:
            if not isinstance(options_data, pd.DataFrame):
                options_data = pd.DataFrame(options_data)

            for index, option in options_data.iterrows():
                model_price = option.get('BS Call') if option['option_type'] == 'call' else option.get('BS Put')
                actual_price = option.get('Actual Call Price') if option['option_type'] == 'call' else option.get('Actual Put Price')
                spread_cost = option.get('spread_cost', 0.0) or 0.0

                if model_price is not None and actual_price is not None and actual_price > 0:
                    if model_price > actual_price + spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'buy',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
                    elif model_price < actual_price - spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'sell',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
        return signals

class MertonStrategy:
    def __init__(self, mispricing_threshold=MISPRICING_THRESHOLD, default_trade_quantity=DEFAULT_TRADE_QUANTITY):
        self.MISPRICING_THRESHOLD = mispricing_threshold
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity

    def generate_signals(self, date, market_data, options_data, portfolio, backtester):
        signals = []
        if options_data is not None:
            if not isinstance(options_data, pd.DataFrame):
                options_data = pd.DataFrame(options_data)

            for index, option in options_data.iterrows():
                model_price = option.get('Merton Call') if option['option_type'] == 'call' else option.get('Merton Put')
                actual_price = option.get('Actual Call Price') if option['option_type'] == 'call' else option.get('Actual Put Price')
                spread_cost = option.get('spread_cost', 0.0) or 0.0

                if model_price is not None and actual_price is not None and actual_price > 0:
                    if model_price > actual_price + spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'buy',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
                    elif model_price < actual_price - spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'sell',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
        return signals

class HestonStrategy:
    def __init__(self, mispricing_threshold=MISPRICING_THRESHOLD, default_trade_quantity=DEFAULT_TRADE_QUANTITY):
        self.MISPRICING_THRESHOLD = mispricing_threshold
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity

    def generate_signals(self, date, market_data, options_data, portfolio, backtester):
        signals = []
        if options_data is not None:
            if not isinstance(options_data, pd.DataFrame):
                options_data = pd.DataFrame(options_data)

            for index, option in options_data.iterrows():
                model_price = option.get('Heston Call') if option['option_type'] == 'call' else option.get('Heston Put')
                actual_price = option.get('Actual Call Price') if option['option_type'] == 'call' else option.get('Actual Put Price')
                spread_cost = option.get('spread_cost', 0.0) or 0.0

                if model_price is not None and actual_price is not None and actual_price > 0:
                    if model_price > actual_price + spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'buy',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
                    elif model_price < actual_price - spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'sell',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
        return signals

class BatesStrategy:
    def __init__(self, mispricing_threshold=MISPRICING_THRESHOLD, default_trade_quantity=DEFAULT_TRADE_QUANTITY):
        self.MISPRICING_THRESHOLD = mispricing_threshold
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity

    def generate_signals(self, date, market_data, options_data, portfolio, backtester):
        signals = []
        if options_data is not None:
            if not isinstance(options_data, pd.DataFrame):
                options_data = pd.DataFrame(options_data)

            for index, option in options_data.iterrows():
                model_price = option.get('Bates Call') if option['option_type'] == 'call' else option.get('Bates Put')
                actual_price = option.get('Actual Call Price') if option['option_type'] == 'call' else option.get('Actual Put Price')
                spread_cost = option.get('spread_cost', 0.0) or 0.0

                if model_price is not None and actual_price is not None and actual_price > 0:
                    if model_price > actual_price + spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'buy',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
                    elif model_price < actual_price - spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'sell',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
        return signals

class MixedStrategy:
    def __init__(self, mispricing_threshold=MISPRICING_THRESHOLD, default_trade_quantity=DEFAULT_TRADE_QUANTITY):
        self.MISPRICING_THRESHOLD = mispricing_threshold
        self.DEFAULT_TRADE_QUANTITY = default_trade_quantity

    def generate_signals(self, date, market_data, options_data, portfolio, backtester):
        signals = []
        if options_data is not None:
            if not isinstance(options_data, pd.DataFrame):
                options_data = pd.DataFrame(options_data)

            for index, option in options_data.iterrows():
                model_price = option.get('Mixed Call') if option['option_type'] == 'call' else option.get('Mixed Put')
                actual_price = option.get('Actual Call Price') if option['option_type'] == 'call' else option.get('Actual Put Price')
                spread_cost = option.get('spread_cost', 0.0) or 0.0

                if model_price is not None and actual_price is not None and actual_price > 0:
                    if model_price > actual_price + spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'buy',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
                    elif model_price < actual_price - spread_cost:
                        signals.append({
                            'contract_symbol': option['contract_symbol'],
                            'type': 'sell',
                            'quantity': self.DEFAULT_TRADE_QUANTITY,
                            'price': actual_price,
                        })
        return signals