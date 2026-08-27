# AlgorithmImports.py
# Stub replacing QuantConnect's AlgorithmImports for standalone backtesting.
# Drop this file in the same directory as the other source files.
# All files that begin with `from AlgorithmImports import *` will
# silently resolve through this stub instead of the QC runtime.

class Resolution:
    DAILY  = 'daily'
    HOUR   = 'hour'
    MINUTE = 'minute'

class OptionRight:
    CALL = 'call'
    PUT  = 'put'

class OrderStatus:
    FILLED     = 'filled'
    SUBMITTED  = 'submitted'
    INVALID    = 'invalid'

class OrderDirection:
    SELL = 'sell'
    BUY  = 'buy'

class DayOfWeek:
    MONDAY    = 0
    TUESDAY   = 1
    WEDNESDAY = 2
    THURSDAY  = 3
    FRIDAY    = 4

# Make `from AlgorithmImports import *` export everything above
__all__ = [
    'Resolution', 'OptionRight', 'OrderStatus',
    'OrderDirection', 'DayOfWeek',
]
