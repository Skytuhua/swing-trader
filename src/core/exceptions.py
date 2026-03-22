class SwingTraderError(Exception):
    pass


class DataProviderError(SwingTraderError):
    pass


class BrokerError(SwingTraderError):
    pass


class BrokerConnectionError(BrokerError):
    pass


class OrderError(BrokerError):
    pass


class ConfigurationError(SwingTraderError):
    pass


class KillSwitchActiveError(SwingTraderError):
    pass


class RiskLimitExceededError(SwingTraderError):
    pass


class StaleDataError(SwingTraderError):
    pass


class NoTradeError(SwingTraderError):
    """Raised when conditions do not support opening a trade."""
    pass


class InsufficientLiquidityError(SwingTraderError):
    """Raised when a stock does not meet liquidity requirements."""
    pass


class BacktestError(SwingTraderError):
    """Raised for backtesting framework errors."""
    pass
