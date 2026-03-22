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
