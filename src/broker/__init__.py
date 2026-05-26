"""
Broker Module — MT5 Connection, Simulated Broker, and Protocol.
"""
from .protocol import BrokerProtocol
from .sim import SimBroker

# MT5Manager is imported lazily because MetaTrader5 may not be installed.
# Use: from broker.mt5 import MT5Manager
__all__ = ["BrokerProtocol", "SimBroker"]
