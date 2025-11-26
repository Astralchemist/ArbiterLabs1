"""
Base Broker Interface

Abstract base class for broker implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    side: str = "long"  # long or short


@dataclass
class Account:
    """Account data structure"""
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    buying_power: float
    positions: List[Position]


class BrokerBase(ABC):
    """
    Abstract base class for broker implementations.

    All broker connectors should inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: Dict):
        """
        Initialize broker connection.

        Args:
            config: Broker configuration dictionary
        """
        self.config = config
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker API.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from broker API.

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    def get_account(self) -> Account:
        """
        Get account information.

        Returns:
            Account object with current account data
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for specific symbol.

        Args:
            symbol: Symbol/ticker

        Returns:
            Position object or None if no position
        """
        pass

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """
        Place an order.

        Args:
            order: Order object

        Returns:
            Order ID as string
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get status of an order.

        Args:
            order_id: Order ID

        Returns:
            OrderStatus enum
        """
        pass

    @abstractmethod
    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get order history.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of Order objects
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """
        Close position for a symbol.

        Args:
            symbol: Symbol/ticker

        Returns:
            True if position closed successfully
        """
        pass

    @abstractmethod
    def close_all_positions(self) -> bool:
        """
        Close all open positions.

        Returns:
            True if all positions closed successfully
        """
        pass

    def buy_market(self, symbol: str, quantity: float) -> str:
        """
        Convenience method to place market buy order.

        Args:
            symbol: Symbol/ticker
            quantity: Quantity to buy

        Returns:
            Order ID
        """
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        return self.place_order(order)

    def sell_market(self, symbol: str, quantity: float) -> str:
        """
        Convenience method to place market sell order.

        Args:
            symbol: Symbol/ticker
            quantity: Quantity to sell

        Returns:
            Order ID
        """
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        return self.place_order(order)

    def buy_limit(self, symbol: str, quantity: float, price: float) -> str:
        """
        Convenience method to place limit buy order.

        Args:
            symbol: Symbol/ticker
            quantity: Quantity to buy
            price: Limit price

        Returns:
            Order ID
        """
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        return self.place_order(order)

    def sell_limit(self, symbol: str, quantity: float, price: float) -> str:
        """
        Convenience method to place limit sell order.

        Args:
            symbol: Symbol/ticker
            quantity: Quantity to sell
            price: Limit price

        Returns:
            Order ID
        """
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        return self.place_order(order)
