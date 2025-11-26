"""
Alpaca Broker Executor

Connects to Alpaca for US stocks trading.
"""

from typing import Dict, List, Optional
from datetime import datetime
import os

from .broker_base import (
    BrokerBase, Order, Position, Account,
    OrderType, OrderSide, OrderStatus
)


class AlpacaExecutor(BrokerBase):
    """
    Alpaca broker implementation.

    Supports US stocks trading with commission-free trades.
    """

    def __init__(self, config: Dict):
        """
        Initialize Alpaca connection.

        Args:
            config: Configuration dict with api_key, api_secret, base_url

        Example config:
            {
                'api_key': 'YOUR_API_KEY',
                'api_secret': 'YOUR_API_SECRET',
                'base_url': 'https://paper-api.alpaca.markets'  # or live URL
            }
        """
        super().__init__(config)

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                MarketOrderRequest, LimitOrderRequest, StopOrderRequest
            )
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
        except ImportError:
            raise ImportError(
                "alpaca-py not installed. Install with: pip install alpaca-py"
            )

        self.api_key = config.get('api_key') or os.getenv('ALPACA_API_KEY')
        self.api_secret = config.get('api_secret') or os.getenv('ALPACA_API_SECRET')
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')

        self.client = None
        self.AlpacaOrderSide = AlpacaOrderSide
        self.TimeInForce = TimeInForce
        self.MarketOrderRequest = MarketOrderRequest
        self.LimitOrderRequest = LimitOrderRequest
        self.StopOrderRequest = StopOrderRequest

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            from alpaca.trading.client import TradingClient

            self.client = TradingClient(
                self.api_key,
                self.api_secret,
                paper=('paper' in self.base_url)
            )

            # Test connection
            account = self.client.get_account()
            self.is_connected = True
            print(f"Connected to Alpaca. Account: {account.account_number}")
            return True

        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from Alpaca."""
        self.client = None
        self.is_connected = False
        return True

    def get_account(self) -> Account:
        """Get account information."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        acc = self.client.get_account()

        return Account(
            balance=float(acc.cash),
            equity=float(acc.equity),
            margin_used=0.0,  # Alpaca doesn't use traditional margin
            margin_available=float(acc.buying_power),
            buying_power=float(acc.buying_power),
            positions=self.get_positions()
        )

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        positions = self.client.get_all_positions()

        return [
            Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pl),
                realized_pnl=0.0,
                side="long" if float(pos.qty) > 0 else "short"
            )
            for pos in positions
        ]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            pos = self.client.get_open_position(symbol)

            return Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pl),
                side="long" if float(pos.qty) > 0 else "short"
            )
        except:
            return None

    def place_order(self, order: Order) -> str:
        """Place an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        # Convert order side
        side = (self.AlpacaOrderSide.BUY if order.side == OrderSide.BUY
                else self.AlpacaOrderSide.SELL)

        # Create order request based on type
        if order.order_type == OrderType.MARKET:
            order_req = self.MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=self.TimeInForce.DAY
            )
        elif order.order_type == OrderType.LIMIT:
            order_req = self.LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=self.TimeInForce.DAY,
                limit_price=order.price
            )
        elif order.order_type == OrderType.STOP:
            order_req = self.StopOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=self.TimeInForce.DAY,
                stop_price=order.stop_price
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        # Submit order
        submitted_order = self.client.submit_order(order_req)

        return submitted_order.id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"Failed to cancel order: {e}")
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        order = self.client.get_order_by_id(order_id)

        status_map = {
            'new': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'pending_new': OrderStatus.PENDING,
        }

        return status_map.get(order.status, OrderStatus.PENDING)

    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        # Get closed orders
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol] if symbol else None
        )

        orders = self.client.get_orders(request)

        return [
            Order(
                symbol=o.symbol,
                side=OrderSide.BUY if o.side == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET if o.type == 'market' else OrderType.LIMIT,
                quantity=float(o.qty),
                price=float(o.limit_price) if o.limit_price else None,
                order_id=o.id,
                status=self._map_status(o.status),
                filled_quantity=float(o.filled_qty or 0),
                average_fill_price=float(o.filled_avg_price or 0),
                timestamp=o.created_at
            )
            for o in orders
        ]

    def close_position(self, symbol: str) -> bool:
        """Close position for a symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            self.client.close_position(symbol)
            return True
        except Exception as e:
            print(f"Failed to close position: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all open positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            self.client.close_all_positions(cancel_orders=True)
            return True
        except Exception as e:
            print(f"Failed to close all positions: {e}")
            return False

    def _map_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca status to OrderStatus enum."""
        status_map = {
            'new': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
        }
        return status_map.get(alpaca_status, OrderStatus.PENDING)
