"""
Binance Executor

Connects to Binance for cryptocurrency trading (spot and futures).
"""

from typing import Dict, List, Optional
from datetime import datetime
import os

from .broker_base import (
    BrokerBase, Order, Position, Account,
    OrderType, OrderSide, OrderStatus
)


class BinanceExecutor(BrokerBase):
    """
    Binance broker implementation for crypto trading.
    """

    def __init__(self, config: Dict):
        """
        Initialize Binance connection.

        Args:
            config: Configuration dict with api_key, api_secret, testnet flag

        Example config:
            {
                'api_key': 'YOUR_API_KEY',
                'api_secret': 'YOUR_API_SECRET',
                'testnet': True  # Use testnet for paper trading
            }
        """
        super().__init__(config)

        try:
            from binance.client import Client
            from binance.enums import *
        except ImportError:
            raise ImportError(
                "python-binance not installed. Install with: pip install python-binance"
            )

        self.api_key = config.get('api_key') or os.getenv('BINANCE_API_KEY')
        self.api_secret = config.get('api_secret') or os.getenv('BINANCE_API_SECRET')
        self.testnet = config.get('testnet', False)

        self.client = None
        self.Client = Client

    def connect(self) -> bool:
        """Connect to Binance API."""
        try:
            if self.testnet:
                # Testnet URLs
                self.client = self.Client(
                    self.api_key,
                    self.api_secret,
                    testnet=True
                )
            else:
                self.client = self.Client(self.api_key, self.api_secret)

            # Test connection
            account = self.client.get_account()
            self.is_connected = True
            print(f"Connected to Binance {'Testnet' if self.testnet else 'Live'}")
            return True

        except Exception as e:
            print(f"Failed to connect to Binance: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from Binance."""
        self.client = None
        self.is_connected = False
        return True

    def get_account(self) -> Account:
        """Get account information."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        account = self.client.get_account()

        # Calculate total balance in USDT equivalent
        total_balance = 0.0
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free + locked > 0:
                # Simplified - in production you'd convert to USDT
                total_balance += free + locked

        return Account(
            balance=total_balance,
            equity=total_balance,
            margin_used=0.0,
            margin_available=total_balance,
            buying_power=total_balance,
            positions=self.get_positions()
        )

    def get_positions(self) -> List[Position]:
        """Get all positions (non-zero balances)."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        account = self.client.get_account()
        positions = []

        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked

            if total > 0 and balance['asset'] != 'USDT':
                # Get current price
                try:
                    ticker = self.client.get_symbol_ticker(
                        symbol=f"{balance['asset']}USDT"
                    )
                    current_price = float(ticker['price'])

                    positions.append(Position(
                        symbol=balance['asset'],
                        quantity=total,
                        entry_price=0.0,  # Not tracked in spot
                        current_price=current_price,
                        unrealized_pnl=0.0,  # Not calculated for spot
                        side="long"
                    ))
                except:
                    pass  # Skip if can't get price

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def place_order(self, order: Order) -> str:
        """Place an order on Binance."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT

        # Map order parameters
        side = SIDE_BUY if order.side == OrderSide.BUY else SIDE_SELL

        try:
            if order.order_type == OrderType.MARKET:
                result = self.client.order_market(
                    symbol=order.symbol,
                    side=side,
                    quantity=order.quantity
                )
            elif order.order_type == OrderType.LIMIT:
                from binance.enums import TIME_IN_FORCE_GTC
                result = self.client.order_limit(
                    symbol=order.symbol,
                    side=side,
                    quantity=order.quantity,
                    price=str(order.price),
                    timeInForce=TIME_IN_FORCE_GTC
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")

            return str(result['orderId'])

        except Exception as e:
            print(f"Failed to place order: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.

        Note: Binance requires symbol to cancel order.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol

        Returns:
            True if cancelled successfully
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            return True
        except Exception as e:
            print(f"Failed to cancel order: {e}")
            return False

    def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """
        Get order status.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        order = self.client.get_order(symbol=symbol, orderId=order_id)

        status_map = {
            'NEW': OrderStatus.PENDING,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.CANCELLED,
        }

        return status_map.get(order['status'], OrderStatus.PENDING)

    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        if symbol is None:
            # Binance requires symbol for order history
            return []

        orders = self.client.get_all_orders(symbol=symbol, limit=500)

        return [
            Order(
                symbol=o['symbol'],
                side=OrderSide.BUY if o['side'] == 'BUY' else OrderSide.SELL,
                order_type=OrderType.MARKET if o['type'] == 'MARKET' else OrderType.LIMIT,
                quantity=float(o['origQty']),
                price=float(o['price']) if o['price'] != '0.00000000' else None,
                order_id=str(o['orderId']),
                status=self._map_status(o['status']),
                filled_quantity=float(o['executedQty']),
                timestamp=datetime.fromtimestamp(o['time'] / 1000)
            )
            for o in orders
        ]

    def close_position(self, symbol: str) -> bool:
        """
        Close position (sell all of the asset).

        Args:
            symbol: Asset symbol (e.g., 'BTC')
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        try:
            # Get current balance
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == symbol:
                    quantity = float(balance['free'])
                    if quantity > 0:
                        # Sell to USDT
                        self.client.order_market_sell(
                            symbol=f"{symbol}USDT",
                            quantity=quantity
                        )
                        return True
            return False
        except Exception as e:
            print(f"Failed to close position: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")

        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] != 'USDT':
                    quantity = float(balance['free'])
                    if quantity > 0:
                        try:
                            self.client.order_market_sell(
                                symbol=f"{balance['asset']}USDT",
                                quantity=quantity
                            )
                        except:
                            pass  # Skip if can't sell
            return True
        except Exception as e:
            print(f"Failed to close all positions: {e}")
            return False

    def _map_status(self, binance_status: str) -> OrderStatus:
        """Map Binance status to OrderStatus enum."""
        status_map = {
            'NEW': OrderStatus.PENDING,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.CANCELLED,
        }
        return status_map.get(binance_status, OrderStatus.PENDING)
