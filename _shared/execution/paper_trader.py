"""
Paper Trading Executor

Simulated trading for testing strategies without real money.
"""

from typing import Dict, List, Optional
from datetime import datetime
import uuid

from .broker_base import (
    BrokerBase, Order, Position, Account,
    OrderType, OrderSide, OrderStatus
)


class PaperTrader(BrokerBase):
    """
    Paper trading implementation for strategy testing.

    Simulates order execution without connecting to real broker.
    """

    def __init__(self, config: Dict):
        """
        Initialize paper trader.

        Args:
            config: Configuration dict

        Example config:
            {
                'initial_balance': 100000,
                'slippage_bps': 5,  # 0.05% slippage
                'commission_bps': 10  # 0.10% commission
            }
        """
        super().__init__(config)

        self.initial_balance = config.get('initial_balance', 100000.0)
        self.slippage_bps = config.get('slippage_bps', 5)
        self.commission_bps = config.get('commission_bps', 10)

        # Internal state
        self.cash = self.initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_count = 0

        # Price function - to be set externally
        self.price_function = None

    def set_price_function(self, func):
        """
        Set function to get current prices.

        Args:
            func: Function that takes symbol and returns current price
                  e.g., lambda symbol: data[symbol]['close'].iloc[-1]
        """
        self.price_function = func

    def connect(self) -> bool:
        """Connect (always successful for paper trading)."""
        self.is_connected = True
        print(f"Paper trader initialized with ${self.initial_balance:,.2f}")
        return True

    def disconnect(self) -> bool:
        """Disconnect."""
        self.is_connected = False
        return True

    def get_account(self) -> Account:
        """Get account information."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        # Calculate equity
        equity = self.cash
        for position in self.positions.values():
            equity += position.quantity * position.current_price

        return Account(
            balance=self.cash,
            equity=equity,
            margin_used=0.0,
            margin_available=self.cash,
            buying_power=self.cash,
            positions=list(self.positions.values())
        )

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        # Update current prices
        for symbol, position in self.positions.items():
            if self.price_function:
                try:
                    current_price = self.price_function(symbol)
                    position.current_price = current_price
                    position.unrealized_pnl = (
                        (current_price - position.entry_price) * position.quantity
                    )
                except:
                    pass

        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        return self.positions.get(symbol)

    def place_order(self, order: Order) -> str:
        """Place an order (execute immediately for paper trading)."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        # Generate order ID
        order_id = str(uuid.uuid4())
        order.order_id = order_id

        # Get current price
        if self.price_function:
            try:
                current_price = self.price_function(order.symbol)
            except:
                current_price = order.price if order.price else 100.0
        else:
            current_price = order.price if order.price else 100.0

        # Apply slippage
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                execution_price = current_price * (1 + self.slippage_bps / 10000)
            else:
                execution_price = current_price * (1 - self.slippage_bps / 10000)
        else:
            execution_price = order.price

        # Calculate cost
        notional = execution_price * order.quantity
        commission = notional * (self.commission_bps / 10000)

        # Execute order
        if order.side == OrderSide.BUY:
            total_cost = notional + commission

            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                print(f"Order rejected: Insufficient funds. Need ${total_cost:,.2f}, have ${self.cash:,.2f}")
                return order_id

            # Update cash
            self.cash -= total_cost

            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                # Average entry price
                total_qty = pos.quantity + order.quantity
                pos.entry_price = (
                    (pos.entry_price * pos.quantity + execution_price * order.quantity) / total_qty
                )
                pos.quantity = total_qty
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=execution_price,
                    current_price=execution_price,
                    unrealized_pnl=0.0,
                    side="long"
                )

        else:  # SELL
            # Check if we have the position
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                print(f"Order rejected: No position in {order.symbol}")
                return order_id

            pos = self.positions[order.symbol]
            if order.quantity > pos.quantity:
                order.status = OrderStatus.REJECTED
                print(f"Order rejected: Insufficient quantity. Have {pos.quantity}, trying to sell {order.quantity}")
                return order_id

            # Calculate proceeds
            proceeds = notional - commission
            self.cash += proceeds

            # Calculate realized PnL
            realized_pnl = (execution_price - pos.entry_price) * order.quantity - commission
            pos.realized_pnl += realized_pnl

            # Update position
            pos.quantity -= order.quantity
            if pos.quantity == 0:
                del self.positions[order.symbol]

        # Mark order as filled
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = execution_price

        # Store order
        self.orders[order_id] = order
        self.order_history.append(order)
        self.trade_count += 1

        print(f"✓ {order.side.value.upper()} {order.quantity} {order.symbol} @ ${execution_price:.2f}")

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (no-op for paper trading with instant execution)."""
        return False  # Orders execute immediately

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.PENDING

    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history."""
        if symbol:
            return [o for o in self.order_history if o.symbol == symbol]
        return self.order_history

    def close_position(self, symbol: str) -> bool:
        """Close position for a symbol."""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=pos.quantity
        )

        self.place_order(order)
        return True

    def close_all_positions(self) -> bool:
        """Close all open positions."""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            self.close_position(symbol)
        return True

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary.

        Returns:
            Dictionary with performance metrics
        """
        account = self.get_account()

        total_return = (account.equity - self.initial_balance) / self.initial_balance * 100
        total_pnl = account.equity - self.initial_balance

        wins = [o for o in self.order_history if o.side == OrderSide.SELL and
                getattr(self.positions.get(o.symbol), 'realized_pnl', 0) > 0]

        return {
            'initial_balance': self.initial_balance,
            'current_equity': account.equity,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': self.trade_count,
            'open_positions': len(self.positions),
            'win_rate': len(wins) / max(self.trade_count / 2, 1) if self.trade_count > 0 else 0
        }
