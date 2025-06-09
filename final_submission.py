
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

import json
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    # def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
    #     compressed = {}
    #     for symbol, order_depth in order_depths.items():
    #         compressed[symbol] = [
    #             list(order_depth.buy_orders.items()),  # Convert dict to list of tuples
    #             list(order_depth.sell_orders.items())
    #         ]
    #     return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        if max_length < 3:  # ✅ Ensure valid slicing
            return "..."
        return value[: max_length - 3] + "..."




logger = Logger()

strike_price = 10000

class Product:
    AMETHYSTS = "RAINFOREST_RESIN"
    STARFRUIT = "KELP"  
    ORCHIDS = "ORCHIDS"
    GIFT_BASKET = "PICNIC_BASKET1"
    CHOCOLATE = "JAMS"
    STRAWBERRIES = "CROISSANTS"
    ROSES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"
    PICNIC_BASKET1 = "PICNIC_BASKET2"
    SQUID_INK = "SQUID_INK"
    COCONUT = "VOLCANIC_ROCK"
    COCONUT_COUPON = f"VOLCANIC_ROCK_VOUCHER_{strike_price}"
    COCONUT_COUPON1 = f"VOLCANIC_ROCK_VOUCHER_9750"
    coupons = ["VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_9750"]


PARAMS = {
    Product.AMETHYSTS: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.STARFRUIT: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "starfruit_min_edge": 2,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "SQUID_INK_min_edge": 2,
    },
    Product.ORCHIDS: {
        "make_edge": 2,
        "make_probability": 0.800,
    },
    Product.SPREAD: {
        "default_spread_mean": 70.0449,
        "default_spread_std": 78.6631717132611,
        "spread_std_window": 50,
        "zscore_threshold":2,
        "target_position": 58,
    },
    Product.SPREAD2: {
        "default_spread_mean": 70.0449,
        "default_spread_std": 78.6631717132611,
        "spread_std_window": 50,
        "zscore_threshold":2,
        "target_position": 58,
    },
    Product.COCONUT_COUPON: {
        "mean_volatility":0.15087620376961405,
        "threshold": 0.00163,
        "strike": strike_price,
        "starting_time_to_expiry": 3/250,
        "std_window": 10,   
        "zscore_threshold": 0.05,
    },
    Product.COCONUT_COUPON1: {
        "mean_volatility":0.15604091359676545,
        "threshold": 0.00163,
        "strike": 9750,
        "starting_time_to_expiry": 3/250,
        "std_window": 10,   
        "zscore_threshold": 0.8,
    },
}

BASKET_WEIGHTS = {
    Product.CHOCOLATE: 3,
    Product.STRAWBERRIES: 6,
    Product.ROSES: 1,
}

BASKET_WEIGHTS2 = {
    Product.CHOCOLATE: 2,
    Product.STRAWBERRIES: 4,
    Product.ROSES: 0,
}


from math import log, sqrt, exp
from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 3
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.AMETHYSTS: 50,
            Product.STARFRUIT: 50,
            Product.SQUID_INK: 50,
            Product.ORCHIDS: 50,
            Product.GIFT_BASKET: 60,
            Product.CHOCOLATE: 350,
            Product.STRAWBERRIES: 250,
            Product.ROSES: 60,
            Product.COCONUT: 400,
            Product.COCONUT_COUPON: 200,
            Product.COCONUT_COUPON1:200,
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume
    def starfruit_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("starfruit_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["starfruit_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("starfruit_last_price", None) != None:
                last_price = traderObject["starfruit_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.STARFRUIT]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["starfruit_last_price"] = mmmid_price
            return fair
        return None
    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None

    def make_amethyst_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ], default=fair_value + 2
        )
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1], default = fair_value - 2
        )

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.AMETHYSTS,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    def make_starfruit_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.STARFRUIT,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def make_SQUID_INK_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.SQUID_INK,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def orchids_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return (
            observation.bidPrice
            - observation.exportTariff
            - observation.transportFees
            - 0.1,
            observation.askPrice + observation.importTariff + observation.transportFees,
        )

    def orchids_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = round(observation.askPrice) - 2

        if ask > implied_ask:
            edge = (ask - implied_ask) * self.params[Product.ORCHIDS]["make_probability"]
        else:
            edge = 0

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(
                    abs(order_depth.sell_orders[price]), buy_quantity
                )  # max amount to buy
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(
                    abs(order_depth.buy_orders[price]), sell_quantity
                )  # max amount to sell
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def orchids_arb_clear(self, position: int) -> int:
        conversions = -position
        return conversions

    def orchids_arb_make(
        self,
        observation: ConversionObservation,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]

        # Implied Bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        # Implied Ask = observation.askPrice + observation.importTariff + observation.transportFees
        implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)

        aggressive_ask = round(observation.askPrice) - 2
        aggressive_bid = round(observation.bidPrice) + 2

        if aggressive_bid < implied_bid:
            bid = aggressive_bid
        else:
            bid = implied_bid - 1

        if aggressive_ask >= implied_ask + 0.5:
            ask = aggressive_ask
        elif aggressive_ask + 1 >= implied_ask + 0.5:
            ask = aggressive_ask + 1
        else:
            ask = implied_ask + 2

        logger.print(f"ALGO_ASK: {round(ask)}")
        logger.print(f"IMPLIED_BID: {implied_bid}")
        logger.print(f"IMPLIED_ASK: {implied_ask}")
        logger.print(f"FOREIGN_ASK: {observation.askPrice}")
        logger.print(f"FOREIGN_BID: {observation.bidPrice}")

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(bid), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(ask), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CHOCOLATE_PER_BASKET = BASKET_WEIGHTS[Product.CHOCOLATE]
        STRAWBERRIES_PER_BASKET = BASKET_WEIGHTS[Product.STRAWBERRIES]
        ROSES_PER_BASKET = BASKET_WEIGHTS[Product.ROSES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        chocolate_best_bid = (
            max(order_depths[Product.CHOCOLATE].buy_orders.keys())
            if order_depths[Product.CHOCOLATE].buy_orders
            else 0
        )
        chocolate_best_ask = (
            min(order_depths[Product.CHOCOLATE].sell_orders.keys())
            if order_depths[Product.CHOCOLATE].sell_orders
            else float("inf")
        )
        strawberries_best_bid = (
            max(order_depths[Product.STRAWBERRIES].buy_orders.keys())
            if order_depths[Product.STRAWBERRIES].buy_orders
            else 0
        )
        strawberries_best_ask = (
            min(order_depths[Product.STRAWBERRIES].sell_orders.keys())
            if order_depths[Product.STRAWBERRIES].sell_orders
            else float("inf")
        )
        roses_best_bid = (
            max(order_depths[Product.ROSES].buy_orders.keys())
            if order_depths[Product.ROSES].buy_orders
            else 0
        )
        roses_best_ask = (
            min(order_depths[Product.ROSES].sell_orders.keys())
            if order_depths[Product.ROSES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            chocolate_best_bid * CHOCOLATE_PER_BASKET
            + strawberries_best_bid * STRAWBERRIES_PER_BASKET
            + roses_best_bid * ROSES_PER_BASKET
        )
        implied_ask = (
            chocolate_best_ask * CHOCOLATE_PER_BASKET
            + strawberries_best_ask * STRAWBERRIES_PER_BASKET
            + roses_best_ask * ROSES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            chocolate_bid_volume = (
                order_depths[Product.CHOCOLATE].buy_orders[chocolate_best_bid]
                // CHOCOLATE_PER_BASKET
            )
            strawberries_bid_volume = (
                order_depths[Product.STRAWBERRIES].buy_orders[strawberries_best_bid]
                // STRAWBERRIES_PER_BASKET
            )
            roses_bid_volume = (
                order_depths[Product.ROSES].buy_orders[roses_best_bid]
                // ROSES_PER_BASKET
            )
            implied_bid_volume = min(
                chocolate_bid_volume, strawberries_bid_volume, roses_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            chocolate_ask_volume = (
                -order_depths[Product.CHOCOLATE].sell_orders[chocolate_best_ask]
                // CHOCOLATE_PER_BASKET
            )
            strawberries_ask_volume = (
                -order_depths[Product.STRAWBERRIES].sell_orders[strawberries_best_ask]
                // STRAWBERRIES_PER_BASKET
            )
            roses_ask_volume = (
                -order_depths[Product.ROSES].sell_orders[roses_best_ask]
                // ROSES_PER_BASKET
            )
            implied_ask_volume = min(
                chocolate_ask_volume, strawberries_ask_volume, roses_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CHOCOLATE: [],
            Product.STRAWBERRIES: [],
            Product.ROSES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                chocolate_price = min(
                    order_depths[Product.CHOCOLATE].sell_orders.keys()
                )
                strawberries_price = min(
                    order_depths[Product.STRAWBERRIES].sell_orders.keys()
                )
                roses_price = min(order_depths[Product.ROSES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                chocolate_price = max(order_depths[Product.CHOCOLATE].buy_orders.keys())
                strawberries_price = max(
                    order_depths[Product.STRAWBERRIES].buy_orders.keys()
                )
                roses_price = max(order_depths[Product.ROSES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            chocolate_order = Order(
                Product.CHOCOLATE,
                chocolate_price,
                quantity * BASKET_WEIGHTS[Product.CHOCOLATE],
            )
            strawberries_order = Order(
                Product.STRAWBERRIES,
                strawberries_price,
                quantity * BASKET_WEIGHTS[Product.STRAWBERRIES],
            )
            roses_order = Order(
                Product.ROSES, roses_price, quantity * BASKET_WEIGHTS[Product.ROSES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CHOCOLATE].append(chocolate_order)
            component_orders[Product.STRAWBERRIES].append(strawberries_order)
            component_orders[Product.ROSES].append(roses_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.GIFT_BASKET not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None

    def get_coconut_coupon_mid_price(
        self, coconut_coupon_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(coconut_coupon_order_depth.buy_orders) > 0
            and len(coconut_coupon_order_depth.sell_orders) > 0
        ):
            best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
            best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]

    def delta_hedge_coconut_position(
        self,
        coconut_order_depth: OrderDepth,
        coconut_coupon_position: int,
        coconut_position: int,
        coconut_buy_orders: int,
        coconut_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the overall position in COCONUT_COUPON by creating orders in COCONUT.

        Args:
            coconut_order_depth (OrderDepth): The order depth for the COCONUT product.
            coconut_coupon_position (int): The current position in COCONUT_COUPON.
            coconut_position (int): The current position in COCONUT.
            coconut_buy_orders (int): The total quantity of buy orders for COCONUT in the current iteration.
            coconut_sell_orders (int): The total quantity of sell orders for COCONUT in the current iteration.
            delta (float): The current value of delta for the COCONUT_COUPON product.
            traderData (Dict[str, Any]): The trader data for the COCONUT_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the COCONUT_COUPON position.
        """

        target_coconut_position = -int(delta * coconut_coupon_position)
        hedge_quantity = target_coconut_position - (
            coconut_position + coconut_buy_orders - coconut_sell_orders
        )

        orders: List[Order] = []
        if hedge_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(hedge_quantity), -coconut_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.COCONUT] - (coconut_position + coconut_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_ask, quantity))
        elif hedge_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(hedge_quantity), coconut_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.COCONUT] + (coconut_position - coconut_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_bid, -quantity))

        return orders

    def delta_hedge_coconut_coupon_orders(
        self,
        coconut_order_depth: OrderDepth,
        coconut_coupon_orders: List[Order],
        coconut_position: int,
        coconut_buy_orders: int,
        coconut_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the new orders for COCONUT_COUPON by creating orders in COCONUT.

        Args:
            coconut_order_depth (OrderDepth): The order depth for the COCONUT product.
            coconut_coupon_orders (List[Order]): The new orders for COCONUT_COUPON.
            coconut_position (int): The current position in COCONUT.
            coconut_buy_orders (int): The total quantity of buy orders for COCONUT in the current iteration.
            coconut_sell_orders (int): The total quantity of sell orders for COCONUT in the current iteration.
            delta (float): The current value of delta for the COCONUT_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the new COCONUT_COUPON orders.
        """
        if len(coconut_coupon_orders) == 0:
            return None

        net_coconut_coupon_quantity = sum(
            order.quantity for order in coconut_coupon_orders
        )
        target_coconut_quantity = -int(delta * net_coconut_coupon_quantity)

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity), -coconut_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.COCONUT] - (coconut_position + coconut_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_ask, quantity))
        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity), coconut_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.COCONUT] + (coconut_position - coconut_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_bid, -quantity))

        return orders

    def coconut_hedge_orders(
        self,
        coconut_order_depth: OrderDepth,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_orders: List[Order],
        coconut_position: int,
        coconut_coupon_position: int,
        delta: float,
    ) -> List[Order]:
        if coconut_coupon_orders == None or len(coconut_coupon_orders) == 0:
            coconut_coupon_position_after_trade = coconut_coupon_position
        else:
            coconut_coupon_position_after_trade = coconut_coupon_position + sum(
                order.quantity for order in coconut_coupon_orders
            )

        target_coconut_position = -delta * coconut_coupon_position_after_trade

        if target_coconut_position == coconut_position:
            return None

        target_coconut_quantity = target_coconut_position - coconut_position

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] - coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_ask, round(quantity)))

        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] + coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_bid, -round(quantity)))

        return orders
    
    def coconut_hedge_orders_all_coupons(
        self,
        coupon_order_depths : dict[Symbol, OrderDepth],
        coconut_order_depth: OrderDepth,

        # coconut_coupon_order_depth: OrderDepth,
        coupon_take_orders : Dict[Symbol , List[Order]],
        # coconut_coupon_orders: List[Order],
        coconut_position: int,
        coupon_positions : dict[Symbol , int],
        # coconut_coupon_position: int,
        coupon_deltas : dict[Symbol, float]
        # delta: float,
    ) -> List[Order]:
        
        target_coconut_position = 0
        for coupon, coconut_coupon_orders in coupon_take_orders.items():
        # for coupon in Product.coupons:
            # coconut_coupon_orders = coupon_take_orders[coupon]
            if coconut_coupon_orders == None: 
            # or len(coconut_coupon_orders) == 0:
                coconut_coupon_position_after_trade = coupon_positions[coupon]
            else:
                coconut_coupon_position_after_trade = coupon_positions[coupon] + sum(
                    order.quantity for order in coupon_take_orders[coupon]
                )

            target_coconut_position += -coupon_deltas[coupon] * coconut_coupon_position_after_trade

        if target_coconut_position == coconut_position:
            return None

        target_coconut_quantity = target_coconut_position - coconut_position

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] - coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_ask, round(quantity)))

        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] + coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_bid, -round(quantity)))

        return orders

    # def coconut_coupon_orders(
    #     self,
    #     coconut_coupon_order_depth: OrderDepth,
    #     coconut_coupon_position: int,
    #     traderData: Dict[str, Any],
    #     volatility: float,
    # ) -> List[Order]:
    #     traderData["past_coupon_vol"].append(volatility)
    #     if (
    #         len(traderData["past_coupon_vol"])
    #         < self.params[Product.COCONUT_COUPON]["std_window"]
    #     ):
    #         return None, None

    #     if (
    #         len(traderData["past_coupon_vol"])
    #         > self.params[Product.COCONUT_COUPON]["std_window"]
    #     ):
    #         traderData["past_coupon_vol"].pop(0)

    #     vol_z_score = (
    #         volatility - self.params[Product.COCONUT_COUPON]["mean_volatility"]
    #     ) / np.std(traderData["past_coupon_vol"])
    #     # print(f"vol_z_score: {vol_z_score}")
    #     # print(f"zscore_threshold: {self.params[Product.COCONUT_COUPON]['zscore_threshold']}")
    #     if vol_z_score >= self.params[Product.COCONUT_COUPON]["zscore_threshold"]:
    #         if coconut_coupon_position != -self.LIMIT[Product.COCONUT_COUPON]:
    #             target_coconut_coupon_position = -self.LIMIT[Product.COCONUT_COUPON]
    #             if len(coconut_coupon_order_depth.buy_orders) > 0:
    #                 best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
    #                 target_quantity = abs(
    #                     target_coconut_coupon_position - coconut_coupon_position
    #                 )
    #                 quantity = min(
    #                     target_quantity,
    #                     abs(coconut_coupon_order_depth.buy_orders[best_bid]),
    #                 )
    #                 quote_quantity = target_quantity - quantity
    #                 if quote_quantity == 0:
    #                     return [Order(Product.COCONUT_COUPON, best_bid, -quantity)], []
    #                 else:
    #                     return [Order(Product.COCONUT_COUPON, best_bid, -quantity)], [
    #                         Order(Product.COCONUT_COUPON, best_bid, -quote_quantity)
    #                     ]

    #     elif vol_z_score <= -self.params[Product.COCONUT_COUPON]["zscore_threshold"]:
    #         if coconut_coupon_position != self.LIMIT[Product.COCONUT_COUPON]:
    #             target_coconut_coupon_position = self.LIMIT[Product.COCONUT_COUPON]
    #             if len(coconut_coupon_order_depth.sell_orders) > 0:
    #                 best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
    #                 target_quantity = abs(
    #                     target_coconut_coupon_position - coconut_coupon_position
    #                 )
    #                 quantity = min(
    #                     target_quantity,
    #                     abs(coconut_coupon_order_depth.sell_orders[best_ask]),
    #                 )
    #                 quote_quantity = target_quantity - quantity
    #                 if quote_quantity == 0:
    #                     return [Order(Product.COCONUT_COUPON, best_ask, quantity)], []
    #                 else:
    #                     return [Order(Product.COCONUT_COUPON, best_ask, quantity)], [
    #                         Order(Product.COCONUT_COUPON, best_ask, quote_quantity)   
    #                     ]

    #     return None, None
    
    def coconut_coupon_orders(
        self,
        product,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData["past_coupon_vol"].append(volatility)
        PRODUCT = product
        if (
            len(traderData["past_coupon_vol"])
            < self.params[PRODUCT]["std_window"]
        ):
            return None, None

        if (
            len(traderData["past_coupon_vol"])
            > self.params[PRODUCT]["std_window"]
        ):
            traderData["past_coupon_vol"].pop(0)

        vol_z_score = (
            volatility - self.params[PRODUCT]["mean_volatility"]
        ) / np.std(traderData["past_coupon_vol"])
        # print(f"vol_z_score: {vol_z_score}")
        # print(f"zscore_threshold: {self.params[PRODUCT]['zscore_threshold']}")
        if vol_z_score >= self.params[PRODUCT]["zscore_threshold"]:
            if coconut_coupon_position != -self.LIMIT[PRODUCT]:
                target_coconut_coupon_position = -self.LIMIT[PRODUCT]
                if len(coconut_coupon_order_depth.buy_orders) > 0:
                    best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
                    target_quantity = abs(
                        target_coconut_coupon_position - coconut_coupon_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(PRODUCT, best_bid, -quantity)], []
                    else:
                        return [Order(PRODUCT, best_bid, -quantity)], [
                            Order(PRODUCT, best_bid, -quote_quantity)
                        ]

        elif vol_z_score <= -self.params[PRODUCT]["zscore_threshold"]:
            if coconut_coupon_position != self.LIMIT[PRODUCT]:
                target_coconut_coupon_position = self.LIMIT[PRODUCT]
                if len(coconut_coupon_order_depth.sell_orders) > 0:
                    best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
                    target_quantity = abs(
                        target_coconut_coupon_position - coconut_coupon_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(PRODUCT, best_ask, quantity)], []
                    else:
                        return [Order(PRODUCT, best_ask, quantity)], [
                            Order(PRODUCT, best_ask, quote_quantity)   
                        ]

        return None, None

    def get_past_returns(
        self,
        traderObject: Dict[str, Any],
        order_depths: Dict[str, OrderDepth],
        timeframes: Dict[str, int],
    ):
        returns_dict = {}

        for symbol, timeframe in timeframes.items():
            traderObject_key = f"{symbol}_price_history"
            if traderObject_key not in traderObject:
                traderObject[traderObject_key] = []

            price_history = traderObject[traderObject_key]

            if symbol in order_depths:
                order_depth = order_depths[symbol]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    current_price = (
                        max(order_depth.buy_orders.keys())
                        + min(order_depth.sell_orders.keys())
                    ) / 2
                else:
                    if len(price_history) > 0:
                        current_price = float(price_history[-1])
                    else:
                        returns_dict[symbol] = None
                        continue
            else:
                if len(price_history) > 0:
                    current_price = float(price_history[-1])
                else:
                    returns_dict[symbol] = None
                    continue

            price_history.append(
                f"{current_price:.1f}"
            )  # Convert float to string with 1 decimal place

            if len(price_history) > timeframe:
                price_history.pop(0)

            if len(price_history) == timeframe:
                past_price = float(price_history[0])  # Convert string back to float
                returns = (current_price - past_price) / past_price
                returns_dict[symbol] = returns
            else:
                returns_dict[symbol] = None

        return returns_dict

    def run(self, state: TradingState):

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        trader_data = ""
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        past_returns_timeframes = {"GIFT_BASKET": 500}
        past_returns_dict = self.get_past_returns(
            traderObject, state.order_depths, past_returns_timeframes
        )

        result = {}
        conversions = 0


        if Product.AMETHYSTS in self.params and Product.AMETHYSTS in state.order_depths:
            amethyst_position = (
                state.position[Product.AMETHYSTS]
                if Product.AMETHYSTS in state.position
                else 0
            )
            amethyst_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.AMETHYSTS,
                    state.order_depths[Product.AMETHYSTS],
                    self.params[Product.AMETHYSTS]["fair_value"],
                    self.params[Product.AMETHYSTS]["take_width"],
                    amethyst_position,
                )
            )
            amethyst_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.AMETHYSTS,
                    state.order_depths[Product.AMETHYSTS],
                    self.params[Product.AMETHYSTS]["fair_value"],
                    self.params[Product.AMETHYSTS]["clear_width"],
                    amethyst_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            amethyst_make_orders, _, _ = self.make_amethyst_orders(
                state.order_depths[Product.AMETHYSTS],
                self.params[Product.AMETHYSTS]["fair_value"],
                amethyst_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.AMETHYSTS]["volume_limit"],
            )

            
            result[Product.AMETHYSTS] = (
                amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
            )
            
        if Product.STARFRUIT in self.params and Product.STARFRUIT in state.order_depths:
            starfruit_position = (
                state.position[Product.STARFRUIT]
                if Product.STARFRUIT in state.position
                else 0
            )
            starfruit_fair_value = self.starfruit_fair_value(
                state.order_depths[Product.STARFRUIT], traderObject
            )
            starfruit_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.STARFRUIT,
                    state.order_depths[Product.STARFRUIT],
                    starfruit_fair_value,
                    self.params[Product.STARFRUIT]["take_width"],
                    starfruit_position,
                    self.params[Product.STARFRUIT]["prevent_adverse"],
                    self.params[Product.STARFRUIT]["adverse_volume"],
                )
            )
            starfruit_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.STARFRUIT,
                    state.order_depths[Product.STARFRUIT],
                    starfruit_fair_value,
                    self.params[Product.STARFRUIT]["clear_width"],
                    starfruit_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            starfruit_make_orders, _, _ = self.make_starfruit_orders(
                state.order_depths[Product.STARFRUIT],
                starfruit_fair_value,
                self.params[Product.STARFRUIT]["starfruit_min_edge"],
                starfruit_position,
                buy_order_volume,
                sell_order_volume,
            )
            
            result[Product.STARFRUIT] = (
                starfruit_take_orders + starfruit_clear_orders + starfruit_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            SQUID_INK_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            SQUID_INK_fair_value = self.SQUID_INK_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    SQUID_INK_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    SQUID_INK_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            SQUID_INK_make_orders, _, _ = self.make_SQUID_INK_orders(
                state.order_depths[Product.SQUID_INK],
                SQUID_INK_fair_value,
                self.params[Product.SQUID_INK]["SQUID_INK_min_edge"],
                SQUID_INK_position,
                buy_order_volume,
                sell_order_volume,
            )
            
            result[Product.SQUID_INK] = (
                SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders
            )


        if Product.ORCHIDS in self.params and Product.ORCHIDS in state.order_depths:
            orchids_position = (
                state.position[Product.ORCHIDS]
                if Product.ORCHIDS in state.position
                else 0
            )
            print(f"ORCHIDS POSITION: {orchids_position}")

            conversions = self.orchids_arb_clear(orchids_position)

            orchids_position = 0

            orchids_take_orders, buy_order_volume, sell_order_volume = (
                self.orchids_arb_take(
                    state.order_depths[Product.ORCHIDS],
                    state.observations.conversionObservations[Product.ORCHIDS],
                    orchids_position,
                )
            )

            orchids_make_orders, _, _ = self.orchids_arb_make(
                state.observations.conversionObservations[Product.ORCHIDS],
                orchids_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.ORCHIDS] = orchids_take_orders + orchids_make_orders
            
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.GIFT_BASKET]
            if Product.GIFT_BASKET in state.position
            else 0
        )
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.GIFT_BASKET,
            basket_position,
            traderObject[Product.SPREAD],
        )
        if spread_orders != None:
            result[Product.CHOCOLATE] = spread_orders[Product.CHOCOLATE]
            result[Product.STRAWBERRIES] = spread_orders[Product.STRAWBERRIES]
            result[Product.ROSES] = spread_orders[Product.ROSES]
            result[Product.GIFT_BASKET] = spread_orders[Product.GIFT_BASKET]
            pass

        coupon_order_depths : dict[Product , OrderDepth] = {}
        coupon_take_orders : dict[Product , List[Order]] = {}
        coupon_positions : dict[Product , int] = {}
        coupon_deltas : dict[Product ,float] = {}
        ######################################################################################################################################################
        #####################################################################################################################################################
        # first coupon
        if Product.COCONUT_COUPON not in traderObject:
            traderObject[Product.COCONUT_COUPON] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": [],
            }

        if (
            Product.COCONUT_COUPON in self.params
            and Product.COCONUT_COUPON in state.order_depths
        ):
            coconut_coupon_position = (
                state.position[Product.COCONUT_COUPON]
                if Product.COCONUT_COUPON in state.position
                else 0
            )

            coconut_position = (
                state.position[Product.COCONUT]
                if Product.COCONUT in state.position
                else 0
            )
            # print(f"coconut_coupon_position: {coconut_coupon_position}")
            # print(f"coconut_position: {coconut_position}")
            coconut_order_depth = state.order_depths[Product.COCONUT]
            coconut_coupon_order_depth = state.order_depths[Product.COCONUT_COUPON]
            coconut_mid_price = (
                min(coconut_order_depth.buy_orders.keys())
                + max(coconut_order_depth.sell_orders.keys())
            ) / 2
            coconut_coupon_mid_price = self.get_coconut_coupon_mid_price(
                coconut_coupon_order_depth, traderObject[Product.COCONUT_COUPON]
            )
            tte = (
                self.params[Product.COCONUT_COUPON]["starting_time_to_expiry"]
                - (state.timestamp) / 1000000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                coconut_coupon_mid_price,
                coconut_mid_price,
                self.params[Product.COCONUT_COUPON]["strike"],
                tte,
            )
            if volatility == 0: 
                volatility = 1e-10
                pass
            delta = BlackScholes.delta(
                coconut_mid_price,
                self.params[Product.COCONUT_COUPON]["strike"],
                tte,
                volatility,
            )

            coconut_coupon_take_orders, coconut_coupon_make_orders = (
                self.coconut_coupon_orders(
                    Product.COCONUT_COUPON,
                    state.order_depths[Product.COCONUT_COUPON],
                    coconut_coupon_position,
                    traderObject[Product.COCONUT_COUPON],
                    volatility,
                )
            )
            # coconut_orders = self.coconut_hedge_orders(
            #     state.order_depths[Product.COCONUT],
            #     state.order_depths[Product.COCONUT_COUPON],
            #     coconut_coupon_take_orders,
            #     coconut_position,
            #     coconut_coupon_position,
            #     delta,
            # )

            if coconut_coupon_take_orders != None or coconut_coupon_make_orders != None:
                result[Product.COCONUT_COUPON] = (
                    coconut_coupon_take_orders + coconut_coupon_make_orders
                )
            # if coconut_orders != None:
            #     result[Product.COCONUT] = coconut_orders
            #     # print(f"COCONUT: {result[Product.COCONUT]}")

            coupon_order_depths[Product.COCONUT_COUPON] = state.order_depths[Product.COCONUT_COUPON]
            coupon_take_orders[Product.COCONUT_COUPON] = coconut_coupon_take_orders
            coupon_positions[Product.COCONUT_COUPON] = coconut_coupon_position
            coupon_deltas[Product.COCONUT_COUPON] = delta

            
        ######################################################################################################################################################
        #####################################################################################################################################################
        # second coupon

        if Product.COCONUT_COUPON1 not in traderObject:
            traderObject[Product.COCONUT_COUPON1] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": [],
            }

        

        if (
            Product.COCONUT_COUPON1 in self.params
            and Product.COCONUT_COUPON1 in state.order_depths
        ):
            coconut_coupon1_position = (
                state.position[Product.COCONUT_COUPON1]
                if Product.COCONUT_COUPON1 in state.position
                else 0
            )

            coconut_position = (
                state.position[Product.COCONUT]
                if Product.COCONUT in state.position
                else 0
            )
            # print(f"coconut_coupon1_position: {coconut_coupon1_position}")
            # print(f"coconut_position: {coconut_position}")
            coconut_order_depth = state.order_depths[Product.COCONUT]
            coconut_coupon1_order_depth = state.order_depths[Product.COCONUT_COUPON1]
            coconut_mid_price = (
                min(coconut_order_depth.buy_orders.keys())
                + max(coconut_order_depth.sell_orders.keys())
            ) / 2
            coconut_coupon1_mid_price = self.get_coconut_coupon_mid_price(
                coconut_coupon1_order_depth, traderObject[Product.COCONUT_COUPON1]
            )
            tte = (
                self.params[Product.COCONUT_COUPON1]["starting_time_to_expiry"]
                - (state.timestamp) / 1000000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                coconut_coupon1_mid_price,
                coconut_mid_price,
                self.params[Product.COCONUT_COUPON1]["strike"],
                tte,
            )
            if volatility == 0: 
                volatility = 1e-10
                pass
            delta = BlackScholes.delta(
                coconut_mid_price,
                self.params[Product.COCONUT_COUPON1]["strike"],
                tte,
                volatility,
            )

            coconut_coupon1_take_orders, coconut_coupon1_make_orders = (
                self.coconut_coupon_orders(
                    Product.COCONUT_COUPON1,
                    state.order_depths[Product.COCONUT_COUPON1],
                    coconut_coupon1_position,
                    traderObject[Product.COCONUT_COUPON1],
                    volatility,
                )
            )

            coupon_order_depths[Product.COCONUT_COUPON1] = state.order_depths[Product.COCONUT_COUPON1]
            coupon_take_orders[Product.COCONUT_COUPON1] = coconut_coupon1_take_orders
            coupon_positions[Product.COCONUT_COUPON1] = coconut_coupon1_position
            coupon_deltas[Product.COCONUT_COUPON1] = delta

            

            if coconut_coupon1_take_orders != None or coconut_coupon1_make_orders != None:
                result[Product.COCONUT_COUPON1] = (
                    coconut_coupon1_take_orders + coconut_coupon1_make_orders
                )

        # coconut_orders = self.coconut_hedge_orders_all_coupons(
        #         state.order_depths[Product.COCONUT],
        #         coupon_order_depths,
        #         coconut_position,
        #         coupon_positions,
        #         coupon_deltas,
        #     )
        # if coconut_orders != None:
        #     result[Product.COCONUT] = coconut_orders
        #         # print(f"COCONUT: {result[Product.COCONUT]}")
        
        
        ######################################################################################################################################################
        #####################################################################################################################################################
        # second coupon

        # if Product.COCONUT_COUPON2 not in traderObject:
        #     traderObject[Product.COCONUT_COUPON2] = {
        #         "prev_coupon_price": 0,
        #         "past_coupon_vol": [],
        #     }

        

        # if (
        #     Product.COCONUT_COUPON2 in self.params
        #     and Product.COCONUT_COUPON2 in state.order_depths
        # ):
        #     coconut_coupon2_position = (
        #         state.position[Product.COCONUT_COUPON2]
        #         if Product.COCONUT_COUPON2 in state.position
        #         else 0
        #     )

        #     coconut_position = (
        #         state.position[Product.COCONUT]
        #         if Product.COCONUT in state.position
        #         else 0
        #     )
        #     # print(f"coconut_coupon2_position: {coconut_coupon2_position}")
        #     # print(f"coconut_position: {coconut_position}")
        #     coconut_order_depth = state.order_depths[Product.COCONUT]
        #     coconut_coupon2_order_depth = state.order_depths[Product.COCONUT_COUPON2]
        #     coconut_mid_price = (
        #         min(coconut_order_depth.buy_orders.keys())
        #         + max(coconut_order_depth.sell_orders.keys())
        #     ) / 2
        #     coconut_coupon2_mid_price = self.get_coconut_coupon2_mid_price(
        #         coconut_coupon2_order_depth, traderObject[Product.COCONUT_COUPON2]
        #     )
        #     tte = (
        #         self.params[Product.COCONUT_COUPON2]["starting_time_to_expiry"]
        #         - (state.timestamp) / 1000000 / 250
        #     )
        #     volatility = BlackScholes.implied_volatility(
        #         coconut_coupon2_mid_price,
        #         coconut_mid_price,
        #         self.params[Product.COCONUT_COUPON2]["strike"],
        #         tte,
        #     )
        #     if volatility == 0: 
        #         volatility = 1e-10
        #         pass
        #     delta = BlackScholes.delta(
        #         coconut_mid_price,
        #         self.params[Product.COCONUT_COUPON2]["strike"],
        #         tte,
        #         volatility,
        #     )

        #     coconut_coupon2_take_orders, coconut_coupon2_make_orders = (
        #         self.coconut_coupon2_orders(
        #             state.order_depths[Product.COCONUT_COUPON2],
        #             coconut_coupon2_position,
        #             traderObject[Product.COCONUT_COUPON2],
        #             volatility,
        #         )
        #     )

            # coconut_orders = self.coconut_hedge_orders(
            #     state.order_depths[Product.COCONUT],
            #     state.order_depths[Product.COCONUT_COUPON],
            #     coconut_coupon_take_orders,
            #     coconut_position,
            #     coconut_coupon_position,
            #     delta,
            # )

            # if coconut_coupon_take_orders != None or coconut_coupon_make_orders != None:
            #     result[Product.COCONUT_COUPON] = (
            #         coconut_coupon_take_orders + coconut_coupon_make_orders
            #     )
                # print(f"COCONUT_COUPON: {result[Product.COCONUT_COUPON]}")

            # if coconut_orders != None:
            #     result[Product.COCONUT] = coconut_orders
            #     # print(f"COCONUT: {result[Product.COCONUT]}")

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData
