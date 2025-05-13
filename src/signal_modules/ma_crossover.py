import pandas as pd
from .base import SignalModule
import matplotlib.pyplot as plt


class MovingAverageCrossover(SignalModule):
    def __init__(
            self, 
            short_window=13,
            mid_window=34,
            long_window=89
        ):
        super().__init__(name="MovingAverageCrossover")
        self.short_window = short_window
        self.mid_window = mid_window
        self.long_window = long_window

    def generate_signals(
            self, 
            data: dict[str, pd.DataFrame], 
            price: str = 'adjClose', 
            diagnostics: bool = False
        ) -> dict[str, pd.Series]:
        """
        For each ticker, compute buy confidence as a function of the difference between
        short and long simple moving averages. Output is a time-indexed Series with
        confidence values in [0.0, 1.0].

        Returns:
            dict[ticker] = pd.Series with index = timestamp, values = buy confidence
        """
        signals: dict[str, pd.Series] = {}

        for ticker, df in data.items():
            df = df.copy()

            df["SMA_short"] = df[price].rolling(window=self.short_window).mean()
            cummean = df[price].expanding(min_periods=1).mean()
            df["SMA_short"] = df["SMA_short"].combine_first(cummean)

            df["SMA_mid"] = df[price].rolling(window=self.mid_window).mean()
            df["SMA_mid"] = df["SMA_mid"].combine_first(cummean)

            df["SMA_long"] = df[price].rolling(window=self.long_window).mean()
            df["SMA_long"] = df["SMA_long"].combine_first(cummean)

            df.dropna(subset=["SMA_short", "SMA_long"], inplace=True)            

            if df.empty:
                signals[ticker] = pd.Series(dtype=float)
                continue

            score = ((df["SMA_short"] - df["SMA_long"]) / df["SMA_mid"]).clip(-1,1)
            score = 0.5 + 0.5 * score

            if diagnostics:
                price_min = df[price].min()
                price_max = df[price].max()

                scaled_score = price_min + (price_max - price_min) * score

                plt.figure(figsize=(12, 6))
                
                # Plot price and MAs
                plt.plot(df.index, df[price], label=price, color="black", alpha=0.5, linewidth=2)
                plt.plot(df.index, df["SMA_short"], label=f"SMA {self.short_window}", color="red")
                plt.plot(df.index, df["SMA_long"], label=f"SMA {self.long_window}", color="green")
                
                # Plot scaled buy confidence
                plt.plot(df.index, scaled_score, label="Score (scaled)", color="orange", linestyle="-", linewidth=1)

                # Draw horizontal lines at price min and max
                plt.axhline(price_min, color="gray", linestyle="--", linewidth=1)
                plt.axhline(price_max, color="gray", linestyle="--", linewidth=1)
                plt.axhline(price_min + (price_max - price_min) * 0.5, color="gray", linestyle="--", linewidth=1)

                # Annotate 0 and 1 labels for confidence
                plt.text(df.index[0], price_min, "-1", va="top", ha="right", fontsize=10, color="gray")
                plt.text(df.index[0], price_max, "1", va="top", ha="right", fontsize=10, color="gray")
                plt.text(df.index[0], price_min + (price_max - price_min) * 0.5, "0", va="top", ha="right", fontsize=10, color="gray")

                plt.title(f"{ticker} - Moving Average Crossover with Buy Confidence")
                plt.xlabel("Date")
                plt.ylabel("Price / Scaled Confidence")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            signals[ticker] = score

        return signals
