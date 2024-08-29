import yfinance as yf
import pandas as pd
import ta


class StockData:
    def __init__(self, ticker_symbols, start_date="2001-01-01", end_date="2024-07-01", period="1d"):
        self.ticker_symbols = ticker_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.datasets = {ticker_symbol: self.__get_data(ticker_symbol) for ticker_symbol in ticker_symbols}

    def __get_data(self, ticker_symbol) -> pd.DataFrame:
        # Download the data
        ticket = yf.Ticker(ticker_symbol)
        historical_data = ticket.history(start=self.start_date, end=self.end_date, period=self.period)[
            ["Open", "High", "Low", "Close", "Volume"]]

        # Calculate indicators
        historical_data = self.__calculate_technical_indicators(historical_data)

        # Set only date as index
        historical_data.index = pd.to_datetime(historical_data.index).strftime("%Y-%m-%d")

        return historical_data

    def __calculate_technical_indicators(self, df) -> pd.DataFrame:
        """
        Calculates popular technical indicators using OHLCV data.

        Parameters:
        - df (DataFrame): DataFrame containing OHLCV data (columns: ['Open', 'High', 'Low', 'Close', 'Volume']).

        Returns:
        - DataFrame: DataFrame with added columns for each calculated technical indicator.
        """
        # Simple Moving Average (SMA)
        # df['SMA_20'] = ta.trend.sma_indicator(close=df['Close'], window=5)

        # Exponential Moving Average (EMA)
        # df['EMA_50'] = ta.trend.ema_indicator(close=df['Close'], window=10)

        # Relative Strength Index (RSI)
        # df['RSI_14'] = ta.momentum.rsi(close=df['Close'], window=10)

        # Calculate the 10-period EMA
        df['EMA5_Close'] = df['Close'].ewm(span=5, adjust=False).mean()

        # Calculate the 26-period EMA
        # df['EMA26'] = df['Close'].ewm(span=10, adjust=False).mean()

        # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
        # df['MACD'] = df['EMA12'] - df['EMA26']

        # Bollinger Bands
        #df['BB_Lower'], df['BB_Middle'], df['BB_Upper'] = ta.volatility.bollinger_lband(close=df['Close'],
        #                                                                                window=10), ta.volatility.bollinger_mavg(
        #    close=df['Close'], window=10), ta.volatility.bollinger_hband(close=df['Close'], window=10)

        # Average True Range (ATR)
        # df['ATR_14'] = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=10)

        # Average Directional Index (ADX)
        # df['ADX_14'] = ta.trend.adx(high=df['High'], low=df['Low'], close=df['Close'], window=6)

        return df
