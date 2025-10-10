import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import csv
import yaml
import logging
from datetime import datetime, timedelta, time as dt_time
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import requests
import ta
import pytz
import schedule
import asyncio
import aiohttp
import threading
from itertools import product
try:
    import talib
except ImportError:
    talib = None
    logging.warning("TA-Lib is not installed. Technical indicators will not work.")
from openai import OpenAI
import tweepy

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('master_trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration and API Initialization
# =============================================================================

def load_configuration(config_file: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        raise

def initialize_alpaca_api(api_key: str, api_secret: str, base_url: str) -> tradeapi.REST:
    """Initialize the Alpaca API instance."""
    try:
        api = tradeapi.REST(api_key, api_secret, base_url)
        logger.info("Alpaca API initialized successfully.")
        return api
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {e}", exc_info=True)
        raise

# =============================================================================
# Data Fetching with Polygon.io and Alpaca
# =============================================================================

def download_historical_data(tickers: List[str], polygon_api_key: str, start_date: str = "2023-02-20", end_date: str = None):
    """Download historical market data from Polygon.io with rate limiting."""
    try:
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        headers = {"Authorization": f"Bearer {polygon_api_key}"}
        request_interval = 12  # 12 seconds between requests (5 calls/min)

        for ticker in tickers:
            logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}...")
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('results', [])
                if data:
                    df = pd.DataFrame([{
                        'time': datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d'),
                        'open': bar['o'],
                        'high': bar['h'],
                        'low': bar['l'],
                        'close': bar['c'],
                        'volume': bar['v']
                    } for bar in data])
                    df.to_csv(f"./data/{ticker}.csv", index=False)
                    logger.info(f"Data for {ticker} downloaded and saved with {len(df)} rows.")
                else:
                    logger.warning(f"No data returned for {ticker}.")
            elif response.status_code == 429:
                logger.error(f"Rate limit exceeded for {ticker}. Waiting...")
                time.sleep(request_interval)
            elif response.status_code == 401:
                logger.error(f"Unauthorized access for {ticker}. Polygon API key: {polygon_api_key}")
            else:
                logger.error(f"Failed to download data for {ticker}. Status: {response.status_code}, Message: {response.text}")
            time.sleep(request_interval)
    except Exception as e:
        logger.error(f"Error downloading data for ticker {ticker}: {e}", exc_info=True)

def fetch_current_market_data(tickers: List[str], polygon_api_key: str, days: int = 200) -> pd.DataFrame:
    """Fetch recent market data from Polygon.io with enough history for features."""
    try:
        logger.info("Fetching current market data with Polygon.io...")
        headers = {"Authorization": f"Bearer {polygon_api_key}"}
        data_list = []
        request_interval = 12  # 12 seconds between requests (5 calls/min)

        end_time = datetime.now().strftime("%Y-%m-%d")
        start_time = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        for ticker in tickers:
            try:
                logger.debug(f"Fetching data for {ticker} from {start_time} to {end_time}...")
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_time}/{end_time}"
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    if results:
                        ticker_data = [{
                            'ticker': ticker,
                            'date': datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d'),
                            'open': bar['o'],
                            'high': bar['h'],
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': bar['v']
                        } for bar in results]
                        logger.debug(f"Fetched {len(ticker_data)} rows for {ticker}")
                        data_list.extend(ticker_data)
                    else:
                        logger.warning(f"No recent data for {ticker}.")
                elif response.status_code == 401:
                    logger.error(f"Unauthorized access for {ticker}. Polygon API key: {polygon_api_key}")
                    return pd.DataFrame()
                elif response.status_code == 429:
                    logger.warning(f"Rate limit exceeded for {ticker}. Waiting...")
                    time.sleep(request_interval)
                else:
                    logger.warning(f"Failed to fetch data for {ticker}. Status: {response.status_code}, Message: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching data for {ticker}: {e}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
            time.sleep(request_interval)

        if not data_list:
            logger.warning("No valid market data retrieved.")
            return pd.DataFrame()

        df = pd.DataFrame(data_list)
        logger.info(f"Market data fetched successfully. Total rows: {len(df)}")
        for ticker in tickers:
            rows = len(df[df['ticker'] == ticker])
            logger.info(f"Rows for {ticker}: {rows}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_current_market_data: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_latest_data(tickers: List[str], api: tradeapi.REST) -> pd.DataFrame:
    """Fetch latest daily bar data (partial if during market) from Alpaca API for real-time trading."""
    try:
        logger.info("Fetching latest daily bar data from Alpaca...")
        data_list = []
        for ticker in tickers:
            try:
                bars = api.get_bars(ticker, tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Day), limit=1)
                if bars:
                    bar = bars[0]
                    data = {
                        'ticker': ticker,
                        'date': datetime.fromtimestamp(bar.t.timestamp()).strftime('%Y-%m-%d'),
                        'open': bar.o,
                        'high': bar.h,
                        'low': bar.l,
                        'close': bar.c,
                        'volume': bar.v
                    }
                    data_list.append(data)
                else:
                    logger.warning(f"No latest daily bar data for {ticker}.")
            except Exception as e:
                logger.error(f"Error fetching latest data for {ticker}: {e}")
        if not data_list:
            logger.warning("No valid latest data retrieved.")
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        logger.info(f"Latest data fetched successfully. Total rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_latest_data: {e}", exc_info=True)
        return pd.DataFrame()

def load_historical_data(directory: str, config: dict) -> pd.DataFrame:
    """Load and combine historical data from CSV files or download if missing."""
    tickers = config['tickers']
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and os.path.splitext(f)[0] in tickers]
    logger.info(f"Files found in {directory} for tickers {tickers}: {all_files}")

    if not all_files:
        logger.warning("No CSV files found. Downloading historical data with Polygon.io...")
        polygon_api_key = config['polygon']['api_key']
        download_historical_data(tickers, polygon_api_key)
        all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and os.path.splitext(f)[0] in tickers]

    combined_data = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            if 'time' in df.columns:
                df['date'] = df['time']  # Already str in '%Y-%m-%d'
                df['ticker'] = os.path.splitext(file)[0]
                combined_data.append(df)
            else:
                logger.warning(f"File {file} does not contain a 'time' column. Skipping.")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}", exc_info=True)

    if combined_data:
        df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Loaded historical data with {len(df)} rows")
        return df
    else:
        logger.error("No valid historical data files found in the directory.")
        return pd.DataFrame()

async def fetch_news(config):
    if not config.get('newsapi_token'):
        logger.error("Missing NewsAPI token")
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": " OR ".join(config['tickers']), "apiKey": config['newsapi_token'], "language": "en", "pageSize": 50}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                data = await resp.json()
                articles = data.get('articles', [])
                for article in articles:
                    title = article.get('title', '') or ''
                    description = article.get('description', '') or ''
                    text = (title + description).lower()
                    for t in config['tickers']:
                        if t.lower() in text:
                            article['ticker'] = t
                            break
                return [a for a in articles if 'ticker' in a]
    except Exception as e:
        logger.error(f"News fetch error: {e}")
        return []

def fetch_twitter_sync(config):
    logger.warning("Skipping Twitter sentiment: free API tier doesn’t support search.")
    return []

def analyze_sentiment(items, config, classification=True):
    if not config.get('grok_api_key'):
        logger.warning("Skipping sentiment analysis: GROK_API_KEY missing.")
        return []
    client = OpenAI(api_key=config['grok_api_key'], base_url='https://api.x.ai/v1')
    out = []
    for it in items:
        text = it.get('text') or it.get('title') or it.get('description', '')
        if not text:
            continue
        sys_prompt = "Respond with one word: Positive, Neutral, or Negative based on the sentiment." if classification else "Explain bullish/bearish reasoning."
        try:
            resp = client.chat.completions.create(
                model='grok-4',
                messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': text}],
                temperature=0,
                max_tokens=50
            )
            if resp.choices and len(resp.choices) > 0 and resp.choices[0].message and resp.choices[0].message.content:
                msg = resp.choices[0].message.content.strip()
                msg_lower = msg.lower()
                if 'positive' in msg_lower:
                    label = 'Positive'
                elif 'negative' in msg_lower:
                    label = 'Negative'
                else:
                    label = 'Neutral'
                out.append({'ticker': it['ticker'], 'label': label, 'detail': msg})
            else:
                logger.warning(f"Invalid Grok response for text: {text[:50]}... Skipping.")
        except Exception as e:
            logger.error(f"LLM error for text {text[:50]}...: {e}")
    logger.info(f"Processed {len(out)} sentiment items successfully.")
    return out

# =============================================================================
# Utility Functions
# =============================================================================

def send_notification(title, message):
    script = f'display notification "{message}" with title "{title}"'
    os.system(f"osascript -e '{script}'")

def initialize_trade_log() -> str:
    """Initialize a CSV file to log trades."""
    log_filename = f"trade_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    try:
        with open(log_filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'ticker', 'type', 'price', 'quantity', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(f"Trade log initialized: {log_filename}")
    except Exception as e:
        logger.error(f"Failed to initialize trade log: {e}", exc_info=True)
    return log_filename

def log_trade(trade_data: dict, log_filename: str):
    """Log individual trades into the CSV file."""
    try:
        with open(log_filename, mode='a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                datetime.now(), trade_data['ticker'], trade_data['type'],
                trade_data['price'], trade_data['quantity'], trade_data['status']
            ])
        logger.info(f"Trade logged: {trade_data}")
    except Exception as e:
        logger.error(f"Failed to log trade: {e}", exc_info=True)

def fetch_current_positions(api: tradeapi.REST) -> list:
    """Fetch current positions from Alpaca API."""
    try:
        positions = api.list_positions()
        current_positions = []
        for position in positions:
            ticker = position.symbol
            qty = float(position.qty)
            avg_price = float(position.avg_entry_price)
            try:
                current_price = float(api.get_latest_trade(ticker).price)
            except Exception as e:
                logger.error(f"Error fetching current price for {ticker}: {e}")
                current_price = avg_price  # Fallback to avg_price if API fails
            current_positions.append({
                'ticker': ticker,
                'quantity': qty,
                'average_price': avg_price,
                'current_price': current_price
            })
            logger.debug(f"Current Position: {ticker} - {qty} shares at an average price of {avg_price}, current price {current_price}")
        return current_positions
    except Exception as e:
        logger.error(f"Error fetching current positions: {e}", exc_info=True)
        return []

def is_market_open():
    """Check if the market is open (9:30 AM - 4:00 PM ET)."""
    now = datetime.now(tz=pytz.timezone('US/Eastern'))
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    return now.weekday() < 5 and market_open <= now.time() <= market_close

# =============================================================================
# Feature Engineering (Combined from both scripts)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the dataset, combining indicators from both scripts."""
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping feature engineering.")
        return pd.DataFrame()

    df.sort_values(['ticker', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df_list = []
    for ticker in df['ticker'].unique():
        df_ticker = df[df['ticker'] == ticker].copy()

        logger.debug(f"Processing {ticker} with {len(df_ticker)} rows")
        if len(df_ticker) < 5:  # Reduced from 50 for testing
            logger.warning(f"Not enough data for {ticker} ({len(df_ticker)} rows). Skipping.")
            continue

        # From Trader_main_Grok3.py
        df_ticker['MA10'] = df_ticker['close'].rolling(window=10, min_periods=1).mean()
        df_ticker['MA50'] = df_ticker['close'].rolling(window=50, min_periods=1).mean()
        df_ticker['RSI'] = ta.momentum.RSIIndicator(df_ticker['close'], window=14).rsi()

        macd_indicator = ta.trend.MACD(df_ticker['close'])
        df_ticker['MACD'] = macd_indicator.macd()
        df_ticker['MACD_Signal'] = macd_indicator.macd_signal()
        df_ticker['MACD_Diff'] = macd_indicator.macd_diff()

        bollinger = ta.volatility.BollingerBands(df_ticker['close'])
        df_ticker['Bollinger_Upper'] = bollinger.bollinger_hband()
        df_ticker['Bollinger_Lower'] = bollinger.bollinger_lband()

        try:
            atr_indicator = ta.volatility.AverageTrueRange(df_ticker['high'], df_ticker['low'], df_ticker['close'], window=5)  # Reduced window
            df_ticker['ATR'] = atr_indicator.average_true_range()
        except IndexError as e:
            logger.warning(f"Insufficient data for ATR calculation on {ticker}: {e}. Defaulting to 1.0.")
            df_ticker['ATR'] = 1.0

        stoch_rsi_indicator = ta.momentum.StochRSIIndicator(df_ticker['close'])
        df_ticker['Stochastic_RSI'] = stoch_rsi_indicator.stochrsi()

        df_ticker['Lag1_Close'] = df_ticker['close'].shift(1)
        df_ticker['Lag2_Close'] = df_ticker['close'].shift(2)

        # From trade_bot.py (additional indicators if TA-Lib available)
        if talib:
            df_ticker['Momentum'] = talib.MOM(df_ticker['close'], timeperiod=10)
            df_ticker['SMA_20'] = talib.SMA(df_ticker['close'], timeperiod=20)

        if len(df_ticker['volume']) > 1:
            df_ticker['Volume_Change'] = df_ticker['volume'] / df_ticker['volume'].shift(1)

        df_ticker['buy_signal'] = (
            (df_ticker['RSI'] < 30) &
            (df_ticker['MACD'] > df_ticker['MACD_Signal']) &
            (df_ticker['Stochastic_RSI'] < 0.2)
        ).astype(int)

        df_ticker['sell_signal'] = (
            (df_ticker['RSI'] > 70) &
            (df_ticker['MACD'] < df_ticker['MACD_Signal']) &
            (df_ticker['Stochastic_RSI'] > 0.8)
        ).astype(int)

        df_ticker['Target'] = (df_ticker['close'].shift(-1) > df_ticker['close']).astype(int)

        df_ticker.dropna(inplace=True)
        logger.debug(f"After NaN drop for {ticker}: {len(df_ticker)} rows")  # Log post-NaN count
        if len(df_ticker) < 5:
            logger.warning(f"Insufficient rows after NaN drop for {ticker} ({len(df_ticker)}). Skipping.")
            continue
        logger.debug(f"After feature engineering, {ticker} has {len(df_ticker)} rows")
        df_list.append(df_ticker)
        
    if df_list:
        df = pd.concat(df_list)
        logger.info(f"Engineered DataFrame shape: {df.shape}")  # Log shape
        logger.info(f"NaNs in DataFrame:\n{df.isna().sum()}")  # Log NaNs
        df.reset_index(drop=True, inplace=True)
    
        # Merge VIX if present (assume '^VIX' data fetched)
        vix_data = df[df['ticker'] == '^VIX'][['date', 'close']].rename(columns={'close': 'VIX'})
    
        if not vix_data.empty:
            df = df.merge(vix_data, on='date', how='left')
            df = df[df['ticker'] != '^VIX']  # Remove VIX row
        else:
            df['VIX'] = 0  # Fallback if no VIX data
    
        logger.info("Feature engineering completed successfully.")
        return df
    else:
        logger.warning("No valid tickers after feature engineering. Returning empty DataFrame.")
        return pd.DataFrame()


def add_sentiment_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """Add sentiment scores as features to the DataFrame."""
    if df is None or df.empty:
        logger.warning("Input DataFrame is None or empty. Returning as is.")
        return df
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    news = loop.run_until_complete(fetch_news(config))
    loop.close()
    twitter = fetch_twitter_sync(config)
    sentiments = analyze_sentiment(news + twitter, config)
    
    sentiment_scores = {}
    for ticker in config['tickers']:
        ticker_sentiments = [s for s in sentiments if s['ticker'] == ticker]
        if ticker_sentiments:
            positive = sum(1 for s in ticker_sentiments if s['label'] == 'Positive')
            negative = sum(1 for s in ticker_sentiments if s['label'] == 'Negative')
            total = len(ticker_sentiments)
            sentiment_scores[ticker] = (positive - negative) / total if total > 0 else 0
        else:
            sentiment_scores[ticker] = 0
    
    df['Sentiment_Score'] = df['ticker'].map(sentiment_scores)
    logger.info(f"DataFrame shape after sentiment: {df.shape}")  # Log shape post-sentiment
    return df

# =============================================================================
# Model Training and Evaluation (from Trader_main_Grok3.py)
# =============================================================================

def prepare_train_test_data(df: pd.DataFrame, selected_features: list):
    """Split the data into training and testing sets using time series split."""
    try:
        X = df[selected_features].reset_index(drop=True)
        y = df['Target'].reset_index(drop=True)

        total_samples = len(df)
        train_size = int(0.8 * total_samples)
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        logger.info(f"Length of X_train: {len(X_train)}, Length of y_train: {len(y_train)}")
        logger.info(f"Length of X_test: {len(X_test)}, Length of y_test: {len(y_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in prepare_train_test_data: {e}", exc_info=True)
        raise

def tune_and_train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Perform hyperparameter tuning and train the model."""
    try:
        param_grid_xgb = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'scale_pos_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = RandomizedSearchCV(
            XGBClassifier(eval_metric='logloss'),
            param_distributions=param_grid_xgb,
            n_iter=100,
            cv=tscv,
            scoring='precision',
            n_jobs=-1,
            random_state=42
        )
        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
        
        if len(X_train) > 0:
            grid_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed.")
            final_model = grid_search.best_estimator_
            joblib.dump(final_model, 'final_model.pkl')
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info("Final model saved as 'final_model.pkl'.")
            return final_model
        else:
            logger.warning("No training data available. Model training skipped.")
            return None
    except Exception as e:
        logger.error(f"Error in tune_and_train_model: {e}", exc_info=True)
        raise

def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the trained model and save confusion matrix."""
    try:
        if X_test.isnull().values.any():
            logger.error("NaN values found in X_test. Dropping NaNs.")
            X_test = X_test.dropna()
            y_test = y_test.loc[X_test.index]
        if y_test.isnull().values.any():
            logger.error("NaN values found in y_test. Dropping NaNs.")
            y_test = y_test.dropna()
            X_test = X_test.loc[y_test.index]

        y_pred = model.predict(X_test)
        logger.info(f"Length of y_test: {len(y_test)}, Length of y_pred: {len(y_pred)}")
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        plt.close()
        logger.info("Confusion matrix saved as 'confusion_matrix.png'.")
        logger.info(f"Classification Report:\n{cr}")
        print(f"Classification Report:\n{cr}")
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}", exc_info=True)
        raise

# =============================================================================
# Q-Learning Integration (from trade_bot.py)
# =============================================================================

def load_q_table(path='q_table.csv'):
    try:
        q_table = pd.read_csv(path, index_col='state')
        if q_table.empty or 'state' not in q_table.index.names:
            raise ValueError("Invalid Q-table format")
        # Ensure all columns exist
        expected_columns = ['hold', 'buy', 'sell']
        for col in expected_columns:
            if col not in q_table.columns:
                q_table[col] = 0.0
        return q_table
    except Exception as e:
        logger.warning(f"Q-table not found or invalid. Creating new table. {e}")
        states = ['-'.join(map(str, combo)) for combo in product(range(3), range(3), range(3), range(3), range(2))]
        q_table = pd.DataFrame(0.0, index=states, columns=['hold', 'buy', 'sell'])
        q_table.index.name = 'state'
        q_table.to_csv(path)
        return q_table

def load_last_decisions(path='last_decisions.csv'):
    try:
        df = pd.read_csv(path)
        return df.set_index('ticker').to_dict('index')
    except Exception as e:
        logger.warning(f"Last decisions not found or invalid. {e}")
        return {}

def save_last_decisions(decisions, path='last_decisions.csv'):
    if decisions:
        df = pd.DataFrame.from_dict(decisions, orient='index')
        df.to_csv(path, index_label='ticker')

def get_state(sentiment_score, indicators, holding_state):
    if sentiment_score < -0.1:
        sentiment_state = 0
    elif sentiment_score > 0.1:
        sentiment_state = 2
    else:
        sentiment_state = 1
    rsi = indicators.get('RSI', 50)
    if rsi < 30:
        rsi_state = 0
    elif rsi > 70:
        rsi_state = 2
    else:
        rsi_state = 1
    macd_hist = indicators.get('MACD_Diff', 0)
    if macd_hist < -0.1:
        macd_state = 0
    elif macd_hist > 0.1:
        macd_state = 2
    else:
        macd_state = 1
    volume_change = indicators.get('Volume_Change', 1)
    if volume_change < 0.8:
        volume_state = 0
    elif volume_change > 1.5:
        volume_state = 2
    else:
        volume_state = 1
    return '-'.join(map(str, [sentiment_state, rsi_state, macd_state, volume_state, holding_state]))

def select_action(q_table, state, holding_state, epsilon=0.05):
    if state not in q_table.index:
        return 'hold'
    valid_actions = ['hold']
    if holding_state == 0:
        valid_actions.append('buy')
    else:
        valid_actions.append('sell')
    if np.random.random() < epsilon:
        return np.random.choice(valid_actions)
    else:
        q_values = q_table.loc[state][valid_actions]
        return q_values.idxmax()

# =============================================================================
# Signal Generation and Trading Logic (Hybrid)
# =============================================================================

def generate_signals(model: XGBClassifier, data: pd.DataFrame, selected_features: List[str], config, q_table, threshold: float = 0.5) -> pd.DataFrame:
    """Generate buy/sell signals using XGBoost predictions and refine with Q-learning."""
    if data.empty:
        logger.warning("generate_signals received empty DataFrame. Exiting early.")
        return pd.DataFrame()

    missing_features = [f for f in selected_features if f not in data.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}. Skipping signal generation.")
        return pd.DataFrame()

    try:
        probabilities = model.predict_proba(data[selected_features])[:, 1]
        data['Prediction'] = (probabilities >= threshold).astype(int)
        data['Signal'] = data.groupby('ticker')['Prediction'].diff().fillna(0)
        data['Signal'] = data['Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        for idx, row in data.iterrows():
            if row.get('VIX', 0) > config.get('vix_threshold', 25):
                data.loc[idx, 'Signal'] = 0  # Skip high-vol trades

        # Refine with Q-learning
        last_decisions = load_last_decisions()
        decisions_dict = {}
        api = tradeapi.REST(config['alpaca']['api_key'], config['alpaca']['api_secret'], config['alpaca']['base_url'])
        account = api.get_account()
        signals = []
        for ticker in config['tickers']:
            ticker_data = data[data['ticker'] == ticker]
            if ticker_data.empty:
                continue
            sentiment_score = ticker_data['Sentiment_Score'].iloc[-1] if 'Sentiment_Score' in ticker_data.columns else 0
            indicators = ticker_data.iloc[-1].to_dict()
            try:
                position = api.get_position(ticker)
                holding_state = 1 if int(position.qty) > 0 else 0
            except:
                holding_state = 0
            state = get_state(sentiment_score, indicators, holding_state)
            current_price = ticker_data['close'].iloc[-1]
            last = last_decisions.get(ticker)
            if last:
                last_state = last['state']
                last_action = last['action']
                last_price = last['price']
                last_state_split = last_state.split('-')
                last_holding_state = int(last_state_split[-1])
                delta = (current_price - last_price) / last_price
                if last_action == 'sell':
                    reward = 0.0
                elif last_action == 'buy':
                    reward = delta
                elif last_action == 'hold':
                    reward = delta if last_holding_state == 1 else 0.0
                else:
                    reward = 0.0
                alpha = 0.1
                gamma = 0.9
                if last_state in q_table.index and last_action in q_table.columns:
                    old_q = q_table.loc[last_state, last_action]
                    max_q_next = q_table.loc[state].max()
                    new_q = old_q + alpha * (reward + gamma * max_q_next - old_q)
                    q_table.loc[last_state, last_action] = new_q
                else:
                    logger.warning(f"State {last_state} or action {last_action} not found in Q-table. Skipping update.")
            action = select_action(q_table, state, holding_state, epsilon=0.05)
            # Override XGBoost signal with Q-learning if conflict
            if action == 'buy' and data.loc[data['ticker'] == ticker, 'Signal'].iloc[-1] != 1:
                data.loc[data['ticker'] == ticker, 'Signal'] = 1
            elif action == 'sell' and data.loc[data['ticker'] == ticker, 'Signal'].iloc[-1] != -1:
                data.loc[data['ticker'] == ticker, 'Signal'] = -1
            decisions_dict[ticker] = {'state': state, 'action': action, 'price': current_price}
            logger.info(f"{ticker}: State={state}, Action={action}, Sentiment={sentiment_score:.2f}")
        q_table.to_csv('q_table.csv')
        save_last_decisions(decisions_dict)
        logger.info("Hybrid signals generated successfully.")
        return data
    except Exception as e:
        logger.error(f"Error in generate_signals: {e}", exc_info=True)
        return pd.DataFrame()

def simulate_trades(data: pd.DataFrame, initial_capital: float, config, max_position: int = 100,
                    stop_loss_pct: float = 0.05, 
                    take_profit_pct: float = 0.10, 
                    buying_power_pct: float = 50, 
                    transaction_cost_pct: float = 0.001,
                    slippage_pct: float = 0.0001) -> Tuple[pd.DataFrame, List[Tuple]]:
    """Simulate trades based on generated signals."""
    try:
        capital = initial_capital
        buying_power = initial_capital * (buying_power_pct / 100)
        capital_history = []
        drawdown_info = []

        data.sort_values(['date', 'ticker'], inplace=True)
        grouped = data.groupby('date')
        positions = {}

        for date, group in grouped:
            for idx, row in group.iterrows():
                ticker = row['ticker']
                signal = row['Signal']
                current_price = row['close']

                if current_price <= 0 or pd.isna(current_price):
                    logger.warning(f"Invalid price for {ticker} on {date}. Skipping.")
                    continue

                if ticker not in positions:
                    positions[ticker] = {'position': 0, 'buy_price': 0}

                position = positions[ticker]['position']
                buy_price = positions[ticker]['buy_price']

                # Take-profit logic
                if position > 0 and current_price > buy_price * (1 + take_profit_pct):
                    sell_price = current_price * (1 - slippage_pct)
                    net_proceeds = position * sell_price * (1 - transaction_cost_pct)
                    capital += net_proceeds
                    buying_power += position * current_price
                    logger.info(f"Take-profit sold {position} shares of {ticker} on {date} at {sell_price:.2f} (net: {net_proceeds:.2f}).")
                    positions[ticker]['position'] = 0
                    positions[ticker]['buy_price'] = 0

                # Stop-loss logic
                if position > 0 and current_price < buy_price * (1 - stop_loss_pct):
                    sell_price = current_price * (1 - slippage_pct)
                    net_proceeds = position * sell_price * (1 - transaction_cost_pct)
                    drawdown_info.append((date, ticker, capital, current_price))
                    capital += net_proceeds
                    buying_power += position * current_price
                    logger.warning(f"Stop-loss triggered for {ticker} on {date} at {sell_price:.2f}. New capital: {capital:.2f}")
                    positions[ticker]['position'] = 0
                    positions[ticker]['buy_price'] = 0

                # Dynamic risk per trade based on current capital
                risk_per_trade = capital * config.get('risk_per_trade_pct', 0.005) / 100
                atr = row.get('ATR', 1.0)
                position_size = max(1, min(int(risk_per_trade / atr), max_position, int(buying_power / current_price)))
                trade_value = position_size * current_price

                # Buy logic with slippage and costs
                if signal == 1 and positions[ticker]['position'] == 0:
                    buy_price = current_price * (1 + slippage_pct)
                    trade_value = position_size * buy_price * (1 + transaction_cost_pct)
                    if trade_value <= buying_power and trade_value > 0:
                        positions[ticker]['position'] = position_size
                        positions[ticker]['buy_price'] = buy_price
                        capital -= trade_value
                        buying_power -= trade_value
                        logger.info(f"Bought {position_size} shares of {ticker} on {date} at {buy_price:.2f} (cost: {trade_value:.2f}).")
                    else:
                        logger.info(f"Insufficient buying power to execute trade for {ticker} on {date}.")

                # Sell logic with slippage and costs
                elif signal == -1 and positions[ticker]['position'] > 0:
                    sell_price = current_price * (1 - slippage_pct)
                    quantity_to_sell = positions[ticker]['position']
                    trade_value = quantity_to_sell * sell_price * (1 - transaction_cost_pct)
                    capital += trade_value
                    buying_power += quantity_to_sell * current_price
                    logger.info(f"Sold {quantity_to_sell} shares of {ticker} on {date} at {sell_price:.2f} (net: {trade_value:.2f}).")
                    positions[ticker]['position'] = 0
                    positions[ticker]['buy_price'] = 0

            total_position_value = 0
            for t in positions:
                position_qty = positions[t]['position']
                subset = data[(data['date'] == date) & (data['ticker'] == t)]
                if not subset.empty:
                    price = subset['close'].values[0]
                    total_position_value += position_qty * price
            capital_history.append({'date': date, 'capital': capital + total_position_value})

        capital_history_df = pd.DataFrame(capital_history)
        capital_history_df.set_index('date', inplace=True)
        logger.info("Trade simulation completed successfully.")
        return capital_history_df, drawdown_info
    except Exception as e:
        logger.error(f"Error in simulate_trades: {e}", exc_info=True)
        return pd.DataFrame(), []

def calculate_performance(capital_history: pd.DataFrame, drawdown_info: List[Tuple]):
    """Calculate and print performance metrics."""
    try:
        capital_history['Returns'] = capital_history['capital'].pct_change().fillna(0)
        cumulative_returns = (capital_history['Returns'] + 1).prod() - 1
        sharpe_ratio = (np.mean(capital_history['Returns']) / np.std(capital_history['Returns'])) * np.sqrt(252) if np.std(capital_history['Returns']) != 0 else 0
        max_drawdown = (capital_history['capital'].cummax() - capital_history['capital']).max()

        logger.info(f"Cumulative Returns: {cumulative_returns:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2f}")

        if drawdown_info:
            logger.info("\nSignificant Drawdown Periods:")
            for date, ticker, cap, price in drawdown_info:
                logger.info(f"Date: {date}, Ticker: {ticker}, Capital: {cap}, Price: {price}")
    except Exception as e:
        logger.error(f"Error in calculate_performance: {e}", exc_info=True)

def log_backtesting_results(capital_history: pd.DataFrame):
    """Log backtesting results into a CSV file."""
    try:
        capital_history.to_csv('backtesting_results.csv')
        logger.debug("Backtesting results logged into 'backtesting_results.csv'.")
        print("Backtesting results logged into 'backtesting_results.csv'.")
    except Exception as e:
        logger.error(f"Failed to log backtesting results: {e}", exc_info=True)

def plot_backtesting_results(capital_history: pd.DataFrame):
    """Plot backtesting results."""
    try:
        capital_history = capital_history.copy()  # Avoid modifying original
        capital_history.index = pd.to_datetime(capital_history.index.str.split('T').str[0])  # Strip time, convert to datetime
        plt.figure(figsize=(12, 6))
        plt.plot(capital_history.index, capital_history['capital'], label='Strategy Capital')
        plt.title('Backtesting Results')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('backtesting_results.png')
        plt.close()
        logger.debug("Backtesting results plot saved as 'backtesting_results.png'.")
        print("Backtesting results plot saved as 'backtesting_results.png'.")
    except Exception as e:
        logger.error(f"Failed to plot backtesting_results: {e}", exc_info=True)

def backtest_strategy(model: XGBClassifier, data: pd.DataFrame, selected_features: List[str], config, q_table, initial_capital: float = 10000):
    """Backtest the trading strategy on historical data."""
    logger.info("Starting backtesting on historical data.")
    data = add_sentiment_features(data, config)
    data = generate_signals(model, data, selected_features, config, q_table, threshold=0.5)

    if data.empty:
        logger.error("No data available after signal generation for backtesting. Exiting.")
        return

    capital_history, drawdown_info = simulate_trades(data, initial_capital, config=config)
    calculate_performance(capital_history, drawdown_info)
    log_backtesting_results(capital_history)
    plot_backtesting_results(capital_history)
    logger.info("Backtesting process completed successfully.")

def place_order(api: tradeapi.REST, symbol: str, qty: int, side: str, order_type: str = 'market', time_in_force: str = 'day'):
    """Place an order through Alpaca API."""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        logger.info(f"Order placed: {order}")
        send_notification('Trade Executed', f"{side.upper()} {qty} {symbol}")
        return order
    except Exception as e:
        logger.error(f"Error placing order for {symbol}: {e}", exc_info=True)
        return None

def monitor_order(api: tradeapi.REST, order_id: str, max_attempts: int = 10, wait_time: int = 5):
    """Monitor the status of an order."""
    attempts = 0
    while attempts < max_attempts:
        try:
            order = api.get_order(order_id)
            if order.status == 'filled':
                logger.info(f"Order {order_id} filled.")
                return True
            elif order.status in ['canceled', 'rejected', 'expired']:
                logger.warning(f"Order {order_id} status: {order.status}. Aborting.")
                return False
            else:
                logger.info(f"Order {order_id} status: {order.status}. Waiting...")
                time.sleep(wait_time)
                attempts += 1
        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {e}", exc_info=True)
            return False
    logger.warning(f"Order {order_id} not filled after {max_attempts} attempts.")
    return False

def execute_trading_logic_live(api: tradeapi.REST, data: pd.DataFrame, config, q_table, initial_capital: float, buying_power_pct: float = 100):
    """Execute trading logic using hybrid signals during live trading."""
    if data.empty:
        logger.warning("No data available for executing trading logic. Skipping trading.")
        return

    try:
        data.sort_values(['date', 'ticker'], inplace=True)
    except KeyError as e:
        logger.error(f"DataFrame missing expected columns: {e}. Skipping trading.")
        return

    capital = initial_capital
    buying_power = initial_capital * (buying_power_pct / 100)
    log_filename = initialize_trade_log()
    logger.info(f"Starting live trading with initial capital: {capital} and buying power: {buying_power}")

    current_positions = fetch_current_positions(api)
    positions_dict = {pos['ticker']: pos for pos in current_positions}

    trade_count = 0
    successful_trades = 0

    for idx, row in data.iterrows():
        ticker = row['ticker']
        signal = row['Signal']
        current_price = row['close']
        date = row['date']

        atr = row.get('ATR', 1.0)
        if np.isnan(atr) or atr < 0.01:
            logger.warning(f"ATR is NaN or too small for {ticker} on {date}. Skipping trade.")
            continue

        risk_per_trade = initial_capital * config.get('risk_per_trade_pct', 0.01) / 100
        position_size = max(1, min(int(risk_per_trade / atr), int(buying_power / current_price)))
        trade_value = position_size * current_price

        logger.debug(f"Ticker: {ticker}, Date: {date}, Signal: {signal}, Position size: {position_size}, Trade value: {trade_value}, Buying power: {buying_power}")

        if trade_value > buying_power:
            logger.info(f"Insufficient buying power to execute trade for {ticker} on {date}.")
            continue

        if signal == 1:
            if ticker not in positions_dict or positions_dict[ticker]['quantity'] == 0:
                order = place_order(api, ticker, position_size, 'buy')
                if order:
                    buying_power -= trade_value
                    trade_count += 1
                    log_trade({
                        'ticker': ticker,
                        'type': 'buy',
                        'price': current_price,
                        'quantity': position_size,
                        'status': 'submitted'
                    }, log_filename)
                    if monitor_order(api, order.id):
                        successful_trades += 1
                        positions_dict[ticker] = {'quantity': position_size}
        elif signal == -1:
            if ticker in positions_dict and positions_dict[ticker]['quantity'] > 0:
                quantity_to_sell = min(positions_dict[ticker]['quantity'], position_size)
                trade_value = quantity_to_sell * current_price
                order = place_order(api, ticker, quantity_to_sell, 'sell')
                if order:
                    buying_power += trade_value
                    trade_count += 1
                    log_trade({
                        'ticker': ticker,
                        'type': 'sell',
                        'price': current_price,
                        'quantity': quantity_to_sell,
                        'status': 'submitted'
                    }, log_filename)
                    if monitor_order(api, order.id):
                        successful_trades += 1
                        positions_dict[ticker]['quantity'] -= quantity_to_sell
                        if positions_dict[ticker]['quantity'] == 0:
                            del positions_dict[ticker]

    success_rate = (successful_trades / trade_count) * 100 if trade_count > 0 else 0
    with open(log_filename, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Summary", "", "Total Trades", trade_count, "Successful Trades", successful_trades, "Success Rate", f"{success_rate:.2f}%"])
    logger.info(f"Live trading session completed. Total trades: {trade_count}, Successful trades: {successful_trades}, Success rate: {success_rate:.2f}%")

# =============================================================================
# Continuous Trading Loop (Hybrid Execution)
# =============================================================================

def run_continuous_trading(api: tradeapi.REST, model: XGBClassifier, config, q_table, selected_features: List[str], initial_capital: float):
    """Run the trading bot continuously during market hours with 30-minute updates."""
    logger.info("Starting continuous trading mode...")
    raw_historical = None
    job_lock = threading.Lock()  # Thread safety lock

    def job():
        with job_lock:
            nonlocal raw_historical
            current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
            logger.info(f"Job running at {current_time.strftime('%H:%M:%S')} EDT")
            if is_market_open():
                logger.info("Market is open. Proceeding with trading logic...")
                # Fetch raw historical data once or update incrementally
                if raw_historical is None:
                    raw_historical = fetch_current_market_data(config['tickers'], config['polygon']['api_key'])
                    if raw_historical.empty:
                        logger.error("Failed to fetch initial historical data. Retrying in 30 minutes...")
                        return

                # Fetch latest data
                latest_data = fetch_latest_data(config['tickers'], api)
                if not latest_data.empty:
                    # Append latest data to raw historical data
                    raw_historical = pd.concat([raw_historical, latest_data]).drop_duplicates(subset=['ticker', 'date'], keep='last')
                    raw_historical = raw_historical.groupby('ticker').tail(200).reset_index(drop=True)  # Keep last 200 for buffer
                    logger.debug(f"Combined raw data rows after tail: {len(raw_historical)}")
                    for ticker in config['tickers']:
                        if ticker != '^VIX':
                            rows = len(raw_historical[raw_historical['ticker'] == ticker])
                            logger.debug(f"Rows for {ticker} after tail: {rows}")
                    engineered_data = engineer_features(raw_historical)
                    if engineered_data.empty:
                        logger.warning("Feature engineering resulted in empty DataFrame. Skipping this cycle.")
                        return
                    engineered_data = add_sentiment_features(engineered_data, config)
                    if not engineered_data.empty:
                        signals = generate_signals(model, engineered_data, selected_features, config, q_table)
                        if not signals.empty:
                            execute_trading_logic_live(api, signals, config, q_table, initial_capital)
                else:
                    logger.warning("No new data fetched. Skipping this cycle.")
            else:
                logger.info("Market is closed. Skipping job.")

    # Run immediately at startup
    logger.info("Running initial job at startup...")
    job()

    # Schedule every 30 minutes
    schedule.every(30).minutes.do(job)
    schedule.every().day.at(config.get('market_hours', {}).get('start', '09:30')).do(job)

    while True:
        schedule.run_pending()
        time.sleep(5)  # Reduced for testing

# =============================================================================
# Main Execution
# =============================================================================

def main():
    global config
    config_file = "config.yaml"
    config = load_configuration(config_file)

    api_key = config['alpaca']['api_key']
    api_secret = config['alpaca']['api_secret']
    base_url = config['alpaca']['base_url']
    polygon_api_key = config['polygon']['api_key']
    tickers = config['tickers']

    # Combined selected features
    selected_features = [
        'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Bollinger_Upper',
        'Bollinger_Lower', 'Lag1_Close', 'Lag2_Close', 'ATR', 'Stochastic_RSI', 'Momentum', 'SMA_20', 'Volume_Change'
    ]  # Exclude Sentiment_Score initially

    api = initialize_alpaca_api(api_key, api_secret, base_url)

    account = api.get_account()
    logger.info(f"Account Details - Cash: {account.cash}, Buying Power: {account.buying_power}")

    trained_today = input("Have you downloaded, trained, and backtested the model today? (yes/no): ").strip().lower()

    if trained_today == "no":
        logger.info("Starting data download, model training, and backtesting...")
        download_historical_data(tickers, polygon_api_key)
        data = load_historical_data("./data", config)
        if data.empty:
            logger.error("No data loaded. Exiting.")
            sys.exit(1)

        data = engineer_features(data)
        data = add_sentiment_features(data, config)
        if data.empty:
            logger.error("No data after feature engineering. Exiting.")
            sys.exit(1)

        selected_features_with_sentiment = selected_features + ['Sentiment_Score']  # Add after sentiment
        X_train, X_test, y_train, y_test = prepare_train_test_data(data, selected_features_with_sentiment)
        model = tune_and_train_model(X_train, y_train)
        if model:
            evaluate_model(model, X_test, y_test)
            q_table = load_q_table()
            backtest_strategy(model, data, selected_features_with_sentiment, config, q_table)

    elif trained_today == "yes":
        logger.info("Skipping data download, training, and backtesting as requested.")
    else:
        logger.error("Invalid input. Please enter 'yes' or 'no'. Exiting.")
        sys.exit(1)

    logger.info("Starting continuous trading...")
    try:
        if not os.path.exists('final_model.pkl'):
            logger.error("Model file 'final_model.pkl' not found. Please train the model first.")
            sys.exit(1)
        model = joblib.load('final_model.pkl')
        logger.info("Loaded pre-trained model from 'final_model.pkl'.")
        q_table = load_q_table()
        run_continuous_trading(api, model, config, q_table, selected_features + ['Sentiment_Score'], float(account.buying_power))  # Use buying_power
    except Exception as e:
        logger.error(f"Error during continuous trading: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted by user. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)