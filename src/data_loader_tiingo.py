import yaml
import pandas as pd
from tiingo import TiingoClient
import os
from datetime import datetime


def load_config(config_path="config/assets.yaml"):
    """
    Loads the configuration from the specified YAML file.

    Args:
        config_path (str, optional): Path to the configuration file,
            relative to the script's location.
            Defaults to "config/assets.yaml".

    Returns:
        dict: The configuration data, or None on error.
    """
    try:
        # Use os.path.dirname to get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go to the parent directory (root)
        root_dir = os.path.dirname(script_dir)
        # Construct the absolute path to the config file
        config_path = os.path.join(root_dir, config_path)

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file at {config_path}: {e}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the config: {e}")
        return None


def download_historical_data(ticker, start_date, end_date, api_key, output_dir="data"):
    """
    Downloads and updates historical trading data for a given ticker using the Tiingo API.
    If a Parquet file already exists, appends only new data.

    Args:
        ticker (str): Ticker symbol (e.g., "SPY", "BRK-B").
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        api_key (str): Tiingo API key.
        output_dir (str): Directory where Parquet files are stored.
    """

    # Convert user ticker to Tiingo-compatible format
    client = TiingoClient({"api_key": api_key})

    # Construct output file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(root_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}.parquet")

    # Determine if we're updating or downloading from scratch
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)

        last_date = existing_df.index.max()
        effective_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if effective_start > end_date:
            print(f"{ticker}: Already up to date.")
            return
        else:
            print(f"{ticker}: Updating from {effective_start} to {end_date}.")
    else:
        print(
            f"{ticker}: No existing file found. Downloading full data from {start_date} to {end_date}."
        )
        existing_df = None
        effective_start = start_date

    # Query Tiingo API
    try:
        df = client.get_dataframe(
            ticker, startDate=effective_start, endDate=end_date, frequency="daily"
        )
    except Exception as e:
        print(f"{ticker}: Error fetching data from Tiingo: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame on error

    if df.empty:
        print(f"{ticker}: No data returned from Tiingo.")
        return

    df.sort_index(inplace=True)

    # Merge with existing data if applicable
    if existing_df is not None:
        combined = pd.concat([existing_df, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = df

    combined.to_parquet(file_path)
    print(f"{ticker}: Data saved to {file_path}")


def main():
    """
    Main function to load configuration, validate API key,
    and download historical data for all tickers.
    """
    config = load_config()
    if config is None:
        print("Failed to load configuration. Exiting.")
        return

    tickers = config.get("tickers")
    start_date = config.get("start_date")
    # end_date = config.get('end_date') # No longer read from config
    interval = config.get("interval")  # read interval

    #  Load API key from environment variable, for security
    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        print(
            "Error: Tiingo API key not found in environment variable 'TIINGO_API_KEY'."
        )
        print("Please set the environment variable and try again.")
        return

    # Basic input validation
    if not tickers:
        print("Error: No tickers found in the configuration file.")
        return
    if not start_date:
        print("Error: Start date is missing in the configuration file.")
        return

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        print("Error: Invalid date format. Please use %Y-%m-%d.")
        return

    # Get today's date in YYYY-MM-DD format
    end_date = datetime.today().strftime("%Y-%m-%d")

    for ticker in tickers:
        download_historical_data(ticker, start_date, end_date, api_key)


if __name__ == "__main__":
    main()
