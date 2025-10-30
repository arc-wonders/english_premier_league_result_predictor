import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# URLs for match result CSVs (Football-Data.co.uk)
season_urls = {
    "2020-21": "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
    "2021-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
}

data_frames = []

print("Downloading match data from Football-Data.co.uk...")
for season, url in season_urls.items():
    try:
        df = pd.read_csv(url)
        df['Season'] = season
        data_frames.append(df)
        print(f"Loaded {season} data with {len(df)} matches.")
    except Exception as e:
        print(f"Failed to load {season}: {e}")

# Example scraping from FBref (team stats)
def scrape_team_stats_fbref():
    base_url = "https://fbref.com"
    stats_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    print("Scraping team stats from FBref...")

    res = requests.get(stats_url)
    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.find('table', {'id': 'stats_squads_standard_for'})

    if not table:
        print("Team stats table not found on FBref.")
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]
    df = df[df['Rk'].apply(lambda x: str(x).isdigit())]  # Drop headers within the table
    df.drop(columns=['Rk'], inplace=True)
    df['Season'] = '2023-24'
    print(f"Scraped {len(df)} team stats rows.")
    return df

fbref_stats_df = scrape_team_stats_fbref()

if not fbref_stats_df.empty:
    data_frames.append(fbref_stats_df)

# Combine all data into one CSV
combined_df = pd.concat(data_frames, axis=0, ignore_index=True)
print(f"\nTotal combined records: {len(combined_df)}")

output_path = "epl_combined_data_2020_2025.csv"
combined_df.to_csv(output_path, index=False)
print(f"\nData saved to {output_path}")