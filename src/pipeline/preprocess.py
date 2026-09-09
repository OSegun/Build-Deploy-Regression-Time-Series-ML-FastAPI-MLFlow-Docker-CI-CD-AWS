"""

- Reads train/test/new CSVs from ../data/raw/
- Cleans and normalizes city names.
- Maps cities to metros and merges latitudes and longitudes.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.
- Preprocessing: city normalization + (optional) latitudes and longitudes merge, duplicate drop, outlier removal.
- Production defaults read from ../data/raw/ and write to ../data/processed/

"""


import re
from pathlib import Path
import pandas as pd


RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# This is subject to changes based on data drift
city_name_mapping = {
    'Atlanta-Sandy Springs-Alpharetta': 'Atlanta-Sandy Springs-Roswell, GA',
    'Austin-Round Rock-Georgetown': 'Austin-Round Rock-San Marcos, TX',
    'Boston-Cambridge-Newton': 'Boston-Cambridge-Newton, MA-NH',
    'Baltimore-Columbia-Towson': 'Baltimore-Columbia-Towson, MD',
    'Chicago-Naperville-Elgin': 'Chicago-Naperville-Elgin, IL-IN',
    'Cincinnati': 'Cincinnati, OH-KY-IN',
    'Charlotte-Concord-Gastonia': 'Charlotte-Concord-Gastonia, NC-SC',
    'Dallas-Fort Worth-Arlington': 'Dallas-Fort Worth-Arlington, TX',
    'DC_Metro': 'Washington-Arlington-Alexandria, DC-VA-MD-WV',
    'Denver-Aurora-Lakewood': 'Denver-Aurora-Centennial, CO',
    'Detroit-Warren-Dearborn': 'Detroit-Warren-Dearborn, MI',
    'Houston-The Woodlands-Sugar Land': 'Houston-Pasadena-The Woodlands, TX',
    'Los Angeles-Long Beach-Anaheim': 'Los Angeles-Long Beach-Anaheim, CA',
    'Las Vegas-Henderson-Paradise': 'Las Vegas-Henderson-Paradise, NV',
    'Miami-Fort Lauderdale-Pompano Beach': 'Miami-Fort Lauderdale-West Palm Beach, FL',
    'Minneapolis-St. Paul-Bloomington': 'Minneapolis-St. Paul-Bloomington, MN-WI',
    'New York-Newark-Jersey City': 'New York-Newark-Jersey City, NY-NJ',
    'Orlando-Kissimmee-Sanford': 'Orlando-Kissimmee-Sanford, FL',
    'Portland-Vancouver-Hillsboro': 'Portland-Vancouver-Hillsboro, OR-WA',
    'Pittsburgh': 'Pittsburgh, PA',
    'Philadelphia-Camden-Wilmington': 'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD',
    'Phoenix-Mesa-Chandler': 'Phoenix-Mesa-Chandler, AZ',
    'Riverside-San Bernardino-Ontario': 'Riverside-San Bernardino-Ontario, CA',
    'San Antonio-New Braunfels': 'San Antonio-New Braunfels, TX',
    'Sacramento-Roseville-Folsom': 'Sacramento-Roseville-Folsom, CA',
    'San Diego-Chula Vista-Carlsbad': 'San Diego-Chula Vista-Carlsbad, CA',
    'Seattle-Tacoma-Bellevue': 'Seattle-Tacoma-Bellevue, WA',
    'San Francisco-Oakland-Berkeley': 'San Francisco-Oakland-Fremont, CA',
    'St. Louis': 'St. Louis, MO-IL',
    'Tampa-St. Petersburg-Clearwater': 'Tampa-St. Petersburg-Clearwater, FL'
}


def normalize_city(data: str) -> str:
    """Lowercase, strip, unify dashes. Safe for NA."""
    if pd.isna(data):
        return data
    data = str(data).strip().lower()
    data = re.sub(r"[–—-]", "-", data)          # unify dashes
    data = re.sub(r"\s+", " ", data)            # collapse spaces
    return data


def clean_and_merge(df: pd.DataFrame, 
                    metros_path: str | None = "../data/raw/map_data/usmetros.csv") -> pd.DataFrame:
    """
    Normalize city names, optionally merge latitudes and longitudes from metros dataset.
    If `city_full` column or `metros_path` is missing, skip gracefully.
    """

    if "city_full" not in df.columns:
        print("⚠️ Skipping city merge: no 'city_full' column present.")
        return df

    # Normalize city_full
    df["city_full"] = df["city_full"].apply(normalize_city)
    # Apply mapping
    norm_mapping = {normalize_city(k): normalize_city(v) for k, v in city_name_mapping.items()}
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # 🚨 If latitudes and longitudes already present, skip merge
    if {"lat", "lng"}.issubset(df.columns):
        print("⚠️ Skipping lat/lng merge: already present in DataFrame.")
        return df

    # If no metros file provided / exists, skip merge
    if not metros_path or not Path(metros_path).exists():
        print("⚠️ Skipping lat/lng merge: metros file not provided or not found.")
        return df

    # Merge latitudes and longitudes
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("⚠️ Skipping latitudes and longitudes merge: metros file missing required columns.")
        return df

    metros["metro_full"] = metros["metro_full"].apply(normalize_city)
    df = df.merge(metros[["metro_full", "lat", "lng"]],
                  how="left", left_on="city_full", right_on="metro_full")
    df.drop(columns=["metro_full"], inplace=True, errors="ignore")

    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("⚠️ Still missing latitudes and longitudes for:", missing)
    else:
        print("✅ All cities matched with metros dataset.")
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    after = df.shape[0]
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    suffix = "" if split.endswith("_data") else "_data"
    out_path = processed_dir / f"processed_{split}{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df



def preprocess(
    splits: tuple[str, ...] = ("train", "test", "new_data"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "../data/raw/usmetros.csv") -> pd.DataFrame:

    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)


if __name__ == "__main__":
    preprocess()
