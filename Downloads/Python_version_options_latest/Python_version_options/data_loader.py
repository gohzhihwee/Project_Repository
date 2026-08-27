import re
import os
import datetime
import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Databento API parameters — kept for reference; not used in CSV-file mode.
# Revert get_chain_df to the _fetch_day path to re-enable API mode.
# ---------------------------------------------------------------------------
DATABENTO_DATASET = os.environ.get('DATABENTO_DATASET', 'OPRA.PILLAR')
DATABENTO_SCHEMA  = os.environ.get('DATABENTO_SCHEMA',  'cbbo-1m')
DATABENTO_STYPE   = os.environ.get('DATABENTO_STYPE',   'parent')

# Candidate column names for bid/ask/symbol across Databento schema versions.
# cbbo-1m uses bid_px / ask_px; mbp-1 uses bid_px_00 / ask_px_00.
# The loader tries each list in order and uses the first match found.
_BID_COLS = ['bid_px_00', 'bid_px', 'close_bid_px', 'open_bid_px']
_ASK_COLS = ['ask_px_00', 'ask_px', 'close_ask_px', 'open_ask_px']
_SYM_COLS = ['symbol', 'raw_symbol']
_SZ_COLS  = ['bid_sz_00', 'bid_sz', 'close_bid_sz', 'size']


class DatabentoCacheLoader:
    """
    Loads SPY option chain data from Databento CBBO-1m CSV files.

    Workflow
    --------
    1. On first call for any date, scan all *.csv files in csv_dir, split
       rows by trading date, and write one parquet file per date to cache_dir.
       This one-time step avoids re-reading large CSVs on every backtest run.
    2. Subsequent calls read the relevant parquet directly (fast path).
    3. If no CSV data exists for a requested date, return None so the caller
       falls back to the synthetic option chain generator.

    CSV expectations
    ----------------
    - Files placed in csv_dir with a .csv extension.
    - Must contain a ts_event column (Databento nanosecond UTC timestamp or
      ISO 8601 string) and a symbol/raw_symbol column in OSI format.
    - bid_px / ask_px (or the fallback names in _BID_COLS / _ASK_COLS) must
      be present.  All other columns are ignored.
    """

    def __init__(self, csv_dir: pathlib.Path, cache_dir: pathlib.Path,
                 symbol: str = "SPY"):
        self._csv_dir   = pathlib.Path(csv_dir)
        self._cache_dir = pathlib.Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._symbol    = symbol.upper()
        self._cache_built = False   # set True after the one-time CSV scan

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_chain_df(self, date: datetime.date,
                     spot: float) -> pd.DataFrame | None:

        cache_path = self._cache_dir / f"{date.isoformat()}.parquet"

        if not cache_path.exists():
            # First miss triggers a one-time scan of all CSV files that
            # populates the entire cache in one pass.
            if not self._cache_built:
                self._build_cache_from_csv()
                self._cache_built = True
            if not cache_path.exists():
                return None   # date absent from CSV data → synthetic fallback

        raw = pd.read_parquet(cache_path)
        if raw.empty:
            return None

        chain_df = self._parse_chain(raw, date)
        if chain_df is None or chain_df.empty:
            return None

        chain_df['ttm'] = chain_df['expiry'].apply(
            lambda e: max((e - date).days / 365.25, 0.001))
        return chain_df

    # ------------------------------------------------------------------
    # One-time CSV → parquet cache builder
    # ------------------------------------------------------------------

    def _build_cache_from_csv(self) -> None:
        """
        Stream every *.csv in csv_dir and write one parquet per trading date.

        Uses ts_recv (always populated in CBBO-1m) to derive the trading date.
        Rows are accumulated for the current date only; when a new date is seen
        the previous date is flushed to parquet and memory is freed.  Peak
        memory is therefore one day's worth of rows (~700 MB for SPY options)
        rather than the full CSV size.  Runs once; subsequent runs read
        parquets directly.
        """
        csv_files = sorted(self._csv_dir.glob('*.csv'))
        if not csv_files:
            print(f"[DataLoader] No CSV files found in {self._csv_dir}. "
                  f"Place Databento CBBO-1m CSV exports there and re-run.")
            return

        print(f"[DataLoader] Building parquet cache from "
              f"{len(csv_files)} CSV file(s) — runs once …")

        current_date: datetime.date | None = None
        current_parts: list[pd.DataFrame] = []
        dates_written = 0

        def _flush(d: datetime.date, parts: list[pd.DataFrame]) -> None:
            nonlocal dates_written
            cache_path = self._cache_dir / f"{d}.parquet"
            if not cache_path.exists():
                pd.concat(parts, ignore_index=True).to_parquet(
                    cache_path, index=False)
            dates_written += 1

        for csv_file in csv_files:
            print(f"  Processing {csv_file.name} …")
            for chunk in pd.read_csv(csv_file, chunksize=50_000,
                                     low_memory=False):
                # ts_recv is the bar-close timestamp and is always present;
                # ts_event can be null for bars with no trades.
                sort_col = ('ts_recv' if 'ts_recv' in chunk.columns
                            else 'ts_event')
                if sort_col not in chunk.columns:
                    continue
                try:
                    ts = pd.to_datetime(chunk[sort_col], utc=True)
                except (ValueError, TypeError):
                    continue
                chunk = chunk.copy()
                chunk['_date'] = ts.dt.date

                for d in sorted(chunk['_date'].dropna().unique()):
                    grp = chunk[chunk['_date'] == d].drop(columns=['_date'])
                    if current_date is not None and d != current_date:
                        _flush(current_date, current_parts)
                        current_date = None
                        current_parts = []
                    if current_date is None:
                        current_date = d
                    current_parts.append(grp)

            # Flush at end of each file; Databento splits files by date range
            # so dates do not span file boundaries.
            if current_date is not None and current_parts:
                _flush(current_date, current_parts)
                current_date = None
                current_parts = []

        if current_date is not None and current_parts:
            _flush(current_date, current_parts)

        print(f"[DataLoader] Cache ready: {dates_written} date(s) → "
              f"{self._cache_dir}.")

    # ------------------------------------------------------------------
    # Databento API fetch — kept for reference, not called in CSV mode.
    # To switch back to API mode, restore the api_key parameter in __init__
    # and call _fetch_day from get_chain_df instead of _build_cache_from_csv.
    # ------------------------------------------------------------------

    def _fetch_day(self, date: datetime.date) -> pd.DataFrame | None:
        """
        NOT CALLED in CSV mode.  Fetches the EOD window (15:40–16:01 ET)
        for one trading day directly from the Databento Historical API.
        """
        try:
            from zoneinfo import ZoneInfo
            import databento as db
            api_key = os.environ.get('DATABENTO_API_KEY', '').strip()
            if not api_key:
                return None
            client = db.Historical(key=api_key)
            ET = ZoneInfo('America/New_York')
            window_start = datetime.datetime.combine(
                date, datetime.time(15, 40), tzinfo=ET)
            window_end   = datetime.datetime.combine(
                date, datetime.time(16,  1), tzinfo=ET)
            data = client.timeseries.get_range(
                dataset  = DATABENTO_DATASET,
                symbols  = [self._symbol],
                stype_in = DATABENTO_STYPE,
                schema   = DATABENTO_SCHEMA,
                start    = window_start,
                end      = window_end,
            )
            df = data.to_df()
            return df if not df.empty else None
        except Exception as exc:
            print(f"[DataLoader] Databento API fetch failed for {date}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Parsing raw DataFrame → calibration-format DataFrame
    # ------------------------------------------------------------------

    def _parse_chain(self, raw: pd.DataFrame,
                     date: datetime.date) -> pd.DataFrame | None:
        bid_col = next((c for c in _BID_COLS if c in raw.columns), None)
        ask_col = next((c for c in _ASK_COLS if c in raw.columns), None)
        sym_col = next((c for c in _SYM_COLS if c in raw.columns), None)
        sz_col  = next((c for c in _SZ_COLS  if c in raw.columns), None)

        if bid_col is None or ask_col is None or sym_col is None:
            print(
                f"[DataLoader] Unrecognised column names in CSV data.\n"
                f"  Expected bid in {_BID_COLS}, ask in {_ASK_COLS}, "
                f"symbol in {_SYM_COLS}.\n"
                f"  Actual columns: {list(raw.columns)}\n"
                f"  Adjust _BID_COLS / _ASK_COLS / _SYM_COLS in data_loader.py."
            )
            return None

        # Keep the last record per contract — for cbbo-1m this is the 16:00
        # bar, giving the EOD consolidated best bid/ask.
        # Prefer ts_recv (bar-close time, always populated) over ts_event
        # (last-event time, can be null for bars with no trades).
        sort_col = ('ts_recv' if 'ts_recv' in raw.columns
                    else 'ts_event' if 'ts_event' in raw.columns
                    else None)
        if sort_col:
            last = (raw.sort_values(sort_col, na_position='first')
                       .drop_duplicates(subset=[sym_col], keep='last')
                       .reset_index(drop=True))
        else:
            last = (raw.drop_duplicates(subset=[sym_col], keep='last')
                       .reset_index(drop=True))

        rows = []
        for _, row in last.iterrows():
            parsed = _parse_osi_symbol(str(row[sym_col]))
            if parsed is None:
                continue
            root, expiry, option_type, strike = parsed
            if root != self._symbol:
                continue
            if expiry <= date:
                continue

            bid = _safe_float(row.get(bid_col))
            ask = _safe_float(row.get(ask_col))
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            vol = int(_safe_float(row.get(sz_col, 0)) or 0)

            rows.append({
                'contract_symbol': str(row[sym_col]),
                'strike':          strike,
                'expiry':          expiry,
                'option_type':     option_type,
                'bid_price':       round(bid, 4),
                'ask_price':       round(ask, 4),
                'last_price':      round(mid, 4),
                'mid_price':       round(mid, 4),
                'volume':          vol,
                'open_interest':   0,
            })

        return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# OSI symbol parser
# ---------------------------------------------------------------------------

_OSI_RE = re.compile(r'^([A-Z]+)(\d{6})([CP])(\d{8})$')


def _parse_osi_symbol(raw_symbol: str) -> tuple | None:

    s = re.sub(r'\s+', '', raw_symbol)
    m = _OSI_RE.match(s)
    if not m:
        return None
    root       = m.group(1)
    exp_str    = m.group(2)
    right      = m.group(3)
    strike_str = m.group(4)
    try:
        expiry = datetime.datetime.strptime(exp_str, '%y%m%d').date()
    except ValueError:
        return None
    strike      = int(strike_str) / 1000.0
    option_type = 'call' if right == 'C' else 'put'
    return root, expiry, option_type, strike


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
