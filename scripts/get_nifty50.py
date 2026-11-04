"""Download the NIFTY 50 daily open/close prices used in the companion demonstration.

The prices come from Yahoo Finance (ticker ^NSEI), whose terms do not allow redistribution,
so they are not included in this package. This script downloads them with the `yfinance`
package and writes data/nifty50_daily_open_close.csv in the format the code expects
(Date,Open,Close; two decimals; 14 Sep 2015 to 21 Sep 2026).

It then compares the SHA-256 checksum with that of the file used for the paper. Yahoo
sometimes revises past values, so a fresh download may not match exactly; the script
says so, and results that use these data may then differ slightly from the stored ones.

Usage:
    pip install yfinance
    python scripts/get_nifty50.py                 # download, write, verify
    python scripts/get_nifty50.py --check FILE    # only verify an existing file
"""
import argparse, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "nifty50_daily_open_close.csv")
SHA256 = "6bd750bd6be1c1f49c4481ffaf95ebb5859ce065b8b526ef5ad018410e93e3af"
START, END_INCLUSIVE, END_EXCLUSIVE, NROWS = "2015-09-14", "2026-09-21", "2026-09-22", 2712


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def verify(path):
    got = sha256(path)
    if got == SHA256:
        print(f"OK: {path} matches the file used for the paper (SHA-256 {SHA256[:16]}...).")
        return True
    print(f"WARNING: SHA-256 of {path} is {got},\n  expected {SHA256}.\n"
          "  The data differ from those used for the paper (for example, Yahoo revised past values).\n"
          "  Results that use the NIFTY 50 data may differ slightly from the stored ones.")
    return False


def download():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("This script needs yfinance:  pip install yfinance")
    df = yf.download("^NSEI", start=START, end=END_EXCLUSIVE, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        sys.exit("Download failed or returned no rows.")
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:   # newer yfinance: (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "Close"]].dropna()
    df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="\n") as f:
        f.write("Date,Open,Close\n")
        for t, row in df.iterrows():
            f.write(f"{t:%Y-%m-%d},{row['Open']:.2f},{row['Close']:.2f}\n")
    print(f"Wrote {len(df)} rows to {os.path.normpath(OUT)} "
          f"({df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}; the paper used {NROWS} rows, "
          f"{START} to {END_INCLUSIVE}).")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", metavar="FILE", help="only verify the checksum of FILE")
    a = ap.parse_args()
    ok = verify(a.check) if a.check else verify(download())
    sys.exit(0 if ok else 1)
