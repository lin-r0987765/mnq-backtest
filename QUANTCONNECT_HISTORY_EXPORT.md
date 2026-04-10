# QuantConnect History Export

Use this when the local `qqq_5m.csv` / `qqq_1d.csv` bridge data is too short and you want QuantConnect to backfill the missing history directly.

## Which File To Use

- Research notebook: `export_qc_qqq_history_research.py`
- Web IDE / backtest project: `lean/QQQ_ORB_DeepBacktest/QQQ_History_Export_WebIDE.py`

The two files are not interchangeable:

- `export_qc_qqq_history_research.py` uses `QuantBook` and must run in QuantConnect Research.
- `QQQ_History_Export_WebIDE.py` inherits from `QCAlgorithm` and can run in the normal Web IDE backtest environment.

## What It Exports

The Research script exports adjusted QQQ history into a `qc_history_export/` folder inside the notebook workspace:

- `qqq_5m_qc.csv`
- `qqq_1h_qc.csv`
- `qqq_1d_qc.csv`

The Web IDE exporter saves yearly gzip CSV chunks into the QuantConnect Object Store:

- `project/qc_history_export_qqq_5m_2017_csv_gz`, `project/qc_history_export_qqq_5m_2018_csv_gz`, ...
- `project/qc_history_export_qqq_1h_2017_csv_gz`, `project/qc_history_export_qqq_1h_2018_csv_gz`, ...
- `project/qc_history_export_qqq_1d_2017_csv_gz`, `project/qc_history_export_qqq_1d_2018_csv_gz`, ...
- `project/qc_history_export_manifest_json`

The CSV layout matches the local repo format:

```text
Datetime,Close,High,Low,Open,Volume
```

## Recommended Flow

### Research Mode

1. Open QuantConnect Research.
2. Paste the contents of `export_qc_qqq_history_research.py` into a notebook cell, or upload the file and run it there.
3. Wait for the script to print row counts and date ranges.
4. Download the exported CSV files from the `qc_history_export/` folder.

### Web IDE Mode

1. Open a normal QuantConnect project.
2. Paste `lean/QQQ_ORB_DeepBacktest/QQQ_History_Export_WebIDE.py` into `main.py`.
3. Run a backtest once.
4. Open the Object Store and download the generated `project/qc_history_export_*` keys.

For either mode, only place the files into the repo after verifying the reported date ranges cover the missing gap.

## Default Range

The script currently exports:

- start: `2017-04-03`
- end inclusive: `2026-04-02`
- symbol: `QQQ`
- normalization: `ADJUSTED`
- market hours: regular session only

## Notes

- The Web IDE exporter writes yearly gzip chunks on purpose. This is safer for QuantConnect Cloud storage quotas than trying to save one huge multi-year intraday CSV.
- The Object Store key format is intentionally `project/<safe_name>` because QuantConnect's documented examples use a project-id prefix and a single key name.
- If you only need part of the missing gap, narrow the year range in the exporter before running it.

## Local Follow-Up

After downloading the files, compare their ranges with:

- `qqq_5m.csv`
- `qqq_1h.csv`
- `qqq_1d.csv`

If the QuantConnect exports are longer and clean, they can replace the local bridge files and unlock matched-sample research for the slow-trend family.
