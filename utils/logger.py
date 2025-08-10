import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

PER_ITER_COLUMNS = [
    "dataset","method","run_id","iteration","accuracy","alignment","loss",
    "circuits","time_sec","subcentroids","noise_level","mitigation","n_samples"
]

PER_RUN_COLUMNS = [
    "dataset","method","run_id","test_accuracy","train_time_sec","circuits_total",
    "subcentroids","noise_level","mitigation","n_samples"
]

class Logger:
    """
    Unified logger that writes:
      - per-iteration rows -> per_iter_logs.csv
      - per-run summary rows -> per_run_summary.csv
    (Schemas above)

    Optional JSON mirrors can be enabled to keep a historical dump.
    """

    def __init__(
        self,
        dataset_name: str,
        log_dir: str,
        per_iter_csv: str = "per_iter_logs.csv",
        per_run_csv: str = "per_run_summary.csv",
        mirror_json: bool = False,
        per_iter_json: str = "per_iter_logs.json",
        per_run_json: str = "per_run_summary.json",
    ):
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir)
        self.per_iter_csv_path = self.log_dir / per_iter_csv
        self.per_run_csv_path = self.log_dir / per_run_csv
        self.mirror_json = mirror_json
        self.per_iter_json_path = self.log_dir / per_iter_json
        self.per_run_json_path = self.log_dir / per_run_json
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Ensure CSV headers exist
        self._ensure_csv_header(self.per_iter_csv_path, PER_ITER_COLUMNS)
        self._ensure_csv_header(self.per_run_csv_path, PER_RUN_COLUMNS)

        # Ensure JSON mirrors exist (optional)
        if self.mirror_json:
            self._ensure_json_array(self.per_iter_json_path)
            self._ensure_json_array(self.per_run_json_path)

    # ------------- public API -------------

    def log_per_iter(self, row: Dict):
        """
        Append one training-iteration row. Missing columns are written as empty strings.
        """
        self._append_csv_row(self.per_iter_csv_path, row, PER_ITER_COLUMNS)
        if self.mirror_json:
            self._append_json_row(self.per_iter_json_path, row)

    def log_per_run(self, row: Dict):
        """
        Append one run-summary row. Missing columns are written as empty strings.
        """
        self._append_csv_row(self.per_run_csv_path, row, PER_RUN_COLUMNS)
        if self.mirror_json:
            self._append_json_row(self.per_run_json_path, row)

    def read_per_iter(self) -> pd.DataFrame:
        return self._read_csv(self.per_iter_csv_path, PER_ITER_COLUMNS)

    def read_per_run(self) -> pd.DataFrame:
        return self._read_csv(self.per_run_csv_path, PER_RUN_COLUMNS)

    # ------------- legacy compatibility (optional) -------------

    def per_run_logs(self, metrics: Dict):
        """Backward compat: pipe to per-run CSV."""
        self.log_per_run(metrics)

    def per_dataset_logs(self, metrics: Dict):
        """Deprecated in the new schema. Kept for compatibility (does nothing)."""
        pass

    def create_df(self) -> pd.DataFrame:
        """Backward compat: returns per-run DataFrame."""
        return self.read_per_run()

    # ------------- helpers -------------

    @staticmethod
    def _ensure_csv_header(path: Path, header_cols: List[str]):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:
                f.write(",".join(header_cols) + "\n")

    @staticmethod
    def _append_csv_row(path: Path, row: Dict, header_cols: List[str]):
        vals = [Logger._to_str(row.get(col, "")) for col in header_cols]
        with path.open("a", newline="") as f:
            f.write(",".join(vals) + "\n")

    @staticmethod
    def _read_csv(path: Path, header_cols: List[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=header_cols)
        df = pd.read_csv(path)
        # Guarantee expected columns (fill missing, drop unknown)
        for c in header_cols:
            if c not in df.columns:
                df[c] = ""
        return df[header_cols]

    @staticmethod
    def _ensure_json_array(path: Path):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump([], f)

    @staticmethod
    def _append_json_row(path: Path, row: Dict):
        with path.open("r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
            data.append(row)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    @staticmethod
    def _to_str(v) -> str:
        if isinstance(v, bool):
            return "on" if v else "off"
        return "" if v is None else str(v)
