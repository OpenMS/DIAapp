"""
osw_utils.py — Utilities for loading and querying OpenSwath OSW SQLite databases.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class OSWFile:
    """Handler for OpenSwath OSW SQLite database files.

    Notes
    -----
    Streamlit reruns can execute in different threads. To avoid cross-thread
    sqlite3 errors, this class opens short-lived read-only connections per query
    instead of keeping one persistent connection on the instance.
    """

    def __init__(self, osw_path: str):
        self.osw_path = Path(osw_path)
        if not self.osw_path.exists():
            raise FileNotFoundError(f"OSW file not found: {osw_path}")
        self._table_cache: Optional[set[str]] = None

    def close(self) -> None:
        """Compatibility no-op; connections are opened per query."""
        return None

    def __del__(self) -> None:
        self.close()

    def _connect(self) -> sqlite3.Connection:
        """Create a fresh SQLite connection for the current thread."""
        conn = sqlite3.connect(str(self.osw_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _normalize_sqlite_value(value: Any) -> Any:
        """Convert SQLite values into Python-friendly scalar values."""
        if isinstance(value, bytes):
            if len(value) in (4, 8):
                try:
                    return int.from_bytes(value, byteorder="little", signed=True)
                except Exception:
                    pass
            try:
                return int.from_bytes(value, byteorder="little", signed=False)
            except Exception:
                return value.hex()
        return value

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize potentially byte-encoded SQLite columns in a dataframe."""
        if df.empty:
            return df
        return df.apply(lambda col: col.map(self._normalize_sqlite_value))

    def _read_sql(self, query: str, params: tuple[Any, ...] | list[Any] | None = None) -> pd.DataFrame:
        """Run a SQL query and return a normalized dataframe."""
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return self._normalize_dataframe(df)

    def _fetch_tables(self) -> set[str]:
        if self._table_cache is None:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables = self._read_sql(query)["name"].tolist()
            self._table_cache = set(str(t) for t in tables)
        return self._table_cache

    def has_table(self, table_name: str) -> bool:
        """Return True if the OSW database contains the requested table."""
        return table_name in self._fetch_tables()

    def list_runs(self) -> pd.DataFrame:
        """List all runs in the OSW file."""
        try:
            df = self._read_sql("SELECT ID, FILENAME FROM RUN ORDER BY FILENAME")
            if "ID" in df.columns:
                df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
            return df
        except Exception:
            return pd.DataFrame(columns=["ID", "FILENAME"])

    def get_run_id_by_filename(self, filename: str) -> Optional[int]:
        """Get RUN.ID by partial filename match."""
        try:
            query = (
                "SELECT ID FROM RUN WHERE FILENAME LIKE ? "
                "ORDER BY LENGTH(FILENAME) ASC LIMIT 1"
            )
            df = self._read_sql(query, params=(f"%{filename}%",))
            if df.empty:
                return None
            return int(df.iloc[0]["ID"])
        except Exception:
            return None

    def _feature_select_sql(self) -> tuple[str, str, str, str]:
        """Build table-aware SELECT/JOIN/ORDER BY SQL for feature queries."""
        score_join = ""
        score_select = "NULL AS SCORE, NULL AS RANK, NULL AS PVALUE, NULL AS QVALUE, NULL AS PEP"
        order_rank = "CASE WHEN 1=1 THEN NULL END"
        if self.has_table("SCORE_MS2"):
            score_join = "LEFT JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID"
            score_select = (
                "s.SCORE AS SCORE, s.RANK AS RANK, s.PVALUE AS PVALUE, "
                "s.QVALUE AS QVALUE, s.PEP AS PEP"
            )
            order_rank = "CASE WHEN s.RANK IS NULL THEN 1 ELSE 0 END, s.RANK ASC"

        ms2_join = ""
        ms2_select = "NULL AS AREA_INTENSITY, NULL AS TOTAL_AREA_INTENSITY, NULL AS APEX_INTENSITY"
        order_intensity = "CASE WHEN 1=1 THEN NULL END"
        if self.has_table("FEATURE_MS2"):
            ms2_join = "LEFT JOIN FEATURE_MS2 m ON f.ID = m.FEATURE_ID"
            ms2_select = (
                "m.AREA_INTENSITY AS AREA_INTENSITY, "
                "m.TOTAL_AREA_INTENSITY AS TOTAL_AREA_INTENSITY, "
                "m.APEX_INTENSITY AS APEX_INTENSITY"
            )
            order_intensity = "CASE WHEN m.AREA_INTENSITY IS NULL THEN 1 ELSE 0 END, m.AREA_INTENSITY DESC"

        order_by = f"{order_rank}, {order_intensity}, f.EXP_RT ASC"
        joins_sql = f"{score_join} {ms2_join}".strip()
        return score_select, ms2_select, joins_sql, order_by

    def get_peak_boundaries(self, precursor_id: int, run_id: int) -> pd.DataFrame:
        """
        Return all feature boundaries for a precursor in a run.

        Ranking preference:
        1. SCORE_MS2.RANK when SCORE_MS2 exists.
        2. FEATURE_MS2.AREA_INTENSITY when SCORE_MS2 is absent.
        3. EXP_RT as a stable final tie-breaker.
        """
        score_select, ms2_select, joins_sql, order_by = self._feature_select_sql()

        query = f"""
        SELECT
            f.ID AS FEATURE_ID,
            f.RUN_ID,
            f.PRECURSOR_ID,
            f.EXP_RT,
            f.LEFT_WIDTH,
            f.RIGHT_WIDTH,
            {ms2_select},
            {score_select}
        FROM FEATURE f
        {joins_sql}
        WHERE f.PRECURSOR_ID = ? AND f.RUN_ID = ?
        ORDER BY {order_by}
        """

        try:
            df = self._read_sql(query, params=(precursor_id, run_id))
            if not df.empty:
                for col in [
                    "FEATURE_ID",
                    "RUN_ID",
                    "PRECURSOR_ID",
                    "EXP_RT",
                    "LEFT_WIDTH",
                    "RIGHT_WIDTH",
                    "AREA_INTENSITY",
                    "TOTAL_AREA_INTENSITY",
                    "APEX_INTENSITY",
                    "SCORE",
                    "RANK",
                    "PVALUE",
                    "QVALUE",
                    "PEP",
                ]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="ignore")

                df["BOUNDARY_WIDTH"] = df["RIGHT_WIDTH"] - df["LEFT_WIDTH"]
                df["TOP_FEATURE"] = False
                if "RANK" in df.columns and df["RANK"].notna().any():
                    rank1 = df.index[df["RANK"] == 1]
                    if len(rank1) > 0:
                        df.loc[rank1[0], "TOP_FEATURE"] = True
                    else:
                        df.loc[df.index[0], "TOP_FEATURE"] = True
                    df["TOP_SOURCE"] = "SCORE_MS2.RANK"
                elif "AREA_INTENSITY" in df.columns and df["AREA_INTENSITY"].notna().any():
                    best_idx = df["AREA_INTENSITY"].fillna(float("-inf")).idxmax()
                    df.loc[best_idx, "TOP_FEATURE"] = True
                    df["TOP_SOURCE"] = "FEATURE_MS2.AREA_INTENSITY"
                else:
                    df.loc[df.index[0], "TOP_FEATURE"] = True
                    df["TOP_SOURCE"] = "FEATURE"
            return df
        except Exception:
            return pd.DataFrame(
                columns=[
                    "FEATURE_ID",
                    "RUN_ID",
                    "PRECURSOR_ID",
                    "EXP_RT",
                    "LEFT_WIDTH",
                    "RIGHT_WIDTH",
                    "AREA_INTENSITY",
                    "TOTAL_AREA_INTENSITY",
                    "APEX_INTENSITY",
                    "SCORE",
                    "RANK",
                    "PVALUE",
                    "QVALUE",
                    "PEP",
                    "BOUNDARY_WIDTH",
                    "TOP_FEATURE",
                    "TOP_SOURCE",
                ]
            )

    def get_selected_peak_boundaries(
        self,
        precursor_id: int,
        run_id: int,
        top_only: bool = False,
    ) -> pd.DataFrame:
        """Return all boundaries or only the selected top boundary."""
        df = self.get_peak_boundaries(precursor_id=precursor_id, run_id=run_id)
        if df.empty or not top_only:
            return df
        top_df = df[df["TOP_FEATURE"] == True]  # noqa: E712
        if top_df.empty:
            return df.head(1).copy()
        return top_df.copy()

    def get_feature_info(self, feature_id: int) -> Optional[dict[str, Any]]:
        """Return detailed information for a single feature."""
        df = self.get_peak_boundaries_for_feature_ids([feature_id])
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_peak_boundaries_for_feature_ids(self, feature_ids: list[int]) -> pd.DataFrame:
        """Return feature information for one or more feature IDs."""
        if not feature_ids:
            return pd.DataFrame()

        score_select, ms2_select, joins_sql, _ = self._feature_select_sql()
        placeholders = ",".join(["?"] * len(feature_ids))
        query = f"""
        SELECT
            f.ID AS FEATURE_ID,
            f.RUN_ID,
            f.PRECURSOR_ID,
            f.EXP_RT,
            f.LEFT_WIDTH,
            f.RIGHT_WIDTH,
            {ms2_select},
            {score_select}
        FROM FEATURE f
        {joins_sql}
        WHERE f.ID IN ({placeholders})
        ORDER BY f.EXP_RT ASC
        """
        try:
            df = self._read_sql(query, params=feature_ids)
            if not df.empty:
                df["BOUNDARY_WIDTH"] = df["RIGHT_WIDTH"] - df["LEFT_WIDTH"]
            return df
        except Exception:
            return pd.DataFrame()

    def get_precursor_qvalue_summary(self, run_ids: Optional[list[int]] = None) -> pd.DataFrame:
        """Return per-precursor q-value summary across OSW features.

        The returned BEST_QVALUE is the minimum feature q-value observed for the
        precursor across the selected runs. TOP_RANK_QVALUE is the minimum q-value
        among rank-1 features when rank information is available.
        """
        if not self.has_table("SCORE_MS2"):
            return pd.DataFrame(
                columns=[
                    "PRECURSOR_ID",
                    "BEST_QVALUE",
                    "BEST_PVALUE",
                    "TOP_RANK_QVALUE",
                    "N_FEATURES",
                    "N_RUNS",
                ]
            )

        params: list[Any] = []
        where_sql = ""
        if run_ids:
            placeholders = ",".join(["?"] * len(run_ids))
            where_sql = f"WHERE f.RUN_ID IN ({placeholders})"
            params.extend(int(r) for r in run_ids)

        query = f"""
        SELECT
            f.PRECURSOR_ID,
            MIN(s.QVALUE) AS BEST_QVALUE,
            MIN(s.PVALUE) AS BEST_PVALUE,
            MIN(CASE WHEN s.RANK = 1 THEN s.QVALUE END) AS TOP_RANK_QVALUE,
            COUNT(*) AS N_FEATURES,
            COUNT(DISTINCT f.RUN_ID) AS N_RUNS
        FROM FEATURE f
        INNER JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID
        {where_sql}
        GROUP BY f.PRECURSOR_ID
        ORDER BY f.PRECURSOR_ID ASC
        """
        try:
            df = self._read_sql(query, params=params)
            for col in [
                "PRECURSOR_ID",
                "BEST_QVALUE",
                "BEST_PVALUE",
                "TOP_RANK_QVALUE",
                "N_FEATURES",
                "N_RUNS",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "PRECURSOR_ID" in df.columns:
                df["PRECURSOR_ID"] = df["PRECURSOR_ID"].astype("Int64")
            if "N_FEATURES" in df.columns:
                df["N_FEATURES"] = df["N_FEATURES"].astype("Int64")
            if "N_RUNS" in df.columns:
                df["N_RUNS"] = df["N_RUNS"].astype("Int64")
            return df
        except Exception:
            return pd.DataFrame(
                columns=[
                    "PRECURSOR_ID",
                    "BEST_QVALUE",
                    "BEST_PVALUE",
                    "TOP_RANK_QVALUE",
                    "N_FEATURES",
                    "N_RUNS",
                ]
            )
