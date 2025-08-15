#!/usr/bin/env python3
"""
Dice Game — End-to-End ETL + Analytics (Star Schema)

What this script does
---------------------
1) Loads the provided CSVs for the 2024 Dice Game app.
2) Cleans and normalizes columns.
3) Builds a **Star Schema** locally with the following tables:
   - dim_date, dim_channel, dim_status, dim_plan, dim_user
   - fact_play_session, fact_subscription, fact_payments
4) Runs **data quality validations** (unit-test like checks) and prints a report.
5) Writes outputs to CSV (and Parquet if the engine is available).
6) Produces **2024 insights** and **2025 trend forecasts** and saves simple charts.

Usage
-----
python dice_game_etl.py \
  --base-dir /path/to/data \
  --out-dir /path/to/warehouse_star \
  --write-charts \
  --run-validations

By default this script expects files to exist in BASE DIR with these names:
- channel_code.csv
- plan_payment_frequency.csv
- plan.csv
- status_code.csv
- user_payment_detail.csv
- user_plan.csv
- user_play_session.csv
- user_registration.csv
- user.csv

Notes
-----
- Parquet output is attempted; if no engine is available, a NOT_WRITTEN marker is created.
- Forecasts are simple linear models using numpy.polyfit over 2024 monthly aggregates.
- Email masking is applied in dim_user to reduce PII risk.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------

@dataclass
class ETLConfig:
    base_dir: Path
    out_dir: Path
    write_charts: bool = True
    run_validations: bool = True

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utility helpers
# ---------------------------

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def mask_email(value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str) or "@" not in value:
        return value
    name, dom = value.split("@", 1)
    if len(name) <= 2:
        masked = "*" * len(name)
    else:
        masked = name[0] + "*" * (len(name) - 2) + name[-1]
    return masked + "@" + dom


def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        with open(str(path) + ".NOT_WRITTEN.txt", "w") as f:
            f.write(f"Parquet not written due to missing engine: {e}\n")


# ---------------------------
# Warehouse builder
# ---------------------------

class WarehouseBuilder:
    def __init__(self, cfg: ETLConfig):
        self.cfg = cfg
        self.raw: Dict[str, pd.DataFrame] = {}
        self.dims: Dict[str, pd.DataFrame] = {}
        self.facts: Dict[str, pd.DataFrame] = {}

    # ---- Load & clean
    def load_inputs(self) -> None:
        names = [
            "channel_code.csv",
            "plan_payment_frequency.csv",
            "plan.csv",
            "status_code.csv",
            "user_payment_detail.csv",
            "user_plan.csv",
            "user_play_session.csv",
            "user_registration.csv",
            "user.csv",
        ]
        for n in names:
            path = self.cfg.base_dir / n
            if path.exists():
                df = pd.read_csv(path)
                self.raw[n] = clean_cols(df)

        # Coerce relevant datetimes/dates
        if "user_play_session.csv" in self.raw:
            ups = self.raw["user_play_session.csv"].copy()
            for col in ["start_datetime", "end_datetime"]:
                if col in ups.columns:
                    ups[col] = to_datetime(ups[col])
            self.raw["user_play_session.csv"] = ups

        if "user_plan.csv" in self.raw:
            up = self.raw["user_plan.csv"].copy()
            for col in ["start_date", "end_date"]:
                if col in up.columns:
                    up[col] = to_date(up[col])
            self.raw["user_plan.csv"] = up

        if "user_payment_detail.csv" in self.raw:
            upd = self.raw["user_payment_detail.csv"].copy()
            if "payment_method_expiry" in upd.columns:
                upd["payment_method_expiry"] = to_date(upd["payment_method_expiry"])
            self.raw["user_payment_detail.csv"] = upd

        if "user_registration.csv" in self.raw:
            ur = self.raw["user_registration.csv"].copy()
            if "registration_date" in ur.columns:
                ur["registration_date"] = to_date(ur["registration_date"])
            if "email" in ur.columns:
                ur["email_masked"] = ur["email"].apply(mask_email)
            self.raw["user_registration.csv"] = ur

        if "plan.csv" in self.raw:
            plan = self.raw["plan.csv"].copy()
            if "cost_amount" in plan.columns:
                plan["cost_amount"] = pd.to_numeric(plan["cost_amount"], errors="coerce")
            self.raw["plan.csv"] = plan

    # ---- Dimensions
    def build_dim_date(self) -> pd.DataFrame:
        df_list: List[pd.DataFrame] = []
        cols_list: List[List[str]] = []
        if "user_play_session.csv" in self.raw:
            df_list.append(self.raw["user_play_session.csv"]) 
            cols_list.append(["start_datetime", "end_datetime"])
        if "user_plan.csv" in self.raw:
            df_list.append(self.raw["user_plan.csv"]) 
            cols_list.append(["start_date", "end_date"])
        if "user_registration.csv" in self.raw:
            df_list.append(self.raw["user_registration.csv"]) 
            cols_list.append(["registration_date"])

        all_dates: List[pd.Series] = []
        for df, cols in zip(df_list, cols_list):
            for c in cols:
                if c in df.columns:
                    s = pd.to_datetime(df[c], errors="coerce").dt.date
                    all_dates.append(s.dropna())
        if not all_dates:
            dim_date = pd.DataFrame(
                columns=["date_key", "full_date", "year", "quarter", "month", "day", "dow", "is_weekend"]
            )
        else:
            dates = pd.Series(pd.unique(pd.concat(all_dates).astype("datetime64[ns]"))).sort_values()
            dim_date = pd.DataFrame({"full_date": dates.dt.date})
            dim_date["date_key"] = dim_date["full_date"].astype(str).str.replace("-", "", regex=False).astype(int)
            dt = pd.to_datetime(dim_date["full_date"])
            dim_date["year"] = dt.dt.year
            dim_date["quarter"] = dt.dt.quarter
            dim_date["month"] = dt.dt.month
            dim_date["day"] = dt.dt.day
            dim_date["dow"] = dt.dt.dayofweek
            dim_date["is_weekend"] = dim_date["dow"].isin([5, 6]).astype(int)
            dim_date = dim_date[["date_key", "full_date", "year", "quarter", "month", "day", "dow", "is_weekend"]]

        self.dims["dim_date"] = dim_date
        return dim_date

    def build_dim_channel(self) -> pd.DataFrame:
        dim = pd.DataFrame()
        if "channel_code.csv" in self.raw:
            cc = self.raw["channel_code.csv"].copy()
            if "play_session_channel_code" in cc.columns:
                cc = cc.rename(columns={"play_session_channel_code": "channel_code"})
            if "english_description" in cc.columns:
                cc = cc.rename(columns={"english_description": "channel_name"})
            if "channel_name" not in cc.columns:
                cc["channel_name"] = cc.iloc[:, 0].astype(str)
            dim = cc[["channel_code", "channel_name"]].drop_duplicates().reset_index(drop=True)
            dim.insert(0, "channel_key", np.arange(1, len(dim) + 1))
        self.dims["dim_channel"] = dim
        return dim

    def build_dim_status(self) -> pd.DataFrame:
        dim = pd.DataFrame()
        if "status_code.csv" in self.raw:
            sc = self.raw["status_code.csv"].copy()
            code_col = None
            for c in ["play_session_status_code", "status_code"]:
                if c in sc.columns:
                    code_col = c
                    break
            if code_col is None:
                code_col = sc.columns[0]
            sc = sc.rename(columns={code_col: "status_code"})
            if "english_description" in sc.columns:
                sc = sc.rename(columns={"english_description": "status_name"})
            if "status_name" not in sc.columns:
                sc["status_name"] = sc["status_code"]
            dim = sc[["status_code", "status_name"]].drop_duplicates().reset_index(drop=True)
            dim.insert(0, "status_key", np.arange(1, len(dim) + 1))
        self.dims["dim_status"] = dim
        return dim

    def build_dim_plan(self) -> pd.DataFrame:
        dim = pd.DataFrame()
        if "plan.csv" in self.raw:
            pl = self.raw["plan.csv"].copy()
            pf = self.raw.get("plan_payment_frequency.csv", pd.DataFrame())
            if not pf.empty and "payment_frequency_code" not in pf.columns and "frequency_code" in pf.columns:
                pf = pf.rename(columns={"frequency_code": "payment_frequency_code"})
            if not pf.empty:
                pl = pl.merge(pf, on="payment_frequency_code", how="left")
            if "english_description" in pl.columns:
                pl = pl.rename(columns={"english_description": "payment_frequency_desc_en"})
            if "french_description" in pl.columns:
                pl = pl.rename(columns={"french_description": "payment_frequency_desc_fr"})
            dim = pl.copy()
            dim.insert(0, "plan_key", np.arange(1, len(dim) + 1))
        self.dims["dim_plan"] = dim
        return dim

    def build_dim_user(self) -> pd.DataFrame:
        dim = pd.DataFrame()
        if "user.csv" in self.raw and "user_registration.csv" in self.raw:
            u = self.raw["user.csv"].copy()
            ur = self.raw["user_registration.csv"].copy()
            if "email" in ur.columns and "email_masked" not in ur.columns:
                ur["email_masked"] = ur["email"].apply(mask_email)
            cols = [c for c in ["user_id", "username", "email_masked", "first_name", "last_name"] if c in (set(u.columns) | set(ur.columns))]
            dim = (
                u.merge(ur, on="user_id", how="left", suffixes=("", "_reg"))
                 .loc[:, cols]
                 .drop_duplicates()
                 .reset_index(drop=True)
            )
            dim.insert(0, "user_key", np.arange(1, len(dim) + 1))
        self.dims["dim_user"] = dim
        return dim

    # ---- Facts
    def build_fact_play_session(self) -> pd.DataFrame:
        fact = pd.DataFrame()
        if "user_play_session.csv" in self.raw:
            ups = self.raw["user_play_session.csv"].copy()
            dim_date = self.dims.get("dim_date", pd.DataFrame())
            if not dim_date.empty:
                key_map = dict(zip(dim_date["full_date"], dim_date["date_key"]))
                for side in ["start_datetime", "end_datetime"]:
                    if side in ups.columns:
                        d = to_datetime(ups[side]).dt.date
                        ups[f"{side}_date_key"] = d.map(key_map)
            dim_channel = self.dims.get("dim_channel", pd.DataFrame())
            if not dim_channel.empty and "channel_code" in ups.columns:
                ch_map = dict(zip(dim_channel["channel_code"], dim_channel["channel_key"]))
                ups["channel_key"] = ups["channel_code"].map(ch_map)
            dim_status = self.dims.get("dim_status", pd.DataFrame())
            if not dim_status.empty:
                st_map = dict(zip(dim_status["status_code"], dim_status["status_key"]))
                code_col = "status_code" if "status_code" in ups.columns else (
                    "play_session_status_code" if "play_session_status_code" in ups.columns else None
                )
                if code_col:
                    ups["status_key"] = ups[code_col].map(st_map)
            fact = ups.rename(columns={"play_session_id": "play_session_key"})
            if "start_datetime" in fact.columns and "end_datetime" in fact.columns:
                fact["duration_minutes"] = (
                    (to_datetime(fact["end_datetime"]) - to_datetime(fact["start_datetime"]))
                    .dt.total_seconds() / 60.0
                )
        self.facts["fact_play_session"] = fact
        return fact

    def build_fact_subscription(self) -> pd.DataFrame:
        fact = pd.DataFrame()
        if "user_plan.csv" in self.raw:
            up = self.raw["user_plan.csv"].copy()
            dim_plan = self.dims.get("dim_plan", pd.DataFrame())
            if not dim_plan.empty and "plan_id" in dim_plan.columns and "plan_id" in up.columns:
                plan_map = dict(zip(dim_plan["plan_id"], dim_plan["plan_key"]))
                up["plan_key"] = up["plan_id"].map(plan_map)
            dim_date = self.dims.get("dim_date", pd.DataFrame())
            if not dim_date.empty:
                kmap = dict(zip(dim_date["full_date"], dim_date["date_key"]))
                for col in ["start_date", "end_date"]:
                    if col in up.columns:
                        up[f"{col}_key"] = up[col].map(kmap)
            fact = up.rename(columns={"user_registration_id": "user_registration_key"})
        self.facts["fact_subscription"] = fact
        return fact

    def build_fact_payments(self) -> pd.DataFrame:
        fact = pd.DataFrame()
        sub = self.facts.get("fact_subscription", pd.DataFrame())
        dim_plan = self.dims.get("dim_plan", pd.DataFrame())
        if not sub.empty and not dim_plan.empty:
            f = sub.merge(
                dim_plan[["plan_key", "plan_id", "payment_frequency_code", "cost_amount"]],
                on="plan_key", how="left"
            )
            freq = self.raw.get("plan_payment_frequency.csv", pd.DataFrame())
            if not freq.empty:
                desc_series = (
                    freq["english_description"] if "english_description" in freq.columns else freq.iloc[:, 1]
                )
                freq_map = dict(zip(freq["payment_frequency_code"], desc_series))
                f["payment_frequency_desc"] = f["payment_frequency_code"].map(freq_map)
            fact = f
        self.facts["fact_payments"] = fact
        return fact

    # ---- Write
    def write_table(self, name: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df.to_csv(self.cfg.out_dir / f"{name}.csv", index=False)
        safe_to_parquet(df, self.cfg.out_dir / f"{name}.parquet")

    def write_all(self) -> None:
        for name, df in {**self.dims, **self.facts}.items():
            self.write_table(name, df)


# ---------------------------
# Validations
# ---------------------------

class Validator:
    def __init__(self):
        self.results: List[Dict[str, object]] = []

    def check(self, condition: bool, name: str) -> None:
        self.results.append({"check": name, "passed": bool(condition)})

    def run(self, wh: WarehouseBuilder) -> pd.DataFrame:
        dim_date = wh.dims.get("dim_date", pd.DataFrame())
        dim_channel = wh.dims.get("dim_channel", pd.DataFrame())
        dim_status = wh.dims.get("dim_status", pd.DataFrame())
        dim_plan = wh.dims.get("dim_plan", pd.DataFrame())
        dim_user = wh.dims.get("dim_user", pd.DataFrame())
        fps = wh.facts.get("fact_play_session", pd.DataFrame())

        # Presence & schema checks
        self.check(not dim_date.empty, "dim_date not empty")
        if not dim_channel.empty:
            self.check("channel_key" in dim_channel.columns, "dim_channel surrogate key present")
            self.check(dim_channel["channel_code"].is_unique, "dim_channel.channel_code unique")
        if not dim_status.empty:
            self.check("status_key" in dim_status.columns, "dim_status surrogate key present")
            self.check(dim_status["status_code"].is_unique, "dim_status.status_code unique")
        if not dim_plan.empty:
            self.check("plan_key" in dim_plan.columns, "dim_plan surrogate key present")
            self.check(dim_plan["plan_id"].is_unique, "dim_plan.plan_id unique")
        if not dim_user.empty:
            self.check("user_key" in dim_user.columns, "dim_user surrogate key present")

        if not fps.empty:
            sdk = "start_datetime_date_key"
            self.check(sdk in fps.columns and fps[sdk].notna().mean() >= 0.95, "fact_play_session.start_date_key >=95% coverage")
            if not dim_channel.empty and "channel_key" in fps.columns:
                self.check(
                    fps["channel_key"].isin(dim_channel["channel_key"]).mean() >= 0.95,
                    "fact_play_session.channel_key valid >=95%"
                )
            self.check("duration_minutes" in fps.columns, "fact_play_session has duration_minutes")

        return pd.DataFrame(self.results)


# ---------------------------
# Insights & Forecasts
# ---------------------------

class Insights:
    def __init__(self, wh: WarehouseBuilder):
        self.wh = wh

    @staticmethod
    def _classify_plan(freq_desc: Optional[str], code: Optional[str]) -> str:
        text = (str(freq_desc) if pd.notna(freq_desc) else str(code)).lower()
        if any(k in text for k in ["one-time", "onetime", "once", "single"]):
            return "One-time"
        if any(k in text for k in ["month", "monthly", "subscription", "subs"]):
            return "Subscription (Monthly)"
        if any(k in text for k in ["year", "annual", "yearly"]):
            return "Subscription (Annual)"
        return "Other/Unknown"

    def analyze(self, write_charts: bool = True) -> Dict[str, pd.DataFrame]:
        insights: Dict[str, pd.DataFrame] = {}
        dim_channel = self.wh.dims.get("dim_channel", pd.DataFrame())
        fps = self.wh.facts.get("fact_play_session", pd.DataFrame())
        fpay = self.wh.facts.get("fact_payments", pd.DataFrame())

        # Sessions by channel in 2024
        if not fps.empty and not dim_channel.empty and "start_datetime" in fps.columns:
            joined = fps.merge(dim_channel, on="channel_key", how="left")
            joined2024 = joined[pd.to_datetime(joined["start_datetime"], errors="coerce").dt.year == 2024]
            by_channel = (
                joined2024.groupby("channel_name", dropna=False)["play_session_key"].count()
                          .reset_index(name="sessions_2024")
                          .sort_values("sessions_2024", ascending=False)
            )
            insights["sessions_by_channel_2024"] = by_channel

            monthly = (
                joined2024.assign(month=lambda d: pd.to_datetime(d["start_datetime"]).dt.to_period("M").astype(str))
                          .groupby("month")["play_session_key"].count()
                          .reset_index(name="sessions")
            )
            if len(monthly) >= 2:
                monthly = monthly.sort_values("month").reset_index(drop=True)
                monthly["t"] = np.arange(1, len(monthly) + 1)
                a, b = np.polyfit(monthly["t"], monthly["sessions"], deg=1)
                last_t = monthly["t"].iloc[-1]
                months = pd.period_range("2025-01", "2025-12", freq="M").astype(str)
                f_list = [(m, float(max(0.0, a * (last_t + i) + b))) for i, m in enumerate(months, start=1)]
                forecast_df = pd.DataFrame(f_list, columns=["month", "forecast_sessions"])
                insights["monthly_sessions_2024"] = monthly[["month", "sessions"]]
                insights["monthly_sessions_forecast_2025"] = forecast_df

                if write_charts:
                    plt.figure()
                    plt.bar(by_channel["channel_name"].astype(str), by_channel["sessions_2024"])
                    plt.title("2024 Sessions by Channel")
                    plt.xlabel("Channel")
                    plt.ylabel("Sessions")
                    plt.tight_layout()
                    plt.savefig(self.wh.cfg.out_dir / "chart_sessions_by_channel_2024.png")
                    plt.close()

                    plt.figure()
                    plt.plot(monthly["month"], monthly["sessions"], marker="o")
                    plt.title("2024 Monthly Sessions")
                    plt.xlabel("Month")
                    plt.ylabel("Sessions")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.wh.cfg.out_dir / "chart_monthly_sessions_2024.png")
                    plt.close()

                    plt.figure()
                    plt.plot(forecast_df["month"], forecast_df["forecast_sessions"], marker="o")
                    plt.title("2025 Forecast: Monthly Sessions")
                    plt.xlabel("Month")
                    plt.ylabel("Forecast Sessions")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.wh.cfg.out_dir / "chart_monthly_sessions_forecast_2025.png")
                    plt.close()

        # Plan mix & revenue proxy (starts in 2024)
        if not fpay.empty:
            fp = fpay.copy()
            fp["start_date"] = pd.to_datetime(fp["start_date"], errors="coerce")
            fp2024 = fp[fp["start_date"].dt.year == 2024].copy()
            fp2024["plan_type"] = [self._classify_plan(r.get("payment_frequency_desc"), r.get("payment_frequency_code")) for _, r in fp2024.iterrows()]
            plan_mix = (
                fp2024.groupby("plan_type")["user_registration_key"].nunique()
                      .reset_index(name="registrations_2024")
                      .sort_values("registrations_2024", ascending=False)
            )
            insights["plan_mix_and_revenue_2024"] = plan_mix

            monthly_rev = (
                fp2024.assign(month=lambda d: d["start_date"].dt.to_period("M").astype(str))
                      .groupby("month")["cost_amount"].sum()
                      .reset_index(name="revenue")
                      .sort_values("month")
                      .reset_index(drop=True)
            )
            if len(monthly_rev) >= 2:
                monthly_rev["t"] = np.arange(1, len(monthly_rev) + 1)
                a2, b2 = np.polyfit(monthly_rev["t"], monthly_rev["revenue"], deg=1)
                last_t2 = monthly_rev["t"].iloc[-1]
                months = pd.period_range("2025-01", "2025-12", freq="M").astype(str)
                f_rev = [(m, float(max(0.0, a2 * (last_t2 + i) + b2))) for i, m in enumerate(months, start=1)]
                forecast_rev = pd.DataFrame(f_rev, columns=["month", "forecast_revenue"])
                insights["monthly_revenue_2024"] = monthly_rev[["month", "revenue"]]
                insights["monthly_revenue_forecast_2025"] = forecast_rev

                if write_charts:
                    plt.figure()
                    plt.plot(monthly_rev["month"], monthly_rev["revenue"], marker="o")
                    plt.title("2024 Monthly Revenue (Proxy)")
                    plt.xlabel("Month")
                    plt.ylabel("Revenue")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.wh.cfg.out_dir / "chart_monthly_revenue_2024.png")
                    plt.close()

                    plt.figure()
                    plt.plot(forecast_rev["month"], forecast_rev["forecast_revenue"], marker="o")
                    plt.title("2025 Forecast: Monthly Revenue (Proxy)")
                    plt.xlabel("Month")
                    plt.ylabel("Forecast Revenue")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.wh.cfg.out_dir / "chart_monthly_revenue_forecast_2025.png")
                    plt.close()

        return insights


# ---------------------------
# Orchestration
# ---------------------------

def run_pipeline(cfg: ETLConfig) -> Tuple[WarehouseBuilder, Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    cfg.ensure_dirs()

    wb = WarehouseBuilder(cfg)
    wb.load_inputs()

    # Build Dims
    wb.build_dim_date()
    wb.build_dim_channel()
    wb.build_dim_status()
    wb.build_dim_plan()
    wb.build_dim_user()

    # Build Facts
    wb.build_fact_play_session()
    wb.build_fact_subscription()
    wb.build_fact_payments()

    # Write
    wb.write_all()

    # Validations
    dq_df: Optional[pd.DataFrame] = None
    if cfg.run_validations:
        v = Validator()
        dq_df = v.run(wb)
        dq_df.to_csv(cfg.out_dir / "data_quality_checks.csv", index=False)

    # Insights
    ins = Insights(wb).analyze(write_charts=cfg.write_charts)
    for name, df in ins.items():
        df.to_csv(cfg.out_dir / f"{name}.csv", index=False)

    # Readme
    readme = f"""# Dice Game — Star Schema Warehouse\n\nGenerated by dice_game_etl.py\n\n## Tables\n- dim_date.*\n- dim_channel.*\n- dim_status.*\n- dim_plan.*\n- dim_user.*\n- fact_play_session.*\n- fact_subscription.*\n- fact_payments.*\n\n## Insights\n- sessions_by_channel_2024.csv\n- monthly_sessions_2024.csv\n- monthly_sessions_forecast_2025.csv\n- plan_mix_and_revenue_2024.csv\n- monthly_revenue_2024.csv\n- monthly_revenue_forecast_2025.csv\n\n## Charts (if enabled)\n- chart_sessions_by_channel_2024.png\n- chart_monthly_sessions_2024.png\n- chart_monthly_sessions_forecast_2025.png\n- chart_monthly_revenue_2024.png\n- chart_monthly_revenue_forecast_2025.png\n\n## Notes\n- Revenue is a proxy using plan cost in the subscription start month.\n- Forecasts are linear trends across 2024 monthly aggregates.\n"""
    (cfg.out_dir / "README.txt").write_text(readme)

    return wb, dq_df, ins


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> ETLConfig:
    ap = argparse.ArgumentParser(description="Dice Game ETL + Analytics (Star Schema)")
    ap.add_argument("--base-dir", default="/data", type=str, help="Directory containing input CSVs")
    ap.add_argument("--out-dir", default="/data/warehouse_star", type=str, help="Output directory for warehouse")
    ap.add_argument("--write-charts", action="store_true", help="If set, write PNG charts")
    ap.add_argument("--run-validations", action="store_true", help="If set, run data quality checks")
    args = ap.parse_args()
    return ETLConfig(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        write_charts=bool(args.write_charts),
        run_validations=bool(args.run_validations),
    )


def main() -> None:
    cfg = parse_args()
    wb, dq_df, ins = run_pipeline(cfg)

    # Console summary (concise)
    print("\n=== Pipeline Complete ===")
    print(f"Inputs loaded: {list(wb.raw.keys())}")
    print(f"Outputs written to: {wb.cfg.out_dir}")
    if dq_df is not None:
        passed = int(dq_df["passed"].sum())
        total = len(dq_df)
        print(f"Data quality checks passed: {passed}/{total}")
    # A couple headline metrics if available
    if "sessions_by_channel_2024" in ins:
        s = ins["sessions_by_channel_2024"].copy()
        print("\n2024 Sessions by Channel:")
        for _, r in s.iterrows():
            print(f" - {r['channel_name']}: {int(r['sessions_2024'])} sessions")
    if "monthly_revenue_2024" in ins:
        rev = ins["monthly_revenue_2024"]["revenue"].sum()
        print(f"\nGross Revenue (Proxy, 2024 starts): ${rev:,.2f}")


if __name__ == "__main__":
    main()
