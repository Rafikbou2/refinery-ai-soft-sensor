"""REST endpoint so the DCS/historian (or a PI/OPC-UA bridge) can push data.

    uvicorn api:app --port 8000
    POST /predict   {"records": [{"timestamp": "...", "feed_rate": 450, ...}]}
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from softsensor import SoftSensor
from softsensor.config import FEATURES, MANIPULATED, TARGETS
from softsensor.optimizer import recommend
from softsensor.risk import risk_level

sensor = SoftSensor.load(Path(__file__).parent / "models" / "soft_sensor.joblib")
app = FastAPI(title="Refinery AI soft sensor", version="1.0")


class Batch(BaseModel):
    records: list[dict] = Field(..., min_length=1)


def _frame(batch: Batch) -> pd.DataFrame:
    df = pd.DataFrame(batch.records)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(422, f"missing tags: {missing}")
    if "timestamp" in df:
        df = df.set_index(pd.to_datetime(df.pop("timestamp")))
    return df


@app.get("/health")
def health():
    return {"status": "ok", "targets": TARGETS, "inputs": FEATURES, "metrics": sensor.metrics}


@app.post("/predict")
def predict(batch: Batch):
    df = _frame(batch)
    pred = sensor.predict(df)
    out = []
    for ts, row in pred.iterrows():
        item = {"timestamp": str(ts), "anomaly": bool(row["anomaly"]),
                "extrapolating": bool(row["extrapolating"])}
        for t in TARGETS:
            p = float(row[f"{t}_p_off"])
            item[t] = {"value": round(float(row[f"{t}_pred"]), 2),
                       "interval_90": [round(float(row[f"{t}_lo"]), 2), round(float(row[f"{t}_hi"]), 2)],
                       "p_offspec": round(p, 4), "risk": risk_level(p)}
        out.append(item)
    return {"predictions": out}


@app.post("/recommend")
def recommend_setpoints(batch: Batch, max_p_offspec: float = 0.05):
    current = _frame(batch).iloc[-1][FEATURES].astype(float)
    rec = recommend(sensor, current, max_p_off=max_p_offspec)
    return {"status": rec["status"], "gain_usd_day": round(rec["gain_usd_day"], 1),
            "setpoints": {t: round(float(rec["setpoints"][t]), 2) for t in MANIPULATED},
            "current": {t: round(float(rec["current"][t]), 2) for t in MANIPULATED}}
