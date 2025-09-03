
import io
import json
import os
import re
import joblib
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ------------------------- env & engine -------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    # You should set DATABASE_URL to a Postgres URL in production.
    # SQLite fallback is for local quick tests only and may not support JSONB schema.
    DATABASE_URL = "sqlite+pysqlite:///:memory:?cache=shared"

engine: Engine = create_engine(DATABASE_URL, future=True)

# ------------------------ schema bootstrap ----------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS configs (
  id SERIAL PRIMARY KEY,
  version TEXT NOT NULL,
  body JSONB NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS uploads (
  id SERIAL PRIMARY KEY,
  filename TEXT NOT NULL,
  uploaded_at TIMESTAMP NOT NULL DEFAULT NOW(),
  sheet_names JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
  id SERIAL PRIMARY KEY,
  upload_id INTEGER NOT NULL,
  pool TEXT NOT NULL,
  sheet TEXT NOT NULL,
  row_index INTEGER NOT NULL,
  raw DOUBLE PRECISION NOT NULL,
  adjusted DOUBLE PRECISION NOT NULL,
  normalized DOUBLE PRECISION NOT NULL,
  sentiment TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_data (
  id SERIAL PRIMARY KEY,
  upload_id INTEGER NOT NULL,
  pool TEXT NOT NULL,
  sheet TEXT NOT NULL,
  features JSONB NOT NULL,
  target DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS models (
  id SERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  metrics JSONB NOT NULL,
  artifact BYTEA NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

with engine.begin() as conn:
    # Split by statement so it works on providers that dislike multi-queries
    for stmt in SCHEMA_SQL.split(";\n"):
        s = stmt.strip()
        if s:
            try:
                conn.execute(text(s))
            except Exception:
                # SQLite fallback schema (loosest). For production, use Postgres.
                if "JSONB" in s:
                    s = s.replace("JSONB", "TEXT").replace("NOW()", "CURRENT_TIMESTAMP").replace("SERIAL", "INTEGER")
                try:
                    conn.execute(text(s))
                except Exception:
                    pass

# -------------------------- utils: scoring ----------------------
def _norm_txt(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _find_sentiment_col(columns: List[str]) -> Optional[str]:
    for c in columns:
        t = _norm_txt(c)
        if "sentiment" in t and "customer" in t:
            return c
    for c in columns:
        if "sentiment" in _norm_txt(c):
            return c
    return None

class Option(BaseModel):
    id: str
    label: str
    weight: float

class Question(BaseModel):
    id: str
    text: str
    questionWeight: float
    type: str  # "categorical" | "rating"
    options: List[Option]

class Pool(BaseModel):
    id: str
    name: str
    poolWeight: float
    questions: List[Question]

class Config(BaseModel):
    version: str
    pools: List[Pool]
    sentimentAdjust: Dict[str, Any]

def calc_pool_min_max(pool: Pool) -> (float, float):
    min_sum, max_sum = 0.0, 0.0
    for q in pool.questions:
        if not q.options:
            continue
        weights = [float(o.weight) for o in q.options]
        min_sum += q.questionWeight * min(weights)
        max_sum += q.questionWeight * max(weights)
    return pool.poolWeight * min_sum, pool.poolWeight * max_sum

def score_row(row: pd.Series, pool: Pool, sentiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    pool_w = float(pool.poolWeight)
    raw = 0.0
    contribs = []

    headers = list(row.index)
    header_norm_map = {_norm_txt(h): h for h in headers}

    for q in pool.questions:
        qw = float(q.questionWeight)
        col = q.text if q.text in headers else header_norm_map.get(_norm_txt(q.text))
        if col is None:
            continue
        val = row[col]
        ow = 0.0
        chosen = None
        if pd.isna(val):
            ow = 0.0
        else:
            chosen = str(val)
            if q.type == "rating":
                try:
                    ow = float(chosen)
                except Exception:
                    mp = {o.label: float(o.weight) for o in q.options}
                    ow = mp.get(chosen, 0.0)
            else:
                mp = {o.label: float(o.weight) for o in q.options}
                if chosen in mp:
                    ow = mp[chosen]
                else:
                    nmap = {_norm_txt(k): v for k, v in mp.items()}
                    ow = nmap.get(_norm_txt(chosen), 0.0)
        part = pool_w * qw * ow
        raw += part
        contribs.append({
            "question": q.text,
            "column": col,
            "chosen": chosen,
            "option_weight": ow,
            "contribution": part,
        })

    sentiment_col = _find_sentiment_col(headers)
    factor = 0.0
    used = None
    if sentiment_col and sentiment_cfg.get("enabled", True):
        sval = row.get(sentiment_col)
        if isinstance(sval, str):
            s = _norm_txt(sval)
            if "positive" in s:
                factor = float(sentiment_cfg.get("positive", 0.0))
                used = "positive"
            elif "negative" in s:
                factor = float(sentiment_cfg.get("negative", 0.0))
                used = "negative"
            else:
                factor = float(sentiment_cfg.get("neutral", 0.0))
                used = "neutral"

    adjusted = raw * (1 + factor)
    pmin, pmax = calc_pool_min_max(pool)
    norm = 0.0
    if pmax > pmin:
        norm = (adjusted - pmin) / (pmax - pmin) * 100
        norm = max(0.0, min(100.0, norm))

    return {
        "raw": raw,
        "adjusted": adjusted,
        "normalized": norm,
        "sentiment_used": used,
        "sentiment_factor": factor,
        "contributions": contribs,
    }

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def extract_target(row: pd.Series) -> Optional[float]:
    for c in row.index:
        t = _norm_txt(c)
        if ("overall" in t and "experience" in t):
            try:
                val = float(row[c])
                return val
            except Exception:
                continue
    return None

def engineer_features(row: pd.Series, pool: Pool) -> Dict[str, float]:
    headers = list(row.index)
    header_norm_map = {_norm_txt(h): h for h in headers}
    feats = {}
    for q in pool.questions:
        col = q.text if q.text in headers else header_norm_map.get(_norm_txt(q.text))
        if col is None:
            continue
        val = row[col]
        val_w = 0.0
        if pd.isna(val):
            val_w = 0.0
        else:
            chosen = str(val)
            if q.type == "rating":
                try:
                    val_w = float(chosen)
                except Exception:
                    mp = {o.label: float(o.weight) for o in q.options}
                    val_w = mp.get(chosen, 0.0)
            else:
                mp = {o.label: float(o.weight) for o in q.options}
                if chosen in mp:
                    val_w = mp[chosen]
                else:
                    nmap = {_norm_txt(k): v for k, v in mp.items()}
                    val_w = nmap.get(_norm_txt(chosen), 0.0)
        feats[f"feat::{q.text}"] = float(pool.poolWeight) * float(q.questionWeight) * val_w
    return feats

app = FastAPI(title="CarryBee API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.get("/config")
def get_config():
    with engine.begin() as conn:
        row = conn.execute(text("SELECT body FROM configs ORDER BY id DESC LIMIT 1")).fetchone()
        if not row:
            raise HTTPException(404, "No config set yet")
        return row[0]

@app.put("/config")
def put_config(config_json: str = Form(...)):
    try:
        body = json.loads(config_json)
        # basic validation
        Config(**body)
    except Exception as e:
        raise HTTPException(400, f"Invalid config JSON: {e}")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO configs(version, body) VALUES (:v, :b)"),
                     {"v": body.get("version", "v1"), "b": json.dumps(body)})
    return {"saved": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT body FROM configs ORDER BY id DESC LIMIT 1")).fetchone()
        if not row:
            raise HTTPException(400, "Upload rejected: please set config first via /config")
        cfg = Config(**row[0])

    content = await file.read()
    try:
        xls = pd.ExcelFile(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid Excel: {e}")

    sheet_names = xls.sheet_names
    with engine.begin() as conn:
        r = conn.execute(text("INSERT INTO uploads(filename, sheet_names) VALUES (:f,:s) RETURNING id"),
                         {"f": file.filename, "s": json.dumps(sheet_names)}).fetchone()
        upload_id = r[0]

    all_rows = []
    td_rows = []

    for pool in cfg.pools:
        cand = None
        pname = pool.name.lower()
        for s in sheet_names:
            if all(t in s.lower() for t in re.split(r"\W+", pname) if t):
                cand = s; break
        if cand is None:
            for key in ["successful", "failed", "return", "pickup", "delivery"]:
                if key in pname:
                    cand = next((s for s in sheet_names if key in s.lower()), None)
                    if cand: break
        if cand is None:
            continue

        df = pd.read_excel(io.BytesIO(content), sheet_name=cand)
        for ridx, row in df.iterrows():
            sres = score_row(row, pool, cfg.sentimentAdjust)
            all_rows.append({
                "upload_id": upload_id,
                "pool": pool.name,
                "sheet": cand,
                "row_index": int(ridx),
                "raw": float(sres["raw"]),
                "adjusted": float(sres["adjusted"]),
                "normalized": float(sres["normalized"]),
                "sentiment": sres["sentiment_used"],
            })

            feats = engineer_features(row, pool)
            target = extract_target(row)
            if target is None:
                target = float(sres["normalized"])
            td_rows.append({
                "upload_id": upload_id,
                "pool": pool.name,
                "sheet": cand,
                "features": json.dumps(feats),
                "target": float(target),
            })

    with engine.begin() as conn:
        for r in all_rows:
            conn.execute(text(
                "INSERT INTO scores(upload_id,pool,sheet,row_index,raw,adjusted,normalized,sentiment) "
                "VALUES (:upload_id,:pool,:sheet,:row_index,:raw,:adjusted,:normalized,:sentiment)"
            ), r)
        for r in td_rows:
            conn.execute(text(
                "INSERT INTO training_data(upload_id,pool,sheet,features,target) "
                "VALUES (:upload_id,:pool,:sheet,:features,:target)"
            ), r)

    return {"ok": True, "upload_id": upload_id, "scored_rows": len(all_rows)}

@app.post("/train")
def train():
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT features, target FROM training_data")).fetchall()
    if not rows:
        raise HTTPException(400, "No training data yet. Upload first.")

    feat_keys = set()
    parsed = []
    for fjson, tgt in rows:
        obj = json.loads(fjson)
        parsed.append((obj, float(tgt)))
        feat_keys |= set(obj.keys())
    feat_keys = sorted(list(feat_keys))

    import numpy as np
    X = np.array([[p.get(k, 0.0) for k in feat_keys] for p, _ in parsed], dtype=float)
    y = np.array([t for _, t in parsed], dtype=float)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    rfr = RandomForestRegressor(n_estimators=200, random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)
    rfr.fit(X_tr, y_tr)
    gbr.fit(X_tr, y_tr)

    from sklearn.metrics import mean_absolute_error
    rfr_mae = mean_absolute_error(y_va, rfr.predict(X_va))
    gbr_mae = mean_absolute_error(y_va, gbr.predict(X_va))

    best, name, mae = (rfr, "RandomForest", rfr_mae) if rfr_mae <= gbr_mae else (gbr, "GradientBoosting", gbr_mae)

    art = io.BytesIO()
    joblib.dump({"model": best, "features": feat_keys}, art)
    artifact_bytes = art.getvalue()

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO models(model_name, metrics, artifact) VALUES (:n, :m, :a)"
        ), {"n": name, "m": json.dumps({"val_mae": mae}), "a": artifact_bytes})

    return {"trained": True, "model": name, "val_mae": mae}

@app.get("/report")
def report():
    with engine.begin() as conn:
        trend = conn.execute(text(
            """
            SELECT u.id as upload_id,
                   CAST(u.uploaded_at AS TEXT) as date,
                   AVG(s.normalized) as avg_norm
            FROM uploads u
            JOIN scores s ON s.upload_id = u.id
            GROUP BY u.id, date
            ORDER BY u.id DESC
            LIMIT 8
            """
        )).mappings().all()

        last_up = conn.execute(text("SELECT id FROM uploads ORDER BY id DESC LIMIT 1")).fetchone()
        by_pool = []
        if last_up:
            by_pool = conn.execute(text(
                "SELECT pool, AVG(normalized) as avg_norm, COUNT(*) as n "
                "FROM scores WHERE upload_id = :u GROUP BY pool ORDER BY pool"
            ), {"u": last_up[0]}).mappings().all()

    return {"trend": list(trend), "pool_breakdown": list(by_pool)}

class PredictPayload(BaseModel):
    pool_name: str
    row: Dict[str, Any]

@app.post("/predict")
def predict(payload: PredictPayload):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT artifact FROM models ORDER BY id DESC LIMIT 1")).fetchone()
        cfgrow = conn.execute(text("SELECT body FROM configs ORDER BY id DESC LIMIT 1")).fetchone()
    if not row or not cfgrow:
        raise HTTPException(400, "Model or config missing")

    bundle = joblib.load(io.BytesIO(row[0]))
    model = bundle["model"]
    feat_keys = bundle["features"]
    cfg = Config(**cfgrow[0])

    pool = next((p for p in cfg.pools if _norm_txt(p.name) == _norm_txt(payload.pool_name)), None)
    if not pool:
        raise HTTPException(400, "Unknown pool")

    series = pd.Series(payload.row)
    feats = engineer_features(series, pool)

    import numpy as np
    X = [[feats.get(k, 0.0) for k in feat_keys]]
    pred = float(model.predict(np.array(X))[0])
    return {"prediction": pred}
