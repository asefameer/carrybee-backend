CarryBee • Backend (FastAPI)
================================

This folder contains a deploy-ready FastAPI app with endpoints:
- /health
- /config (GET/PUT)
- /upload (Excel upload → scoring)
- /train (train/refresh model)
- /report (trends + pool breakdown)
- /predict (optional single-row prediction)

Deploy (free) on Render in minutes:
1) Create a Neon (or Supabase) Postgres and copy its connection string.
2) Create a new GitHub repo (web UI: New → Repository), then click "Upload files" and drag this entire folder in.
3) In Render → New + → Blueprint → select that repo (it reads render.yaml).
4) Set env var DATABASE_URL to your Postgres connection string.
5) Deploy. Your API will be at https://<something>.onrender.com  (free tier may cold start).

Then open https://<your-site>/docs for the interactive UI,
or use the single-file frontend we provided to call these endpoints.
