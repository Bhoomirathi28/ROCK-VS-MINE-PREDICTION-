# Deployment guide

The app is a standard Python web service, so it runs anywhere Python or Docker runs. Pick one option; Render is the easiest free public URL for the demo.

## Option A: Render (free public URL, recommended)

1. Push this repository to GitHub.
2. Sign in at render.com with GitHub → **New +** → **Blueprint** → select the repository. Render reads `render.yaml` and configures everything.
3. Wait for the build (about 3 minutes). Your app is live at `https://rock-vs-mine-xxxx.onrender.com`.
4. Paste that URL into the README table and the report.

Free instances sleep after 15 minutes idle; the first request after that takes around 30 seconds. Open the link a minute before the demo.

Manual setup instead of the blueprint: **New Web Service** → build command `pip install -r requirements.txt` → start command `gunicorn app.app:app --bind 0.0.0.0:$PORT`.

## Option B: Railway or any Heroku-style host

The `Procfile` and `runtime.txt` are picked up automatically. Connect the repo and deploy.

## Option C: Docker (any operating system)

```bash
docker build -t rock-vs-mine .
docker run -p 5000:5000 rock-vs-mine
```

Open http://localhost:5000.

## Option D: Local machine, reachable from your phone

```bash
python app/app.py
```

The server listens on all interfaces. Find your laptop's IP (`ipconfig` on Windows, `ifconfig` or `ip a` on macOS/Linux) and open `http://<that-ip>:5000` on a phone connected to the same Wi-Fi. Useful if the venue's internet is unreliable during the demo.

## Installing as an app

The site ships a web app manifest and a service worker, so browsers offer to install it:

* **Android, Chrome, Edge (desktop):** browser menu → *Install app*.
* **iOS / iPadOS:** Safari → Share → *Add to Home Screen*.

The interface itself is cached, so it opens even without a connection. Predictions still need the server, because the models run there.

## Checking a deployment

```bash
curl https://<your-url>/api/health
# {"model":"svm","status":"ok"}
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: src` | Start from the repository root (`gunicorn app.app:app`, not from inside `app/`). |
| Model loading warning about scikit-learn version | Install the exact versions in `requirements.txt`, or delete `models/*.joblib` and run `python -m src.train`. |
| Port already in use locally | `PORT=8000 python app/app.py` (Windows PowerShell: `$env:PORT=8000; python app/app.py`). |
