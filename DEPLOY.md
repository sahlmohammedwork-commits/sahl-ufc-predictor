# Deploying Sahl UFC Predictor to Render (Free)

This guide walks you through getting Sahl on a public URL like `sahl-ufc-predictor.onrender.com` for **free**, in about 10 minutes.

## What you'll need
- A free GitHub account (https://github.com/signup)
- A free Render account (https://render.com — sign up with GitHub, no credit card needed)

## Step 1 — Put the code on GitHub

1. Go to https://github.com/new
2. Repository name: `sahl-ufc-predictor`
3. Set it to **Public** (required for Render's free tier auto-deploy)
4. Click **Create repository**
5. On the next page, scroll to **"…or push an existing repository from the command line"**

In the unzipped `sahl-ufc-predictor` folder, open Terminal/Command Prompt and run the commands GitHub shows you. They'll look like this (use the URL from YOUR repo, not this one):

```bash
cd path/to/sahl-ufc-predictor
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sahl-ufc-predictor.git
git push -u origin main
```

If you don't have git installed, download it from https://git-scm.com/

## Step 2 — Deploy on Render

1. Go to https://dashboard.render.com
2. Click **New +** → **Blueprint**
3. Connect your GitHub account (one-time)
4. Pick the `sahl-ufc-predictor` repo
5. Render reads the `render.yaml` automatically and shows you the service it'll create
6. Click **Apply**
7. Wait ~5 minutes — Render installs Python deps, generates synthetic data, trains the model, and starts the server

When the status turns green, click the URL at the top. That's your public site!

The URL will look like: `https://sahl-ufc-predictor.onrender.com`

## ⚠️ Important: Free tier behavior

- **Sleeps after 15 min idle.** First visitor after a quiet period waits ~30-60 seconds for the server to wake up. Subsequent requests are instant.
- **750 free hours/month.** Enough to run 24/7 if you only have one service.
- **No credit card required.**

## Updating Sahl later

Whenever you want to push changes (new model, frontend tweaks, real scraped data):

```bash
git add .
git commit -m "describe what changed"
git push
```

Render auto-deploys on every push. ~3-5 minutes for a new version to go live.

## Training on real UFC data after deploy

The default deploy uses synthetic data so it boots fast. To train on real UFC data:

**Option A — locally, then push:**
```bash
cd backend
python scraper.py --years 10
python features.py
python train.py
git add ../models ../data
git commit -m "Real UFC data trained"
git push
```

**Option B — let Render do it** (slower deploys but cleaner):
Edit `render.yaml`'s `buildCommand` to:
```yaml
buildCommand: |
  pip install -r requirements.txt
  cd backend && python scraper.py --years 10 && python features.py && python train.py
```
This makes deploys take ~1 hour because of the scrape. Only do this if you want fully fresh data on every deploy.

## Custom domain (optional, free)

Render gives you a free `*.onrender.com` URL. If you own a domain (e.g. `sahl.com`):
1. Render dashboard → your service → **Settings** → **Custom Domains**
2. Add your domain
3. Update DNS at your registrar with the records Render shows you
4. SSL is automatic and free

## Troubleshooting

| Problem | Fix |
|---|---|
| "Build failed: Out of memory" | Free tier has 512 MB. Comment out `import shap` in `train.py`'s SHAP block — it's the heaviest dependency. |
| Site is slow to first load | Normal — free tier wakes up on demand. Upgrade to paid ($7/mo) for always-on. |
| Upcoming card endpoint hangs | Some scrapers get blocked from Render's IP range. Test with manual matchups. |
| 502 Bad Gateway briefly after deploy | Wait 30 seconds — server is still starting. |
