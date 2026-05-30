# Garmin Running Dashboard

A Streamlit dashboard for amateur runners who want a clearer view of their Garmin running history. It turns a Garmin account export into practical charts for recent runs, weekly volume, heart-rate zones, VO2 max, training load, race predictions, and half-marathon planning.

The app can run in two ways:

- **Public website mode:** visitors upload their own Garmin export zip in the sidebar. The file is processed in memory for that session.
- **Local mode:** you place your own Garmin export zip in `data/raw/`, run the ETL, and use the dashboard from your computer.

## What You Get

- **Overview:** headline mileage, current training read, weekly volume, recent windows, and activity mix.
- **Recent Runs:** last-run analysis for pace, HR, hard-zone share, recovery gaps, load jumps, and what to adjust next.
- **Performance:** pace trends, pace vs HR, Garmin VO2 max history, race predictions, and race-like runs.
- **Training Load:** Garmin training load, long-run progression, heart-rate zones, and acute/chronic distance ratio.
- **Zones & VO2:** zone definitions, easy/recovery pace guidance, intensity by run type, and VO2 rebuild levers.
- **Half Marathon Plan:** configurable 8-12 week half-marathon block with target paces and long-run progression.
- **Cambridge 2027:** long-range 1:38 half-marathon readiness view and roadmap.
- **Data:** normalized run table, flagged outlier review, and CSV download.

## Privacy First

Garmin exports and generated processed data can contain sensitive personal information, including location history. Do **not** commit Garmin zip files or generated `data/processed/` files to a public GitHub repository.

This repo includes a `.gitignore` that excludes:

- `*.zip`
- `data/raw/*.zip`
- `data/processed/*.csv`
- `data/processed/*.json`

Before publishing, run:

```bash
git status --short
```

Only source files, `requirements.txt`, `.gitignore`, `.streamlit/config.toml`, `README.md`, and `data/raw/.gitkeep` should be staged for a public repo.

## Request Your Garmin Data

Use Garmin's Account Management data export, not the single-activity export from Garmin Connect.

1. Open Garmin Account Management: https://www.garmin.com/account/datamanagement/
2. Sign in with your Garmin account.
3. Choose **Manage Your Data**.
4. Choose the option to export/request your data.
5. Wait for Garmin's email with a download link. Export time can vary.
6. Download the zip file.
7. Do not unzip it.

For local use, put the downloaded `.zip` file here:

```text
data/raw/
```

Example:

```text
data/raw/garmin-export.zip
```

If there is more than one zip in `data/raw/`, the app uses the newest one.

## Run Locally

### 1. Install Python

Use Python 3.12 if possible. Python 3.10 or newer should also work.

Check your version:

```bash
python3 --version
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Garmin export

Put the Garmin export zip in:

```text
data/raw/
```

### 5. Build the processed data

```bash
python3 garmin_etl.py
```

You can also point to a specific zip:

```bash
python3 garmin_etl.py --zip data/raw/garmin-export.zip
```

### 6. Start the dashboard

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## Use Without Local Files

You can run the app and upload a Garmin export zip directly in the sidebar:

```bash
streamlit run app.py
```

Then use **Data source -> Garmin export .zip**. This is the recommended pattern for the public website because visitors should not need their data committed to the repository.

## Publish From `javcamposz` GitHub

This is a Streamlit app, so GitHub Pages is not the right host for the interactive dashboard. GitHub Pages hosts static HTML/CSS/JavaScript files; it does not run a Python Streamlit server. Use GitHub for the public source repo, then deploy the app with Streamlit Community Cloud.

Official references:

- GitHub local project import: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- GitHub Pages overview: https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages
- Streamlit Community Cloud deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

### 1. Make sure private data is not staged

```bash
git status --short
```

If this folder is not already a Git repo, initialize it:

```bash
git init -b main
```

Stage only the public-safe files:

```bash
git add .gitignore .streamlit/config.toml README.md app.py garmin_etl.py requirements.txt data/raw/.gitkeep
git commit -m "Prepare public Garmin running dashboard"
```

### 2. Create the GitHub repo

With GitHub CLI:

```bash
gh repo create javcamposz/garmin-running-dashboard --public --source=. --remote=origin --push
```

Without GitHub CLI:

1. Go to https://github.com/new
2. Owner: `javcamposz`
3. Repository name: `garmin-running-dashboard`
4. Visibility: **Public**
5. Do not add a README, license, or gitignore on GitHub.
6. Create the repository.
7. Copy the repository URL and run:

```bash
git remote add origin REPLACE_WITH_THE_GITHUB_REPO_URL
git push -u origin main
```

### 3. Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub.
3. Select **Create app**.
4. Choose the `javcamposz/garmin-running-dashboard` repository.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Choose a public app URL.
8. Deploy.

The public app should open without personal data. Each runner can upload their own Garmin export zip in the sidebar.

## Refresh Your Data Locally

When Garmin sends a newer export:

1. Delete or move the old zip from `data/raw/`.
2. Put the new zip in `data/raw/`.
3. Run:

```bash
python3 garmin_etl.py
streamlit run app.py
```

You can also press **Rebuild Garmin data** in the app sidebar when running locally.

## Troubleshooting

### No running activities found

Make sure you used the full Garmin account export zip from Account Management. A single-activity CSV export is not enough for this app.

### The dashboard opens but has no VO2 max, training load, or race predictions

Some Garmin accounts or devices do not include every metric. The dashboard still works with run-level activity data.

### Upload is slow

Large Garmin exports can take time to parse. Wait for Streamlit to finish processing the file.

### The public site shows no data

That is expected until a visitor uploads a Garmin export zip. Personal data should not live in the public GitHub repository.

## Project Structure

```text
app.py                 Streamlit dashboard
garmin_etl.py          Garmin export parser and processed-data builder
requirements.txt       Python dependencies
.streamlit/config.toml Streamlit theme and upload limit
data/raw/.gitkeep      Empty folder placeholder for local Garmin zip files
data/processed/        Local generated outputs, ignored by Git
```

## Training Disclaimer

The dashboard provides training guidance from Garmin activity data. It is not medical advice. Reduce load or seek professional advice for unusual pain, illness, persistent fatigue, or health concerns.
