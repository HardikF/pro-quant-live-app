# How to Host ProQuant 24/7 for Free (Always Visible on Phone & PC)

You can host this web app for **free** on **Streamlit Community Cloud** so you get a permanent website link (e.g., `https://your-proquant.streamlit.app`) that you can open anytime on your phone or laptop without keeping your PC running.

## One-Time Hosting Steps (5 Minutes, Zero Coding)
1. Create a free account on **GitHub** (`https://github.com`).
2. Click **+** (top right) → **New repository** → name it `proquant-live` → select **Public** (or Private) → click **Create repository**.
3. Click **uploading an existing file**, drag and drop all files from inside this unzipped folder (`app.py`, `config.py`, `backtester.py`, `requirements.txt`, and the `engines/` folder) → click **Commit changes**.
4. Go to **Streamlit Community Cloud** (`https://share.streamlit.io`) and click **Sign in with GitHub**.
5. Click **Create app** → choose your `proquant-live` repository → set **Main file path** to `app.py` → click **Deploy**.
6. In ~2 minutes, your permanent live URL will open. Bookmark it on your phone's home screen!
