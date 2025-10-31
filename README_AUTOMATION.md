# Automation additions — Retrain + Full pipeline

New files added:
- retrain.py : checks data checksum and runs train_model.py when data changes. Saves timestamped models under `models/`.
- pipeline.sh : one-shot script to run install, retrain, and build Docker image.
- deploy.sh : simple docker run wrapper to deploy locally.
- Makefile : convenience commands (install, retrain, build, run, deploy).
- .github/workflows/retrain.yml : GitHub Actions workflow to trigger retrain on data changes or schedule.

How to run locally (commands):
1. Install dependencies:
   python -m pip install -r requirements.txt

2. Run full pipeline (retrain + build):
   ./pipeline.sh

3. To manually trigger retrain:
   python retrain.py

4. To run app locally after build:
   docker run -p 5000:5000 cicd-sales-prediction:latest

Notes:
- The GitHub Actions workflow will only push images if you set repository secrets:
  REGISTRY_IMAGE, REGISTRY_HOST, REGISTRY_USERNAME, REGISTRY_PASSWORD.
- retrain.py uses a checksum file at data/last_retrain.txt to detect changes.
