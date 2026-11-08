# SUBMIT CHECKLIST — BITS Pilani Digital, Taxi Forecasting MLOps Capstone
# Deadline: TODAY Sun Nov 8, 2026 — extended, NO further extensions.
# Assume 11:59 PM IST = 12:29 PM CST. Aim to submit by 11:30 AM CST.

## 1. Push the repo FIRST (blocked on Varun's GitHub auth — see PUSH_COMMANDS.md)
- [ ] `gh auth login` (or sign in to GitHub in the browser)
- [ ] Create PUBLIC repo `taxi-forecasting-mlops-capstone`, push the
      capstone-starter folder, copy the repo URL and the commit ID
      (git log -1 --format=%H, or the commit hash shown on GitHub).
- [ ] Open the repo in a browser while LOGGED OUT / in a private window
      and confirm it is public and the files are visible.

## 2. Record the video (see VIDEO_SCRIPT.md, ~11 min, hard limit 12 min)
- [ ] Webcam intro FIRST: full name "Varun Arikatla" + BITS ID 2026a1c100024
- [ ] Full-screen capture, show repo + commit ID on screen
- [ ] No secrets / passwords / tokens on screen at any point
- [ ] File: mp4, 10–12 minutes. Check duration before uploading — over 12 min fails.

## 3. Submit on Lumen (lumen.bitspilani-digital.edu.in)
- [ ] Sign in via "Login using BITS Mail" (Google SSO, BITS ID
      2026a1c100024@bitspilani-digital.edu.in). Do NOT use the direct-password
      login — it was rejected before.
- [ ] Open the course C1-26_AIEOPC151, tab "C1 AIEMLOps Capstone",
      assignment "Capstone Project Submission".
- [ ] Upload the mp4 video file.
- [ ] Upload / paste the repo link as a .txt file link (per the brief:
      mp4 + txt link). The txt file must contain the public repo URL.
- [ ] Submit, then screenshot / save the submission confirmation page.
- [ ] If Lumen says "already submitted", stop — do not submit twice.

## 4. What is in this submission (for the record)
- Best model: LightGBM, val RMSE 31.00 / MAE 19.91, test RMSE 31.45 / MAE 20.45
- MLflow run: e38677e286a64196b5eca00998c86197
- Data: NYC TLC yellow taxi, Jan 2024, 2,964,624 trips, top-20 zones
- Fairness: Manhattan 29.90 / Queens 43.06, gap 13.16; mitigation did NOT
  reduce the gap (13.32 after) — reported honestly.
- Docker: NOT verified (Docker not installed on the build machine).
- Monitoring/drift: plan only, in MONITORING.md — no live AWS deployment.

## If anything fails
- Lumen won't load / SSO fails: try a fresh browser, sign in to BITS Mail
  first, then open Lumen. Do not keep retrying the password login.
- Video over 12 min: re-record the last section only if your editor allows
  a clean cut, otherwise do a fresh single take and speak faster in
  Phases 6–7.
- Repo push fails: submit the video with the repo link as soon as the repo
  is public — do not submit a private-repo link.
