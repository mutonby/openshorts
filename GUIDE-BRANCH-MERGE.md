# 🌿 Branch Merge Guide — put OpenShorts+ on your `main`

Everything we built lives on the branch **`arena/019feea4-openshort`**.
Your `main` is still the old original code. This guide merges the branch into
`main` so your deploys (Vercel + Render) get the new zero-budget app.

> ⏱️ Takes ~2 minutes, phone-friendly, free. Do this **once**, before deploying.
> Your `main` has not been touched, so this merge is trivially clean (no
> conflicts expected).

---

## Option A — GitHub app on your phone (easiest) ✅

1. Open the **GitHub app** → open your repo **`tahsinxiao/openshort-`**.
2. Tap **Branches** (top bar, next to the branch name).
3. Find **`arena/019feea4-openshort`** → tap it → tap **New pull request**.
   (If you don't see "New pull request" inside the branch, tap the branch,
   then the **"Compare & pull request"** banner.)
4. GitHub shows: **base: `main`** ← **compare: `arena/019feea4-openshort`**.
   Leave it exactly like that.
5. Tap **Create pull request** (you can write "OpenShorts+ zero-budget" as
   the title — or leave the default).
6. Tap **Merge pull request** → **Confirm merge**.
7. *(Optional)* When asked **"Delete branch?"** — you can tap **Delete**; the
   code is safe on `main` now.

✅ Done. Your `main` now has the full OpenShorts+ app.

## Option B — GitHub website on your phone's browser

1. Open this exact link (it pre-fills everything):
   **https://github.com/tahsinxiao/openshort-/pull/new/arena/019feea4-openshort**
2. Confirm it says base **`main`** ← compare **`arena/019feea4-openshort`**.
3. Tap **Create pull request** → **Merge pull request** → **Confirm merge**.

Same result as Option A.

## How to verify it worked (30 seconds)

- Go to your repo → **Code** tab. The README should now say
  **"OpenShorts+ — Zero-Budget Edition"**.
- Check the file list contains **`ai_gateway.py`**, **`publish_kit.py`**,
  **`GUIDE-DEPLOY-PHONE.md`** and a **`dashboard/`** folder with a
  `vercel.json`. Those only exist in the new code.

If you still see the old README ("OpenShorts.app"), the merge didn't
complete — redo steps A4–A6 (or B2–B3).

## If GitHub says "can't automatically merge" (rare)

Your `main` hasn't changed since the fork, so this *shouldn't* happen. If it
does:

1. Don't panic — no data is lost.
2. On the PR page choose **Create a merge commit** (instead of squash/rebase).
3. If it still refuses, tell me and I'll resolve the conflict for you.

## Already deployed and seeing old code?

If you deployed Vercel/Render *before* merging, they're showing the old app.
After merging:

- **Render:** open your service → **Manual Deploy → Deploy latest commit**
  (or just wait — it auto-deploys on new `main` commits).
- **Vercel:** it auto-deploys on merge; if not, go to your project →
  **Deployments** → **Redeploy**.

## Computer version (if you ever use one)

```bash
git checkout main
git pull origin main
git fetch origin arena/019feea4-openshort
git merge origin/arena/019feea4-openshort --no-edit
git push origin main
```

---

### What's next after the merge

Follow **`GUIDE-DEPLOY-PHONE.md`** from **Step 2**:

1. Deploy the backend on **Render** (free, with a 1 GB disk at `/opt/data`).
2. Deploy the dashboard on **Vercel** (root directory `dashboard`,
   `VITE_API_URL` = your Render URL).
3. Open the app → **Settings → Free AI keys (server)** → paste keys
   (+ add new) → Save. Pick a caption theme.
4. Generate clips from any Kick/YouTube link → review → **publish kit** →
   copy title + description + hashtags → post yourself.
