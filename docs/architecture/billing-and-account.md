# Billing flows, lifecycle emails and account erasure

## What a minute pays for (`config.billable_minutes`, `EDITOR_ACTION_MINUTES`)

Minutes pay for the source a user imports: whole minutes **rounded down**, at
least 1 (a 4.6-min video costs 4). The download's safety cap
(`SOURCE_CAP_MINUTES`) still covers the whole file, so it uses the probed length
rounded **up**. Editing a clip that exists is free: AI edit and effects, the
hook overlay, recut/rerender and reframe reserve `EDITOR_ACTION_MINUTES` (0), and
`reserve_managed_action` bounds 0-minute actions (captions included) with
`EDITOR_ACTIONS_PER_HOUR` per user instead (429). AI edit and AI effects send
the clip to Gemini on the managed key, so only `AI_EDITS_FREE_PER_DAY` (5) a day
per user are free and each later one costs `MANAGED_ANALYSIS_MINUTES` again
(`app._ai_edit_minutes`, in memory per UTC day). Thumbnail image generation,
thumbnail transcription/analysis and SaaS-shorts analysis keep their fixed cost.

## Cost line and picked stretch (`/api/process/estimate`, `SourceCost.jsx`)

Under the source input the dashboard says what the video will cost before it
runs. `POST /api/process/estimate` takes `{url}` (probed like the submit, but
never on the per-GB proxy, in its own hourly probe window, and cached per user
for 15 min so the submit does not probe again) or `{seconds}` (a local file the
browser already measured), and answers `estimate_outcome`: `whole`,
`first_video_free`, `first_part` or `needs_more`, mirroring
`reserve_process_minutes` without reserving. "Clip only part of it" opens a
from/to picker; `/api/process` then takes `source_start` / `source_end`
(seconds). `plan_source_range` validates them (at least `MIN_RANGE_SECONDS`; the
whole video is an ordinary job), only that stretch is billed (no free overflow,
no partial offer on its 402), and the job is a partial with `start_seconds`,
`end_seconds` and `chosen: true`. `_apply_source_cut` sets
`MAX_SOURCE_MINUTES` to the exact length and `SOURCE_START_SECONDS`; main.py
downloads up to the stretch's end and `cap_source_duration` cuts from the start,
leaving a `.range` stamp so a resumed job does not cut the cut file again. The
"first N minutes" banners stay off for a chosen stretch, and clip times seek the
source preview shifted by its start.

## Quota wall offers the first N minutes (`app.partial_offer`)

The wall (`TopUpModal`, context `wall`) opens on a 402 from `/api/process`. The
402 carries `partial_minutes` (the floored balance, when it is at least
`PARTIAL_MIN_MINUTES` and shorter than the source), so a user with free minutes
left is offered "clip the first N min" next to the plans instead of being asked
to pay before seeing a clip. The dashboard resubmits the same job with
`max_minutes=N`. `reserve_process_minutes` reserves N (never more than the
balance, whatever the client asks) and sets `MAX_SOURCE_MINUTES` for `main.py`,
whose `cap_source_duration` cuts the downloaded/uploaded file **in place** before
anything reads it, so transcription, the layout picker, the editor and
`/api/source` all see a short video. The cut travels in the resume manifest: a
resumed job downloads the source again and would otherwise process the full
length on a short reservation. `/api/status` and the process response carry
`partial`, and the results view says which part of the video the clips came
from, with the upsell for the rest.

## Free sources past the balance never meet the wall (`app.free_overflow`)

The wall only shows to paid plans and to free accounts with less than
`PARTIAL_MIN_MINUTES` left. For a free account whose source is longer than its
balance, `reserve_process_minutes` decides on its own (the client sends no
`max_minutes`):

- **First video, up to `FIRST_VIDEO_MAX_MINUTES` (60):** clipped whole. Only the
  floored balance is reserved (it lands at zero) and the download's safety cap
  (`SOURCE_CAP_MINUTES`) is the probed length, not the reservation. Once per
  account (`metering.has_processed_before`: any reserved/committed `process` row;
  a released one keeps the grant for the retry), and once per client IP every
  `FIRST_VIDEO_IP_WINDOW_DAYS` (30): `first_video_grants` holds an HMAC of the IP
  (never the IP), is not tied to the user row so account deletion does not reset
  it, and only counts grants whose job reservation is live. A blocked network
  gets the first-N-minutes cut. The response carries `first_video: true`; the
  dashboard says so and tracks `FirstVideoGrant`.
- **Anything else:** `max_minutes` becomes the balance, so the job clips the
  first N minutes as if the user had taken the wall's offer. Tracked
  client-side as `AutoPartial`.

## Paths to a plan (`TopUpModal.jsx`, `PricingSection.jsx`)

- Every surface that opens the plan modal passes a `source` (`meter`, `menu`, `results_banner`, `partial_banner`, `partial_processing`, `first_video_banner`, `watermark`, `autopilot`, `queue`, `thumbnails`, `clip_editor`, `autopilot_run`); it rides on `UpsellModalSeen`/`QuotaWallSeen` and on the checkout events, so each surface can be measured on its own.
- A 402 outside the main clip flow calls `lib/api.openQuotaWall(err, source)`; the app shell listens for it and opens the quota wall instead of an error string.
- The wall's "this video is N min long" copy only applies when the user still has at least a minute: the pre-probe 402 sends a floor of 1 for `minutes_required`, which is not a duration.
- A plan picked while signed out is stashed (`lib/pendingPlan.js`, two-hour TTL). `#/pricing` opens the sign-in at once, AuthContext sends the new session back to `#/pricing`, and the section opens that checkout (`CheckoutStarted` with `resumed=1`).
- `#/pricing?plan=<plan>[&interval=year]` opens that plan's checkout (through the sign-in when signed out) and drops the param afterwards.
- Emails link to `#/pricing?src=email_<kind>`; checkouts from there are tagged `pricing_email_<kind>`. In nginx, `/pricing?<query>` redirects to the picker at `/#/pricing?<query>`; a bare `/pricing` goes to the homepage plan table (`/#pricing`).

## Lifecycle emails (`cloud/lifecycle.py`)

Welcome (minutes after sign-up), first-clip nudge (24-72 h, nothing processed),
win-back (2-7 days after the first committed video, no plan; with
`WINBACK_PROMO_CODE` when set) and checkout recovery (`checkout.session.expired`
with the `after_expiration.recovery` URL that `create_checkout` enables; sessions
keep Stripe's default 24 h expiry (a 1 h expiry was tried and reverted), and the
email names the plan or pack from the session metadata, `lifecycle.cart_label`). Each at
most once per account: a `lifecycle_emails` row is claimed before the send
(unique `user_id, kind`). The loop sends at most `BATCH` per 10-minute tick. All
four are commercial: `emails.send_commercial_email` skips `marketing_opt_out`
accounts and adds the unsubscribe footer and `List-Unsubscribe` headers (the
out-of-minutes upsell goes through it too). The Stripe webhook endpoint must
have `checkout.session.expired` enabled. Promotion codes and coupon ids live in
the env / Stripe, never in this repo.

Two more go to customers: `paid_idle` (a live paid plan 3+ days into its period
with nothing processed in it and 5+ days left, plans set to cancel included:
the minutes waiting plus Autopilot) and `comeback` (3+ days after the
subscription ended, free-plan accounts only: the free monthly minutes, no
discount). Both join the one-per-user subscription row and skip comp plans
(`sub_comp…`); `tests/test_lifecycle_emails.py` compiles every candidate query
for Postgres, since a correlation mistake only fails at run time.

## Cancel flow (`cloud/cancellation.py`, `CancelPlanModal.jsx`)

Account → "Cancel subscription" opens three steps: reason (closed list,
`CANCEL_REASONS`, plus an optional detail box), a 1-5 rating with an optional
review (and an "OK to quote publicly" box), then confirm.
`POST /api/billing/cancel` sets `cancel_at_period_end` in Stripe with
`cancellation_details` (our reason mapped to Stripe's feedback enum), and only
after Stripe accepts writes a `cancellation_feedback` row (user-owned, erased
with the account) and flips the local row, so the webhook sees no transition and
the generic churn alert does not fire twice. The alert names the reason and
rating, never the written text (see `alerts.user_ref`): read reviews in the
table.

The last step leads with a retention offer (`RETENTION_COUPON_ID` env):
`GET /api/billing/retention-offer` says whether this subscription gets it (live
monthly plan, no discount on it, never offered before: `retention_offer` in the
Stripe subscription metadata; any Stripe error means no offer),
`POST /api/billing/retention-offer/accept` applies it and stores the feedback row
with `outcome="retained"` (cancels store `"canceled"`). Cancelling is off in the
Stripe customer portal, so this flow is the only way to cancel;
`POST /api/billing/resume` undoes a scheduled cancel ("Keep my subscription" on
the account page). The webhook alert for a portal cancel stays for cancels made
from the Stripe Dashboard.

For `PAUSE_REASONS` (not using it, one-off project) the last step first offers
a month on hold (`POST /api/billing/pause`, monthly live plans, once per
subscription). It sets Stripe `pause_collection` (`behavior: void`,
`resumes_at`) and a hold window `[hold_from, hold_until)` in the subscription
metadata, where `hold_from` is the end of the paid period and `hold_until` is
`HOLD_DAYS` (27) later, shorter than any month. The webhook mirrors the window
to `subscriptions.hold_from/hold_until` and `config.subscription_status` turns a
period that starts inside it into `on_hold`: no plan minutes (free plan
instead), checkout blocked, library durable (`metering.is_free_user` is False),
`/api/me` reports `hold_until`. The next period starts past the window, Stripe
charges it and the row is `active` again (and the library is unmarked, as on
any upgrade). `POST /api/billing/unpause` drops a hold not yet started, or ends
a running one by charging a new period now (`billing_cycle_anchor=now`,
window closed at that instant). Feedback rows store `outcome="paused"`.

## Sign-up survey (`cloud/onboarding.py`, `OnboardingSurvey.jsx`)

`signup_attribution` says which page a user came from, not why. Accounts younger
than 7 days get one skippable screen before the clip tutorial (never over a
running job): what they want to make (multi-select), how they heard of us, and
who they are. Closed lists mirrored in the JSX (a test checks), stored in
`onboarding_surveys` (user-owned; a skip is a row too, so it is asked once), and
tracked as `SignupSurveyAnswered` with one `goal_<x>` prop per goal.

## Account erasure (GDPR art. 17, `cloud/account.py`)

`DELETE /api/account` (dashboard: Account → Delete account) is immediate and
irreversible: after the delete there is nothing left to authenticate a recovery
request against. It refuses API-key auth (a leaked `osk_` must not destroy its own
account) and requires the caller to retype the account email.

The order of the steps is the design, and each one is a failure mode:

1. **Stripe cancel first**, aborting everything if it fails, so we never erase a
   user we are still billing.
2. **R2 before the database**: those rows are the only index of which objects
   are theirs; dropping them first turns a failed purge into permanent orphans.
3. The DB delete is **one transaction** over an explicit table list
   (`USER_OWNED_TABLES`) rather than declared ON DELETE CASCADEs, since
   `create_all` never ALTERs an existing table and a constraint added later
   exists in the models but not in the live schema.
   `tests/test_account_erasure.py` fails if a new table references `users.id`
   without joining that list.

`app.py` registers a callback for the local working files, which record ownership
three ways: the `.owner` file clip jobs write (so jobs recovered from disk after a
restart count too), `saas_jobs`, and `thumbnail_sessions`. That last one is the
only thing that ever deletes generated thumbnails: the hourly sweep skips their
directory and they are served publicly at `/thumbnails/`.

What deliberately survives: the Stripe customer and its invoices (6-year
retention, Spanish commercial law) and one `account_deletions` row holding a
sha256 of the email as proof the erasure happened, itself purged after 5 years.
The "why are you leaving" answer is a closed list (`DELETION_REASONS`), never
free text: anything typed would land in a row designed to outlive the user.
`_apply_topup` reads the user id from Stripe metadata, so it confirms the row
still exists before inserting; otherwise the FK violation makes Stripe retry the
same doomed event for days.

## Checkout session (`create_checkout`)

- The client sends the surface that opened it as `source` (`wall`, `watermark`,
  `meter`, `plan_choice`, `account_topup`, `pricing_email_*`…). It is stored in the
  session metadata with `plan`, and decides the `cancel_url`: `#/pricing?checkout=cancel`
  for pricing-page clicks (the page scrolls to the plans), `#app?checkout=cancel`
  for everything opened inside the app. Both emit `CheckoutCanceled`.
- `allow_promotion_codes` for everyone (limiting it to win-back recipients was
  tried and reverted).
- The terms checkbox stays (`consent_collection`, art. 16(m) request for immediate
  performance) with a one-sentence `CONSENT_MESSAGE`.
- Checkout buttons reset their spinner on a back/forward-cache restore
  (`lib/backFromStripe.js`); without it every plan stayed disabled after Back.
