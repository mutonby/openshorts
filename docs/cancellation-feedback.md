# Private cancellation feedback

Account → **Cancel subscription** opens an optional form. Both continuing with
feedback and skipping open Stripe. Stripe is still responsible for confirmation,
cancellation timing and any outstanding invoices; this endpoint never changes a
subscription. Manage billing remains available without the form.

`POST /api/billing/cancel` requires the existing authenticated bearer session.
The user and subscription identifiers are resolved server-side. Accepted fields:
`reason` (too_expensive, not_using, quality, missing_features, technical_issues,
switching, other), `comment` (maximum 2000 characters), `rating` (integer 1–5).
All fields may be omitted. Arbitrary identifiers and extra fields are rejected.
The response contains only the Stripe portal URL.

## Storage / rollout

`CancellationFeedback` registers a new `cancellation_feedback` table. The existing
`cloud.database.init_engine()` / SQLAlchemy `create_all` bootstrap creates it on
startup, including indexes, on both existing and fresh installations. No existing
tables need ALTER statements. No production migration has been executed here.
Feedback is best-effort: a failed write does not block the portal URL. Deep-link
failure falls back to the existing standard Stripe portal. The UI also falls back
to the standard portal if the request fails or times out.

Every saved row has `status = 'intent'`, including skipped feedback (null answers).
An intent is NOT a completed cancellation. The existing Stripe webhooks continue
to own `subscriptions.status` and `cancel_at_period_end`. Do not count intent rows
as churn or treat a Stripe return URL as proof. Repeated visits can create several
intents. Match by Stripe subscription ID, not just user ID, when analysing churn.

## Telegram delivery

Non-empty feedback is sent after the portal response using FastAPI background
 tasks and the existing `cloud.alerts.send_telegram` configuration
(`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`). No new destination is configured.
The message contains a short account reference, reason, optional rating/plan,
and comment, explicitly labelled as feedback received, NOT a confirmed cancellation.
Skipping feedback sends no notification. The form discloses Telegram sharing.
Obvious emails and long phone/card-like numbers are redacted heuristically;
this is NOT guaranteed anonymization. Full original feedback stays in the private DB.
Messages are bounded below Telegram's length limit and sent as plain text.

Delivery is best-effort, not a durable queue: Telegram failures or process shutdown
can lose a notification, but cannot block cancellation. There are no automatic retries
or confirmed delivery receipts. Operators can retrieve stored feedback below.
Account deletion removes DB feedback; it does not retract already delivered Telegram
messages. Apply access controls and an appropriate retention policy to the admin chat.

## Operator-only retrieval

No public feedback endpoint is exposed. Use the existing authenticated private
PostgreSQL operator connection (prefer a read-only role through the normal SSH/VPN
path). Do not expose database credentials in a browser or shell history. Example
read-only SQL, run inside that authenticated psql session:

```sql
BEGIN READ ONLY;
SELECT f.created_at, f.user_id, f.stripe_subscription_id,
       f.reason, f.rating, f.comment, f.status AS feedback_status,
       s.status AS current_subscription_status, s.cancel_at_period_end
FROM cancellation_feedback AS f
LEFT JOIN subscriptions AS s
  ON s.stripe_subscription_id = f.stripe_subscription_id
ORDER BY f.created_at DESC
LIMIT 100;
COMMIT;
```

Comments are private customer data. Limit access to authorised operators, do not
copy them to analytics, public reviews, or logs. Account erasure explicitly deletes
these rows (also protected by the user FK cascade). Normal account-data retention
applies; no independent indefinite retention or public review integration is added.

## Verification

Backend tests isolate Stripe and the database with fixtures; they make no external
Stripe requests or production database changes. Run:

```sh
python -m pytest tests/test_cancellation_feedback.py tests/test_billing_states.py tests/test_account_erasure.py -q
cd dashboard && npm run build
```

Before rollout, verify the Stripe **test-mode** portal configuration allows
subscription cancellation with the intended timing. Test keyboard navigation,
Escape/return focus, skip, failed network retries, and the actual test-mode Stripe
handoff. The native dialog provides modal focus management; no public review is
submitted. This implementation does not claim live Stripe or database integration
verification.
