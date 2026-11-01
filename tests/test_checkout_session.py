"""What /api/billing/checkout asks Stripe for: Stripe's default expiry, the
surface on the session, a cancel URL back to where the click came from, and
the promotion-code field for everyone.
"""
import asyncio
from types import SimpleNamespace

import pytest

billing = pytest.importorskip("cloud.billing")  # needs the stripe SDK

ENTRY = {"price_id": "price_s", "kind": "subscription", "plan": "starter",
         "interval": "month", "minutes": 100}


@pytest.fixture()
def created(monkeypatch):
    calls = []

    async def _user(request):
        return SimpleNamespace(id="u1", email="a@b.co")

    async def _catalog():
        return {"by_id": {"price_s": ENTRY}}

    async def _none(*a, **k):
        return None

    async def _customer(*a, **k):
        return "cus_1"

    def _create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(url="https://checkout.stripe.com/c/x")

    monkeypatch.setattr(billing, "get_current_user_required", _user)
    monkeypatch.setattr(billing, "get_catalog", _catalog)
    monkeypatch.setattr(billing, "_blocking_sub_status", _none)
    monkeypatch.setattr(billing, "ensure_stripe_customer", _customer)
    monkeypatch.setattr(billing.stripe.checkout.Session, "create", _create)
    return calls


def _checkout(source=""):
    body = billing.CheckoutRequest(price_id="price_s", source=source)
    return asyncio.run(billing.create_checkout(body, request=None))


def test_default_expiry(created):
    assert _checkout("wall") == {"url": "https://checkout.stripe.com/c/x"}
    assert "expires_at" not in created[0]  # Stripe's 24 h


def test_source_and_plan_on_the_session(created):
    _checkout("watermark")
    md = created[0]["metadata"]
    assert md["source"] == "watermark" and md["plan"] == "starter" and md["user_id"] == "u1"


def test_cancel_goes_back_to_the_surface(created):
    _checkout("watermark")
    _checkout("pricing_email_expiring")
    _checkout("")
    assert created[0]["cancel_url"].endswith("/#app?checkout=cancel")
    assert created[1]["cancel_url"].endswith("/#/pricing?checkout=cancel")
    assert created[2]["cancel_url"].endswith("/#/pricing?checkout=cancel")


def test_bad_source_is_dropped(created):
    _checkout("<script>")
    assert created[0]["metadata"]["source"] == ""
    assert created[0]["cancel_url"].endswith("/#/pricing?checkout=cancel")


def test_promo_field_for_everyone(created):
    _checkout("wall")
    assert created[0]["allow_promotion_codes"] is True
    assert created[0]["after_expiration"]["recovery"]["allow_promotion_codes"] is True


def test_consent_box_kept_and_short(created):
    _checkout("wall")
    assert created[0]["consent_collection"] == {"terms_of_service": "required"}
    msg = created[0]["custom_text"]["terms_of_service_acceptance"]["message"]
    assert "14 days" in msg and len(msg) <= 160
