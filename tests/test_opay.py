"""OPay client, with the emphasis on callback verification.

A callback endpoint is reachable by anyone. If verification is wrong in the
permissive direction, a forged "payment succeeded" grants a free subscription;
wrong in the strict direction, every real payment is rejected. Both are tested.
"""

import hashlib
import hmac
import json

import httpx
import pytest

from app.services.opay import (
    OPayClient,
    OPayConfig,
    OPayError,
    OPayNotConfigured,
    callback_signing_string,
    compute_callback_signature,
    naira_to_kobo,
    new_reference,
    sign_status_body,
    verify_callback,
)

SECRET = "OPAYPRV-test-secret-key-for-unit-tests"
CONFIG = OPayConfig(merchant_id="256612345678901", public_key="OPAYPUB-test", secret_key=SECRET)

PAYLOAD = {
    "amount": "500000",
    "channel": "Web",
    "country": "NG",
    "currency": "NGN",
    "displayedFailure": "",
    "fee": "737",
    "feeCurrency": "NGN",
    "instrumentType": "BankCard",
    "reference": "SUB-ABC123",
    "refunded": False,
    "status": "SUCCESS",
    "timestamp": "2026-09-13T10:20:46Z",
    "token": "220507145660712931829",
    "transactionId": "220507145660712931829",
    "updated_at": "2026-09-13T10:20:46Z",
}


def _signed(payload: dict) -> dict:
    return {"payload": payload, "sha512": compute_callback_signature(payload, SECRET),
            "type": "transaction-status"}


# --- amounts -----------------------------------------------------------------


def test_naira_is_converted_to_kobo() -> None:
    """Sending 5000 would charge fifty naira."""
    assert naira_to_kobo(5000) == 500_000


def test_fractional_naira_avoids_float_error() -> None:
    # 4999.99 * 100 is 499998.999... in binary floating point.
    assert naira_to_kobo(4999.99) == 499_999
    assert naira_to_kobo("4999.99") == 499_999


@pytest.mark.parametrize("bad", [0, -5, "0"])
def test_non_positive_amounts_are_refused(bad) -> None:
    with pytest.raises(ValueError):
        naira_to_kobo(bad)


def test_references_are_unique_and_unguessable() -> None:
    references = {new_reference() for _ in range(500)}
    assert len(references) == 500
    assert all(len(r) >= 24 for r in references)


# --- callback signature ------------------------------------------------------


def test_signing_string_matches_the_documented_format() -> None:
    assert callback_signing_string(PAYLOAD) == (
        '{Amount:"500000",Currency:"NGN",Reference:"SUB-ABC123",Refunded:f,'
        'Status:"SUCCESS",Timestamp:"2026-09-13T10:20:46Z",'
        'Token:"220507145660712931829",TransactionID:"220507145660712931829"}'
    )


def test_refunded_is_rendered_as_a_bare_letter() -> None:
    assert "Refunded:t," in callback_signing_string({**PAYLOAD, "refunded": True})
    assert "Refunded:f," in callback_signing_string({**PAYLOAD, "refunded": False})
    assert "Refunded:t," in callback_signing_string({**PAYLOAD, "refunded": "true"})


def test_callback_uses_sha3_512_not_sha_512() -> None:
    """The field is called sha512, but the algorithm is SHA3-512."""
    message = callback_signing_string(PAYLOAD).encode()
    sha3 = hmac.new(SECRET.encode(), message, hashlib.sha3_512).hexdigest()
    sha2 = hmac.new(SECRET.encode(), message, hashlib.sha512).hexdigest()

    assert compute_callback_signature(PAYLOAD, SECRET) == sha3
    assert compute_callback_signature(PAYLOAD, SECRET) != sha2


def test_a_genuine_callback_verifies() -> None:
    assert verify_callback(_signed(PAYLOAD), SECRET)["reference"] == "SUB-ABC123"


def test_a_sha2_signed_callback_is_rejected() -> None:
    body = {"payload": PAYLOAD, "sha512": hmac.new(
        SECRET.encode(), callback_signing_string(PAYLOAD).encode(), hashlib.sha512
    ).hexdigest()}
    with pytest.raises(OPayError, match="does not match"):
        verify_callback(body, SECRET)


@pytest.mark.parametrize("field,value", [
    ("amount", "1"),            # pay one kobo, claim the full subscription
    ("status", "SUCCESS"),      # changed below from FAIL
    ("reference", "SUB-OTHER"), # reuse a signature for another order
    ("refunded", True),
    ("currency", "USD"),
])
def test_tampering_with_any_signed_field_is_rejected(field, value) -> None:
    original = {**PAYLOAD, "status": "FAIL"} if field == "status" else PAYLOAD
    body = _signed(original)
    body["payload"] = {**original, field: value}
    with pytest.raises(OPayError, match="does not match"):
        verify_callback(body, SECRET)


@pytest.mark.parametrize("field", [
    "amount", "currency", "reference", "refunded", "status", "timestamp", "token", "transactionId",
])
def test_a_callback_missing_any_single_field_is_rejected(field) -> None:
    """OPay's sample only rejects when every field is absent. This must not."""
    payload = {k: v for k, v in PAYLOAD.items() if k != field}
    with pytest.raises(OPayError, match="missing"):
        verify_callback({"payload": payload, "sha512": "anything"}, SECRET)


def test_a_signature_made_with_another_key_is_rejected() -> None:
    body = {"payload": PAYLOAD, "sha512": compute_callback_signature(PAYLOAD, "some-other-key")}
    with pytest.raises(OPayError, match="does not match"):
        verify_callback(body, SECRET)


@pytest.mark.parametrize("body", [
    None, [], {}, {"payload": PAYLOAD}, {"sha512": "abc"}, {"payload": "text", "sha512": "abc"},
    {"payload": PAYLOAD, "sha512": ""},
])
def test_malformed_callbacks_are_rejected(body) -> None:
    with pytest.raises(OPayError):
        verify_callback(body, SECRET)


def test_verification_refuses_to_run_without_a_key() -> None:
    """An empty key would make every HMAC trivially reproducible."""
    with pytest.raises(OPayNotConfigured):
        verify_callback(_signed(PAYLOAD), "")


def test_signature_comparison_is_constant_time() -> None:
    import inspect

    from app.services import opay

    assert "compare_digest" in inspect.getsource(opay.verify_callback)


# --- status signing ----------------------------------------------------------


def test_status_request_is_signed_with_sha512() -> None:
    body = '{"reference":"SUB-ABC123","country":"NG"}'
    expected = hmac.new(SECRET.encode(), body.encode(), hashlib.sha512).hexdigest()
    assert sign_status_body(body, SECRET) == expected


# --- HTTP --------------------------------------------------------------------


def _client(handler) -> OPayClient:
    return OPayClient(CONFIG, transport=httpx.MockTransport(handler))


def test_create_sends_kobo_with_public_key_auth() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["merchant"] = request.headers["MerchantId"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "00000", "message": "SUCCESSFUL", "data": {
            "reference": "SUB-ABC123", "orderNo": "2110091408", "status": "INITIAL",
            "cashierUrl": "https://sandboxcashier.opaycheckout.com/checkout?orderToken=T",
        }})

    data = _client(handler).create_checkout(
        reference="SUB-ABC123", amount_kobo=500_000, return_url="https://x/r",
        callback_url="https://x/c", cancel_url="https://x/x", user_id=7,
        user_email="u@example.com", user_name="U", product_name="Monthly",
        product_description="30 days",
    )

    assert data["cashierUrl"].startswith("https://")
    assert seen["auth"] == "Bearer OPAYPUB-test"
    assert seen["merchant"] == "256612345678901"
    assert seen["url"].startswith("https://testapi.opaycheckout.com/"), "staging unless told otherwise"
    assert seen["body"]["amount"] == {"total": 500_000, "currency": "NGN"}
    assert seen["body"]["country"] == "NG"


def test_status_signature_covers_the_bytes_actually_sent() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"].removeprefix("Bearer ")
        seen["raw"] = request.content.decode()
        return httpx.Response(200, json={"code": "00000", "data": {"status": "SUCCESS"}})

    _client(handler).query_status("SUB-ABC123")
    assert seen["auth"] == sign_status_body(seen["raw"], SECRET)


def test_production_flag_switches_host() -> None:
    assert OPayConfig("m", "p", "s", production=True).base_url.startswith("https://liveapi.")
    assert OPayConfig("m", "p", "s").base_url.startswith("https://testapi.")


def test_a_non_success_code_raises() -> None:
    def handler(request):
        return httpx.Response(200, json={"code": "00004", "message": "Invalid request parameters"})

    with pytest.raises(OPayError, match="00004"):
        _client(handler).query_status("SUB-ABC123")


def test_an_http_error_raises() -> None:
    with pytest.raises(OPayError, match="HTTP 502"):
        _client(lambda r: httpx.Response(502, text="bad gateway")).query_status("SUB-ABC123")


def test_a_missing_checkout_url_raises() -> None:
    def handler(request):
        return httpx.Response(200, json={"code": "00000", "data": {"status": "INITIAL"}})

    with pytest.raises(OPayError, match="checkout URL"):
        _client(handler).create_checkout(
            reference="R", amount_kobo=1, return_url="r", callback_url="c", cancel_url="x",
            user_id=1, user_email="e", user_name="n", product_name="p", product_description="d",
        )


def test_unconfigured_client_refuses_to_call_out() -> None:
    client = OPayClient(OPayConfig("", "", ""), transport=httpx.MockTransport(
        lambda r: pytest.fail("must not reach the network")
    ))
    with pytest.raises(OPayNotConfigured):
        client.query_status("SUB-ABC123")


def test_status_body_uses_the_field_order_opay_accepts() -> None:
    """OPay re-serialises the body to check the signature, so order is load-bearing.

    Against the sandbox, country-first returned 00000 and reference-first
    returned 02000 Authentication failed, with the same key and reference.
    """
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw"] = request.content.decode()
        return httpx.Response(200, json={"code": "00000", "data": {"status": "SUCCESS"}})

    OPayClient(CONFIG, transport=httpx.MockTransport(handler)).query_status("SUB-ABC123")

    assert seen["raw"] == '{"country":"NG","reference":"SUB-ABC123"}'
    assert " " not in seen["raw"], "the spaced form is rejected too"
