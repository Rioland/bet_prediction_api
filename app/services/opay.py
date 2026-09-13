"""OPay Online Gateway (Cashier) client.

Written against https://documentation.opaycheckout.com - the online checkout
product, not the POS integration documented at documentation.opayweb.com, which
uses an unrelated encrypted-payload scheme.

Three operations, each with its own authentication:

    create   Authorization: Bearer <public key>
    status   Authorization: Bearer <HMAC-SHA512(body, secret key)>
    callback verified with   HMAC-SHA3-512(canonical string, secret key)

Note the two different hash functions. Status requests are signed with SHA-512;
callbacks are signed with SHA3-512, despite the callback field being named
"sha512". Using one where the other belongs rejects every genuine payment, or
worse, if a check is relaxed to make it pass.

OPay's own sample callback handler must not be copied. Its field-presence test
joins every !isset with &&, so it rejects a callback only when all fields are
missing; and it computes the signature without ever comparing it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from typing import Any

import httpx

STAGING_BASE = "https://testapi.opaycheckout.com"
PRODUCTION_BASE = "https://liveapi.opaycheckout.com"

CREATE_PATH = "/api/v1/international/cashier/create"
STATUS_PATH = "/api/v1/international/cashier/status"

SUCCESS_CODE = "00000"
COUNTRY = "NG"
CURRENCY = "NGN"

# Fields the callback signature is computed over, in the order OPay expects.
CALLBACK_SIGNED_FIELDS = (
    "amount", "currency", "reference", "refunded",
    "status", "timestamp", "token", "transactionId",
)


class OPayError(Exception):
    """The gateway rejected a request, or returned something unusable."""


class OPayNotConfigured(OPayError):
    """Credentials are missing; payments cannot be taken."""


@dataclass(frozen=True)
class OPayConfig:
    merchant_id: str
    public_key: str
    secret_key: str
    production: bool = False

    @property
    def base_url(self) -> str:
        return PRODUCTION_BASE if self.production else STAGING_BASE

    @property
    def configured(self) -> bool:
        return bool(self.merchant_id and self.public_key and self.secret_key)


def naira_to_kobo(naira: float | int | str) -> int:
    """Convert a naira amount to kobo, the unit the gateway expects.

    The gateway reads amount.total in the currency's minor unit, so sending
    5000 would charge fifty naira rather than five thousand.
    """
    from decimal import ROUND_HALF_UP, Decimal

    value = Decimal(str(naira))
    if value <= 0:
        raise ValueError("Amount must be positive.")
    return int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def new_reference() -> str:
    """An order reference that cannot be guessed or enumerated."""
    return f"SUB-{secrets.token_hex(12).upper()}"


def sign_status_body(body: str, secret_key: str) -> str:
    """HMAC-SHA512 over the exact request body string being sent."""
    return hmac.new(secret_key.encode(), body.encode(), hashlib.sha512).hexdigest()


def callback_signing_string(payload: dict[str, Any]) -> str:
    """The canonical string OPay signs a callback over.

    Not JSON: keys are capitalised and unquoted, string values are quoted, and
    refunded is the bare letter t or f.
    """
    missing = [field for field in CALLBACK_SIGNED_FIELDS if field not in payload]
    if missing:
        raise OPayError(f"Callback payload missing fields: {', '.join(missing)}")

    refunded = payload["refunded"]
    if isinstance(refunded, str):
        refunded = refunded.strip().lower() in {"true", "t", "1"}

    return (
        '{Amount:"%s",Currency:"%s",Reference:"%s",Refunded:%s,'
        'Status:"%s",Timestamp:"%s",Token:"%s",TransactionID:"%s"}'
    ) % (
        payload["amount"],
        payload["currency"],
        payload["reference"],
        "t" if refunded else "f",
        payload["status"],
        payload["timestamp"],
        payload["token"],
        payload["transactionId"],
    )


def compute_callback_signature(payload: dict[str, Any], secret_key: str) -> str:
    message = callback_signing_string(payload)
    return hmac.new(secret_key.encode(), message.encode(), hashlib.sha3_512).hexdigest()


def verify_callback(body: dict[str, Any], secret_key: str) -> dict[str, Any]:
    """Return the callback payload if and only if its signature is genuine.

    Raises on anything short of that. A signature that verifies proves the
    callback came from OPay; it does not by itself prove the payment is good,
    which is why the caller re-queries the status before granting access.
    """
    if not secret_key:
        raise OPayNotConfigured("OPay secret key is not configured.")
    if not isinstance(body, dict):
        raise OPayError("Callback body is not an object.")

    payload = body.get("payload")
    provided = body.get("sha512")
    if not isinstance(payload, dict) or not isinstance(provided, str) or not provided:
        raise OPayError("Callback is missing its payload or signature.")

    expected = compute_callback_signature(payload, secret_key)
    # Constant-time: an ordinary == leaks how many leading characters matched.
    if not hmac.compare_digest(expected, provided.strip().lower()):
        raise OPayError("Callback signature does not match.")
    return payload


class OPayClient:
    def __init__(self, config: OPayConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.config = config
        self._transport = transport

    def _require_config(self) -> None:
        if not self.config.configured:
            raise OPayNotConfigured(
                "OPay is not configured. Set OPAY_MERCHANT_ID, OPAY_PUBLIC_KEY and OPAY_SECRET_KEY."
            )

    def _post(self, path: str, body: str, authorization: str) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {authorization}",
            "MerchantId": self.config.merchant_id,
        }
        with httpx.Client(timeout=30, transport=self._transport) as client:
            response = client.post(f"{self.config.base_url}{path}", content=body, headers=headers)

        if response.status_code != 200:
            raise OPayError(f"OPay returned HTTP {response.status_code}.")
        try:
            data = response.json()
        except ValueError as exc:
            raise OPayError("OPay returned a response that is not JSON.") from exc

        if data.get("code") != SUCCESS_CODE:
            raise OPayError(f"OPay rejected the request: {data.get('code')} {data.get('message')}")
        return data.get("data") or {}

    def create_checkout(
        self,
        *,
        reference: str,
        amount_kobo: int,
        return_url: str,
        callback_url: str,
        cancel_url: str,
        user_id: int,
        user_email: str,
        user_name: str,
        product_name: str,
        product_description: str,
    ) -> dict[str, Any]:
        """Start a hosted checkout and return the page to send the customer to."""
        self._require_config()
        body = json.dumps(
            {
                "country": COUNTRY,
                "reference": reference,
                "amount": {"total": amount_kobo, "currency": CURRENCY},
                "returnUrl": return_url,
                "callbackUrl": callback_url,
                "cancelUrl": cancel_url,
                "customerVisitSource": "BROWSER",
                "expireAt": 30,
                "userInfo": {
                    "userId": str(user_id),
                    "userEmail": user_email,
                    "userName": user_name,
                },
                "product": {"name": product_name, "description": product_description},
            },
            separators=(",", ":"),
        )
        data = self._post(CREATE_PATH, body, self.config.public_key)
        if not data.get("cashierUrl"):
            raise OPayError("OPay did not return a checkout URL.")
        return data

    def query_status(self, reference: str) -> dict[str, Any]:
        """Ask OPay directly for a payment's state.

        Field order is load-bearing. OPay does not verify the signature against
        the bytes it receives; it re-serialises the body in its own order and
        compares. Verified against the sandbox with one key and one reference:

            {"country":"NG","reference":"..."}   00000 SUCCESSFUL
            {"reference":"...","country":"NG"}   02000 Authentication failed

        Whitespace matters for the same reason - the spaced form is rejected -
        so the body is built in the documented order with compact separators,
        and that exact string is both signed and posted.
        """
        self._require_config()
        body = json.dumps({"country": COUNTRY, "reference": reference}, separators=(",", ":"))
        return self._post(STATUS_PATH, body, sign_status_body(body, self.config.secret_key))
