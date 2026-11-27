"""Async transport for OpenRouter's typed Decisions API."""

import httpx


class DecisionsClient:
    def __init__(
        self, model_name: str, base_url: str, api_key: str, timeout: float, *,
        headers: dict[str, str] | None = None, transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model_name = model_name
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={**(headers or {}), "Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self._client.__aexit__(*exc)

    async def decide(self, payload: dict) -> dict:
        response = await self._client.post("decisions", json=payload)
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            return {"unparsed_response": response.text}
        return body if isinstance(body, dict) else {"unexpected_body": body}
