# app/services/tools.py
import httpx
import logging

log = logging.getLogger(__name__)

async def fetch_url(url: str, timeout: float = 10.0) -> dict:
    """Fetch a URL and return minimal payload: status, text (first N chars), headers."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            text = resp.text
            return {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "snippet": text[:2000],  # keep small for now
                "url": str(resp.url),
            }
    except Exception as e:
        log.exception("fetch_url failed")
        return {"error": str(e), "url": url}