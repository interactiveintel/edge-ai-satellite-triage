"""Live data sources — pull real satellite imagery into the triage pipeline.

Three free, no-auth-required sources:

1. **Sentinel-2 L2A via Element84 Earth Search STAC**
   - Global 10 m multispectral, ~2-5 day latency
   - Endpoint: https://earth-search.aws.element84.com/v1
   - We pull preview JPEG thumbnails (small, fast) + scene metadata

2. **NOAA GOES-16 / GOES-18 via public S3 previews**
   - Americas + E. Pacific, ~10 min latency
   - Endpoint: https://cdn.star.nesdis.noaa.gov/GOES18/ABI/
   - Direct HTTPS pulls of PNG previews

3. **NASA FIRMS fire hotspots** (optional — requires free MAP_KEY)
   - Global active fire detections, ~3 h latency
   - Endpoint: https://firms.modaps.eosdis.nasa.gov/api

All sources return :class:`LiveFeedItem` with a numpy image array ready for
:class:`edge_triage.triage.EdgeTriageEngine.process_tile`.

Offline / CI: any ``requests.RequestException`` is caught and returns an empty
list — callers must tolerate this.
"""

from __future__ import annotations

import io
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class LiveFeedItem:
    """A single live-feed scene / tile ready for triage."""

    name: str
    image: np.ndarray   # HWC float32 in [0, 1]
    source: str         # "sentinel-2" | "goes-18" | "firms"
    acquired_utc: str   # ISO-8601 timestamp
    cloud_cover_pct: float | None = None
    bbox: tuple[float, float, float, float] | None = None  # minlon, minlat, maxlon, maxlat
    scene_id: str = ""
    preview_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ── Preset Areas of Interest ────────────────────────────────────────────────


AOI_PRESETS: dict[str, tuple[float, float, float, float]] = {
    # Wildfire-prone
    "California — wildfire zone":      (-122.0, 36.0, -118.5, 39.5),
    "Australia — bushfire belt":       (145.0, -39.0, 150.0, -35.0),
    # Defense / maritime
    "Strait of Hormuz":                (55.0, 26.0, 57.5, 27.5),
    "South China Sea — Scarborough":   (117.0, 14.5, 119.0, 16.5),
    "Black Sea — Odessa":              (30.0, 45.5, 32.5, 47.5),
    # Disaster response
    "Turkey — SE quake zone":          (36.0, 36.5, 39.0, 38.5),
    "Haiti — earthquake zone":         (-74.0, 18.0, -72.5, 19.0),
    # Energy
    "Permian Basin — pipelines":       (-103.0, 31.0, -101.5, 33.0),
    "North Sea — offshore rigs":       (1.0, 56.5, 4.0, 58.0),
}


# ── HTTP helper (stdlib only, no requests dep) ─────────────────────────────


def _http_get(url: str, timeout: float = 10.0, headers: dict[str, str] | None = None) -> bytes:
    """GET a URL using only stdlib. Raises urllib.error on failure."""
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "edge-triage/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = 15.0,
) -> dict[str, Any]:
    """POST JSON and parse JSON response."""
    import json
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "edge-triage/0.1",
            "Accept": "application/geo+json,application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _decode_image(raw: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes to HWC float32 array in [0, 1]."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


# ── Sentinel-2 L2A via Element84 STAC ──────────────────────────────────────


STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1/search"


class Sentinel2Source:
    """Fetch recent Sentinel-2 L2A scenes from the free Element84 STAC API.

    No authentication required. Returns preview JPEG thumbnails for speed
    (full COG download would be 100 MB+ per scene).
    """

    name = "sentinel-2"

    def __init__(
        self,
        max_cloud_cover: float = 30.0,
        max_age_days: int = 14,
        limit: int = 6,
    ) -> None:
        self.max_cloud_cover = max_cloud_cover
        self.max_age_days = max_age_days
        self.limit = limit

    def fetch(self, bbox: tuple[float, float, float, float]) -> list[LiveFeedItem]:
        """Search the STAC API for recent scenes over ``bbox``."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=self.max_age_days)
        datetime_range = f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now.strftime('%Y-%m-%dT%H:%M:%SZ')}"

        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": list(bbox),
            "datetime": datetime_range,
            "limit": max(self.limit, 1),
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
            "query": {"eo:cloud_cover": {"lt": self.max_cloud_cover}},
        }

        try:
            response = _http_post_json(STAC_ENDPOINT, payload)
        except Exception as exc:  # broad — stdlib urllib raises many types
            logger.warning("Sentinel-2 STAC search failed: %s", exc)
            return []

        features = response.get("features", [])
        items: list[LiveFeedItem] = []
        for feat in features[: self.limit]:
            props = feat.get("properties", {})
            assets = feat.get("assets", {})
            thumb = assets.get("thumbnail") or assets.get("preview")
            if not thumb or "href" not in thumb:
                continue

            try:
                raw = _http_get(thumb["href"], timeout=15.0)
                arr = _decode_image(raw)
            except Exception as exc:
                logger.warning("Failed to download thumbnail %s: %s", thumb["href"], exc)
                continue

            scene_bbox_raw = feat.get("bbox")
            scene_bbox: tuple[float, float, float, float] | None = None
            if scene_bbox_raw is not None and len(scene_bbox_raw) >= 4:
                scene_bbox = (
                    float(scene_bbox_raw[0]),
                    float(scene_bbox_raw[1]),
                    float(scene_bbox_raw[2]),
                    float(scene_bbox_raw[3]),
                )

            items.append(LiveFeedItem(
                name=f"S2 {props.get('datetime', '?')} — tile {feat.get('id', '?')}",
                image=arr,
                source=self.name,
                acquired_utc=str(props.get("datetime", "")),
                cloud_cover_pct=float(props.get("eo:cloud_cover") or 0.0),
                bbox=scene_bbox,
                scene_id=str(feat.get("id", "")),
                preview_url=str(thumb["href"]),
                extra={
                    "platform": str(props.get("platform", "")),
                    "instruments": props.get("instruments", []),
                    "grid_code": str(props.get("grid:code", "")),
                },
            ))

        logger.info("Sentinel-2 STAC: %d scenes fetched for bbox=%s", len(items), bbox)
        return items


# ── NOAA GOES-18 near-real-time previews ───────────────────────────────────


GOES_SECTOR_PRESETS: dict[str, str] = {
    # Sector name -> URL component on cdn.star.nesdis.noaa.gov
    "Full Disk (GOES-18)":      "FD",
    "CONUS (GOES-18)":          "CONUS",
    "Pacific Northwest":        "PNW",
    "California":               "CA",
    "West Central":             "WUS",
    "Pacific Southwest":        "PSW",
    "Hawaii":                   "hi",
}

GOES_PRODUCT = "GEOCOLOR"
GOES_SIZE = "2400x2400"


class NOAAGOESSource:
    """Fetch the latest GOES-18 imagery preview (near-real-time, ~10 min latency)."""

    name = "goes-18"

    def __init__(self, satellite: str = "GOES18") -> None:
        self.satellite = satellite

    def fetch(self, sector_code: str = "CA") -> list[LiveFeedItem]:
        """Pull the latest ``_latest.jpg`` preview for a sector."""
        # NOAA CDN pattern:
        # https://cdn.star.nesdis.noaa.gov/{SAT}/ABI/SECTOR/{sector}/{product}/latest.jpg
        # Top-level sector (FD/CONUS) uses a slightly different pattern.
        sector = sector_code.upper()
        if sector in {"FD", "CONUS"}:
            url = (
                f"https://cdn.star.nesdis.noaa.gov/{self.satellite}/ABI/{sector}/"
                f"{GOES_PRODUCT}/latest.jpg"
            )
        else:
            url = (
                f"https://cdn.star.nesdis.noaa.gov/{self.satellite}/ABI/SECTOR/{sector.lower()}/"
                f"{GOES_PRODUCT}/latest.jpg"
            )

        try:
            raw = _http_get(url, timeout=15.0)
            arr = _decode_image(raw)
        except Exception as exc:
            logger.warning("GOES fetch failed (%s): %s", url, exc)
            return []

        # GOES previews are one giant frame — we can tile-split for the engine
        return [LiveFeedItem(
            name=f"GOES-18 {sector} GeoColor — latest",
            image=arr,
            source=self.name,
            acquired_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            cloud_cover_pct=None,
            bbox=None,
            scene_id=f"goes18_{sector}_latest",
            preview_url=url,
            extra={"product": GOES_PRODUCT, "sector": sector},
        )]


# ── NASA FIRMS fire hotspots (CSV, requires free MAP_KEY) ──────────────────


FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


class FIRMSFireSource:
    """Fetch recent VIIRS active fire detections.

    Requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/

    Key lookup priority:
      1. Explicit ``map_key`` argument
      2. :class:`edge_triage.secrets_store.SecretsStore` (env var → local file)
    """

    name = "firms"

    def __init__(self, map_key: str | None = None, days: int = 1) -> None:
        if map_key is not None and map_key.strip():
            self.map_key = map_key.strip()
        else:
            # Defer import to avoid cyclic issues
            from .secrets_store import secrets as _secrets
            self.map_key = _secrets.get("firms")
        self.days = max(1, min(days, 10))

    def fetch(self, bbox: tuple[float, float, float, float]) -> list[LiveFeedItem]:
        """Fetch a CSV of fire detections and rasterize into a 512x512 heatmap."""
        if not self.map_key:
            logger.info("FIRMS: no MAP_KEY set (EDGE_TRIAGE_FIRMS_KEY) — skipping")
            return []

        # FIRMS expects: minlon,minlat,maxlon,maxlat
        area = ",".join(f"{v:.3f}" for v in bbox)
        url = (
            f"{FIRMS_API}/{self.map_key}/VIIRS_SNPP_NRT/{urllib.parse.quote(area)}/{self.days}"
        )

        try:
            raw = _http_get(url, timeout=20.0).decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("FIRMS fetch failed: %s", exc)
            return []

        lines = raw.strip().splitlines()
        if len(lines) < 2:
            return []

        header = [h.strip() for h in lines[0].split(",")]
        try:
            lat_i = header.index("latitude")
            lon_i = header.index("longitude")
            conf_i = header.index("confidence") if "confidence" in header else -1
            frp_i = header.index("frp") if "frp" in header else -1
        except ValueError:
            logger.warning("FIRMS: unexpected header %s", header)
            return []

        fires: list[tuple[float, float, float]] = []  # (lat, lon, intensity)
        for row in lines[1:]:
            parts = row.split(",")
            if len(parts) <= max(lat_i, lon_i):
                continue
            try:
                lat = float(parts[lat_i])
                lon = float(parts[lon_i])
            except ValueError:
                continue
            intensity = 0.5
            if frp_i >= 0 and frp_i < len(parts):
                try:
                    intensity = min(1.0, float(parts[frp_i]) / 50.0)
                except ValueError:
                    pass
            fires.append((lat, lon, intensity))

        # Rasterize to a 512x512 heatmap over the bbox
        minlon, minlat, maxlon, maxlat = bbox
        w, h = 512, 512
        heat = np.zeros((h, w), dtype=np.float32)
        for lat, lon, intensity in fires:
            if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
                continue
            x = int((lon - minlon) / (maxlon - minlon) * (w - 1))
            y = int((maxlat - lat) / (maxlat - minlat) * (h - 1))
            # Gaussian-ish stamp
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < w and 0 <= yy < h:
                        heat[yy, xx] = max(
                            heat[yy, xx],
                            intensity * np.exp(-(dx * dx + dy * dy) / 6.0),
                        )

        # Convert to RGB preview (red channel = fire intensity)
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[..., 0] = heat                    # red = fire
        rgb[..., 1] = heat * 0.3              # hint of orange
        rgb[..., 2] = 0.05                    # dark blue background

        return [LiveFeedItem(
            name=f"FIRMS VIIRS fires — {len(fires)} detections ({self.days}d)",
            image=rgb,
            source=self.name,
            acquired_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            cloud_cover_pct=0.0,
            bbox=bbox,
            scene_id=f"firms_{int(datetime.now().timestamp())}",
            preview_url=url,
            extra={"fire_count": len(fires)},
        )]


# ── Dispatch ────────────────────────────────────────────────────────────────


SOURCES = {
    "Sentinel-2 L2A (Earth Search)": Sentinel2Source,
    "NOAA GOES-18 (near-real-time)": NOAAGOESSource,
    "NASA FIRMS (active fires)":      FIRMSFireSource,
}
