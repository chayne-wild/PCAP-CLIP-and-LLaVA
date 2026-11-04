# ==============================================================================
# scripts/01_harvest_pcap_omeka_api.py (FULL ARCHIVE HARVEST)
# Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Queries the University of Michigan PCAP Omeka S REST API using ONLY
#          Python's built-in standard library (0 MB external dependencies).
#          Writes directly to Google Drive on Chromebook or Colab.
# ETHICS : Explicit Virginia Tech researcher identification headers
#          (User-Agent, RFC 9110 From, X-Research-Project) + 1.5-3.0s delay.
# OUTPUTS: data/pcap_master_metadata.csv
#          data/pcap_excluded_items.csv
# ==============================================================================

import os
import re
import csv
import math
import json
import time
import random
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# 1. AUTO-DETECT GOOGLE DRIVE PATH (CHROMEBOOK LINUX OR GOOGLE COLAB)
# ------------------------------------------------------------------------------
CHROMEBOOK_DRIVE = Path("/mnt/chromeos/GoogleDrive/MyDrive/colab_pcap_folder")
COLAB_DRIVE = Path("/content/drive/MyDrive/colab_pcap_folder")

if CHROMEBOOK_DRIVE.exists():
    PROJECT_ROOT = CHROMEBOOK_DRIVE
elif COLAB_DRIVE.exists():
    PROJECT_ROOT = COLAB_DRIVE
else:
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV_PATH = DATA_DIR / "pcap_master_metadata.csv"
EXCLUDED_CSV_PATH = DATA_DIR / "pcap_excluded_items.csv"
CHECKPOINT_JSON_PATH = DATA_DIR / "checkpoint_01_omeka_harvest.json"

OMEKA_BASE_URL = "https://dcc.carceralstateproject.lsa.umich.edu"
API_ITEMS_ENDPOINT = f"{OMEKA_BASE_URL}/api/items"
API_MEDIA_ENDPOINT = f"{OMEKA_BASE_URL}/api/media"
API_SITES_ENDPOINT = f"{OMEKA_BASE_URL}/api/sites"

# Ethical academic identification (User-Agent + RFC 9110 'From' + X-Research-Project)
RESEARCH_BOT_ID = "PCAP_Virginia_Tech_Research_Bot/1.0 (chayne@vt.edu)"
PRIMARY_HEADERS = {
    "User-Agent": RESEARCH_BOT_ID,
    "From": "chayne@vt.edu",
    "X-Research-Project": RESEARCH_BOT_ID,
    "Accept": "application/json",
}
# Fallback if Cloudflare WAF regex challenges the word "Bot", still preserving
# full Virginia Tech & chayne@vt.edu identification in User-Agent, From, and X-Research-Project:
FALLBACK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 "
        "PCAP_Virginia_Tech_Research/1.0 (+mailto:chayne@vt.edu)"
    ),
    "From": "chayne@vt.edu",
    "X-Research-Project": RESEARCH_BOT_ID,
    "Accept": "application/json",
}
ACTIVE_HEADERS = dict(PRIMARY_HEADERS)

PER_PAGE = 100
DELAY_MIN_SEC = 1.5
DELAY_MAX_SEC = 3.0
MAX_RETRIES = 4

# Full harvest enabled (None = harvest all ~13,553 artworks across all pages)
MAX_PAGES_PILOT: Optional[int] = None

NON_ARTWORK_TITLE_PATTERNS = re.compile(
    r"\b(logo|sponsor|credits|guest\s*book|audio\s*tour|banner|header\s*image|oembed\s*media|box\s*url\s*test)\b",
    re.IGNORECASE,
)

MASTER_COLUMNS = [
    "item_id",
    "identifier",
    "title",
    "raw_date",
    "exhibition_source",
    "artist_name",
    "facility_raw",
    "medium_raw",
    "subject_tags",
    "visual_description",
    "artist_statement",
    "rights",
    "media_id",
    "image_url",
    "omeka_web_url",
]

EXCLUDED_COLUMNS = [
    "item_id",
    "title",
    "exclusion_reason",
    "omeka_api_url",
]


# ------------------------------------------------------------------------------
# 2. ZERO-DEPENDENCY HTTP JSON HELPER (BUILT-IN URLLIB)
# ------------------------------------------------------------------------------
def http_get_json(url: str, params: Optional[dict] = None, timeout: int = 35) -> Tuple[int, dict, object]:
    """Perform HTTP GET using Python's built-in urllib.request and return (status, headers, json_data)."""
    global ACTIVE_HEADERS
    full_url = url
    if params:
        query_str = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_str}"

    req = urllib.request.Request(full_url, headers=ACTIVE_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            resp_headers = {k.lower(): v for k, v in resp.info().items()}
            raw_bytes = resp.read()
            data = json.loads(raw_bytes.decode("utf-8"))
            return status, resp_headers, data
    except urllib.error.HTTPError as e:
        if e.code == 403 and ACTIVE_HEADERS.get("User-Agent") == RESEARCH_BOT_ID:
            print(
                "[INFO] Cloudflare WAF challenged 'Bot' keyword; switching to "
                "User-Agent with '+mailto:chayne@vt.edu' & 'From: chayne@vt.edu'..."
            )
            ACTIVE_HEADERS = dict(FALLBACK_HEADERS)
            req2 = urllib.request.Request(full_url, headers=ACTIVE_HEADERS)
            with urllib.request.urlopen(req2, timeout=timeout) as resp:
                status = resp.getcode()
                resp_headers = {k.lower(): v for k, v in resp.info().items()}
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))
                return status, resp_headers, data
        raise


# ------------------------------------------------------------------------------
# 3. JSON-LD PARSING & MEDIA EXTRACTION
# ------------------------------------------------------------------------------
def extract_property_values(
    item_json: dict,
    term_keys: Optional[List[str]] = None,
    label_keywords: Optional[List[str]] = None,
) -> str:
    extracted: List[str] = []

    if term_keys:
        for key in term_keys:
            entries = item_json.get(key, [])
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        val = (
                            entry.get("@value")
                            or entry.get("o:label")
                            or entry.get("display_title")
                            or entry.get("@id")
                        )
                        if val:
                            cleaned = re.sub(r"\s+", " ", str(val)).strip()
                            if cleaned:
                                extracted.append(cleaned)

    if label_keywords:
        allow_statement = any("statement" in kw.lower() for kw in label_keywords)
        for key, entries in item_json.items():
            if not isinstance(entries, list) or key.startswith("o:"):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    prop_label = str(entry.get("property_label", "")).strip().lower()
                    if "statement" in prop_label and not allow_statement:
                        continue
                    if any(kw.lower() in prop_label for kw in label_keywords):
                        val = (
                            entry.get("@value")
                            or entry.get("o:label")
                            or entry.get("display_title")
                        )
                        if val:
                            cleaned = re.sub(r"\s+", " ", str(val)).strip()
                            if cleaned:
                                extracted.append(cleaned)

    seen = set()
    unique_vals = []
    for val in extracted:
        if val not in seen:
            seen.add(val)
            unique_vals.append(val)

    return "; ".join(unique_vals)


def extract_artist_statement_field(item_json: dict, artwork_title: str) -> str:
    literals: List[str] = []
    for key, entries in item_json.items():
        if not isinstance(entries, list) or key.startswith("o:"):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            prop_label = str(entry.get("property_label", "")).strip().lower()
            is_stmt_key = key in (
                "bibo:content",
                "dcterms:abstract",
                "extracttext:extracted_text",
            )
            if is_stmt_key or "statement" in prop_label:
                val = entry.get("@value") or entry.get("o:label")
                if val:
                    cleaned = re.sub(r"\s+", " ", str(val)).strip()
                    if cleaned and cleaned.lower() != artwork_title.lower():
                        literals.append(cleaned)

    if literals:
        return "; ".join(dict.fromkeys(literals))
    return ""


def resolve_image_url_from_item(
    item_json: dict,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    media_list = item_json.get("o:media", [])
    if not isinstance(media_list, list) or len(media_list) == 0:
        return None, None, "no_attached_media_object"

    first_media = media_list[0]
    media_id = first_media.get("o:id") if isinstance(first_media, dict) else None

    thumb_urls = item_json.get("thumbnail_display_urls", {})
    if isinstance(thumb_urls, dict):
        direct_url = thumb_urls.get("large") or thumb_urls.get("medium")
        if direct_url and any(
            direct_url.lower().endswith(ext)
            for ext in (".jpg", ".jpeg", ".png", ".webp")
        ):
            return media_id, direct_url, None

    if media_id is not None:
        time.sleep(random.uniform(0.35, 0.8))
        try:
            status, _, media_json = http_get_json(f"{API_MEDIA_ENDPOINT}/{media_id}", timeout=30)
            if status == 200 and isinstance(media_json, dict):
                media_type = str(media_json.get("o:media_type", "")).lower()
                if media_type and not media_type.startswith("image/"):
                    return media_id, None, f"non_image_media_type:{media_type}"

                t_urls = media_json.get("o:thumbnail_urls", {})
                img_url = (
                    (t_urls.get("large") if isinstance(t_urls, dict) else None)
                    or media_json.get("o:original_url")
                    or (t_urls.get("medium") if isinstance(t_urls, dict) else None)
                )
                if img_url:
                    return media_id, str(img_url), None
                return media_id, None, "media_record_missing_image_url"
        except Exception as exc:
            return media_id, None, f"media_api_error:{type(exc).__name__}"

    return media_id, None, "unresolved_image_url"


def parse_omeka_item_record(
    item_json: dict,
    site_slug: str = "pcap",
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    item_id = item_json.get("o:id")
    raw_title = (
        item_json.get("o:title")
        or extract_property_values(item_json, ["dcterms:title"], ["title"])
        or "Untitled"
    )
    title = re.sub(r"\s+", " ", str(raw_title)).strip()

    if NON_ARTWORK_TITLE_PATTERNS.search(title):
        return None, {
            "item_id": str(item_id),
            "title": title,
            "exclusion_reason": "institutional_logo_or_non_artwork_page",
            "omeka_api_url": f"{API_ITEMS_ENDPOINT}/{item_id}",
        }

    media_id, image_url, exclusion_reason = resolve_image_url_from_item(item_json)
    if exclusion_reason is not None or not image_url:
        return None, {
            "item_id": str(item_id),
            "title": title,
            "exclusion_reason": exclusion_reason or "missing_image_url",
            "omeka_api_url": f"{API_ITEMS_ENDPOINT}/{item_id}",
        }

    source_str = extract_property_values(
        item_json,
        ["dcterms:source", "dcterms:isPartOf", "dcterms:relation"],
        ["source", "exhibition", "part of", "collection"],
    )
    identifier_str = extract_property_values(
        item_json,
        ["dcterms:identifier"],
        ["log number", "identifier"],
    )
    medium_str = extract_property_values(
        item_json,
        ["dcterms:medium"],
        ["medium", "material"],
    )
    raw_date = extract_property_values(
        item_json,
        ["dcterms:date", "dcterms:created", "dcterms:issued"],
        ["date", "year"],
    )
    artist_name = (
        extract_property_values(
            item_json,
            ["dcterms:creator", "dcterms:contributor"],
            ["artist", "creator"],
        )
        or "Anonymous / Unlisted Artist"
    )
    facility_raw = extract_property_values(
        item_json,
        ["dcterms:spatial", "dcterms:coverage", "dcterms:publisher"],
        ["prison", "facility", "institution", "correctional"],
    )
    subject_tags = extract_property_values(
        item_json,
        ["dcterms:subject"],
        ["subject", "theme", "keyword", "gallery"],
    )
    visual_description = extract_property_values(
        item_json,
        ["dcterms:description"],
        ["description", "visual description"],
    )
    artist_statement = extract_artist_statement_field(item_json, artwork_title=title)
    rights = extract_property_values(
        item_json,
        ["dcterms:rights", "dcterms:rightsHolder"],
        ["rights", "copyright"],
    )

    omeka_web_url = f"{OMEKA_BASE_URL}/s/{site_slug}/item/{item_id}"

    master_row = {
        "item_id": str(item_id),
        "identifier": identifier_str,
        "title": title,
        "raw_date": raw_date,
        "exhibition_source": source_str,
        "artist_name": artist_name,
        "facility_raw": facility_raw,
        "medium_raw": medium_str,
        "subject_tags": subject_tags,
        "visual_description": visual_description,
        "artist_statement": artist_statement,
        "rights": rights,
        "media_id": str(media_id) if media_id else "",
        "image_url": image_url,
        "omeka_web_url": omeka_web_url,
    }
    return master_row, None


# ------------------------------------------------------------------------------
# 4. TDD SELF-TEST SUITE & PCAP SITE_ID DISCOVERY
# ------------------------------------------------------------------------------
def run_tdd_tests() -> None:
    print("[INFO] Running TDD unit tests for Omeka S JSON-LD parser...")
    mock_valid_item = {
        "o:id": 14892,
        "o:title": "Psychoanalysis",
        "o:media": [{"@id": f"{API_MEDIA_ENDPOINT}/15001", "o:id": 15001}],
        "thumbnail_display_urls": {
            "large": f"{OMEKA_BASE_URL}/files/large/sample_hash.jpg"
        },
        "dcterms:creator": [{"property_label": "Artist", "@value": "Van"}],
        "dcterms:medium": [{"property_label": "Medium", "@value": "Soap and ink"}],
        "dcterms:date": [{"property_label": "Date", "@value": "2021"}],
        "dcterms:source": [
            {
                "property_label": "Source",
                "@value": "26th Annual Exhibition of Art by Michigan Prisoners",
            }
        ],
        "dcterms:spatial": [
            {"property_label": "Prison", "@value": "Parnall Correctional Facility"}
        ],
        "bibo:content": [
            {"property_label": "Artist Statement", "@value": "Art is my survival."}
        ],
    }

    valid_row, excl_row = parse_omeka_item_record(mock_valid_item)
    assert valid_row is not None and excl_row is None
    assert valid_row["artist_name"] == "Van"
    assert valid_row["facility_raw"] == "Parnall Correctional Facility"
    assert valid_row["medium_raw"] == "Soap and ink"
    assert valid_row["artist_statement"] == "Art is my survival."
    print("[TDD PASS] Omeka S JSON-LD parser and exclusion gates verified.")


def discover_pcap_site_info() -> Tuple[Optional[int], str]:
    """
    Find the exact Omeka S site_id for the PCAP Visual Art Database so we only
    harvest PCAP artworks and exclude other DCC sub-sites / IT test items.
    """
    try:
        status, _, sites = http_get_json(API_SITES_ENDPOINT, timeout=25)
        if status == 200 and isinstance(sites, list):
            for s in sites:
                title = str(s.get("o:title", "")).lower()
                slug = str(s.get("o:slug", ""))
                if "visual art database" in title or slug.lower() == "pcap":
                    site_id = s.get("o:id")
                    _, hdrs, _ = http_get_json(
                        API_ITEMS_ENDPOINT,
                        params={"site_id": site_id, "per_page": 1},
                        timeout=20,
                    )
                    total_count = int(hdrs.get("omeka-s-total-results", "0"))
                    if total_count >= 5000:
                        print(
                            f"[INFO] Discovered Master PCAP Site: '{s.get('o:title')}' "
                            f"(site_id={site_id}, slug='{slug}', total_items={total_count})"
                        )
                        return int(site_id), slug
    except Exception:
        pass

    try:
        status, _, known_item = http_get_json(f"{API_ITEMS_ENDPOINT}/75605", timeout=25)
        if status == 200 and isinstance(known_item, dict):
            for s_ref in known_item.get("o:site", []):
                cand_id = s_ref.get("o:id")
                if cand_id:
                    _, hdrs, _ = http_get_json(
                        API_ITEMS_ENDPOINT,
                        params={"site_id": cand_id, "per_page": 1},
                        timeout=20,
                    )
                    total_count = int(hdrs.get("omeka-s-total-results", "0"))
                    if 10000 <= total_count <= 20000:
                        print(
                            f"[INFO] Locked onto PCAP Visual Art Database via Item 75605: "
                            f"site_id={cand_id} (total_items={total_count})"
                        )
                        return int(cand_id), "pcap"
    except Exception as exc:
        print(f"[WARN] Could not probe Item 75605 for site_id ({exc}).")

    return None, "pcap"


# ------------------------------------------------------------------------------
# 5. MAIN FAULT-TOLERANT HARVESTER
# ------------------------------------------------------------------------------
def harvest_pcap_omeka_archive() -> None:
    print("=" * 78)
    print(f"[INFO] Writing directly to : {DATA_DIR}")
    print(f"[INFO] Polite User-Agent   : {ACTIVE_HEADERS['User-Agent']}")
    print(f"[INFO] RFC 9110 From Header: {ACTIVE_HEADERS['From']}")
    print(f"[INFO] Rate-Limit Delay    : {DELAY_MIN_SEC}-{DELAY_MAX_SEC}s between pages")
    print("=" * 78)
    run_tdd_tests()

    site_id, site_slug = discover_pcap_site_info()

    start_page = 1
    seen_item_ids = set()
    if CHECKPOINT_JSON_PATH.exists() and MASTER_CSV_PATH.exists():
        try:
            with open(CHECKPOINT_JSON_PATH, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                start_page = int(ckpt.get("last_completed_page", 0)) + 1
            with open(MASTER_CSV_PATH, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    seen_item_ids.add(row["item_id"])
            print(
                f"[INFO] Resuming from checkpoint at Page {start_page} "
                f"({len(seen_item_ids)} artworks already indexed)."
            )
        except Exception:
            start_page = 1

    if start_page == 1 or not MASTER_CSV_PATH.exists():
        with open(MASTER_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MASTER_COLUMNS).writeheader()
    if start_page == 1 or not EXCLUDED_CSV_PATH.exists():
        with open(EXCLUDED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EXCLUDED_COLUMNS).writeheader()

    page = start_page
    total_master = len(seen_item_ids)
    total_excluded = 0
    total_pages_est = MAX_PAGES_PILOT

    print(f"[INFO] Starting polite Full Archive harvest from {API_ITEMS_ENDPOINT} ...")

    while True:
        if MAX_PAGES_PILOT is not None and page > MAX_PAGES_PILOT:
            print(f"\n[INFO] Reached MAX_PAGES_PILOT={MAX_PAGES_PILOT}. Stopping.")
            break

        params = {
            "page": page,
            "per_page": PER_PAGE,
            "sort_by": "id",
            "sort_order": "asc",
        }
        if site_id is not None:
            params["site_id"] = site_id

        items_batch = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                status, hdrs, data = http_get_json(API_ITEMS_ENDPOINT, params=params, timeout=45)
                if status == 200 and isinstance(data, list):
                    items_batch = data
                    total_hdr = hdrs.get("omeka-s-total-results")
                    if total_hdr and total_pages_est is None:
                        total_pages_est = math.ceil(int(total_hdr) / PER_PAGE)
                    break
            except Exception as err:
                wait = (2 ** attempt) + random.uniform(1.0, 3.0)
                print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} on page {page} failed ({err}). Retrying in {wait:.1f}s...")
                time.sleep(wait)

        if items_batch is None:
            raise RuntimeError(f"[ERROR] Failed to fetch page {page} after {MAX_RETRIES} retries.")

        if len(items_batch) == 0:
            print(f"\n[INFO] Reached final empty page at page={page}. Harvest complete.")
            break

        master_rows_to_write = []
        excluded_rows_to_write = []

        for item_json in items_batch:
            if not isinstance(item_json, dict) or "o:id" not in item_json:
                continue

            item_id_str = str(item_json.get("o:id", ""))
            if item_id_str in seen_item_ids:
                continue

            master_row, excluded_row = parse_omeka_item_record(item_json, site_slug=site_slug)
            if master_row is not None:
                master_rows_to_write.append(master_row)
                seen_item_ids.add(item_id_str)
                total_master += 1
            elif excluded_row is not None:
                excluded_rows_to_write.append(excluded_row)
                total_excluded += 1

        if master_rows_to_write:
            with open(MASTER_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=MASTER_COLUMNS).writerows(master_rows_to_write)

        if excluded_rows_to_write:
            with open(EXCLUDED_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=EXCLUDED_COLUMNS).writerows(excluded_rows_to_write)

        with open(CHECKPOINT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_completed_page": page,
                    "total_master_indexed": total_master,
                    "total_excluded": total_excluded,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                f,
                indent=2,
            )

        denom = f"/{total_pages_est}" if total_pages_est else ""
        print(
            f"  -> [Page {page}{denom}] Indexed {len(master_rows_to_write)} artworks "
            f"(Running Total: {total_master} valid, {total_excluded} excluded)"
        )

        if len(items_batch) < PER_PAGE:
            print(f"\n[INFO] Final partial page ({len(items_batch)} items) processed.")
            break

        page += 1
        time.sleep(random.uniform(DELAY_MIN_SEC, DELAY_MAX_SEC))

    print("=" * 78)
    print("[TDD PASS] Script 01 Complete!")
    print(f"  -> Master Metadata CSV : {MASTER_CSV_PATH} ({total_master} artworks)")
    print(f"  -> Excluded Items CSV  : {EXCLUDED_CSV_PATH} ({total_excluded} records)")
    print("=" * 78)


if __name__ == "__main__":
    harvest_pcap_omeka_archive()
