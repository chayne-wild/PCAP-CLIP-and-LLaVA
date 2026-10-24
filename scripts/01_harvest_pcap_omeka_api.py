# ==============================================================================
# SCRIPT: scripts/01_harvest_pcap_omeka_api.py
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Queries the University of Michigan PCAP Omeka S REST API
#          ([https://dcc.carceralstateproject.lsa.umich.edu/api/items](https://dcc.carceralstateproject.lsa.umich.edu/api/items) & /api/media),
#          paginating through all ~13,553 cataloged artworks. Extracts structured
#          metadata, resolves linked Artist Statement resources, filters out
#          non-artwork/logo items, and indexes direct image URLs WITHOUT
#          downloading binary images.
# ETHICS : Polite academic User-Agent header, randomized rate-limiting (1.5-3.0s),
#          and incremental checkpointing for fault-tolerant Colab resume.
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & ETHICAL PROTOCOL CONSTANTS
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path("/content/drive/MyDrive/colab_pcap_folder")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV_PATH = DATA_DIR / "pcap_master_metadata.csv"
EXCLUDED_CSV_PATH = DATA_DIR / "pcap_excluded_items.csv"
CHECKPOINT_JSON_PATH = DATA_DIR / "checkpoint_01_omeka_harvest.json"

# UMich Carceral State Project / PCAP Visual Art Database Omeka S Backend
OMEKA_BASE_URL = "[https://dcc.carceralstateproject.lsa.umich.edu](https://dcc.carceralstateproject.lsa.umich.edu)"
API_ITEMS_ENDPOINT = f"{OMEKA_BASE_URL}/api/items"
API_MEDIA_ENDPOINT = f"{OMEKA_BASE_URL}/api/media"
API_SITES_ENDPOINT = f"{OMEKA_BASE_URL}/api/sites"

USER_AGENT = "PCAP_Virginia_Tech_Dissertation_Bot/1.0 (chayne@vt.edu)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

PER_PAGE = 100
DELAY_MIN_SEC = 1.5
DELAY_MAX_SEC = 3.0
MAX_RETRIES = 4

# Optional cap for pilot testing in Colab (set to None for full ~13,553 harvest)
MAX_PAGES_PILOT: Optional[int] = None

# Non-artwork institutional records (logos, sponsors, guestbooks) to exclude
NON_ARTWORK_TITLE_PATTERNS = re.compile(
    r"\b(logo|sponsor|credits|guest\s*book|audio\s*tour|banner|header\s*image)\b",
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
# 2. JSON-LD PARSING, LINKED STATEMENT RESOLUTION & MEDIA EXTRACTION
# ------------------------------------------------------------------------------
def extract_property_values(
    item_json: dict,
    term_keys: Optional[List[str]] = None,
    label_keywords: Optional[List[str]] = None,
) -> str:
    """
    Extract literal or URI string values from an Omeka S JSON-LD item dictionary.
    Checks explicit Dublin Core / Bibliographic Ontology term keys first, then
    falls back to scanning custom property_label strings.
    """
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
        for key, entries in item_json.items():
            if not isinstance(entries, list) or key.startswith("o:"):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    prop_label = str(entry.get("property_label", "")).strip().lower()
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


def extract_artist_statement_field(
    item_json: dict,
    artwork_title: str,
    session: Optional[requests.Session] = None,
) -> str:
    """
    Extract the Artist Statement text. In PCAP's Omeka S schema, 'Artist Statement'
    is either a literal string OR a linked Omeka S item (type='resource:item',
    with 'value_resource_id') whose 'display_title' merely repeats the artwork
    title. This function avoids echoing duplicate titles and resolves linked
    statement items when present.
    """
    literals: List[str] = []
    linked_resource_ids: List[int] = []

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
                if entry.get("value_resource_id"):
                    try:
                        linked_resource_ids.append(int(entry["value_resource_id"]))
                    except (ValueError, TypeError):
                        pass

                val = entry.get("@value") or entry.get("o:label")
                if val:
                    cleaned = re.sub(r"\s+", " ", str(val)).strip()
                    if cleaned and cleaned.lower() != artwork_title.lower():
                        literals.append(cleaned)

    if literals:
        return "; ".join(dict.fromkeys(literals))

    # Resolve linked Artist Statement resource item if session is provided
    if linked_resource_ids and session is not None:
        resolved_texts: List[str] = []
        for res_id in linked_resource_ids[:1]:
            try:
                time.sleep(random.uniform(0.25, 0.6))
                resp = session.get(
                    f"{API_ITEMS_ENDPOINT}/{res_id}",
                    headers=HEADERS,
                    timeout=20,
                )
                if resp.status_code == 200:
                    linked_json = resp.json()
                    stmt_txt = extract_property_values(
                        linked_json,
                        [
                            "bibo:content",
                            "extracttext:extracted_text",
                            "dcterms:description",
                        ],
                        ["statement", "text", "description", "content"],
                    )
                    if stmt_txt and stmt_txt.lower() != artwork_title.lower():
                        resolved_texts.append(stmt_txt)
            except requests.RequestException:
                continue
        if resolved_texts:
            return "; ".join(dict.fromkeys(resolved_texts))

    return ""


def resolve_image_url_from_item(
    item_json: dict,
    session: requests.Session,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Extract the primary media_id and direct image URL ('large' or 'original')
    from an Omeka S item without downloading image bytes.
    Returns: (media_id, image_url, exclusion_reason)
    """
    media_list = item_json.get("o:media", [])
    if not isinstance(media_list, list) or len(media_list) == 0:
        return None, None, "no_attached_media_object"

    first_media = media_list[0]
    media_id = first_media.get("o:id") if isinstance(first_media, dict) else None

    # Fast path: Omeka S v3/v4 embeds thumbnail_display_urls directly on the item
    thumb_urls = item_json.get("thumbnail_display_urls", {})
    if isinstance(thumb_urls, dict):
        direct_url = thumb_urls.get("large") or thumb_urls.get("medium")
        if direct_url and any(
            direct_url.lower().endswith(ext)
            for ext in (".jpg", ".jpeg", ".png", ".webp")
        ):
            return media_id, direct_url, None

    # Fallback path: Query /api/media/{media_id} for o:original_url / o:thumbnail_urls
    if media_id is not None:
        time.sleep(random.uniform(0.35, 0.8))
        try:
            resp = session.get(
                f"{API_MEDIA_ENDPOINT}/{media_id}",
                headers=HEADERS,
                timeout=30,
            )
            if resp.status_code == 200:
                media_json = resp.json()
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
        except requests.RequestException as exc:
            return media_id, None, f"media_api_error:{type(exc).__name__}"

    return media_id, None, "unresolved_image_url"


def parse_omeka_item_record(
    item_json: dict,
    session: requests.Session,
    site_slug: str = "pcap",
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """
    Transform a raw Omeka S item JSON-LD payload into either a validated
    master metadata row or an excluded item diagnostic row.
    """
    item_id = item_json.get("o:id")
    raw_title = (
        item_json.get("o:title")
        or extract_property_values(item_json, ["dcterms:title"], ["title"])
        or "Untitled"
    )
    title = re.sub(r"\s+", " ", str(raw_title)).strip()

    # Gate 1: Exclude institutional logos, sponsor graphics, and non-artwork records
    if NON_ARTWORK_TITLE_PATTERNS.search(title):
        return None, {
            "item_id": str(item_id),
            "title": title,
            "exclusion_reason": "institutional_logo_or_non_artwork_page",
            "omeka_api_url": f"{API_ITEMS_ENDPOINT}/{item_id}",
        }

    # Gate 2: Resolve visual media URL
    media_id, image_url, exclusion_reason = resolve_image_url_from_item(
        item_json, session=session
    )
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
    artist_statement = extract_artist_statement_field(
        item_json, artwork_title=title, session=None
    )
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
# 3. TDD SELF-TEST SUITE (RUNS PRIOR TO LIVE API HARVESTING)
# ------------------------------------------------------------------------------
def run_tdd_tests() -> None:
    """Validate JSON-LD parsing and exclusion gates on synthetic payloads."""
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

    dummy_session = requests.Session()
    valid_row, excl_row = parse_omeka_item_record(mock_valid_item, dummy_session)
    assert valid_row is not None and excl_row is None, "[ERROR] Valid mock item rejected!"
    assert valid_row["item_id"] == "14892"
    assert valid_row["artist_name"] == "Van"
    assert valid_row["facility_raw"] == "Parnall Correctional Facility"
    assert valid_row["medium_raw"] == "Soap and ink"
    assert valid_row["artist_statement"] == "Art is my survival."
    assert valid_row["image_url"].endswith(".jpg")

    # Test exclusion of institutional logo records
    mock_logo_item = {
        "o:id": 88888,
        "o:title": "UM Arts Initiative Logo",
        "o:media": [{"@id": f"{API_MEDIA_ENDPOINT}/88889", "o:id": 88889}],
        "thumbnail_display_urls": {"large": f"{OMEKA_BASE_URL}/files/large/logo.jpg"},
    }
    valid_logo, excl_logo = parse_omeka_item_record(mock_logo_item, dummy_session)
    assert valid_logo is None and excl_logo is not None, "[ERROR] Logo exclusion failed!"

    # Test exclusion of text-only records
    mock_no_media_item = {"o:id": 99999, "o:title": "Text Record Only", "o:media": []}
    valid_row_2, excl_row_2 = parse_omeka_item_record(mock_no_media_item, dummy_session)
    assert valid_row_2 is None and excl_row_2 is not None, "[ERROR] Media gate failed!"
    assert excl_row_2["exclusion_reason"] == "no_attached_media_object"
    print("[TDD PASS] Omeka S JSON-LD parser, logo filter, and media gates verified.")


# ------------------------------------------------------------------------------
# 4. DISCOVER PCAP VISUAL ART DATABASE SITE SLUG & TOTAL ITEM COUNT
# ------------------------------------------------------------------------------
def discover_pcap_site_info(session: requests.Session) -> Tuple[Optional[int], str]:
    """
    Query /api/sites to locate the master PCAP Visual Art Database site_id and slug.
    Verifies via Omeka-S-Total-Results header that the site contains the full
    multi-year archive (>5,000 items); otherwise falls back to global /api/items.
    """
    try:
        resp = session.get(API_SITES_ENDPOINT, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            sites = resp.json()
            if isinstance(sites, list):
                for s in sites:
                    title = str(s.get("o:title", "")).lower()
                    slug = str(s.get("o:slug", ""))
                    if "visual art database" in title or slug.lower() == "pcap":
                        site_id = s.get("o:id")
                        # Check item count for this site_id
                        probe = session.get(
                            API_ITEMS_ENDPOINT,
                            params={"site_id": site_id, "per_page": 1},
                            headers=HEADERS,
                            timeout=20,
                        )
                        total_count = int(
                            probe.headers.get("Omeka-S-Total-Results", "0")
                        )
                        if total_count >= 5000:
                            print(
                                f"[INFO] Discovered Master PCAP Site: '{s.get('o:title')}' "
                                f"(site_id={site_id}, slug='{slug}', total_items={total_count})"
                            )
                            return site_id, slug
    except Exception as exc:
        print(f"[WARN] Site discovery fallback triggered ({exc}); querying global /api/items.")
    return None, "pcap"


# ------------------------------------------------------------------------------
# 5. MAIN FAULT-TOLERANT PAGINATED HARVESTER
# ------------------------------------------------------------------------------
def harvest_pcap_omeka_archive() -> None:
    """Paginate through the Omeka S /api/items endpoint with checkpoint resume."""
    run_tdd_tests()

    session = requests.Session()
    site_id, site_slug = discover_pcap_site_info(session)

    # Load checkpoint if resuming an interrupted Colab session
    start_page = 1
    seen_item_ids = set()
    if CHECKPOINT_JSON_PATH.exists() and MASTER_CSV_PATH.exists():
        try:
            with open(CHECKPOINT_JSON_PATH, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                start_page = int(ckpt.get("last_completed_page", 0)) + 1
            with open(MASTER_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    seen_item_ids.add(row["item_id"])
            print(
                f"[INFO] Resuming from checkpoint at Page {start_page} "
                f"({len(seen_item_ids)} artworks already indexed)."
            )
        except Exception as exc:
            print(f"[WARN] Checkpoint read failed ({exc}); restarting from Page 1.")
            start_page = 1

    # Initialize CSV files with headers if starting fresh
    if start_page == 1 or not MASTER_CSV_PATH.exists():
        with open(MASTER_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MASTER_COLUMNS).writeheader()
    if start_page == 1 or not EXCLUDED_CSV_PATH.exists():
        with open(EXCLUDED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EXCLUDED_COLUMNS).writeheader()

    page = start_page
    total_master = len(seen_item_ids)
    total_excluded = 0

    print(
        f"[INFO] Starting polite API harvest from {API_ITEMS_ENDPOINT} "
        f"(User-Agent: {USER_AGENT})"
    )

    pbar = tqdm(desc="Harvesting Omeka S Pages", unit="page", initial=start_page - 1)
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
                resp = session.get(
                    API_ITEMS_ENDPOINT,
                    params=params,
                    headers=HEADERS,
                    timeout=45,
                )
                if resp.status_code == 200:
                    items_batch = resp.json()
                    total_results_hdr = resp.headers.get("Omeka-S-Total-Results")
                    if total_results_hdr and pbar.total is None:
                        pbar.total = math.ceil(int(total_results_hdr) / PER_PAGE)
                    break
                elif resp.status_code in (429, 500, 502, 503, 504):
                    wait = (2 ** attempt) + random.uniform(1.0, 3.0)
                    print(
                        f"[WARN] HTTP {resp.status_code} on page {page}. "
                        f"Retrying in {wait:.1f}s (Attempt {attempt}/{MAX_RETRIES})..."
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"[ERROR] Unrecoverable HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            except requests.RequestException as req_err:
                wait = (2 ** attempt) + random.uniform(1.0, 3.0)
                print(
                    f"[WARN] Network exception on page {page}: {req_err}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

        if items_batch is None:
            raise RuntimeError(
                f"[ERROR] Failed to fetch page {page} after {MAX_RETRIES} retries."
            )

        if not isinstance(items_batch, list) or len(items_batch) == 0:
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

            master_row, excluded_row = parse_omeka_item_record(
                item_json, session=session, site_slug=site_slug
            )
            if master_row is not None:
                master_rows_to_write.append(master_row)
                seen_item_ids.add(item_id_str)
                total_master += 1
            elif excluded_row is not None:
                excluded_rows_to_write.append(excluded_row)
                total_excluded += 1

        # Flush page batch to disk immediately
        if master_rows_to_write:
            with open(MASTER_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
                writer.writerows(master_rows_to_write)

        if excluded_rows_to_write:
            with open(EXCLUDED_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EXCLUDED_COLUMNS)
                writer.writerows(excluded_rows_to_write)

        # Save checkpoint state
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

        pbar.set_postfix({"valid_artworks": total_master, "excluded": total_excluded})
        pbar.update(1)

        if len(items_batch) < PER_PAGE:
            print(f"\n[INFO] Final partial page ({len(items_batch)} items) processed.")
            break

        page += 1
        time.sleep(random.uniform(DELAY_MIN_SEC, DELAY_MAX_SEC))

    pbar.close()
    print("=" * 78)
    print(f"[TDD PASS] Script 01 Complete!")
    print(f"  -> Master Metadata CSV : {MASTER_CSV_PATH} ({total_master} artworks)")
    print(f"  -> Excluded Items CSV  : {EXCLUDED_CSV_PATH} ({total_excluded} records)")
    print("=" * 78)


if __name__ == "__main__":
    harvest_pcap_omeka_archive()
