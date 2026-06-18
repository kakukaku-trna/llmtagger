from __future__ import annotations

import argparse
import json
import os
import sys

from adwsdk.adw_client import AdwClient
from adwsdk import cmm


def _build_client_from_env() -> AdwClient | None:
    """Build AdwClient from environment variables."""
    env = os.getenv("ADW_ENV", "prod")
    user = os.getenv("ADW_USER")
    passwd = os.getenv("ADW_PASS")
    site_path = os.getenv("ADW_SITE_PATH")

    if not user or not passwd:
        return None

    if site_path and os.path.isfile(site_path):
        client = AdwClient(adw_site_path=site_path, adw_user=user, adw_pass=passwd)
    else:
        client = AdwClient(env=env, adw_user=user, adw_pass=passwd)
    return client


def _check_completeness(extend_json: str) -> bool:
    """Return True if the record is considered data-complete.

    Completeness criteria (checked in order):
    1. ``miss_file_cnt`` must be 0.
    2. If ``excepted_files`` is present, every expected category must
       have a count > 0.
    """
    if not extend_json:
        return False
    try:
        ext = json.loads(extend_json) if isinstance(extend_json, str) else extend_json
    except (json.JSONDecodeError, TypeError):
        return False

    # 1. miss_file_cnt == 0
    if ext.get("miss_file_cnt", -1) != 0:
        return False

    # 2. All expected file categories present
    expected = ext.get("excepted_files")
    if expected is not None:
        if not isinstance(expected, dict):
            return False
        for category, count in expected.items():
            if count == 0:
                return False

    return True


def _extract_rules_version(extra_data_json: str) -> str | None:
    """Extract ``rules_version`` from the ``extra_data`` JSON blob."""
    if not extra_data_json:
        return None
    try:
        data = (
            json.loads(extra_data_json)
            if isinstance(extra_data_json, str)
            else extra_data_json
        )
    except (json.JSONDecodeError, TypeError):
        return None
    return data.get("rules_version")


def find_uuids_by_rule(
    client: AdwClient,
    namespace: str,
    rule_name: str,
    rules_version: str | None = None,
    require_complete: bool = True,
) -> list[str]:
    """Retrieve all UUIDs from keydata filtered by rule_name.

    Additional optional filters:

    * ``rules_version`` — only keep records whose ``extra_data.rules_version``
      matches the provided value.
    * ``require_complete`` — if ``True`` (default), only keep records whose
      ``extend`` JSON indicates zero missing files and all expected file
      categories are present.

    Results are paginated in chunks of 1000.
    """
    table_name = f"{namespace}/keydata"
    where = f"rule_name='{rule_name}'"
    all_uuids: list[str] = []
    offset = 0
    limit = 1000

    # We need extend (for completeness) and extra_data (for rules_version).
    columns = "uuid,extend,extra_data"

    while True:
        scan = cmm.Scan(
            table_name=table_name,
            where=where,
            columns=columns,
            limit=limit,
            offset=offset,
        )
        results = client.scan(scan)
        if not results:
            break

        for res in results:
            meta = res.get_meta()
            if not meta or "uuid" not in meta:
                continue

            # --- Filter: data completeness ---
            if require_complete:
                extend_json = meta.get("extend", "")
                if not _check_completeness(extend_json):
                    continue

            # --- Filter: rules_version ---
            if rules_version is not None:
                extra_data_json = meta.get("extra_data", "")
                actual_version = _extract_rules_version(extra_data_json)
                if actual_version != rules_version:
                    continue

            all_uuids.append(meta["uuid"])

        if len(results) < limit:
            break
        offset += limit

    return all_uuids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve UUIDs from ADW keydata filtered by rule_name, "
        "with optional rules_version and data-completeness constraints."
    )
    parser.add_argument(
        "--namespace",
        default="production/collection",
        help='ADW namespace (default: "production/collection")',
    )
    parser.add_argument(
        "--rule",
        required=True,
        help='Rule name to filter, e.g. "perception_dust_air"',
    )
    parser.add_argument(
        "--rules-version",
        default=None,
        help="Rules version to filter (matches extra_data.rules_version). "
        'E.g. "rules_cn_0_v190.4"',
    )
    parser.add_argument(
        "--no-completeness-check",
        action="store_true",
        help="By default only complete records (miss_file_cnt==0) are kept. "
        "Use this flag to include incomplete records as well.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help='Output file path (default: auto-generated as "{rule}_{rules_version}_uuids.txt")',
    )
    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        safe_rule = args.rule.replace("/", "_").replace("\\", "_")
        if args.rules_version:
            safe_version = args.rules_version.replace("/", "_").replace("\\", "_")
            args.output = f"{safe_rule}_{safe_version}_uuids.txt"
        else:
            args.output = f"{safe_rule}_uuids.txt"

    client = _build_client_from_env()
    if client is None:
        print(
            "ERROR: ADW credentials not found. "
            "Set ADW_USER and ADW_PASS environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Namespace        : {args.namespace}")
    print(f"Rule             : {args.rule}")
    print(f"Rules version    : {args.rules_version or '(any)'}")
    print(f"Require complete : {not args.no_completeness_check}")
    print(f"Output           : {args.output}")
    print("-" * 40)

    uuids = find_uuids_by_rule(
        client=client,
        namespace=args.namespace,
        rule_name=args.rule,
        rules_version=args.rules_version,
        require_complete=not args.no_completeness_check,
    )

    with open(args.output, mode="w", encoding="utf-8") as f:
        for uid in uuids:
            f.write(f"{uid}\n")

    print(f"Retrieved {len(uuids)} UUID(s), written to {args.output}")


if __name__ == "__main__":
    main()
