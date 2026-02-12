#!/usr/bin/env python3
"""UC-native smoke evaluation.

Deterministic retrieval checks over a synthetic in-memory dataset.
Used as a lightweight regression gate in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CASES_PATH = ROOT / "eval" / "uc_smoke_cases.json"


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: str


def _rid(value: object) -> str:
    return str(value)


async def _run_smoke(cases_path: Path) -> tuple[bool, list[CaseResult]]:
    from universal_context.db.client import UCDatabase
    from universal_context.db.queries import (
        create_artifact,
        create_scope,
        hybrid_search,
        search_artifacts,
        semantic_search,
        store_embedding,
    )
    from universal_context.db.schema import apply_schema

    db = UCDatabase.in_memory()
    await db.connect()
    await apply_schema(db)

    try:
        # Two scopes to validate project scoping behavior.
        scope_a = await create_scope(db, "alpha", "/tmp/alpha")
        scope_b = await create_scope(db, "beta", "/tmp/beta")
        sid_a = _rid(scope_a["id"])
        sid_b = _rid(scope_b["id"])

        # Seed summaries in each scope.
        a_auth = await create_artifact(
            db,
            kind="summary",
            content="Implemented JWT auth middleware with token refresh.",
            scope_id=sid_a,
        )
        a_db = await create_artifact(
            db,
            kind="summary",
            content="Added SurrealDB schema migration for run and turn tables.",
            scope_id=sid_a,
        )
        b_ui = await create_artifact(
            db,
            kind="summary",
            content="Refined dashboard typography and navigation layout.",
            scope_id=sid_b,
        )

        # Deterministic vectors (3D) for semantic/hybrid checks.
        await store_embedding(db, a_auth, [1.0, 0.0, 0.0])
        await store_embedding(db, a_db, [0.0, 1.0, 0.0])
        await store_embedding(db, b_ui, [0.0, 0.0, 1.0])

        scope_map = {"alpha": sid_a, "beta": sid_b}
        label_map = {
            "auth_summary": a_auth,
            "db_summary": a_db,
            "ui_summary": b_ui,
        }

        case_specs = json.loads(cases_path.read_text(encoding="utf-8"))
        cases: list[CaseResult] = []
        for spec in case_specs:
            name = spec["name"]
            mode = spec["mode"]
            query = spec.get("query", "")
            scope_key = spec.get("scope", "alpha")
            scope_id = scope_map[scope_key]

            results: list[dict[str, object]] = []
            if mode == "keyword":
                results = await search_artifacts(db, query, kind="summary", scope_id=scope_id)
            elif mode == "semantic":
                results = await semantic_search(
                    db,
                    spec["query_embedding"],
                    kind="summary",
                    scope_id=scope_id,
                )
            elif mode == "hybrid":
                results = await hybrid_search(
                    db,
                    query,
                    spec["query_embedding"],
                    kind="summary",
                    scope_id=scope_id,
                    limit=5,
                )
            else:
                raise ValueError(f"Unsupported case mode: {mode}")

            if "expected_count" in spec:
                expected_count = int(spec["expected_count"])
                passed = len(results) == expected_count
                details = f"count={len(results)} expected={expected_count}"
            else:
                expected_id = label_map[spec["expected_label"]]
                top = _rid(results[0]["id"]) if results else "none"
                passed = bool(results) and top == expected_id
                details = f"top={top} expected={expected_id}"

            cases.append(CaseResult(name=name, passed=passed, details=details))

        ok = all(c.passed for c in cases)
        return ok, cases
    finally:
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UC smoke retrieval eval.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="JSON file containing smoke case specs",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write JSON report to this path",
    )
    args = parser.parse_args()

    ok, cases = asyncio.run(_run_smoke(args.cases))
    payload = {
        "ok": ok,
        "cases": [{"name": c.name, "passed": c.passed, "details": c.details} for c in cases],
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
