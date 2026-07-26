"""CLI driver and library for JSON Schema validation.

Wires the three schemas under `docs/spec/schemas/` to their producers and
consumers (`gen_fixtures.py`, `measure_setup.sh`, `measure.sh`,
`analyze.py`). Single source of truth for "is this artifact shape-correct
against its declared schema?".

CLI usage::

    python -m grover_tax.validate_schemas <path>                # auto-detect
    python -m grover_tax.validate_schemas --schema fixture <path>
    python -m grover_tax.validate_schemas --schema setup <path>
    python -m grover_tax.validate_schemas --schema discards <path>

Auto-detect rules (filename-based):

* `fixtures/v*.json` or `*v0.1.json` → `fixture-v0.1`
* `*setup*.json`                     → `setup-v1`
* `discards.log`                      → `discards-v1`  (one record per line)

Exit codes:
  0  — validation passed.
  4  — `FIXTURE.SCHEMA_INVALID` for a fixture; structured error on stderr.
  6  — `REPORT.SCHEMA_INVALID` for a results-side artifact (setup, discards).
  2  — usage / argument error.

The CLI prints `<SUBCODE>: <field-level reason>` per
`docs/spec/04-error-model.md`'s logging convention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from jsonschema import Draft202012Validator, ValidationError  # type: ignore[import-untyped]

from grover_tax.errors import (
    FIXTURE_EXIT_CODE,
    REPORT_EXIT_CODE,
    FixtureSubcode,
    ReportSubcode,
)
from grover_tax.paths import schemas_dir

__all__ = [
    "AUTO_DETECT_NAMES",
    "SCHEMA_NAMES",
    "main",
    "validate_artifact",
    "validate_file",
]

# Map from a short label (`--schema` arg) to a schema file.
SCHEMA_NAMES: Final[dict[str, str]] = {
    "fixture": "fixture-v0.2.schema.json",  # v0.2 is the current default (RFC-0015)
    "fixture-v0.1": "fixture-v0.1.schema.json",
    "fixture-v0.2": "fixture-v0.2.schema.json",
    "fixture-v0.3-iadd": "fixture-v0.3-iadd.schema.json",  # KB-4 (#116)
    "setup": "setup-v1.schema.json",
    "setup-v1": "setup-v1.schema.json",
    "discards": "discards-v1.schema.json",
    "discards-v1": "discards-v1.schema.json",
}

# Exit-code routing per category.
_EXIT_BY_SCHEMA: Final[dict[str, int]] = {
    "fixture-v0.1.schema.json": FIXTURE_EXIT_CODE,
    "fixture-v0.2.schema.json": FIXTURE_EXIT_CODE,
    "fixture-v0.3-iadd.schema.json": FIXTURE_EXIT_CODE,
    "setup-v1.schema.json": REPORT_EXIT_CODE,
    "discards-v1.schema.json": REPORT_EXIT_CODE,
}

_SUBCODE_BY_SCHEMA: Final[dict[str, str]] = {
    "fixture-v0.1.schema.json": FixtureSubcode.SCHEMA_INVALID.value,
    "fixture-v0.2.schema.json": FixtureSubcode.SCHEMA_INVALID.value,
    "fixture-v0.3-iadd.schema.json": FixtureSubcode.SCHEMA_INVALID.value,
    "setup-v1.schema.json": ReportSubcode.SCHEMA_INVALID.value,
    "discards-v1.schema.json": ReportSubcode.SCHEMA_INVALID.value,
}

# Detection rules: ordered tuples of (filename or path substring, schema).
# The first match wins; `discards.log` must be checked before "setup" because
# a runner might one day rename a discards log to include "setup" in its
# prefix and we want the JSON-lines path to take precedence. Fixture-version
# detection by content (the JSON's `version` field) takes precedence over
# filename — see `_detect_fixture_version`.
AUTO_DETECT_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("discards.log", "discards-v1.schema.json"),
    ("setup", "setup-v1.schema.json"),
    ("v0.3-iadd", "fixture-v0.3-iadd.schema.json"),  # before the generic "fixture" fallback
    ("v0.2.json", "fixture-v0.2.schema.json"),
    ("v0.1.json", "fixture-v0.1.schema.json"),
    ("fixture", "fixture-v0.2.schema.json"),  # default to v0.2 for unversioned filenames
)


def _detect_fixture_version(path: Path) -> str | None:
    """Read the JSON file and return the matching fixture schema iff the file
    looks like a fixture (has a `version` key whose value is one of the known
    fixture versions). Returns None if the file is unreadable or unrecognised."""
    try:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(4096)
        # Cheap version probe — we look for `"version": "v0.X"` in the first
        # 4 KiB; the version field is always emitted near the top by the
        # generators. Full JSON parse happens later in validate_file().
        m = re.search(r'"version"\s*:\s*"(v0\.1|v0\.2|v0\.3-iadd)"', head)
        if m is None:
            return None
        return f"fixture-{m.group(1)}.schema.json"
    except OSError:
        return None


def _resolve_schema(*, explicit: str | None, path: Path) -> str:
    """Return the schema filename to use; raise if it can't be determined."""
    if explicit is not None:
        if explicit not in SCHEMA_NAMES:
            raise SystemExit(
                f"validate_schemas: unknown --schema {explicit!r}; "
                f"valid: {sorted(set(SCHEMA_NAMES))}"
            )
        return SCHEMA_NAMES[explicit]

    # Content-based detection wins over filename: a file named `v0.1.json`
    # but containing `"version": "v0.2"` should route to the v0.2 schema.
    # gen_fixtures.py tests in CI write to `tmp_path / "v0.1.json"` regardless
    # of which version the generator currently emits.
    if path.is_file():
        content_match = _detect_fixture_version(path)
        if content_match is not None:
            return content_match

    # Search both the filename and the full path so that files under
    # `python/tests/fixtures/bad/` match by parent-directory hint.
    haystack = str(path)
    for hint, schema in AUTO_DETECT_NAMES:
        if hint in haystack:
            return schema
    raise SystemExit(
        f"validate_schemas: cannot auto-detect schema for {path.name!r}; "
        f"pass --schema explicitly. Valid labels: {sorted(set(SCHEMA_NAMES))}"
    )


def _load_schema(filename: str) -> dict[str, object]:
    """Load and cache one schema file by basename."""
    schema_path = schemas_dir() / filename
    data: dict[str, object] = json.loads(schema_path.read_text(encoding="utf-8"))
    return data


def validate_artifact(*, document: object, schema_filename: str) -> list[ValidationError]:
    """Validate one in-memory JSON document against the named schema.

    Returns the (possibly empty) list of `ValidationError`s. An empty
    list means the document validated.
    """
    schema = _load_schema(schema_filename)
    validator = Draft202012Validator(schema)
    return list(validator.iter_errors(document))


def validate_file(path: Path, *, schema_filename: str) -> list[ValidationError]:
    """Validate a file against a schema. Each line of `discards.log` is one record."""
    if schema_filename == "discards-v1.schema.json":
        return _validate_jsonl_file(path, schema_filename=schema_filename)
    document = json.loads(path.read_text(encoding="utf-8"))
    return validate_artifact(document=document, schema_filename=schema_filename)


def _validate_jsonl_file(path: Path, *, schema_filename: str) -> list[ValidationError]:
    """Per-line validation for the discards JSON-lines log."""
    errors: list[ValidationError] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        document = json.loads(raw)
        line_errors = validate_artifact(document=document, schema_filename=schema_filename)
        # Prefix each error path with the line number so the human report points
        # at the right record.
        for err in line_errors:
            err.message = f"line {line_number}: {err.message}"
        errors.extend(line_errors)
    return errors


def _format_errors(errors: Iterable[ValidationError]) -> str:
    out: list[str] = []
    for err in errors:
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        out.append(f"  at {path}: {err.message}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m grover_tax.validate_schemas`."""
    parser = argparse.ArgumentParser(
        prog="validate_schemas",
        description="Validate a fixture, setup record, or discards log against its schema.",
    )
    parser.add_argument("path", type=Path, help="Artifact file to validate.")
    parser.add_argument(
        "--schema",
        choices=sorted(SCHEMA_NAMES),
        default=None,
        help="Force a specific schema; otherwise inferred from the filename.",
    )
    args = parser.parse_args(argv)

    try:
        schema_filename = _resolve_schema(explicit=args.schema, path=args.path)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    if not args.path.is_file():
        subcode = _SUBCODE_BY_SCHEMA[schema_filename]
        print(f"{subcode}: file not found: {args.path}", file=sys.stderr)
        return _EXIT_BY_SCHEMA[schema_filename]

    try:
        errors = validate_file(args.path, schema_filename=schema_filename)
    except json.JSONDecodeError as e:
        subcode = _SUBCODE_BY_SCHEMA[schema_filename]
        print(f"{subcode}: not valid JSON: {e}", file=sys.stderr)
        return _EXIT_BY_SCHEMA[schema_filename]

    if not errors:
        return 0

    subcode = _SUBCODE_BY_SCHEMA[schema_filename]
    print(f"{subcode}: {len(errors)} validation error(s) in {args.path}", file=sys.stderr)
    print(_format_errors(errors), file=sys.stderr)
    return _EXIT_BY_SCHEMA[schema_filename]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
