$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$artifactRoot = Join-Path $repo "artifacts\tests"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $artifactRoot "system") | Out-Null

python -m pip install -r requirements-test.txt

python -m pytest tests/unit -m unit --junitxml "$artifactRoot\unit_junit.xml" 2>&1 | Tee-Object -FilePath "$artifactRoot\unit.log"
python -m pytest tests/integration -m integration --junitxml "$artifactRoot\integration_junit.xml" 2>&1 | Tee-Object -FilePath "$artifactRoot\integration.log"
python -m pytest tests/system -m system --junitxml "$artifactRoot\system_junit.xml" 2>&1 | Tee-Object -FilePath "$artifactRoot\system.log"

@'
import xml.etree.ElementTree as ET
from pathlib import Path
root = Path("artifacts/tests")
rows = []
for name in ["unit", "integration", "system"]:
    p = root / f"{name}_junit.xml"
    t = ET.parse(p).getroot()
    suite = t.find("testsuite") if t.tag == "testsuites" else t
    tests = int(suite.attrib.get("tests", 0))
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    passed = tests - failures - errors - skipped
    rows.append((name, tests, passed, failures, errors, skipped))
out = Path("qa/test_report.md")
lines = [
    "# 試験結果レポート",
    "",
    "## 実行サマリ",
    "",
    "| レベル | Tests | Passed | Failures | Errors | Skipped |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
lines += [
    "",
    "## エビデンス",
    "",
    "- `artifacts/tests/unit_junit.xml`",
    "- `artifacts/tests/integration_junit.xml`",
    "- `artifacts/tests/system_junit.xml`",
    "- `artifacts/tests/unit.log`",
    "- `artifacts/tests/integration.log`",
    "- `artifacts/tests/system.log`",
    "- `artifacts/tests/system/step1_loaded.png`",
    "- `artifacts/tests/system/step2_path_browser_opened.png`",
    "- `artifacts/tests/system/step3_path_applied.png`",
    "- `artifacts/tests/system/step4_model_thumbnail.png`",
]
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out}")
'@ | python -
