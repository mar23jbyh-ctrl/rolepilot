"""Offline checks for the cleaned source layout and current documentation."""
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location('layout_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_retained_markdown_local_links_exist():
    broken = []
    for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]:
        text = path.read_text(encoding='utf-8-sig')
        for match in re.finditer(r'\]\(([^)]+)\)', text):
            url = match.group(1).strip().strip('<>')
            if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:|^#', url):
                continue
            target = unquote(url.split('#', 1)[0])
            if target and not (path.parent / target).exists():
                broken.append((path.relative_to(ROOT).as_posix(), target))
    assert not broken


def test_current_project_paths_and_versions():
    from app.config import PROJECT_ROOT, settings
    from app.models.schemas import SCHEMA_VERSION
    from app.telemetry.summary import summarize_usage
    assert PROJECT_ROOT.resolve() == ROOT
    assert settings.data_root.resolve().is_relative_to(ROOT)
    assert SCHEMA_VERSION == '3.4.0'
    assert summarize_usage([])['schema_version'] == 'usage-v1'


def test_current_inventory_keeps_public_readme_without_archives():
    inventory = load_script('acceptance_cleanup').build_inventory()
    entries = {row['path']: row for row in inventory['files']}
    assert entries['README.md']['class'] == 'KEEP'
    assert not any(row['class'] == 'ARCHIVE' for row in inventory['files'])


def test_inventory_default_does_not_write(tmp_path, monkeypatch, capsys):
    inventory = load_script('acceptance_cleanup')
    monkeypatch.setattr(inventory, 'ROOT', tmp_path)
    assert inventory.main([]) == 0
    assert not list(tmp_path.iterdir())
    assert json.loads(capsys.readouterr().out)['files'] == []


def test_finish_works_without_optional_release_metadata(tmp_path, monkeypatch, capsys):
    finish = load_script('acceptance_finish')
    for relative in ('README.md', '.env.example', 'requirements.txt',
                     'frontend/package.json', 'frontend/package-lock.json'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
    monkeypatch.setattr(finish, 'ROOT', tmp_path)
    assert finish.main([]) == 0
    output = tmp_path / 'docs/release/evidence/my-source-validation.json'
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['broken_local_links'] == []
    assert set(result['source_sha256']) == {
        'README.md', '.env.example', 'requirements.txt',
        'frontend/package.json', 'frontend/package-lock.json',
    }
    assert 'supplied_offline_result' not in result
    assert 'cleanup' not in result
    assert json.loads(capsys.readouterr().out)['source_files'] == 5


def test_runtime_dependencies_and_regression_fixture_are_retained():
    for path in ('api/main.py', 'app/service.py', 'app/graph/builder.py',
                 'frontend/package-lock.json', 'app/prompts/references.py',
                 'tests/fixtures/intent/replay-metadata.json', 'tests/fixtures/guard/cases.json'):
        assert (ROOT / path).is_file()
