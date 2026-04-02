# /scaffold — 신규 프로젝트 초기화

새 프로젝트의 전체 디렉토리 구조를 생성합니다.

## 사용자에게 확인할 것
1. **프로젝트 이름**: 예) `video-curation-pipeline`
2. **프로젝트 타입**: `data-curation` / `model-training` / `general`

`$ARGUMENTS`에 이름이 있으면 그걸 사용. 없으면 물어보세요.

## 생성할 구조

```
{프로젝트명}/
├── CLAUDE.md
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
│
├── .claude/
│   ├── settings.json
│   ├── agents/   (planner, developer, tester, reviewer, recommender, analyzer, atlassian-manager, github-manager, researcher, doc-writer, knowledge-writer, migrator)
│   ├── commands/ (task-plan, scaffold, restore, check, auto, issue, migrate)
│   └── skills/   (version-control, plan-writing, review-rules, testing, error-logging, profiling, debug, dependency-check, research, doc-update, jira-workflow, machine-profile, knowledge-management, migration)
│
├── .plans/            (.gitkeep)
├── .manuals/errors/   (.gitkeep)
├── configs/           (.gitkeep)
├── docs/              (.gitkeep)
├── src/__init__.py
├── scripts/           (.gitkeep)
├── tests/             (.gitkeep)
├── results/           (.gitkeep)
├── experiments/       (.gitkeep + TEMPLATE.md)
└── tmp/sandbox/
```

## 타입별 차이
- `data-curation` / `model-training`: 해당 타입에 맞는 도메인 skill을 `.claude/skills/`에 직접 추가
- `general`: 기본 skill 세트만 포함 (추가 도메인 skill 없음)

## pyproject.toml 템플릿
```toml
[project]
name = "{프로젝트명}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[tool.uv]
dev-dependencies = ["pytest", "ruff"]
```

## .gitignore 기본값
```
.plans/
.manuals/
.claude/settings.local.json
tmp/
.env
*.pyc
__pycache__/
results/
experiments/**/outputs/
experiments/**/checkpoints/
experiments/**/*.pt
experiments/**/*.ckpt
```

## 완료 후 보고
```
✅ 프로젝트 '{프로젝트명}' 초기화 완료
📁 위치: ./{프로젝트명}/
▶️ 시작: cd {프로젝트명} && claude
```
