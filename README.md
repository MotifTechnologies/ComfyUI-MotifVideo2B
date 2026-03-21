# Claude Code 설정 키트

Claude Code CLI 커스텀 설정. Hook, Agent, Skill, Manual 아키텍처로 LLM 코딩 품질을 제어한다.

## 설치

### 빈 프로젝트

```bash
# clone 후 직접 실행
git clone https://github.com/MotifTechnologies/claude-code-kit.git
cd my-project
bash /path/to/claude-code-kit/setup.sh
```

### 기존 프로젝트

```bash
# --src로 template 경로 지정
bash setup.sh --src /path/to/claude-code-kit/template
```

> setup.sh는 기존 파일을 절대 덮어쓰지 않는다. 없는 파일만 생성.

## 머지 전략

기존 `.claude/` 설정이 있는 프로젝트에서의 동작:

| 파일 유형 | 기존 존재 시 동작 |
|-----------|-----------------|
| CLAUDE.md | `.new` 파일 생성 -- 수동 병합 필요 |
| settings.json | 스킵 (기존 유지) |
| agents/, skills/, hooks/, commands/ | 스킵 (기존 유지) |
| .manuals/ 하위 파일 | 스킵 (기존 유지) |

## 커스터마이징 포인트

- **CLAUDE.md**: `## 프로젝트` 섹션에 프로젝트 설명 작성
- **mirror-check.sh**: `src/` 구조가 없으면 자동 스킵 (가드 내장)
- **.manuals/**: 프로젝트에 맞게 dev/, process/ 하위 내용 수정 가능
- **agents/**: 불필요한 에이전트 파일 삭제 가능

## 설치 후 확인

```bash
# 1. Claude Code 실행
claude

# 2. SessionStart hook 동작 확인
#    - 머신 프로파일 자동 생성 (.manuals/env/)
#    - 상태 메시지 출력
```

## 디렉토리 구조 (설치 후)

```
my-project/
├── CLAUDE.md              # 전역 규칙 (매 세션 로드)
├── .claude/
│   ├── settings.json      # hooks + permissions
│   ├── hooks/             # shell scripts
│   ├── agents/            # agent 정의
│   ├── commands/          # /slash commands
│   └── skills/            # 경량 목차
├── .manuals/
│   ├── dev/               # 개발 실무 규칙
│   ├── process/           # 프로세스 규칙
│   ├── knowledge/         # 축적형 지식
│   ├── env/               # 머신 환경 캐시
│   └── templates/         # 문서 템플릿
├── .plans/                # 작업 기억 저장소
└── experiments/           # 실험 관리
```

## FAQ

**Q: 기존 CLAUDE.md가 있을 때?**
A: `CLAUDE.md.new`로 생성된다. 기존 파일과 수동 병합 필요.

**Q: 특정 에이전트/스킬만 사용하고 싶을 때?**
A: 설치 후 `.claude/agents/`, `.claude/skills/`에서 불필요한 파일 삭제.

**Q: `src/` 구조가 없는 프로젝트에서 mirror-check 경고가 뜨나?**
A: 아니다. `mirror-check.sh`에 `src/` 존재 여부 가드가 있어 자동 스킵된다.

**Q: setup.sh를 두 번 실행하면?**
A: 멱등성 보장. 이미 존재하는 파일은 스킵되고, 없는 파일만 추가 생성.

**Q: .plans/와 .manuals/는 git에 올라가나?**
A: setup.sh가 `.gitignore`에 자동 추가한다. 필요 시 변경 가능.
