# HIERARCHY.md

> Repo map + ownership for Galaxy-X-os

```
Galaxy-X-os/
├── README.md               # Entry point
├── CLAUDE.md               # ★ Orchestrator kernel (auto-loaded)
├── KIMI.md                 # ★ Identical to CLAUDE.md
├── AGENTS.md               # Alias
├── HANDOFF.md              # ★ Session state
├── HIERARCHY.md            # ★ This file
├── HOW_TO_RUN.md           # Workflow guide
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guide
├── OS_SETUP.md             # Methodology source
│
├── .claude/                # Claude Code settings
│
├── orchestrator/           # TIER 1 apparatus
│   ├── ROLE.md
│   ├── core/               # Governance files
│   ├── commands/           # Slash commands
│   ├── skills/             # SKILL.md files
│   ├── agents/             # Sub-agent definitions
│   ├── hooks/              # Auto-action scripts
│   ├── recipes/            # Parameterized workflows
│   ├── rules/              # Path-scoped rules
│   ├── memory/             # MEMORY.md + states + session log
│   └── scripts/            # Utility scripts
│
├── evals/                  # Eval-driven development
│
├── work/                   # ★ THE BRIDGE
│   ├── TASK_TEMPLATE.md
│   ├── REPORT_TEMPLATE.md
│   ├── WORKER_PROMPT.md
│   └── wave-1/             # Task files per wave
│
├── plan/                   # 3 LIVING strategic docs
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── EXECUTION.md
│
├── .specify/               # Spec-driven
│
├── docs/                   # Documentation
│
├── prompts/                # Evolving prompts
│
├── attic/                  # Archive (never delete)
│
├── src/                    # Source code
├── app/                    # Streamlit demo
├── notebooks/              # Jupyter notebooks
├── config/                 # YAML configuration
├── data/                   # Datasets
├── checkpoints/            # Model weights
├── results/                # Outputs
├── tests/                  # Test suites
├── models/                 # Versioned models
├── scripts/                # Utility scripts
├── deployment/             # IaC
└── resources/              # Reference materials
```
