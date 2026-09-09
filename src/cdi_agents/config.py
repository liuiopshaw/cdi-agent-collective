"""Configuration loading for CDI Agent Collective.

Configuration is a single YAML file describing LLM providers, role
assignment and per-role temperatures. Secrets are never stored in the
file; each provider names an environment variable that holds its key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    model: str

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass
class RoleConfig:
    provider: str
    temperature: float


@dataclass
class FrameworkConfig:
    providers: dict[str, ProviderConfig]
    roles: dict[str, RoleConfig]
    orchestration: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def role(self, name: str) -> RoleConfig:
        if name not in self.roles:
            raise KeyError(f"unknown role: {name}")
        return self.roles[name]

    def provider_for(self, role_name: str) -> ProviderConfig:
        return self.providers[self.role(role_name).provider]


def load_config(path: str | Path) -> FrameworkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    providers = {
        name: ProviderConfig(name=name, **spec)
        for name, spec in raw.get("providers", {}).items()
    }
    roles = {
        name: RoleConfig(**spec) for name, spec in raw.get("roles", {}).items()
    }
    return FrameworkConfig(
        providers=providers,
        roles=roles,
        orchestration=raw.get("orchestration", {}),
        evaluation=raw.get("evaluation", {}),
        paths=raw.get("paths", {}),
    )
