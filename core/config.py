"""
Configuration module for Autonomous AI Data Scientist.
Loads environment variables and sets system-wide defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

# LLM Model Configurations
DEFAULT_REASONING_MODEL = os.getenv("GEMINI_REASONING_MODEL", "gemini-2.5-pro")
DEFAULT_FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash")

# Sandbox Guardrails
SANDBOX_DEFAULT_TIMEOUT = int(os.getenv("SANDBOX_DEFAULT_TIMEOUT", "60"))
SANDBOX_MAX_MEMORY_MB = int(os.getenv("SANDBOX_MAX_MEMORY_MB", "4096"))

# Workspace Path
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Research Paths (Phase 5)
RESEARCH_DIR = PROJECT_ROOT / "research"
RESEARCH_EXPERIMENTS_DIR = RESEARCH_DIR / "experiments"
RESEARCH_RESULTS_DIR = RESEARCH_DIR / "results"
RESEARCH_FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_REPORTS_DIR = RESEARCH_DIR / "reports"
for _d in [RESEARCH_EXPERIMENTS_DIR, RESEARCH_RESULTS_DIR, RESEARCH_FIGURES_DIR, RESEARCH_REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
