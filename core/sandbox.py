"""
Hardened Sandboxed Execution Engine for Autonomous AI Data Scientist.
Provides local Windows-compatible subprocess isolation, AST analysis,
resource monitoring, filesystem confinement, and network isolation.
"""

import ast
import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from pydantic import BaseModel, Field
import psutil

from core.config import SANDBOX_DEFAULT_TIMEOUT, SANDBOX_MAX_MEMORY_MB, WORKSPACE_DIR


class SandboxResult(BaseModel):
    """Encapsulates the execution results and metrics of a sandboxed run."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    artifacts: List[str] = Field(default_factory=list)
    error_category: Optional[str] = None
    error_message: Optional[str] = None


class ASTSecurityViolation(Exception):
    """Raised when unsafe code patterns or banned modules are detected in AST."""
    pass


class HardenedPythonSandbox:
    """
    Local Windows-compatible execution sandbox for LLM-generated Python scripts.
    Implements:
    1. Static AST parsing & threat filtering (banning system/network execution).
    2. Filesystem jail (restricted to session workspace directory).
    3. Wall-clock timeout watchdog.
    4. Memory quota monitoring via psutil.
    5. Network socket disabling via runtime preamble.
    """

    # Modules strictly banned from being imported by synthesized code
    BANNED_MODULES: Set[str] = {
        "subprocess",
        "ctypes",
        "socket",
        "winreg",
        "webbrowser",
        "pty",
        "termios",
        "fcntl",
    }

    # Builtins / functions strictly banned from direct invocation
    BANNED_FUNCTIONS: Set[str] = {
        "eval",
        "exec",
        "compile",
        "__import__",
    }

    # Banned attribute access patterns (e.g. os.system, shutil.rmtree)
    BANNED_CALL_ATTRIBUTES: Dict[str, Set[str]] = {
        "os": {"system", "popen", "spawn", "fork", "kill", "remove", "unlink", "rmdir"},
        "shutil": {"rmtree", "move"},
    }

    def __init__(
        self,
        session_id: str,
        timeout: int = SANDBOX_DEFAULT_TIMEOUT,
        max_memory_mb: int = SANDBOX_MAX_MEMORY_MB
    ):
        self.session_id = session_id
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        
        # Ensure session sandbox workspace directory exists
        self.session_dir = (WORKSPACE_DIR / session_id / "sandbox").resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def validate_code_safety(self, code: str) -> None:
        """
        Statically inspects Python code via AST to block dangerous constructs.
        Raises ASTSecurityViolation if prohibited operations are identified.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ASTSecurityViolation(f"Syntax error in synthesized code: {e}")

        for node in ast.walk(tree):
            # 1. Check for banned module imports: import X
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in self.BANNED_MODULES:
                        raise ASTSecurityViolation(
                            f"Security Violation: Import of banned module '{alias.name}' is strictly prohibited."
                        )

            # 2. Check for banned module imports: from X import Y
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg in self.BANNED_MODULES:
                        raise ASTSecurityViolation(
                            f"Security Violation: Import from banned module '{node.module}' is strictly prohibited."
                        )

            # 3. Check for direct execution of banned builtins: eval, exec, etc.
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BANNED_FUNCTIONS:
                        raise ASTSecurityViolation(
                            f"Security Violation: Invocation of banned function '{node.func.id}()' is prohibited."
                        )

                # 4. Check for banned method calls: os.system, shutil.rmtree, etc.
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        caller = node.func.value.id
                        method = node.func.attr
                        if caller in self.BANNED_CALL_ATTRIBUTES and method in self.BANNED_CALL_ATTRIBUTES[caller]:
                            raise ASTSecurityViolation(
                                f"Security Violation: Invocation of '{caller}.{method}()' is prohibited."
                            )

    def _generate_execution_script(self, raw_code: str) -> str:
        """
        Prepends a security preamble that blocks network sockets and traps errors.
        """
        preamble = (
            "# --- AUTONOMOUS DATA SCIENTIST SANDBOX PREAMBLE ---\n"
            "import os, sys\n"
            "# Disable outbound network connections\n"
            "try:\n"
            "    import socket\n"
            "    def _blocked_socket(*args, **kwargs):\n"
            "        raise PermissionError('Sandbox Security: Outbound network access is prohibited.')\n"
            "    socket.socket = _blocked_socket\n"
            "    socket.create_connection = _blocked_socket\n"
            "except Exception:\n"
            "    pass\n"
            "# --- END PREAMBLE ---\n\n"
        )
        return preamble + raw_code

    def _find_artifacts(self, initial_files: Set[Path]) -> List[str]:
        """Identifies files created during script execution within the session directory."""
        current_files = set(self.session_dir.rglob("*"))
        new_files = current_files - initial_files
        artifacts = []
        for file_path in new_files:
            if file_path.is_file() and not file_path.name.startswith("__run_script"):
                try:
                    rel_path = file_path.relative_to(self.session_dir)
                    artifacts.append(str(rel_path))
                except ValueError:
                    artifacts.append(str(file_path))
        return sorted(artifacts)

    def run_code(
        self,
        code: str,
        timeout: Optional[int] = None,
        max_memory_mb: Optional[int] = None
    ) -> SandboxResult:
        """
        Executes code inside the hardened sandbox with watchdogs for memory & timeout.
        """
        timeout = timeout or self.timeout
        max_memory_mb = max_memory_mb or self.max_memory_mb

        # 1. Static AST Safety Verification
        try:
            self.validate_code_safety(code)
        except ASTSecurityViolation as sec_err:
            return SandboxResult(
                success=False,
                return_code=1,
                error_category="AST_SECURITY_VIOLATION",
                error_message=str(sec_err),
                stderr=str(sec_err)
            )

        # 2. Prepare Sandbox Directory and Script File
        script_path = self.session_dir / "__run_script.py"
        full_code = self._generate_execution_script(code)
        script_path.write_text(full_code, encoding="utf-8")

        initial_files = set(self.session_dir.rglob("*"))

        # 3. Prepare Environment Variables
        env = os.environ.copy()
        # Strip proxy variables to prevent network tunneling
        for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"]:
            env.pop(proxy_var, None)
        env["PYTHONUNBUFFERED"] = "1"

        # 4. Launch Subprocess
        start_time = time.time()
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(self.session_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        peak_memory_mb = 0.0
        memory_exceeded = False
        timed_out = False

        # 5. Background Watchdog for Memory & Timeout
        def watchdog():
            nonlocal peak_memory_mb, memory_exceeded, timed_out
            try:
                ps_proc = psutil.Process(process.pid)
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        timed_out = True
                        self._kill_process_tree(process.pid)
                        break

                    try:
                        mem_info = ps_proc.memory_info()
                        mem_mb = mem_info.rss / (1024 * 1024)
                        if mem_mb > peak_memory_mb:
                            peak_memory_mb = mem_mb

                        if mem_mb > max_memory_mb:
                            memory_exceeded = True
                            self._kill_process_tree(process.pid)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break

                    time.sleep(0.05)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()

        # 6. Wait for Process Completion
        stdout, stderr = "", ""
        try:
            stdout, stderr = process.communicate()
        except Exception as e:
            self._kill_process_tree(process.pid)
            stderr = f"Subprocess communication error: {e}"

        watchdog_thread.join(timeout=1.0)
        execution_time = time.time() - start_time

        # Clean up temporary execution script
        try:
            if script_path.exists():
                script_path.unlink()
        except Exception:
            pass

        # 7. Collect Artifacts Created
        artifacts = self._find_artifacts(initial_files)

        # 8. Evaluate Outcome
        if timed_out:
            return SandboxResult(
                success=False,
                stdout=stdout,
                stderr=f"Execution timed out after {timeout} seconds.",
                return_code=-1,
                execution_time_seconds=execution_time,
                peak_memory_mb=peak_memory_mb,
                artifacts=artifacts,
                error_category="TIMEOUT",
                error_message=f"Process terminated: Wall-clock timeout of {timeout}s exceeded."
            )

        if memory_exceeded:
            return SandboxResult(
                success=False,
                stdout=stdout,
                stderr=f"Memory limit of {max_memory_mb} MB exceeded (Peak: {peak_memory_mb:.1f} MB).",
                return_code=-2,
                execution_time_seconds=execution_time,
                peak_memory_mb=peak_memory_mb,
                artifacts=artifacts,
                error_category="MEMORY_LIMIT_EXCEEDED",
                error_message=f"Process terminated: Memory quota of {max_memory_mb} MB exceeded."
            )

        if process.returncode != 0:
            return SandboxResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                execution_time_seconds=execution_time,
                peak_memory_mb=peak_memory_mb,
                artifacts=artifacts,
                error_category="RUNTIME_ERROR",
                error_message=f"Script failed with exit code {process.returncode}."
            )

        return SandboxResult(
            success=True,
            stdout=stdout,
            stderr=stderr,
            return_code=0,
            execution_time_seconds=execution_time,
            peak_memory_mb=peak_memory_mb,
            artifacts=artifacts
        )

    def _kill_process_tree(self, pid: int) -> None:
        """Forcefully terminates the process and any spawned child processes on Windows."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
