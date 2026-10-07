import uuid
import shutil
import pytest
from core.sandbox import HardenedPythonSandbox, ASTSecurityViolation


@pytest.fixture
def sandbox():
    """Initializes an isolated sandbox instance for each test."""
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    s = HardenedPythonSandbox(session_id=session_id, timeout=10, max_memory_mb=1024)
    yield s
    # Clean up test session workspace after test
    shutil.rmtree(s.session_dir.parent, ignore_errors=True)


def test_clean_execution(sandbox):
    """Verifies that safe, standard data science code executes successfully."""
    code = (
        "import math\n"
        "val = math.sqrt(144)\n"
        "print(f'Result={val}')\n"
        "with open('output.txt', 'w') as f:\n"
        "    f.write('artifact_created')\n"
    )
    result = sandbox.run_code(code)
    assert result.success is True
    assert "Result=12.0" in result.stdout
    assert "output.txt" in result.artifacts
    assert result.return_code == 0


def test_banned_subprocess_import(sandbox):
    """Verifies that attempting to import subprocess is blocked by AST analysis."""
    code = "import subprocess\nsubprocess.run(['dir'])\n"
    result = sandbox.run_code(code)
    assert result.success is False
    assert result.error_category == "AST_SECURITY_VIOLATION"
    assert "banned module 'subprocess'" in result.error_message


def test_banned_os_system(sandbox):
    """Verifies that calling os.system is intercepted by AST analysis."""
    code = "import os\nos.system('echo hacked')\n"
    result = sandbox.run_code(code)
    assert result.success is False
    assert result.error_category == "AST_SECURITY_VIOLATION"
    assert "os.system" in result.error_message


def test_banned_eval(sandbox):
    """Verifies that calling eval() is intercepted by AST analysis."""
    code = "x = eval('2 + 2')\nprint(x)\n"
    result = sandbox.run_code(code)
    assert result.success is False
    assert result.error_category == "AST_SECURITY_VIOLATION"
    assert "eval()" in result.error_message


def test_banned_socket_import(sandbox):
    """Verifies that importing socket is intercepted by AST analysis."""
    code = "from socket import socket\ns = socket()\n"
    result = sandbox.run_code(code)
    assert result.success is False
    assert result.error_category == "AST_SECURITY_VIOLATION"
    assert "socket" in result.error_message


def test_timeout_termination(sandbox):
    """Verifies that runaway infinite loops are killed when timeout expires."""
    code = (
        "import time\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    # Use short timeout of 2 seconds
    result = sandbox.run_code(code, timeout=2)
    assert result.success is False
    assert result.error_category == "TIMEOUT"
    assert "timed out" in result.stderr.lower()


def test_runtime_error_capture(sandbox):
    """Verifies that Python runtime exceptions are gracefully captured."""
    code = (
        "data = [1, 2, 3]\n"
        "print(data[99])\n"  # IndexError
    )
    result = sandbox.run_code(code)
    assert result.success is False
    assert result.error_category == "RUNTIME_ERROR"
    assert "IndexError" in result.stderr


def test_syntax_error_rejection(sandbox):
    """Verifies that malformed code is flagged immediately."""
    code = "def broken_func(:\n    pass\n"
    result = sandbox.run_code(code)
    assert result.success is False
    assert result.error_category == "AST_SECURITY_VIOLATION"
    assert "Syntax error" in result.error_message
