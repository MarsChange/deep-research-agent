"""
E2B Code Sandbox tool for executing Python code in isolated environment.

Based on miroflow's implementation, simplified for web data processing.

Allows the agent to:
1. Create and manage sandboxes
2. Execute Python code safely in a sandboxed environment
3. Download files from URLs and process them
4. Process data from Playwright browser sessions (tables, files, etc.)

Environment variables:
- E2B_API_KEY: E2B API key (required)
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Get API key
E2B_API_KEY = os.getenv("E2B_API_KEY")

# Default timeout in seconds
DEFAULT_TIMEOUT = 300  # 5 minutes

# Common packages pre-installed or to install
COMMON_PACKAGES = [
    "pandas",
    "numpy",
    "openpyxl",  # for Excel files
    "xlrd",  # for old Excel files
    "beautifulsoup4",  # for HTML parsing
    "lxml",  # for XML/HTML parsing
    "requests",  # for HTTP requests
    "tabulate",  # for table formatting
]


async def create_sandbox() -> Optional[str]:
    """
    Create a Linux sandbox and get the sandbox_id for safely executing code.

    The sandbox may timeout and automatically shutdown. If so, you will need
    to create a new sandbox.

    IMPORTANT: Do not execute create_sandbox and other sandbox tools in the
    same turn. You must wait for create_sandbox to return the sandbox_id,
    then use that sandbox_id in subsequent turns.

    Returns:
        The sandbox_id of the newly created sandbox. Use this sandbox_id
        to run other tools in the sandbox.
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY environment variable is required"

    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return "Error: e2b-code-interpreter package not installed. Run: pip install e2b-code-interpreter"

    max_retries = 3
    sandbox = None

    for attempt in range(1, max_retries + 1):
        try:
            sandbox = Sandbox.create()
            sandbox_id = sandbox.sandbox_id

            # Install common packages
            logger.info(f"Installing common packages in sandbox {sandbox_id}")
            packages_str = " ".join(COMMON_PACKAGES)
            sandbox.commands.run(f"pip install -q {packages_str}")

            return f"Sandbox created successfully. sandbox_id: {sandbox_id}"

        except Exception as e:
            if sandbox:
                try:
                    sandbox.kill()
                except Exception:
                    pass
            if attempt == max_retries:
                return f"Error: Failed to create sandbox after {max_retries} attempts: {e}"
            await asyncio.sleep(attempt * 2)  # Exponential backoff


async def run_python_code(sandbox_id: str, code: str) -> Optional[str]:
    """
    Run Python code in the sandbox and return the execution result.

    The sandbox has common packages pre-installed: pandas, numpy, openpyxl,
    beautifulsoup4, lxml, requests, tabulate.

    Use this tool to:
    - Process HTML tables extracted from Playwright browser sessions
    - Parse and analyze data structures
    - Perform calculations and data transformations
    - Process downloaded files (CSV, Excel, JSON, etc.)

    Args:
        sandbox_id: The ID of the existing sandbox (from create_sandbox).
        code: The Python code to run. Use print() to output results.

    Returns:
        Execution result including stdout, stderr, and any errors.

    Example:
        code = '''
        import pandas as pd
        from bs4 import BeautifulSoup

        # Parse HTML table from Playwright
        html = "<table><tr><td>Name</td><td>Value</td></tr></table>"
        soup = BeautifulSoup(html, 'html.parser')

        # Extract and process data
        tables = pd.read_html(str(soup))
        print(tables[0])
        '''
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY environment variable is required"

    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return "Error: e2b-code-interpreter package not installed"

    sandbox = None
    try:
        sandbox = Sandbox.connect(sandbox_id)
    except Exception as e:
        return f"Error: Failed to connect to sandbox {sandbox_id}. Make sure the sandbox exists and hasn't timed out. Details: {e}"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Refresh timeout
            sandbox.set_timeout(DEFAULT_TIMEOUT)

            # Execute code
            execution = sandbox.run_code(code)

            # Format result
            results = []

            # Stdout
            if execution.logs.stdout:
                stdout_text = "".join(execution.logs.stdout)
                if stdout_text.strip():
                    results.append(f"**Output:**\n```\n{stdout_text}\n```")

            # Stderr
            if execution.logs.stderr:
                stderr_text = "".join(execution.logs.stderr)
                if stderr_text.strip():
                    results.append(f"**Stderr:**\n```\n{stderr_text}\n```")

            # Results (for expressions)
            if execution.results:
                for result in execution.results:
                    if hasattr(result, "text") and result.text:
                        results.append(f"**Result:**\n```\n{result.text}\n```")

            # Error
            if execution.error:
                error_msg = f"**Error:** {execution.error.name}: {execution.error.value}"
                if execution.error.traceback:
                    error_msg += f"\n```\n{execution.error.traceback}\n```"
                results.append(error_msg)

            if not results:
                return "Code executed successfully (no output)"

            return "\n\n".join(results)

        except Exception as e:
            if attempt == max_retries:
                return f"Error: Failed to run code after {max_retries} attempts: {e}"
            await asyncio.sleep(attempt * 2)
        finally:
            try:
                sandbox.set_timeout(DEFAULT_TIMEOUT)
            except Exception:
                pass


async def download_file_to_sandbox(
    sandbox_id: str,
    url: str,
    filename: Optional[str] = None
) -> Optional[str]:
    """
    Download a file from the internet to the sandbox for processing.

    Use this when you need to download and process files found during
    Playwright browser sessions (Excel, CSV, PDF, etc.).

    Args:
        sandbox_id: The ID of the existing sandbox (from create_sandbox).
        url: The URL of the file to download.
        filename: Optional filename to save as. If not provided, uses the
                 filename from URL.

    Returns:
        The path of the downloaded file in the sandbox if successful.

    Example workflow:
        1. Use Playwright to find a download link for an Excel file
        2. Use download_file_to_sandbox to download it
        3. Use run_python_code to process the file with pandas
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY environment variable is required"

    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return "Error: e2b-code-interpreter package not installed"

    sandbox = None
    try:
        sandbox = Sandbox.connect(sandbox_id)
    except Exception as e:
        return f"Error: Failed to connect to sandbox {sandbox_id}. Details: {e}"

    try:
        sandbox.set_timeout(DEFAULT_TIMEOUT)

        # Determine filename
        if not filename:
            filename = os.path.basename(url.split("?")[0]) or "downloaded_file"

        download_path = f"/home/user/{filename}"

        # Download with wget
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            result = sandbox.commands.run(f'wget -q "{url}" -O {download_path}')
            if result.exit_code == 0:
                # Verify file exists
                check = sandbox.commands.run(f"ls -la {download_path}")
                return f"File downloaded successfully to: {download_path}\n\n{check.stdout}"
            elif attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"Error: Failed to download file from {url} after {max_retries} attempts. Exit code: {result.exit_code}, stderr: {result.stderr}"

    except Exception as e:
        return f"Error: Failed to download file: {e}"
    finally:
        try:
            sandbox.set_timeout(DEFAULT_TIMEOUT)
        except Exception:
            pass


async def run_shell_command(sandbox_id: str, command: str) -> Optional[str]:
    """
    Execute a shell command in the Linux sandbox.

    Use this for system-level operations like:
    - Installing additional packages (pip install, apt-get)
    - File operations (ls, cat, head, tail)
    - Running command-line tools

    Args:
        sandbox_id: The ID of the existing sandbox (from create_sandbox).
        command: The shell command to execute.

    Returns:
        Command execution result including stdout, stderr, and exit code.
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY environment variable is required"

    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return "Error: e2b-code-interpreter package not installed"

    sandbox = None
    try:
        sandbox = Sandbox.connect(sandbox_id)
    except Exception as e:
        return f"Error: Failed to connect to sandbox {sandbox_id}. Details: {e}"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            sandbox.set_timeout(DEFAULT_TIMEOUT)
            result = sandbox.commands.run(command)

            output = []
            if result.stdout:
                output.append(f"**stdout:**\n```\n{result.stdout}\n```")
            if result.stderr:
                output.append(f"**stderr:**\n```\n{result.stderr}\n```")
            output.append(f"**exit_code:** {result.exit_code}")

            return "\n\n".join(output)

        except Exception as e:
            if attempt == max_retries:
                return f"Error: Failed to run command after {max_retries} attempts: {e}"
            await asyncio.sleep(attempt * 2)
        finally:
            try:
                sandbox.set_timeout(DEFAULT_TIMEOUT)
            except Exception:
                pass


async def close_sandbox(sandbox_id: str) -> str:
    """
    Close and destroy a sandbox to release remote resources.

    Always call this tool when you are done using the sandbox.

    Args:
        sandbox_id: The ID of the sandbox to close (from create_sandbox).

    Returns:
        Confirmation message.
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY environment variable is required"

    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return "Error: e2b-code-interpreter package not installed"

    try:
        sandbox = Sandbox.connect(sandbox_id)
        sandbox.kill()
        return f"Sandbox {sandbox_id} closed and destroyed successfully."
    except Exception as e:
        return f"Error: Failed to close sandbox {sandbox_id}: {e}"


# Export the tool functions
SANDBOX_TOOLS = [
    create_sandbox,
    run_python_code,
    download_file_to_sandbox,
    run_shell_command,
    close_sandbox,
]
