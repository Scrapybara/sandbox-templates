#!/usr/bin/env python3
"""
FastAPI server providing a Bash tool endpoint
"""

import asyncio
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Tuple, Union
import re

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path


# Base classes and data structures
@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""
    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    system: str | None = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolError(Exception):
    """Raised when a tool encounters an error."""
    def __init__(self, message):
        self.message = message


class BaseAnthropicTool(metaclass=ABCMeta):
    """Abstract base class for Anthropic-defined tools."""

    @abstractmethod
    async def __call__(self, **kwargs) -> Any:
        """Executes the tool with the given arguments."""
        ...


# Bash Session implementation
class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process
    _current_directory: str
    _is_running_command: bool
    _last_command: str
    _partial_output: str
    _partial_error: str
    _session_id: int

    command: str = "/bin/bash"
    _output_delay: float = 0.2
    _timeout: float = 10.0
    _sentinel: str = "<<exit>>"

    def __init__(self, session_id: int):
        self._started = False
        self._current_directory = ""
        self._is_running_command = False
        self._last_command = ""
        self._partial_output = ""
        self._partial_error = ""
        self._session_id = session_id
        self._process = None

    @property
    def session_id(self) -> int:
        return self._session_id
        
    @property
    def is_running_command(self) -> bool:
        return self._is_running_command
        
    @property
    def last_command(self) -> str:
        return self._last_command
        
    @property
    def current_directory(self) -> str:
        return self._current_directory
        
    async def check_command_completion(self) -> bool:
        """Check if a running command has completed."""
        if not self._is_running_command:
            return True
            
        if not self._process or self._process.stdout is None:
            self._is_running_command = False
            return True
            
        try:
            if self._process.stdout._buffer:
                output = self._process.stdout._buffer.decode()
                self._partial_output = output
                
                if self._sentinel in output:
                    self._is_running_command = False
                    return True
        except Exception:
            pass
            
        return False

    def _process_output(self, output: str, prev_dir: str) -> Tuple[str, List[str]]:
        """Process command output to handle directory changes and file events."""
        system_messages: List[str] = []
        raw_created: Set[str] = set()
        raw_deleted: Set[str] = set()

        # Extract and parse the multi-line FILE_EVENTS block
        events_pattern = r'FILE_EVENTS:\s*created=\[(.*?)\]\s*deleted=\[(.*?)\]'
        match = re.search(events_pattern, output, flags=re.S)
        if match:
            created_blob = match.group(1).strip()
            deleted_blob = match.group(2).strip()
            
            raw_created = set(f for f in created_blob.split() if f)
            raw_deleted = set(f for f in deleted_blob.split() if f)

            if raw_created:
                joined = ' '.join(sorted(raw_created))
                system_messages.append(f'Created files: {joined}')

            if raw_deleted:
                joined = ' '.join(sorted(raw_deleted))
                system_messages.append(f'Removed files: {joined}')

            output = output[:match.start()] + output[match.end():]

        # Handle working directory changes
        lines = output.split('\n')
        cwd_change_line = next((l for l in lines if l.startswith('CWD_CHANGE:')), None)
        if cwd_change_line:
            new_dir = cwd_change_line.split(':', 1)[1].strip()
            self._current_directory = new_dir
            system_messages.append(f'Current directory: {new_dir}')

        output = '\n'.join(
            l for l in lines
            if not l.startswith(('FILE_EVENTS:', 'CWD_CHANGE:'))
        ).rstrip('\n')

        return output, system_messages
        
    def _filter_error_output(self, error: str) -> str:
        """Filter out common error messages that don't affect command execution."""
        if not error:
            return error
            
        if error.endswith("\n"):
            error = error[:-1]
            
        filtered_error = []
        for line in error.split('\n'):
            if not any(x in line.lower() for x in [
                'failed to connect to the bus',
                'failed to call method',
                'viz_main_impl',
                'object_proxy',
                'dbus',
                'setting up watches',
                'watches established'
            ]):
                filtered_error.append(line)
                
        return '\n'.join(filtered_error)

    async def start(self):
        """Start the bash session."""
        if self._started:
            return

        try:
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                preexec_fn=os.setsid,
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )

            self._started = True
            self._current_directory = os.getcwd()
                
        except Exception as e:
            raise ToolError(f"Failed to start bash session: {str(e)}")

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            return
            
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
            except Exception:
                pass
        
        self._is_running_command = False

    async def get_current_output(self) -> CLIResult:
        """Get the current output of a running command."""
        if not self._started:
            return CLIResult(
                output="",
                error="Session not started",
                system=f"Session ID: {self._session_id} not started"
            )
            
        await self.check_command_completion()
            
        if not self._is_running_command:
            return CLIResult(
                output="No command currently running in this session.",
                error="",
                system=f"Session ID: {self._session_id}"
            )
        
        if not self._process or self._process.stdout is None or self._process.stderr is None:
            self._is_running_command = False
            return CLIResult(
                output="",
                error="Process terminated unexpectedly",
                system=f"Session ID: {self._session_id} process terminated"
            )
        
        output = self._partial_output
        if self._process.stdout._buffer:
            output += self._process.stdout._buffer.decode()
            
        error = self._partial_error
        if self._process.stderr._buffer:
            error += self._process.stderr._buffer.decode()
            
        filtered_error = self._filter_error_output(error)
            
        if self._sentinel in output:
            self._is_running_command = False
            output = output.replace(self._sentinel, "")
            
            prev_dir = self._current_directory
            processed_output, system_messages = self._process_output(output, prev_dir)
            
            system_msg = f"Command completed. {', '.join(system_messages) if system_messages else ''} Session ID: {self._session_id}".strip()
            return CLIResult(
                output=processed_output,
                error=filtered_error,
                system=system_msg
            )
            
        return CLIResult(
            output=output,
            error=filtered_error,
            system=f"Command still running. Session ID: {self._session_id}"
        )

    async def run(self, command: str, timeout: float | None = None):
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
            
        await self.check_command_completion()
            
        if not self._process or self._process.returncode is not None:
            return ToolResult(
                system=f"Session {self._session_id} must be restarted",
                error=f"Bash has exited with returncode {self._process.returncode if self._process else 'None'}",
            )
            
        if self._is_running_command:
            return ToolResult(
                system=f"A command is already running in this session (ID: {self._session_id}). Please use another session or check the status of the current command."
            )

        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        self._partial_output = ""
        self._partial_error = ""
        self._last_command = command
        self._is_running_command = True
        
        prev_dir = self._current_directory

        wrapped_command = f"""
{command}

# Detect file changes if in a git repo
if [ -d .git ]; then
    created=$(git ls-files --others --exclude-standard 2>/dev/null || echo "")
    deleted=$(git ls-files --deleted 2>/dev/null || echo "")
else
    created=""
    deleted=""
fi

echo "FILE_EVENTS: created=[$created] deleted=[$deleted]"
echo "CWD_CHANGE: $(pwd)"
echo '{self._sentinel}'
"""
        self._process.stdout._buffer.clear()
        self._process.stderr._buffer.clear()
        
        try:
            self._process.stdin.write(wrapped_command.encode())
            await self._process.stdin.drain()
        except Exception as e:
            self._is_running_command = False
            return ToolResult(
                error=f"Failed to send command to bash: {str(e)}",
                system="Session may need to be restarted"
            )

        try:
            command_timeout = timeout if timeout is not None else self._timeout
            data = await asyncio.wait_for(
                self._process.stdout.readuntil(self._sentinel.encode()),
                timeout=command_timeout
            )
            output = data.decode().replace(self._sentinel, "")
            self._partial_output = output
            error = self._process.stderr._buffer.decode()
            self._partial_error = error
            self._is_running_command = False
        except asyncio.TimeoutError:
            output = self._partial_output
            error = self._partial_error
            filtered_error = self._filter_error_output(error)
            
            command_timeout = timeout if timeout is not None else self._timeout
            return ToolResult(
                output=output,
                error=filtered_error,
                system=f"Process timed out after {command_timeout} seconds. This process will continue to run in session {self._session_id}."
            )
        except asyncio.LimitOverrunError:
            output_chunks = []
            try:
                while True:
                    chunk = await asyncio.wait_for(
                        self._process.stdout.read(8192),
                        timeout=0.1
                    )
                    if not chunk:
                        break
                    output_chunks.append(chunk.decode())
                    accumulated = ''.join(output_chunks)
                    if self._sentinel in accumulated:
                        output = accumulated.replace(self._sentinel, "")
                        self._partial_output = output
                        error = self._process.stderr._buffer.decode()
                        self._partial_error = error
                        self._is_running_command = False
                        break
            except asyncio.TimeoutError:
                output = ''.join(output_chunks)
                self._partial_output = output
                error = self._process.stderr._buffer.decode()
                self._partial_error = error
        except Exception as e:
            self._is_running_command = False
            return ToolResult(
                error=f"Error executing command: {str(e)}",
                system="An unexpected error occurred"
            )

        processed_output, system_messages = self._process_output(output, prev_dir)
        filtered_error = self._filter_error_output(error)

        self._process.stdout._buffer.clear()
        self._process.stderr._buffer.clear()

        system = ", ".join(system_messages) if system_messages else None
        return CLIResult(output=processed_output, error=filtered_error, system=system)


# Bash Tool implementation
class BashTool(BaseAnthropicTool):
    """A tool that allows the agent to run bash commands."""

    _sessions: Dict[int, _BashSession]
    name: ClassVar[Literal["bash"]] = "bash"

    def __init__(self):
        self._sessions = {}
        super().__init__()

    async def __call__(
        self, command: str | None = None, 
        session: int | None = None,
        restart: bool = False,
        list_sessions: bool = False, 
        check_session: int | None = None,
        timeout: float | None = None,
        **kwargs
    ):
        if list_sessions:
            if not self._sessions:
                return ToolResult(system="No active sessions.")
                
            sessions_info = []
            for session_id, session_obj in self._sessions.items():
                await session_obj.check_command_completion()
                status = "running command" if session_obj.is_running_command else "idle"
                last_cmd = session_obj.last_command if session_obj.last_command else "None"
                sessions_info.append(f"Session {session_id}: {status}, Last command: '{last_cmd}', Directory: {session_obj.current_directory}")
                
            return ToolResult(output="\n".join(sessions_info))
        
        if check_session is not None:
            if check_session not in self._sessions:
                return ToolResult(error=f"Session {check_session} not found.")
                
            session_obj = self._sessions[check_session]
            
            await session_obj.check_command_completion()
                
            if not session_obj.is_running_command:
                return ToolResult(
                    system=f"No command running in session {check_session}. Last command: '{session_obj.last_command}'",
                )
                
            return await session_obj.get_current_output()

        if restart:
            session_id = session if session is not None else 1
            try:
                if session_id in self._sessions:
                    try:
                        self._sessions[session_id].stop()
                    except Exception:
                        pass
                self._sessions[session_id] = _BashSession(session_id=session_id)
                await self._sessions[session_id].start()
                return ToolResult(system=f"Session {session_id} has been restarted.")
            except Exception as e:
                return ToolResult(error=f"Failed to restart session {session_id}: {str(e)}")
            
        if session is None and command is not None:
            session_id = 1
            while True:
                if session_id not in self._sessions:
                    session = session_id
                    break
                
                await self._sessions[session_id].check_command_completion()
                if not self._sessions[session_id].is_running_command:
                    session = session_id
                    break
                
                session_id += 1
        
        session = session if session is not None else 1
            
        created_msg = None
        try:
            if session not in self._sessions:                
                self._sessions[session] = _BashSession(session_id=session)
                await self._sessions[session].start()
                created_msg = f"Created new session with ID: {session}"
        except Exception as e:
            return ToolResult(error=f"Failed to create session {session}: {str(e)}")
            
        current_session = self._sessions[session]
        
        await current_session.check_command_completion()

        if command is not None:
            if current_session.is_running_command:
                busy_message = f"Session {session} is busy running '{current_session.last_command}'. Please use another session number."
                return ToolResult(system=busy_message)
                
            try:
                result = await current_session.run(command, timeout)
                
                if isinstance(result, ToolResult) and (
                    (result.system and "must be restarted" in result.system) or
                    (result.error and "0 bytes read on a total of undefined expected bytes" in result.error)
                ):
                    try:
                        current_session.stop()
                        self._sessions[session] = _BashSession(session_id=session)
                        await self._sessions[session].start()
                        current_session = self._sessions[session]
                        
                        result = await current_session.run(command, timeout)
                        
                        if isinstance(result, CLIResult):
                            new_system_msg = f"Session {session} was automatically restarted and the command was re-run."
                            if result.system:
                                new_system_msg = f"{new_system_msg} {result.system}"
                            return CLIResult(
                                output=result.output,
                                error=result.error,
                                system=new_system_msg,
                                base64_image=getattr(result, 'base64_image', None)
                            )
                        elif isinstance(result, ToolResult):
                            new_system_msg = f"Session {session} was automatically restarted and the command was re-run."
                            if result.system:
                                new_system_msg = f"{new_system_msg} {result.system}"
                            return ToolResult(
                                output=result.output,
                                error=result.error,
                                system=new_system_msg,
                                base64_image=getattr(result, 'base64_image', None)
                            )
                    except Exception as e:
                        return ToolResult(error=f"Failed to automatically restart session {session}: {str(e)}")
                
                if created_msg and isinstance(result, CLIResult):
                    new_system_msg = created_msg
                    if result.system:
                        new_system_msg = f"{created_msg}. {result.system}"
                    return CLIResult(
                        output=result.output,
                        error=result.error,
                        system=new_system_msg,
                        base64_image=getattr(result, 'base64_image', None)
                    )
                return result
            except Exception as e:
                return ToolResult(error=f"Error executing command: {str(e)}")

        if created_msg:
            return ToolResult(system=created_msg)
            
        raise ToolError("no command provided.")


# FastAPI app and endpoints
app = FastAPI(
    title="Bash Tool API",
    description="REST API for bash command execution",
    version="1.0.0"
)

# Initialize bash tool
bash_tool = BashTool()

# Define the workspace directory
WORKSPACE_DIR = Path("/project/workspace")
if not WORKSPACE_DIR.exists():
    # Fallback for local development
    WORKSPACE_DIR = Path.cwd()

# Mount static file server
app.mount("/static", StaticFiles(directory=str(WORKSPACE_DIR), html=True), name="static")


# Request/Response models
class BashRequest(BaseModel):
    command: Optional[str] = None
    session: Optional[int] = None
    restart: Optional[bool] = False
    list_sessions: Optional[bool] = False
    check_session: Optional[int] = None
    timeout: Optional[float] = None


class ToolResponse(BaseModel):
    output: Optional[str] = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None


# Helper function
def _tool_result_to_response(result: ToolResult) -> Dict[str, Any]:
    return {
        "output": result.output,
        "error": result.error,
        "base64_image": result.base64_image,
        "system": result.system
    }


# API Endpoints
@app.post("/bash", response_model=ToolResponse)
async def bash_action(request: BashRequest):
    try:
        result = await bash_tool(**request.model_dump(exclude_none=True))
        return _tool_result_to_response(result)
    except ToolError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/status")
async def get_status():
    return {"status": "ok", "service": "bash-tool-api"}


@app.get("/list-files")
async def list_files():
    """List all files in the workspace directory with absolute paths"""
    try:
        files = []
        for file_path in WORKSPACE_DIR.rglob("*"):
            if file_path.is_file():
                # Skip hidden files and directories
                if not any(part.startswith('.') for part in file_path.parts):
                    files.append(str(file_path))
        
        files.sort()
        return {
            "total_files": len(files),
            "workspace": str(WORKSPACE_DIR),
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.get("/file/{file_path:path}")
async def get_file(file_path: str):
    """Get a specific file from the workspace"""
    try:
        full_path = WORKSPACE_DIR / file_path
        
        # Security check: ensure the path is within workspace
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(WORKSPACE_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied: Path outside workspace")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        return FileResponse(path=str(full_path), filename=full_path.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "Bash Tool API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/bash", "method": "POST", "description": "Execute bash commands"},
            {"path": "/status", "method": "GET", "description": "Check service status"},
            {"path": "/list-files", "method": "GET", "description": "List all files in workspace"},
            {"path": "/file/{file_path}", "method": "GET", "description": "Get a specific file"},
            {"path": "/static", "description": "Static file server (browse to /static)"},
            {"path": "/docs", "method": "GET", "description": "API documentation"}
        ]
    }


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 