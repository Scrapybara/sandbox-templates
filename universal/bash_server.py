#!/usr/bin/env python3
"""
FastAPI server providing Bash and File tool endpoints
"""

import asyncio
import os
import re
import base64
import shutil
import inspect
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, get_args

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path


# Command types for file operations
Command = Literal[
    "read", "write", "append", "delete", "exists", "list", "mkdir", "rmdir", "move", "copy",
    "view", "create", "replace", "insert", "delete_lines", "undo", "grep"
]


def _shorten(text: str, limit: int = 120) -> str:
    """Return *text* truncated to *limit* chars, escaping newlines for readability."""
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


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
        """Process command output to handle directory changes."""
        system_messages: List[str] = []

        # Handle working directory changes
        lines = output.split('\n')
        cwd_change_line = next((l for l in lines if l.startswith('CWD_CHANGE:')), None)
        if cwd_change_line:
            new_dir = cwd_change_line.split(':', 1)[1].strip()
            self._current_directory = new_dir
            system_messages.append(f'Current directory: {new_dir}')

        # Remove any CWD_CHANGE marker lines from the output
        output = '\n'.join(
            l for l in lines
            if not l.startswith('CWD_CHANGE:')
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
        self._sessions_lock = asyncio.Lock()
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
                async with self._sessions_lock:
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
            async with self._sessions_lock:
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
                        async with self._sessions_lock:
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


# File Tool implementation
class FileTool(BaseAnthropicTool):
    """
    A filesystem editor tool that allows the agent to view, create, and edit files.
    """

    name: ClassVar[Literal["file"]] = "file"
    _file_history: Dict[Path, List[str]]  # Undo history for text edits

    def __init__(self, base_path: Path | None = None):
        self._file_history = defaultdict(list)
        self.base_path = base_path or Path.cwd()
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
        super().__init__()
    
    def _validate_path(self, path: str) -> Path:
        try:
            path_obj = Path(path)
            full_path = path_obj if path_obj.is_absolute() else (self.base_path / path_obj).resolve()
            # Allow access to the entire workspace area
            if not str(full_path).startswith(str(self.base_path)):
                raise ToolError("Path is outside the allowed base directory")
            return full_path
        except Exception as e:
            raise ToolError(f"Invalid path: {str(e)}")
        
    async def __call__(self, command: Command, **kwargs) -> ToolResult:
        try:
            if command not in get_args(Command):
                raise ToolError(f"Unsupported command: {command}. Supported commands: {', '.join(get_args(Command))}")
            
            method_map = {
                "read": self.read, "write": self.write, "append": self.append, "delete": self.delete,
                "exists": self.exists, "list": self.list_dir, "mkdir": self.mkdir, "rmdir": self.rmdir,
                "move": self.move, "copy": self.copy, "view": self.view, "create": self.create,
                "replace": self.replace, "insert": self.insert, "delete_lines": self.delete_lines,
                "undo": self.undo, "grep": self.grep
            }
            
            if command not in method_map:
                raise ToolError(f"Command '{command}' is valid but not implemented")
                
            method = method_map[command]
            
            # Filter kwargs to only include parameters that the method accepts
            sig = inspect.signature(method)
            valid_params = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            
            return await method(**filtered_kwargs)
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"Unexpected error: {str(e)}")

    async def read(self, path: str, mode: str = "text", encoding: str = "utf-8", line_numbers: bool = True) -> ToolResult:
        """Read the content of a file in text or binary mode."""
        full_path = self._validate_path(path)
        if not full_path.is_file():
            raise ToolError("Path is not a file")
        try:
            if mode == "text":
                content = full_path.read_text(encoding=encoding)
                if line_numbers:
                    numbered_content = "\n".join(
                        f"{str(i + 1).rjust(6)}\t{line}" for i, line in enumerate(content.splitlines())
                    )
                    return ToolResult(output=numbered_content)
                return ToolResult(output=content)
            elif mode == "binary":
                content = base64.b64encode(full_path.read_bytes()).decode()
                return ToolResult(output=content, system="binary")
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
        except Exception as e:
            raise ToolError(f"Failed to read file: {str(e)}")

    async def write(self, path: str, content: str, mode: str = "text", encoding: str = "utf-8") -> ToolResult:
        """Write content to a file, overwriting if it exists."""
        full_path = self._validate_path(path)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "text":
                full_path.write_text(content, encoding=encoding)
            elif mode == "binary":
                full_path.write_bytes(base64.b64decode(content))
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
            return ToolResult(output=f"File written to {path}")
        except Exception as e:
            raise ToolError(f"Failed to write file: {str(e)}")

    async def append(self, path: str, content: str, mode: str = "text", encoding: str = "utf-8") -> ToolResult:
        """Append content to an existing file or create it if it doesn't exist."""
        full_path = self._validate_path(path)
        try:
            if mode == "text":
                with full_path.open('a', encoding=encoding) as f:
                    f.write(content)
            elif mode == "binary":
                decoded_content = base64.b64decode(content)
                with full_path.open('ab') as f:
                    f.write(decoded_content)
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
            return ToolResult(output=f"Appended to file {path}")
        except Exception as e:
            raise ToolError(f"Failed to append to file: {str(e)}")

    async def delete(self, path: str, recursive: bool = False) -> ToolResult:
        """Delete a file or directory, optionally recursively."""
        full_path = self._validate_path(path)
        try:
            if full_path.is_file():
                full_path.unlink()
            elif full_path.is_dir():
                if recursive:
                    shutil.rmtree(full_path)
                else:
                    full_path.rmdir()
            else:
                raise ToolError("Path does not exist")
            self._file_history.pop(full_path, None)  # Clear undo history
            return ToolResult(output=f"Deleted {path}")
        except Exception as e:
            raise ToolError(f"Failed to delete: {str(e)}")

    async def exists(self, path: str) -> ToolResult:
        """Check if a path exists."""
        try:
            full_path = self._validate_path(path)
            exists = full_path.exists()
            return ToolResult(output=str(exists))
        except Exception as e:
            return ToolResult(error=f"Failed to check existence: {str(e)}")

    async def list_dir(self, path: str) -> ToolResult:
        """List the contents of a directory."""
        full_path = self._validate_path(path)
        if not full_path.is_dir():
            raise ToolError("Path is not a directory")
        try:
            contents = []
            for p in sorted(full_path.iterdir()):
                if p.is_dir():
                    contents.append(f"{p.name}/")
                else:
                    contents.append(p.name)
            return ToolResult(output="\n".join(contents))
        except Exception as e:
            raise ToolError(f"Failed to list directory: {str(e)}")

    async def mkdir(self, path: str) -> ToolResult:
        """Create a directory, including parent directories if needed."""
        full_path = self._validate_path(path)
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            return ToolResult(output=f"Directory created: {path}")
        except Exception as e:
            raise ToolError(f"Failed to create directory: {str(e)}")

    async def rmdir(self, path: str) -> ToolResult:
        """Remove an empty directory."""
        full_path = self._validate_path(path)
        try:
            full_path.rmdir()
            return ToolResult(output=f"Directory removed: {path}")
        except Exception as e:
            raise ToolError(f"Failed to remove directory: {str(e)}")

    async def move(self, src: str, dst: str) -> ToolResult:
        """Move or rename a file or directory."""
        src_path = self._validate_path(src)
        dst_path = self._validate_path(dst)
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            src_path.rename(dst_path)
            if src_path in self._file_history:
                self._file_history[dst_path] = self._file_history.pop(src_path)
            return ToolResult(output=f"Moved {src} to {dst}")
        except Exception as e:
            raise ToolError(f"Failed to move: {str(e)}")

    async def copy(self, src: str, dst: str) -> ToolResult:
        """Copy a file or directory."""
        src_path = self._validate_path(src)
        dst_path = self._validate_path(dst)
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            elif src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                raise ToolError("Source path does not exist")
            return ToolResult(output=f"Copied {src} to {dst}")
        except Exception as e:
            raise ToolError(f"Failed to copy: {str(e)}")

    async def view(
        self,
        path: str,
        view_range: Optional[List[int]] = None,
        line_numbers: bool = True,
    ) -> ToolResult:
        full_path = self._validate_path(path)
        if full_path.is_dir():
            if view_range:
                raise ToolError("view_range not applicable for directories")
            # List directory contents using pathlib
            contents = []
            try:
                for item in sorted(full_path.iterdir()):
                    if item.is_dir():
                        contents.append(f"  {item.name}/")
                    else:
                        contents.append(f"  {item.name}")
                output = f"Directory contents of {path}:\n" + "\n".join(contents)
                return ToolResult(output=output)
            except Exception as e:
                raise ToolError(f"Failed to list directory: {str(e)}")
        
        try:
            content = full_path.read_text()
            if view_range:
                lines = content.splitlines()
                start, end = view_range
                if start < 1 or start > len(lines) or end < start or end > len(lines):
                    raise ToolError(f"Invalid view_range: {view_range}")
                content = "\n".join(lines[start - 1 : end])

            if line_numbers:
                start_num = view_range[0] if view_range else 1
                numbered_content = "\n".join(
                    f"{str(start_num + i).rjust(6)}\t{line}" for i, line in enumerate(content.splitlines())
                )
                return ToolResult(output=numbered_content)

            return ToolResult(output=content)
        except Exception as e:
            raise ToolError(f"Failed to view file: {str(e)}")

    async def create(self, path: str, content: str, mode: str = "text", encoding: str = "utf-8") -> ToolResult:
        """Create a new file with the given content, failing if it already exists."""
        full_path = self._validate_path(path)
        if full_path.exists():
            raise ToolError("File already exists")
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "text":
                full_path.write_text(content, encoding=encoding)
            elif mode == "binary":
                full_path.write_bytes(base64.b64decode(content))
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
            return ToolResult(output=f"File created: {path}")
        except Exception as e:
            raise ToolError(f"Failed to create file: {str(e)}")

    async def replace(self, path: str, old_str: str, new_str: str, all_occurrences: bool = False) -> ToolResult:
        """Replace a string in a file, optionally all occurrences."""
        full_path = self._validate_path(path)
        if not full_path.is_file():
            raise ToolError("Path is not a file")
        try:
            content = full_path.read_text()

            # CASE 1 – literal text matches
            if old_str in content:
                if all_occurrences:
                    new_content = content.replace(old_str, new_str)
                else:
                    if content.count(old_str) > 1:
                        raise ToolError("Multiple occurrences found; set all_occurrences=True to replace all")
                    new_content = content.replace(old_str, new_str, 1)
            # CASE 2 – literal differs only by line-ending or trailing-space -> replace on normalised text then restore original line endings
            else:
                # Compare using normalised versions so CRLF/LF or trailing spaces do not cause false negatives.
                cmp_content = content.replace("\r\n", "\n")
                cmp_old = old_str.replace("\r\n", "\n")
                if cmp_old not in cmp_content:
                    raise ToolError(f"'{old_str}' not found")

                norm_new = new_str.replace("\r\n", "\n")
                if all_occurrences:
                    norm_new_content = cmp_content.replace(cmp_old, norm_new)
                else:
                    if cmp_content.count(cmp_old) > 1:
                        raise ToolError("Multiple occurrences found; set all_occurrences=True to replace all")
                    norm_new_content = cmp_content.replace(cmp_old, norm_new, 1)

                # Convert the normalised content back to the original EOL style
                if "\r\n" in content:
                    new_content = norm_new_content.replace("\n", "\r\n")
                else:
                    new_content = norm_new_content

            self._file_history[full_path].append(content)
            if len(self._file_history[full_path]) > 5:
                self._file_history[full_path].pop(0)
            full_path.write_text(new_content)
            return ToolResult(output=f"Replaced \"{_shorten(old_str)}\" with \"{_shorten(new_str)}\"")
        except Exception as e:
            raise ToolError(f"Failed to replace string: {str(e)}")

    async def insert(self, path: str, line: int, text: str) -> ToolResult:
        """Insert text at a specific line in a file."""
        full_path = self._validate_path(path)
        if not full_path.is_file():
            raise ToolError("Path is not a file")
        try:
            content = full_path.read_text()
            lines = content.splitlines()
            if line < 1 or line > len(lines) + 1:
                raise ToolError(f"Line number {line} is out of range")
            lines.insert(line - 1, text)
            new_content = "\n".join(lines)
            self._file_history[full_path].append(content)
            if len(self._file_history[full_path]) > 5:
                self._file_history[full_path].pop(0)
            full_path.write_text(new_content)
            return ToolResult(output=f"Inserted \"{_shorten(text)}\" at line {line}")
        except Exception as e:
            raise ToolError(f"Failed to insert text: {str(e)}")

    async def delete_lines(self, path: str, lines: List[int]) -> ToolResult:
        """Delete specified lines from a file."""
        full_path = self._validate_path(path)
        if not full_path.is_file():
            raise ToolError("Path is not a file")
        try:
            content = full_path.read_text()
            file_lines = content.splitlines()
            lines_to_delete = set(lines)
            new_lines = [line for i, line in enumerate(file_lines, 1) if i not in lines_to_delete]
            new_content = "\n".join(new_lines)
            self._file_history[full_path].append(content)
            if len(self._file_history[full_path]) > 5:
                self._file_history[full_path].pop(0)
            full_path.write_text(new_content)
            return ToolResult(output=f"Deleted lines {lines}")
        except Exception as e:
            raise ToolError(f"Failed to delete lines: {str(e)}")

    async def undo(self, path: str) -> ToolResult:
        """Undo the last text editing operation on a file."""
        full_path = self._validate_path(path)
        if not full_path.is_file():
            raise ToolError("File does not exist")
        if not self._file_history.get(full_path):
            raise ToolError("No undo history available")
        try:
            previous_content = self._file_history[full_path].pop()
            full_path.write_text(previous_content)
            return ToolResult(output=f"Undid last edit on {path}")
        except Exception as e:
            raise ToolError(f"Failed to undo edit: {str(e)}")

    async def grep(
        self,
        pattern: str,
        path: str,
        case_sensitive: bool = True,
        recursive: bool = False,
        line_numbers: bool = True
    ) -> ToolResult:
        """Search for a pattern in a file or directory."""
        full_path = self._validate_path(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        results = []

        def search_file(file_path):
            try:
                # Skip if not a regular file (sockets, pipes, etc.)
                if not file_path.is_file() or (file_path.is_symlink() and not file_path.resolve().is_file()):
                    return
                
                with file_path.open('r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line, flags):
                            results.append({
                                'file': str(file_path.relative_to(self.base_path)),
                                'line_number': i if line_numbers else None,
                                'content': line.strip()
                            })
            except (UnicodeDecodeError, IOError, OSError):
                # Skip binary files and files that can't be read
                pass

        if full_path.is_file():
            search_file(full_path)
        elif full_path.is_dir():
            if not recursive:
                raise ToolError("Recursive search must be enabled for directories")
            for root, _, files in os.walk(full_path):
                for file in files:
                    search_file(Path(root) / file)
        else:
            raise ToolError("Path does not exist")

        if not results:
            return ToolResult(output="No matches found")

        output = "\n".join([
            f"{r['file']}:{r['line_number']}:{r['content']}" if r['line_number'] else f"{r['file']}:{r['content']}"
            for r in results
        ])
        return ToolResult(output=output)


# FastAPI app and endpoints
app = FastAPI(
    title="Bash and File Tool API",
    description="REST API for bash command execution and file operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the workspace directory
WORKSPACE_DIR = Path("/project/workspace")
if not WORKSPACE_DIR.exists():
    # Fallback for local development
    WORKSPACE_DIR = Path.cwd()

# Initialize tools
bash_tool = BashTool()
file_tool = FileTool(base_path=WORKSPACE_DIR)

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


class FileRequest(BaseModel):
    command: str
    path: Optional[str] = None
    content: Optional[str] = None
    mode: Optional[str] = "text"
    encoding: Optional[str] = "utf-8"
    line_numbers: Optional[bool] = True
    recursive: Optional[bool] = False
    src: Optional[str] = None
    dst: Optional[str] = None
    view_range: Optional[List[int]] = None
    old_str: Optional[str] = None
    new_str: Optional[str] = None
    all_occurrences: Optional[bool] = False
    line: Optional[int] = None
    text: Optional[str] = None
    lines: Optional[List[int]] = None
    pattern: Optional[str] = None
    case_sensitive: Optional[bool] = True


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


@app.post("/file", response_model=ToolResponse)
async def file_action(request: FileRequest):
    """Execute file operations"""
    try:
        # Convert request to kwargs, excluding None values
        kwargs = request.model_dump(exclude_none=True)
        result = await file_tool(**kwargs)
        return _tool_result_to_response(result)
    except ToolError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/status")
async def get_status():
    return {"status": "ok", "service": "bash-and-file-tool-api"}


@app.get("/list-files")
async def list_files(path: str = None):
    """List first-level files and directories at the given path"""
    try:
        # Use provided path or default to workspace
        if path:
            target_path = Path(path)
            # Security check: ensure the path exists and is a directory
            if not target_path.exists():
                raise HTTPException(status_code=404, detail="Path not found")
            if not target_path.is_dir():
                raise HTTPException(status_code=400, detail="Path is not a directory")
        else:
            target_path = WORKSPACE_DIR
        
        items = []
        # List only immediate children (first level)
        for item_path in target_path.iterdir():
            # Skip hidden files and directories
            if not item_path.name.startswith('.'):
                item_info = {
                    "name": item_path.name,
                    "path": str(item_path),
                    "type": "directory" if item_path.is_dir() else "file"
                }
                items.append(item_info)
        
        # Sort items by name
        items.sort(key=lambda x: x["name"])
        
        return {
            "path": str(target_path),
            "total_items": len(items),
            "items": items
        }
    except HTTPException:
        raise
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
        "service": "Bash and File Tool API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/bash", "method": "POST", "description": "Execute bash commands"},
            {"path": "/file", "method": "POST", "description": "File operations (read, write, create, delete, etc.)"},
            {"path": "/status", "method": "GET", "description": "Check service status"},
            {"path": "/list-files", "method": "GET", "description": "List first-level files/dirs in path (query param: ?path=/absolute/path)"},
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