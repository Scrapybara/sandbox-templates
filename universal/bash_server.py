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
import aiofiles
import aiofiles.os
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Dict, List, Literal, Optional, get_args

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

# Files and directories to exclude from serving/listing
EXCLUDED_PATTERNS = [
    "bash_server.py",
    "lsp.py",
    ".codesandbox",
    ".devcontainer",
    "__pycache__",
    "README",
    "README.md",
    "README.txt",
    "README.rst"
]


def _is_excluded_path(path: Path) -> bool:
    """Check if a path should be excluded from serving/listing."""
    # Get the workspace directory for comparison
    workspace_dir = WORKSPACE_DIR
    
    try:
        # Resolve the path to handle any symlinks or relative paths
        resolved_path = path.resolve()
        workspace_resolved = workspace_dir.resolve()
        
        # Check if this is a root-level file/directory
        # Get the relative path from workspace root
        try:
            relative_path = resolved_path.relative_to(workspace_resolved)
        except ValueError:
            # Path is not within workspace
            return False
            
        # Get the first part of the path (root-level item)
        parts = relative_path.parts
        if not parts:
            return False
            
        root_item = parts[0]
        
        # Check if the root item matches any excluded pattern
        for pattern in EXCLUDED_PATTERNS:
            if root_item == pattern:
                return True
                
        # Also check for hidden files/directories at root level only
        if root_item.startswith('.'):
            return True
            
    except Exception:
        # If there's any error in path resolution, don't exclude
        return False
        
    return False


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


def validate_command_length(command: str) -> None:
    """Validate command length to prevent issues with very long commands."""
    MAX_COMMAND_LENGTH = 100000  # 100KB limit
    if len(command.encode('utf-8')) > MAX_COMMAND_LENGTH:
        raise ToolError(f"Command too long ({len(command.encode('utf-8'))} bytes). Maximum allowed: {MAX_COMMAND_LENGTH} bytes")


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
        return str(WORKSPACE_DIR)
        
    async def check_command_completion(self) -> bool:
        """Check if a running command has completed."""
        if not self._is_running_command:
            return True
            
        if not self._process or self._process.stdout is None:
            self._is_running_command = False
            return True
            
        try:
            if self._process.stdout._buffer:
                output = self._process.stdout._buffer.decode(errors="replace")
                self._partial_output = output
                
                if self._sentinel in output:
                    self._is_running_command = False
                    return True
        except Exception:
            pass
            
        return False

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
                limit=3 * 1024 * 1024,
            )

            self._started = True
                
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
            output += self._process.stdout._buffer.decode(errors="replace")
            
        error = self._partial_error
        if self._process.stderr._buffer:
            error += self._process.stderr._buffer.decode(errors="replace")
            
        filtered_error = self._filter_error_output(error)
            
        if self._sentinel in output:
            self._is_running_command = False
            output = output.replace(self._sentinel, "")
            
            processed_output = output.rstrip('\n')
            system_msg = f"Command completed. Session ID: {self._session_id}"
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
        
        base_dir = str(WORKSPACE_DIR)

        wrapped_command = f"""
{command}

cd "{base_dir}"
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
            
            # Add timeout protection for the readuntil operation
            try:
                data = await asyncio.wait_for(
                    self._process.stdout.readuntil(self._sentinel.encode()),
                    timeout=command_timeout
                )
                output = data.decode(errors="replace").replace(self._sentinel, "")
                self._partial_output = output
                error = self._process.stderr._buffer.decode(errors="replace")
                self._partial_error = error
                self._is_running_command = False
                
            except (asyncio.IncompleteReadError, ConnectionResetError) as e:
                # Handle the "0 bytes read" error and similar stream issues
                # This is the main fix for the reported error
                output = self._partial_output
                error = self._partial_error
                
                # Try to get any remaining data from buffers
                if hasattr(self._process.stdout, '_buffer') and self._process.stdout._buffer:
                    try:
                        remaining = self._process.stdout._buffer.decode(errors="replace")
                        output += remaining
                    except Exception:
                        pass
                        
                if hasattr(self._process.stderr, '_buffer') and self._process.stderr._buffer:
                    try:
                        remaining = self._process.stderr._buffer.decode(errors="replace")
                        error += remaining
                    except Exception:
                        pass
                
                # Check if the command actually completed successfully despite the stream error
                if self._sentinel in output:
                    # Command completed, just had a stream reading issue
                    output = output.replace(self._sentinel, "")
                    self._is_running_command = False
                    filtered_error = self._filter_error_output(error)
                    return CLIResult(
                        output=output.rstrip('\n'), 
                        error=filtered_error,
                        system=f"Command completed despite stream reading issue. Session ID: {self._session_id}"
                    )
                else:
                    # Command didn't complete normally
                    self._is_running_command = False
                    filtered_error = self._filter_error_output(error)
                    return ToolResult(
                        output=output,
                        error=filtered_error,
                        system=f"Stream reading error: {str(e)}. Command may have failed or produced output exceeding buffer limits."
                    )
                
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
            # Enhanced limit overrun handling with timeout protection
            output_chunks = []
            total_timeout = command_timeout if 'command_timeout' in locals() else (timeout if timeout is not None else self._timeout)
            start_time = asyncio.get_event_loop().time()
            
            try:
                while True:
                    current_time = asyncio.get_event_loop().time()
                    remaining_time = total_timeout - (current_time - start_time)
                    
                    if remaining_time <= 0:
                        break
                        
                    chunk = await asyncio.wait_for(
                        self._process.stdout.read(8192),
                        timeout=min(remaining_time, 30.0)  # Max 30s per read
                    )
                    if not chunk:
                        break
                        
                    output_chunks.append(chunk.decode(errors="replace"))
                    accumulated = ''.join(output_chunks)
                    
                    if self._sentinel in accumulated:
                        output = accumulated.replace(self._sentinel, "")
                        self._partial_output = output
                        error = self._process.stderr._buffer.decode(errors="replace")
                        self._partial_error = error
                        self._is_running_command = False
                        break
                        
            except asyncio.TimeoutError:
                output = ''.join(output_chunks)
                self._partial_output = output
                error = self._process.stderr._buffer.decode(errors="replace")
                self._partial_error = error
        except Exception as e:
            # Catch any other unexpected errors
            self._is_running_command = False
            return ToolResult(
                error=f"Unexpected error executing command: {str(e)}",
                system="Session may need to be restarted"
            )

        processed_output = output.rstrip('\n')
        filtered_error = self._filter_error_output(error)

        self._process.stdout._buffer.clear()
        self._process.stderr._buffer.clear()

        return CLIResult(output=processed_output, error=filtered_error)

    async def stream_command(self, command: str):
        """Run command and yield stdout and stderr lines as they arrive until the sentinel appears."""
        if not self._started:
            await self.start()

        await self.check_command_completion()
        if self._is_running_command:
            raise ToolError("Session busy running another command")

        assert (
            self._process
            and self._process.stdin
            and self._process.stdout
            and self._process.stderr
        )

        self._partial_output = ""
        self._partial_error = ""
        self._last_command = command
        self._is_running_command = True

        wrapped_command = f"""
{command}

cd \"{WORKSPACE_DIR}\"
echo '{self._sentinel}'
"""

        self._process.stdout._buffer.clear()
        self._process.stderr._buffer.clear()

        self._process.stdin.write(wrapped_command.encode())
        await self._process.stdin.drain()

        sentinel = self._sentinel

        queue: asyncio.Queue[str] = asyncio.Queue()

        async def _reader(stream: asyncio.StreamReader, is_stdout: bool):
            """Continuously read chunks (not lines) from *stream* and enqueue them."""
            buffer = ""
            chunk_size = 256  # bytes

            while True:
                data = await stream.read(chunk_size)
                if not data:
                    break  # EOF

                text = data.decode(errors="replace")

                if is_stdout:
                    buffer += text

                    if sentinel in buffer:
                        before, _sent, _after = buffer.partition(sentinel)
                        if before:
                            await queue.put(before)
                        await queue.put(None)
                        break

                    if buffer:
                        await queue.put(buffer)
                        buffer = ""
                else:
                    filtered = self._filter_error_output(text)
                    if filtered:
                        await queue.put(filtered)

        stdout_task = asyncio.create_task(_reader(self._process.stdout, True))
        stderr_task = asyncio.create_task(_reader(self._process.stderr, False))

        try:
            while True:
                line = await queue.get()
                if line is None:  # Sentinel from stdout reader
                    break
                yield line
        finally:
            for t in (stdout_task, stderr_task):
                t.cancel()
            self._is_running_command = False


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
            
        created_msg = None
        try:
            async with self._sessions_lock:
                if session is None and command is not None:
                    session_id = 1
                    while True:
                        if session_id not in self._sessions:
                            session = session_id
                            break
                        
                        if session_id in self._sessions and not self._sessions[session_id].is_running_command:
                            session = session_id
                            break
                        
                        session_id += 1
                
                session = session if session is not None else 1
                
                # Create session if it doesn't exist
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
                # Validate command length to prevent issues with very long commands
                validate_command_length(command)
                result = await current_session.run(command, timeout)
                
                if isinstance(result, ToolResult) and (
                    (result.system and "must be restarted" in result.system) or
                    (result.error and "0 bytes read on a total of undefined expected bytes" in result.error) or
                    (result.error and "Stream reading error" in result.error) or
                    (result.system and "stream reading issue" in result.system)
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
        # Note: We'll check/create the base_path in the first async call
        super().__init__()
    
    async def _ensure_base_path_exists(self):
        """Ensure base path exists - call this in async methods that need it"""
        if not await aiofiles.os.path.exists(str(self.base_path)):
            await asyncio.to_thread(self.base_path.mkdir, parents=True, exist_ok=True)
    
    async def _validate_path(self, path: str) -> Path:
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
        full_path = await self._validate_path(path)
        if not await aiofiles.os.path.isfile(str(full_path)):
            raise ToolError("Path is not a file")
        try:
            if mode == "text":
                async with aiofiles.open(str(full_path), 'r', encoding=encoding) as f:
                    content = await f.read()
                if line_numbers:
                    numbered_content = "\n".join(
                        f"{str(i + 1).rjust(6)}\t{line}" for i, line in enumerate(content.splitlines())
                    )
                    return ToolResult(output=numbered_content)
                return ToolResult(output=content)
            elif mode == "binary":
                async with aiofiles.open(str(full_path), 'rb') as f:
                    binary_content = await f.read()
                content = base64.b64encode(binary_content).decode()
                return ToolResult(output=content, system="binary")
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
        except Exception as e:
            raise ToolError(f"Failed to read file: {str(e)}")

    async def write(self, path: str, content: str, mode: str = "text", encoding: str = "utf-8") -> ToolResult:
        """Write content to a file, overwriting if it exists."""
        full_path = await self._validate_path(path)
        try:
            await self._ensure_base_path_exists()
            await asyncio.to_thread(full_path.parent.mkdir, parents=True, exist_ok=True)
            if mode == "text":
                async with aiofiles.open(str(full_path), 'w', encoding=encoding) as f:
                    await f.write(content)
            elif mode == "binary":
                decoded_content = base64.b64decode(content)
                async with aiofiles.open(str(full_path), 'wb') as f:
                    await f.write(decoded_content)
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
            return ToolResult(output=f"File written to {path}")
        except Exception as e:
            raise ToolError(f"Failed to write file: {str(e)}")

    async def append(self, path: str, content: str, mode: str = "text", encoding: str = "utf-8") -> ToolResult:
        """Append content to an existing file or create it if it doesn't exist."""
        full_path = await self._validate_path(path)
        try:
            await self._ensure_base_path_exists()
            if mode == "text":
                async with aiofiles.open(str(full_path), 'a', encoding=encoding) as f:
                    await f.write(content)
            elif mode == "binary":
                decoded_content = base64.b64decode(content)
                async with aiofiles.open(str(full_path), 'ab') as f:
                    await f.write(decoded_content)
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
            return ToolResult(output=f"Appended to file {path}")
        except Exception as e:
            raise ToolError(f"Failed to append to file: {str(e)}")

    async def delete(self, path: str, recursive: bool = False) -> ToolResult:
        """Delete a file or directory, optionally recursively."""
        full_path = await self._validate_path(path)
        try:
            if await aiofiles.os.path.isfile(str(full_path)):
                await aiofiles.os.remove(str(full_path))
            elif await aiofiles.os.path.isdir(str(full_path)):
                if recursive:
                    await asyncio.to_thread(shutil.rmtree, str(full_path))
                else:
                    await aiofiles.os.rmdir(str(full_path))
            else:
                raise ToolError("Path does not exist")
            self._file_history.pop(full_path, None)  # Clear undo history
            return ToolResult(output=f"Deleted {path}")
        except Exception as e:
            raise ToolError(f"Failed to delete: {str(e)}")

    async def exists(self, path: str) -> ToolResult:
        """Check if a path exists."""
        try:
            full_path = await self._validate_path(path)
            exists = await aiofiles.os.path.exists(str(full_path))
            return ToolResult(output=str(exists))
        except Exception as e:
            return ToolResult(error=f"Failed to check existence: {str(e)}")

    async def list_dir(self, path: str) -> ToolResult:
        """List the contents of a directory."""
        full_path = await self._validate_path(path)
        if not await aiofiles.os.path.isdir(str(full_path)):
            raise ToolError("Path is not a directory")
        try:
            contents = []
            items = await asyncio.to_thread(list, full_path.iterdir())
            for p in sorted(items):
                if await aiofiles.os.path.isdir(str(p)):
                    contents.append(f"{p.name}/")
                else:
                    contents.append(p.name)
            return ToolResult(output="\n".join(contents))
        except Exception as e:
            raise ToolError(f"Failed to list directory: {str(e)}")

    async def mkdir(self, path: str) -> ToolResult:
        """Create a directory, including parent directories if needed."""
        full_path = await self._validate_path(path)
        try:
            await self._ensure_base_path_exists()
            await asyncio.to_thread(full_path.mkdir, parents=True, exist_ok=True)
            return ToolResult(output=f"Directory created: {path}")
        except Exception as e:
            raise ToolError(f"Failed to create directory: {str(e)}")

    async def rmdir(self, path: str) -> ToolResult:
        """Remove an empty directory."""
        full_path = await self._validate_path(path)
        try:
            await self._ensure_base_path_exists()
            await aiofiles.os.rmdir(str(full_path))
            return ToolResult(output=f"Directory removed: {path}")
        except Exception as e:
            raise ToolError(f"Failed to remove directory: {str(e)}")

    async def move(self, src: str, dst: str) -> ToolResult:
        """Move or rename a file or directory."""
        src_path = await self._validate_path(src)
        dst_path = await self._validate_path(dst)
        try:
            await self._ensure_base_path_exists()
            await asyncio.to_thread(dst_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(src_path.rename, dst_path)
            if src_path in self._file_history:
                self._file_history[dst_path] = self._file_history.pop(src_path)
            return ToolResult(output=f"Moved {src} to {dst}")
        except Exception as e:
            raise ToolError(f"Failed to move: {str(e)}")

    async def copy(self, src: str, dst: str) -> ToolResult:
        """Copy a file or directory."""
        src_path = await self._validate_path(src)
        dst_path = await self._validate_path(dst)
        try:
            await self._ensure_base_path_exists()
            await asyncio.to_thread(dst_path.parent.mkdir, parents=True, exist_ok=True)
            if await aiofiles.os.path.isfile(str(src_path)):
                await asyncio.to_thread(shutil.copy2, str(src_path), str(dst_path))
            elif await aiofiles.os.path.isdir(str(src_path)):
                await asyncio.to_thread(shutil.copytree, str(src_path), str(dst_path))
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
        full_path = await self._validate_path(path)
        if await aiofiles.os.path.isdir(str(full_path)):
            if view_range:
                raise ToolError("view_range not applicable for directories")
            # List directory contents using pathlib
            contents = []
            try:
                items = await asyncio.to_thread(list, full_path.iterdir())
                for item in sorted(items):
                    if await aiofiles.os.path.isdir(str(item)):
                        contents.append(f"  {item.name}/")
                    else:
                        contents.append(f"  {item.name}")
                output = f"Directory contents of {path}:\n" + "\n".join(contents)
                return ToolResult(output=output)
            except Exception as e:
                raise ToolError(f"Failed to list directory: {str(e)}")
        
        try:
            async with aiofiles.open(str(full_path), 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()
            if view_range:
                lines = content.splitlines()
                start, end = view_range
                # Allow negative offsets: -1 refers to the last line, -2 the line before, etc.
                if start < 0:
                    start = len(lines) + start + 1  # convert to 1-indexed positive
                if end < 0:
                    end = len(lines) + end + 1
                if start < 1 or start > len(lines) or end < start or end > len(lines):
                    raise ToolError(f"Invalid view_range: {view_range}. File has {len(lines)} lines, but requested range is [{start}, {end}]")
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
        full_path = await self._validate_path(path)
        if await aiofiles.os.path.exists(str(full_path)):
            raise ToolError("File already exists")
        try:
            await self._ensure_base_path_exists()
            await asyncio.to_thread(full_path.parent.mkdir, parents=True, exist_ok=True)
            if mode == "text":
                async with aiofiles.open(str(full_path), 'w', encoding=encoding) as f:
                    await f.write(content)
            elif mode == "binary":
                decoded_content = base64.b64decode(content)
                async with aiofiles.open(str(full_path), 'wb') as f:
                    await f.write(decoded_content)
            else:
                raise ToolError("Invalid mode: choose 'text' or 'binary'")
            return ToolResult(output=f"File created: {path}")
        except Exception as e:
            raise ToolError(f"Failed to create file: {str(e)}")

    async def replace(self, path: str, old_str: str, new_str: str, all_occurrences: bool = False) -> ToolResult:
        """Replace a string in a file, optionally all occurrences."""
        full_path = await self._validate_path(path)
        if not await aiofiles.os.path.isfile(str(full_path)):
            raise ToolError("Path is not a file")
        try:
            async with aiofiles.open(str(full_path), 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()

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
            if len(self._file_history[full_path]) > 2:
                self._file_history[full_path].pop(0)
            async with aiofiles.open(str(full_path), 'w', encoding='utf-8', errors='replace') as f:
                await f.write(new_content)
            return ToolResult(output=f"Replaced \"{_shorten(old_str)}\" with \"{_shorten(new_str)}\"")
        except Exception as e:
            raise ToolError(f"Failed to replace string: {str(e)}")

    async def insert(self, path: str, line: int, text: str) -> ToolResult:
        """Insert text at a specific line in a file."""
        full_path = await self._validate_path(path)
        if not await aiofiles.os.path.isfile(str(full_path)):
            raise ToolError("Path is not a file")
        try:
            async with aiofiles.open(str(full_path), 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()
            lines = content.splitlines()
            if line < 1 or line > len(lines) + 1:
                raise ToolError(f"Line number {line} is out of range")
            lines.insert(line - 1, text)
            new_content = "\n".join(lines)
            self._file_history[full_path].append(content)
            if len(self._file_history[full_path]) > 2:
                self._file_history[full_path].pop(0)
            async with aiofiles.open(str(full_path), 'w', encoding='utf-8', errors='replace') as f:
                await f.write(new_content)
            return ToolResult(output=f"Inserted \"{_shorten(text)}\" at line {line}")
        except Exception as e:
            raise ToolError(f"Failed to insert text: {str(e)}")

    async def delete_lines(self, path: str, lines: List[int]) -> ToolResult:
        """Delete specified lines from a file."""
        full_path = await self._validate_path(path)
        if not await aiofiles.os.path.isfile(str(full_path)):
            raise ToolError("Path is not a file")
        try:
            async with aiofiles.open(str(full_path), 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()
            file_lines = content.splitlines()
            lines_to_delete = set(lines)
            new_lines = [line for i, line in enumerate(file_lines, 1) if i not in lines_to_delete]
            new_content = "\n".join(new_lines)
            self._file_history[full_path].append(content)
            if len(self._file_history[full_path]) > 2:
                self._file_history[full_path].pop(0)
            async with aiofiles.open(str(full_path), 'w', encoding='utf-8', errors='replace') as f:
                await f.write(new_content)
            return ToolResult(output=f"Deleted lines {lines}")
        except Exception as e:
            raise ToolError(f"Failed to delete lines: {str(e)}")

    async def undo(self, path: str) -> ToolResult:
        """Undo the last text editing operation on a file."""
        full_path = await self._validate_path(path)
        if not await aiofiles.os.path.isfile(str(full_path)):
            raise ToolError("File does not exist")
        if not self._file_history.get(full_path):
            raise ToolError("No undo history available")
        try:
            previous_content = self._file_history[full_path].pop()
            async with aiofiles.open(str(full_path), 'w', encoding='utf-8', errors='replace') as f:
                await f.write(previous_content)
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
        full_path = await self._validate_path(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        results = []

        async def search_file(file_path):
            try:
                # Skip if not a regular file (sockets, pipes, etc.)
                is_file = await aiofiles.os.path.isfile(str(file_path))
                is_symlink = await asyncio.to_thread(file_path.is_symlink)
                if is_symlink:
                    resolved = await asyncio.to_thread(file_path.resolve)
                    is_resolved_file = await aiofiles.os.path.isfile(str(resolved))
                    if not is_resolved_file:
                        return
                elif not is_file:
                    return
                
                async with aiofiles.open(str(file_path), 'r', encoding='utf-8') as f:
                    i = 0
                    async for line in f:
                        i += 1
                        if re.search(pattern, line, flags):
                            results.append({
                                'file': str(file_path.relative_to(self.base_path)),
                                'line_number': i if line_numbers else None,
                                'content': line.strip()
                            })
            except (UnicodeDecodeError, IOError, OSError):
                # Skip binary files and files that can't be read
                pass

        if await aiofiles.os.path.isfile(str(full_path)):
            await search_file(full_path)
        elif await aiofiles.os.path.isdir(str(full_path)):
            if not recursive:
                raise ToolError("Recursive search must be enabled for directories")
            # Use asyncio.to_thread for os.walk since there's no native async version
            for root, _, files in await asyncio.to_thread(os.walk, str(full_path)):
                for file in files:
                    await search_file(Path(root) / file)
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


@app.websocket("/bash/ws")
async def bash_websocket(websocket: WebSocket):
    """WebSocket endpoint providing live bash output suitable for xterm.js clients."""
    await websocket.accept()

    # Create a dedicated bash session for this WebSocket connection
    async with bash_tool._sessions_lock:
        session_id = max(bash_tool._sessions.keys(), default=0) + 1
        session = _BashSession(session_id=session_id)
        await session.start()
        bash_tool._sessions[session_id] = session

    try:
        while True:
            try:
                command = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            command = command.strip()
            if not command:
                continue  # Ignore empty commands

            try:
                async for chunk in session.stream_command(command):
                    await websocket.send_text(chunk)
            except ToolError as e:
                await websocket.send_text(f"ERROR: {e.message}\n")
            except Exception as e:
                await websocket.send_text(f"Unexpected error: {str(e)}\n")
    finally:
        # Clean up the session when the WebSocket disconnects
        session.stop()
        async with bash_tool._sessions_lock:
            bash_tool._sessions.pop(session_id, None)


@app.get("/status")
async def get_status():
    return {"status": "ok", "service": "bash-and-file-tool-api"}


@app.get("/list-files")
async def list_files(git_ignore: bool = False):
    try:
        items: List[Dict[str, Any]] = []
        try:
            # Fast path: ripgrep respects nested .gitignore files
            rg_cmd = [
                "rg",
                "--files",
                "--hidden",
                "--color",
                "never",
            ]
            if not git_ignore:
                rg_cmd.append("--no-ignore")

            proc = await asyncio.create_subprocess_exec(
                *rg_cmd,
                cwd=str(WORKSPACE_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"ripgrep failed with code {proc.returncode}: {stderr.decode().strip()}"
                )

            all_paths = [WORKSPACE_DIR / Path(p.strip()) for p in stdout.decode().splitlines() if p.strip()]
        except Exception:
            # Fallback to Python walk if ripgrep is unavailable or errors
            all_paths = await asyncio.to_thread(list, WORKSPACE_DIR.rglob('*'))
         
        for item_path in all_paths:
            # Check if path should be excluded
            if _is_excluded_path(item_path):
                continue
            
            is_dir = await aiofiles.os.path.isdir(str(item_path))
            relative_path = item_path.relative_to(WORKSPACE_DIR)
            
            item_info = {
                "name": item_path.name,
                "path": str(item_path),
                "relative_path": str(relative_path),
                "type": "directory" if is_dir else "file"
            }
            items.append(item_info)
         
        # Sort items by relative path
        items.sort(key=lambda x: x["relative_path"])
         
        return {
            "workspace_path": str(WORKSPACE_DIR),
            "total_items": len(items),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.get("/file/{file_path:path}")
async def get_file(file_path: str):
    """Get a specific file from the workspace"""
    try:
        full_path = WORKSPACE_DIR / file_path
        
        # Security check: ensure the path is within workspace
        full_path = await asyncio.to_thread(full_path.resolve)
        workspace_resolved = await asyncio.to_thread(WORKSPACE_DIR.resolve)
        if not str(full_path).startswith(str(workspace_resolved)):
            raise HTTPException(status_code=403, detail="Access denied: Path outside workspace")
        
        # Check if file is excluded
        if _is_excluded_path(full_path):
            raise HTTPException(status_code=403, detail="Access denied: File is excluded from serving")
        
        if not await aiofiles.os.path.exists(str(full_path)):
            raise HTTPException(status_code=404, detail="File not found")
        
        if not await aiofiles.os.path.isfile(str(full_path)):
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
            {"path": "/list-files", "method": "GET", "description": "List all files and directories recursively in /project/workspace"},
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