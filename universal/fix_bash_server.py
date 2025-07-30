#!/usr/bin/env python3
"""
Script to fix async bottlenecks in bash_server.py
"""

def fix_bash_server():
    with open('bash_server.py', 'r') as f:
        content = f.read()
    
    # 1. Add logging imports
    if 'import logging' not in content:
        content = content.replace(
            'import uvicorn',
            'import uvicorn\nimport logging\nimport time\n\n# Configure logging for performance monitoring\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)'
        )
    
    # 2. Add file size constant to FileTool class
    content = content.replace(
        'class FileTool(BaseAnthropicTool):\n    """\n    A filesystem editor tool that allows the agent to view, create, and edit files.\n    """\n\n    name: ClassVar[Literal["file"]] = "file"',
        'class FileTool(BaseAnthropicTool):\n    """\n    A filesystem editor tool that allows the agent to view, create, and edit files.\n    """\n\n    name: ClassVar[Literal["file"]] = "file"\n    # Maximum file size (in bytes) to store in history to prevent memory issues\n    MAX_HISTORY_FILE_SIZE = 1024 * 1024  # 1MB'
    )
    
    # 3. Add history helper method after _ensure_base_path_exists
    history_method = '''
    def _add_to_history(self, full_path: Path, content: str) -> None:
        """Add content to file history with size limits to prevent memory issues"""
        # Skip history for very large files to prevent memory problems
        if len(content.encode("utf-8")) > self.MAX_HISTORY_FILE_SIZE:
            # Keep only the most recent version for large files
            self._file_history[full_path] = []
            return
        
        self._file_history[full_path].append(content)
        if len(self._file_history[full_path]) > 5:
            self._file_history[full_path].pop(0)
'''
    
    content = content.replace(
        'async def _validate_path(self, path: str) -> Path:',
        history_method + '\n    async def _validate_path(self, path: str) -> Path:'
    )
    
    # 4. Replace direct history usage with helper method
    content = content.replace(
        'self._file_history[full_path].append(content)\n            if len(self._file_history[full_path]) > 5:\n                self._file_history[full_path].pop(0)',
        'self._add_to_history(full_path, content)'
    )
    
    # 5. Add timing to file_action endpoint
    old_file_action = '''@app.post("/file", response_model=ToolResponse)
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
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")'''
        
    new_file_action = '''@app.post("/file", response_model=ToolResponse)
async def file_action(request: FileRequest):
    """Execute file operations"""
    start_time = time.time()
    try:
        # Convert request to kwargs, excluding None values
        kwargs = request.model_dump(exclude_none=True)
        logger.info(f"Processing file command: {request.command} for path: {getattr(request, 'path', 'unknown')}")
        result = await file_tool(**kwargs)
        elapsed = time.time() - start_time
        logger.info(f"File operation completed in {elapsed:.2f}s")
        return _tool_result_to_response(result)
    except ToolError as e:
        elapsed = time.time() - start_time
        logger.error(f"File operation failed after {elapsed:.2f}s: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Unexpected error after {elapsed:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")'''
    
    content = content.replace(old_file_action, new_file_action)
    
    # 6. Update uvicorn configuration
    old_uvicorn = '''if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )'''
    
    new_uvicorn = '''if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
        # Prevent hanging connections from blocking the server
        limit_concurrency=100,
        # Add request timeout to prevent indefinite hangs
        timeout_notify=30
    )'''
    
    content = content.replace(old_uvicorn, new_uvicorn)
    
    # Write the fixed content
    with open('bash_server.py', 'w') as f:
        f.write(content)
    
    print("Successfully applied async bottleneck fixes to bash_server.py")

if __name__ == '__main__':
    fix_bash_server()