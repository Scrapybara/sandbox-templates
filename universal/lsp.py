#!/usr/bin/env python3
"""
LSP CLI Tool - Language Server Protocol client
Supports Python, TypeScript, JavaScript, Java, Rust, C#, Go, Dart, Ruby, and Kotlin with auto-detection
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import multilspy components
try:
    from multilspy import SyncLanguageServer
    from multilspy.multilspy_config import MultilspyConfig, Language
    from multilspy.multilspy_logger import MultilspyLogger
    from multilspy import multilspy_types
except ImportError:
    print("Error: multilspy not found. Please install with: pip install multilspy", file=sys.stderr)
    sys.exit(1)


class LanguageDetector:
    """Detect language based on file extension"""
    
    PYTHON_EXTENSIONS = {'.py', '.pyw', '.pyi'}
    TYPESCRIPT_EXTENSIONS = {'.ts', '.tsx'}
    JAVASCRIPT_EXTENSIONS = {'.js', '.jsx'}
    JAVA_EXTENSIONS = {'.java'}
    RUST_EXTENSIONS = {'.rs'}
    CSHARP_EXTENSIONS = {'.cs'}
    GO_EXTENSIONS = {'.go'}
    DART_EXTENSIONS = {'.dart'}
    RUBY_EXTENSIONS = {'.rb'}
    KOTLIN_EXTENSIONS = {'.kt', '.kts'}
    
    @classmethod
    def detect_language(cls, file_path: str) -> Optional[Language]:
        """Detect language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext in cls.PYTHON_EXTENSIONS:
            return Language.PYTHON
        elif ext in cls.TYPESCRIPT_EXTENSIONS:
            return Language.TYPESCRIPT
        elif ext in cls.JAVASCRIPT_EXTENSIONS:
            return Language.JAVASCRIPT
        elif ext in cls.JAVA_EXTENSIONS:
            return Language.JAVA
        elif ext in cls.RUST_EXTENSIONS:
            return Language.RUST
        elif ext in cls.CSHARP_EXTENSIONS:
            return Language.CSHARP
        elif ext in cls.GO_EXTENSIONS:
            return Language.GO
        elif ext in cls.DART_EXTENSIONS:
            return Language.DART
        elif ext in cls.RUBY_EXTENSIONS:
            return Language.RUBY
        elif ext in cls.KOTLIN_EXTENSIONS:
            return Language.KOTLIN
        
        return None


class MultilspyCLI:
    """Main CLI class for multilspy operations"""
    
    def __init__(self, repo_root: str, language: Optional[Language] = None, verbose: bool = False):
        self.repo_root = os.path.abspath(repo_root)
        self.language = language
        self.verbose = verbose
        self.logger = MultilspyLogger()
        
        # Configure logging level
        if verbose:
            self.logger.logger.setLevel(logging.DEBUG)
        else:
            self.logger.logger.setLevel(logging.WARNING)
    
    def _get_language_for_file(self, file_path: str) -> Language:
        """Get language for a specific file or directory, with auto-detection if not specified"""
        if self.language:
            return self.language
        
        # Check if it's a directory
        if os.path.isdir(file_path):
            # For directories, detect language based on files inside
            language_counts = {}
            
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    detected = LanguageDetector.detect_language(file)
                    if detected:
                        language_counts[detected] = language_counts.get(detected, 0) + 1
            
            if language_counts:
                # Return the most common language
                return max(language_counts, key=language_counts.get)
            else:
                # Default to Python if no recognizable files found
                return Language.PYTHON
        
        detected = LanguageDetector.detect_language(file_path)
        if not detected:
            raise ValueError(f"Could not detect language for file: {file_path}")
        
        return detected
    
    def _create_language_server(self, language: Language) -> SyncLanguageServer:
        """Create and configure language server"""
        config = MultilspyConfig.from_dict({"code_language": language})
        return SyncLanguageServer.create(config, self.logger, self.repo_root)
    
    def _format_output(self, data: Any) -> str:
        """Format output as JSON"""
        return json.dumps(data, indent=2, default=str)
    
    def _convert_locations_to_dict(self, locations: List[multilspy_types.Location]) -> List[Dict[str, Any]]:
        """Convert Location objects to dictionaries for JSON serialization"""
        result = []
        for loc in locations:
            result.append({
                'uri': loc['uri'],
                'range': loc['range'],
                'absolutePath': loc['absolutePath'],
                'relativePath': loc['relativePath']
            })
        return result
    
    def _convert_symbols_to_dict(self, symbols: List[multilspy_types.UnifiedSymbolInformation]) -> List[Dict[str, Any]]:
        """Convert symbol objects to dictionaries for JSON serialization"""
        # SymbolKind enum mapping
        symbol_kind_names = {
            1: 'File', 2: 'Module', 3: 'Namespace', 4: 'Package', 5: 'Class',
            6: 'Method', 7: 'Property', 8: 'Field', 9: 'Constructor', 10: 'Enum',
            11: 'Interface', 12: 'Function', 13: 'Variable', 14: 'Constant',
            15: 'String', 16: 'Number', 17: 'Boolean', 18: 'Array', 19: 'Object',
            20: 'Key', 21: 'Null', 22: 'EnumMember', 23: 'Struct', 24: 'Event',
            25: 'Operator', 26: 'TypeParameter'
        }
        
        result = []
        for symbol in symbols:
            kind_value = symbol['kind']
            kind_int = int(kind_value)
            symbol_dict = {
                'name': symbol['name'],
                'kind': kind_int,
                'kindName': symbol_kind_names.get(kind_int, f'Unknown({kind_int})')
            }
            
            # Add optional fields if present
            if 'detail' in symbol:
                symbol_dict['detail'] = symbol['detail']
            if 'range' in symbol:
                symbol_dict['range'] = symbol['range']
            if 'selectionRange' in symbol:
                symbol_dict['selectionRange'] = symbol['selectionRange']
            if 'location' in symbol:
                symbol_dict['location'] = symbol['location']
            if 'containerName' in symbol:
                symbol_dict['containerName'] = symbol['containerName']
            
            result.append(symbol_dict)
        return result
    
    def get_definitions(self, file_path: str, line: int, column: int) -> str:
        """Get symbol definitions"""
        language = self._get_language_for_file(file_path)
        lsp = self._create_language_server(language)
        
        relative_path = os.path.relpath(file_path, self.repo_root)
        
        with lsp.start_server():
            locations = lsp.request_definition(relative_path, line, column)
            return self._format_output(self._convert_locations_to_dict(locations))
    
    def get_references(self, file_path: str, line: int, column: int) -> str:
        """Get symbol references"""
        language = self._get_language_for_file(file_path)
        lsp = self._create_language_server(language)
        
        relative_path = os.path.relpath(file_path, self.repo_root)
        
        with lsp.start_server():
            locations = lsp.request_references(relative_path, line, column)
            return self._format_output(self._convert_locations_to_dict(locations))
    
    def get_hover(self, file_path: str, line: int, column: int) -> str:
        """Get hover information"""
        language = self._get_language_for_file(file_path)
        lsp = self._create_language_server(language)
        
        relative_path = os.path.relpath(file_path, self.repo_root)
        
        with lsp.start_server():
            hover_info = lsp.request_hover(relative_path, line, column)
            if hover_info:
                return self._format_output({
                    'contents': hover_info['contents'],
                    'range': hover_info.get('range')
                })
            else:
                return self._format_output({'contents': None, 'range': None})
    
    def get_symbols(self, file_path: str) -> str:
        """Get document symbols"""
        language = self._get_language_for_file(file_path)
        lsp = self._create_language_server(language)
        
        relative_path = os.path.relpath(file_path, self.repo_root)
        
        with lsp.start_server():
            symbols, tree = lsp.request_document_symbols(relative_path)
            return self._format_output(self._convert_symbols_to_dict(symbols))
    
    def get_workspace_symbols(self, query: str, language: Optional[Language] = None) -> str:
        """Get workspace symbols"""
        if not language:
            language = self.language or Language.PYTHON  # Default to Python if no language specified
        
        lsp = self._create_language_server(language)
        
        with lsp.start_server():
            symbols = lsp.request_workspace_symbol(query)
            if symbols:
                return self._format_output(self._convert_symbols_to_dict(symbols))
            else:
                return self._format_output([])
    
    def get_diagnostics(self, file_path: str) -> str:
        """Get diagnostics for a file"""
        language = self._get_language_for_file(file_path)
        
        # Dispatch to language-specific diagnostic functions
        diagnostic_methods = {
            Language.PYTHON: self._get_python_diagnostics,
            Language.TYPESCRIPT: self._get_typescript_diagnostics,
            Language.JAVASCRIPT: self._get_javascript_diagnostics,
            Language.JAVA: self._get_java_diagnostics,
            Language.RUST: self._get_rust_diagnostics,
            Language.CSHARP: self._get_csharp_diagnostics,
            Language.GO: self._get_go_diagnostics,
            Language.DART: self._get_dart_diagnostics,
            Language.RUBY: self._get_ruby_diagnostics,
            Language.KOTLIN: self._get_kotlin_diagnostics,
        }
        
        diagnostic_method = diagnostic_methods.get(language)
        if diagnostic_method:
            return diagnostic_method(file_path)
        else:
            return self._format_output({
                'error': f'Diagnostics not supported for language: {language}'
            })
    
    def _get_python_diagnostics(self, file_path: str) -> str:
        """
        Get Python diagnostics using pyright and ruff
        Supports both individual files and directories
        """
        diagnostics = []
        errors = []
        
        # Convert to absolute path if needed
        abs_file_path = os.path.abspath(file_path)
        is_directory = os.path.isdir(abs_file_path)
        
        # Run pyright for type checking
        try:
            result = subprocess.run(
                ['pyright', '--outputjson', abs_file_path],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                pyright_data = json.loads(result.stdout)
                for diag in pyright_data.get('generalDiagnostics', []):
                    diagnostics.append({
                        'source': 'pyright',
                        'severity': diag['severity'],
                        'message': diag['message'],
                        'file': diag['file'],  # Include the specific file path for directory analysis
                        'range': {
                            'start': {
                                'line': diag['range']['start']['line'],
                                'character': diag['range']['start']['character']
                            },
                            'end': {
                                'line': diag['range']['end']['line'],
                                'character': diag['range']['end']['character']
                            }
                        },
                        'code': diag.get('rule', ''),
                        'relatedInformation': []
                    })
        except subprocess.TimeoutExpired:
            errors.append('Pyright timed out after 30 seconds')
        except FileNotFoundError:
            errors.append('Pyright not found. Please install pyright: npm install -g pyright')
        except json.JSONDecodeError:
            errors.append('Failed to parse pyright output')
        except Exception as e:
            errors.append(f'Pyright error: {str(e)}')
        
        # Run ruff for linting
        try:
            result = subprocess.run(
                ['ruff', 'check', '--output-format', 'json', abs_file_path],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                ruff_data = json.loads(result.stdout)
                for diag in ruff_data:
                    # Convert ruff format to LSP-like format
                    severity = 'warning'  # Most ruff issues are warnings
                    if diag['code'] and (diag['code'].startswith('E9') or 
                                       diag['code'].startswith('F63') or 
                                       diag['code'].startswith('F7') or 
                                       diag['code'].startswith('F82')):
                        severity = 'error'  # Some codes are more serious
                    
                    diagnostics.append({
                        'source': 'ruff',
                        'severity': severity,
                        'message': diag['message'],
                        'file': diag['filename'],  # Include the specific file path for directory analysis
                        'range': {
                            'start': {
                                'line': diag['location']['row'] - 1,  # Ruff uses 1-based lines
                                'character': diag['location']['column'] - 1  # Ruff uses 1-based columns
                            },
                            'end': {
                                'line': diag['end_location']['row'] - 1,
                                'character': diag['end_location']['column'] - 1
                            }
                        },
                        'code': diag['code'],
                        'relatedInformation': []
                    })
        except subprocess.TimeoutExpired:
            errors.append('Ruff timed out after 30 seconds')
        except FileNotFoundError:
            errors.append('Ruff not found. Please install ruff: pip install ruff')
        except json.JSONDecodeError:
            errors.append('Failed to parse ruff output')
        except Exception as e:
            errors.append(f'Ruff error: {str(e)}')
        
        # Group diagnostics by file if analyzing a directory
        files_analyzed = set()
        if is_directory:
            for diag in diagnostics:
                if 'file' in diag:
                    files_analyzed.add(diag['file'])
        
        return self._format_output({
            'diagnostics': diagnostics,
            'file': file_path,
            'is_directory': is_directory,
            'language': 'python',
            'errors': errors if errors else None,
            'summary': {
                'total': len(diagnostics),
                'pyright': len([d for d in diagnostics if d['source'] == 'pyright']),
                'ruff': len([d for d in diagnostics if d['source'] == 'ruff']),
                'files_analyzed': len(files_analyzed) if is_directory else 1
            }
        })
    
    def _get_typescript_diagnostics(self, file_path: str) -> str:
        """
        Get TypeScript diagnostics using tsc and eslint
        Supports both individual files and directories
        """
        diagnostics = []
        errors = []
        
        # Convert to absolute path if needed
        abs_file_path = os.path.abspath(file_path)
        is_directory = os.path.isdir(abs_file_path)
        
        # Run tsc for type checking with no color output
        try:
            if is_directory:
                # For directories, run tsc on the whole project
                result = subprocess.run(
                    ['tsc', '--noEmit', '--pretty', 'false'],
                    cwd=abs_file_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            else:
                # For single files
                result = subprocess.run(
                    ['tsc', '--noEmit', '--pretty', 'false', abs_file_path],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            # Parse tsc output (can be in stderr or stdout)
            output = result.stderr or result.stdout
            if output:
                lines = output.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Found ') and '(' in line and ')' in line and ':' in line:
                        # Parse tsc error format: file.ts(line,col): error TS####: message
                        try:
                            # Split on the first colon after the position
                            parts = line.split(': ', 1)
                            if len(parts) == 2:
                                file_and_pos = parts[0]
                                message_part = parts[1]
                                
                                # Extract file path and position
                                if '(' in file_and_pos and ')' in file_and_pos:
                                    file_name = file_and_pos.split('(')[0]
                                    pos_part = file_and_pos.split('(')[1].split(')')[0]
                                    
                                    if ',' in pos_part:
                                        line_num, col_num = pos_part.split(',')
                                        line_num = int(line_num) - 1  # Convert to 0-based
                                        col_num = int(col_num) - 1    # Convert to 0-based
                                        
                                        # Extract error code and message
                                        message = message_part.strip()
                                        code = ''
                                        if 'TS' in message:
                                            parts = message.split()
                                            for part in parts:
                                                if part.startswith('TS') and ':' in part:
                                                    code = part.rstrip(':')
                                                    break
                                        
                                        severity = 'error' if 'error' in line else 'warning'
                                        
                                        diagnostics.append({
                                            'source': 'tsc',
                                            'severity': severity,
                                            'message': message,
                                            'file': file_name,
                                            'range': {
                                                'start': {
                                                    'line': line_num,
                                                    'character': col_num
                                                },
                                                'end': {
                                                    'line': line_num,
                                                    'character': col_num + 1
                                                }
                                            },
                                            'code': code,
                                            'relatedInformation': []
                                        })
                        except (ValueError, IndexError):
                            # Skip lines that don't match expected format
                            continue
                            
        except subprocess.TimeoutExpired:
            errors.append('TypeScript compiler (tsc) timed out after 60 seconds')
        except FileNotFoundError:
            errors.append('TypeScript compiler (tsc) not found. Please install: npm install -g typescript')
        except Exception as e:
            errors.append(f'TypeScript compiler error: {str(e)}')
        
        # Run eslint for linting
        try:
            if is_directory:
                # For directories, lint all TypeScript files
                eslint_result = subprocess.run(
                    ['eslint', '--format', 'json', '**/*.{ts,tsx}'],
                    cwd=abs_file_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            else:
                # For single files
                eslint_result = subprocess.run(
                    ['eslint', '--format', 'json', abs_file_path],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            if eslint_result.stdout:
                eslint_data = json.loads(eslint_result.stdout)
                for file_data in eslint_data:
                    for msg in file_data.get('messages', []):
                        severity = 'error' if msg['severity'] == 2 else 'warning'
                        
                        diagnostics.append({
                            'source': 'eslint',
                            'severity': severity,
                            'message': msg['message'],
                            'file': file_data['filePath'],
                            'range': {
                                'start': {
                                    'line': (msg.get('line', 1) - 1),  # Convert to 0-based
                                    'character': (msg.get('column', 1) - 1)  # Convert to 0-based
                                },
                                'end': {
                                    'line': (msg.get('endLine', msg.get('line', 1)) - 1),
                                    'character': (msg.get('endColumn', msg.get('column', 1)) - 1)
                                }
                            },
                            'code': msg.get('ruleId', ''),
                            'relatedInformation': []
                        })
                        
        except subprocess.TimeoutExpired:
            errors.append('ESLint timed out after 60 seconds')
        except FileNotFoundError:
            errors.append('ESLint not found. Please install: npm install -g eslint')
        except json.JSONDecodeError:
            errors.append('Failed to parse ESLint output')
        except Exception as e:
            errors.append(f'ESLint error: {str(e)}')
        
        # Group diagnostics by file if analyzing a directory
        files_analyzed = set()
        if is_directory:
            for diag in diagnostics:
                if 'file' in diag:
                    files_analyzed.add(diag['file'])
        
        return self._format_output({
            'diagnostics': diagnostics,
            'file': file_path,
            'is_directory': is_directory,
            'language': 'typescript',
            'errors': errors if errors else None,
            'summary': {
                'total': len(diagnostics),
                'tsc': len([d for d in diagnostics if d['source'] == 'tsc']),
                'eslint': len([d for d in diagnostics if d['source'] == 'eslint']),
                'files_analyzed': len(files_analyzed) if is_directory else 1
            }
        })
    
    def _get_javascript_diagnostics(self, file_path: str) -> str:
        """
        Get JavaScript diagnostics using eslint
        Supports both individual files and directories
        """
        diagnostics = []
        errors = []
        
        # Convert to absolute path if needed
        abs_file_path = os.path.abspath(file_path)
        is_directory = os.path.isdir(abs_file_path)
        
        # Run eslint for linting
        try:
            if is_directory:
                # For directories, lint all JavaScript files
                eslint_result = subprocess.run(
                    ['eslint', '--format', 'json', '**/*.{js,jsx}'],
                    cwd=abs_file_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            else:
                # For single files
                eslint_result = subprocess.run(
                    ['eslint', '--format', 'json', abs_file_path],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            if eslint_result.stdout:
                eslint_data = json.loads(eslint_result.stdout)
                for file_data in eslint_data:
                    for msg in file_data.get('messages', []):
                        severity = 'error' if msg['severity'] == 2 else 'warning'
                        
                        diagnostics.append({
                            'source': 'eslint',
                            'severity': severity,
                            'message': msg['message'],
                            'file': file_data['filePath'],
                            'range': {
                                'start': {
                                    'line': (msg.get('line', 1) - 1),  # Convert to 0-based
                                    'character': (msg.get('column', 1) - 1)  # Convert to 0-based
                                },
                                'end': {
                                    'line': (msg.get('endLine', msg.get('line', 1)) - 1),
                                    'character': (msg.get('endColumn', msg.get('column', 1)) - 1)
                                }
                            },
                            'code': msg.get('ruleId', ''),
                            'relatedInformation': []
                        })
                        
        except subprocess.TimeoutExpired:
            errors.append('ESLint timed out after 60 seconds')
        except FileNotFoundError:
            errors.append('ESLint not found. Please install: npm install -g eslint')
        except json.JSONDecodeError:
            errors.append('Failed to parse ESLint output')
        except Exception as e:
            errors.append(f'ESLint error: {str(e)}')
        
        # Group diagnostics by file if analyzing a directory
        files_analyzed = set()
        if is_directory:
            for diag in diagnostics:
                if 'file' in diag:
                    files_analyzed.add(diag['file'])
        
        return self._format_output({
            'diagnostics': diagnostics,
            'file': file_path,
            'is_directory': is_directory,
            'language': 'javascript',
            'errors': errors if errors else None,
            'summary': {
                'total': len(diagnostics),
                'eslint': len([d for d in diagnostics if d['source'] == 'eslint']),
                'files_analyzed': len(files_analyzed) if is_directory else 1
            }
        })
    
    def _get_java_diagnostics(self, file_path: str) -> str:
        """
        Skeleton function for Java diagnostics
        TODO: Implement custom Java diagnostics (outside of multilspy scope)
        """
        return self._format_output({
            'diagnostics': [],
            'message': 'Java diagnostics implementation pending',
            'file': file_path,
            'language': 'java'
        })
    
    def _get_rust_diagnostics(self, file_path: str) -> str:
        """
        Skeleton function for Rust diagnostics
        TODO: Implement custom Rust diagnostics (outside of multilspy scope)
        """
        return self._format_output({
            'diagnostics': [],
            'message': 'Rust diagnostics implementation pending',
            'file': file_path,
            'language': 'rust'
        })
    
    def _get_csharp_diagnostics(self, file_path: str) -> str:
        """
        Skeleton function for C# diagnostics
        TODO: Implement custom C# diagnostics (outside of multilspy scope)
        """
        return self._format_output({
            'diagnostics': [],
            'message': 'C# diagnostics implementation pending',
            'file': file_path,
            'language': 'csharp'
        })
    
    def _get_go_diagnostics(self, file_path: str) -> str:
        """
        Skeleton function for Go diagnostics
        TODO: Implement custom Go diagnostics (outside of multilspy scope)
        """
        return self._format_output({
            'diagnostics': [],
            'message': 'Go diagnostics implementation pending',
            'file': file_path,
            'language': 'go'
        })
    
    def _get_dart_diagnostics(self, file_path: str) -> str:
        """
        Skeleton function for Dart diagnostics
        TODO: Implement custom Dart diagnostics (outside of multilspy scope)
        """
        return self._format_output({
            'diagnostics': [],
            'message': 'Dart diagnostics implementation pending',
            'file': file_path,
            'language': 'dart'
        })
    
    def _get_ruby_diagnostics(self, file_path: str) -> str:
        """
        Skeleton function for Ruby diagnostics
        TODO: Implement custom Ruby diagnostics (outside of multilspy scope)
        """
        return self._format_output({
            'diagnostics': [],
            'message': 'Ruby diagnostics implementation pending',
            'file': file_path,
            'language': 'ruby'
        })
    
    def _get_kotlin_diagnostics(self, file_path: str) -> str:
        """
        Skeleton function for Kotlin diagnostics
        TODO: Implement custom Kotlin diagnostics (outside of multilspy scope)
        """
        return self._format_output({
            'diagnostics': [],
            'message': 'Kotlin diagnostics implementation pending',
            'file': file_path,
            'language': 'kotlin'
        })


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Multilspy CLI - Language Server Protocol client for multiple languages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get definitions at line 10, column 5 in main.py
  %(prog)s definitions src/main.py 10 5

  # Get references with auto-detected language
  %(prog)s references app.ts 25 12

  # Get symbols in a file
  %(prog)s symbols utils.py

  # Get hover information
  %(prog)s hover main.py 15 8

  # Get workspace symbols
  %(prog)s workspace-symbols "MyClass"

  # Get diagnostics (Python uses pyright and ruff)
  %(prog)s diagnostics main.py
  
  # Get diagnostics for TypeScript (uses tsc and eslint)
  %(prog)s diagnostics app.ts
  
  # Get diagnostics for JavaScript (uses eslint)
  %(prog)s diagnostics script.js
  
  # Get diagnostics for a directory
  %(prog)s diagnostics src/

Supported languages:
  - Python: .py, .pyw, .pyi
  - TypeScript: .ts, .tsx
  - JavaScript: .js, .jsx  
  - Java: .java
  - Rust: .rs
  - C#: .cs
  - Go: .go
  - Dart: .dart
  - Ruby: .rb
  - Kotlin: .kt, .kts

Language is auto-detected from file extension unless --language is specified.
        """)
    
    parser.add_argument('--repo-root', default='.', 
                       help='Repository root directory (default: current directory)')
    parser.add_argument('--language', choices=['python', 'typescript', 'javascript', 'java', 'rust', 'csharp', 'go', 'dart', 'ruby', 'kotlin'],
                       help='Force specific language (auto-detected if not specified)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Definitions command
    def_parser = subparsers.add_parser('definitions', help='Get symbol definitions')
    def_parser.add_argument('file', help='Source file path')
    def_parser.add_argument('line', type=int, help='Line number (0-based)')
    def_parser.add_argument('column', type=int, help='Column number (0-based)')
    
    # References command
    ref_parser = subparsers.add_parser('references', help='Get symbol references')
    ref_parser.add_argument('file', help='Source file path')
    ref_parser.add_argument('line', type=int, help='Line number (0-based)')
    ref_parser.add_argument('column', type=int, help='Column number (0-based)')
    
    # Hover command
    hover_parser = subparsers.add_parser('hover', help='Get hover information')
    hover_parser.add_argument('file', help='Source file path')
    hover_parser.add_argument('line', type=int, help='Line number (0-based)')
    hover_parser.add_argument('column', type=int, help='Column number (0-based)')
    
    # Symbols command
    symbols_parser = subparsers.add_parser('symbols', help='Get document symbols')
    symbols_parser.add_argument('file', help='Source file path')
    
    # Workspace symbols command
    ws_parser = subparsers.add_parser('workspace-symbols', help='Get workspace symbols')
    ws_parser.add_argument('query', help='Symbol search query')
    ws_parser.add_argument('--language', choices=['python', 'typescript', 'javascript', 'java', 'rust', 'csharp', 'go', 'dart', 'ruby', 'kotlin'],
                          help='Language for workspace search')
    
    # Diagnostics command
    diag_parser = subparsers.add_parser('diagnostics', help='Get diagnostics for a file or directory')
    diag_parser.add_argument('file', help='Source file or directory path')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Convert language string to enum if provided
    language = None
    if hasattr(args, 'language') and args.language:
        language = Language(args.language)
    elif args.language:
        language = Language(args.language)
    
    try:
        cli = MultilspyCLI(args.repo_root, language, args.verbose)
        
        if args.command == 'definitions':
            result = cli.get_definitions(args.file, args.line, args.column)
        elif args.command == 'references':
            result = cli.get_references(args.file, args.line, args.column)
        elif args.command == 'hover':
            result = cli.get_hover(args.file, args.line, args.column)
        elif args.command == 'symbols':
            result = cli.get_symbols(args.file)
        elif args.command == 'workspace-symbols':
            ws_language = None
            if hasattr(args, 'language') and args.language:
                ws_language = Language(args.language)
            result = cli.get_workspace_symbols(args.query, ws_language)
        elif args.command == 'diagnostics':
            result = cli.get_diagnostics(args.file)
        else:
            parser.print_help()
            return 1
        
        print(result)
        return 0
        
    except Exception as e:
        error_output = {
            'error': str(e),
            'command': args.command
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())