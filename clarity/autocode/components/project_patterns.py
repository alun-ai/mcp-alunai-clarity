"""
Project pattern management component for AutoCode domain.
"""

import json
import os
import uuid
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from clarity.shared.infrastructure import get_cache
from clarity.shared.exceptions import AutoCodeError
from clarity.shared.async_utils import AsyncFileProcessor, async_timed, AsyncBatcher, async_timer
from ..interfaces import ProjectPatternManager, AutoCodeComponent
from ..pattern_detector import PatternDetector


class ProjectPatternManagerImpl(ProjectPatternManager, AutoCodeComponent):
    """Implementation of project pattern management"""
    
    def __init__(self, config: Dict[str, Any], persistence_domain):
        """Initialize project pattern manager"""
        self.config = config
        self.persistence_domain = persistence_domain
        self.autocode_config = config.get("autocode", {})
        
        # Pattern caching
        self.pattern_cache = get_cache(
            "project_patterns",
            max_size=500,       # Cache 500 projects
            max_memory_mb=100,  # 100MB for pattern data
            default_ttl=3600.0  # 1 hour TTL
        )
        
        # Initialize pattern detector
        self.pattern_detector = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the pattern manager"""
        if self._initialized:
            return
            
        logger.info("Initializing Project Pattern Manager")
        
        try:
            # Initialize pattern detector
            self.pattern_detector = PatternDetector(self.autocode_config)
            await self.pattern_detector.initialize()
            
            # Load existing patterns
            await self.load_existing_patterns()
            
            self._initialized = True
            logger.info("Project Pattern Manager initialized successfully")
            
        except (OSError, ValueError, ImportError, KeyError) as e:
            logger.error(f"Failed to initialize Project Pattern Manager: {e}")
            raise AutoCodeError("Pattern manager initialization failed", cause=e)
    
    async def shutdown(self) -> None:
        """Shutdown the pattern manager"""
        if self.pattern_detector:
            await self.pattern_detector.shutdown()
        self._initialized = False
        logger.info("Project Pattern Manager shutdown complete")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": "ProjectPatternManager",
            "initialized": self._initialized,
            "cache_info": self.pattern_cache.get_info() if self._initialized else None,
            "pattern_detector_enabled": self.pattern_detector is not None
        }
    
    async def get_project_patterns(self, project_path: str, pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get patterns for a specific project with caching"""
        if not self._initialized:
            await self.initialize()
            
        # Create cache key
        cache_key = f"patterns_{project_path}"
        if pattern_types:
            cache_key += f"_{hash(tuple(sorted(pattern_types)))}"
        
        # Try cache first
        cached_patterns = self.pattern_cache.get(cache_key)
        if cached_patterns is not None:
            return cached_patterns
        
        try:
            # Detect patterns
            patterns = await self.detect_project_patterns(project_path)
            
            # Filter by requested types if specified
            if pattern_types:
                filtered_patterns = {}
                for pattern_type in pattern_types:
                    if pattern_type in patterns:
                        filtered_patterns[pattern_type] = patterns[pattern_type]
                patterns = filtered_patterns
            
            # Add metadata
            patterns["metadata"] = {
                "project_path": project_path,
                "pattern_types_requested": pattern_types,
                "total_patterns": len(patterns) - 1,  # Exclude metadata
                "scan_timestamp": datetime.utcnow().isoformat(),
                "cache_key": cache_key
            }
            
            # Cache the result
            self.pattern_cache.set(cache_key, patterns)
            
            logger.debug(f"Generated patterns for project: {project_path}")
            return patterns
            
        except (AutoCodeError, OSError, ValueError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to get project patterns for {project_path}: {e}")
            raise AutoCodeError(f"Pattern detection failed for {project_path}", cause=e)
    
    async def detect_project_patterns(self, project_path: str) -> Dict[str, Any]:
        """Detect and analyze patterns in a project"""
        if not self.pattern_detector:
            raise AutoCodeError("Pattern detector not initialized")
        
        try:
            # Use pattern detector to analyze project
            patterns = await self.pattern_detector.detect_patterns(project_path)
            
            # Enhance patterns with additional metadata
            enhanced_patterns = {
                "file_patterns": patterns.get("file_patterns", {}),
                "dependency_patterns": patterns.get("dependency_patterns", {}),
                "build_patterns": patterns.get("build_patterns", {}),
                "test_patterns": patterns.get("test_patterns", {}),
                "documentation_patterns": patterns.get("documentation_patterns", {}),
                "configuration_patterns": patterns.get("configuration_patterns", {}),
                "project_structure": await self._analyze_project_structure(project_path),
                "technology_stack": await self._detect_technology_stack(project_path),
                "complexity_metrics": await self._calculate_complexity_metrics(project_path)
            }
            
            return enhanced_patterns
            
        except (OSError, FileNotFoundError, PermissionError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Pattern detection failed for {project_path}: {e}")
            raise AutoCodeError(f"Pattern detection failed", cause=e)
    
    async def cache_project_patterns(self, project_path: str, patterns: Dict[str, Any]) -> None:
        """Cache patterns for a project"""
        cache_key = f"patterns_{project_path}"
        self.pattern_cache.set(cache_key, patterns)
        logger.debug(f"Cached patterns for project: {project_path}")
    
    async def load_existing_patterns(self) -> None:
        """Load existing patterns from storage"""
        try:
            # Query for stored project patterns
            pattern_memories = await self.persistence_domain.retrieve_memories(
                query="project patterns analysis",
                types=["project_pattern", "code_analysis"],
                limit=100
            )
            
            patterns_loaded = 0
            for memory in pattern_memories:
                try:
                    # Extract project path and patterns from memory
                    content = memory.get("content", {})
                    if isinstance(content, dict) and "project_path" in content:
                        project_path = content["project_path"]
                        patterns = content.get("patterns", {})
                        
                        # Cache the patterns
                        await self.cache_project_patterns(project_path, patterns)
                        patterns_loaded += 1
                        
                except (KeyError, ValueError, AttributeError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to load pattern from memory {memory.get('id')}: {e}")
                    continue
            
            logger.info(f"Loaded {patterns_loaded} existing project patterns")
            
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load existing patterns: {e}")
    
    @async_timed("project_structure_analysis")
    async def _analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze the structure of a project with async optimization"""
        async with async_timer("project_structure_analysis"):
            try:
                structure = {
                    "total_files": 0,
                    "directories": 0,
                    "file_types": {},
                    "max_depth": 0,
                    "large_files": [],
                    "empty_directories": []
                }
                
                # Use asyncio to make file system operations non-blocking
                loop = asyncio.get_event_loop()
                
                def analyze_directory_tree():
                    """Synchronous tree analysis to run in executor"""
                    dirs_to_process = []
                    
                    for root, dirs, files in os.walk(project_path):
                        # Calculate depth
                        depth = root[len(project_path):].count(os.sep)
                        structure["max_depth"] = max(structure["max_depth"], depth)
                        
                        # Count directories
                        structure["directories"] += len(dirs)
                        
                        # Collect file info for batch processing
                        file_paths = []
                        for file in files:
                            structure["total_files"] += 1
                            
                            # Count file types
                            ext = os.path.splitext(file)[1].lower()
                            if ext:
                                structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
                            
                            file_path = os.path.join(root, file)
                            file_paths.append(file_path)
                        
                        dirs_to_process.append((root, dirs, files, file_paths))
                        
                        # Check for empty directories
                        if not dirs and not files:
                            structure["empty_directories"].append(
                                os.path.relpath(root, project_path)
                            )
                    
                    return dirs_to_process
                
                # Run directory analysis in executor to avoid blocking
                dirs_to_process = await loop.run_in_executor(None, analyze_directory_tree)
                
                # Process large file detection concurrently
                large_file_tasks = []
                
                async def check_file_size(file_path: str):
                    """Check if file is large, run in executor"""
                    try:
                        size = await loop.run_in_executor(None, os.path.getsize, file_path)
                        if size > 1024 * 1024:  # Files > 1MB
                            return {
                                "path": os.path.relpath(file_path, project_path),
                                "size_mb": round(size / (1024 * 1024), 2)
                            }
                    except OSError:
                        pass
                    return None
                
                # Collect all file paths for concurrent processing
                all_file_paths = []
                for root, dirs, files, file_paths in dirs_to_process:
                    all_file_paths.extend(file_paths)
                
                # Process files in batches to find large ones
                if all_file_paths:
                    batcher = AsyncBatcher(max_concurrency=20, batch_size=50)
                    batch_result = await batcher.execute_batch(
                        all_file_paths,
                        check_file_size,
                        progress_callback=lambda done, total: logger.debug(f"Analyzed {done}/{total} files")
                    )
                    
                    # Collect large files from results
                    structure["large_files"] = [
                        result for result in batch_result.results 
                        if result is not None
                    ]
                
                return structure
                
            except (OSError, FileNotFoundError, PermissionError, ValueError) as e:
                logger.warning(f"Failed to analyze project structure: {e}")
                return {"error": str(e)}
    
    @async_timed("technology_stack_detection")
    async def _detect_technology_stack(self, project_path: str) -> Dict[str, Any]:
        """Detect the technology stack used in the project with async optimization"""
        async with async_timer("technology_stack_detection"):
            try:
                stack = {
                    "languages": [],
                    "frameworks": [],
                    "build_tools": [],
                    "package_managers": [],
                    "databases": [],
                    "testing_frameworks": [],
                    "ci_cd": []
                }
                
                # Check for common files that indicate technology stack
                indicators = {
                    "package.json": {"language": "JavaScript/TypeScript", "package_manager": "npm"},
                    "yarn.lock": {"package_manager": "Yarn"},
                    "requirements.txt": {"language": "Python", "package_manager": "pip"},
                    "Pipfile": {"language": "Python", "package_manager": "pipenv"},
                    "pyproject.toml": {"language": "Python", "build_tool": "Poetry/setuptools"},
                    "Cargo.toml": {"language": "Rust", "package_manager": "Cargo"},
                    "go.mod": {"language": "Go", "package_manager": "Go modules"},
                    "pom.xml": {"language": "Java", "build_tool": "Maven"},
                    "build.gradle": {"language": "Java/Kotlin", "build_tool": "Gradle"},
                    "Gemfile": {"language": "Ruby", "package_manager": "Bundler"},
                    "composer.json": {"language": "PHP", "package_manager": "Composer"},
                    "mix.exs": {"language": "Elixir", "build_tool": "Mix", "package_manager": "Hex"},
                    "mix.lock": {"language": "Elixir", "package_manager": "Hex"},
                    "config/config.exs": {"language": "Elixir"},
                    "config/prod.exs": {"language": "Elixir"},
                    "config/dev.exs": {"language": "Elixir"},
                    "config/test.exs": {"language": "Elixir"},
                    "lib/*/application.ex": {"language": "Elixir", "framework": "Phoenix"},
                    "rebar.config": {"language": "Erlang", "build_tool": "Rebar"},
                    "CMakeLists.txt": {"language": "C/C++", "build_tool": "CMake"},
                    "Makefile": {"build_tool": "Make"},
                    "dockerfile": {"technology": "Docker"},
                    "docker-compose.yml": {"technology": "Docker Compose"},
                    ".github/workflows": {"ci_cd": "GitHub Actions"},
                    ".gitlab-ci.yml": {"ci_cd": "GitLab CI"},
                    "Jenkinsfile": {"ci_cd": "Jenkins"}
                }
                
                # Use async file processing for concurrent analysis
                async def analyze_file_for_stack(file_path: str) -> List[Dict[str, Any]]:
                    """Analyze a single file for technology indicators"""
                    findings = []
                    file_name = os.path.basename(file_path).lower()
                    
                    if file_name in indicators:
                        info = indicators[file_name]
                        findings.append({
                            "file": file_path,
                            "indicators": info
                        })
                    
                    return findings
                
                # Collect all files for concurrent processing
                all_files = []
                all_dirs = []
                
                loop = asyncio.get_event_loop()
                
                def collect_files_and_dirs():
                    """Collect files and directories for processing"""
                    for root, dirs, files in os.walk(project_path):
                        for file in files:
                            all_files.append(os.path.join(root, file))
                        for dir_name in dirs:
                            all_dirs.append((root, dir_name))
                    
                    return all_files, all_dirs
                
                # Run file collection in executor
                collected_files, collected_dirs = await loop.run_in_executor(None, collect_files_and_dirs)
                
                # Process files concurrently
                if collected_files:
                    batcher = AsyncBatcher(max_concurrency=15, batch_size=30)
                    batch_result = await batcher.execute_batch(
                        collected_files,
                        analyze_file_for_stack
                    )
                    
                    # Process findings
                    for findings_list in batch_result.results:
                        for finding in findings_list:
                            info = finding["indicators"]
                            for category, value in info.items():
                                stack_key = f"{category}s" if not category.endswith("s") else category
                                if stack_key in stack and value not in stack[stack_key]:
                                    stack[stack_key].append(value)
                
                # Check directories for technology indicators
                for root, dir_name in collected_dirs:
                    dir_lower = dir_name.lower()
                    
                    # Check exact directory matches
                    if dir_lower in indicators:
                        info = indicators[dir_lower]
                        for category, value in info.items():
                            if category == "ci_cd" and value not in stack["ci_cd"]:
                                stack["ci_cd"].append(value)
                    
                    # Check for Elixir/Phoenix specific directories with error handling
                    try:
                        if dir_name == "lib" and os.path.exists(os.path.join(root, dir_name, "mix.exs")):
                            if "Elixir" not in stack["languages"]:
                                stack["languages"].append("Elixir")
                        
                        dir_path = os.path.join(root, dir_name)
                        if dir_name == "priv" and os.path.exists(dir_path) and os.path.isdir(dir_path):
                            try:
                                files = os.listdir(dir_path)
                                if any(f.endswith('.ex') for f in files if os.path.isfile(os.path.join(dir_path, f))):
                                    if "Phoenix" not in stack["frameworks"]:
                                        stack["frameworks"].append("Phoenix")
                            except (OSError, PermissionError):
                                pass
                        
                        # Check for web directory (typical Phoenix structure)
                        if dir_name in ["assets", "web"]:
                            # Look for parent directory containing mix.exs
                            parent_dir = os.path.dirname(root)
                            if os.path.exists(os.path.join(parent_dir, "mix.exs")):
                                if "Phoenix" not in stack["frameworks"]:
                                    stack["frameworks"].append("Phoenix")
                        
                        # Check for Elixir test directory
                        if dir_name == "test" and os.path.exists(os.path.join(os.path.dirname(root), "mix.exs")):
                            if "ExUnit" not in stack["testing_frameworks"]:
                                stack["testing_frameworks"].append("ExUnit")
                                    
                    except (OSError, PermissionError):
                        # Skip directories we can't access
                        continue
                
                return stack
                
            except (OSError, FileNotFoundError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to detect technology stack: {e}")
                return {"error": str(e)}
    
    @async_timed("complexity_metrics_calculation")
    async def _calculate_complexity_metrics(self, project_path: str) -> Dict[str, Any]:
        """Calculate basic complexity metrics for the project with async optimization"""
        async with async_timer("complexity_metrics_calculation"):
            try:
                metrics = {
                    "total_lines": 0,
                    "code_lines": 0,
                    "comment_lines": 0,
                    "blank_lines": 0,
                    "files_analyzed": 0,
                    "average_file_size": 0,
                    "largest_file": {"path": "", "lines": 0},
                    "cyclomatic_complexity_estimate": 0
                }
                
                code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.ex', '.exs', '.erl', '.hrl'}
                
                # Collect all code files for concurrent processing
                code_files = []
                loop = asyncio.get_event_loop()
                
                def collect_code_files():
                    """Collect all code files for processing"""
                    files_found = []
                    for root, dirs, files in os.walk(project_path):
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            if ext in code_extensions:
                                file_path = os.path.join(root, file)
                                files_found.append(file_path)
                    return files_found
                
                # Run file collection in executor
                code_files = await loop.run_in_executor(None, collect_code_files)
                
                # Process files concurrently for complexity analysis
                if code_files:
                    async def analyze_code_file(file_path: str) -> Dict[str, Any]:
                        """Analyze a single code file for complexity metrics"""
                        try:
                            def process_file():
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines = f.readlines()
                                    file_lines = len(lines)
                                    
                                    file_metrics = {
                                        "total_lines": file_lines,
                                        "code_lines": 0,
                                        "comment_lines": 0,
                                        "blank_lines": 0,
                                        "complexity_count": 0,
                                        "file_path": file_path,
                                        "relative_path": os.path.relpath(file_path, project_path)
                                    }
                                    
                                    # Analyze line types (expanded for Elixir/Erlang)
                                    complexity_keywords = ['if', 'for', 'while', 'switch', 'case', 'catch', 'except', 'cond', 'with', 'receive', 'try', 'when']
                                    for line in lines:
                                        line_stripped = line.strip()
                                        if not line_stripped:
                                            file_metrics["blank_lines"] += 1
                                        elif (line_stripped.startswith('#') or line_stripped.startswith('//') or 
                                              line_stripped.startswith('/*') or line_stripped.startswith('%')):
                                            file_metrics["comment_lines"] += 1
                                        else:
                                            file_metrics["code_lines"] += 1
                                            
                                            # Count complexity indicators
                                            line_lower = line_stripped.lower()
                                            for keyword in complexity_keywords:
                                                if keyword in line_lower:
                                                    file_metrics["complexity_count"] += 1
                                    
                                    return file_metrics
                            
                            return await loop.run_in_executor(None, process_file)
                            
                        except (UnicodeDecodeError, PermissionError, OSError) as e:
                            logger.debug(f"Skipping file {file_path}: {e}")
                            return None
                    
                    # Use AsyncBatcher for concurrent file processing
                    batcher = AsyncBatcher(max_concurrency=10, batch_size=25)
                    batch_result = await batcher.execute_batch(
                        code_files,
                        analyze_code_file,
                        progress_callback=lambda done, total: logger.debug(f"Analyzed {done}/{total} code files")
                    )
                    
                    # Aggregate results from all files
                    largest_file_lines = 0
                    largest_file_path = ""
                    
                    for file_metrics in batch_result.results:
                        if file_metrics is not None:
                            metrics["total_lines"] += file_metrics["total_lines"]
                            metrics["code_lines"] += file_metrics["code_lines"]
                            metrics["comment_lines"] += file_metrics["comment_lines"]
                            metrics["blank_lines"] += file_metrics["blank_lines"]
                            metrics["cyclomatic_complexity_estimate"] += file_metrics["complexity_count"]
                            metrics["files_analyzed"] += 1
                            
                            # Track largest file
                            if file_metrics["total_lines"] > largest_file_lines:
                                largest_file_lines = file_metrics["total_lines"]
                                largest_file_path = file_metrics["relative_path"]
                    
                    # Set largest file info
                    metrics["largest_file"] = {
                        "path": largest_file_path,
                        "lines": largest_file_lines
                    }
                
                # Calculate averages
                if metrics["files_analyzed"] > 0:
                    metrics["average_file_size"] = round(metrics["total_lines"] / metrics["files_analyzed"], 1)
                
                return metrics
                
            except (OSError, FileNotFoundError, ValueError, ZeroDivisionError) as e:
                logger.warning(f"Failed to calculate complexity metrics: {e}")
                return {"error": str(e)}