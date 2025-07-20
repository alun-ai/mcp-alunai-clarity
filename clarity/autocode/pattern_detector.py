"""
Pattern detection engine for the AutoCode system.
"""

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class DetectedPattern:
    """Represents a detected code pattern."""
    pattern_type: str
    name: str
    details: Dict[str, Any]
    confidence: float
    file_paths: List[str]
    language: str = "unknown"
    framework: str = "unknown"


class PatternDetector:
    """
    Detects and indexes code project patterns.
    
    This class scans projects to identify:
    - Architectural patterns (MVC, Provider, Component-based, etc.)
    - Naming conventions (file names, functions, variables)
    - Framework usage patterns
    - Component relationships and dependencies
    - Testing patterns
    - Build and deployment patterns
    """
    
    def __init__(self, domain_manager):
        """
        Initialize the pattern detector.
        
        Args:
            domain_manager: The memory domain manager instance
        """
        self.domain_manager = domain_manager
        
        # Supported file extensions and their languages
        self.supported_languages = {
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript", 
            ".jsx": "javascript",
            ".py": "python",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".vue": "vue",
            ".svelte": "svelte"
        }
        
        # Framework detection patterns
        self.framework_indicators = {
            "react": {
                "files": ["package.json"],
                "content_patterns": [r'"react":', r'import.*React', r'from ["\']react["\']'],
                "file_patterns": ["*.jsx", "*.tsx"],
                "directory_patterns": ["components", "pages", "hooks"]
            },
            "vue": {
                "files": ["package.json"],
                "content_patterns": [r'"vue":', r'<template>', r'<script>'],
                "file_patterns": ["*.vue"],
                "directory_patterns": ["components", "views", "store"]
            },
            "angular": {
                "files": ["package.json", "angular.json"],
                "content_patterns": [r'"@angular/', r'@Component', r'@Injectable'],
                "file_patterns": ["*.component.ts", "*.service.ts"],
                "directory_patterns": ["app", "components", "services"]
            },
            "nextjs": {
                "files": ["next.config.js", "package.json"],
                "content_patterns": [r'"next":', r'next/'],
                "directory_patterns": ["pages", "app", "public"]
            },
            "nuxt": {
                "files": ["nuxt.config.js", "package.json"],
                "content_patterns": [r'"nuxt":', r'nuxt/'],
                "directory_patterns": ["pages", "components", "layouts"]
            },
            "svelte": {
                "files": ["package.json"],
                "content_patterns": [r'"svelte":', r'<script>', r'\.svelte'],
                "file_patterns": ["*.svelte"],
                "directory_patterns": ["src", "components"]
            },
            "django": {
                "files": ["manage.py", "settings.py"],
                "content_patterns": [r'django', r'from django', r'DJANGO_SETTINGS_MODULE'],
                "directory_patterns": ["templates", "static", "migrations"]
            },
            "flask": {
                "files": ["app.py"],
                "content_patterns": [r'from flask', r'Flask\(', r'@app\.route'],
                "directory_patterns": ["templates", "static"]
            },
            "fastapi": {
                "files": ["main.py"],
                "content_patterns": [r'from fastapi', r'FastAPI\(', r'@app\.(get|post|put|delete)'],
                "directory_patterns": ["routers", "models", "schemas"]
            },
            "express": {
                "files": ["package.json"],
                "content_patterns": [r'"express":', r'express\(\)', r'app\.use'],
                "directory_patterns": ["routes", "middleware", "controllers"]
            },
            "rails": {
                "files": ["Gemfile", "config/routes.rb"],
                "content_patterns": [r'gem ["\']rails["\']', r'Rails\.application'],
                "directory_patterns": ["app", "config", "db"]
            },
            "spring": {
                "files": ["pom.xml", "build.gradle"],
                "content_patterns": [r'spring-boot', r'@SpringBootApplication', r'@Controller'],
                "directory_patterns": ["src/main/java", "src/main/resources"]
            },
            "rust": {
                "files": ["Cargo.toml"],
                "content_patterns": [r'\[package\]', r'extern crate', r'use std::'],
                "directory_patterns": ["src", "tests"]
            }
        }
        
        # Architectural pattern indicators
        self.architecture_patterns = {
            "mvc": {
                "directories": ["models", "views", "controllers"],
                "files": ["*model*", "*view*", "*controller*"],
                "content_patterns": [r'class.*Controller', r'class.*Model', r'class.*View']
            },
            "mvvm": {
                "directories": ["models", "views", "viewmodels"],
                "files": ["*viewmodel*", "*model*", "*view*"],
                "content_patterns": [r'class.*ViewModel', r'class.*Model']
            },
            "component_based": {
                "directories": ["components", "widgets", "elements"],
                "files": ["*component*", "*widget*", "*.component.*"],
                "content_patterns": [r'@Component', r'React\.Component', r'component.*{']
            },
            "layered": {
                "directories": ["domain", "application", "infrastructure", "presentation"],
                "files": ["*service*", "*repository*", "*dto*"],
                "content_patterns": [r'class.*Service', r'class.*Repository', r'interface.*Repository']
            },
            "clean_architecture": {
                "directories": ["entities", "usecases", "adapters", "frameworks"],
                "files": ["*usecase*", "*entity*", "*adapter*"],
                "content_patterns": [r'class.*UseCase', r'class.*Entity', r'interface.*Gateway']
            },
            "microservices": {
                "directories": ["services", "api", "gateway", "shared"],
                "files": ["*service*", "*gateway*", "docker-compose.yml"],
                "content_patterns": [r'@Service', r'@RestController', r'microservice']
            },
            "monorepo": {
                "files": ["lerna.json", "nx.json", "workspace"],
                "directories": ["packages", "apps", "libs"],
                "content_patterns": [r'"workspaces":', r'lerna', r'nx']
            },
            "ddd": {
                "directories": ["domain", "aggregates", "valueobjects", "repositories"],
                "files": ["*aggregate*", "*valueobject*", "*domain*"],
                "content_patterns": [r'class.*Aggregate', r'class.*ValueObject', r'class.*DomainService']
            }
        }
        
        # Common naming conventions
        self.naming_patterns = {
            "camelCase": r'^[a-z][a-zA-Z0-9]*$',
            "PascalCase": r'^[A-Z][a-zA-Z0-9]*$',
            "snake_case": r'^[a-z][a-z0-9_]*$',
            "kebab-case": r'^[a-z][a-z0-9-]*$',
            "UPPER_SNAKE_CASE": r'^[A-Z][A-Z0-9_]*$'
        }
        
        # Testing framework indicators
        self.testing_frameworks = {
            "jest": {
                "files": ["jest.config.js", "package.json"],
                "content_patterns": [r'"jest":', r'describe\(', r'test\(', r'it\('],
                "file_patterns": ["*.test.js", "*.spec.js", "*.test.ts", "*.spec.ts"]
            },
            "vitest": {
                "files": ["vitest.config.ts", "package.json"],
                "content_patterns": [r'"vitest":', r'describe\(', r'test\(', r'it\('],
                "file_patterns": ["*.test.ts", "*.spec.ts"]
            },
            "pytest": {
                "files": ["pytest.ini", "conftest.py"],
                "content_patterns": [r'import pytest', r'def test_', r'@pytest'],
                "file_patterns": ["test_*.py", "*_test.py"]
            },
            "mocha": {
                "files": ["package.json"],
                "content_patterns": [r'"mocha":', r'describe\(', r'it\('],
                "file_patterns": ["*.test.js", "*.spec.js"]
            },
            "rspec": {
                "files": ["Gemfile", ".rspec"],
                "content_patterns": [r'gem ["\']rspec["\']', r'describe\s+', r'it\s+'],
                "file_patterns": ["*_spec.rb"]
            },
            "cargo_test": {
                "files": ["Cargo.toml"],
                "content_patterns": [r'#\[test\]', r'#\[cfg\(test\)\]'],
                "directory_patterns": ["tests"]
            }
        }
        
        # Build tool indicators
        self.build_tools = {
            "webpack": {
                "files": ["webpack.config.js", "package.json"],
                "content_patterns": [r'"webpack":', r'module\.exports\s*=']
            },
            "vite": {
                "files": ["vite.config.ts", "vite.config.js", "package.json"],
                "content_patterns": [r'"vite":', r'import.*vite']
            },
            "rollup": {
                "files": ["rollup.config.js", "package.json"],
                "content_patterns": [r'"rollup":', r'export default']
            },
            "parcel": {
                "files": ["package.json"],
                "content_patterns": [r'"parcel":']
            },
            "esbuild": {
                "files": ["package.json"],
                "content_patterns": [r'"esbuild":']
            },
            "cargo": {
                "files": ["Cargo.toml"],
                "content_patterns": [r'\[package\]']
            },
            "maven": {
                "files": ["pom.xml"],
                "content_patterns": [r'<project', r'<groupId>']
            },
            "gradle": {
                "files": ["build.gradle", "build.gradle.kts"],
                "content_patterns": [r'apply plugin:', r'dependencies\s*{']
            },
            "make": {
                "files": ["Makefile"],
                "content_patterns": [r'^[a-zA-Z_-]+:']
            }
        }
    
    async def analyze_file_content(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Analyze a single file's content for patterns.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Framework detection patterns
            if file_ext in ['.ts', '.tsx', '.js', '.jsx']:
                # React patterns
                if any(pattern in content for pattern in ['import React', 'from "react"', 'from \'react\'', 'jsx']):
                    patterns.append({
                        "type": "framework_usage",
                        "framework": "react",
                        "file_path": file_path,
                        "confidence": 0.9,
                        "evidence": ["React imports detected"]
                    })
                
                # Vue patterns
                if any(pattern in content for pattern in ['<template>', '<script>', '<style>', 'Vue']):
                    patterns.append({
                        "type": "framework_usage",
                        "framework": "vue",
                        "file_path": file_path,
                        "confidence": 0.9,
                        "evidence": ["Vue template structure detected"]
                    })
                
                # Angular patterns
                if any(pattern in content for pattern in ['@Component', '@Injectable', '@NgModule']):
                    patterns.append({
                        "type": "framework_usage",
                        "framework": "angular",
                        "file_path": file_path,
                        "confidence": 0.9,
                        "evidence": ["Angular decorators detected"]
                    })
                
                # Component patterns
                if any(pattern in file_path.lower() for pattern in ['component', 'comp']) or 'Component' in content:
                    patterns.append({
                        "type": "component_pattern",
                        "language": "typescript" if file_ext in ['.ts', '.tsx'] else "javascript",
                        "file_path": file_path,
                        "confidence": 0.8,
                        "evidence": ["Component naming or structure detected"]
                    })
            
            elif file_ext == '.py':
                # Django patterns
                if any(pattern in content for pattern in ['from django', 'import django', 'models.Model', 'views.View']):
                    patterns.append({
                        "type": "framework_usage",
                        "framework": "django",
                        "file_path": file_path,
                        "confidence": 0.9,
                        "evidence": ["Django imports/patterns detected"]
                    })
                
                # Flask patterns
                if any(pattern in content for pattern in ['from flask', 'Flask(__name__)', '@app.route']):
                    patterns.append({
                        "type": "framework_usage",
                        "framework": "flask",
                        "file_path": file_path,
                        "confidence": 0.9,
                        "evidence": ["Flask patterns detected"]
                    })
                
                # FastAPI patterns
                if any(pattern in content for pattern in ['from fastapi', 'FastAPI()', '@app.get', '@app.post']):
                    patterns.append({
                        "type": "framework_usage",
                        "framework": "fastapi",
                        "file_path": file_path,
                        "confidence": 0.9,
                        "evidence": ["FastAPI patterns detected"]
                    })
            
            elif file_ext == '.rs':
                # Rust patterns
                if any(pattern in content for pattern in ['use actix_web', 'use rocket', 'use warp']):
                    framework = "actix-web" if "actix_web" in content else "rocket" if "rocket" in content else "warp"
                    patterns.append({
                        "type": "framework_usage",
                        "framework": framework,
                        "file_path": file_path,
                        "confidence": 0.9,
                        "evidence": [f"{framework} patterns detected"]
                    })
            
            # Architectural patterns
            if any(pattern in content.lower() for pattern in ['controller', 'service', 'repository', 'model']):
                patterns.append({
                    "type": "architectural_pattern",
                    "pattern": "mvc_like",
                    "file_path": file_path,
                    "confidence": 0.7,
                    "evidence": ["MVC-like naming conventions detected"]
                })
            
            # Testing patterns
            if any(pattern in file_path.lower() for pattern in ['test', 'spec']) or any(pattern in content for pattern in ['describe(', 'it(', 'test(', 'def test_']):
                patterns.append({
                    "type": "testing_pattern",
                    "file_path": file_path,
                    "confidence": 0.9,
                    "evidence": ["Test patterns detected"]
                })
            
        except Exception as e:
            logger.error(f"Error analyzing file content {file_path}: {e}")
        
        return patterns
    
    async def scan_project(self, project_root: str) -> Dict[str, Any]:
        """
        Perform comprehensive project scan for patterns.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary containing all detected patterns
        """
        try:
            import os
            project_path = Path(project_root)
            logger.debug(f"Pattern detector: checking path existence for {project_root}")
            
            # Use multiple validation methods to ensure robust path checking
            path_exists_pathlib = project_path.exists()
            path_exists_os = os.path.exists(project_root)
            path_is_dir_pathlib = project_path.is_dir()
            path_is_dir_os = os.path.isdir(project_root)
            
            logger.debug(f"Pattern detector: pathlib.Path.exists(): {path_exists_pathlib}")
            logger.debug(f"Pattern detector: os.path.exists(): {path_exists_os}")
            logger.debug(f"Pattern detector: pathlib.Path.is_dir(): {path_is_dir_pathlib}")
            logger.debug(f"Pattern detector: os.path.isdir(): {path_is_dir_os}")
            
            # Accept path if ANY of the validation methods confirm it exists and is a directory
            path_valid = (path_exists_pathlib or path_exists_os) and (path_is_dir_pathlib or path_is_dir_os)
            
            if not path_valid:
                logger.error(f"Pattern detector: Path validation failed for {project_root}")
                logger.error(f"Pattern detector: All validation methods failed - pathlib.exists={path_exists_pathlib}, os.exists={path_exists_os}, pathlib.is_dir={path_is_dir_pathlib}, os.is_dir={path_is_dir_os}")
                raise ValueError(f"Project path {project_root} does not exist or is not a directory")
            
            logger.info(f"Starting comprehensive project scan: {project_root}")
            
            # Initialize scan results
            scan_results = {
                "project_path": str(project_path),
                "scan_timestamp": datetime.utcnow().isoformat(),
                "framework": await self.detect_framework(project_root),
                "language": await self.detect_primary_language(project_root),
                "architecture": await self.detect_architecture_patterns(project_root),
                "naming_conventions": await self.extract_naming_conventions(project_root),
                "component_relationships": await self.map_component_relationships(project_root),
                "testing_patterns": await self.detect_testing_patterns(project_root),
                "build_patterns": await self.detect_build_patterns(project_root),
                "file_structure": await self.analyze_file_structure(project_root),
                "dependencies": await self.analyze_dependencies(project_root),
                "configuration": await self.analyze_configuration(project_root)
            }
            
            # Store patterns in memory
            await self._store_scan_results(project_root, scan_results)
            
            logger.info(f"Project scan completed: {len(scan_results)} pattern categories detected")
            return scan_results
            
        except Exception as e:
            logger.error(f"Error scanning project {project_root}: {e}")
            return {"error": str(e), "project_path": project_root}
    
    async def detect_framework(self, project_root: str) -> Dict[str, Any]:
        """
        Detect the primary framework(s) used in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary with detected frameworks and confidence scores
        """
        try:
            project_path = Path(project_root)
            detected_frameworks = {}
            
            for framework, indicators in self.framework_indicators.items():
                confidence = await self._calculate_framework_confidence(
                    project_path, framework, indicators
                )
                
                if confidence > 0.3:  # Minimum confidence threshold
                    detected_frameworks[framework] = {
                        "confidence": confidence,
                        "evidence": await self._get_framework_evidence(
                            project_path, framework, indicators
                        )
                    }
            
            # Sort by confidence
            sorted_frameworks = dict(
                sorted(detected_frameworks.items(), 
                      key=lambda x: x[1]["confidence"], 
                      reverse=True)
            )
            
            return {
                "primary": list(sorted_frameworks.keys())[0] if sorted_frameworks else "unknown",
                "all_detected": sorted_frameworks,
                "confidence_threshold": 0.3
            }
            
        except Exception as e:
            logger.error(f"Error detecting framework for {project_root}: {e}")
            return {"primary": "unknown", "error": str(e)}
    
    async def detect_primary_language(self, project_root: str) -> Dict[str, Any]:
        """
        Detect the primary programming language(s) of the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary with language statistics
        """
        try:
            project_path = Path(project_root)
            language_counts = {}
            total_files = 0
            
            for file_path in project_path.rglob("*"):
                if file_path.is_file() and not self._should_ignore_path(file_path):
                    suffix = file_path.suffix.lower()
                    if suffix in self.supported_languages:
                        lang = self.supported_languages[suffix]
                        language_counts[lang] = language_counts.get(lang, 0) + 1
                        total_files += 1
            
            if not language_counts:
                return {"primary": "unknown", "distribution": {}, "total_files": 0}
            
            # Calculate percentages
            language_distribution = {
                lang: {
                    "count": count,
                    "percentage": (count / total_files) * 100
                }
                for lang, count in language_counts.items()
            }
            
            primary_language = max(language_counts, key=language_counts.get)
            
            return {
                "primary": primary_language,
                "distribution": language_distribution,
                "total_files": total_files,
                "languages_detected": len(language_counts)
            }
            
        except Exception as e:
            logger.error(f"Error detecting language for {project_root}: {e}")
            return {"primary": "unknown", "error": str(e)}
    
    async def detect_architecture_patterns(self, project_root: str) -> Dict[str, Any]:
        """
        Detect architectural patterns in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of detected architectural patterns
        """
        try:
            project_path = Path(project_root)
            detected_patterns = {}
            
            for pattern_name, indicators in self.architecture_patterns.items():
                confidence = await self._calculate_architecture_confidence(
                    project_path, pattern_name, indicators
                )
                
                if confidence > 0.4:  # Architecture patterns need higher confidence
                    detected_patterns[pattern_name] = {
                        "confidence": confidence,
                        "evidence": await self._get_architecture_evidence(
                            project_path, pattern_name, indicators
                        )
                    }
            
            return {
                "patterns": detected_patterns,
                "primary_pattern": max(detected_patterns, key=lambda x: detected_patterns[x]["confidence"]) if detected_patterns else "unknown",
                "total_patterns": len(detected_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error detecting architecture patterns for {project_root}: {e}")
            return {"patterns": {}, "error": str(e)}
    
    async def extract_naming_conventions(self, project_root: str) -> Dict[str, Any]:
        """
        Extract naming conventions from the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of naming conventions with examples
        """
        try:
            project_path = Path(project_root)
            conventions = {
                "files": {"patterns": {}, "examples": []},
                "directories": {"patterns": {}, "examples": []},
                "functions": {"patterns": {}, "examples": []},
                "classes": {"patterns": {}, "examples": []},
                "variables": {"patterns": {}, "examples": []}
            }
            
            # Collect file and directory names
            files = []
            directories = []
            
            for path in project_path.rglob("*"):
                if self._should_ignore_path(path):
                    continue
                    
                if path.is_file():
                    files.append(path.name)
                elif path.is_dir():
                    directories.append(path.name)
            
            # Analyze naming patterns
            conventions["files"] = self._analyze_naming_patterns(files, "file")
            conventions["directories"] = self._analyze_naming_patterns(directories, "directory")
            
            # Analyze code naming conventions
            code_conventions = await self._extract_code_naming_conventions(project_path)
            conventions.update(code_conventions)
            
            return conventions
            
        except Exception as e:
            logger.error(f"Error extracting naming conventions for {project_root}: {e}")
            return {"error": str(e)}
    
    async def map_component_relationships(self, project_root: str) -> Dict[str, Any]:
        """
        Map relationships between components in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary mapping component relationships
        """
        try:
            project_path = Path(project_root)
            relationships = {
                "imports": {},
                "exports": {},
                "dependencies": {},
                "component_tree": {},
                "circular_dependencies": []
            }
            
            # Analyze import/export relationships for supported languages
            for file_path in project_path.rglob("*"):
                if (file_path.is_file() and 
                    file_path.suffix in self.supported_languages and
                    not self._should_ignore_path(file_path)):
                    
                    try:
                        file_relationships = await self._analyze_file_relationships(file_path, project_path)
                        relative_path = str(file_path.relative_to(project_path))
                        
                        relationships["imports"][relative_path] = file_relationships.get("imports", [])
                        relationships["exports"][relative_path] = file_relationships.get("exports", [])
                        
                    except Exception as e:
                        logger.debug(f"Error analyzing relationships for {file_path}: {e}")
                        continue
            
            # Build dependency graph
            relationships["dependencies"] = self._build_dependency_graph(relationships["imports"])
            
            # Detect circular dependencies
            relationships["circular_dependencies"] = self._detect_circular_dependencies(
                relationships["dependencies"]
            )
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error mapping component relationships for {project_root}: {e}")
            return {"error": str(e)}
    
    async def detect_testing_patterns(self, project_root: str) -> Dict[str, Any]:
        """
        Detect testing patterns and conventions.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of testing patterns
        """
        try:
            project_path = Path(project_root)
            patterns = {
                "frameworks": {},
                "structure": {},
                "naming_conventions": {},
                "coverage_tools": []
            }
            
            # Detect test frameworks
            for framework, indicators in self.testing_frameworks.items():
                confidence = await self._calculate_framework_confidence(
                    project_path, framework, indicators
                )
                
                if confidence > 0.3:
                    patterns["frameworks"][framework] = {
                        "confidence": confidence,
                        "evidence": await self._get_framework_evidence(
                            project_path, framework, indicators
                        )
                    }
            
            # Analyze test structure
            test_files = self._find_test_files(project_path)
            test_dirs = [d for d in project_path.rglob("*") 
                        if d.is_dir() and any(test_word in d.name.lower() 
                                            for test_word in ["test", "spec", "__tests__"])]
            
            patterns["structure"] = {
                "test_files": [str(f.relative_to(project_path)) for f in test_files],
                "test_directories": [str(d.relative_to(project_path)) for d in test_dirs],
                "total_test_files": len(test_files)
            }
            
            # Analyze naming conventions
            patterns["naming_conventions"] = self._analyze_test_naming(test_files)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting testing patterns for {project_root}: {e}")
            return {"error": str(e)}
    
    async def detect_build_patterns(self, project_root: str) -> Dict[str, Any]:
        """
        Detect build and deployment patterns.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of build patterns
        """
        try:
            project_path = Path(project_root)
            patterns = {
                "build_tools": {},
                "package_managers": {},
                "ci_cd": {},
                "containerization": {},
                "deployment": {}
            }
            
            # Detect build tools
            for tool, indicators in self.build_tools.items():
                confidence = await self._calculate_framework_confidence(
                    project_path, tool, indicators
                )
                
                if confidence > 0.5:
                    patterns["build_tools"][tool] = {
                        "confidence": confidence,
                        "evidence": await self._get_framework_evidence(
                            project_path, tool, indicators
                        )
                    }
            
            # Detect package managers
            patterns["package_managers"] = await self._detect_package_managers(project_path)
            
            # Detect CI/CD
            patterns["ci_cd"] = await self._detect_ci_cd(project_path)
            
            # Detect containerization
            patterns["containerization"] = await self._detect_containerization(project_path)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting build patterns for {project_root}: {e}")
            return {"error": str(e)}
    
    async def analyze_file_structure(self, project_root: str) -> Dict[str, Any]:
        """
        Analyze overall file structure and organization.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary describing file structure
        """
        try:
            project_path = Path(project_root)
            structure = {
                "total_files": 0,
                "total_directories": 0,
                "file_types": {},
                "directory_depth": 0,
                "largest_directories": {},
                "file_size_distribution": {},
                "organization_patterns": {}
            }
            
            file_counts = {}
            dir_sizes = {}
            max_depth = 0
            total_size = 0
            
            for path in project_path.rglob("*"):
                if self._should_ignore_path(path):
                    continue
                
                # Calculate depth
                depth = len(path.relative_to(project_path).parts)
                max_depth = max(max_depth, depth)
                
                if path.is_file():
                    structure["total_files"] += 1
                    
                    # Count file types
                    suffix = path.suffix.lower()
                    file_counts[suffix] = file_counts.get(suffix, 0) + 1
                    
                    # Track directory sizes
                    parent = str(path.parent.relative_to(project_path))
                    dir_sizes[parent] = dir_sizes.get(parent, 0) + 1
                    
                    # File size
                    try:
                        size = path.stat().st_size
                        total_size += size
                    except:
                        pass
                
                elif path.is_dir():
                    structure["total_directories"] += 1
            
            structure["file_types"] = dict(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:15])
            structure["directory_depth"] = max_depth
            structure["largest_directories"] = dict(sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)[:10])
            structure["total_size_bytes"] = total_size
            
            # Analyze organization patterns
            structure["organization_patterns"] = self._analyze_organization_patterns(project_path)
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing file structure for {project_root}: {e}")
            return {"error": str(e)}
    
    async def analyze_dependencies(self, project_root: str) -> Dict[str, Any]:
        """
        Analyze project dependencies.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of dependency information
        """
        try:
            project_path = Path(project_root)
            dependencies = {
                "package_files": [],
                "dependencies": {},
                "dev_dependencies": {},
                "total_dependencies": 0,
                "dependency_categories": {}
            }
            
            # Check for dependency files
            dependency_files = [
                "package.json", "Cargo.toml", "requirements.txt", 
                "pyproject.toml", "pom.xml", "build.gradle", "Gemfile"
            ]
            
            for dep_file in dependency_files:
                file_path = project_path / dep_file
                if file_path.exists():
                    dependencies["package_files"].append(dep_file)
                    deps = await self._parse_dependency_file(file_path)
                    dependencies.update(deps)
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies for {project_root}: {e}")
            return {"error": str(e)}
    
    async def analyze_configuration(self, project_root: str) -> Dict[str, Any]:
        """
        Analyze project configuration files.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary of configuration information
        """
        try:
            project_path = Path(project_root)
            config = {
                "config_files": [],
                "environment_files": [],
                "build_configs": [],
                "linting_configs": [],
                "type_configs": []
            }
            
            # Define config file patterns
            config_patterns = {
                "config_files": [
                    "*.config.js", "*.config.ts", "*.config.json",
                    "config.py", "settings.py", "application.properties"
                ],
                "environment_files": [
                    ".env*", "*.env", "environment.yml", "docker-compose.yml"
                ],
                "build_configs": [
                    "webpack.config.*", "vite.config.*", "rollup.config.*",
                    "Makefile", "CMakeLists.txt", "build.gradle"
                ],
                "linting_configs": [
                    ".eslintrc.*", ".prettierrc.*", ".pylintrc", 
                    "tslint.json", ".flake8", "rustfmt.toml"
                ],
                "type_configs": [
                    "tsconfig.json", "mypy.ini", "pyproject.toml"
                ]
            }
            
            for category, patterns in config_patterns.items():
                for pattern in patterns:
                    matching_files = list(project_path.rglob(pattern))
                    config[category].extend([
                        str(f.relative_to(project_path)) for f in matching_files
                        if not self._should_ignore_path(f)
                    ])
            
            return config
            
        except Exception as e:
            logger.error(f"Error analyzing configuration for {project_root}: {e}")
            return {"error": str(e)}
    
    # Private implementation methods
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored during analysis."""
        ignore_patterns = {
            "node_modules", ".git", "__pycache__", ".pytest_cache",
            "target", "build", "dist", ".next", ".nuxt", "coverage",
            ".vscode", ".idea", "venv", "env", ".env", "vendor"
        }
        
        return any(pattern in path.parts for pattern in ignore_patterns)
    
    async def _calculate_framework_confidence(
        self, 
        project_path: Path, 
        framework: str, 
        indicators: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for framework detection."""
        try:
            confidence = 0.0
            total_checks = 0
            
            # Check for required files
            files = indicators.get("files", [])
            for file_name in files:
                total_checks += 1
                if (project_path / file_name).exists():
                    confidence += 0.3
            
            # Check content patterns
            content_patterns = indicators.get("content_patterns", [])
            if content_patterns and files:
                for file_name in files:
                    file_path = project_path / file_name
                    if file_path.exists():
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                for pattern in content_patterns:
                                    total_checks += 1
                                    if re.search(pattern, content):
                                        confidence += 0.2
                        except:
                            pass
            
            # Check file patterns
            file_patterns = indicators.get("file_patterns", [])
            for pattern in file_patterns:
                total_checks += 1
                if list(project_path.rglob(pattern)):
                    confidence += 0.15
            
            # Check directory patterns
            dir_patterns = indicators.get("directory_patterns", [])
            for pattern in dir_patterns:
                total_checks += 1
                if any(project_path.rglob(pattern)):
                    confidence += 0.1
            
            # Normalize confidence
            if total_checks > 0:
                confidence = min(1.0, confidence)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating framework confidence for {framework}: {e}")
            return 0.0
    
    async def _get_framework_evidence(
        self, 
        project_path: Path, 
        framework: str, 
        indicators: Dict[str, Any]
    ) -> List[str]:
        """Get evidence for framework detection."""
        evidence = []
        
        try:
            # File evidence
            files = indicators.get("files", [])
            for file_name in files:
                if (project_path / file_name).exists():
                    evidence.append(f"Found {file_name}")
            
            # Directory evidence
            dir_patterns = indicators.get("directory_patterns", [])
            for pattern in dir_patterns:
                if any(project_path.rglob(pattern)):
                    evidence.append(f"Found {pattern} directory")
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error getting framework evidence for {framework}: {e}")
            return []
    
    async def _calculate_architecture_confidence(
        self, 
        project_path: Path, 
        pattern_name: str, 
        indicators: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for architectural pattern detection."""
        try:
            confidence = 0.0
            total_weight = 0.0
            
            # Check directories (highest weight)
            directories = indicators.get("directories", [])
            if directories:
                directory_weight = 0.5
                total_weight += directory_weight
                matches = sum(1 for d in directories if any(project_path.rglob(d)))
                confidence += (matches / len(directories)) * directory_weight
            
            # Check file patterns (medium weight)
            files = indicators.get("files", [])
            if files:
                file_weight = 0.3
                total_weight += file_weight
                matches = sum(1 for f in files if list(project_path.rglob(f)))
                confidence += (matches / len(files)) * file_weight
            
            # Check content patterns (lower weight)
            content_patterns = indicators.get("content_patterns", [])
            if content_patterns:
                content_weight = 0.2
                total_weight += content_weight
                pattern_matches = 0
                
                for file_path in project_path.rglob("*"):
                    if (file_path.is_file() and 
                        file_path.suffix in self.supported_languages and
                        not self._should_ignore_path(file_path)):
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                for pattern in content_patterns:
                                    if re.search(pattern, content):
                                        pattern_matches += 1
                                        break  # Only count one match per file
                        except:
                            continue
                
                if pattern_matches > 0:
                    confidence += content_weight
            
            return confidence if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating architecture confidence for {pattern_name}: {e}")
            return 0.0
    
    async def _get_architecture_evidence(
        self, 
        project_path: Path, 
        pattern_name: str, 
        indicators: Dict[str, Any]
    ) -> List[str]:
        """Get evidence for architectural pattern detection."""
        evidence = []
        
        try:
            # Directory evidence
            directories = indicators.get("directories", [])
            for directory in directories:
                if any(project_path.rglob(directory)):
                    evidence.append(f"Found {directory} directory")
            
            # File pattern evidence
            files = indicators.get("files", [])
            for file_pattern in files:
                if list(project_path.rglob(file_pattern)):
                    evidence.append(f"Found files matching {file_pattern}")
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error getting architecture evidence for {pattern_name}: {e}")
            return []
    
    def _analyze_naming_patterns(self, names: List[str], name_type: str) -> Dict[str, Any]:
        """Analyze naming patterns in a list of names."""
        try:
            if not names:
                return {"patterns": {}, "examples": []}
            
            pattern_counts = {}
            examples = {}
            
            for name in names:
                # Skip hidden files/directories for naming analysis
                if name.startswith('.'):
                    continue
                    
                # Remove file extensions for analysis
                if name_type == "file" and '.' in name:
                    name_for_analysis = name.rsplit('.', 1)[0]
                else:
                    name_for_analysis = name
                
                # Check against naming patterns
                for pattern_name, pattern_regex in self.naming_patterns.items():
                    if re.match(pattern_regex, name_for_analysis):
                        pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                        if pattern_name not in examples:
                            examples[pattern_name] = []
                        if len(examples[pattern_name]) < 5:  # Limit examples
                            examples[pattern_name].append(name)
            
            # Find dominant pattern
            dominant_pattern = max(pattern_counts, key=pattern_counts.get) if pattern_counts else "mixed"
            
            return {
                "patterns": pattern_counts,
                "dominant": dominant_pattern,
                "examples": examples,
                "total_analyzed": len([n for n in names if not n.startswith('.')])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing naming patterns: {e}")
            return {"patterns": {}, "examples": []}
    
    async def _extract_code_naming_conventions(self, project_path: Path) -> Dict[str, Any]:
        """Extract naming conventions from code files."""
        try:
            conventions = {
                "functions": {"patterns": {}, "examples": []},
                "classes": {"patterns": {}, "examples": []},
                "variables": {"patterns": {}, "examples": []}
            }
            
            function_names = []
            class_names = []
            variable_names = []
            
            # Analyze code files
            for file_path in project_path.rglob("*"):
                if (file_path.is_file() and 
                    file_path.suffix in ['.py', '.js', '.ts', '.java', '.rb'] and
                    not self._should_ignore_path(file_path)):
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Extract names based on language
                        lang = self.supported_languages.get(file_path.suffix, "unknown")
                        names = self._extract_names_from_code(content, lang)
                        
                        function_names.extend(names.get("functions", []))
                        class_names.extend(names.get("classes", []))
                        variable_names.extend(names.get("variables", []))
                        
                    except Exception as e:
                        logger.debug(f"Error analyzing {file_path}: {e}")
                        continue
            
            # Analyze patterns
            conventions["functions"] = self._analyze_naming_patterns(function_names, "function")
            conventions["classes"] = self._analyze_naming_patterns(class_names, "class")
            conventions["variables"] = self._analyze_naming_patterns(variable_names, "variable")
            
            return conventions
            
        except Exception as e:
            logger.error(f"Error extracting code naming conventions: {e}")
            return {}
    
    def _extract_names_from_code(self, content: str, language: str) -> Dict[str, List[str]]:
        """Extract function, class, and variable names from code."""
        names = {"functions": [], "classes": [], "variables": []}
        
        try:
            if language == "python":
                # Python patterns
                functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                variables = re.findall(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=', content, re.MULTILINE)
                
            elif language in ["javascript", "typescript"]:
                # JavaScript/TypeScript patterns
                functions = re.findall(r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
                functions.extend(re.findall(r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\(', content))
                classes = re.findall(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
                variables = re.findall(r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
                
            elif language == "java":
                # Java patterns
                functions = re.findall(r'(?:public|private|protected)?\s*(?:static\s+)?[a-zA-Z<>\[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
                classes = re.findall(r'(?:public\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                
            elif language == "ruby":
                # Ruby patterns
                functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            
            names["functions"] = functions[:50]  # Limit to prevent memory issues
            names["classes"] = classes[:50]
            names["variables"] = variables[:50]
            
        except Exception as e:
            logger.debug(f"Error extracting names from {language} code: {e}")
        
        return names
    
    async def _store_scan_results(self, project_root: str, scan_results: Dict[str, Any]) -> None:
        """Store scan results in memory."""
        try:
            # Store overall project pattern
            await self.domain_manager.store_project_pattern(
                pattern_type="comprehensive_scan",
                framework=scan_results.get("framework", {}).get("primary", "unknown"),
                language=scan_results.get("language", {}).get("primary", "unknown"),
                structure=scan_results,
                importance=0.8,
                metadata={
                    "project_root": project_root,
                    "scan_timestamp": scan_results.get("scan_timestamp"),
                    "pattern_count": len([k for k, v in scan_results.items() 
                                        if isinstance(v, dict) and v])
                }
            )
            
            logger.info(f"Stored comprehensive scan results for {project_root}")
            
        except Exception as e:
            logger.error(f"Error storing scan results: {e}")
    
    # Additional helper methods for specific analyses...
    async def _analyze_file_relationships(self, file_path: Path, project_root: Path) -> Dict[str, Any]:
        """Analyze import/export relationships for a file."""
        relationships = {"imports": [], "exports": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = self.supported_languages.get(file_path.suffix, "unknown")
            
            if language in ["javascript", "typescript"]:
                # JavaScript/TypeScript imports
                imports = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content)
                exports = re.findall(r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
                
            elif language == "python":
                # Python imports
                imports = re.findall(r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import', content)
                imports.extend(re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content))
                
            relationships["imports"] = imports
            relationships["exports"] = exports
            
        except Exception as e:
            logger.debug(f"Error analyzing relationships for {file_path}: {e}")
        
        return relationships
    
    def _build_dependency_graph(self, imports: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Build dependency graph from import relationships."""
        return {file: deps for file, deps in imports.items() if deps}
    
    def _detect_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in the dependency graph."""
        circular_deps = []
        
        def has_path(start: str, end: str, visited: Set[str]) -> bool:
            if start == end:
                return True
            if start in visited:
                return False
            
            visited.add(start)
            for dep in dependencies.get(start, []):
                if has_path(dep, end, visited):
                    return True
            return False
        
        for file, deps in dependencies.items():
            for dep in deps:
                if has_path(dep, file, set()):
                    circular_deps.append([file, dep])
        
        return circular_deps
    
    def _find_test_files(self, project_path: Path) -> List[Path]:
        """Find test files in the project."""
        test_files = []
        
        test_patterns = [
            "test_*.py", "*_test.py", "test*.py",
            "*.test.js", "*.spec.js", "*.test.ts", "*.spec.ts",
            "*_spec.rb", "*_test.rb",
            "*Test.java", "*Tests.java"
        ]
        
        for pattern in test_patterns:
            test_files.extend(project_path.rglob(pattern))
        
        return [f for f in test_files if not self._should_ignore_path(f)]
    
    def _analyze_test_naming(self, test_files: List[Path]) -> Dict[str, Any]:
        """Analyze test file naming conventions."""
        if not test_files:
            return {}
        
        naming_patterns = {
            "prefix_test": 0,  # test_something.py
            "suffix_test": 0,  # something_test.py
            "dot_test": 0,     # something.test.js
            "dot_spec": 0      # something.spec.js
        }
        
        for test_file in test_files:
            name = test_file.name.lower()
            if name.startswith("test"):
                naming_patterns["prefix_test"] += 1
            elif ".test." in name:
                naming_patterns["dot_test"] += 1
            elif ".spec." in name:
                naming_patterns["dot_spec"] += 1
            elif name.endswith("_test.py") or name.endswith("test.py"):
                naming_patterns["suffix_test"] += 1
        
        dominant = max(naming_patterns, key=naming_patterns.get) if any(naming_patterns.values()) else "mixed"
        
        return {
            "patterns": naming_patterns,
            "dominant": dominant,
            "total_files": len(test_files)
        }
    
    async def _detect_package_managers(self, project_path: Path) -> Dict[str, Any]:
        """Detect package managers used in the project."""
        managers = {}
        
        manager_indicators = {
            "npm": ["package.json", "package-lock.json"],
            "yarn": ["yarn.lock"],
            "pnpm": ["pnpm-lock.yaml"],
            "pip": ["requirements.txt"],
            "poetry": ["pyproject.toml", "poetry.lock"],
            "cargo": ["Cargo.toml", "Cargo.lock"],
            "maven": ["pom.xml"],
            "gradle": ["build.gradle", "build.gradle.kts"],
            "composer": ["composer.json", "composer.lock"],
            "bundler": ["Gemfile", "Gemfile.lock"]
        }
        
        for manager, files in manager_indicators.items():
            evidence = [f for f in files if (project_path / f).exists()]
            if evidence:
                managers[manager] = {
                    "confidence": len(evidence) / len(files),
                    "evidence": evidence
                }
        
        return managers
    
    async def _detect_ci_cd(self, project_path: Path) -> Dict[str, Any]:
        """Detect CI/CD configurations."""
        ci_cd = {}
        
        ci_indicators = {
            "github_actions": [".github/workflows"],
            "gitlab_ci": [".gitlab-ci.yml"],
            "travis": [".travis.yml"],
            "circle_ci": [".circleci"],
            "jenkins": ["Jenkinsfile"],
            "azure_pipelines": ["azure-pipelines.yml", ".azure"]
        }
        
        for ci_system, indicators in ci_indicators.items():
            evidence = []
            for indicator in indicators:
                path = project_path / indicator
                if path.exists():
                    evidence.append(indicator)
            
            if evidence:
                ci_cd[ci_system] = {
                    "confidence": 1.0,
                    "evidence": evidence
                }
        
        return ci_cd
    
    async def _detect_containerization(self, project_path: Path) -> Dict[str, Any]:
        """Detect containerization configurations."""
        containers = {}
        
        if (project_path / "Dockerfile").exists():
            containers["docker"] = {
                "confidence": 1.0,
                "evidence": ["Dockerfile"]
            }
        
        if (project_path / "docker-compose.yml").exists():
            containers["docker_compose"] = {
                "confidence": 1.0,
                "evidence": ["docker-compose.yml"]
            }
        
        k8s_files = list(project_path.rglob("*.yaml")) + list(project_path.rglob("*.yml"))
        k8s_evidence = []
        for file in k8s_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(keyword in content for keyword in ["apiVersion:", "kind:", "kubernetes"]):
                        k8s_evidence.append(str(file.relative_to(project_path)))
            except:
                pass
        
        if k8s_evidence:
            containers["kubernetes"] = {
                "confidence": 0.8,
                "evidence": k8s_evidence[:5]  # Limit evidence
            }
        
        return containers
    
    async def _parse_dependency_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse dependency information from package files."""
        deps_info = {"dependencies": {}, "dev_dependencies": {}}
        
        try:
            if file_path.name == "package.json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    deps_info["dependencies"] = data.get("dependencies", {})
                    deps_info["dev_dependencies"] = data.get("devDependencies", {})
            
            elif file_path.name == "requirements.txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            dep_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                            deps_info["dependencies"][dep_name] = line
            
            # Add more parsers for other dependency files as needed
            
        except Exception as e:
            logger.debug(f"Error parsing dependency file {file_path}: {e}")
        
        return deps_info
    
    def _analyze_organization_patterns(self, project_path: Path) -> Dict[str, Any]:
        """Analyze how the project is organized."""
        patterns = {
            "flat_structure": False,
            "feature_based": False,
            "layer_based": False,
            "domain_driven": False
        }
        
        try:
            # Get top-level directories
            top_dirs = [d.name for d in project_path.iterdir() 
                       if d.is_dir() and not self._should_ignore_path(d)]
            
            # Check for flat structure (few top-level directories)
            if len(top_dirs) <= 3:
                patterns["flat_structure"] = True
            
            # Check for layer-based organization
            layer_keywords = ["models", "views", "controllers", "services", "repositories", "dto"]
            if any(keyword in top_dirs for keyword in layer_keywords):
                patterns["layer_based"] = True
            
            # Check for feature-based organization
            if len(top_dirs) > 5 and not patterns["layer_based"]:
                patterns["feature_based"] = True
            
            # Check for domain-driven design
            ddd_keywords = ["domain", "aggregates", "entities", "valueobjects"]
            if any(keyword in top_dirs for keyword in ddd_keywords):
                patterns["domain_driven"] = True
            
        except Exception as e:
            logger.debug(f"Error analyzing organization patterns: {e}")
        
        return patterns