"""
Lazy import utilities for reducing startup time.

This module provides utilities for deferring heavy imports until they are actually needed,
significantly reducing application startup time.
"""

import sys
import threading
from typing import Any, Dict, Optional, Tuple, Type, Union
from functools import wraps
import importlib

from loguru import logger


class LazyImport:
    """Lazy import container that only imports when first accessed."""
    
    def __init__(self, module_name: str, attribute: Optional[str] = None):
        self.module_name = module_name
        self.attribute = attribute
        self._cached_import = None
        self._import_lock = threading.Lock()
        self._import_attempted = False
    
    def __call__(self) -> Any:
        """Get the lazily imported module or attribute."""
        if self._cached_import is None and not self._import_attempted:
            with self._import_lock:
                if self._cached_import is None and not self._import_attempted:
                    self._import_attempted = True
                    try:
                        module = importlib.import_module(self.module_name)
                        if self.attribute:
                            self._cached_import = getattr(module, self.attribute)
                        else:
                            self._cached_import = module
                        
                        logger.debug(f"Lazy imported: {self.module_name}.{self.attribute or ''}")
                    except ImportError as e:
                        logger.warning(f"Failed to lazy import {self.module_name}: {e}")
                        self._cached_import = None
        
        return self._cached_import
    
    @property
    def available(self) -> bool:
        """Check if the import is available without actually importing."""
        if self._import_attempted:
            return self._cached_import is not None
        
        try:
            importlib.util.find_spec(self.module_name)
            return True
        except ImportError:
            return False


class MLDependencies:
    """Lazy loading container for machine learning dependencies."""
    
    def __init__(self):
        self._numpy = LazyImport("numpy")
        self._sentence_transformers = LazyImport("sentence_transformers", "SentenceTransformer")
        self._torch = LazyImport("torch")
        
    @property
    def numpy(self):
        """Get numpy module (lazy loaded)."""
        return self._numpy()
    
    @property
    def SentenceTransformer(self):
        """Get SentenceTransformer class (lazy loaded)."""
        return self._sentence_transformers()
    
    @property
    def torch(self):
        """Get torch module (lazy loaded)."""
        return self._torch()
    
    @property
    def available(self) -> Dict[str, bool]:
        """Check availability of ML dependencies."""
        return {
            "numpy": self._numpy.available,
            "sentence_transformers": self._sentence_transformers.available,
            "torch": self._torch.available
        }


class DatabaseDependencies:
    """Lazy loading container for database dependencies."""
    
    def __init__(self):
        self._qdrant_client = LazyImport("qdrant_client", "QdrantClient")
        self._qdrant_models_distance = LazyImport("qdrant_client.models", "Distance")
        self._qdrant_models_vector = LazyImport("qdrant_client.models", "VectorParams")
        self._qdrant_models_point = LazyImport("qdrant_client.models", "PointStruct")
        
    @property
    def QdrantClient(self):
        """Get QdrantClient class (lazy loaded)."""
        return self._qdrant_client()
    
    @property
    def Distance(self):
        """Get Distance enum (lazy loaded)."""
        return self._qdrant_models_distance()
    
    @property
    def VectorParams(self):
        """Get VectorParams class (lazy loaded)."""
        return self._qdrant_models_vector()
    
    @property
    def PointStruct(self):
        """Get PointStruct class (lazy loaded)."""
        return self._qdrant_models_point()
    
    @property
    def available(self) -> Dict[str, bool]:
        """Check availability of database dependencies."""
        return {
            "qdrant_client": self._qdrant_client.available,
            "qdrant_models": self._qdrant_models_distance.available
        }


class OptionalDependencies:
    """Lazy loading container for optional dependencies."""
    
    def __init__(self):
        self._psutil = LazyImport("psutil")
        self._uvicorn = LazyImport("uvicorn")
        self._fastapi = LazyImport("fastapi", "FastAPI")
        
    @property
    def psutil(self):
        """Get psutil module (lazy loaded)."""
        return self._psutil()
    
    @property
    def uvicorn(self):
        """Get uvicorn module (lazy loaded)."""
        return self._uvicorn()
    
    @property
    def FastAPI(self):
        """Get FastAPI class (lazy loaded)."""
        return self._fastapi()
    
    @property
    def available(self) -> Dict[str, bool]:
        """Check availability of optional dependencies."""
        return {
            "psutil": self._psutil.available,
            "uvicorn": self._uvicorn.available,
            "fastapi": self._fastapi.available
        }


# Global lazy import containers
ml_deps = MLDependencies()
db_deps = DatabaseDependencies()
optional_deps = OptionalDependencies()


def lazy_import_decorator(dependencies: Dict[str, str]):
    """Decorator to lazy import dependencies for a function or class."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import dependencies when function is called
            imported = {}
            for name, module_path in dependencies.items():
                try:
                    if '.' in module_path:
                        module_name, attr_name = module_path.rsplit('.', 1)
                        module = importlib.import_module(module_name)
                        imported[name] = getattr(module, attr_name)
                    else:
                        imported[name] = importlib.import_module(module_path)
                except ImportError as e:
                    logger.warning(f"Could not import {name} from {module_path}: {e}")
                    imported[name] = None
            
            # Add imported dependencies to kwargs
            kwargs.update(imported)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def check_dependency_availability() -> Dict[str, Any]:
    """Check availability of all lazy-loaded dependencies."""
    return {
        "ml": ml_deps.available,
        "database": db_deps.available,
        "optional": optional_deps.available
    }


def warm_up_critical_dependencies():
    """Pre-load critical dependencies during startup if needed."""
    logger.info("Warming up critical dependencies...")
    
    # Pre-load database dependencies if available
    if db_deps.available["qdrant_client"]:
        logger.debug("Pre-loading Qdrant client classes")
        db_deps.QdrantClient
        db_deps.Distance
        db_deps.VectorParams
    
    # Only pre-load numpy (lightweight) from ML dependencies
    if ml_deps.available["numpy"]:
        logger.debug("Pre-loading numpy")
        ml_deps.numpy


class ImportTimer:
    """Utility to measure import times for performance analysis."""
    
    def __init__(self):
        self.import_times: Dict[str, float] = {}
    
    def time_import(self, module_name: str) -> Any:
        """Time the import of a module."""
        import time
        
        start_time = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
            elapsed = time.perf_counter() - start_time
            self.import_times[module_name] = elapsed
            logger.debug(f"Import {module_name}: {elapsed:.3f}s")
            return module
        except ImportError as e:
            elapsed = time.perf_counter() - start_time
            self.import_times[module_name] = elapsed
            logger.warning(f"Failed import {module_name}: {elapsed:.3f}s - {e}")
            return None
    
    def get_report(self) -> Dict[str, Any]:
        """Get import timing report."""
        if not self.import_times:
            return {"total_time": 0, "imports": {}, "slowest": []}
        
        total_time = sum(self.import_times.values())
        slowest = sorted(
            self.import_times.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "total_time": total_time,
            "imports": self.import_times,
            "slowest": slowest,
            "count": len(self.import_times)
        }


# Global import timer for startup analysis
import_timer = ImportTimer()