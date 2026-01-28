import hashlib
import os
from typing import Optional
from src.core.logger import get_logger

log = get_logger(name="cache_version")

class CacheVersionManager:
    """Manages cache versioning based on code changes to invalidate stale cache entries."""
    
    def __init__(self, monitored_files: list = None):
        self.version_hash = self._compute_version_hash(monitored_files)
        
    def _compute_version_hash(self, files: list = None) -> str:
        """Compute a hash based on the content of critical files."""
        if not files:
            # Default to monitoring config and core files implies semantic changes
            files = [
                "config/settings.py",
                "src/rag/retrieval.py"
            ]
            
        hasher = hashlib.md5()
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        buf = f.read()
                        hasher.update(buf)
                except Exception as e:
                    log.warning(f"Could not read file {file_path} for versioning: {e}")
            else:
                # log.debug(f"File {file_path} not found for cache versioning")
                pass
                
        return hasher.hexdigest()[:8]

    def get_version(self) -> str:
        return self.version_hash

    def get_cache_prefix(self) -> str:
        return f"v{self.version_hash}:"

def get_cache_version_manager() -> CacheVersionManager:
    return CacheVersionManager()
