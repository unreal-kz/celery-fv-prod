import os
import logging
import tempfile
from pathlib import Path
from typing import Optional
from ..config import settings

logger = logging.getLogger(__name__)

class LocalStorage:
    def __init__(self):
        """Initialize local storage handler."""
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, file_content: bytes, filename: str) -> Optional[str]:
        """
        Save an uploaded file to local storage.
        
        Args:
            file_content: Raw bytes of the file
            filename: Original filename
            
        Returns:
            Optional[str]: Path to saved file, None if save fails
        """
        try:
            # Create a unique filename
            ext = Path(filename).suffix
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                dir=self.upload_dir,
                suffix=ext
            )
            
            # Write content to file
            with open(temp_file.name, 'wb') as f:
                f.write(file_content)
                
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            return None

    def cleanup_file(self, file_path: str) -> None:
        """
        Delete a file from local storage.
        
        Args:
            file_path: Path to the file to delete
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")

    def cleanup_old_files(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up files older than specified age.
        
        Args:
            max_age_seconds: Maximum age of files in seconds (default: 1 hour)
        """
        try:
            current_time = Path().stat().st_mtime
            for file_path in self.upload_dir.glob('*'):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        self.cleanup_file(str(file_path))
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")

# Create singleton instance
storage = LocalStorage()
