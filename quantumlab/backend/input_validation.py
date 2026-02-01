"""
Input validation utilities to prevent injection attacks and malicious input.
"""

import re
from typing import Any, Optional
from fastapi import HTTPException, status


class InputValidator:
    """
    Comprehensive input validation to prevent injection attacks.
    """
    
    # Regex patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|\/\*|\*\/|xp_|sp_)",
        r"(\bOR\b.*=.*|1=1|' OR ')",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<embed",
        r"<object",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\(",
        r"\.\.\/",
        r"\/etc\/passwd",
    ]
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input by removing potentially dangerous characters.
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
        
        Returns:
            Sanitized string
        
        Raises:
            HTTPException: If input is invalid
        """
        if not isinstance(value, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input must be a string"
            )
        
        # Check length
        if len(value) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input too long. Maximum length is {max_length}"
            )
        
        # Check for SQL injection
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Potentially malicious SQL pattern detected"
                )
        
        # Check for XSS
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Potentially malicious XSS pattern detected"
                )
        
        # Check for command injection
        for pattern in InputValidator.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Potentially malicious command injection pattern detected"
                )
        
        # Remove null bytes
        value = value.replace("\x00", "")
        
        # Trim whitespace
        value = value.strip()
        
        return value
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        return email.lower()
    
    @staticmethod
    def validate_number(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Validate numeric input with bounds"""
        try:
            num = float(value)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid number format"
            )
        
        if min_val is not None and num < min_val:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Value must be at least {min_val}"
            )
        
        if max_val is not None and num > max_val:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Value must be at most {max_val}"
            )
        
        return num
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Validate filename to prevent path traversal"""
        # Remove directory separators
        filename = filename.replace("/", "").replace("\\", "").replace("..", "")
        
        # Allow only alphanumeric, dash, underscore, and dot
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename. Only alphanumeric characters, dash, underscore, and dot allowed"
            )
        
        return filename
    
    @staticmethod
    def validate_molecule_data(data: dict) -> dict:
        """Validate molecule generation parameters"""
        if "num_atoms" in data:
            data["num_atoms"] = int(InputValidator.validate_number(
                data["num_atoms"], min_val=1, max_val=1000
            ))
        
        if "band_gap" in data:
            data["band_gap"] = InputValidator.validate_number(
                data["band_gap"], min_val=0, max_val=10
            )
        
        if "name" in data:
            data["name"] = InputValidator.sanitize_string(data["name"], max_length=100)
        
        return data
