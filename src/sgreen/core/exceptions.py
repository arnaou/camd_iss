"""Custom exceptions for sgreen package."""


class SgreenError(Exception):
    """Base exception for sgreen package."""
    pass


class CompoundError(SgreenError):
    """Exception for compound-related errors."""
    pass


class ValidationError(CompoundError):
    """Exception for validation errors."""
    pass


class ConnectivityError(ValidationError):
    """Exception for connectivity validation errors."""
    pass


class GroupDataError(CompoundError):
    """Exception for group data loading errors."""
    pass