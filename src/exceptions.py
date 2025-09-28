"""Custom exception hierarchy for the Agentic Ticker application."""


class AgenticTickerError(Exception):
    """Base exception class for all Agentic Ticker errors."""
    pass


class UtilityModuleError(AgenticTickerError):
    """Base exception for utility module related errors."""
    pass


class UtilityFunctionError(AgenticTickerError):
    """Base exception for utility function related errors."""
    pass


class CodeDuplicationError(AgenticTickerError):
    """Base exception for code duplication detection errors."""
    pass


class RefactoringError(AgenticTickerError):
    """Base exception for refactoring progress errors."""
    pass


# Utility Module specific exceptions
class ValidationError(UtilityModuleError):
    """Raised when validation fails."""
    pass


class ModuleNotFoundError(UtilityModuleError):
    """Raised when a utility module is not found."""
    pass


class FunctionNotFoundError(UtilityModuleError):
    """Raised when a utility function is not found."""
    pass


class DependencyError(UtilityModuleError):
    """Raised when there's an issue with module dependencies."""
    pass


class FilePathError(UtilityModuleError):
    """Raised when there's an issue with file path validation."""
    pass


# Utility Function specific exceptions
class ParameterError(UtilityFunctionError):
    """Raised when there's an issue with function parameters."""
    pass


class FunctionGenerationError(UtilityFunctionError):
    """Raised when function generation fails."""
    pass


class CodeGenerationError(UtilityFunctionError):
    """Raised when code generation fails."""
    pass


# Code Duplication specific exceptions
class PatternDetectionError(CodeDuplicationError):
    """Raised when pattern detection fails."""
    pass


class AnalysisError(CodeDuplicationError):
    """Raised when code analysis fails."""
    pass


class SimilarityError(CodeDuplicationError):
    """Raised when similarity calculation fails."""
    pass


# Refactoring specific exceptions
class ProgressError(RefactoringError):
    """Raised when progress tracking fails."""
    pass


class RefactoringStepError(RefactoringError):
    """Raised when a refactoring step fails."""
    pass


class CompletionError(RefactoringError):
    """Raised when refactoring completion fails."""
    pass


class NotFoundError(AgenticTickerError):
    """Raised when a requested resource is not found."""
    pass


class ConflictError(AgenticTickerError):
    """Raised when there's a conflict with existing resources."""
    pass
