"""
Code duplication detection and tracking system for the Agentic Ticker.

This module provides functionality to scan source files for common duplication patterns,
identify identical code blocks, similar structures, and repeated logic. It calculates
similarity scores and severity levels, generates CodeDuplicationPattern entities with
locations, and provides analysis results for API endpoints.
"""

import os
import re
import hashlib
import ast
import difflib
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from enum import Enum

# Define enums and models inline for now
class PatternType(str, Enum):
    """Types of code duplication patterns."""
    IDENTICAL_CODE = "identical_code"
    SIMILAR_STRUCTURE = "similar_structure"
    REPEATED_LOGIC = "repeated_logic"
    MAGIC_STRINGS = "magic_strings"
    HARDCODED_VALUES = "hardcoded_values"


class SeverityLevel(str, Enum):
    """Severity levels for code duplication patterns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternStatus(str, Enum):
    """Status of code duplication pattern handling."""
    DETECTED = "detected"
    ANALYZED = "analyzed"
    REFACTORED = "refactored"
    IGNORED = "ignored"


class CodeLocation:
    """Represents a specific location in the codebase with detailed context."""

    def __init__(self, file_path: str, start_line: int, end_line: int,
                 function_name: Optional[str] = None, class_name: Optional[str] = None,
                 module_name: Optional[str] = None):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.function_name = function_name
        self.class_name = class_name
        self.module_name = module_name or self._extract_module_name()

    def _extract_module_name(self) -> str:
        """Extract module name from file path."""
        path = Path(self.file_path)
        if path.suffix == '.py':
            return path.stem
        return path.name

    @property
    def line_count(self) -> int:
        """Calculate the number of lines in this location."""
        return self.end_line - self.start_line + 1


class CodeDuplicationPattern:
    """Represents a code duplication pattern detected in the codebase."""

    def __init__(self, id: str, pattern_type: PatternType, severity: SeverityLevel,
                 locations: List[CodeLocation], lines_affected: int,
                 suggested_refactoring: str, estimated_savings: int,
                 detection_date: Optional[datetime] = None,
                 status: PatternStatus = PatternStatus.DETECTED):
        self.id = id
        self.pattern_type = pattern_type
        self.severity = severity
        self.locations = locations
        self.lines_affected = lines_affected
        self.suggested_refactoring = suggested_refactoring
        self.estimated_savings = estimated_savings
        self.detection_date = detection_date or datetime.now()
        self.status = status

        # Validate that we have at least 2 locations
        if len(locations) < 2:
            raise ValueError('At least 2 locations are required for a duplication pattern')

        # Validate estimated savings
        if estimated_savings > lines_affected:
            raise ValueError('estimated_savings cannot exceed lines_affected')


class CodeBlock:
    """Represents a code block for duplication analysis."""
    
    def __init__(self, file_path: str, start_line: int, end_line: int, 
                 content: str, function_name: Optional[str] = None, 
                 class_name: Optional[str] = None, module_name: Optional[str] = None):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.content = content
        self.function_name = function_name
        self.class_name = class_name
        self.module_name = module_name or self._extract_module_name()
        self.normalized_content = self._normalize_content()
        self.hash = self._calculate_hash()
    
    def _extract_module_name(self) -> str:
        """Extract module name from file path."""
        path = Path(self.file_path)
        if path.suffix == '.py':
            return path.stem
        return path.name
    
    def _calculate_hash(self) -> str:
        """Calculate hash of the code block content."""
        return hashlib.md5(self.normalized_content.encode()).hexdigest()
    
    def _normalize_content(self) -> str:
        """Normalize content for comparison (remove comments, whitespace, etc.)."""
        lines = self.content.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove comments
            line = re.sub(r'#.*$', '', line)
            # Remove leading/trailing whitespace
            line = line.strip()
            # Skip empty lines
            if line:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def to_code_location(self) -> CodeLocation:
        """Convert to CodeLocation model."""
        return CodeLocation(
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            function_name=self.function_name,
            class_name=self.class_name,
            module_name=self.module_name
        )
    
    @property
    def line_count(self) -> int:
        """Get the number of lines in this code block."""
        return self.end_line - self.start_line + 1


class DuplicationDetector:
    """Main class for detecting code duplication patterns."""
    
    def __init__(self, min_lines: int = 3, similarity_threshold: float = 0.8,
                 ignore_comments: bool = True, ignore_whitespace: bool = True,
                 safe_root: Optional[str] = None):
        self.min_lines = min_lines
        self.similarity_threshold = similarity_threshold
        self.ignore_comments = ignore_comments
        self.ignore_whitespace = ignore_whitespace
        self.patterns: List[CodeDuplicationPattern] = []
        self._file_cache: Dict[str, List[str]] = {}
        # Default to current working directory if not specified
        self.safe_root = os.path.abspath(safe_root) if safe_root else os.path.abspath(os.getcwd())
    
    def scan_codebase(self, analysis_scope: str, file_patterns: List[str],
                     include_tests: bool = False) -> List[CodeDuplicationPattern]:
        """
        Scan the codebase for duplication patterns.
        
        Args:
            analysis_scope: Directory path to scan (e.g., 'src/')
            file_patterns: List of file patterns to include (e.g., ['*.py'])
            include_tests: Whether to include test files in the analysis
            
        Returns:
            List of detected CodeDuplicationPattern objects
        """
        self.patterns = []
        
        # Find all files matching the patterns
        files_to_scan = self._find_files(analysis_scope, file_patterns, include_tests)
        
        if not files_to_scan:
            return []
        
        # Extract code blocks from all files
        all_blocks = []
        for file_path in files_to_scan:
            try:
                blocks = self._extract_code_blocks(file_path)
                all_blocks.extend(blocks)
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue
        
        # Find identical code blocks
        identical_patterns = self._find_identical_blocks(all_blocks)
        self.patterns.extend(identical_patterns)
        
        # Find similar structures
        similar_patterns = self._find_similar_structures(all_blocks)
        self.patterns.extend(similar_patterns)
        
        # Find repeated logic patterns
        logic_patterns = self._find_repeated_logic(all_blocks)
        self.patterns.extend(logic_patterns)
        
        # Find magic strings and hardcoded values
        magic_patterns = self._find_magic_strings_and_hardcoded_values(all_blocks)
        self.patterns.extend(magic_patterns)
        
        return self.patterns
    
    def _find_files(self, analysis_scope: str, file_patterns: List[str], 
                    include_tests: bool) -> List[str]:
        """Find all files matching the given patterns, restricting to safe root."""
        files = []
        # Defensive validation of analysis_scope: must not be absolute or contain traversal
        if os.path.isabs(analysis_scope):
            raise ValueError(f"Analysis scope must be a relative path under the safe root (got absolute path: {analysis_scope!r})")
        if '..' in Path(analysis_scope).parts:
            raise ValueError(f"Analysis scope must not contain parent directory traversal '..' (got: {analysis_scope!r})")
        # Compute the normalized/real analysis scope joined to safe root
        abs_safe_root = os.path.realpath(os.path.abspath(self.safe_root))
        candidate_scope = os.path.realpath(os.path.abspath(os.path.join(abs_safe_root, analysis_scope)))
        if not os.path.exists(candidate_scope):
            raise ValueError(f"Analysis scope path does not exist: {candidate_scope}")
        # Use os.path.commonpath to securely check that candidate_scope is within abs_safe_root
        if os.path.commonpath([abs_safe_root, candidate_scope]) != abs_safe_root:
            raise ValueError(f"Analysis scope path escapes the allowed directory: {analysis_scope}")
        base_path = Path(candidate_scope)

        for pattern in file_patterns:
            # Handle glob patterns
            if '**' in pattern:
                # Recursive search
                matched_files = base_path.rglob(pattern.replace('**/', ''))
            else:
                # Non-recursive search
                matched_files = base_path.glob(pattern)
            
            for file_path in matched_files:
                if file_path.is_file():
                    file_real_path = os.path.realpath(str(file_path))
                    # Ensure each file is contained in safe root
                    if not self._is_within_safe_root(file_real_path, abs_safe_root):
                        continue
                    
                    # Skip test files if not including tests
                    if not include_tests and self._is_test_file(file_real_path):
                        continue
                    
                    # Skip non-Python files for now (can be extended later)
                    if not file_real_path.endswith('.py'):
                        continue
                    
                    files.append(file_real_path)
        
    def _is_within_safe_root(self, path: str, safe_root: str) -> bool:
        """Return True if the given path is within the safe_root directory."""
        try:
            return os.path.commonpath([os.path.realpath(path), safe_root]) == safe_root
        except ValueError:
            # Raised if paths are on different drives (Windows), treat as not allowed
            return False

        return sorted(list(set(files)))
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file."""
        test_indicators = ['test_', '_test.py', 'tests/', '/tests/', 'test/']
        return any(indicator in file_path.lower() for indicator in test_indicators)
    
    def _extract_code_blocks(self, file_path: str) -> List[CodeBlock]:
        """Extract code blocks from a Python file."""
        blocks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                self._file_cache[file_path] = lines
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")
        
        try:
            # Parse AST to extract function and class definitions
            tree = ast.parse(content, filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function block
                    func_block = self._extract_function_block(node, file_path, lines)
                    if func_block and func_block.line_count >= self.min_lines:
                        blocks.append(func_block)
                
                elif isinstance(node, ast.ClassDef):
                    # Extract class block
                    class_block = self._extract_class_block(node, file_path, lines)
                    if class_block and class_block.line_count >= self.min_lines:
                        blocks.append(class_block)
                    
                    # Also extract methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_block = self._extract_method_block(item, node, file_path, lines)
                            if method_block and method_block.line_count >= self.min_lines:
                                blocks.append(method_block)
            
            # Extract standalone code blocks (not in functions/classes)
            standalone_blocks = self._extract_standalone_blocks(tree, file_path, lines)
            blocks.extend(standalone_blocks)
            
        except SyntaxError as e:
            print(f"Warning: Syntax error in {file_path}: {e}")
            # Fall back to simple line-based extraction
            blocks.extend(self._extract_line_based_blocks(file_path, lines))
        
        return blocks
    
    def _extract_function_block(self, node: ast.FunctionDef, file_path: str, 
                               lines: List[str]) -> Optional[CodeBlock]:
        """Extract a function code block."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Extract the function content
        content = '\n'.join(lines[start_line-1:end_line])
        
        return CodeBlock(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            function_name=node.name
        )
    
    def _extract_class_block(self, node: ast.ClassDef, file_path: str, 
                            lines: List[str]) -> Optional[CodeBlock]:
        """Extract a class code block."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Extract the class content
        content = '\n'.join(lines[start_line-1:end_line])
        
        return CodeBlock(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            class_name=node.name
        )
    
    def _extract_method_block(self, node: ast.FunctionDef, class_node: ast.ClassDef, 
                             file_path: str, lines: List[str]) -> Optional[CodeBlock]:
        """Extract a method code block."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Extract the method content
        content = '\n'.join(lines[start_line-1:end_line])
        
        return CodeBlock(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            function_name=node.name,
            class_name=class_node.name
        )
    
    def _extract_standalone_blocks(self, tree: ast.AST, file_path: str, 
                                  lines: List[str]) -> List[CodeBlock]:
        """Extract standalone code blocks (not in functions/classes)."""
        blocks = []
        
        # Find top-level statements that are not function/class definitions
        if hasattr(tree, 'body'):
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    if end_line - start_line + 1 >= self.min_lines:
                        content = '\n'.join(lines[start_line-1:end_line])
                        blocks.append(CodeBlock(
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            content=content
                        ))
        
        return blocks
    
    def _extract_line_based_blocks(self, file_path: str, lines: List[str]) -> List[CodeBlock]:
        """Fallback method to extract blocks based on line analysis."""
        blocks = []
        
        # Look for consecutive lines that might be duplicated
        # This is a simple heuristic approach
        current_block_start = None
        current_block_lines = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                if current_block_start and len(current_block_lines) >= self.min_lines:
                    # End of a potential block
                    content = '\n'.join(current_block_lines)
                    blocks.append(CodeBlock(
                        file_path=file_path,
                        start_line=current_block_start,
                        end_line=i-1,
                        content=content
                    ))
                current_block_start = None
                current_block_lines = []
                continue
            
            # Start a new block or continue existing one
            if current_block_start is None:
                current_block_start = i
            current_block_lines.append(line)
        
        # Handle final block
        if current_block_start and len(current_block_lines) >= self.min_lines:
            content = '\n'.join(current_block_lines)
            blocks.append(CodeBlock(
                file_path=file_path,
                start_line=current_block_start,
                end_line=len(lines),
                content=content
            ))
        
        return blocks
    
    def _find_identical_blocks(self, blocks: List[CodeBlock]) -> List[CodeDuplicationPattern]:
        """Find identical code blocks."""
        patterns = []
        hash_groups = defaultdict(list)
        
        # Group blocks by their hash
        for block in blocks:
            hash_groups[block.hash].append(block)
        
        # Create patterns for groups with multiple identical blocks
        for hash_value, block_group in hash_groups.items():
            if len(block_group) >= 2:
                pattern = self._create_pattern_from_blocks(
                    block_group, PatternType.IDENTICAL_CODE
                )
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _find_similar_structures(self, blocks: List[CodeBlock]) -> List[CodeDuplicationPattern]:
        """Find similar code structures."""
        patterns = []
        processed_blocks = set()
        
        for i, block1 in enumerate(blocks):
            if block1.hash in processed_blocks:
                continue
            
            similar_blocks = [block1]
            
            for j, block2 in enumerate(blocks[i+1:], i+1):
                if block2.hash in processed_blocks:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(block1, block2)
                
                if similarity >= self.similarity_threshold:
                    similar_blocks.append(block2)
                    processed_blocks.add(block2.hash)
            
            if len(similar_blocks) >= 2:
                pattern = self._create_pattern_from_blocks(
                    similar_blocks, PatternType.SIMILAR_STRUCTURE
                )
                if pattern:
                    patterns.append(pattern)
            
            processed_blocks.add(block1.hash)
        
        return patterns
    
    def _find_repeated_logic(self, blocks: List[CodeBlock]) -> List[CodeDuplicationPattern]:
        """Find repeated logic patterns."""
        patterns = []
        
        # Look for common logic patterns like input validation, error handling, etc.
        logic_patterns = {
            'input_validation': [
                r'if.*is None', r'if.*==.*None', r'if.*not.*', r'if.*len\(.*\)',
                r'try:.*except', r'raise.*Error', r'assert.*'
            ],
            'error_handling': [
                r'try:.*except.*:', r'except.*Error.*:', r'raise.*', r'logger\.error',
                r'print.*error', r'logging\.error'
            ],
            'data_processing': [
                r'for.*in.*:', r'while.*:', r'if.*in.*:', r'\.append\(',
                r'\.extend\(', r'\.update\(', r'dict\(', r'list\('
            ]
        }
        
        for pattern_name, regex_patterns in logic_patterns.items():
            matching_blocks = []
            
            for block in blocks:
                # Check if block matches any of the regex patterns for this logic type
                for regex_pattern in regex_patterns:
                    if re.search(regex_pattern, block.content, re.IGNORECASE):
                        matching_blocks.append(block)
                        break
            
            if len(matching_blocks) >= 2:
                pattern = self._create_pattern_from_blocks(
                    matching_blocks, PatternType.REPEATED_LOGIC
                )
                if pattern:
                    # Add specific description for logic pattern
                    pattern.suggested_refactoring = f"Consider creating a reusable {pattern_name.replace('_', ' ')} utility function"
                    patterns.append(pattern)
        
        return patterns
    
    def _find_magic_strings_and_hardcoded_values(self, blocks: List[CodeBlock]) -> List[CodeDuplicationPattern]:
        """Find magic strings and hardcoded values."""
        patterns = []
        
        # Find magic strings (string literals used in multiple places)
        string_literals = defaultdict(list)
        
        for block in blocks:
            # Extract string literals from the content
            strings = re.findall(r'["\'](.*?)["\']', block.content)
            for string in strings:
                # Skip very short strings and common strings
                if len(string) > 3 and string.lower() not in ['true', 'false', 'none', 'null']:
                    string_literals[string].append(block)
        
        # Create patterns for strings that appear in multiple blocks
        for string_value, block_list in string_literals.items():
            if len(block_list) >= 2:
                pattern = self._create_pattern_from_blocks(
                    block_list, PatternType.MAGIC_STRINGS
                )
                if pattern:
                    pattern.suggested_refactoring = f"Extract magic string '{string_value}' into a constant or configuration variable"
                    patterns.append(pattern)
        
        # Find hardcoded numeric values
        numeric_values = defaultdict(list)
        
        for block in blocks:
            # Extract numeric literals (excluding single digits and common values)
            numbers = re.findall(r'\b([0-9]{2,})\b', block.content)
            for number in numbers:
                numeric_values[number].append(block)
        
        # Create patterns for numbers that appear in multiple blocks
        for number_value, block_list in numeric_values.items():
            if len(block_list) >= 2:
                pattern = self._create_pattern_from_blocks(
                    block_list, PatternType.HARDCODED_VALUES
                )
                if pattern:
                    pattern.suggested_refactoring = f"Extract hardcoded value {number_value} into a constant or configuration variable"
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_similarity(self, block1: CodeBlock, block2: CodeBlock) -> float:
        """Calculate similarity score between two code blocks."""
        # Use difflib to calculate sequence similarity
        similarity = difflib.SequenceMatcher(
            None, block1.normalized_content, block2.normalized_content
        ).ratio()
        
        # Adjust similarity based on structural similarity
        structure_bonus = 0.0
        
        # Bonus for same function/class context
        if block1.function_name and block2.function_name:
            if block1.function_name == block2.function_name:
                structure_bonus += 0.1
        
        if block1.class_name and block2.class_name:
            if block1.class_name == block2.class_name:
                structure_bonus += 0.1
        
        return min(similarity + structure_bonus, 1.0)
    
    def _create_pattern_from_blocks(self, blocks: List[CodeBlock], 
                                   pattern_type: PatternType) -> Optional[CodeDuplicationPattern]:
        """Create a CodeDuplicationPattern from a list of similar blocks."""
        if len(blocks) < 2:
            return None
        
        # Calculate severity based on number of occurrences and lines affected
        severity = self._calculate_severity(blocks)
        
        # Calculate estimated savings
        estimated_savings = self._calculate_estimated_savings(blocks)
        
        # Generate suggested refactoring
        suggested_refactoring = self._generate_refactoring_suggestion(blocks, pattern_type)
        
        # Create locations
        locations = [block.to_code_location() for block in blocks]
        
        # Calculate total lines affected
        lines_affected = sum(block.line_count for block in blocks)
        
        pattern = CodeDuplicationPattern(
            id=self._generate_pattern_id(blocks, pattern_type),
            pattern_type=pattern_type,
            severity=severity,
            locations=locations,
            lines_affected=lines_affected,
            suggested_refactoring=suggested_refactoring,
            estimated_savings=estimated_savings,
            detection_date=datetime.now(),
            status=PatternStatus.DETECTED
        )
        
        return pattern
    
    def _calculate_severity(self, blocks: List[CodeBlock]) -> SeverityLevel:
        """Calculate severity level based on duplication characteristics."""
        num_blocks = len(blocks)
        total_lines = sum(block.line_count for block in blocks)
        
        # Critical: Many occurrences or large blocks
        if num_blocks >= 5 or total_lines >= 50:
            return SeverityLevel.CRITICAL
        
        # High: Multiple occurrences or medium-sized blocks
        if num_blocks >= 3 or total_lines >= 20:
            return SeverityLevel.HIGH
        
        # Medium: Few occurrences or small blocks
        if num_blocks >= 2 or total_lines >= 10:
            return SeverityLevel.MEDIUM
        
        # Low: Minimal duplication
        return SeverityLevel.LOW
    
    def _calculate_estimated_savings(self, blocks: List[CodeBlock]) -> int:
        """Calculate estimated lines of code that could be saved."""
        if len(blocks) < 2:
            return 0
        
        # Estimate: keep one copy, remove duplicates
        total_lines = sum(block.line_count for block in blocks)
        max_block_lines = max(block.line_count for block in blocks)
        
        # Savings = total lines - lines needed for shared implementation
        estimated_savings = total_lines - max_block_lines - 2  # -2 for function call overhead
        
        return max(0, estimated_savings)
    
    def _generate_refactoring_suggestion(self, blocks: List[CodeBlock], 
                                        pattern_type: PatternType) -> str:
        """Generate refactoring suggestion based on pattern type."""
        if pattern_type == PatternType.IDENTICAL_CODE:
            return "Extract identical code into a shared utility function or base class"
        elif pattern_type == PatternType.SIMILAR_STRUCTURE:
            return "Refactor similar structures using template method pattern or composition"
        elif pattern_type == PatternType.REPEATED_LOGIC:
            return "Consolidate repeated logic into a single, reusable function"
        elif pattern_type == PatternType.MAGIC_STRINGS:
            return "Extract magic strings into constants or configuration files"
        elif pattern_type == PatternType.HARDCODED_VALUES:
            return "Extract hardcoded values into constants, configuration, or parameters"
        else:
            return "Consider refactoring to reduce duplication and improve maintainability"
    
    def _generate_pattern_id(self, blocks: List[CodeBlock], 
                           pattern_type: PatternType) -> str:
        """Generate a unique ID for the pattern."""
        # Create a hash based on the first block's content and pattern type
        content_hash = hashlib.md5(
            f"{blocks[0].hash}_{pattern_type.value}".encode()
        ).hexdigest()[:8]
        
        return f"{pattern_type.value}_{content_hash}"
    
    def get_patterns_by_filter(self, file_path: Optional[str] = None,
                             pattern_type: Optional[str] = None,
                             severity: Optional[str] = None) -> List[CodeDuplicationPattern]:
        """
        Filter patterns by various criteria.
        
        Args:
            file_path: Filter by file path (substring match)
            pattern_type: Filter by pattern type
            severity: Filter by severity level
            
        Returns:
            Filtered list of patterns
        """
        filtered_patterns = self.patterns
        
        if file_path:
            filtered_patterns = [
                p for p in filtered_patterns
                if any(file_path in loc.file_path for loc in p.locations)
            ]
        
        if pattern_type:
            filtered_patterns = [
                p for p in filtered_patterns
                if p.pattern_type.value == pattern_type
            ]
        
        if severity:
            filtered_patterns = [
                p for p in filtered_patterns
                if p.severity.value == severity
            ]
        
        return filtered_patterns
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected patterns."""
        if not self.patterns:
            return {
                'total_patterns': 0,
                'by_type': {},
                'by_severity': {},
                'total_lines_affected': 0,
                'total_estimated_savings': 0
            }
        
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        total_lines = 0
        total_savings = 0
        
        for pattern in self.patterns:
            by_type[pattern.pattern_type.value] += 1
            by_severity[pattern.severity.value] += 1
            total_lines += pattern.lines_affected
            total_savings += pattern.estimated_savings
        
        return {
            'total_patterns': len(self.patterns),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'total_lines_affected': total_lines,
            'total_estimated_savings': total_savings
        }


def scan_for_duplications(analysis_scope: str, file_patterns: List[str],
                         min_duplication_lines: int = 5,
                         ignore_comments: bool = True,
                         ignore_whitespace: bool = True,
                         include_tests: bool = False,
                         similarity_threshold: float = 0.8,
                         safe_root: Optional[str] = None) -> List[CodeDuplicationPattern]:
    """
    Convenience function to scan for code duplications.
    
    Args:
        analysis_scope: Directory path to scan
        file_patterns: List of file patterns to include
        min_duplication_lines: Minimum lines for duplication detection
        ignore_comments: Whether to ignore comments in comparison
        ignore_whitespace: Whether to ignore whitespace in comparison
        include_tests: Whether to include test files
        similarity_threshold: Threshold for similarity detection (0.0-1.0)
        
    Returns:
        List of detected CodeDuplicationPattern objects
    """
    detector = DuplicationDetector(
        min_lines=min_duplication_lines,
        similarity_threshold=similarity_threshold,
        ignore_comments=ignore_comments,
        ignore_whitespace=ignore_whitespace,
        safe_root=safe_root
    )
    
    return detector.scan_codebase(analysis_scope, file_patterns, include_tests)


def get_patterns_by_filter(patterns: List[CodeDuplicationPattern],
                         file_path: Optional[str] = None,
                         pattern_type: Optional[str] = None,
                         severity: Optional[str] = None) -> List[CodeDuplicationPattern]:
    """
    Filter patterns by various criteria.
    
    Args:
        patterns: List of patterns to filter
        file_path: Filter by file path (substring match)
        pattern_type: Filter by pattern type
        severity: Filter by severity level
        
    Returns:
        Filtered list of patterns
    """
    filtered_patterns = patterns
    
    if file_path:
        filtered_patterns = [
            p for p in filtered_patterns
            if any(file_path in loc.file_path for loc in p.locations)
        ]
    
    if pattern_type:
        filtered_patterns = [
            p for p in filtered_patterns
            if p.pattern_type.value == pattern_type
        ]
    
    if severity:
        filtered_patterns = [
            p for p in filtered_patterns
            if p.severity.value == severity
        ]
    
    return filtered_patterns


# Example usage and testing
if __name__ == "__main__":
    # Test the detector on the current codebase
    print("Testing code duplication detector...")
    
    patterns = scan_for_duplications(
        analysis_scope="src/",
        file_patterns=["*.py"],
        min_duplication_lines=5,
        include_tests=False
    )
    
    print(f"Found {len(patterns)} duplication patterns")
    
    for pattern in patterns[:5]:  # Show first 5 patterns
        print(f"\nPattern ID: {pattern.id}")
        print(f"Type: {pattern.pattern_type.value}")
        print(f"Severity: {pattern.severity.value}")
        print(f"Locations: {len(pattern.locations)} files")
        print(f"Lines affected: {pattern.lines_affected}")
        print(f"Estimated savings: {pattern.estimated_savings}")
        print(f"Suggestion: {pattern.suggested_refactoring}")
        
        for loc in pattern.locations[:2]:  # Show first 2 locations
            print(f"  - {loc.file_path}:{loc.start_line}-{loc.end_line}")
        
        if len(pattern.locations) > 2:
            print(f"  ... and {len(pattern.locations) - 2} more locations")
    
    # Show statistics
    if patterns:
        detector = DuplicationDetector()
        detector.patterns = patterns
        stats = detector.get_pattern_statistics()
        print(f"\nStatistics:")
        print(f"Total patterns: {stats['total_patterns']}")
        print(f"By type: {stats['by_type']}")
        print(f"By severity: {stats['by_severity']}")
        print(f"Total lines affected: {stats['total_lines_affected']}")
        print(f"Total estimated savings: {stats['total_estimated_savings']}")