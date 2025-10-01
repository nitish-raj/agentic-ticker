"""
Thread-safe session state manager for Streamlit applications.
Provides atomic operations and prevents race conditions.
"""

import threading
import streamlit as st
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class SessionStateManager:
    """
    Thread-safe manager for Streamlit session state operations.
    Provides atomic operations and prevents race conditions.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure consistent locking across the app."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the session state manager."""
        if not hasattr(self, '_initialized'):
            self._locks = {}
            self._global_lock = threading.Lock()
            self._initialized = True
    
    def _get_lock(self, key: str) -> threading.Lock:
        """Get or create a lock for a specific session state key."""
        if key not in self._locks:
            with self._global_lock:
                if key not in self._locks:
                    self._locks[key] = threading.Lock()
        return self._locks[key]
    
    @contextmanager
    def atomic_operation(self, keys: Optional[list] = None):
        """
        Context manager for atomic operations on session state.
        
        Args:
            keys: List of session state keys to lock. If None, uses global lock.
        """
        if keys is None:
            # Use global lock for operations involving multiple keys
            lock = self._global_lock
        else:
            # For single key operations, use key-specific lock
            if len(keys) == 1:
                lock = self._get_lock(keys[0])
            else:
                # For multiple keys, acquire locks in sorted order to prevent deadlock
                locks = [self._get_lock(key) for key in sorted(keys)]
                lock = MultiLock(locks)
        
        try:
            lock.acquire()
            yield
        finally:
            lock.release()
    
    def set_atomic(self, key: str, value: Any) -> None:
        """
        Atomically set a session state value.
        
        Args:
            key: Session state key
            value: Value to set
        """
        with self.atomic_operation([key]):
            st.session_state[key] = value
            logger.debug(f"Atomically set session state key: {key}")
    
    def get_atomic(self, key: str, default: Any = None) -> Any:
        """
        Atomically get a session state value.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Session state value or default
        """
        with self.atomic_operation([key]):
            return st.session_state.get(key, default)
    
    def update_atomic(self, updates: Dict[str, Any]) -> None:
        """
        Atomically update multiple session state values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        if not updates:
            return
            
        keys = list(updates.keys())
        with self.atomic_operation(keys):
            for key, value in updates.items():
                st.session_state[key] = value
            logger.debug(f"Atomically updated session state keys: {keys}")
    
    def delete_atomic(self, keys: list) -> None:
        """
        Atomically delete multiple session state keys.
        
        Args:
            keys: List of keys to delete
        """
        if not keys:
            return
            
        with self.atomic_operation(keys):
            for key in keys:
                if key in st.session_state:
                    del st.session_state[key]
            logger.debug(f"Atomically deleted session state keys: {keys}")
    
    def clear_all_atomic(self) -> None:
        """Atomically clear all session state."""
        with self.atomic_operation():
            keys_to_delete = list(st.session_state.keys())
            for key in keys_to_delete:
                del st.session_state[key]
            logger.debug("Atomically cleared all session state")
    
    def compare_and_swap(self, key: str, expected_value: Any, new_value: Any) -> bool:
        """
        Atomically compare and swap a session state value.
        
        Args:
            key: Session state key
            expected_value: Expected current value
            new_value: New value to set
            
        Returns:
            True if swap was successful, False if expected value didn't match
        """
        with self.atomic_operation([key]):
            current_value = st.session_state.get(key)
            if current_value == expected_value:
                st.session_state[key] = new_value
                logger.debug(f"Compare-and-swap successful for key: {key}")
                return True
            else:
                logger.debug(f"Compare-and-swap failed for key: {key} (expected: {expected_value}, got: {current_value})")
                return False
    
    def execute_with_lock(self, func: Callable, keys: Optional[list] = None, *args, **kwargs):
        """
        Execute a function with session state locks held.
        
        Args:
            func: Function to execute
            keys: Session state keys to lock
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        with self.atomic_operation(keys):
            return func(*args, **kwargs)


class MultiLock:
    """
    Helper class to manage multiple locks safely.
    Acquires locks in sorted order to prevent deadlock.
    """
    
    def __init__(self, locks: list):
        """Initialize with a list of locks."""
        self.locks = sorted(locks, key=lambda x: id(x))  # Sort by id to ensure consistent ordering
    
    def acquire(self):
        """Acquire all locks."""
        for lock in self.locks:
            lock.acquire()
    
    def release(self):
        """Release all locks in reverse order."""
        for lock in reversed(self.locks):
            lock.release()


# Global instance for easy access
session_manager = SessionStateManager()


# Convenience functions for common operations
def set_session_state(key: str, value: Any) -> None:
    """Convenience function to atomically set a session state value."""
    session_manager.set_atomic(key, value)


def get_session_state(key: str, default: Any = None) -> Any:
    """Convenience function to atomically get a session state value."""
    return session_manager.get_atomic(key, default)


def update_session_state(updates: Dict[str, Any]) -> None:
    """Convenience function to atomically update multiple session state values."""
    session_manager.update_atomic(updates)


def delete_session_state(keys: list) -> None:
    """Convenience function to atomically delete session state keys."""
    session_manager.delete_atomic(keys)


def clear_session_state() -> None:
    """Convenience function to atomically clear all session state."""
    session_manager.clear_all_atomic()