"""Manages concurrent threads and tasks."""
import logging
import asyncio
from typing import Dict, Optional, Callable
from asyncio import Task

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    Manages concurrent threads and async tasks.
    """
    
    def __init__(self):
        """Initialize thread manager."""
        self._tasks: Dict[str, Task] = {}
        self._running = False
        logger.info("ThreadManager initialized")
    
    def start_background_task(
        self,
        coroutine: Callable,
        name: str,
        *args,
        **kwargs
    ) -> Task:
        """
        Starts named background task.
        
        Parameters:
            coroutine: Async function to run
            name: Task name for identification
            *args: Positional arguments for coroutine
            **kwargs: Keyword arguments for coroutine
        
        Returns:
            asyncio.Task: The created task
        """
        if name in self._tasks and not self._tasks[name].done():
            logger.warning(f"Task '{name}' is already running")
            return self._tasks[name]
        
        task = asyncio.create_task(coroutine(*args, **kwargs), name=name)
        self._tasks[name] = task
        
        logger.info(f"Started background task: {name}")
        return task
    
    def stop_task(self, name: str, timeout: Optional[float] = None) -> bool:
        """
        Stop a specific task.
        
        Parameters:
            name: Task name
            timeout: Optional timeout in seconds
        
        Returns:
            bool: True if stopped successfully
        """
        if name not in self._tasks:
            logger.warning(f"Task '{name}' not found")
            return False
        
        task = self._tasks[name]
        
        if task.done():
            logger.info(f"Task '{name}' already completed")
            return True
        
        task.cancel()
        logger.info(f"Cancelled task: {name}")
        
        return True
    
    async def stop_all(self, timeout: float = 5.0) -> None:
        """
        Stops all managed tasks.
        
        Parameters:
            timeout: Maximum time to wait for tasks to stop
        """
        if not self._tasks:
            return
        
        logger.info(f"Stopping {len(self._tasks)} tasks...")
        
        # Cancel all tasks
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelling task: {name}")
        
        # Wait for tasks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=timeout
            )
            logger.info("All tasks stopped")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for tasks to stop")
        except Exception as e:
            logger.error(f"Error stopping tasks: {e}")
        
        self._tasks.clear()
    
    def get_task_status(self) -> Dict[str, str]:
        """
        Get status of all tasks.
        
        Returns:
            dict: Task names mapped to their status
        """
        status = {}
        
        for name, task in self._tasks.items():
            if task.done():
                if task.cancelled():
                    status[name] = "cancelled"
                elif task.exception():
                    status[name] = f"failed: {task.exception()}"
                else:
                    status[name] = "completed"
            else:
                status[name] = "running"
        
        return status
    
    def is_task_running(self, name: str) -> bool:
        """
        Check if a task is currently running.
        
        Parameters:
            name: Task name
        
        Returns:
            bool: True if running
        """
        return name in self._tasks and not self._tasks[name].done()
    
    def get_active_count(self) -> int:
        """
        Get number of active tasks.
        
        Returns:
            int: Number of running tasks
        """
        return sum(1 for task in self._tasks.values() if not task.done())
