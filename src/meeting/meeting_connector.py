"""Base meeting connector interface."""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MeetingState(Enum):
    """Meeting connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IN_MEETING = "in_meeting"
    ERROR = "error"


@dataclass
class MeetingInfo:
    """Information about a meeting."""
    meeting_id: str
    title: str
    host: str
    start_time: Optional[datetime] = None
    participants: int = 0
    platform: str = "unknown"
    metadata: Dict = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}


class MeetingConnector(ABC):
    """
    Base class for meeting platform connectors.
    """
    
    def __init__(self):
        """Initialize base connector."""
        self._state = MeetingState.DISCONNECTED
        self._meeting_info: Optional[MeetingInfo] = None
        
        # Callbacks
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None
        self._on_participant_joined: Optional[Callable] = None
        self._on_participant_left: Optional[Callable] = None
        self._on_audio_received: Optional[Callable] = None
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def connect(self, meeting_url: str) -> bool:
        """
        Connects to meeting platform.
        
        Parameters:
            meeting_url: URL or ID of the meeting
        
        Returns:
            bool: Success
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnects from meeting."""
        pass
    
    @abstractmethod
    async def join_meeting(self, meeting_id: str, **kwargs) -> bool:
        """
        Joins a specific meeting.
        
        Parameters:
            meeting_id: Meeting identifier
            **kwargs: Platform-specific options
        
        Returns:
            bool: Success
        """
        pass
    
    @abstractmethod
    async def leave_meeting(self) -> None:
        """Leaves current meeting."""
        pass
    
    @abstractmethod
    async def get_audio_stream(self):
        """
        Gets audio stream from meeting.
        
        Returns:
            Audio stream generator
        """
        pass
    
    @abstractmethod
    async def send_audio(self, audio_data) -> None:
        """
        Sends audio to meeting.
        
        Parameters:
            audio_data: Audio samples to send
        """
        pass
    
    def get_state(self) -> MeetingState:
        """Get current connection state."""
        return self._state
    
    def get_meeting_info(self) -> Optional[MeetingInfo]:
        """Get current meeting info."""
        return self._meeting_info
    
    def set_on_connected(self, callback: Callable) -> None:
        """Set connection callback."""
        self._on_connected = callback
    
    def set_on_disconnected(self, callback: Callable) -> None:
        """Set disconnection callback."""
        self._on_disconnected = callback
    
    def set_on_participant_joined(self, callback: Callable) -> None:
        """Set participant joined callback."""
        self._on_participant_joined = callback
    
    def set_on_participant_left(self, callback: Callable) -> None:
        """Set participant left callback."""
        self._on_participant_left = callback
    
    def set_on_audio_received(self, callback: Callable) -> None:
        """Set audio received callback."""
        self._on_audio_received = callback
    
    def _set_state(self, state: MeetingState) -> None:
        """Update state and trigger callbacks."""
        old_state = self._state
        self._state = state
        
        if state == MeetingState.CONNECTED and old_state != MeetingState.CONNECTED:
            if self._on_connected:
                self._on_connected()
        elif state == MeetingState.DISCONNECTED and old_state != MeetingState.DISCONNECTED:
            if self._on_disconnected:
                self._on_disconnected()
        
        logger.debug(f"State changed: {old_state} -> {state}")
