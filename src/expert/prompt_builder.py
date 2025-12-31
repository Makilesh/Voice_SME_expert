"""Builds prompts for expert LLM queries."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from string import Template

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""
    name: str
    template: str
    required_vars: List[str]
    optional_vars: List[str] = None
    
    def __post_init__(self):
        self.optional_vars = self.optional_vars or []


class PromptBuilder:
    """
    Builds prompts for expert LLM queries.
    """
    
    # Default templates
    DEFAULT_TEMPLATES = {
        "expert_answer": PromptTemplate(
            name="expert_answer",
            template="""You are a Subject Matter Expert assistant participating in a meeting.

CONTEXT FROM KNOWLEDGE BASE:
$knowledge_context

RECENT CONVERSATION:
$conversation_context

CURRENT SPEAKER: $speaker
QUESTION: $question

Provide a helpful, accurate, and concise answer based on the knowledge base and conversation context.
If you don't have enough information to answer confidently, say so.
Keep your response natural and conversational, suitable for voice output.""",
            required_vars=["question"],
            optional_vars=["knowledge_context", "conversation_context", "speaker"]
        ),
        
        "clarification": PromptTemplate(
            name="clarification",
            template="""Based on the conversation, ask a clarifying question to better understand what is being discussed.

RECENT CONVERSATION:
$conversation_context

TOPIC: $topic

Generate a brief, relevant clarifying question.""",
            required_vars=["conversation_context"],
            optional_vars=["topic"]
        ),
        
        "summary": PromptTemplate(
            name="summary",
            template="""Summarize the following meeting conversation:

$conversation_context

Provide a concise summary covering:
1. Main topics discussed
2. Key decisions made
3. Action items identified

Keep the summary brief and actionable.""",
            required_vars=["conversation_context"]
        ),
        
        "fact_check": PromptTemplate(
            name="fact_check",
            template="""Verify the following statement using the knowledge base:

STATEMENT: $statement

RELEVANT KNOWLEDGE:
$knowledge_context

Is this statement accurate? Provide a brief assessment.""",
            required_vars=["statement", "knowledge_context"]
        )
    }
    
    def __init__(
        self,
        system_prompt: str = "",
        custom_templates: Optional[Dict[str, PromptTemplate]] = None
    ):
        """
        Initializes prompt builder.
        
        Parameters:
            system_prompt: Base system prompt
            custom_templates: Additional custom templates
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Merge templates
        self.templates = dict(self.DEFAULT_TEMPLATES)
        if custom_templates:
            self.templates.update(custom_templates)
        
        logger.info(f"PromptBuilder initialized with {len(self.templates)} templates")
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are an expert Subject Matter Expert (SME) assistant designed to help during meetings.
Your role is to:
1. Listen to the conversation and provide relevant information when asked
2. Answer questions accurately based on your knowledge base
3. Clarify complex topics in simple terms
4. Be concise - your responses will be spoken aloud

Key behaviors:
- Be helpful and professional
- Acknowledge when you don't know something
- Keep responses brief for voice output (under 100 words when possible)
- Reference specific sources when available"""
    
    def build_prompt(
        self,
        template_name: str,
        variables: Dict[str, str]
    ) -> str:
        """
        Builds prompt from template.
        
        Parameters:
            template_name: Name of template to use
            variables: Variable values to substitute
        
        Returns:
            Built prompt string
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        
        # Check required variables
        missing = [v for v in template.required_vars if v not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Set defaults for optional variables
        for var in template.optional_vars:
            if var not in variables:
                variables[var] = ""
        
        # Substitute
        try:
            prompt = Template(template.template).safe_substitute(variables)
            return prompt.strip()
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise
    
    def build_chat_messages(
        self,
        template_name: str,
        variables: Dict[str, str],
        include_system: bool = True
    ) -> List[Dict]:
        """
        Builds chat messages for LLM.
        
        Parameters:
            template_name: Template name
            variables: Template variables
            include_system: Include system message
        
        Returns:
            List of message dicts
        """
        messages = []
        
        # System message
        if include_system:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # User message with built prompt
        user_prompt = self.build_prompt(template_name, variables)
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        return messages
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add or update a template."""
        self.templates[template.name] = template
        logger.debug(f"Template added: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())
    
    def set_system_prompt(self, prompt: str) -> None:
        """Update system prompt."""
        self.system_prompt = prompt
