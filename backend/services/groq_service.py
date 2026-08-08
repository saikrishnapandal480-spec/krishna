import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("groq_service")

class GroqService:
    """
    Dedicated service for managing interactions with the Groq API.
    Handles client initialization, prompt completion, structured JSON extraction,
    error logging, and transient retries.
    """
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        self.client = None
        
        # Only instantiate Groq client if key is set and not default placeholder
        if self.api_key and self.api_key != "PASTE_MY_GROQ_KEY_HERE":
            try:
                from groq import AsyncGroq
                self.client = AsyncGroq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")

    def is_configured(self) -> bool:
        """Check if Groq API key is present and client initialized."""
        return self.client is not None and bool(self.api_key) and self.api_key != "PASTE_MY_GROQ_KEY_HERE"

    async def health_check(self) -> Dict[str, Any]:
        """
        Backend health/test function. Verifies Groq API Key, client, and completion.
        Never prints or exposes actual API key.
        """
        if not self.is_configured():
            return {
                "status": "unconfigured",
                "message": "GROQ_API_KEY is missing or set to placeholder in .env"
            }
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Respond with 'OK'"}],
                max_tokens=10,
                temperature=0.1
            )
            text = response.choices[0].message.content.strip()
            return {
                "status": "connected",
                "message": "Groq connection successful.",
                "model": self.model,
                "response": text
            }
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            return {
                "status": "error",
                "message": f"Groq API call failed: {str(e)}"
            }

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.5,
        max_retries: int = 2
    ) -> Optional[str]:
        """
        Send prompt completion request to Groq with retry logic and error handling.
        """
        if not self.is_configured():
            logger.warning("Groq API client is not configured. Returning None.")
            return None

        extra_args = {}
        if json_mode:
            extra_args["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    timeout=25.0,
                    **extra_args
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq API call attempt {attempt+1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.error("Max retries reached for Groq completion.")
                    return None

    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.4
    ) -> Optional[Dict[str, Any]]:
        """
        Helper method to get and parse JSON responses from Groq.
        Robustly strips markdown fences if returned and parses valid dict.
        """
        raw_text = await self.generate_completion(messages, json_mode=True, temperature=temperature)
        if not raw_text:
            # Try without json_mode if json_mode wasn't supported
            raw_text = await self.generate_completion(messages, json_mode=False, temperature=temperature)

        if not raw_text:
            return None

        # Clean potential markdown formatting ```json ... ```
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.error(f"Failed to parse JSON from Groq response: {err}\nRaw text: {raw_text}")
            return None

# Singleton instance
groq_service = GroqService()
