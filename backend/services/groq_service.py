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
    error logging, transient retries, and automatic model fallback on rate limits.
    """
    FALLBACK_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.primary_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        self.client = None
        self.last_error_message = ""
        
        # Only instantiate Groq client if key is set and not default placeholder
        if self.api_key and self.api_key != "PASTE_MY_GROQ_KEY_HERE":
            try:
                from groq import AsyncGroq
                self.client = AsyncGroq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.last_error_message = f"Client init error: {e}"

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
                "message": "GROQ_API_KEY is missing or set to placeholder in environment"
            }
        
        models_to_test = [self.primary_model] + [m for m in self.FALLBACK_MODELS if m != self.primary_model]
        
        for m in models_to_test:
            try:
                response = await self.client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": "Respond with 'OK'"}],
                    max_tokens=10,
                    temperature=0.1
                )
                text = response.choices[0].message.content.strip()
                return {
                    "status": "connected",
                    "message": "Groq connection successful.",
                    "model": m,
                    "response": text
                }
            except Exception as e:
                logger.warning(f"Groq health check failed for model {m}: {e}")
                self.last_error_message = str(e)

        return {
            "status": "error",
            "message": f"Groq API call failed across all models: {self.last_error_message}"
        }

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.4,
        max_retries: int = 1
    ) -> Optional[str]:
        """
        Send prompt completion request to Groq with model fallback logic and error logging.
        """
        if not self.is_configured():
            self.last_error_message = "GROQ_API_KEY environment variable is not configured or missing."
            logger.warning(self.last_error_message)
            return None

        models_queue = [self.primary_model] + [m for m in self.FALLBACK_MODELS if m != self.primary_model]
        extra_args = {}
        if json_mode:
            extra_args["response_format"] = {"type": "json_object"}

        for model_name in models_queue:
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Attempting Groq completion using model: {model_name} (attempt {attempt+1})")
                    response = await self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        timeout=20.0,
                        **extra_args
                    )
                    content = response.choices[0].message.content
                    if content and content.strip():
                        self.last_error_message = ""
                        return content
                except Exception as e:
                    err_str = str(e)
                    self.last_error_message = f"Model {model_name} error: {err_str}"
                    logger.warning(f"Groq completion error for model {model_name} (attempt {attempt+1}): {err_str}")
                    if "429" in err_str or "rate_limit" in err_str.lower():
                        # Switch immediately to next fallback model on 429 rate limits
                        logger.info(f"Rate limit hit on {model_name}. Switching to next fallback model...")
                        break
                    if attempt < max_retries:
                        await asyncio.sleep(0.5)

        logger.error(f"All Groq completion attempts failed. Last error: {self.last_error_message}")
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
            # Try without json_mode if json_mode format wasn't supported
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
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            else:
                self.last_error_message = f"Groq JSON response is not a dict: {cleaned}"
                logger.error(self.last_error_message)
                return None
        except json.JSONDecodeError as err:
            self.last_error_message = f"Failed to parse JSON from Groq response: {err}\nRaw text: {raw_text}"
            logger.error(self.last_error_message)
            return None

# Singleton instance
groq_service = GroqService()
