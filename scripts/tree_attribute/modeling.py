from typing import Any, Optional, List
import time
import asyncio
import aiohttp
from typing import Any, Annotated, Optional, List
import traceback
import random

class vllm_server_model:
  """Class for storing any single language model."""
  def __init__(
      self,
      temperature: float = 0.5,
      max_tokens: int = 2048,
      api_url: str = "http://localhost:2341/v1/chat/completions",
      model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
  ) -> None:
    """Initializes a model."""
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.api_url = api_url.split(";")
  
  # Call ChatGPT with the given prompt, asynchronously.
  async def call_chatgpt_async(self, session, prompt: str, temperature, max_tokens, max_attempts, retry_interval, sys_prompt=None):
    if sys_prompt is None:
        payload = {
            'model': self.model_name,
            'messages': [
                {"role": "user", "content": prompt}
            ],
            "temperature": str(temperature or self.temperature),
            "max_tokens": str(max_tokens or self.max_tokens)
        }
    else:
        payload = {
            'model': self.model_name,
            'messages': [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": str(temperature or self.temperature),
            "max_tokens": str(max_tokens or self.max_tokens)
        }
    result, num_attempts = "", 0 
    while not result and num_attempts < max_attempts:
      try:
        async with session.post(
          url=random.choice(self.api_url),
          headers={"Content-Type": "application/json"},
          json=payload,
          timeout=aiohttp.ClientTimeout(total=90 * 60),
        ) as response:
          response = await response.json()
          if "choices" not in response:
            raise ValueError(f"Invalid response: {response}")
          result = response['choices'][0]['message']['content']
      except Exception as e:
        print(f"Encounter the following error when calling api: {repr(e)}")
        traceback.print_exc()
        time.sleep(retry_interval)
        retry_interval = retry_interval * 2
      num_attempts += 1
    
    return result

  # Call chatGPT for all the given prompts in parallel.
  async def call_chatgpt_bulk(self, prompts, temperature, max_tokens, max_attempts, retry_interval, sys_prompt=None):
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as tg:
      responses = [tg.create_task(self.call_chatgpt_async(session, prompt, temperature, max_tokens, max_attempts, retry_interval, sys_prompt)) for prompt in prompts]
    return responses

  def generate_batched(
      self,
      prompt_batch: List[str],
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      max_attempts: int = 30,
      retry_interval: int = 5,
      sys_prompt: str = None
  ) -> List[str]:
    """Generates a response to a prompt."""
    prompt_batch = [prompt for prompt in prompt_batch]
    gen_temp = temperature or self.temperature
    gen_max_tokens = max_tokens or self.max_tokens
    response, num_attempts = '', 0

    response = asyncio.run(self.call_chatgpt_bulk(prompt_batch, gen_temp, gen_max_tokens, max_attempts, retry_interval, sys_prompt))
    response = [tmp.result().strip() for tmp in response]
    print(response)
    return response
  
  def generate(
      self,
      prompts: str | list,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
  ) -> str:
    """Generates a response to a prompt."""
    return self.generate_batched([prompts], temperature, max_tokens)[0]
