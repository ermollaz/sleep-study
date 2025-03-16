import os
from openai import AsyncOpenAI
import asyncio
import logging
from typing import List, Optional, Union, Dict
import yaml
from pydantic import BaseModel
from aiolimiter import AsyncLimiter

def load_config(
    config_file:str, 
    field: Optional[str] = None
)-> Dict:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    if field:
        return config[field]
    return config


config = load_config("configs.yaml")
# Set your OpenAI API key for normal (non-Azure) connections.
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
) 

async def generate_embedding(
    text: str, 
    model: str = None
) -> List[float]:
    """
    Generate an embedding for the provided text using the new OpenAI API.
    
    Since the asynchronous embeddings method is no longer available,
    we run the synchronous call in a thread using asyncio.to_thread.
    
    Args:
        text (str): Input text to embed.
        model (str, optional): Model to use. Defaults to config["DEFAULT"]["embeddings"].
    
    Returns:
        list: Embedding vector.
    """
    try:
        model = model or config["DEFAULT"]["embeddings"]
        response = await client.embeddings.create(input=text, model="text-embedding-3-large")
        logging.debug("Received embedding")
        return response.data[0].embedding
    except KeyError as e:
        logging.error(f"Configuration error: Missing key {str(e)}")
        return [0] * config["EMBEDDING"][model]["vectordim"]
    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
        return [0] * config["EMBEDDING"][model]["vectordim"]
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return [0] * config["EMBEDDING"][model]["vectordim"]

async def batch_generate_embeddings(
    texts: List[str], 
    model: str = None
) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts asynchronously using a shared rate limiter.
    """
    model = model or config["DEFAULT"]["embeddings"]
    if model not in config["EMBEDDING"]:
        logging.error(f"Invalid model. Choose from {list(config['EMBEDDING'].keys())}")
        raise ValueError("Invalid embedding model")
    
    embedding_limiter = AsyncLimiter(
        max_rate=config["EMBEDDING"][model]["max-requests"],
        time_period=config["EMBEDDING"][model]["time-span"]
    )
    logging.info(f"Generating embeddings with limits: max_rate={embedding_limiter.max_rate}, time_period={embedding_limiter.time_period}")
    tasks = []

    try:
        for text in texts:
            async def task(text=text):
                async with embedding_limiter:
                    return await generate_embedding(text, model)
            tasks.append(task())
        results = await asyncio.gather(*tasks)
        logging.info(f"Embeddings generated for {len(texts)} texts")
        return results

    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

async def generate_completion(
    text: str, 
    model: str = None, 
    agent: str = None, 
    response_model: Optional[BaseModel] = None, 
    extra: Optional[str] = None
) -> Union[str, BaseModel]:
    """
    Generate a completion response for the provided text using the normal OpenAI API.
    
    Args:
        text (str): Input text for completion.
        model (str, optional): Model to use.
        agent (str, optional): Agent type.
        response_model (BaseModel, optional): Pydantic model to parse the response.
        extra (str, optional): Additional context used to construct the prompt.
    
    Returns:
        str or BaseModel: Generated completion or parsed response.
    """
    try:
        agent = agent or config["DEFAULT"]["agent"]

        if extra:
            if len(extra) > 200:
                prompt = [
                    {"role": "system", "content": config["PROMPTS"][agent]["system_message"].format(junk=extra)},
                    {"role": "user", "content": config["PROMPTS"][agent]["user_message"].format(input=text)},
                ]
            else:
                prompt = [
                    {"role": "system", "content": config["PROMPTS"][agent]["system_message"].format(title=extra)},
                    {"role": "user", "content": config["PROMPTS"][agent]["user_message"].format(input=text)},
                ]
        else:
            prompt = [
                {"role": "system", "content": config["PROMPTS"][agent]["system_message"]},
                {"role": "user", "content": config["PROMPTS"][agent]["user_message"].format(input=text)},
            ]

        model = model or config["DEFAULT"]['deployment']["completion"]
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0,
            max_tokens=4096,
        )
        if response_model:
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=prompt,
                temperature=0,
                max_tokens=4096,
                response_format = response_model
                )
            return response.choices[0].message.parsed
        else:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt,
                temperature=0,
                max_tokens=4096,
                )
            return response.choices[0].message.content

    except KeyError as e:
        logging.error(f"Configuration error: Missing key {str(e)}")
        return "Error Transforming Text"
    except asyncio.TimeoutError:
        logging.error("Request timeout: The request took too long to complete")
        return "Error Transforming Text"
    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
        return "Error Transforming Text"
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return "Error Transforming Text"

async def batch_generate_completions(
    texts: List[str], 
    model: str = None, 
    agent: str = None, 
    response_model: Optional[BaseModel] = None, 
    extras: Optional[List[str]] = None,
) -> List[Union[str, BaseModel]]:
    """
    Generate completions for a batch of texts asynchronously using a shared rate limiter.
    """
    model = model or config["DEFAULT"]['deployment']["completion"]
    if model not in config["COMPLETION"]:
        logging.error(f"Invalid model. Choose from {list(config['COMPLETION'].keys())}")
        raise ValueError("Invalid completion model")
    
    agent = agent or config["DEFAULT"]["agent"]
    if agent not in config["PROMPTS"]:
        logging.error(f"Invalid agent. Choose from {list(config['PROMPTS'].keys())}")
        raise ValueError("Invalid agent")
    
    completion_limiter = AsyncLimiter(
        max_rate=config["COMPLETION"][model]["max-requests"],
        time_period=config["COMPLETION"][model]["time-span"]
    )
    logging.info(f"Generating completions with limits: max_rate={completion_limiter.max_rate}, time_period={completion_limiter.time_period}")
    tasks = []

    try:
        # If extras is not provided, default to None for each text.
        if extras is None:
            extras = [None] * len(texts)
        for text, extra in zip(texts, extras):
            async def task(text=text, extra=extra):
                async with completion_limiter:
                    return await generate_completion(text, model, agent, response_model, extra)
            tasks.append(task())
        results = await asyncio.gather(*tasks)
        logging.info(f"Completions generated for {len(texts)} texts")
        return results

    except Exception as e:
        logging.error(f"Error generating completions: {e}")
        raise
