"""
LiteLLM-based OpenIE for HippoRAG - uses litellm instead of vllm.
Works on CPU with OpenAI-compatible APIs.
"""
import json
from typing import Dict, Tuple
import os
import time

import litellm
from copy import deepcopy

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager

logger = get_logger(__name__)


class LiteLLMOffline:
    """LLM using litellm for OpenAI-compatible APIs (works on CPU)."""

    def __init__(self, global_config):
        self.global_config = global_config
        self.llm_name = global_config.llm_name
        self.llm_base_url = getattr(global_config, 'llm_base_url', None)
        self.temperature = getattr(global_config, 'temperature', 0.0)

        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"{self.llm_name.replace('/', '_')}_litellm_cache.json")

        self.max_retries = 3
        self.retry_delay = 1.0

        logger.info(f"[LiteLLM] Initialized with model: {self.llm_name}, base_url: {self.llm_base_url}")

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: Dict):
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

    def _get_cache_key(self, messages: list, max_tokens: int) -> str:
        import hashlib
        msg_str = json.dumps(messages, sort_keys=True)
        key_str = f"{self.llm_name}:{max_tokens}:{msg_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _call_llm(self, messages: list, max_tokens: int = 2048) -> Tuple[str, dict]:
        cache = self._load_cache()
        cache_key = self._get_cache_key(messages, max_tokens)

        if cache_key in cache:
            logger.debug(f"[LiteLLM] Cache hit for key: {cache_key[:16]}...")
            return cache[cache_key]['response'], cache[cache_key]['metadata']

        params = {
            'model': self.llm_name,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': self.temperature,
        }
        if self.llm_base_url:
            params['api_base'] = self.llm_base_url

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(**params)
                text = response.choices[0].message.content
                metadata = {
                    'prompt_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                    'completion_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                }

                cache[cache_key] = {'response': text, 'metadata': metadata}
                self._save_cache(cache)

                return text, metadata
            except Exception as e:
                logger.warning(f"[LiteLLM] Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def infer(self, messages: list, max_tokens: int = 2048) -> Tuple[str, dict]:
        return self._call_llm(messages, max_tokens)

    def batch_infer(self, messages_list: list, max_tokens: int = 2048, json_template: str = None) -> Tuple[list, dict]:
        all_responses = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, messages in enumerate(messages_list):
            try:
                response, metadata = self._call_llm(messages, max_tokens)
                all_responses.append(response)
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
            except Exception as e:
                logger.error(f"[LiteLLM] batch_infer failed for message {i}: {e}")
                all_responses.append("")

        metadata = {
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'num_request': len(messages_list),
        }
        return all_responses, metadata


class LiteLLMOfflineOpenIE(OpenIE):
    """OpenIE implementation using LiteLLM instead of vllm."""

    def __init__(self, global_config):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = LiteLLMOffline(global_config)

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE using LiteLLM, including NER and triple extraction.
        """
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_input_messages = [
            self.prompt_template_manager.render(name='ner', passage=p)
            for p in chunk_passages.values()
        ]

        ner_output, ner_metadata = self.llm_model.batch_infer(ner_input_messages, json_template='ner', max_tokens=512)

        triple_extract_input_messages = [
            self.prompt_template_manager.render(
                name='triple_extraction',
                passage=passage,
                named_entity_json=named_entities
            )
            for passage, named_entities in zip(chunk_passages.values(), ner_output)
        ]
        triple_output, triple_metadata = self.llm_model.batch_infer(triple_extract_input_messages, json_template='triples', max_tokens=2048)

        ner_raw_outputs = []
        for idx, ner_output_instance in enumerate(ner_output):
            chunk_id = list(chunks.keys())[idx]
            response = ner_output_instance
            try:
                if response.strip().startswith("```json"):
                    response = response.strip()[7:]
                if response.strip().endswith("```"):
                    response = response.strip()[:-3]
                ner_json = json.loads(response.strip())
                if isinstance(ner_json, dict) and 'entities' in ner_json:
                    ner_json = ner_json['entities']
                ner_raw_outputs.append({
                    'chunk_id': chunk_id,
                    'ner_result': ner_json
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse NER response: {e}")
                ner_raw_outputs.append({
                    'chunk_id': chunk_id,
                    'ner_result': []
                })

        triple_raw_outputs = []
        for idx, triple_output_instance in enumerate(triple_output):
            chunk_id = list(chunks.keys())[idx]
            response = triple_output_instance
            try:
                if response.strip().startswith("```json"):
                    response = response.strip()[7:]
                if response.strip().endswith("```"):
                    response = response.strip()[:-3]
                triple_json = json.loads(response.strip())
                if isinstance(triple_json, dict) and 'triples' in triple_json:
                    triple_json = triple_json['triples']
                triple_raw_outputs.append({
                    'chunk_id': chunk_id,
                    'triple_result': triple_json
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse Triple response: {e}")
                triple_raw_outputs.append({
                    'chunk_id': chunk_id,
                    'triple_result': []
                })

        ner_results_dict = {item['chunk_id']: item['ner_result'] for item in ner_raw_outputs}
        triple_results_dict = {item['chunk_id']: item['triple_result'] for item in triple_raw_outputs}

        return ner_results_dict, triple_results_dict