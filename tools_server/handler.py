"""
IGPO Tool Server - Handler (Web Search Only)

Lightweight handler for web search tool calls.
Supports Serper API (Google) and Azure Bing Search.
"""

import os
import json
import time
import threading
import concurrent.futures
from typing import List, Dict, Any

from tools_server.search.search_api import web_search, local_retrieve


class Handler:
    """Web search / local retrieval handler with local caching."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = config.get('cache_dir', './cache/tool_cache')
        self.cache_ttl = config.get('cache_ttl_days', 7) * 24 * 60 * 60

        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(self.cache_dir, 'search_cache.json')
        self.retrieve_cache_file = os.path.join(self.cache_dir, 'retrieve_cache.json')
        self.search_cache = self._load_cache(self.cache_file)
        self.retrieve_cache = self._load_cache(self.retrieve_cache_file)
        self.cache_lock = threading.Lock()

    def _load_cache(self, cache_file: str) -> Dict:
        """Load a cache from file."""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_cache(self, cache_file: str, cache: Dict):
        """Save a cache to file."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Handler] Cache save error: {e}")
    
    def _is_cache_valid(self, entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return entry.get('timestamp', 0) and (time.time() - entry['timestamp']) < self.cache_ttl
    
    def handle_all(self, task_list: List[Dict]) -> List[Dict]:
        """Process all web search tasks."""
        if not task_list:
            return task_list
        
        print(f"[Handler] Processing {len(task_list)} tasks...")
        start_time = time.time()

        # Pre-fetch all queries per tool in parallel
        self._prefetch(task_list, 'web_search', self.search_cache,
                       web_search, self.cache_file)
        self._prefetch(task_list, 'local_retrieve', self.retrieve_cache,
                       local_retrieve, self.retrieve_cache_file)

        # Process each task
        for task in task_list:
            tool_call = task.get('tool_call', {})
            # Ensure tool_call is a dict
            if not isinstance(tool_call, dict):
                task['content'] = f"Invalid tool_call format: expected dict, got {type(tool_call).__name__}"
                continue

            tool_name = tool_call.get('name', '')
            arguments = tool_call.get('arguments', {})
            if not isinstance(arguments, dict):
                arguments = {}
            if tool_name == 'web_search':
                task['content'], task['retrieved_doc_ids'] = self._handle_web_search(arguments)
            elif tool_name == 'local_retrieve':
                task['content'], task['retrieved_doc_ids'] = self._handle_local_retrieve(arguments)
            else:
                task['content'] = f"Unknown tool: {tool_name}"
                task['retrieved_doc_ids'] = []

        print(f"[Handler] Completed in {time.time() - start_time:.2f}s")
        return task_list

    def _prefetch(self, task_list: List[Dict], tool_name: str, cache: Dict,
                  fetch_fn, cache_file: str):
        """Pre-fetch all queries for a given tool in parallel into its cache."""
        queries_to_fetch = set()

        for task in task_list:
            tool_call = task.get('tool_call', {})
            # Ensure tool_call is a dict
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get('name') != tool_name:
                continue

            arguments = tool_call.get('arguments', {})
            if not isinstance(arguments, dict):
                continue

            query_list = arguments.get('query', [])
            if not isinstance(query_list, list):
                query_list = [query_list] if query_list else []

            for query in query_list[:3]:
                if isinstance(query, str):
                    with self.cache_lock:
                        if query not in cache or not self._is_cache_valid(cache[query]):
                            queries_to_fetch.add(query)

        if not queries_to_fetch:
            return

        print(f"[Handler] Fetching {len(queries_to_fetch)} {tool_name} queries...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(fetch_fn, q, self.config): q for q in queries_to_fetch}
            for future in concurrent.futures.as_completed(futures):
                query = futures[future]
                try:
                    results = future.result(timeout=30)
                    with self.cache_lock:
                        cache[query] = {'timestamp': time.time(), 'results': results}
                except Exception as e:
                    print(f"[Handler] {tool_name} error for '{query}': {e}")

        self._save_cache(cache_file, cache)
    
    def _handle_web_search(self, arguments: Dict) -> List[Dict]:
        """Handle web_search tool call."""
        query_list = arguments.get('query', [])
        
        # Handle both single string and list of strings
        if isinstance(query_list, str):
            query_list = [query_list]
        elif not isinstance(query_list, list):
            return []
        
        results = []
        doc_ids = []
        for query in query_list[:3]:
            if not isinstance(query, str):
                continue

            # Get from cache or fetch
            with self.cache_lock:
                entry = self.search_cache.get(query, {})
                if self._is_cache_valid(entry):
                    search_results = entry.get('results', [])
                else:
                    search_results = web_search(query, self.config)
                    self.search_cache[query] = {'timestamp': time.time(), 'results': search_results}

            # Format results
            page_infos = []
            for r in search_results[:5]:
                url = r.get('link', r.get('url', ''))
                page_infos.append({
                    "title": r.get('title', ''),
                    "url": url,
                    "quick_summary": r.get('snippet', r.get('description', ''))
                })
                if url:
                    doc_ids.append(url)
            results.append({"search_query": query, "web_page_info_list": page_infos})

        return results, doc_ids

    def _handle_local_retrieve(self, arguments: Dict) -> List[Dict]:
        """Handle local_retrieve tool call against the hybrid retrieval server."""
        query_list = arguments.get('query', [])

        # Handle both single string and list of strings
        if isinstance(query_list, str):
            query_list = [query_list]
        elif not isinstance(query_list, list):
            return []

        results = []
        doc_ids = []
        for query in query_list[:3]:
            if not isinstance(query, str):
                continue

            # Get from cache or fetch
            with self.cache_lock:
                entry = self.retrieve_cache.get(query, {})
                if self._is_cache_valid(entry):
                    retrieved = entry.get('results', [])
                else:
                    retrieved = local_retrieve(query, self.config)
                    self.retrieve_cache[query] = {'timestamp': time.time(), 'results': retrieved}

            # Format results (hybrid server returns [{"document": {...}, "score": ...}, ...])
            docs = []
            for item in retrieved[:5]:
                doc = item.get('document', item) if isinstance(item, dict) else {}
                title, text = self._split_doc(doc)
                docs.append({
                    "title": title,
                    "content": text,
                    "score": item.get('score') if isinstance(item, dict) else None,
                })
                key = self._doc_key(doc)
                if key is not None:
                    doc_ids.append(key)

            results.append({"search_query": query, "retrieved_docs": docs})

        return results, doc_ids

    @staticmethod
    def _split_doc(doc: Dict) -> tuple:
        """Split a retrieved document into (title, text).

        The corpus packs title + body into 'contents' (title on the first line);
        fall back to explicit 'title'/'text' fields when 'contents' is absent.
        """
        contents = doc.get('contents', '')
        if contents:
            lines = contents.split("\n")
            return lines[0].strip('"'), "\n".join(lines[1:])
        return doc.get('title', ''), doc.get('text', '')

    @staticmethod
    def _doc_key(doc: Dict):
        """Stable dedup key for a retrieved document (for redundancy tracking).

        Mirrors the hybrid server's fusion key preference: docid first, then a
        content/title fallback so BM25 hits (which may lack an explicit id) still
        dedup consistently against dense hits.
        """
        for k in ('docid', 'doc_id', 'id'):
            if doc.get(k) is not None:
                return str(doc[k])
        text = doc.get('contents') or doc.get('text') or doc.get('title') or ''
        text = text.strip()
        return text[:200] if text else None
