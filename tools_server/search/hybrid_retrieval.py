"""
Hybrid Retrieval Fusion - 混合检索融合策略

增强 Search-R1 的召回阶段，通过 RRF (Reciprocal Rank Fusion) 算法
融合 BM25 稀疏检索和 Dense 向量检索的结果，提升召回质量。

设计思路：
1. 保持与原有 retrieval_server.py 的接口兼容
2. 新增 HybridRetriever 类，支持双路召回 + RRF 融合
3. 可通过配置选择不同的融合策略：RRF、Score-based Weighted、Convex Combination

论文参考：
- "Reciprocal Rank Fusion for Multiple Retrieval Modalities" (IRNLP, 2023)
- "Sparse, Dense, and Learned Retrieval for RAG" (Google Research, 2024)
"""

import json
import argparse
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import faiss
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm


# ===================== 基础工具函数 =====================

def load_corpus(corpus_path: str):
    """加载语料库"""
    import datasets
    corpus = datasets.load_dataset('json', data_files=corpus_path, split="train", num_proc=4)
    return corpus


def read_jsonl(file_path: str) -> List[dict]:
    """读取 jsonl 文件"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs: List[int]) -> List[dict]:
    """根据索引加载文档"""
    return [corpus[int(idx)] for idx in doc_idxs]


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    """池化操作"""
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError(f"Pooling method {pooling_method} not implemented!")


def load_model(model_path: str, use_fp16: bool = False):
    """加载编码器模型"""
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


# ===================== 配置类 =====================

@dataclass
class HybridRetrievalConfig:
    """混合检索配置"""
    # BM25 配置
    bm25_index_path: str = "./index/bm25"
    bm25_contain_doc: bool = True  # Lucene 索引是否用 --storeRaw 存了原文
    # Dense 配置
    dense_index_path: str = "./index/e5_Flat.index"
    # 共享配置
    corpus_path: str = "./data/corpus.jsonl"
    topk: int = 10
    # 融合配置
    fusion_method: str = "rrf"  # "rrf" | "weighted" | "convex"
    rrf_k: float = 60.0  # RRF 算法参数，越大越平滑
    dense_weight: float = 0.5  # Dense 检索权重 (0-1)
    # Dense 编码器配置
    retrieval_model_path: str = "intfloat/e5-base-v2"
    retrieval_pooling_method: str = "mean"
    retrieval_query_max_length: int = 256
    retrieval_use_fp16: bool = True
    retrieval_batch_size: int = 128
    faiss_gpu: bool = True
    # FAISS GPU 加载：索引是否用 fp16 存(省一半显存,但转换瞬间需 fp32+fp16 两份)。
    # 若 fp16 转换 OOM,可关掉它用 fp32(64GB 直接上 80GB 卡,不走转换)。
    faiss_use_float16: bool = True
    faiss_temp_mem_mb: int = 512  # 限制 FAISS GPU 临时显存池,避免默认大量预留
    # Rerank 配置（可选阶段，接在 RRF 融合之后做精排）
    enable_rerank: bool = False
    rerank_model_path: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    rerank_batch_size: int = 32
    rerank_candidate_factor: int = 3  # 召回 topk*factor 个候选送 rerank
    rerank_max_length: int = 512


# ===================== Encoder =====================

class Encoder:
    """文本编码器"""
    def __init__(self, model_name: str, model_path: str, pooling_method: str,
                 max_length: int, use_fp16: bool):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model, self.tokenizer = load_model(model_path, use_fp16)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # E5 模型需要添加前缀
        if "e5" in self.model_name.lower():
            query_list = [f"query: {q}" for q in query_list] if is_query else [f"passage: {q}" for q in query_list]

        # BGE 模型需要添加前缀
        if "bge" in self.model_name.lower() and is_query:
            query_list = [f"Represent this sentence for searching relevant passages: {q}" for q in query_list]

        inputs = self.tokenizer(query_list, max_length=self.max_length, padding=True,
                                truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0], 1), dtype=torch.long).to(inputs['input_ids'].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state,
                               inputs['attention_mask'], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        del inputs, output
        torch.cuda.empty_cache()
        return query_emb


# ===================== BM25 Retriever =====================

class BM25Retriever:
    """BM25 稀疏检索器"""
    def __init__(self, config: HybridRetrievalConfig):
        from pyserini.search.lucene import LuceneSearcher
        self.config = config
        self.searcher = LuceneSearcher(config.bm25_index_path)
        self.contain_doc = config.bm25_contain_doc
        # 仅在索引未存原文时才需要 corpus 兜底
        self.corpus = None if self.contain_doc else load_corpus(config.corpus_path)

    def search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.config.topk

        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            return ([], []) if return_score else []

        scores = [hit.score for hit in hits]
        hits = hits[:num]

        if self.contain_doc:
            results = []
            for hit in hits:
                content = json.loads(self.searcher.doc(hit.docid).raw())['contents']
                results.append({
                    'title': content.split("\n")[0].strip('"'),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content,
                    # 关键：带上 docid，否则 RRF 无法与 Dense 结果对齐去重
                    'docid': str(hit.docid),
                })
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])
            for hit, r in zip(hits, results):
                r['docid'] = str(hit.docid)

        return (results, scores) if return_score else results

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results, scores = [], []
        for query in query_list:
            r, s = self.search(query, num, True)
            results.append(r)
            scores.append(s)
        return (results, scores) if return_score else results


# ===================== Dense Retriever =====================

class DenseRetriever:
    """Dense 向量检索器"""
    def __init__(self, config: HybridRetrievalConfig):
        self.config = config
        self.index = faiss.read_index(config.dense_index_path)
        if config.faiss_gpu:
            # 单卡克隆(dedicated 一张卡做索引时用):显式 resource + 限制临时显存,
            # 避免 index_cpu_to_all_gpus 默认大量预留临时显存导致 OOM。
            # 要用多卡分片时,改回 index_cpu_to_all_gpus 即可。
            res = faiss.StandardGpuResources()
            res.setTempMemory(config.faiss_temp_mem_mb * 1024 * 1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = config.faiss_use_float16
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
            # 持有 resource 引用,防止被 GC 回收
            self._gpu_res = res

        self.corpus = load_corpus(config.corpus_path)
        self.encoder = Encoder(
            model_name="e5",
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )
        self.batch_size = config.retrieval_batch_size

    def search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.config.topk

        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        results = load_docs(self.corpus, idxs[0])
        for i, r in enumerate(results):
            r['docid'] = int(idxs[0][i])

        return (results, scores[0].tolist()) if return_score else results

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.config.topk

        results, scores = [], []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Dense retrieval"):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            for i, r in enumerate(batch_results):
                r['docid'] = int(flat_idxs[i])

            batch_results = [batch_results[i*num:(i+1)*num] for i in range(len(batch_idxs))]
            scores.extend(batch_scores)
            results.extend(batch_results)

            del batch_emb, batch_scores, batch_idxs
            torch.cuda.empty_cache()

        return (results, scores) if return_score else results


# ===================== 融合策略 =====================

def _fusion_doc_key(doc: dict) -> str:
    """统一的文档去重 key。

    归一化为 str：BM25 的 docid 是字符串、Dense 是 int，str() 后同一篇能对齐，
    保证 RRF / 加权 / 凸组合真正把多路命中的同一篇文档合并。
    """
    docid = doc.get('docid')
    if docid is not None:
        return str(docid)
    return str(doc.get('contents') or doc.get('text') or doc.get('title') or id(doc))


class FusionStrategy:
    """检索结果融合策略基类"""

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        """
        融合多路检索结果
        Args:
            retrieval_results: List of (documents, scores) tuples from different retrievers
        Returns:
            Fused list of (document, fused_score) sorted by score descending
        """
        raise NotImplementedError


class RRFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) - 倒数排序融合

    核心思想：对每个检索器返回的结果，按排名赋分 (1/rank)，最后累加。
    优势：对单个检索器的排序质量不敏感，泛化能力强。

    公式: RRF(d) = Σ 1/(k + rank_i(d))

    参考：F.以北等, "Reciprocal Rank Fusion for Multiple Retrieval Modalities", 2023
    """

    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        doc_scores = defaultdict(float)
        doc_info = {}

        for docs, _ in retrieval_results:
            for rank, doc in enumerate(docs, start=1):
                doc_key = _fusion_doc_key(doc)
                doc_scores[doc_key] += 1.0 / (self.k + rank)
                doc_info[doc_key] = doc

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_info[k], s) for k, s in sorted_items[:kwargs.get('topk', 10)]]


class ScoreWeightedFusion(FusionStrategy):
    """
    Score-based Weighted Fusion - 基于分数的加权融合

    核心思想：将不同检索器的分数归一化后加权求和。
    适用于已知各检索器质量差异的场景。

    公式: score(d) = Σ w_i * norm(score_i(d))
    """

    def __init__(self, weights: List[float] = None):
        self.weights = weights or [0.5, 0.5]

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        assert len(self.weights) == len(retrieval_results), "权重数量必须与检索器数量一致"

        doc_scores = defaultdict(float)
        doc_info = {}

        for retriever_idx, (docs, scores) in enumerate(retrieval_results):
            if not scores:
                continue

            # Min-Max 归一化
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s != min_s else 1.0

            for doc, score in zip(docs, scores):
                norm_score = (score - min_s) / range_s
                doc_key = _fusion_doc_key(doc)
                doc_scores[doc_key] += self.weights[retriever_idx] * norm_score
                doc_info[doc_key] = doc

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_info.get(k), s) for k, s in sorted_items[:kwargs.get('topk', 10)]]


class ConvexCombinationFusion(FusionStrategy):
    """
    Convex Combination Fusion - 凸组合融合

    核心思想：在分数空间中做线性插值，假设不同检索器的分数分布相似。
    适用于检索器质量相近、且分数分布可比的场景。
    """

    def __init__(self, weights: List[float] = None):
        self.weights = weights or [0.5, 0.5]

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        assert len(self.weights) == len(retrieval_results), "权重数量必须与检索器数量一致"

        doc_scores = defaultdict(float)
        doc_info = {}

        for retriever_idx, (docs, scores) in enumerate(retrieval_results):
            if not scores:
                continue

            for doc, score in zip(docs, scores):
                doc_key = _fusion_doc_key(doc)
                doc_scores[doc_key] += self.weights[retriever_idx] * score
                doc_info[doc_key] = doc

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_info.get(k), s) for k, s in sorted_items[:kwargs.get('topk', 10)]]


def get_fusion_strategy(method: str, **kwargs) -> FusionStrategy:
    """获取融合策略实例"""
    strategies = {
        "rrf": RRFusion(k=kwargs.get("rrf_k", 60.0)),
        "weighted": ScoreWeightedFusion(weights=kwargs.get("weights")),
        "convex": ConvexCombinationFusion(weights=kwargs.get("weights")),
    }
    if method not in strategies:
        raise ValueError(f"Unknown fusion method: {method}. Available: {list(strategies.keys())}")
    return strategies[method]


# ===================== Rerank（精排）=====================

class CrossEncoderReranker:
    """
    Cross-Encoder 精排器

    职责：在 RRF 融合产出的 topk*factor 个候选上，用 (query, doc) 联合编码的
    cross-encoder 重新打分，输出最终 topk。

    为什么要精排：
      - BM25 / Dense 都是"双塔"模型——query 和 doc 各自独立编码，损失了细粒度交互
      - cross-encoder 把 (query, doc) 拼起来过一次 transformer，对相关性更敏感
      - 代价是慢——所以只在融合的小候选集 (topk*3 ~ topk*5) 上跑

    与 rerank_server.py 的关系：
      - rerank_server.py 是独立的 HTTP 服务，需单独部署
      - 这里把 reranker 内嵌到 HybridRetriever 里，单服务即可端到端"召回→融合→精排"

    论文背景：cross-encoder 始于 monoBERT (Nogueira & Cho, 2019)
    """

    def __init__(self, model_path: str, batch_size: int = 32, max_length: int = 512,
                 device: str = None):
        from sentence_transformers import CrossEncoder
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # max_length 通过 model 参数透传给底层 tokenizer
        self.model = CrossEncoder(model_path, max_length=max_length, device=self.device)

    @staticmethod
    def _doc_to_text(doc: dict) -> str:
        """将检索结果中的 doc dict 拼成 cross-encoder 输入文本"""
        content = doc.get('contents', '')
        if not content:
            # 兼容 BM25 / Dense 检索结果里没有 contents 的情况
            title = doc.get('title', '')
            text = doc.get('text', '')
            content = f"{title}\n{text}" if title else text
        # 截断到合理长度（cross-encoder tokenizer 会再次截断）
        return content[:2000]

    def rerank(self, query: str, docs: List[dict], topk: int) -> Tuple[List[dict], List[float]]:
        """
        单查询精排
        Args:
            query: 用户查询
            docs:  RRF 融合后的候选文档列表
            topk:  最终保留数量
        Returns:
            (重排后的 docs, 重排分数)
        """
        if not docs:
            return [], []

        pairs = [(query, self._doc_to_text(d)) for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif isinstance(scores, np.ndarray):
            scores = scores.tolist()

        # 按分数降序排序
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        reranked_docs = [docs[i] for i in order[:topk]]
        reranked_scores = [scores[i] for i in order[:topk]]
        return reranked_docs, reranked_scores

    def batch_rerank(self, queries: List[str], docs_per_query: List[List[dict]],
                     topk: int) -> Tuple[List[List[dict]], List[List[float]]]:
        """批量精排：把所有 (q, d) 对拍平做一次大 batch，效率最佳"""
        if len(queries) != len(docs_per_query):
            raise ValueError("queries 数量必须等于 docs_per_query 数量")

        # 拍平所有 pair，记录每条 query 对应的 pair 区间
        all_pairs = []
        offsets = []  # 第 i 个 query 的 pair 在 all_pairs 中的 [start, end)
        cursor = 0
        for q, docs in zip(queries, docs_per_query):
            start = cursor
            for d in docs:
                all_pairs.append((q, self._doc_to_text(d)))
            cursor += len(docs)
            offsets.append((start, cursor))

        if not all_pairs:
            return [[] for _ in queries], [[] for _ in queries]

        all_scores = self.model.predict(all_pairs, batch_size=self.batch_size,
                                        show_progress_bar=False)
        if hasattr(all_scores, 'tolist'):
            all_scores = all_scores.tolist()
        elif isinstance(all_scores, np.ndarray):
            all_scores = all_scores.tolist()

        out_docs, out_scores = [], []
        for (start, end), docs in zip(offsets, docs_per_query):
            scores = all_scores[start:end]
            order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
            out_docs.append([docs[i] for i in order[:topk]])
            out_scores.append([scores[i] for i in order[:topk]])
        return out_docs, out_scores


# ===================== Hybrid Retriever =====================

class HybridRetriever:
    """
    混合检索器 - 融合 BM25 和 Dense 双路召回

    相比单一检索器的优势：
    1. BM25 对关键词匹配敏感，擅长实体、术语检索
    2. Dense 对语义相似度敏感，擅长同义词、语义扩展
    3. 融合后可覆盖更多查询类型

    使用方式：
        retriever = HybridRetriever(config)
        results = retriever.search("Python 教程", topk=5)
    """

    def __init__(self, config: HybridRetrievalConfig):
        self.config = config
        self.bm25_retriever = BM25Retriever(config)
        self.dense_retriever = DenseRetriever(config)
        self.fusion_strategy = get_fusion_strategy(
            config.fusion_method,
            rrf_k=config.rrf_k,
            weights=[1 - config.dense_weight, config.dense_weight]
        )

        # 可选 rerank 阶段
        self.reranker: Optional[CrossEncoderReranker] = None
        if config.enable_rerank:
            print(f"[HybridRetriever] Loading cross-encoder reranker: {config.rerank_model_path}")
            self.reranker = CrossEncoderReranker(
                model_path=config.rerank_model_path,
                batch_size=config.rerank_batch_size,
                max_length=config.rerank_max_length,
            )
            print(f"[HybridRetriever] Rerank ENABLED "
                  f"(candidate_factor={config.rerank_candidate_factor})")

    def search(self, query: str, num: int = None, return_score: bool = False):
        """单查询混合检索：BM25 + Dense → RRF 融合 → (可选) cross-encoder 精排"""
        if num is None:
            num = self.config.topk

        # 召回阶段：rerank 启用时召回更多候选给精排
        recall_k = num * (self.config.rerank_candidate_factor if self.reranker else 2)

        bm25_results, bm25_scores = self.bm25_retriever.search(query, recall_k, True)
        dense_results, dense_scores = self.dense_retriever.search(query, recall_k, True)

        # 融合阶段
        fuse_topk = recall_k if self.reranker else num
        fused = self.fusion_strategy.fuse(
            [(bm25_results, bm25_scores), (dense_results, dense_scores)],
            topk=fuse_topk
        )
        fused_docs = [d for d, _ in fused]
        fused_scores = [s for _, s in fused]

        # 精排阶段（可选）
        if self.reranker:
            fused_docs, fused_scores = self.reranker.rerank(query, fused_docs, topk=num)

        if return_score:
            return fused_docs, fused_scores
        else:
            return fused_docs

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        """批量查询混合检索"""
        if num is None:
            num = self.config.topk

        recall_k = num * (self.config.rerank_candidate_factor if self.reranker else 2)

        bm25_results, bm25_scores = self.bm25_retriever.batch_search(query_list, recall_k, True)
        dense_results, dense_scores = self.dense_retriever.batch_search(query_list, recall_k, True)

        # 逐条融合
        fuse_topk = recall_k if self.reranker else num
        fused_results, fused_scores = [], []
        for i in range(len(query_list)):
            fused = self.fusion_strategy.fuse(
                [(bm25_results[i], bm25_scores[i]), (dense_results[i], dense_scores[i])],
                topk=fuse_topk
            )
            fused_results.append([d for d, _ in fused])
            fused_scores.append([s for _, s in fused])

        # 批量精排（可选）—— 一次大 batch 比逐条快
        if self.reranker:
            fused_results, fused_scores = self.reranker.batch_rerank(
                query_list, fused_results, topk=num
            )

        return (fused_results, fused_scores) if return_score else fused_results


# ===================== FastAPI 服务 =====================

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI(title="Hybrid Retrieval Fusion API")


def _passages2string(retrieval_result: List[dict]) -> str:
    """将检索结果格式化为字符串"""
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item.get('contents', '')
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    混合检索接口，融合 BM25 和 Dense 双路召回

    请求格式：
    {
        "queries": ["query1", "query2"],
        "topk": 3,
        "return_scores": true
    }

    返回格式：
    {
        "result": [
            [{"document": {...}, "score": 0.95}, ...],
            [...]
        ]
    }
    """
    if not request.topk:
        request.topk = config.topk

    results, scores = hybrid_retriever.batch_search(
        request.queries,
        num=request.topk,
        return_score=True
    )

    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = [{"document": doc, "score": scores[i][j]}
                        for j, doc in enumerate(single_result)]
            resp.append(combined)
        else:
            resp.append(single_result)

    return {"result": resp}


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok", "fusion_method": config.fusion_method}


@app.get("/info")
def retrieval_info():
    """获取检索器配置信息"""
    return {
        "fusion_method": config.fusion_method,
        "rrf_k": config.rrf_k,
        "topk": config.topk,
        "bm25_index": config.bm25_index_path,
        "dense_index": config.dense_index_path,
    }


# ===================== 入口 =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Fusion Server")
    # 路径配置
    parser.add_argument("--bm25_index_path", type=str, default="./index/bm25")
    parser.add_argument("--bm25_contain_doc", action=argparse.BooleanOptionalAction, default=True,
                       help="Lucene 索引是否存了原文（--storeRaw 建的）。用 --no-bm25_contain_doc 关闭")
    parser.add_argument("--dense_index_path", type=str, default="./index/e5_Flat.index")
    parser.add_argument("--corpus_path", type=str, default="./data/corpus.jsonl")
    # 检索配置
    parser.add_argument("--topk", type=int, default=10, help="返回的 topk 结果数")
    # 融合配置
    parser.add_argument("--fusion_method", type=str, default="rrf",
                       choices=["rrf", "weighted", "convex"],
                       help="融合策略: rrf(倒数排序融合) / weighted(加权融合) / convex(凸组合融合)")
    parser.add_argument("--rrf_k", type=float, default=60.0,
                       help="RRF 算法参数，建议范围 30-100，越大越平滑")
    parser.add_argument("--dense_weight", type=float, default=0.5,
                       help="Dense 检索权重 (0-1)，BM25 权重自动为 1-dense_weight")
    # Dense 模型配置
    parser.add_argument("--retrieval_model_path", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--retrieval_pooling_method", type=str, default="mean")
    parser.add_argument("--retrieval_query_max_length", type=int, default=256)
    parser.add_argument("--retrieval_use_fp16", action=argparse.BooleanOptionalAction, default=True,
                       help="用 --no-retrieval_use_fp16 在 CPU 上跑（CPU 不支持 fp16）")
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--faiss_gpu", action=argparse.BooleanOptionalAction, default=True,
                       help="用 --no-faiss_gpu 让 FAISS 走 CPU（单卡冒烟测试用）")
    parser.add_argument("--faiss_use_float16", action=argparse.BooleanOptionalAction, default=True,
                       help="GPU 索引用 fp16(省显存)。若转换 OOM,用 --no-faiss_use_float16 走 fp32")
    parser.add_argument("--faiss_temp_mem_mb", type=int, default=512,
                       help="FAISS GPU 临时显存池大小(MB),默认 512")
    # Rerank 配置
    parser.add_argument("--enable_rerank", action='store_true', default=False,
                       help="启用 cross-encoder 精排（接在 RRF 融合之后）")
    parser.add_argument("--rerank_model_path", type=str,
                       default="cross-encoder/ms-marco-MiniLM-L12-v2",
                       help="rerank 模型路径或 HF id")
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--rerank_candidate_factor", type=int, default=3,
                       help="送入 rerank 的候选数 = topk * factor（建议 3-5）")
    parser.add_argument("--rerank_max_length", type=int, default=512)
    # 服务配置
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    config = HybridRetrievalConfig(
        bm25_index_path=args.bm25_index_path,
        bm25_contain_doc=args.bm25_contain_doc,
        dense_index_path=args.dense_index_path,
        corpus_path=args.corpus_path,
        topk=args.topk,
        fusion_method=args.fusion_method,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        retrieval_model_path=args.retrieval_model_path,
        retrieval_pooling_method=args.retrieval_pooling_method,
        retrieval_query_max_length=args.retrieval_query_max_length,
        retrieval_use_fp16=args.retrieval_use_fp16,
        retrieval_batch_size=args.retrieval_batch_size,
        faiss_gpu=args.faiss_gpu,
        faiss_use_float16=args.faiss_use_float16,
        faiss_temp_mem_mb=args.faiss_temp_mem_mb,
        enable_rerank=args.enable_rerank,
        rerank_model_path=args.rerank_model_path,
        rerank_batch_size=args.rerank_batch_size,
        rerank_candidate_factor=args.rerank_candidate_factor,
        rerank_max_length=args.rerank_max_length,
    )

    hybrid_retriever = HybridRetriever(config)

    print("=" * 60)
    print("Hybrid Retrieval Fusion Server")
    print("=" * 60)
    print(f"Fusion Method: {config.fusion_method}")
    print(f"RRF K: {config.rrf_k}")
    print(f"Dense Weight: {config.dense_weight}")
    print(f"TopK: {config.topk}")
    print(f"Rerank Enabled: {config.enable_rerank}")
    if config.enable_rerank:
        print(f"Rerank Model: {config.rerank_model_path}")
        print(f"Rerank Candidate Factor: {config.rerank_candidate_factor}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
