"""Comprehensive unit test suite verifying modularity, persistent model reuse, and hybrid search."""

import pytest
import numpy as np
import os
from typing import List

from core.models import ImageDocument, SearchQuery
from core.tokenizer import JiebaTokenizer
from core.model_manager import ModelManager
from core.data_loader import DataLoader
from core.bm25 import BM25Retriever
from core.dense import DenseRetriever
from core.reranker import CrossEncoderReranker
from core.fusion import RRFHybridFusion
from core.engine import HybridSearchEngine


@pytest.fixture(scope="session")
def sample_documents() -> List[ImageDocument]:
    """Provides mock ImageDocuments with multi-surrounding texts."""
    docs = [
        ImageDocument(
            image_id="doc_001_img_0",
            uid="uid_001",
            doc_name="山藥桃園4號栽培手冊",
            image_name="image_0000",
            file_path="image_data/mock/image_0000.png",
            page=1,
            figure_title="地下塊莖外觀特寫",
            surrounding_texts=[
                "效益：本品種係晚熟品種，可延遲1至3個月收穫而不發芽，避開盛產期調節供需。",
                "特點：地下塊莖肉色雪白不氧化，黏度極高，產量優異適合砂質壤土。",
                "桃園區農業改良場育種成果。"
            ]
        ).build_text_representations(),
        ImageDocument(
            image_id="doc_002_img_0",
            uid="uid_002",
            doc_name="自行車維修保養指南",
            image_name="image_0001",
            file_path="image_data/mock/image_0001.png",
            page=2,
            figure_title="Shimano 105 BR-R7000 煞車夾器安裝",
            surrounding_texts=[
                "安裝煞車夾器時請使用 5mm 內六角扳手以 8-10 Nm 扭力鎖固前叉沉頭螺母。",
                "定期檢查來令片磨耗狀況，厚度低於 0.5mm 應即刻更換。"
            ]
        ).build_text_representations(),
        ImageDocument(
            image_id="doc_003_img_0",
            uid="uid_003",
            doc_name="水稻病蟲害防治專刊",
            image_name="image_0002",
            file_path="image_data/mock/image_0002.png",
            page=5,
            figure_title="稻熱病葉片典型紡錘形病斑",
            surrounding_texts=[
                "稻熱病好發於高濕環境，葉片出現褐色至灰白色紡錘形病斑。",
                "建議於發病初期施用三賽唑或嘉賜黴素進行有效防治。"
            ]
        ).build_text_representations()
    ]
    return docs


def test_tokenizer():
    tokenizer = JiebaTokenizer()
    tokens = tokenizer.tokenize("BR-R7000 煞車夾器安裝扭力")
    assert "BR" in tokens or "BR-R7000" in tokens
    assert "煞車" in tokens or "夾器" in tokens
    explanation = tokenizer.explain("山藥塊莖")
    assert explanation["total_tokens"] >= 2


def test_model_manager_singleton():
    m1 = ModelManager()
    m2 = ModelManager()
    assert m1 is m2
    status = m1.get_status()
    assert "device" in status


def test_bm25_standalone(sample_documents):
    bm25 = BM25Retriever()
    bm25.fit(sample_documents)
    
    # Keyword search for mountain yam
    results = bm25.search("山藥 塊莖 肉色雪白", top_k=2)
    assert len(results) > 0
    top_idx, top_score = results[0]
    assert top_idx == 0  # Should match doc 0
    assert top_score > 0

    # Keyword search for brake caliper
    results_brake = bm25.search("BR-R7000 夾器", top_k=2)
    assert len(results_brake) > 0
    assert results_brake[0][0] == 1  # Should match doc 1


def test_dense_standalone(sample_documents):
    dense = DenseRetriever()
    dense.fit(sample_documents)
    
    # Semantic synonym search (without literal words)
    results = dense.search("腳踏車制動器鎖緊力道", top_k=2)
    assert len(results) > 0
    top_idx, top_score = results[0]
    assert top_idx == 1  # Should semantically match bicycle brake caliper


def test_reranker_standalone():
    reranker = CrossEncoderReranker()
    diag = reranker.explain(
        query="水稻 葉片 病害 防治藥劑",
        document_text="文獻: 水稻病蟲害防治專刊 | 內文: 稻熱病出現紡錘形病斑，建議施用三賽唑或嘉賜黴素防治。"
    )
    assert diag["rerank_score"] > -5.0
    assert 0.0 <= diag["sigmoid_probability"] <= 1.0


def test_rrf_fusion():
    bm25_scores = np.array([10.0, 0.0, 5.0], dtype=np.float32)
    dense_scores = np.array([0.9, 0.3, 0.4], dtype=np.float32)
    fusion = RRFHybridFusion(rrf_k=60)
    fused = fusion.fuse(bm25_scores, dense_scores, recall_k=2)
    assert len(fused) == 2
    assert fused[0]["doc_index"] == 0  # Top rank in both


def test_engine_end_to_end(sample_documents):
    engine = HybridSearchEngine()
    engine.documents = sample_documents
    engine.doc_id_to_index = {doc.image_id: i for i, doc in enumerate(sample_documents)}
    engine.bm25_retriever.fit(sample_documents)
    engine.dense_retriever.fit(sample_documents)
    engine.is_indexed = True

    response = engine.search(
        SearchQuery(
            query="山藥 晚熟 避開盛產期",
            top_n=2,
            recall_k=3,
            enable_rerank=True
        )
    )
    assert response.returned_count > 0
    assert response.results[0].image_id == "doc_001_img_0"
    assert response.results[0].scores.rerank_score is not None
    assert response.latency_ms > 0
