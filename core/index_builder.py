"""CLI and utility script to pre-build and persist search index cache."""

import argparse
import logging
import time
import os

from core.engine import HybridSearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("index_builder")


def main():
    parser = argparse.ArgumentParser(description="Build and cache BM25 and Dense vector search indices.")
    parser.add_argument("--data-dir", default="image_data", help="Directory containing image_data documents.")
    parser.add_argument("--cache-dir", default="index_cache", help="Directory to store serialized indices.")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of document folders to index.")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to index.")
    parser.add_argument("--force", action="store_true", help="Force rebuild index even if cache exists.")
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting index building process...")
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Cache Directory: {args.cache_dir}")
    logger.info(f"Max Docs: {args.max_docs}, Max Images: {args.max_images}, Force: {args.force}")
    
    engine = HybridSearchEngine(data_dir=args.data_dir, cache_dir=args.cache_dir)
    engine.build_or_load_index(
        max_documents=args.max_docs,
        max_images=args.max_images,
        force_rebuild=args.force,
        save_to_cache=True
    )
    
    status = engine.get_status()
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Index build completed in {elapsed:.2f} seconds.")
    logger.info(f"Total Unique Documents: {status.total_documents}")
    logger.info(f"Total Indexed Images: {status.total_images}")
    logger.info(f"Device: {status.device}")
    logger.info(f"Dense Model: {status.dense_model_name}")
    logger.info(f"Reranker Model: {status.reranker_model_name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
