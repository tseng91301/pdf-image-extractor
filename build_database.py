"""Build database from raw PDF files by extracting images and surrounding text descriptions."""

import os
import sys
import json
import time
import logging
import argparse
import shutil
from typing import Set, List

# Suppress slow PaddleX model hoster check
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("build_database")


def load_processed_log(log_path: str) -> Set[str]:
    """Loads the set of previously processed PDF paths."""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data) if isinstance(data, list) else set()
        except Exception as e:
            logger.warning(f"Failed to read processed log {log_path}: {e}")
            return set()
    return set()


def save_processed_log(log_path: str, processed_set: Set[str]):
    """Atomically saves the set of processed PDF paths."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        tmp_path = f"{log_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(processed_set)), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, log_path)
    except Exception as e:
        logger.error(f"Failed to save processed log to {log_path}: {e}")


def find_pdf_files(source_dir: str) -> List[str]:
    """Recursively searches for all PDF files in the source directory."""
    pdf_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    pdf_files.sort()
    return pdf_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract images and contextual descriptions from PDF files into image_data database."
    )
    parser.add_argument(
        "source_dir_pos",
        nargs="?",
        default=None,
        help="Source directory containing raw PDF files (positional)."
    )
    parser.add_argument(
        "-s", "--source-dir",
        default=None,
        help="Source directory containing raw PDF files."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="image_data",
        help="Output directory to store extracted image data folders (default: image_data)."
    )
    parser.add_argument(
        "-l", "--processed-log",
        default=None,
        help="Path to processed documents tracking file (default: <output-dir>/processed_docs.json)."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI used for rendering PDF pages to images (default: 300)."
    )
    parser.add_argument(
        "--optimize-dpi",
        type=int,
        default=400,
        help="DPI used for high-resolution image crops (default: 400)."
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Interval of processed PDFs to update progress log (default: 5)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of PDFs to process."
    )
    parser.add_argument(
        "--use-xref",
        action="store_true",
        default=False,
        help="Attempt extracting embedded raster images via PDF xref before falling back to render."
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Force CPU execution for layout detection and OCR."
    )

    args = parser.parse_args()

    # Determine source directory
    source_dir = args.source_dir or args.source_dir_pos
    if not source_dir:
        logger.error("Source directory is required! Specify as positional argument or with --source-dir.")
        sys.exit(1)

    source_dir = os.path.abspath(source_dir)
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_path = args.processed_log
    if not log_path:
        log_path = os.path.join(output_dir, "processed_docs.json")
    log_path = os.path.abspath(log_path)

    # Import PdfInfo here so arguments and paths are validated first
    from core.converter import PdfInfo

    logger.info("=" * 60)
    logger.info("Starting PDF Database Build Process")
    logger.info(f"Source Directory: {source_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Progress Log:     {log_path}")
    logger.info(f"Render DPI:       {args.dpi}, Crop DPI: {args.optimize_dpi}")
    logger.info(f"Use GPU:          {not args.no_gpu}")
    logger.info("=" * 60)

    # 1. Discover all PDFs
    all_pdfs = find_pdf_files(source_dir)
    logger.info(f"Discovered {len(all_pdfs)} total PDF files in {source_dir}.")
    if not all_pdfs:
        logger.warning("No PDF files found to process.")
        sys.exit(0)

    # 2. Load resume state
    processed_docs = load_processed_log(log_path)
    logger.info(f"Loaded {len(processed_docs)} previously processed documents from log.")

    # 3. Process loop
    newly_processed = 0
    total_images_extracted = 0
    start_time = time.time()

    for idx, pdf_path in enumerate(all_pdfs, start=1):
        rel_path = os.path.relpath(pdf_path, source_dir)

        if rel_path in processed_docs:
            logger.info(f"[{idx}/{len(all_pdfs)}] Skipping already processed: {rel_path}")
            continue

        if args.limit is not None and newly_processed >= args.limit:
            logger.info(f"Reached processing limit of {args.limit} files. Stopping.")
            break

        logger.info(f"\n[{idx}/{len(all_pdfs)}] Processing PDF: {rel_path}")
        pdf = None
        try:
            pdf = PdfInfo(pdf_path, gpu=not args.no_gpu)
            pdf.to_images(dpi=args.dpi)
            pdf.label_layout()
            pdf.label_images(
                optimize_resolution=True,
                optimize_dpi=args.optimize_dpi,
                use_xref=args.use_xref
            )
            pdf.extract_image_description(export=False)

            doc_output_dir = os.path.join(output_dir, pdf.pdf_uid, "image_datas")
            pdf.export_all_image_datas(doc_output_dir, doc_name=rel_path)

            img_count = len(pdf.pdf_imgdatas)
            total_images_extracted += img_count
            processed_docs.add(rel_path)
            newly_processed += 1

            logger.info(f"Successfully extracted {img_count} image(s) from {rel_path} -> {doc_output_dir}")

            if newly_processed % args.save_interval == 0:
                logger.info(f"💾 Saving progress checkpoint ({newly_processed} new docs processed)...")
                save_processed_log(log_path, processed_docs)

        except KeyboardInterrupt:
            logger.warning("\n🛑 Interrupted by user! Saving current progress before exiting...")
            save_processed_log(log_path, processed_docs)
            if pdf:
                pdf.cleanup()
            sys.exit(130)

        except Exception as e:
            logger.exception(f"Error processing {rel_path}: {e}")

        finally:
            if pdf:
                pdf.cleanup()

    # Final progress save
    if newly_processed > 0:
        save_processed_log(log_path, processed_docs)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Extraction Completed in {elapsed:.2f} seconds.")
    logger.info(f"Newly Processed PDFs:    {newly_processed}")
    logger.info(f"Total Processed in DB:   {len(processed_docs)}")
    logger.info(f"Total Images Extracted:  {total_images_extracted}")
    logger.info(f"Output Directory:        {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
