# API Documentation - PDF Image Search System

This document describes the available API endpoints for the PDF Image Search system.

## 1. Search API

Performs a multi-modal search combining text (title, content, keywords) and image features.

- **Endpoint**: `/search`
- **Method**: `GET`
- **Response Format**: `application/json`

### Query Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `q` | `string` | **Required** | The search query string (e.g., "螞蟻聚集"). |
| `topk` | `integer` | `10` | The number of results to return. |
| `w_text` | `float` | `0.7` | Weight for text-based features (0.0 to 1.0). |
| `w_image` | `float` | `0.3` | Weight for image-based features (0.0 to 1.0). |
| `w_title` | `float` | `0.5` | Within text, weight for document titles. |
| `w_content` | `float` | `0.3` | Within text, weight for surrounding text snippets. |
| `w_keyword` | `float` | `0.2` | Within text, weight for keyword matching. |

> [!NOTE]
> All weight parameters are normalized internally. `w_text` and `w_image` are normalized to sum to 1.0. Similarly, `w_title`, `w_content`, and `w_keyword` are normalized to sum to 1.0.

### Success Response

Returns a JSON object containing extracted query keywords and a list of search hits.

```json
{
  "query_keywords": ["螞蟻", "聚集"],
  "results": [
    {
      "score": 0.8542,
      "s_text": 0.9123,
      "s_title": 0.8845,
      "s_sur": 0.7532,
      "s_keyword": 1.0,
      "s_img": 0.7845,
      "best_sur_chunk": "許多螞蟻在水管邊聚集...",
      "matched_keywords": ["螞蟻"],
      "doc_name": "tech_doc_01.pdf",
      "image_name": "fig_01",
      "image_path": "output/doc_01/fig_01.png",
      "page": 5,
      "keywords": ["螞蟻", "害蟲", "防治"],
      "figure_title": "螞蟻聚集示意圖"
    }
  ]
}
```

---

## 2. Static Resources

The system serves static files directly from the filesystem.

### Image Assets
- **Endpoint**: `/output/{path}`
- **Description**: Serves the extracted PNG images.
- **Example**: `GET /output/doc_01/fig_01.png`

### PDF Documents
- **Endpoint**: `/agriculture_tech_docs/{path}`
- **Description**: Serves the original PDF files for preview.
- **Example**: `GET /agriculture_tech_docs/tech_doc_01.pdf`

---

## 3. Web Interface

- **Endpoint**: `/`
- **Method**: `GET`
- **Description**: Returns the HTML search interface (`index.html`).
