/* Interactive Frontend Logic for Hybrid Multimodal Image Search */

document.addEventListener('DOMContentLoaded', () => {
  fetchEngineStatus();
  updateParamLabels();
});

async function fetchEngineStatus() {
  const badge = document.getElementById('statusBadge');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      text.textContent = `引擎就緒 (已索引 ${data.total_images} 張圖 / ${data.device.toUpperCase()})`;
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      text.textContent = '後端連線異常';
      badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    }
  } catch (err) {
    text.textContent = '伺服器離線';
  }
}

function updateParamLabels() {
  const topN = document.getElementById('topNInput').value;
  const recallK = document.getElementById('recallKInput').value;
  const bm25W = document.getElementById('bm25WeightInput').value;
  const denseW = document.getElementById('denseWeightInput').value;
  const capW = document.getElementById('captionWeightInput').value;
  const ctxW = document.getElementById('contextWeightInput').value;
  const rerank = document.getElementById('rerankToggle').checked;

  document.getElementById('topNVal').textContent = topN;
  document.getElementById('recallKVal').textContent = recallK;
  document.getElementById('bm25WeightVal').textContent = bm25W;
  document.getElementById('denseWeightVal').textContent = denseW;
  document.getElementById('captionWeightVal').textContent = capW;
  document.getElementById('contextWeightVal').textContent = ctxW;

  document.getElementById('paramSummaryBadge').textContent = 
    `Top ${topN} | Recall ${recallK} | Reranker: ${rerank ? 'ON' : 'OFF'}`;
}

function applyPreset(queryText) {
  document.getElementById('queryInput').value = queryText;
  handleSearch(new Event('submit'));
}

function clearSearch() {
  document.getElementById('queryInput').value = '';
  document.getElementById('resultsGrid').innerHTML = '';
  document.getElementById('resultsHeader').style.display = 'none';
  document.getElementById('emptyState').style.display = 'block';
}

async function handleSearch(event) {
  if (event) event.preventDefault();
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const topN = parseInt(document.getElementById('topNInput').value);
  const recallK = parseInt(document.getElementById('recallKInput').value);
  const bm25W = parseFloat(document.getElementById('bm25WeightInput').value);
  const denseW = parseFloat(document.getElementById('denseWeightInput').value);
  const capW = parseFloat(document.getElementById('captionWeightInput').value);
  const ctxW = parseFloat(document.getElementById('contextWeightInput').value);
  const rerank = document.getElementById('rerankToggle').checked;

  // Show loading
  document.getElementById('loadingState').style.display = 'flex';
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('resultsGrid').innerHTML = '';
  document.getElementById('resultsHeader').style.display = 'none';

  try {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        top_n: topN,
        recall_k: recallK,
        bm25_weight: bm25W,
        dense_weight: denseW,
        caption_weight: capW,
        context_weight: ctxW,
        enable_rerank: rerank
      })
    });

    const data = await response.json();
    document.getElementById('loadingState').style.display = 'none';

    if (response.ok) {
      renderResults(data);
    } else {
      alert('檢索失敗: ' + (data.detail || '未知錯誤'));
    }
  } catch (err) {
    document.getElementById('loadingState').style.display = 'none';
    alert('連線失敗: ' + err.message);
  }
}

function renderResults(data) {
  const grid = document.getElementById('resultsGrid');
  const header = document.getElementById('resultsHeader');
  const countSpan = document.getElementById('resultsCount');

  header.style.display = 'flex';
  countSpan.textContent = data.returned_count;

  // Latency tags
  document.getElementById('totalLatencyBadge').textContent = `總耗時: ${data.latency_ms} ms`;
  const bk = data.breakdown_latency_ms || {};
  document.getElementById('bm25LatencyBadge').textContent = `BM25: ${bk.bm25_ms || 0}ms`;
  document.getElementById('denseLatencyBadge').textContent = `Dense: ${bk.dense_ms || 0}ms`;
  document.getElementById('rerankLatencyBadge').textContent = `Rerank: ${bk.rerank_ms || 0}ms`;

  if (!data.results || data.results.length === 0) {
    grid.innerHTML = '<div class="empty-state"><h3>未找到相符之文獻圖片</h3><p>可嘗試縮短關鍵字或調高候選召回數 (Recall K)。</p></div>';
    return;
  }

  let html = '';
  data.results.forEach((item) => {
    const imageUrl = `/api/image/${item.uid}/${item.image_name}`;
    const scores = item.scores || {};
    const rankClass = item.rank === 1 ? 'rank-pill top-1' : 'rank-pill';

    const surroundingsHtml = (item.surrounding_texts && item.surrounding_texts.length > 0)
      ? `<details class="surrounding-accordion">
          <summary class="surrounding-summary">展開 ${item.surrounding_texts.length} 段提取之周圍內文</summary>
          <ul class="surrounding-list">
            ${item.surrounding_texts.map(st => `<li class="surrounding-item">${escapeHtml(st)}</li>`).join('')}
          </ul>
         </details>`
      : '';

    html += `
      <div class="result-card">
        <div class="${rankClass}">#${item.rank}</div>
        <div class="card-image-wrap" onclick="openModal('${escapeHtml(JSON.stringify(item))}')">
          <img src="${imageUrl}" class="card-image" alt="${escapeHtml(item.figure_title || item.image_name)}" loading="lazy" onerror="this.src='https://via.placeholder.com/400x250/1e293b/94a3b8?text=Image+Not+Found'">
          <div class="card-image-overlay">
            <span class="zoom-hint">🔍 點擊檢視大圖與文獻座標</span>
          </div>
        </div>
        <div class="card-body">
          <div class="doc-badge-row">
            <span class="doc-name-badge" title="${escapeHtml(item.doc_name)}">📄 ${escapeHtml(item.doc_name || item.uid)}</span>
            <span class="page-badge">第 ${item.page} 頁</span>
          </div>
          <h3 class="card-title">${escapeHtml(item.figure_title || '（無圖題）')}</h3>
          
          <div class="snippet-box">
            ${escapeHtml(item.matched_snippet || '（無擷取文字片段）')}
          </div>

          ${surroundingsHtml}

          <div class="score-pill-row">
            <span class="score-pill final">總分: ${scores.final_score.toFixed(4)}</span>
            <span class="score-pill bm25">BM25: ${scores.bm25_score.toFixed(2)}</span>
            <span class="score-pill dense">語意: ${scores.dense_score.toFixed(2)}</span>
            ${scores.rerank_score !== null ? `<span class="score-pill rerank">Cross-Encoder: ${scores.rerank_score.toFixed(4)}</span>` : ''}
          </div>
        </div>
      </div>
    `;
  });

  grid.innerHTML = html;
}

function openModal(rawJsonStr) {
  const item = JSON.parse(rawJsonStr);
  const modal = document.getElementById('imageModal');
  const modalBody = document.getElementById('modalBody');
  const imageUrl = `/api/image/${item.uid}/${item.image_name}`;

  modalBody.innerHTML = `
    <div class="modal-img-wrap">
      <img src="${imageUrl}" class="modal-img" alt="${escapeHtml(item.figure_title)}">
    </div>
    <div class="modal-details">
      <div class="doc-badge-row" style="margin-bottom: 12px;">
        <span class="doc-name-badge">📄 文獻: ${escapeHtml(item.doc_name)}</span>
        <span class="page-badge">頁碼: P.${item.page} | UID: ${item.uid}</span>
      </div>
      <h2 style="font-size: 1.3rem; margin-bottom: 12px; color: #fff;">${escapeHtml(item.figure_title || item.image_name)}</h2>
      
      <div style="background: rgba(0,0,0,0.4); padding: 14px; border-radius: 8px; margin-bottom: 16px;">
        <h4 style="font-size: 0.85rem; color: #818cf8; margin-bottom: 8px;">📑 完整提取之周圍內文（${item.surrounding_texts ? item.surrounding_texts.length : 0} 段）：</h4>
        ${item.surrounding_texts && item.surrounding_texts.length > 0 
          ? item.surrounding_texts.map((st, i) => `<p style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;"><strong>[${i+1}]</strong> ${escapeHtml(st)}</p>`).join('')
          : '<p style="font-size:0.8rem; color:#64748b;">無周圍內文</p>'
        }
      </div>

      <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted);">
        <p>圖檔路徑: ${escapeHtml(item.file_path)}</p>
        <p>Bounding Box 座標: ${JSON.stringify(item.coordinate || [])}</p>
      </div>
    </div>
  `;

  modal.style.display = 'flex';
}

function closeModal(event) {
  if (event.target.id === 'imageModal') {
    closeModalDirect();
  }
}

function closeModalDirect() {
  document.getElementById('imageModal').style.display = 'none';
}

function toggleDiagnostics() {
  const sec = document.getElementById('diagnosticsSection');
  sec.style.display = sec.style.display === 'none' ? 'block' : 'none';
}

function switchDiagTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-body').forEach(b => b.style.display = 'none');
  event.target.classList.add('active');
  document.getElementById(tabId).style.display = 'block';
}

// Standalone Diagnostics Fetchers
async function runDiagTokenize() {
  const text = document.getElementById('diagTokInput').value;
  const out = document.getElementById('diagTokOutput');
  out.textContent = '斷詞中...';
  try {
    const res = await fetch(`/api/debug/tokenize?text=${encodeURIComponent(text)}`, { method: 'POST' });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

async function runDiagBM25() {
  const q = document.getElementById('diagBm25Input').value;
  const out = document.getElementById('diagBm25Output');
  out.textContent = 'BM25 檢索中...';
  try {
    const res = await fetch(`/api/debug/bm25?query=${encodeURIComponent(q)}&top_k=5`, { method: 'POST' });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

async function runDiagDense() {
  const q = document.getElementById('diagDenseInput').value;
  const out = document.getElementById('diagDenseOutput');
  out.textContent = 'Dense 向量比對中...';
  try {
    const res = await fetch(`/api/debug/dense?query=${encodeURIComponent(q)}&top_k=5`, { method: 'POST' });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

async function runDiagRerank() {
  const q = document.getElementById('diagRerankQuery').value;
  const rawTexts = document.getElementById('diagRerankTexts').value.split('\n').filter(t => t.trim());
  const out = document.getElementById('diagRerankOutput');
  out.textContent = 'Cross-Encoder 重排序評分中...';
  try {
    const params = new URLSearchParams();
    params.append('query', q);
    rawTexts.forEach(t => params.append('candidate_texts', t));
    const res = await fetch(`/api/debug/rerank?${params.toString()}`, { method: 'POST' });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
