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
  const item = typeof rawJsonStr === 'string' ? JSON.parse(rawJsonStr) : rawJsonStr;
  const modal = document.getElementById('imageModal');
  const modalBody = document.getElementById('modalBody');
  const imageUrl = item.image_url || (item.uid && item.image_name ? `/api/image/${item.uid}/${item.image_name}` : '');

  modalBody.innerHTML = `
    <div class="modal-img-wrap">
      <img src="${imageUrl}" class="modal-img" alt="${escapeHtml(item.figure_title || item.image_id)}">
    </div>
    <div class="modal-details">
      <div class="doc-badge-row" style="margin-bottom: 12px;">
        <span class="doc-name-badge">📄 文獻: ${escapeHtml(item.doc_name || '文獻庫')}</span>
        <span class="page-badge">頁碼: P.${item.page || '-'} | ID: ${escapeHtml(item.image_id || item.image_name || item.uid || '-')}</span>
      </div>
      <h2 style="font-size: 1.3rem; margin-bottom: 12px; color: #fff;">${escapeHtml(item.figure_title || item.image_name || item.image_id || '（無圖題）')}</h2>
      
      <div style="background: rgba(0,0,0,0.4); padding: 14px; border-radius: 8px; margin-bottom: 16px;">
        <h4 style="font-size: 0.85rem; color: #818cf8; margin-bottom: 8px;">📑 提取之周圍內文/說明（${item.surrounding_texts ? item.surrounding_texts.length : (item.reason ? 1 : 0)} 段）：</h4>
        ${item.surrounding_texts && item.surrounding_texts.length > 0 
          ? item.surrounding_texts.map((st, i) => `<p style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;"><strong>[${i+1}]</strong> ${escapeHtml(st)}</p>`).join('')
          : (item.reason ? `<p style="font-size: 0.82rem; color: #cbd5e1;">${escapeHtml(item.reason)}</p>` : '<p style="font-size:0.8rem; color:#64748b;">無周圍內文</p>')
        }
      </div>

      <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted);">
        <p>圖片 API 路徑: ${escapeHtml(imageUrl)}</p>
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

let devConsoleInitialized = false;
let currentBenchmarkReport = null;
let currentQueryEvaluations = [];
let benchmarkPollInterval = null;

function toggleDiagnostics() {
  const sec = document.getElementById('diagnosticsSection');
  const isOpen = sec.style.display !== 'none';
  sec.style.display = isOpen ? 'none' : 'block';

  if (!isOpen && !devConsoleInitialized) {
    devConsoleInitialized = true;
    loadBenchmarkDatasetsAndReports();
  }
}

function switchDevMode(mode) {
  const btnBm = document.getElementById('btnModeBenchmark');
  const btnIso = document.getElementById('btnModeIsolation');
  const viewBm = document.getElementById('devBenchmarkView');
  const viewIso = document.getElementById('devIsolationView');

  if (mode === 'benchmark') {
    btnBm.classList.add('active');
    btnIso.classList.remove('active');
    viewBm.style.display = 'block';
    viewIso.style.display = 'none';
  } else {
    btnIso.classList.add('active');
    btnBm.classList.remove('active');
    viewIso.style.display = 'block';
    viewBm.style.display = 'none';
  }
}

async function loadBenchmarkDatasetsAndReports() {
  await Promise.all([
    fetchBenchmarkDatasets(),
    fetchBenchmarkReports()
  ]);
  pollBenchmarkStatus();
}

async function fetchBenchmarkDatasets() {
  const select = document.getElementById('benchmarkDatasetSelect');
  try {
    const res = await fetch('/api/benchmark/datasets');
    if (res.ok) {
      const datasets = await res.json();
      if (datasets.length === 0) {
        select.innerHTML = '<option value="">(無可用資料集)</option>';
        return;
      }
      select.innerHTML = datasets.map(d => {
        const queryCount = d.total_queries ? `${d.total_queries} 題` : '未載入題數';
        return `<option value="${escapeHtml(d.filename)}">${escapeHtml(d.filename)} (${queryCount})</option>`;
      }).join('');
    } else {
      select.innerHTML = '<option value="">載入資料集失敗</option>';
    }
  } catch (err) {
    select.innerHTML = '<option value="">載入失敗: ' + escapeHtml(err.message) + '</option>';
  }
}

async function fetchBenchmarkReports(preferredFilename = null) {
  const select = document.getElementById('benchmarkReportSelect');
  try {
    const res = await fetch('/api/benchmark/reports');
    if (res.ok) {
      const reports = await res.json();
      if (reports.length === 0) {
        select.innerHTML = '<option value="">(目前無歷史報告)</option>';
        return;
      }
      select.innerHTML = reports.map(r => {
        const fullPipe = (r.summary && r.summary.ablation_comparison && r.summary.ablation_comparison.Full_Pipeline) || {};
        const hit1 = fullPipe.hit_at_1 !== undefined ? `Hit@1: ${(fullPipe.hit_at_1 * 100).toFixed(1)}%` : '';
        const mrr = fullPipe.mrr_at_10 !== undefined ? `MRR: ${fullPipe.mrr_at_10.toFixed(3)}` : '';
        const metaStr = [r.modified_time, `${r.total_queries} 題`, hit1, mrr].filter(Boolean).join(' | ');
        const isSelected = (preferredFilename && r.filename === preferredFilename) ? 'selected' : '';
        return `<option value="${escapeHtml(r.filename)}" ${isSelected}>${escapeHtml(r.filename)} (${metaStr})</option>`;
      }).join('');

      // Auto-load currently selected report
      if (select.value) {
        loadSelectedReport();
      }
    } else {
      select.innerHTML = '<option value="">載入報告清單失敗</option>';
    }
  } catch (err) {
    select.innerHTML = '<option value="">載入失敗: ' + escapeHtml(err.message) + '</option>';
  }
}

async function loadSelectedReport() {
  const select = document.getElementById('benchmarkReportSelect');
  const filename = select.value;
  if (!filename) return;

  try {
    const res = await fetch(`/api/benchmark/reports/${encodeURIComponent(filename)}`);
    if (res.ok) {
      const report = await res.json();
      currentBenchmarkReport = report;
      renderBenchmarkReport(report);
    } else {
      alert('無法載入報告: ' + filename);
    }
  } catch (err) {
    alert('載入報告失敗: ' + err.message);
  }
}

function renderBenchmarkReport(report) {
  // 1. Meta Tags
  document.getElementById('tagDataset').textContent = report.dataset_file || '未指定';
  document.getElementById('tagTime').textContent = report.benchmark_time || '-';
  const totalQ = (report.dataset_summary && report.dataset_summary.total_queries) || 0;
  document.getElementById('tagTotalQueries').textContent = `${totalQ} 題`;
  const engStatus = report.engine_status || {};
  document.getElementById('tagTotalImages').textContent = `${engStatus.total_images || 0} 張`;
  document.getElementById('tagDevice').textContent = (engStatus.device || 'CPU').toUpperCase();

  // 2. Executive KPI Cards from Full_Pipeline
  const arms = report.arms || {};
  const fullPipe = arms['Full_Pipeline'] || {};
  const fullOm = fullPipe.overall_metrics || {};
  const fullLat = fullPipe.latency_ms || {};
  const fullPm = fullPipe.per_level_metrics || {};

  const hit1Pct = (fullOm['Hit@1'] !== undefined) ? (fullOm['Hit@1'] * 100).toFixed(2) + '%' : '-';
  const hit5Pct = (fullOm['Hit@5'] !== undefined) ? (fullOm['Hit@5'] * 100).toFixed(2) + '%' : '-';
  const mrrVal = (fullOm['MRR@10'] !== undefined) ? fullOm['MRR@10'].toFixed(4) : '-';
  const p50Lat = fullLat.p50 ? `${fullLat.p50.toFixed(1)} ms` : '-';
  const meanLat = fullLat.mean ? `${fullLat.mean.toFixed(1)} ms` : '-';

  document.getElementById('kpiHit1').textContent = hit1Pct;
  const l1Hit1 = fullPm.level1_exact_keyword ? (fullPm.level1_exact_keyword['Hit@1'] * 100).toFixed(1) + '%' : '-';
  const l2Hit1 = fullPm.level2_semantic_qa ? (fullPm.level2_semantic_qa['Hit@1'] * 100).toFixed(1) + '%' : '-';
  document.getElementById('kpiHit1Sub').textContent = `精確詞(L1): ${l1Hit1} | 語意(L2): ${l2Hit1}`;

  document.getElementById('kpiHit5').textContent = hit5Pct;
  const l1Hit5 = fullPm.level1_exact_keyword ? (fullPm.level1_exact_keyword['Hit@5'] * 100).toFixed(1) + '%' : '-';
  const l2Hit5 = fullPm.level2_semantic_qa ? (fullPm.level2_semantic_qa['Hit@5'] * 100).toFixed(1) + '%' : '-';
  document.getElementById('kpiHit5Sub').textContent = `前五名命中 (L1: ${l1Hit5} | L2: ${l2Hit5})`;

  document.getElementById('kpiMrr').textContent = mrrVal;
  document.getElementById('kpiLatency').textContent = p50Lat;
  document.getElementById('kpiLatencySub').textContent = `平均延遲: ${meanLat} (包含精排)`;

  // 3. Overall Ablation Comparison Table
  renderAblationTable(arms);

  // 4. Per-Level Breakdown Table
  renderLevelBreakdownTable(arms);

  // 5. Per-Query Inspector
  currentQueryEvaluations = fullPipe.query_evaluations || [];
  filterQueryList();

  // 6. Raw JSON
  document.getElementById('rawJsonPre').textContent = JSON.stringify(report, null, 2);
}

function renderAblationTable(arms) {
  const tbody = document.getElementById('ablationTableBody');
  const armOrder = ['BM25_Only', 'Dense_Only', 'RRF_Hybrid', 'Full_Pipeline'];
  const armPills = {
    'BM25_Only': '<span class="arm-pill bm25">BM25 Only</span>',
    'Dense_Only': '<span class="arm-pill dense">Dense Only</span>',
    'RRF_Hybrid': '<span class="arm-pill rrf">RRF Hybrid</span>',
    'Full_Pipeline': '<span class="arm-pill full">🏆 Full Pipeline</span>'
  };

  let rowsHtml = '';
  armOrder.forEach(name => {
    const data = arms[name];
    if (!data) return;
    const om = data.overall_metrics || {};
    const lat = data.latency_ms || {};
    const isChamp = name === 'Full_Pipeline';
    const trClass = isChamp ? 'champion-row' : '';

    rowsHtml += `
      <tr class="${trClass}">
        <td>${armPills[name] || name}</td>
        <td style="font-size:0.78rem; color: var(--text-muted);">${escapeHtml(data.description || '')}</td>
        <td class="metric-val" style="color: ${isChamp ? '#10b981' : '#fff'};">${(om['Hit@1'] * 100).toFixed(2)}%</td>
        <td class="metric-val">${(om['Hit@3'] * 100).toFixed(2)}%</td>
        <td class="metric-val" style="color: ${isChamp ? '#06b6d4' : '#fff'};">${(om['Hit@5'] * 100).toFixed(2)}%</td>
        <td class="metric-val">${(om['Hit@10'] * 100).toFixed(2)}%</td>
        <td class="metric-val" style="color: ${isChamp ? '#a855f7' : '#fff'};">${om['MRR@10'] ? om['MRR@10'].toFixed(4) : '-'}</td>
        <td class="metric-val">${om['nDCG@10'] ? om['nDCG@10'].toFixed(4) : '-'}</td>
        <td class="metric-val">${lat.p50 ? lat.p50.toFixed(1) + ' ms' : '-'}</td>
        <td class="metric-val">${lat.mean ? lat.mean.toFixed(1) + ' ms' : '-'}</td>
      </tr>
    `;
  });

  tbody.innerHTML = rowsHtml || '<tr><td colspan="10" style="text-align:center;">無消融數據</td></tr>';
}

function renderLevelBreakdownTable(arms) {
  const tbody = document.getElementById('levelTableBody');
  const bm25Pm = (arms['BM25_Only'] && arms['BM25_Only'].per_level_metrics) || {};
  const densePm = (arms['Dense_Only'] && arms['Dense_Only'].per_level_metrics) || {};
  const rrfPm = (arms['RRF_Hybrid'] && arms['RRF_Hybrid'].per_level_metrics) || {};
  const fullPm = (arms['Full_Pipeline'] && arms['Full_Pipeline'].per_level_metrics) || {};

  const levelConfigs = [
    {
      key: 'level1_exact_keyword',
      title: 'Level 1: 精確圖題與關鍵字',
      count: '50 題',
      badgeClass: 'l1',
      desc: '測試對文獻圖題、專有名詞之精確字面召回。BM25 與 Full Pipeline 均達 80% 頂尖水準。'
    },
    {
      key: 'level2_semantic_qa',
      title: 'Level 2: 農業問答與語意內文',
      count: '60 題',
      badgeClass: 'l2',
      desc: '測試內文段落理解與多行脈絡關聯。Full Pipeline 透過 Cross-Encoder 將首位命中率由 23.3% 大幅提升至 31.7%。'
    },
    {
      key: 'level3_visual_description',
      title: 'Level 3: 視覺外觀與病徵特徵',
      count: '40 題',
      badgeClass: 'l3',
      desc: '病徵、昆蟲外觀及顏色描述。雙塔語意融合 (RRF) 與重排可達到 42.5%~45% 之前五召回率。'
    },
    {
      key: 'level4_hard_negative',
      title: 'Level 4: 負向干擾與抗拒檢索',
      count: '10 題',
      badgeClass: 'l4',
      desc: '未包含圖檔之干擾難題，檢驗系統防誤判與置信度拒絕能力。'
    }
  ];

  let html = '';
  levelConfigs.forEach(cfg => {
    if (cfg.key === 'level4_hard_negative') {
      const fullL4 = fullPm[cfg.key] || {};
      const fpRate = fullL4.False_Positive_Rate !== undefined ? (fullL4.False_Positive_Rate * 100).toFixed(1) + '%' : '-';
      html += `
        <tr>
          <td><span class="query-level-badge ${cfg.badgeClass}">${cfg.title}</span></td>
          <td class="metric-val">${cfg.count}</td>
          <td colspan="4" style="text-align:center; font-family:var(--font-mono); color:var(--text-muted);">
            誤報率 (False Positive Rate): <strong>${fpRate}</strong>
          </td>
          <td style="font-size:0.75rem; color:var(--text-muted);">${cfg.desc}</td>
        </tr>
      `;
    } else {
      const b = bm25Pm[cfg.key] || {};
      const d = densePm[cfg.key] || {};
      const r = rrfPm[cfg.key] || {};
      const f = fullPm[cfg.key] || {};

      const bStr = `${(b['Hit@1'] * 100 || 0).toFixed(1)}% / ${(b['Hit@5'] * 100 || 0).toFixed(1)}%`;
      const dStr = `${(d['Hit@1'] * 100 || 0).toFixed(1)}% / ${(d['Hit@5'] * 100 || 0).toFixed(1)}%`;
      const rStr = `${(r['Hit@1'] * 100 || 0).toFixed(1)}% / ${(r['Hit@5'] * 100 || 0).toFixed(1)}%`;
      const fStr = `${(f['Hit@1'] * 100 || 0).toFixed(1)}% / ${(f['Hit@5'] * 100 || 0).toFixed(1)}%`;

      html += `
        <tr>
          <td><span class="query-level-badge ${cfg.badgeClass}">${cfg.title}</span></td>
          <td class="metric-val">${cfg.count}</td>
          <td class="metric-val">${bStr}</td>
          <td class="metric-val">${dStr}</td>
          <td class="metric-val">${rStr}</td>
          <td class="metric-val" style="color:#10b981; font-weight:700;">${fStr}</td>
          <td style="font-size:0.75rem; color:var(--text-muted);">${cfg.desc}</td>
        </tr>
      `;
    }
  });

  tbody.innerHTML = html;
}

function filterQueryList() {
  const levelEl = document.getElementById('queryLevelFilter');
  const levelVal = levelEl ? levelEl.value : 'all';
  const hitEl = document.getElementById('queryHitFilter');
  const hitVal = hitEl ? hitEl.value : 'all';
  const searchEl = document.getElementById('querySearchFilter');
  const searchVal = searchEl ? searchEl.value.trim().toLowerCase() : '';
  const container = document.getElementById('benchmarkQueryList');
  const countSpan = document.getElementById('queryFilterCount');

  if (!currentQueryEvaluations || currentQueryEvaluations.length === 0) {
    if (container) container.innerHTML = '<div class="empty-state sm">本報告未包含逐筆評測明細</div>';
    if (countSpan) countSpan.textContent = '顯示 0 / 0 筆';
    return;
  }

  const filtered = currentQueryEvaluations.filter(item => {
    // Level filter
    if (levelVal && levelVal !== 'all' && String(item.difficulty_level) !== levelVal) {
      return false;
    }
    // Hit filter
    if (hitVal === 'hit1' && !item.is_hit_at_1) return false;
    if (hitVal === 'hit5' && !item.is_hit_at_5) return false;
    if (hitVal === 'miss' && item.is_hit_at_5) return false;

    // Search text filter
    if (searchVal) {
      const qText = (item.query || '').toLowerCase();
      const docText = (item.source_doc || '').toLowerCase();
      const targetId = (item.target_image_id || '').toLowerCase();
      if (!qText.includes(searchVal) && !docText.includes(searchVal) && !targetId.includes(searchVal)) {
        return false;
      }
    }
    return true;
  });

  if (countSpan) countSpan.textContent = `顯示 ${filtered.length} / ${currentQueryEvaluations.length} 筆`;

  if (filtered.length === 0) {
    if (container) container.innerHTML = '<div class="empty-state sm">未找到符合篩選條件的評測題目</div>';
    return;
  }

  // Render cards (first 60 for optimal DOM speed)
  const displaySlice = filtered.slice(0, 60);
  let html = '';

  displaySlice.forEach(q => {
    const lvl = q.difficulty_level || 1;
    let hitBadgeHtml = '';
    if (q.is_hit_at_1) {
      hitBadgeHtml = '<span class="query-hit-status hit1">✅ Hit@1 (首位命中)</span>';
    } else if (q.is_hit_at_5) {
      hitBadgeHtml = `<span class="query-hit-status hit5">🎯 Hit@5 (第 ${q.first_hit_rank} 名)</span>`;
    } else {
      hitBadgeHtml = '<span class="query-hit-status miss">❌ Miss (未命中前5名)</span>';
    }

    // Ground Truth Block
    let gtHtml = '';
    if (q.ground_truth && q.ground_truth.length > 0) {
      const gt = q.ground_truth[0];
      const gtUrl = gt.image_url || '';
      gtHtml = `
        <div class="gt-target-box">
          <div class="gt-box-title">
            <span>🎯 標準答案 (Ground Truth)</span>
            <span style="font-size:0.7rem; color:#86efac;">★★ 核心目標</span>
          </div>
          <div class="gt-thumb-wrap" onclick="openModal('${escapeHtml(JSON.stringify({ image_url: gtUrl, image_id: gt.image_id, figure_title: gt.reason, doc_name: q.source_doc }))}')">
            <img src="${gtUrl}" class="gt-thumb" alt="${escapeHtml(gt.image_id)}" loading="lazy" onerror="this.src='https://via.placeholder.com/200x140/1e293b/94a3b8?text=GT+Image'">
            <div class="card-image-overlay"><span class="zoom-hint">🔍 點擊檢視大圖</span></div>
          </div>
          <div class="gt-desc">圖檔 ID: ${escapeHtml(gt.image_id)}</div>
          <div class="gt-reason">${escapeHtml(gt.reason || '目標判定標準')}</div>
        </div>
      `;
    } else {
      gtHtml = `
        <div class="gt-target-box">
          <div class="gt-box-title"><span>🎯 負向干擾題 (Hard Negative)</span></div>
          <p style="font-size:0.78rem; color:var(--text-muted); margin-top:20px;">此題無對應之文獻圖片，驗證模型對未知查詢之抗拒與低置信度。</p>
        </div>
      `;
    }

    // Retrieved Gallery
    let retCardsHtml = '';
    const results = q.retrieved_results || [];
    results.forEach((ret, idx) => {
      const isHit = ret.is_hit;
      const retUrl = ret.image_url || '';
      const hitClass = isHit ? 'is-hit' : '';
      const pillClass = isHit ? 'hit' : 'miss';
      const pillText = isHit ? '🎯 命中' : '文獻召回';

      retCardsHtml += `
        <div class="retrieved-card ${hitClass}">
          <div class="ret-rank-pill">#${ret.rank || idx + 1}</div>
          <div class="ret-hit-pill ${pillClass}">${pillText}</div>
          <div class="ret-thumb-wrap" onclick="openModal('${escapeHtml(JSON.stringify(ret))}')">
            <img src="${retUrl}" class="ret-thumb" alt="${escapeHtml(ret.figure_title || ret.image_id)}" loading="lazy" onerror="this.src='https://via.placeholder.com/200x120/1e293b/94a3b8?text=Image'">
            <div class="card-image-overlay"><span class="zoom-hint">🔍 放大檢視</span></div>
          </div>
          <div class="ret-doc-badge" title="${escapeHtml(ret.doc_name)}">📄 P.${ret.page} ${escapeHtml(ret.doc_name)}</div>
          <div class="ret-title" title="${escapeHtml(ret.figure_title)}">${escapeHtml(ret.figure_title || '（無圖題）')}</div>
          <div class="ret-score-badge">Score: ${(ret.score || 0).toFixed(4)}</div>
        </div>
      `;
    });

    html += `
      <div class="query-card">
        <div class="query-card-header">
          <div class="query-meta-left">
            <span class="query-id-badge">#${escapeHtml(q.query_id)}</span>
            <span class="query-level-badge l${lvl}">${escapeHtml(q.level_name || 'Level ' + lvl)}</span>
            ${hitBadgeHtml}
          </div>
          <div class="query-scores-right">
            <span>MRR@10: <strong>${q.mrr_at_10}</strong></span>
            <span>nDCG@10: <strong>${q.ndcg_at_10}</strong></span>
          </div>
        </div>

        <div class="query-prompt-box">
          <div class="query-prompt-text">題目：${escapeHtml(q.query)}</div>
          <div class="query-prompt-doc">📄 出處來源文獻：${escapeHtml(q.source_doc || '專訊文獻庫')}</div>
        </div>

        <div class="query-comparison-layout">
          ${gtHtml}
          <div class="retrieved-gallery-wrap">
            <div class="retrieved-gallery-header">
              <span class="retrieved-gallery-title">⚡ 系統檢索排序前 ${results.length} 名圖片：</span>
              <span class="gallery-scroll-hint">↔ 可左右拖曳滑動或滾輪捲動 (共 ${results.length} 張)</span>
            </div>
            <div class="retrieved-gallery" onwheel="handleGalleryWheel(event, this)">
              ${retCardsHtml || '<div style="color:var(--text-muted); font-size:0.8rem;">無檢索結果</div>'}
            </div>
          </div>
        </div>
      </div>
    `;
  });

  if (filtered.length > displaySlice.length) {
    html += `<div style="text-align:center; padding:16px; color:var(--text-muted); font-size:0.8rem;">已呈現前 ${displaySlice.length} 筆，請使用篩選器縮小範圍檢視更多。</div>`;
  }

  container.innerHTML = html;
}

async function triggerBenchmarkRun() {
  const datasetSelect = document.getElementById('benchmarkDatasetSelect');
  const dataset = datasetSelect.value;
  if (!dataset) {
    alert('請先選擇欲評測之資料集 (Dataset)');
    return;
  }
  const reportName = document.getElementById('benchmarkReportNameInput').value.trim();

  const runBtn = document.getElementById('runBenchmarkBtn');
  const cancelBtn = document.getElementById('cancelBenchmarkBtn');
  const progressBox = document.getElementById('benchmarkProgressBox');

  runBtn.style.display = 'none';
  cancelBtn.style.display = 'flex';
  cancelBtn.disabled = false;
  cancelBtn.textContent = '⏹ 中斷評測';
  progressBox.style.display = 'block';

  try {
    const params = new URLSearchParams();
    params.append('dataset_filename', dataset);
    if (reportName) params.append('report_name', reportName);

    const res = await fetch(`/api/benchmark/run?${params.toString()}`, { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      if (benchmarkPollInterval) clearInterval(benchmarkPollInterval);
      benchmarkPollInterval = setInterval(pollBenchmarkStatus, 1500);
    } else {
      alert('無法啟動評測: ' + (data.detail || '未知錯誤'));
      runBtn.style.display = 'flex';
      cancelBtn.style.display = 'none';
      progressBox.style.display = 'none';
    }
  } catch (err) {
    alert('連線失敗: ' + err.message);
    runBtn.style.display = 'flex';
    cancelBtn.style.display = 'none';
    progressBox.style.display = 'none';
  }
}

async function pollBenchmarkStatus() {
  try {
    const res = await fetch('/api/benchmark/run/status');
    if (!res.ok) return;
    const st = await res.json();

    const runBtn = document.getElementById('runBenchmarkBtn');
    const cancelBtn = document.getElementById('cancelBenchmarkBtn');
    const progressBox = document.getElementById('benchmarkProgressBox');
    const armText = document.getElementById('bmProgressArmText');
    const countText = document.getElementById('bmProgressCountText');
    const elapsedText = document.getElementById('bmProgressElapsedText');
    const bar = document.getElementById('bmProgressBar');

    if (st.status === 'running') {
      runBtn.style.display = 'none';
      cancelBtn.style.display = 'flex';
      progressBox.style.display = 'block';

      armText.textContent = `[${st.arm_index}/${st.total_arms}] 評測中: ${st.current_arm}`;
      countText.textContent = `${st.query_index} / ${st.total_queries} 題 (${st.progress_percent}%)`;
      elapsedText.textContent = `耗時: ${st.elapsed_seconds}s`;
      bar.style.width = `${st.progress_percent}%`;

      if (!benchmarkPollInterval) {
        benchmarkPollInterval = setInterval(pollBenchmarkStatus, 1500);
      }
    } else if (st.status === 'completed') {
      if (benchmarkPollInterval) {
        clearInterval(benchmarkPollInterval);
        benchmarkPollInterval = null;
      }
      runBtn.style.display = 'flex';
      cancelBtn.style.display = 'none';
      progressBox.style.display = 'none';

      // Reload reports dropdown and select newly produced report
      if (st.latest_report_file) {
        await fetchBenchmarkReports(st.latest_report_file);
      }
    } else {
      // idle or error
      if (benchmarkPollInterval) {
        clearInterval(benchmarkPollInterval);
        benchmarkPollInterval = null;
      }
      runBtn.style.display = 'flex';
      cancelBtn.style.display = 'none';
      progressBox.style.display = 'none';
      if (st.error) {
        console.warn('Benchmark notice/error:', st.error);
      }
    }
  } catch (err) {
    console.error('Status poll error:', err);
  }
}

async function cancelBenchmarkRun() {
  const cancelBtn = document.getElementById('cancelBenchmarkBtn');
  cancelBtn.disabled = true;
  cancelBtn.textContent = '中斷中...';
  try {
    await fetch('/api/benchmark/run/cancel', { method: 'POST' });
  } catch (err) {
    alert('中斷請求失敗: ' + err.message);
  }
}

function toggleRawJson() {
  const sec = document.getElementById('rawJsonSection');
  sec.style.display = sec.style.display === 'none' ? 'block' : 'none';
  if (sec.style.display === 'block') {
    sec.scrollIntoView({ behavior: 'smooth' });
  }
}

function copyRawJson() {
  const pre = document.getElementById('rawJsonPre');
  const btn = document.getElementById('copyJsonBtn');
  navigator.clipboard.writeText(pre.textContent).then(() => {
    const originalText = btn.textContent;
    btn.textContent = '✅ 已複製至剪貼簿！';
    setTimeout(() => {
      btn.textContent = originalText;
    }, 2000);
  });
}

function switchDiagTab(tabId, evt) {
  const target = (evt && evt.target) || (typeof event !== 'undefined' ? event.target : null);
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-body').forEach(b => b.style.display = 'none');
  if (target && target.classList) target.classList.add('active');
  const el = document.getElementById(tabId);
  if (el) el.style.display = 'block';
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

function handleGalleryWheel(evt, el) {
  if (evt.deltaY !== 0) {
    evt.preventDefault();
    el.scrollLeft += evt.deltaY * 1.5;
  }
}

// Explicit window assignments to ensure all inline onclick handlers resolve
window.handleGalleryWheel = handleGalleryWheel;
window.toggleDiagnostics = toggleDiagnostics;
window.switchDevMode = switchDevMode;
window.loadBenchmarkDatasetsAndReports = loadBenchmarkDatasetsAndReports;
window.loadSelectedReport = loadSelectedReport;
window.filterQueryList = filterQueryList;
window.triggerBenchmarkRun = triggerBenchmarkRun;
window.pollBenchmarkStatus = pollBenchmarkStatus;
window.cancelBenchmarkRun = cancelBenchmarkRun;
window.toggleRawJson = toggleRawJson;
window.copyRawJson = copyRawJson;
window.switchDiagTab = switchDiagTab;
window.runDiagTokenize = runDiagTokenize;
window.runDiagBM25 = runDiagBM25;
window.runDiagDense = runDiagDense;
window.runDiagRerank = runDiagRerank;
window.handleSearch = handleSearch;
window.applyPreset = applyPreset;
window.clearSearch = clearSearch;
window.openModal = openModal;
window.closeModal = closeModal;
window.closeModalDirect = closeModalDirect;
window.updateParamLabels = updateParamLabels;
