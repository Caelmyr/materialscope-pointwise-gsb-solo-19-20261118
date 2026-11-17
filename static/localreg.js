/* 可审阅局部波段配准前端：叠看、对应点/区域圈定、候选审阅、冲突与冻结。 */
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const lr = {
    sessions: [], session: null, detail: null, draft: null, regionDraft: [],
    warpedPreview: false, shape: [0, 0, 0],
  };

  function api(path, options = {}) {
    return window.appHelpers.request(path, options);
  }
  function projectId() { return window.appHelpers.state.project.id; }
  function asset() { return window.appHelpers.state.asset; }
  function regBase() { return `/api/projects/${projectId()}/registration`; }
  function sessionBase() { return `${regBase()}/${lr.session}`; }
  function toast(msg, error) { window.appHelpers.toast(msg, error); }

  async function loadSessions(selectId) {
    const data = await api(regBase());
    lr.sessions = data.sessions;
    const select = $('lr-sessions');
    select.innerHTML = '<option value="">选择审阅会话</option>' + lr.sessions.map((s) =>
      `<option value="${s.id}">${esc(s.name)} · ${s.status === 'frozen' ? '已冻结' : '审阅中'} · ${s.point_count}点</option>`).join('');
    if (selectId) select.value = selectId;
  }

  async function openSession(id) {
    if (!id) { lr.session = null; lr.detail = null; renderFrozen(null); return; }
    lr.session = id;
    lr.detail = await api(sessionBase());
    lr.shape = lr.detail.shape;
    $('lr-name').value = lr.detail.name;
    $('lr-ref-band').value = String(lr.detail.reference_band);
    $('lr-max-disp').value = lr.detail.max_displacement;
    $('lr-policy').value = lr.detail.unconstrained_policy;
    $('lr-sessions').value = id;
    const frozen = lr.detail.status === 'frozen';
    // 复制审阅在冻结时仍可用——这是调整已冻结判断的唯一入口
    $('lr-clone').disabled = false;
    ['lr-new', 'lr-save-settings', 'lr-confirm-pair', 'lr-pending-pair', 'lr-freeze', 'lr-delete', 'lr-candidates', 'lr-finish-region', 'lr-cancel-region'].forEach((bid) => { $(bid).disabled = frozen; });
    ['lr-name', 'lr-ref-band', 'lr-moving-band', 'lr-max-disp', 'lr-policy', 'lr-blend-mode', 'lr-alpha', 'lr-template-radius', 'lr-search-radius', 'lr-ambiguity'].forEach((bid) => { $(bid).disabled = frozen; });
    document.querySelectorAll('input[name="lr-mode"]').forEach((r) => { r.disabled = frozen; });
    renderConflicts();
    renderPoints();
    renderRegions();
    renderFrozen(lr.detail);
    drawCanvases();
  }

  function bandOptions() {
    if (!lr.detail) return '';
    return lr.detail.bands.map((b, i) => `<option value="${i}">${i} · ${esc(b.name)}</option>`).join('');
  }

  function syncBandSelectors() {
    if (!lr.detail) return;
    const ref = lr.detail.reference_band;
    $('lr-ref-band').innerHTML = bandOptions();
    $('lr-ref-band').value = String(ref);
    const moving = $('lr-moving-band');
    const current = moving.value;
    moving.innerHTML = lr.detail.bands.map((b, i) => i === ref ? '' : `<option value="${i}">${i} · ${esc(b.name)}</option>`).join('');
    moving.value = current && Number(current) !== ref ? current : lr.detail.bands.findIndex((_, i) => i !== ref);
  }

  function movingBand() { return Number($('lr-moving-band').value); }
  function mode() { return document.querySelector('input[name="lr-mode"]:checked').value; }

  async function drawCanvases() {
    if (!lr.detail) return;
    syncBandSelectors();
    const sid = lr.session;
    const band = movingBand();
    const blend = $('lr-blend-mode').value;
    const alpha = $('lr-alpha').value;
    const warp = lr.warpedPreview ? '&warped=1' : '';
    const refUrl = `${sessionBase()}/annotations.png?band=${lr.detail.reference_band}`;
    const movUrl = `${sessionBase()}/blend.png?band=${band}&mode=${blend}&alpha=${alpha}${warp}`;
    await Promise.all([loadImage($('lr-ref-canvas'), refUrl), loadImage($('lr-mov-canvas'), movUrl)]);
    drawAnnotations($('lr-ref-canvas'), 'reference');
    drawAnnotations($('lr-mov-canvas'), 'moving');
  }

  function loadImage(canvas, url) {
    return api(url, { raw: true }).then(async (resp) => {
      const blob = await resp.blob();
      const bitmap = await createImageBitmap(blob);
      const maxW = 300;
      const scale = Math.min(1, maxW / bitmap.width);
      canvas.width = Math.round(bitmap.width * scale);
      canvas.height = Math.round(bitmap.height * scale);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      canvas.dataset.scale = scale;
    });
  }

  function toImageXY(event, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scale = Number(canvas.dataset.scale || 1);
    const x = (event.clientX - rect.left) / scale;
    const y = (event.clientY - rect.top) / scale;
    return [Math.max(0, Math.min(x, lr.shape[1] - 1)), Math.max(0, Math.min(y, lr.shape[0] - 1))];
  }

  function drawAnnotations(canvas, plane) {
    const ctx = canvas.getContext('2d');
    const scale = Number(canvas.dataset.scale || 1);
    const band = movingBand();
    ctx.save();
    ctx.scale(scale, scale);
    for (const region of lr.detail.regions) {
      ctx.strokeStyle = region.color; ctx.lineWidth = 1.5;
      const pts = region.kind === 'polygon' ? region.shape : rectToPolygon(region.shape);
      ctx.beginPath();
      pts.forEach((p, i) => i ? ctx.lineTo(p[0] + 0.5, p[1] + 0.5) : ctx.moveTo(p[0] + 0.5, p[1] + 0.5));
      ctx.stroke();
    }
    if (mode() === 'region' && lr.regionDraft.length) {
      ctx.strokeStyle = '#67dbc1'; ctx.setLineDash([4, 3]);
      ctx.beginPath();
      lr.regionDraft.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.stroke(); ctx.setLineDash([]);
    }
    for (const point of lr.detail.points) {
      if (point.band !== band) continue;
      const [x, y] = plane === 'reference' ? point.reference : point.moving;
      ctx.fillStyle = point.status === 'confirmed' ? '#f4b860' : '#9aafff';
      ctx.strokeStyle = plane === 'reference' ? '#ffffff' : '#ff7d7d';
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    if (lr.draft) {
      const p = plane === 'reference' ? lr.draft.reference : lr.draft.moving;
      if (p) { ctx.fillStyle = '#67dbc1'; ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2); ctx.fill(); }
    }
    ctx.restore();
  }

  function rectToPolygon([x0, y0, x1, y1]) { return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; }

  function canvasClick(canvas, plane) {
    canvas.addEventListener('click', async (event) => {
      if (!lr.detail || lr.detail.status === 'frozen') return;
      const [x, y] = toImageXY(event, canvas);
      if (mode() === 'region') {
        if (plane !== 'reference') return toast('材料区域在参考波段（左图）上圈定');
        lr.regionDraft.push([Math.round(x * 2) / 2, Math.round(y * 2) / 2]);
        $('lr-finish-region').hidden = lr.regionDraft.length < 3;
        $('lr-cancel-region').hidden = false;
        drawCanvases();
        return;
      }
      lr.draft = lr.draft || { band: movingBand(), reference: null, moving: null, candidates: [] };
      lr.draft.band = movingBand();
      lr.draft[plane] = [Math.round(x * 2) / 2, Math.round(y * 2) / 2];
      $('lr-draft').textContent = `草稿点：波段${lr.draft.band} 参考=${fmt(lr.draft.reference)} 待配=${fmt(lr.draft.moving)}`;
      drawCanvases();
    });
  }

  const fmt = (p) => p ? `(${p[0].toFixed(1)}, ${p[1].toFixed(1)})` : '未选';

  async function searchCandidates() {
    if (!lr.draft || !lr.draft.reference) return toast('请先在左图点击参考位置');
    try {
      const data = await api(`${sessionBase()}/candidates`, {
        method: 'POST',
        body: JSON.stringify({
          band: movingBand(), reference: lr.draft.reference,
          template_radius: Number($('lr-template-radius').value),
          search_radius: Number($('lr-search-radius').value),
          ambiguity_margin: Number($('lr-ambiguity').value),
        }),
      });
      lr.draft.candidates = data.candidates;
      const hint = data.translation_hint.available
        ? `平移提示 dx=${data.translation_hint.dx.toFixed(2)} dy=${data.translation_hint.dy.toFixed(2)}（不约束结果）`
        : '平移提示不可用：' + data.translation_hint.reason;
      $('lr-candidate-list').innerHTML = `<p class="muted">${esc(hint)}</p>` + data.candidates.map((c, i) =>
        `<div class="lr-candidate ${c.ambiguous ? 'ambiguous' : ''}"><button data-pick-candidate="${i}">采用</button>
         <small>#${i + 1} 坐标(${c.moving[0]}, ${c.moving[1]}) 相关分 ${c.score.toFixed(3)} ${c.ambiguous ? '⚠ 位置歧义' : ''}<br>${esc(c.reason)}</small></div>`).join('');
      document.querySelectorAll('[data-pick-candidate]').forEach((btn) => {
        btn.onclick = () => {
          const c = lr.draft.candidates[Number(btn.dataset.pickCandidate)];
          lr.draft.moving = c.moving;
          $('lr-draft').textContent = `草稿点：波段${lr.draft.band} 参考=${fmt(lr.draft.reference)} 待配=${fmt(lr.draft.moving)}（候选，需确认）`;
          drawCanvases();
        };
      });
    } catch (error) { toast(error.message, true); }
  }

  async function savePoint(status) {
    if (!lr.draft || !lr.draft.reference || !lr.draft.moving) return toast('需要先在两张图分别点出参考点和待配点');
    const payload = {
      revision: lr.detail.revision, band: lr.draft.band,
      reference: lr.draft.reference, moving: lr.draft.moving, status,
      candidates: lr.draft.candidates, note: status === 'pending' ? '人工保留未决' : '',
    };
    try {
      lr.detail = await api(`${sessionBase()}/points`, { method: 'POST', body: JSON.stringify(payload) });
      lr.draft = null; $('lr-draft').textContent = '尚未选择草稿点。';
      $('lr-candidate-list').innerHTML = '<p class="muted">已保存。可继续点选下一组对应点。</p>';
      await afterMutation();
    } catch (error) { toast(error.message, true); }
  }

  async function afterMutation() {
    await loadSessions(lr.session);
    await openSession(lr.session);
  }

  function renderPoints() {
    const band = movingBand();
    const rows = lr.detail.points.filter((p) => p.band === band);
    $('lr-point-list').innerHTML = rows.map((p) =>
      `<div class="lr-row ${p.status === 'pending' ? 'pending' : ''}">
        <small>${esc(p.status === 'confirmed' ? '确认' : '未决')} · 参考(${p.reference[0]}, ${p.reference[1]}) → 待配(${p.moving[0]}, ${p.moving[1]})${p.candidates.some((c) => c.ambiguous) ? ' · 含歧义候选' : ''}</small>
        <div class="row"><button data-toggle-point="${p.id}">${p.status === 'confirmed' ? '转未决' : '转确认'}</button><button data-delete-point="${p.id}">删除</button></div>
      </div>`).join('') || '<p class="muted">当前待配波段还没有对应点。</p>';
    document.querySelectorAll('[data-delete-point]').forEach((b) => {
      b.onclick = async () => {
        try { await api(`${sessionBase()}/points/${b.dataset.deletePoint}`, { method: 'DELETE', body: JSON.stringify({ revision: lr.detail.revision }) }); await afterMutation(); }
        catch (e) { toast(e.message, true); }
      };
    });
    document.querySelectorAll('[data-toggle-point]').forEach((b) => {
      b.onclick = async () => {
        const point = lr.detail.points.find((p) => p.id === b.dataset.togglePoint);
        try {
          await api(`${sessionBase()}/points/${point.id}`, { method: 'PATCH', body: JSON.stringify({ revision: lr.detail.revision, status: point.status === 'confirmed' ? 'pending' : 'confirmed' }) });
          await afterMutation();
        } catch (e) { toast(e.message, true); }
      };
    });
  }

  function renderRegions() {
    $('lr-region-list').innerHTML = lr.detail.regions.map((r) =>
      `<div class="lr-row"><small><b style="color:${r.color}">${esc(r.name)}</b> · ${r.kind} · ${r.shape.length}顶点</small>
      <button data-delete-region="${r.id}">删除区域</button></div>`).join('') || '<p class="muted">未圈定独立材料区域时，所有确认点参与同一个全局分段变形。</p>';
    document.querySelectorAll('[data-delete-region]').forEach((b) => {
      b.onclick = async () => {
        try { await api(`${sessionBase()}/regions/${b.dataset.deleteRegion}`, { method: 'DELETE', body: JSON.stringify({ revision: lr.detail.revision }) }); await afterMutation(); }
        catch (e) { toast(e.message, true); }
      };
    });
  }

  function renderConflicts() {
    const summary = lr.detail.conflict_summary;
    const box = $('lr-conflicts');
    if (!summary) { box.innerHTML = ''; return; }
    const blocking = summary.conflicts.filter((c) => c.severity === 'blocking');
    const pending = summary.conflicts.filter((c) => c.severity === 'pending');
    box.innerHTML = blocking.length
      ? `<div class="lr-blocking"><b>必须先处理的判断冲突（${blocking.length}）</b>${blocking.map((c) => `<div>• ${esc(c.message)}</div>`).join('')}</div>` : '';
    if (pending.length) box.innerHTML += `<div class="lr-pending"><b>保留未决（${pending.length}，不阻断，对应位置不会生成有效测量）</b>${pending.slice(0, 8).map((c) => `<div>• ${esc(c.message)}</div>`).join('')}${pending.length > 8 ? `<div>…另有${pending.length - 8}条</div>` : ''}</div>`;
    if (!blocking.length && !pending.length) box.innerHTML = '<div class="lr-ok">当前没有冲突；未覆盖到的位置仍按无约束策略处理。</div>';
  }

  function renderFrozen(detail) {
    const box = $('lr-frozen');
    if (!detail || detail.status !== 'frozen') { box.hidden = true; return; }
    box.hidden = false;
    const report = detail.warp_report || {};
    const bands = (report.bands || []).map((b) =>
      `<tr><td>${esc(b.band)}</td><td>${esc(b.status)}</td><td>${b.confirmed_points}</td><td>${b.warped_pixels}</td><td>${b.pending_pixels}</td><td>${b.unconstrained_pixels}</td></tr>`).join('');
    box.innerHTML = `<b>已冻结 · 结果影像：${esc(detail.result_asset_id)}</b>
      <p class="muted">参考带 ${report.reference_band} · 模型 ${esc(report.model)} · 确认点 ${report.confirmed_point_count} · 未决点 ${report.pending_point_count} · 区域 ${report.region_count}</p>
      <table class="lr-table"><thead><tr><th>波段</th><th>状态</th><th>确认点</th><th>已变形</th><th>未决</th><th>无约束保留</th></tr></thead><tbody>${bands}</tbody></table>
      <p class="muted">冻结判断与结果不可修改；调整请复制为新审阅，旧结果保持不变。</p>`;
  }

  function bind() {
    canvasClick($('lr-ref-canvas'), 'reference');
    canvasClick($('lr-mov-canvas'), 'moving');
    $('lr-new').onclick = async () => {
      if (!asset()) return toast('请先选择影像');
      try {
        const data = await api(`/api/projects/${projectId()}/assets/${asset().id}/registration`, {
          method: 'POST', body: JSON.stringify({ revision: window.appHelpers.state.project.revision, reference_band: Math.floor(asset().summary.band_count / 2) }),
        });
        await loadSessions(data.id); await openSession(data.id); toast('已创建局部配准审阅');
      } catch (e) { toast(e.message, true); }
    };
    $('lr-clone').onclick = async () => {
      if (!lr.session) return;
      try { const d = await api(`${sessionBase()}/clone`, { method: 'POST', body: JSON.stringify({ revision: lr.detail.revision }) }); await loadSessions(d.id); await openSession(d.id); }
      catch (e) { toast(e.message, true); }
    };
    $('lr-delete').onclick = async () => {
      if (!lr.session) return;
      try { await api(sessionBase(), { method: 'DELETE', body: JSON.stringify({ revision: lr.detail.revision }) }); lr.session = null; await loadSessions(); $('lr-panel').hidden = true; }
      catch (e) { toast(e.message, true); }
    };
    $('lr-save-settings').onclick = async () => {
      try {
        lr.detail = await api(sessionBase(), {
          method: 'PATCH', body: JSON.stringify({ revision: lr.detail.revision, name: $('lr-name').value.trim(), reference_band: Number($('lr-ref-band').value), max_displacement: Number($('lr-max-disp').value), unconstrained_policy: $('lr-policy').value }),
        }); await afterMutation(); toast('审阅设置已保存');
      } catch (e) { toast(e.message, true); }
    };
    $('lr-candidates').onclick = searchCandidates;
    $('lr-confirm-pair').onclick = () => savePoint('confirmed');
    $('lr-pending-pair').onclick = () => savePoint('pending');
    $('lr-toggle-warp').onclick = async () => { lr.warpedPreview = !lr.warpedPreview; await drawCanvases(); };
    $('lr-refresh-conflicts').onclick = async () => { lr.detail = await api(sessionBase()); renderConflicts(); };
    $('lr-finish-region').onclick = async () => {
      if (lr.regionDraft.length < 3) return;
      try {
        lr.detail = await api(`${sessionBase()}/regions`, { method: 'POST', body: JSON.stringify({ revision: lr.detail.revision, name: `材料区域 ${lr.detail.regions.length + 1}`, kind: 'polygon', vertices: lr.regionDraft, color: '#67dbc1' }) });
        lr.regionDraft = []; $('lr-finish-region').hidden = true; $('lr-cancel-region').hidden = true; await afterMutation();
      } catch (e) { toast(e.message, true); }
    };
    $('lr-cancel-region').onclick = () => { lr.regionDraft = []; $('lr-finish-region').hidden = true; $('lr-cancel-region').hidden = true; drawCanvases(); };
    $('lr-freeze').onclick = async () => {
      try {
        const data = await api(`${sessionBase()}/freeze`, { method: 'POST', body: JSON.stringify({ revision: lr.detail.revision }) });
        await loadSessions(data.session.id); await openSession(data.session.id);
        await window.appHelpers.openProject(projectId(), data.asset.id);
        toast('局部配准完成，结果影像已保存；未决位置未生成有效测量');
      } catch (e) { toast(e.message, true); await openSession(lr.session); }
    };
    $('lr-sessions').onchange = () => openSession($('lr-sessions').value);
    ['lr-moving-band', 'lr-blend-mode', 'lr-alpha'].forEach((id) => { $(id).onchange = drawCanvases; $(id).oninput = drawCanvases; });
    $('lr-ref-band').onchange = () => $('lr-save-settings').click();
    document.querySelectorAll('input[name="lr-mode"]').forEach((r) => { r.onchange = drawCanvases; });
  }

  window.localReg = {
    async enter() {
      if (!asset()) { $('lr-no-asset').hidden = false; $('lr-panel').hidden = true; return; }
      $('lr-no-asset').hidden = true; $('lr-panel').hidden = false;
      await loadSessions(lr.session);
      if (lr.session && lr.sessions.some((s) => s.id === lr.session)) await openSession(lr.session);
      else if (lr.sessions.length) await openSession(lr.sessions[0].id);
    },
  };

  document.addEventListener('DOMContentLoaded', bind);
})();
