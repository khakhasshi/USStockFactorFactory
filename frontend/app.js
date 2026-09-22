/* USStockFactorFactory 前端 — Vue3 全局构建 + ECharts */
const {
  createApp, ref, reactive, computed, onMounted, onUnmounted,
  onActivated, onDeactivated, watch, nextTick,
} = Vue;

const appState = reactive({
  experimentId: null,
  experimentVersion: 0,
  activeTab: localStorage.getItem("factorfactory.tab") || "dash",
  requestedTab: null,
  backtestDraft: "",
  switching: false,
  switchMessage: "",
});
const responseCache = new Map();
const inflightGets = new Map();
let apiCacheGeneration = 0;

async function api(path, opts = {}) {
  const method = (opts.method || "GET").toUpperCase();
  const requestGeneration = apiCacheGeneration;
  const cacheKey = method === "GET" ? `${requestGeneration}:${path}` : null;
  const ttl = opts.cacheTtl ?? 0;
  const { cacheTtl: _cacheTtl, ...fetchOptions } = opts;
  if (cacheKey && ttl > 0) {
    const hit = responseCache.get(cacheKey);
    if (hit && Date.now() - hit.time < ttl) return hit.value;
    if (inflightGets.has(cacheKey)) return inflightGets.get(cacheKey);
  }
  const request = (async () => {
    const res = await fetch("/api" + path, {
      headers: { "Content-Type": "application/json" },
      ...fetchOptions,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    const value = await res.json();
    if (method !== "GET") invalidateApiCache();
    if (cacheKey && ttl > 0 && requestGeneration === apiCacheGeneration) {
      responseCache.set(cacheKey, { time: Date.now(), value });
    }
    return value;
  })();
  if (cacheKey && ttl > 0) inflightGets.set(cacheKey, request);
  try { return await request; }
  finally { if (cacheKey) inflightGets.delete(cacheKey); }
}

function invalidateApiCache() {
  apiCacheGeneration += 1;
  responseCache.clear();
  inflightGets.clear();
}

async function activateExperiment(experimentId) {
  appState.switching = true;
  appState.switchMessage = "正在切换研究任务…";
  try {
    const result = await api(`/experiments/${experimentId}/activate`, { method: "POST" });
    appState.experimentId = Number(result.active_id || experimentId);
    appState.experimentVersion += 1;
    appState.switchMessage = `已切换到 ${result.experiment?.name || "研究任务"}`;
    return result;
  } finally {
    appState.switching = false;
    setTimeout(() => { appState.switchMessage = ""; }, 1200);
  }
}

function mountChart(el, option) {
  let c = echarts.getInstanceByDom(el);
  if (!c) c = echarts.init(el, "dark", { renderer: "canvas" });
  c.setOption(option, true);
  return c;
}

const DARK = { backgroundColor: "transparent", textStyle: { fontFamily: "SF Mono, Menlo, monospace" } };

/* ============ Dashboard ============ */
const Dashboard = {
  template: `
  <div>
    <div class="grid cols-4" style="margin-bottom:14px">
      <div class="card"><h3>引擎状态</h3>
        <div class="big-num" :style="{color: st.state==='running' ? 'var(--green)' : 'var(--red)'}">{{ st.state }}</div>
        <div class="sub">实验: {{ st.experiment?.name ?? '—' }} · {{ st.counts?.evaluation_protocol || '—' }} · 外层步 {{ st.outer_step }} · 内层评估 {{ st.inner_evals }}</div>
        <div style="margin-top:10px; display:flex; gap:8px">
          <button class="btn primary" @click="start" :disabled="st.state==='running'">启动 7×24</button>
          <button class="btn danger" @click="stop" :disabled="st.state!=='running'">停止</button>
        </div>
      </div>
      <div class="card"><h3>当前协议因子</h3><div class="big-num">{{ st.counts?.factors ?? '—' }}</div><div class="sub">全历史 {{ st.counts?.factors_all ?? '—' }} · 旧协议只读保留</div></div>
      <div class="card"><h3>当前协议节点</h3><div class="big-num">{{ st.counts?.nodes ?? '—' }}</div><div class="sub">全历史 {{ st.counts?.nodes_all ?? '—' }} · 不混入当前上下文</div></div>
      <div class="card"><h3>外层接受率</h3>
        <div class="big-num">{{ acceptRate }}</div>
        <div class="sub">{{ st.counts?.accepted ?? 0 }} / {{ st.counts?.outer_steps ?? 0 }} 当前协议步 · 全历史 {{ st.counts?.outer_steps_all ?? '—' }}</div>
      </div>
    </div>
    <div class="card" style="margin-bottom:14px">
      <div class="panel-title-row"><div><h3>并行任务与数据身份</h3><span class="sub">同一端口内独立 worker；历史数据按任务 ID 隔离</span></div><span class="tag blue">{{ (st.workers || []).length }} workers</span></div>
      <table><tr><th>任务</th><th>市场</th><th>实际模式</th><th>方向</th><th>状态</th><th>搜索健康</th><th>评价/预筛拒绝</th><th>本次效率</th></tr>
        <tr v-for="w in (st.workers || [])" :key="w.experiment_id"><td>{{ w.experiment_id }}</td><td>{{ w.task_config?.market || '—' }}</td><td>{{ w.effective_portfolio_mode || w.task_config?.portfolio_mode || '—' }}</td><td>{{ w.task_config?.direction_policy==='both_train_select' ? '双向训练 · 同分'+(Number(w.task_config?.direction || 1)>0?'+1':'-1') : (Number(w.task_config?.direction || 1)===1 ? '固定 +1' : '固定 -1') }}</td><td>{{ w.state }}</td><td><span class="tag" :class="w.search_health?.state==='healthy'?'green':w.search_health?'amber':'blue'">{{ w.search_health?.state || '等待样本' }}</span><div class="sub" v-if="w.search_health">重复 {{ ((w.search_health.recent_duplicate_rate||0)*100).toFixed(0) }}% · epoch {{ w.search_health.search_epoch||0 }}</div></td><td>{{ w.candidate_evaluations ?? w.inner_evals }} / 重复{{ w.pre_eval_duplicate_rejections || 0 }} / 预筛{{ w.pre_eval_signal_rejections || 0 }}</td><td>{{ Number(w.effective_evaluations_per_hour||0).toFixed(1) }}/h<div class="sub">重复浪费 {{ (Number(w.duplicate_waste_rate||0)*100).toFixed(0) }}% · 预筛 {{ (Number(w.signal_preflight_rejection_rate||0)*100).toFixed(0) }}%</div></td></tr>
      </table>
    </div>
    <div class="grid cols-2">
      <div class="card">
        <h3>外层 Meta-Score 步进 (候选 vs 在位)</h3>
        <div class="chart" ref="progressEl"></div>
      </div>
      <div class="card">
        <h3>在位 Miner 配置 (v{{ st.incumbent?.version_no ?? '—' }} · score {{ fmt(st.incumbent?.meta_score) }})</h3>
        <table v-if="st.incumbent">
          <tr v-for="(v,k) in st.incumbent.spec" :key="k"><td style="color:var(--muted)">{{ k }}</td><td>{{ v }}</td></tr>
        </table>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h3>实时日志</h3>
      <div class="logs" ref="logEl">
        <div v-for="(l,i) in st.logs" :key="i" :class="l.level">[{{ l.t }}] {{ l.msg }}</div>
      </div>
    </div>
  </div>`,
  setup() {
    const st = ref({ state: "…", logs: [] });
    const progressEl = ref(null), logEl = ref(null);
    let timer = null, refreshing = false;
    const fmt = (v) => (v == null ? "—" : Number(v).toFixed(4));
    const acceptRate = computed(() => {
      const c = st.value.counts;
      return c && c.outer_steps ? ((100 * c.accepted) / c.outer_steps).toFixed(0) + "%" : "—";
    });
    async function refresh() {
      if (refreshing) return;
      refreshing = true;
      try {
        const [status, prog] = await Promise.all([
          api("/engine/status", { cacheTtl: 900 }),
          api("/engine/progress", { cacheTtl: 900 }),
        ]);
        st.value = status;
        drawProgress(prog.steps);
        nextTick(() => { if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight; });
      } catch (e) { /* server booting */ }
      finally { refreshing = false; }
    }
    function drawProgress(steps) {
      if (!progressEl.value) return;
      mountChart(progressEl.value, {
        ...DARK,
        grid: { left: 50, right: 20, top: 30, bottom: 30 },
        tooltip: { trigger: "axis" },
        legend: { data: ["候选", "在位"], top: 0, textStyle: { color: "#8b949e" } },
        xAxis: { type: "category", data: steps.map((s) => "S" + s.step), axisLine: { lineStyle: { color: "#30363d" } } },
        yAxis: { type: "value", scale: true, splitLine: { lineStyle: { color: "#21262d" } } },
        series: [
          { name: "候选", type: "line", data: steps.map((s) => s.candidate), symbol: "circle", symbolSize: 7,
            itemStyle: { color: (p) => (steps[p.dataIndex].accepted ? "#3fb950" : "#f85149") }, lineStyle: { color: "#58a6ff" } },
          { name: "在位", type: "line", step: "end", data: steps.map((s) => s.incumbent), lineStyle: { color: "#d29922", type: "dashed" }, itemStyle: { color: "#d29922" } },
        ],
      });
    }
    async function start() { await api("/engine/start", { method: "POST", body: { mode: "v2" } }); refresh(); }
    async function stop() { await api("/engine/stop", { method: "POST" }); refresh(); }
    function startPolling() {
      refresh();
      clearInterval(timer);
      timer = setInterval(refresh, 3000);
    }
    function stopPolling() { clearInterval(timer); timer = null; }
    watch(
      () => appState.experimentVersion,
      () => { if (appState.activeTab === "dash") refresh(); },
    );
    onActivated(startPolling);
    onDeactivated(stopPolling);
    onUnmounted(stopPolling);
    return { st, progressEl, logEl, start, stop, fmt, acceptRate };
  },
};

/* ============ 研发树 ============ */
const ResearchTree = {
  template: `
  <div>
    <div class="card" style="margin-bottom:14px">
      <h3>Miner 版本演化 (外层)</h3>
      <table>
        <tr><th>版本 / 协议</th><th>状态</th><th>meta-score</th><th>反馈报告</th><th>结果反思</th><th>提案</th></tr>
        <tr v-for="v in data.versions" :key="v.id" class="clickable" @click="selectVersion(v.id)"
            :style="{background: v.id===selected ? '#1c2733' : ''}">
          <td>v{{ v.version_no }}<div class="sub">{{ v.evaluation_protocol || 'legacy' }}</div></td>
          <td><span class="tag" :class="{green: v.status==='incumbent', red: v.status==='rejected', amber: v.status==='superseded', blue: v.status==='candidate'}">{{ v.status }}</span></td>
          <td>{{ v.meta_score == null ? '—' : v.meta_score.toFixed(4) }}</td>
          <td>{{ v.feedback_summary?.attempts ?? '—' }} attempts<div class="sub">学习 {{ fmt(v.feedback_summary?.score_mean ?? v.feedback_summary?.seed_score_mean) }} · 门槛 {{ fmt(v.feedback_summary?.gate_score_mean) }} · pass {{ v.feedback_summary?.pass_rate == null ? '—' : (100*v.feedback_summary.pass_rate).toFixed(0)+'%' }}</div></td>
          <td>{{ v.reflection?.outcome?.hypothesis_result || v.reflection?.proposal?.hypothesis || '—' }}<div class="sub">{{ v.reflection?.outcome?.source || '—' }}</div></td>
          <td style="color:var(--muted); max-width:500px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">{{ v.note }}</td>
        </tr>
      </table>
    </div>
    <div class="card">
      <h3>内层搜索树 {{ selected ? '(v' + versionNo(selected) + ')' : '(全部, 最近800节点)' }} — 点击节点看表达式</h3>
      <div class="chart tall" ref="treeEl"></div>
      <div v-if="picked" style="margin-top:10px; padding:10px; border:1px solid var(--border); border-radius:6px">
        <div class="mono-expr">{{ picked.expression }}</div>
        <div class="sub">op={{ picked.op }} · source={{ picked.source }} · task={{ picked.task }} · seed={{ picked.seed ?? 'legacy' }} · protocol={{ picked.evaluation_protocol || 'legacy' }} · 学习分={{ fmt(picked.learning_score ?? picked.public_score) }} · 硬门槛分={{ fmt(picked.gate_score) }} · 选中方向={{ Number(picked.selected_direction || 1)===1?'+1 高值偏多':'-1 低值偏多' }} · 状态: {{ picked.status }}</div>
        <div v-if="picked.direction_selection?.candidates" class="sub" style="margin-top:6px">双向训练评价：+1 学习分 {{ fmt(picked.direction_selection.candidates['+1']?.learning_score) }} / 门槛 {{ fmt(picked.direction_selection.candidates['+1']?.gate_score) }}；-1 学习分 {{ fmt(picked.direction_selection.candidates['-1']?.learning_score) }} / 门槛 {{ fmt(picked.direction_selection.candidates['-1']?.gate_score) }}</div>
        <div v-if="picked.feedback_summary?.failure_reasons?.length" class="bad-text" style="margin-top:8px">评价反馈：{{ picked.feedback_summary.failure_reasons.join('；') }}</div>
        <div v-if="picked.feedback_summary?.improvement_targets?.length" class="sub" style="margin-top:6px">下一步：{{ picked.feedback_summary.improvement_targets.join('；') }}</div>
        <details v-if="picked.feedback_summary || picked.proposal_meta" class="ops-details">
          <summary>查看本节点反馈信封与提案反思</summary>
          <pre>{{ JSON.stringify({proposal:picked.proposal_meta, feedback:picked.feedback_summary}, null, 2) }}</pre>
        </details>
      </div>
    </div>
  </div>`,
  setup() {
    const data = ref({ versions: [], nodes: [] });
    const selected = ref(null), picked = ref(null);
    const treeEl = ref(null);
    let timer = null, refreshing = false;
    const fmt = (value) => value == null ? "—" : Number(value).toFixed(4);
    const versionNo = (id) => data.value.versions.find((v) => v.id === id)?.version_no;
    async function refresh() {
      if (refreshing) return;
      refreshing = true;
      const q = selected.value ? "?miner_version_id=" + selected.value : "";
      try {
        data.value = await api("/tree" + q, { cacheTtl: 1200 });
        draw();
      } finally { refreshing = false; }
    }
    function buildForest(nodes) {
      const byId = {}, roots = [];
      nodes.forEach((n) => (byId[n.id] = { ...n, children: [] }));
      nodes.forEach((n) => {
        if (n.parent_id && byId[n.parent_id]) byId[n.parent_id].children.push(byId[n.id]);
        else roots.push(byId[n.id]);
      });
      const conv = (n) => ({
        name: "#" + n.id,
        value: n.public_score,
        raw: n,
        itemStyle: { color: n.status === "error" ? "#f85149" : n.public_score > 0.3 ? "#3fb950" : n.public_score > 0.1 ? "#d29922" : "#58a6ff" },
        children: n.children.map(conv),
      });
      return { name: "root", itemStyle: { color: "#30363d" }, children: roots.map(conv) };
    }
    function draw() {
      if (!treeEl.value) return;
      const chart = mountChart(treeEl.value, {
        ...DARK,
        tooltip: { formatter: (p) => p.data.raw ? `${p.data.name}<br/>learning=${(p.data.raw.learning_score ?? p.data.raw.public_score ?? 0).toFixed(3)} · gate=${(p.data.raw.gate_score ?? 0).toFixed(3)} · dir=${Number(p.data.raw.selected_direction || 1)>0?'+1':'-1'}<br/>${p.data.raw.expression?.slice(0, 60)}` : "" },
        series: [{
          type: "tree", data: [buildForest(data.value.nodes)], layout: "orthogonal", orient: "LR",
          top: 10, bottom: 10, left: 40, right: 120, symbol: "circle", symbolSize: 12,
          initialTreeDepth: -1, roam: true,
          label: { color: "#8b949e", fontSize: 10, position: "top" },
          leaves: { label: { position: "right" } },
          lineStyle: { color: "#30363d", curveness: 0.4 },
        }],
      });
      chart.off("click");
      chart.on("click", (p) => { if (p.data.raw) picked.value = p.data.raw; });
    }
    function selectVersion(id) { selected.value = selected.value === id ? null : id; refresh(); }
    function startPolling() {
      refresh();
      clearInterval(timer);
      timer = setInterval(refresh, 6000);
    }
    function stopPolling() { clearInterval(timer); timer = null; }
    watch(() => appState.experimentVersion, () => {
      selected.value = null;
      picked.value = null;
      if (appState.activeTab === "tree") refresh();
    });
    onActivated(startPolling);
    onDeactivated(stopPolling);
    onUnmounted(stopPolling);
    return { data, selected, picked, treeEl, selectVersion, versionNo, fmt };
  },
};

/* ============ 因子库 ============ */
const FactorLibrary = {
  template: `
  <div>
    <div class="card" style="margin-bottom:14px">
      <h3>手动评估 / 录入表达式</h3>
      <div class="form-row">
        <div style="flex:3"><input v-model="evalExpr" placeholder="例: -rank(ts_delta(close, 20))" /></div>
        <div><select v-model.number="evalUniv"><option :value="500">Top500</option><option :value="1500">Top1500</option></select></div>
        <div><select v-model.number="evalHzn"><option v-for="h in [1,5,10,20]" :key="h" :value="h">{{h}}日</option></select></div>
        <div><button class="btn primary" @click="runEval" :disabled="evaling">{{ evaling ? '评估中…' : '评估' }}</button></div>
      </div>
      <div v-if="evalResult" style="margin-top:10px">
        <table><tr><th></th><th>RankIC均值</th><th>RankICIR</th><th>期一致性</th><th>换手</th><th>综合分</th></tr>
          <tr v-for="(m,k) in {public: evalResult.public, gate: evalResult.gate}" :key="k">
            <td>{{ k==='public' ? 'PUBLIC(训练可见)' : 'GATE(门禁)' }}</td>
            <td>{{ f(m.rank_ic_mean ?? m.ic_mean) }}</td><td>{{ f(m.rank_icir ?? m.icir) }}</td><td>{{ f(m.era_consistency) }}</td><td>{{ f(m.turnover) }}</td><td><b>{{ f(m.score) }}</b></td>
          </tr></table>
      </div>
      <div v-if="evalErr" style="color:var(--red); margin-top:8px">{{ evalErr }}</div>
    </div>
    <div class="card">
      <h3>研究因子记录 ({{ factors.length }})</h3>
      <div class="sub" style="margin-bottom:10px">训练通过只代表进入任务专属研究记录；正式晋级还需隔离层、事件回测、输入冻结、DSR/PBO、当前评级与双盲审查全部通过。这里的相关性指标为 Spearman RankIC，旧 IC 字段只作兼容读取。</div>
      <table>
        <tr><th>名称</th><th>表达式</th><th>状态</th><th>任务</th><th>PUB RankICIR</th><th>GATE RankICIR</th><th>GATE 分</th><th>时间</th></tr>
        <tr v-for="fa in factors" :key="fa.id" class="clickable" @click="open(fa)">
          <td>{{ fa.name }}</td>
          <td class="mono-expr" style="max-width:380px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">{{ fa.expression }}</td>
          <td><span class="tag" :class="{green: fa.evidence_state?.formal_factor, amber: !fa.evidence_state?.formal_factor, red: fa.raw_status==='retired'}">{{ fa.evidence_state?.label || fa.status }}</span></td>
          <td>{{ fa.task }}</td>
          <td>{{ f(fa.public?.rank_icir ?? fa.public?.icir) }}</td><td>{{ f(fa.gate?.rank_icir ?? fa.gate?.icir) }}</td><td>{{ f(fa.gate?.score) }}</td>
          <td class="sub">{{ fa.created_at?.slice(5,16) }}</td>
        </tr>
      </table>
    </div>

    <div class="drawer" v-if="detail">
      <button class="btn close" @click="detail=null">✕ 关闭</button>
      <h2 style="margin-bottom:6px">{{ detail.factor.name }}</h2>
      <div class="mono-expr" style="margin-bottom:8px">{{ detail.factor.expression }}</div>
      <div class="sub" style="margin-bottom:6px">{{ detail.factor.hypothesis }}</div>
      <div class="protocol-card" style="margin-bottom:12px"><b>{{ detail.factor.evidence_state?.label }}</b><span v-if="!detail.factor.evidence_state?.formal_factor">不是正式因子；待完成：{{ detail.factor.evidence_state?.promotion_blockers?.join('、') }}</span></div>
      <div style="display:flex; gap:8px; margin-bottom:14px">
        <button class="btn" v-for="s in ['library-admitted','paper','retired']" :key="s" @click="setStatus(s)">标记 {{ s }}</button>
      </div>
      <div class="card" style="margin-bottom:12px">
        <h3>分层 · 分期 RankIC 步进 (含隔离层)</h3>
        <div class="chart" ref="eraEl"></div>
      </div>
      <div class="card">
        <h3>各层指标</h3>
        <table><tr><th>层</th><th>RankIC均值</th><th>RankICIR</th><th>一致性</th><th>综合分</th></tr>
          <tr v-for="(m,layer) in detail.layers" :key="layer">
            <td>{{ layer }}</td><td>{{ f(m.rank_ic_mean ?? m.ic_mean) }}</td><td>{{ f(m.rank_icir ?? m.icir) }}</td><td>{{ f(m.era_consistency) }}</td><td>{{ f(m.score) }}</td>
          </tr></table>
      </div>
    </div>
  </div>`,
  setup() {
    const factors = ref([]), detail = ref(null);
    const evalExpr = ref(""), evalUniv = ref(500), evalHzn = ref(5), evalResult = ref(null), evalErr = ref(""), evaling = ref(false);
    const eraEl = ref(null);
    let timer = null;
    const f = (v) => (v == null ? "—" : Number(v).toFixed(3));
    async function refresh() { factors.value = (await api("/factors")).factors; }
    async function open(fa) {
      detail.value = await api(`/factors/${fa.id}/detail`);
      nextTick(drawEra);
    }
    function drawEra() {
      if (!eraEl.value || !detail.value) return;
      const eras = detail.value.eras || [];
      const colors = { INNER_PUBLIC: "#58a6ff", META_TRAIN: "#d29922", META_HOLDOUT: "#a371f7", FACTOR_VAULT: "#f85149" };
      mountChart(eraEl.value, {
        ...DARK,
        grid: { left: 55, right: 20, top: 30, bottom: 40 },
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: eras.map((e) => e.era), axisLabel: { rotate: 45, color: "#8b949e" } },
        yAxis: { type: "value", name: "era RankIC", splitLine: { lineStyle: { color: "#21262d" } } },
        series: [{
          type: "bar",
          data: eras.map((e) => ({ value: e.rank_ic_mean ?? e.ic_mean, itemStyle: { color: colors[e.layer] || "#8b949e" } })),
        }],
        legend: { show: false },
        graphic: Object.entries(colors).map(([k, c], i) => ({
          type: "text", right: 10, top: 8 + i * 16, style: { text: "■ " + k, fill: c, fontSize: 11 },
        })),
      });
    }
    async function runEval() {
      evaling.value = true; evalErr.value = ""; evalResult.value = null;
      try { evalResult.value = await api("/factors/evaluate", { method: "POST", body: { expression: evalExpr.value, universe_n: evalUniv.value, horizon: evalHzn.value } }); }
      catch (e) { evalErr.value = e.message; }
      evaling.value = false;
    }
    async function setStatus(s) {
      await api(`/factors/${detail.value.factor.id}/status`, { method: "POST", body: { status: s } });
      refresh();
      detail.value.factor.status = s;
    }
    onMounted(() => { refresh(); timer = setInterval(refresh, 8000); });
    onUnmounted(() => clearInterval(timer));
    return { factors, detail, open, f, evalExpr, evalUniv, evalHzn, evalResult, evalErr, evaling, runEval, setStatus, eraEl };
  },
};

/* ============ 因子研究工作台 ============ */
const FactorLibraryWorkbench = {
  template: `
  <div class="factor-workbench">
    <div class="card" style="margin-bottom:14px">
      <div class="panel-title-row"><div><div class="eyebrow">LIVE-RANKED FACTOR RESEARCH</div><h1>因子研究资产库</h1><span class="sub">研究评估 V4.2 在训练安全层双向选优并冻结方向；Rating V4.3 使用 2020 至最新交易日完整窗口，HOLDOUT/Vault 仍作为独立硬门槛，且不反馈给 LLM。</span></div><div><span class="tag blue">{{ factors.length }} 条研究记录</span> <span class="tag green">{{ factors.filter(f=>f.evidence_state?.formal_factor).length }} 个正式研究因子</span> <span class="tag amber">{{ factors.filter(f=>!f.evidence_state?.formal_factor).length }} 个非正式候选</span> <span class="tag green">{{ groupStats.groups || 0 }} 个结构组</span> <span class="tag amber">NON_PIT_RESEARCH</span></div></div>
      <div class="form-row" style="margin-top:14px">
        <input style="flex:3" v-model="query" @keyup.enter="refresh" placeholder="搜索名称、表达式、经济学假设…" />
        <select v-model="status"><option value="">全部生命周期</option><option value="discovery_only">F1 · discovery_only</option><option value="research_pass">F2 · research_pass</option><option value="oos_pass">F3 · oos_pass</option><option value="paper_candidate">F4 · paper_candidate</option><option value="live_candidate_non_pit">F5 · live_candidate_non_pit</option><option value="legacy_unreviewed">旧协议未审计</option><option value="invalid_provenance">来源无效</option><option value="configuration_changed_requires_reaudit">配置变更待复审</option></select>
        <select v-model="groupFilter"><option value="">全部相似组</option><option v-for="g in groups" :key="g.id" :value="g.id">{{ g.id }} · {{ familyLabel(g.family) }} · {{ g.size }}个</option></select>
        <select v-model="sort"><option value="live_rank">按实盘排序</option><option value="score">按连续学习分</option><option value="grade">按实战等级</option><option value="icir">按 RankICIR</option><option value="created">按最新</option></select>
        <button class="btn" @click="refresh">刷新</button><button class="btn primary" @click="compare" :disabled="selected.length<2">比较 {{ selected.length }} 个</button>
      </div>
      <div class="similarity-summary">
        <span>SimHash LSH + 加权 Jaccard</span>
        <b>{{ groupStats.duplicate_groups || 0 }}</b><small>个重复簇</small>
        <b>{{ pct(groupStats.redundancy_ratio) }}</b><small>结构冗余率</small>
        <button class="text-btn" @click="groupFilter=''">清除分组筛选</button>
      </div>
      <div class="rank-calibration" :class="rankingDiagnostics.status">
        <div><span>排序校准</span><b>{{ calibrationLabel(rankingDiagnostics.status) }}</b><small>{{ rankingDiagnostics.message || '等待 V4 审计样本' }}</small></div>
        <template v-if="rankingDiagnostics.metrics">
          <div><span>Rank ↔ Vault</span><b>{{ f(rankingDiagnostics.metrics.spearman_score_vs_vault_return) }}</b><small>Spearman</small></div>
          <div><span>Top 组盈利率</span><b>{{ pct(rankingDiagnostics.metrics.top_quartile_positive_rate) }}</b><small>Bottom {{ pct(rankingDiagnostics.metrics.bottom_quartile_positive_rate) }}</small></div>
          <div><span>分组单调性</span><b>{{ f(rankingDiagnostics.metrics.bucket_monotonicity) }}</b><small>{{ rankingDiagnostics.sample_size }} 个冻结样本</small></div>
        </template>
        <div v-else><span>有效样本</span><b>{{ rankingDiagnostics.sample_size || 0 }} / {{ rankingDiagnostics.minimum_sample || 8 }}</b><small>不足时拒绝给出校准结论</small></div>
      </div>
    </div>

    <div class="grid cols-2" v-if="comparison">
      <div class="card">
        <div class="panel-title-row"><h2>V4.2 冻结方向训练层复评</h2><span class="tag amber">{{ comparison.portfolio_mode }}</span></div>
        <table><tr><th>表达式</th><th>方向</th><th>PUB RankICIR</th><th>GATE RankICIR</th><th>GATE 费后 Sharpe</th><th>学习分</th></tr>
          <tr v-for="r in comparison.results" :key="r.expression"><td class="mono-expr">{{ r.expression }}</td><td>{{ Number(r.direction || 1)>0?'+1':'-1' }}</td><td>{{ f(r.public?.rank_icir ?? r.public?.icir) }}</td><td>{{ f(r.gate?.rank_icir ?? r.gate?.icir) }}</td><td>{{ f(layerSharpe(r.gate)) }}</td><td><b>{{ f(r.discovery?.score) }}</b></td></tr></table>
      </div>
      <div class="card"><h2>横截面冗余检查</h2><div class="sub">{{ comparison.correlation?.date }} · {{ comparison.correlation?.n }} 只股票</div>
        <table><tr><th></th><th v-for="(_,i) in comparison.correlation.matrix" :key="i">F{{ i+1 }}</th></tr>
          <tr v-for="(row,i) in comparison.correlation.matrix" :key="i"><th>F{{ i+1 }}</th><td v-for="(v,j) in row" :key="j" :class="Math.abs(v)>=0.8 && i!==j ? 'corr-high' : ''">{{ v.toFixed(2) }}</td></tr></table>
        <div class="sub" style="margin-top:8px">相关系数绝对值 ≥ 0.80 标红，表示候选可能是同一风险暴露的重复表达。</div>
      </div>
    </div>

    <div class="card">
      <div class="panel-title-row"><h2>研究资产</h2><span class="sub">{{ visibleFactors.length }} 条 · 点击行读取详情与相似因子</span></div>
      <table><tr><th><input type="checkbox" @change="toggleAll" /></th><th>名称</th><th>冻结评级</th><th>判断</th><th>结构组</th><th>表达式</th><th>方向</th><th>协议</th><th>等级</th><th>评级 Sharpe LCB</th><th>成本缓冲</th><th>学习分</th><th>来源</th></tr>
        <tr v-for="fa in visibleFactors" :key="fa.id" class="clickable" @click="open(fa)">
          <td @click.stop><input type="checkbox" :value="fa.id" v-model="selected" /></td><td><b>{{ fa.name }}</b></td>
          <td><div class="live-rank-cell"><b>{{ !fa.ranking?.current || fa.ranking?.score == null ? '—' : Number(fa.ranking.score).toFixed(1) }}</b><small v-if="fa.ranking?.position">#{{ fa.ranking.position }}</small></div></td>
          <td><span class="tag" :class="rankStatusClass(fa.ranking?.status)">{{ rankStatusLabel(fa.ranking?.status) }}</span></td>
          <td><button class="group-chip" @click.stop="groupFilter=groupFor(fa.id)">{{ groupFor(fa.id) || '—' }}</button></td>
          <td class="mono-expr factor-expression">{{ fa.expression }}</td>
          <td><span class="tag">{{ Number(fa.research_meta?.direction || 1)>0?'+1':'-1' }}</span></td>
          <td><span class="tag" :class="String(fa.evaluation_protocol || '').startsWith('v4.') ? 'green' : 'amber'">{{ fa.evaluation_protocol }}</span></td>
          <td><b :class="gradeClass(fa.eligibility?.grade)">{{ fa.eligibility?.grade || '—' }}</b><small :class="fa.evidence_state?.formal_factor?'good-text':'sub'">{{ fa.evidence_state?.label }}</small></td>
          <td>{{ fa.ranking?.current ? f(fa.ranking?.evidence?.rating_sharpe_lcb) : '—' }}</td><td>{{ !fa.ranking?.current || fa.ranking?.evidence?.cost_cushion_multiple == null ? '—' : f(fa.ranking.evidence.cost_cushion_multiple) + '×' }}</td><td><b>{{ f(fa.public?.score) }}</b></td>
          <td><span class="tag" :class="fa.provenance_status?.includes('invalid') ? 'red' : ''">{{ fa.provenance_status }}</span></td>
        </tr></table>
      <div v-if="!visibleFactors.length" class="selector-empty"><h2>当前筛选没有结果</h2><p>降低筛选条件，或清除相似组筛选。</p></div>
    </div>

    <div class="drawer" v-if="detail">
      <button class="btn close" @click="detail=null">✕ 关闭</button>
      <div class="panel-title-row"><div><h2>{{ detail.factor.name }}</h2><div class="sub">{{ detail.factor.evidence_state?.label }} · {{ detail.factor.lifecycle_stage }} · {{ detail.factor.provenance_status }}</div></div><div><span class="tag">{{ Number(detail.factor.research_meta?.direction || 1)>0?'+1 高值偏多':'-1 低值偏多' }}</span> <span class="tag" :class="String(detail.factor.evaluation_protocol || '').startsWith('v4.')?'green':'amber'">{{ detail.factor.evaluation_protocol }}</span> <span class="tag amber">NON_PIT</span></div></div>
      <div v-if="!detail.factor.evidence_state?.formal_factor" class="protocol-card"><b>研究候选，不是正式因子</b><span>训练通过不能代替隔离层审计。晋级阻断项：{{ detail.factor.evidence_state?.promotion_blockers?.join('、') }}</span></div>
      <div class="expression-display">
        <label class="latex-toggle"><input type="checkbox" v-model="showLatex" /> 以 Web LaTeX 渲染 DSL</label>
        <div v-if="showLatex" ref="latexEl" class="latex-expression"></div>
        <div v-else class="mono-expr">{{ detail.factor.expression }}</div>
        <div class="sub">DSL: <code>{{ detail.factor.expression }}</code></div>
      </div>
      <p class="sub">{{ detail.factor.hypothesis }}</p>
      <div v-if="detail.factor.research_meta?.direction_selection?.candidates" class="protocol-card"><b>双向训练选择</b><span>+1 学习 {{ f(detail.factor.research_meta.direction_selection.candidates['+1']?.learning_score) }} / 门槛 {{ f(detail.factor.research_meta.direction_selection.candidates['+1']?.gate_score) }}；-1 学习 {{ f(detail.factor.research_meta.direction_selection.candidates['-1']?.learning_score) }} / 门槛 {{ f(detail.factor.research_meta.direction_selection.candidates['-1']?.gate_score) }}</span><small>选中 {{ Number(detail.factor.research_meta.direction || 1)>0?'+1':'-1' }}，后续隔离层与实盘工具沿用冻结方向</small></div>
      <div class="live-rank-hero" v-if="detail.factor.ranking?.available && detail.factor.ranking?.current">
        <div class="live-rank-score"><span>冻结评级分</span><b>{{ Number(detail.factor.ranking.score).toFixed(1) }}</b><small>{{ detail.factor.ranking.rating_window?.start || '2020-01-01' }} 至 {{ detail.factor.ranking.rating_window?.end || '最新' }} · Vault 门槛 {{ detail.factor.ranking.vault_seal }}</small></div>
        <div class="live-rank-evidence">
          <div><span>评级 Sharpe LCB</span><b>{{ f(detail.factor.ranking.evidence?.rating_sharpe_lcb ?? detail.factor.ranking.evidence?.holdout_sharpe_lcb) }}</b></div>
          <div><span>评级年化收益 LCB</span><b>{{ pct(detail.factor.ranking.evidence?.rating_ann_return_lcb ?? detail.factor.ranking.evidence?.holdout_ann_return_lcb) }}</b></div>
          <div><span>评级收益 HAC t</span><b>{{ f(detail.factor.ranking.evidence?.rating_return_hac_t ?? detail.factor.ranking.evidence?.holdout_return_hac_t) }}</b></div>
          <div><span>成本缓冲</span><b>{{ f(detail.factor.ranking.evidence?.cost_cushion_multiple) }}×</b></div>
        </div>
        <div class="rank-components">
          <div v-for="(value,key) in detail.factor.ranking.components" :key="key"><span>{{ componentLabel(key) }}</span><i><em :style="{width:(Number(value)*100).toFixed(0)+'%'}"></em></i><b>{{ (Number(value)*100).toFixed(0) }}</b></div>
        </div>
        <div class="sub">冻结评级是 2020 至最新交易日的全窗口 NON-PIT 统计，不是独立样本外证明；HOLDOUT/Vault 继续独立决定硬门槛，评级结果不会反馈给两层 LLM。</div>
        <div v-if="detail.factor.ranking.warnings?.length" class="failure-list"><b>排序警告</b><ul><li v-for="warning in detail.factor.ranking.warnings" :key="warning">{{ warning }}</li></ul></div>
      </div>
      <div v-else class="warn-banner">尚无当前 Rating V4.3 冻结评级。旧评级仅保留为历史记录，请运行完整审计。</div>
      <div class="card similarity-detail" v-if="detail.similarity">
        <div class="panel-title-row"><div><h3>结构近邻</h3><span class="sub">{{ detail.similarity.group_id || '单因子组' }} · 不读取未来收益</span></div><span class="tag blue">{{ detail.similarity.nearest?.length || 0 }} 个近邻</span></div>
        <div class="similar-factor-list">
          <button v-for="row in detail.similarity.nearest" :key="row.id" @click="openSimilar(row)"><span>{{ row.name }}</span><code>{{ row.expression }}</code><b>{{ (row.similarity*100).toFixed(0) }}%</b></button>
        </div>
      </div>
      <div v-if="detail.factor.validation?.source_provenance_warning" class="warn-banner">{{ detail.factor.validation.source_provenance_warning }}</div>
      <div class="card" v-if="detail.factor.validation">
        <div class="panel-title-row"><div><h3>正式晋级证据链</h3><span class="sub">向量分数不等于可交易证据；缺少原始试验路径时 DSR/PBO 不会以代理数据补成通过。</span></div><span class="tag" :class="detail.factor.evidence_state?.formal_factor?'green':'amber'">{{ detail.factor.evidence_state?.formal_factor ? '全部通过' : '非正式候选' }}</span></div>
        <div class="metric-strip">
          <div class="metric-card"><span>事件回测硬门槛</span><b :class="evidenceClass(auditEvidence.event)">{{ evidenceStatus(auditEvidence.event) }}</b><small>{{ auditEvidence.event.protocol || '尚无事件审计' }}</small></div>
          <div class="metric-card"><span>不可变输入</span><b :class="auditEvidence.provenance.immutable_inputs_available?'ok-text':'bad-text'">{{ auditEvidence.provenance.immutable_inputs_available?'已冻结':'缺少完整冻结' }}</b><small>{{ auditEvidence.provenance.actual_start || '—' }} 至 {{ auditEvidence.provenance.actual_end || '—' }}</small></div>
          <div class="metric-card"><span>DSR / PBO</span><b :class="evidenceClass(auditEvidence.overfit)">{{ evidenceStatus(auditEvidence.overfit) }}</b><small>DSR {{ pct(auditEvidence.overfit.dsr?.dsr_probability) }} · PBO {{ pct(auditEvidence.overfit.pbo?.pbo) }}</small></div>
        </div>
        <div class="sub">实际 {{ auditEvidence.provenance.actual_sessions ?? '—' }} 个交易日 · Git {{ auditEvidence.provenance.panel?.code?.git_commit || '—' }}<br/>代码 SHA256：<code>{{ auditEvidence.provenance.panel?.code?.code_sha256 || '—' }}</code><br/>面板 SHA256：<code>{{ auditEvidence.provenance.panel?.data_sha256 || '—' }}</code><br/>表达式 SHA256：<code>{{ auditEvidence.provenance.expression_sha256 || '—' }}</code></div>
        <table v-if="auditEvidence.event.windows"><tr><th>事件窗口</th><th>实际日期</th><th>Sharpe</th><th>MDD</th><th>日均换手</th><th>状态 / 阻断</th></tr><tr v-for="(row,key) in auditEvidence.event.windows" :key="key"><td>{{ key }}</td><td>{{ row.actual_start || '—' }} 至 {{ row.actual_end || '—' }}</td><td>{{ f(row.stats?.sharpe) }}</td><td>{{ pct(row.stats?.max_dd) }}</td><td>{{ pct(row.stats?.avg_daily_turnover) }}</td><td><span :class="evidenceClass(row)">{{ evidenceStatus(row) }}</span><small>{{ row.failure_reasons?.join('；') }}</small></td></tr></table>
        <div class="sub" v-if="auditEvidence.overfit.registered_trials != null">登记 {{ auditEvidence.overfit.registered_trials }} 次 · 已评价 {{ auditEvidence.overfit.evaluated_trials ?? '—' }} 次 · 有原始路径 {{ auditEvidence.overfit.raw_evidence_trials ?? '—' }} 次 · 缺失 {{ auditEvidence.overfit.missing_evidence_trials ?? '—' }} 次 · 双向检验 {{ auditEvidence.overfit.attempted_direction_trials ?? '—' }} 次 · 有效试验上界 {{ auditEvidence.overfit.effective_trials ?? '—' }}<br/>门槛：DSR ≥ {{ pct(auditEvidence.overfit.thresholds?.dsr_min) }}，PBO ≤ {{ pct(auditEvidence.overfit.thresholds?.pbo_max) }}</div>
        <div class="failure-list" v-if="auditEvidence.overfit.reasons?.length"><b>过拟合治理阻断</b><ul><li v-for="reason in auditEvidence.overfit.reasons" :key="reason">{{ reason }}</li></ul></div>
        <div class="failure-list" v-if="auditEvidence.event.failure_reasons?.length"><b>事件审计阻断</b><ul><li v-for="reason in auditEvidence.event.failure_reasons" :key="reason">{{ reason }}</li></ul></div>
      </div>
      <div class="card audit-controls">
        <div class="panel-title-row"><div><h3>完整审计 + Rating V4.3</h3><span class="sub">冻结面板、代码、实际日期与方向；隔离标签采用 purge/embargo，向量通过后必须通过事件回测与 DSR/PBO 门槛。手动翻向作为新假设重审。</span></div><button class="btn primary" @click="runAudit" :disabled="auditing">{{ auditing ? '审计中…' : '运行完整审计' }}</button></div>
        <div class="form-row"><div><label>股票池</label><input type="number" v-model.number="auditForm.universe_n" /></div><div><label>持有期</label><select v-model.number="auditForm.horizon"><option :value="1">1日</option><option :value="5">5日</option><option :value="10">10日</option><option :value="20">20日</option></select></div><div><label>冻结方向</label><select v-model.number="auditForm.direction"><option :value="1">+1 高值偏多</option><option :value="-1">-1 低值偏多</option></select></div><div><label>基础成本 bps</label><input type="number" v-model.number="auditForm.cost_bps" /></div><div><label>目标资金规模</label><input type="number" v-model.number="auditForm.target_capital" /></div></div>
        <div v-if="auditErr" style="color:var(--red)">{{ auditErr }}</div>
      </div>
      <div class="card" v-if="detail.factor.validation?.layers">
        <div class="panel-title-row"><h3>四层隔离指标 + 冻结评级窗口</h3><div><span class="grade-pill" :class="gradeClass(detail.factor.eligibility?.grade)">{{ detail.factor.eligibility?.grade }}</span> <span class="tag">{{ detail.factor.eligibility?.stage }}</span></div></div>
        <table><tr><th>层</th><th>费后 Sharpe</th><th>Sharpe LCB</th><th>年化 LCB</th><th>收益 t</th><th>成本盈亏平衡</th><th>盈利 era</th><th>压力最差</th></tr>
          <tr v-for="(m,k) in detail.factor.validation.layers" :key="k"><td>{{ k.toUpperCase() }}</td><td>{{ f(layerSharpe(m)) }}</td><td>{{ f(m?.return_confidence?.sharpe_lcb) }}</td><td>{{ pct(m?.return_confidence?.ann_return_lcb) }}</td><td>{{ f(m?.return_confidence?.hac_t_stat) }}</td><td>{{ f(m?.cost_breakeven_bps) }} bps</td><td>{{ pct(m?.profitable_era_rate) }}</td><td>{{ f(worstStress(m)) }}</td></tr>
        </table>
        <div v-if="detail.factor.eligibility?.failure_reasons?.length" class="failure-list"><b>未通过原因</b><ul><li v-for="reason in detail.factor.eligibility.failure_reasons" :key="reason">{{ reason }}</li></ul></div>
      </div>
      <div v-else class="card selector-empty"><h3>尚未完成 Rating V4.3 全层审计</h3><p>旧评分仅作为历史记录。运行审计后才会生成 F1–F5 等级与 2020 至最新冻结评级分。</p></div>
      <div class="card"><h3>训练反馈层</h3><div class="sub">RankIC 是因子截面排名与前向收益排名的 Spearman 相关性；不是原值 Pearson IC。历史 ic_mean/icir 仅保留兼容读取。</div><table><tr><th>层</th><th>RankICIR</th><th>一致性</th><th>费后 Sharpe</th><th>日均等效换手</th><th>学习分</th></tr><tr v-for="(m,k) in {PUBLIC:detail.factor.public,GATE:detail.factor.gate}" :key="k"><td>{{ k }}</td><td>{{ f(m?.rank_icir ?? m?.icir) }}</td><td>{{ f(m?.era_consistency) }}</td><td>{{ f(layerSharpe(m)) }}</td><td>{{ pct(m?.daily_turnover ?? m?.turnover) }}</td><td>{{ f(m?.score) }}</td></tr></table></div>
      <label>标签（逗号分隔）</label><input v-model="review.tags" placeholder="momentum, quality, low-turnover" /><label>研究备注</label><textarea v-model="review.note" rows="5" placeholder="记录经济机制、已知暴露、失败原因和后续动作"></textarea>
      <div style="margin-top:10px"><button class="btn primary" @click="saveReview">保存研究备注</button><span class="sub" style="margin-left:8px">experiment={{ detail.factor.experiment_id }}</span></div>
    </div>
  </div>`,
  setup() {
    const factors = ref([]), selected = ref([]), detail = ref(null), comparison = ref(null);
    const query = ref(""), status = ref(""), sort = ref("live_rank");
    const groups = ref([]), groupFilter = ref("");
    const groupStats = reactive({ groups: 0, duplicate_groups: 0, redundancy_ratio: 0 });
    const rankingDiagnostics = reactive({ status:"insufficient_sample", sample_size:0, minimum_sample:8, metrics:null, message:"" });
    const showLatex = ref(true), latexEl = ref(null);
    const review = reactive({ tags: "", note: "" });
    const auditForm = reactive({ universe_n: 500, horizon: 5, direction: 1, cost_bps: 20, target_capital: 10000000 });
    const auditing = ref(false), auditErr = ref("");
    const auditEvidence = computed(() => {
      const validation = detail.value?.factor?.validation || {};
      return {event:validation.event_audit || {}, provenance:validation.audit_provenance || {}, overfit:validation.overfit_governance || {}};
    });
    const evidenceStatus = evidence => evidence?.status || (evidence?.passed === true ? "PASS" : evidence?.passed === false ? "FAIL" : "NOT_RUN");
    const evidenceClass = evidence => evidence?.passed === true ? "ok-text" : "bad-text";
    let loadedExperimentVersion = -1;
    const f = v => v == null ? "—" : Number(v).toFixed(3);
    const pct = v => v == null ? "—" : (Number(v) * 100).toFixed(1) + "%";
    const layerSharpe = m => m?.active?.sharpe ?? m?.net?.sharpe ?? m?.long_only_sharpe;
    const layerReturn = m => m?.active?.ann_return ?? m?.net?.ann_return;
    const worstStress = m => m?.cost_stress?.length ? Math.min(...m.cost_stress.map(x=>Number(x.sharpe))) : null;
    const gradeClass = grade => grade === "F5" ? "grade-f5" : grade === "F4" ? "grade-f4" : grade === "F3" ? "grade-f3" : "grade-low";
    const calibrationLabel = value => ({
      calibrated:"已校准", needs_review:"未通过", insufficient_sample:"样本不足",
      not_applicable_full_window_rating:"全窗口口径",
    })[value] || "未知";
    const rankStatusLabel = value => ({
      capital_priority_non_pit:"资本候选", paper_priority:"模拟优先",
      passed_low_conviction:"低置信通过", capacity_limited:"容量受限",
      vault_rejected:"Vault 拒绝", holdout_rejected:"OOS 拒绝",
      research_rejected:"研究拒绝", missing_holdout:"缺少 OOS",
      missing_rating_window:"缺少评级窗口",
    })[value] || "未审计";
    const rankStatusClass = value => (
      value === "capital_priority_non_pit" ? "green"
      : value === "paper_priority" ? "blue"
      : value?.includes("rejected") ? "red"
      : value ? "amber" : ""
    );
    const componentLabel = value => ({
      net_profitability_lcb:"费后盈利下界", selection_confidence:"选择置信度",
      cost_regime_robustness:"成本/状态稳健", oos_generalization:"样本外保持",
      implementability:"可执行性", signal_quality:"信号质量",
    })[value] || value;
    const factorGroupMap = computed(() => {
      const out = new Map();
      groups.value.forEach(g => (g.factor_ids || []).forEach(id => out.set(Number(id), g.id)));
      return out;
    });
    const visibleFactors = computed(() => {
      if (!groupFilter.value) return factors.value;
      return factors.value.filter(factor => factorGroupMap.value.get(Number(factor.id)) === groupFilter.value);
    });
    const groupFor = id => factorGroupMap.value.get(Number(id)) || "";
    const familyLabel = family => ({
      valuation:"估值", capital_flow:"资金流", volatility:"波动",
      liquidity_volume:"流动性/量价", relationship:"关系结构",
      price_trend_reversal:"趋势/反转", composite:"复合", other:"其他",
    })[family] || family;
    async function renderDetailLatex() {
      await nextTick();
      if (!showLatex.value || !latexEl.value || !detail.value) return;
      const latex = detail.value.dsl?.latex || detail.value.factor.expression;
      if (window.katex?.render) {
        window.katex.render(latex, latexEl.value, { throwOnError:false, strict:"warn", trust:false, displayMode:true });
      } else {
        latexEl.value.textContent = latex;
      }
    }
    async function refresh() {
      const params = new URLSearchParams({ limit: "500", sort: sort.value });
      if (query.value) params.set("q", query.value); if (status.value) params.set("lifecycle", status.value);
      const [factorData, groupData, diagnostics] = await Promise.all([
        api("/factors?" + params.toString(), { cacheTtl: 1000 }),
        api("/factors/similarity-groups", { cacheTtl: 5000 }),
        api("/factors/ranking-diagnostics", { cacheTtl: 5000 }),
      ]);
      factors.value = factorData.factors;
      groups.value = groupData.groups || [];
      Object.assign(groupStats, groupData.stats || {});
      Object.assign(rankingDiagnostics, diagnostics || {});
      if (groupFilter.value && !groups.value.some(g => g.id === groupFilter.value)) groupFilter.value = "";
      loadedExperimentVersion = appState.experimentVersion;
    }
    function ensureFresh() {
      if (loadedExperimentVersion !== appState.experimentVersion) refresh();
    }
    function toggleAll(e) { selected.value = e.target.checked ? visibleFactors.value.map(fa=>fa.id) : []; }
    async function open(fa) {
      detail.value = await api(`/factors/${fa.id}/detail`, { cacheTtl: 2000 });
      review.tags = (detail.value.factor.research_meta?.tags || []).join(", ");
      review.note = detail.value.factor.research_meta?.note || "";
      const defaults = detail.value.audit_defaults || {};
      auditForm.universe_n = detail.value.factor.research_meta?.last_audit_universe_n || defaults.universe_n || 500;
      auditForm.horizon = detail.value.factor.research_meta?.last_audit_horizon || defaults.horizon || 5;
      auditForm.direction = Number(detail.value.factor.research_meta?.direction || defaults.direction || 1);
      auditForm.cost_bps = defaults.cost_bps ?? 15;
      auditForm.target_capital = defaults.target_capital ?? 1000000;
      renderDetailLatex();
    }
    async function openSimilar(row) { await open({ id: row.id }); }
    async function compare() { comparison.value = await api("/factors/compare", { method:"POST", body:{ factor_ids:selected.value } }); }
    async function runAudit() {
      auditing.value = true; auditErr.value = "";
      try {
        const factorId = detail.value.factor.id;
        await api(`/factors/${factorId}/audit`, { method:"POST", body:{ ...auditForm } });
        invalidateApiCache();
        detail.value = await api(`/factors/${factorId}/detail`);
        await Promise.all([refresh(), renderDetailLatex()]);
      } catch (e) { auditErr.value = e.message; }
      finally { auditing.value = false; }
    }
    async function saveReview() {
      await api(`/factors/${detail.value.factor.id}/review`, { method:"PATCH", body:{ tags:review.tags.split(","), note:review.note } });
      detail.value.factor.research_meta = { ...detail.value.factor.research_meta, tags:review.tags.split(",").filter(Boolean), note:review.note };
      invalidateApiCache(); refresh();
    }
    watch(() => appState.experimentVersion, () => {
      selected.value = [];
      detail.value = null;
      comparison.value = null;
      groupFilter.value = "";
      if (appState.activeTab === "factors") ensureFresh();
    });
    watch(showLatex, renderDetailLatex);
    onActivated(ensureFresh);
    return {
      factors, visibleFactors, selected, detail, comparison, query, status, sort,
      groups, groupFilter, groupStats, rankingDiagnostics, groupFor, familyLabel, showLatex, latexEl,
      review, auditForm, auditing, auditErr, auditEvidence, evidenceStatus, evidenceClass, f, pct, layerSharpe, layerReturn,
      worstStress, gradeClass, calibrationLabel, rankStatusLabel, rankStatusClass,
      componentLabel, refresh, toggleAll, open, openSimilar, compare,
      runAudit, saveReview,
    };
  },
};

/* ============ 回测 ============ */
const BacktestView = {
  template: `
  <section class="backtest-lab">
    <div class="selector-heading">
      <div><div class="eyebrow">CHRONOLOGICAL EXECUTION / EVENT LEDGER</div><h1>步进事件式回测实验室</h1><p>唯一真相来自订单、成交、现金、持仓与费用账本；信号在 t 日收盘生成，只允许 t+1 原始开盘价成交。</p></div>
      <div><span class="tag blue">{{ market==='ashare' ? 'A股' : '美股' }}</span> <span class="tag green">{{ feeLabel }}</span> <span class="tag amber">{{ form.mode==='long_only' ? '纯多头' : '多空' }}</span> <span class="tag">{{ factorRows.length }} 因子</span></div>
    </div>

    <div class="card backtest-config">
      <div class="execution-timeline">
        <span><b>01</b> t 日收盘读取信号</span><i>→</i><span><b>02</b> t+1开盘风险检查</span><i>→</i><span><b>03</b> 成交与日内止损止盈</span><i>→</i><span><b>04</b> 收盘估值、风控与对账</span>
      </div>
      <div class="factor-sleeve-editor">
        <div class="panel-title-row">
          <div><h3>加权因子袖套</h3><span class="sub">每个因子独立分配资本、生成事件账本和费用；组合收益与逐因子贡献可以精确对账。</span></div>
          <button class="btn" @click="addFactor" :disabled="factorRows.length>=12">＋ 添加因子</button>
        </div>
        <div class="factor-sleeve-head"><span>#</span><span>名称</span><span>DSL 表达式</span><span>方向</span><span>原始权重</span><span>归一权重</span><span></span></div>
        <div class="factor-sleeve-row" v-for="(factor,index) in factorRows" :key="factor.uid">
          <b>{{ index+1 }}</b>
          <input v-model="factor.name" :placeholder="'因子'+(index+1)" />
          <textarea v-model="factor.expression" rows="2" class="backtest-expression-input" placeholder="例: -rank(ts_delta(close, 20))"></textarea>
          <select v-model.number="factor.direction"><option :value="1">+1 高值偏多</option><option :value="-1">-1 低值偏多</option></select>
          <input type="number" v-model.number="factor.weight" min="0.000001" step="0.1" />
          <span class="weight-preview">{{ pct(normalizedWeights[index]) }}</span>
          <button class="btn danger" @click="removeFactor(index)" :disabled="factorRows.length===1">×</button>
        </div>
        <div class="factor-sleeve-summary"><span>权重无需手工加总为 1，运行时按正权重自动归一。</span><b>当前合计 {{ num(factorWeightTotal,4) }}</b></div>
      </div>
      <div class="backtest-form-grid">
        <div><label>股票池</label><select v-model.number="form.universe_n"><option :value="100">Top100</option><option :value="300">Top300</option><option :value="500">Top500</option><option :value="1000">Top1000</option><option :value="1500">Top1500</option></select></div>
        <div><label>开始日期</label><input type="date" v-model="form.start" /></div>
        <div><label>结束日期</label><input type="date" v-model="form.end" /></div>
        <div><label>初始资金</label><input type="number" v-model.number="form.initial_capital" min="10000" /></div>
        <div><label>调仓步长</label><select v-model.number="form.rebalance_every"><option :value="1">每日</option><option :value="5">每5日</option><option :value="10">每10日</option><option :value="20">每20日</option></select></div>
        <div><label>单边选股比例</label><input type="number" v-model.number="form.top_fraction" min="0.01" max="0.5" step="0.05" /></div>
        <div><label>基础滑点 bps</label><input type="number" v-model.number="form.slippage_bps" min="0" step="0.5" /></div>
        <div><label>最大成交量参与率</label><input type="number" v-model.number="form.max_volume_participation" min="0.01" max="1" step="0.01" /></div>
        <div><label>组合版本</label><select v-model="form.mode"><option value="long_only">纯多头（Long Only）</option><option value="long_short" :disabled="market==='ashare'">多空（Long/Short）</option></select><small>每次回测独立冻结，不改写任务默认值 {{ taskMode }}</small></div>
        <div><label>组合方法</label><input value="独立资金袖套（精确归因）" disabled /><small>不跨因子净额抵销订单，避免主观拆分成交贡献</small></div>
        <div v-if="form.mode==='long_short'"><label>年化借券成本 bps</label><input type="number" v-model.number="form.borrow_cost_bps_annual" min="0" /></div>
      </div>
      <div class="backtest-v2-toolbar"><span class="tag green">step_event_v2</span><button class="btn" @click="applyRiskPreset">载入稳健风控预设</button><button class="btn" @click="clearExitPolicy">关闭全部个股退出规则</button></div>
      <details class="backtest-policy" open>
        <summary>个股止损、止盈与持仓生命周期</summary>
        <div class="backtest-form-grid">
          <div><label>固定止损比例</label><input type="number" v-model.number="form.exit_policy.fixed_stop_loss_pct" min="0.001" max="0.99" step="0.01" placeholder="留空关闭" /></div>
          <div><label>固定止盈比例</label><input type="number" v-model.number="form.exit_policy.fixed_take_profit_pct" min="0.001" max="0.99" step="0.01" placeholder="留空关闭" /></div>
          <div><label>移动止损回撤</label><input type="number" v-model.number="form.exit_policy.trailing_stop_pct" min="0.001" max="0.99" step="0.01" placeholder="留空关闭" /></div>
          <div><label>ATR周期</label><input type="number" v-model.number="form.exit_policy.atr_period" min="2" max="252" /></div>
          <div><label>ATR止损倍数</label><input type="number" v-model.number="form.exit_policy.atr_stop_multiple" min="0.1" step="0.25" placeholder="留空关闭" /></div>
          <div><label>ATR止盈倍数</label><input type="number" v-model.number="form.exit_policy.atr_take_profit_multiple" min="0.1" step="0.25" placeholder="留空关闭" /></div>
          <div><label>ATR移动止损倍数</label><input type="number" v-model.number="form.exit_policy.atr_trailing_multiple" min="0.1" step="0.25" placeholder="留空关闭" /></div>
          <div><label>保本启动盈利比例</label><input type="number" v-model.number="form.exit_policy.break_even_activation_pct" min="0.001" max="0.99" step="0.01" placeholder="留空关闭" /></div>
          <div><label>最长持有交易日</label><input type="number" v-model.number="form.exit_policy.time_stop_sessions" min="1" max="2520" placeholder="留空关闭" /></div>
          <div><label>同日双触发</label><select v-model="form.exit_policy.intrabar_conflict_policy"><option value="conservative">保守：先止损</option><option value="optimistic">乐观：先止盈</option></select></div>
        </div>
        <p class="sub">跳空越过阈值按开盘价；日线无法判断先后时默认先止损；移动锚点只在收盘后更新；A股当日买入数量遵守T+1。</p>
      </details>
      <details class="backtest-policy">
        <summary>仓位、账户与组合风险</summary>
        <div class="backtest-form-grid">
          <div><label>账户类型</label><select v-model="form.account_type"><option value="auto">自动</option><option value="cash">现金账户</option><option value="margin">保证金账户</option></select></div>
          <div><label>仓位模型</label><select v-model="form.position_sizing"><option value="equal_weight">等权</option><option value="inverse_volatility">逆波动率</option><option value="atr_risk">ATR风险定仓</option></select></div>
          <div><label>最大持仓数</label><input type="number" v-model.number="form.max_positions" min="1" max="10000" /></div>
          <div><label>单股最大权重</label><input type="number" v-model.number="form.max_position_weight" min="0.001" max="2" step="0.01" /></div>
          <div><label>现金缓冲比例</label><input type="number" v-model.number="form.cash_buffer_fraction" min="0" max="0.99" step="0.01" /></div>
          <div><label>最大总杠杆</label><input type="number" v-model.number="form.max_gross_leverage" min="1" max="10" step="0.1" /></div>
          <div><label>多头目标敞口</label><input type="number" v-model.number="form.long_gross_target" min="0" max="10" step="0.1" /></div>
          <div v-if="form.mode==='long_short'"><label>空头目标敞口</label><input type="number" v-model.number="form.short_gross_target" min="0" max="10" step="0.1" /></div>
          <div v-if="form.position_sizing==='atr_risk'"><label>每只股票风险预算</label><input type="number" v-model.number="form.risk_per_position_fraction" min="0.001" max="0.25" step="0.005" /></div>
          <div><label>最小调仓金额</label><input type="number" v-model.number="form.min_trade_notional" min="0" step="100" /></div>
          <div><label>调仓缓冲比例</label><input type="number" v-model.number="form.rebalance_buffer_pct" min="0" max="0.99" step="0.01" /></div>
          <div><label>融资年利率 bps</label><input type="number" v-model.number="form.margin_interest_bps_annual" min="0" /></div>
          <div><label>组合回撤退出触发阈值</label><input type="number" v-model.number="form.portfolio_stop_drawdown_pct" min="0.001" max="0.99" step="0.01" placeholder="留空关闭" /><small>收盘触发、次日开盘清仓，不是最大回撤保证</small></div>
          <div><label>组合单日亏损退出</label><input type="number" v-model.number="form.portfolio_daily_loss_pct" min="0.001" max="0.99" step="0.01" placeholder="留空关闭" /></div>
          <div><label>风险退出冷却期</label><input type="number" v-model.number="form.risk_cooldown_sessions" min="0" max="252" /></div>
        </div>
        <p class="sub" :class="{'bad-text':leveragePlan.invalid}">
          目标总敞口 {{ num(leveragePlan.configured,2) }}×；实际运行目标 {{ num(leveragePlan.effective,2) }}×；硬上限 {{ num(form.max_gross_leverage,2) }}×；预留 {{ num(leveragePlan.headroom,2) }}×。
          每笔增仓成交均执行硬上限检查，跳空越线会在开盘自动去杠杆；只有无法修复的剩余超限才判定账本失败。
          <b v-if="leveragePlan.invalid">当前多空目标之和超过硬上限，无法运行。</b>
        </p>
      </details>
      <details class="backtest-policy">
        <summary>订单、流动性与终止规则</summary>
        <div class="backtest-form-grid">
          <div><label>买卖价差 bps</label><input type="number" v-model.number="form.spread_bps" min="0" step="0.5" /></div>
          <div><label>市场冲击模型</label><select v-model="form.impact_model"><option value="fixed">固定滑点</option><option value="linear">线性参与率</option><option value="square_root">平方根冲击</option></select></div>
          <div v-if="form.impact_model!=='fixed'"><label>冲击系数 bps</label><input type="number" v-model.number="form.impact_coefficient_bps" min="0" step="1" /></div>
          <div><label>部分成交处理</label><select v-model="form.unfilled_order_policy"><option value="cancel">DAY取消</option><option value="carry">GTC续挂</option></select></div>
          <div v-if="form.unfilled_order_policy==='carry'"><label>最多续挂交易日</label><input type="number" v-model.number="form.max_order_age_sessions" min="1" max="20" /></div>
          <div><label>行情缺失减记阈值</label><input type="number" v-model.number="form.max_stale_sessions" min="1" max="252" /></div>
          <div><label><input type="checkbox" v-model="form.liquidate_at_end" /> 区间结束强制平仓</label></div>
        </div>
        <p class="sub">成交容量使用此前20个交易日ADV，不再使用开盘时尚未知的当日总成交量。</p>
      </details>
      <details class="backtest-policy" open>
        <summary>时间稳定性、IC 与蒙特卡洛诊断</summary>
        <div class="backtest-form-grid">
          <div><label><input type="checkbox" v-model="form.monte_carlo_enabled" /> 启用移动区块蒙特卡洛</label><small>年度/滚动稳定性与IC始终计算</small></div>
          <div v-if="form.monte_carlo_enabled"><label>模拟路径数</label><input type="number" v-model.number="form.monte_carlo_simulations" min="100" max="20000" step="100" /></div>
          <div v-if="form.monte_carlo_enabled"><label>区块长度（交易日）</label><input type="number" v-model.number="form.monte_carlo_block_size_sessions" min="1" max="252" /></div>
          <div v-if="form.monte_carlo_enabled"><label>确定性随机种子</label><input type="number" v-model.number="form.monte_carlo_seed" /></div>
        </div>
        <p class="sub">收益稳定性只读取费后事件账本；IC使用 t 收盘信号对应 t+1 至 t+1+h 复权开盘收益；蒙特卡洛按连续区块重采样，并对所有 sleeve 使用相同日期索引。</p>
      </details>
      <div class="fee-disclosure">
        <b>{{ feeLabel }}</b>
        <span v-if="market==='ashare'">券商佣金万2免5；卖出印花税按历史日期；过户费双向按历史日期。</span>
        <span v-else>每股 $0.005、每单最低 $1、最高成交额 1%；固定费率中的交易规费不重复扣除。</span>
      </div>
      <div class="run-row"><button class="btn primary" @click="run" :disabled="running || !factorInputValid || leveragePlan.invalid">{{ running ? '正在生成多因子事件账本…' : '运行多因子事件回测' }}</button><span v-if="err" class="selector-error inline-error">{{ err }}</span></div>
    </div>

    <template v-if="result">
      <div class="metric-strip backtest-metrics">
        <div class="metric-card accent"><span>费后期末净值</span><b>{{ num(result.stats.final_nav, 4) }}</b><small>{{ money(result.stats.final_nlv) }}</small></div>
        <div class="metric-card"><span>年化 / Sharpe</span><b>{{ pct(result.stats.ann_ret) }}</b><small>Sharpe {{ num(result.stats.sharpe, 2) }}</small></div>
        <div class="metric-card"><span>最大回撤</span><b>{{ pct(result.stats.max_dd) }}</b><small>日均换手 {{ pct(result.stats.avg_daily_turnover) }}</small></div>
        <div class="metric-card"><span>订单 / 成交</span><b>{{ result.stats.orders }} / {{ result.stats.fills }}</b><small>成交率 {{ pct(result.stats.fill_rate) }}</small></div>
        <div class="metric-card"><span>佣金税费</span><b>{{ money(result.stats.commission_and_tax) }}</b><small>滑点 {{ money(result.stats.slippage_cost) }}</small></div>
        <div class="metric-card"><span>账本完整性</span><b :class="result.integrity?.all_pass ? 'ok-text' : 'bad-text'">{{ result.integrity?.all_pass ? 'PASS' : 'FAIL' }}</b><small>{{ result.stats.protocol }}</small></div>
        <div class="metric-card"><span>Sortino / Calmar</span><b>{{ num(result.stats.sortino, 2) }}</b><small>Calmar {{ num(result.stats.calmar, 2) }}</small></div>
        <div class="metric-card" :title="result.stats.trade_statistics_disclosure"><span>已平仓批次 / 胜率</span><b>{{ result.stats.closed_lots ?? result.stats.closed_trades }} / {{ pct((result.stats.closed_lots ?? result.stats.closed_trades) === 0 ? null : (result.stats.closed_lot_win_rate ?? result.stats.win_rate)) }}</b><small>已平仓 PF {{ num((result.stats.closed_lots ?? result.stats.closed_trades) === 0 ? null : (result.stats.closed_lot_profit_factor ?? result.stats.profit_factor), 2) }} · 持有 {{ num(result.stats.avg_holding_sessions,1) }}日</small></div>
      </div>
      <div class="protocol-card"><b>交易统计与组合收益分开解释</b><span>胜率和 PF 仅统计已平仓批次，包含部分减仓；未平仓浮盈亏与未分摊借券/融资成本不计入该比率。零平仓样本显示“—”，不是 0% 胜率。组合净值、Sharpe 与回撤仍包含全部持仓及已入账费用。</span><small>期末未平仓 {{ result.stats.open_positions ?? '—' }} 个 · 未实现盈亏（扣未分摊入场费）{{ money(result.stats.open_unrealized_pnl_after_entry_fees) }} · 未分摊借券/融资成本 {{ money(result.stats.unallocated_financing_cost) }}</small></div>
      <div class="card" v-if="result.execution">
        <div class="panel-title-row">
          <div><h3>Rust 镜像内核诊断</h3><span class="sub">Python事件账本永久作为影子权威；只有逐笔成交与每日NLV全部对齐才标记Rust通过。</span></div>
          <span class="grade-pill" :class="result.execution.alignment?.all_pass ? 'grade-f5' : (result.execution.requested_backend==='python' ? 'grade-low' : 'grade-low')">{{ result.execution.backend_used }}</span>
        </div>
        <div class="metric-strip">
          <div class="metric-card"><span>请求 / 实际后端</span><b>{{ result.execution.requested_backend }}</b><small>{{ result.execution.backend_used }}</small></div>
          <div class="metric-card"><span>账本对齐</span><b :class="result.execution.alignment?.all_pass?'ok-text':'bad-text'">{{ result.execution.alignment?.all_pass ? 'PASS' : 'FALLBACK' }}</b><small>逐笔误差 {{ num(result.execution.alignment?.max_trade_numeric_error,8) }}</small></div>
          <div class="metric-card"><span>事件内核加速</span><b>{{ num(result.execution.timing_seconds?.event_kernel_speedup,2) }}×</b><small>Python {{ num(result.execution.timing_seconds?.python_event_simulation,4) }}s / Rust {{ num(result.execution.timing_seconds?.rust_kernel_only,4) }}s</small></div>
          <div class="metric-card"><span>端到端预测加速</span><b>{{ num(result.execution.timing_seconds?.end_to_end_projected_speedup,2) }}×</b><small>因子物化 {{ num(result.execution.timing_seconds?.materialization,4) }}s</small></div>
        </div>
        <p class="sub bad-text" v-if="result.execution.rust_fallback_reasons?.length">回退原因：{{ result.execution.rust_fallback_reasons.join('；') }}</p>
      </div>
      <div class="card stability-card" v-if="result.stability_analysis?.status==='OK'">
        <div class="panel-title-row"><div><h3>时间切片稳定性</h3><span class="sub">CAGR、Sharpe、MDD 与换手全部来自同一费后事件账本；年度末不足全年时标记 YTD。</span></div><span class="tag green">{{ result.stability_analysis.protocol }}</span></div>
        <div class="metric-strip" v-if="result.stability_analysis.latest_rolling">
          <div class="metric-card" v-for="key in ['12m','24m']" :key="key"><span>最新滚动 {{ key }}</span><template v-if="result.stability_analysis.latest_rolling[key]"><b>{{ pct(result.stability_analysis.latest_rolling[key].cagr) }}</b><small>Sharpe {{ num(result.stability_analysis.latest_rolling[key].sharpe,2) }} · MDD {{ pct(result.stability_analysis.latest_rolling[key].max_drawdown) }}</small></template><b v-else>—</b></div>
          <div class="metric-card"><span>年度收益翻转</span><b>{{ result.stability_analysis.regime_reversal?.reversal_count || 0 }}</b><small>重大翻转 {{ result.stability_analysis.regime_reversal?.material_reversal_count || 0 }}</small></div>
        </div>
        <div class="ledger-scroll"><table><thead><tr><th>切片</th><th>区间/交易日</th><th>总收益</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>日均换手</th><th v-for="factor in stabilitySleeves" :key="factor.factor_id">{{ factor.factor_id }}贡献</th></tr></thead>
          <tbody><tr v-for="row in result.stability_analysis.annual" :key="row.period"><td><b>{{ row.period }}{{ row.is_ytd?' YTD':'' }}</b></td><td>{{ row.start }}~{{ row.end }} / {{ row.sessions }}</td><td>{{ pct(row.total_return) }}</td><td>{{ pct(row.cagr) }}</td><td :class="row.sharpe>=0?'ok-text':'bad-text'">{{ num(row.sharpe,2) }}</td><td>{{ pct(row.max_drawdown) }}</td><td>{{ pct(row.avg_daily_turnover) }}</td><td v-for="factor in stabilitySleeves" :key="factor.factor_id" :class="sleeveContribution(row,factor.factor_id)>=0?'ok-text':'bad-text'">{{ pct(sleeveContribution(row,factor.factor_id)) }}</td></tr></tbody>
        </table></div>
        <details v-if="result.stability_analysis.regime_reversal?.events?.length"><summary>查看 sleeve regime reversal（{{ result.stability_analysis.regime_reversal.events.length }}）</summary><table><tr><th>Sleeve</th><th>切片</th><th>独立收益翻转</th><th>组合贡献翻转</th><th>显著</th></tr><tr v-for="(row,index) in result.stability_analysis.regime_reversal.events" :key="index"><td>{{ row.factor_id }} · {{ row.name }}</td><td>{{ row.from_period }} → {{ row.to_period }}</td><td>{{ pct(row.from_standalone_return) }} → {{ pct(row.to_standalone_return) }}</td><td>{{ pct(row.from_contribution) }} → {{ pct(row.to_contribution) }}</td><td :class="row.material?'bad-text':''">{{ row.material?'MATERIAL':'轻微' }}</td></tr></table></details>
      </div>
      <div class="card" v-if="signalFactorRows.length">
        <div class="panel-title-row"><div><h3>因果 IC / RankIC 诊断</h3><span class="sub">方向已冻结后计算；ICIR 按 √(252/h) 年化。独立资金 sleeve 不伪造单一组合 RankIC。</span></div><span class="tag blue">causal_forward_open_ic_v1</span></div>
        <table><tr><th>因子</th><th>h</th><th>有效截面</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>RankIC胜率</th><th>最新12m RankIC/IR</th></tr>
          <tr v-for="row in signalFactorRows" :key="row.factor_id"><td><b>{{ row.factor_id }} · {{ row.name }}</b></td><td>{{ row.horizon_sessions }}</td><td>{{ row.overall?.n_dates || 0 }} × {{ num(row.overall?.mean_cross_section_n,0) }}</td><td>{{ num(row.overall?.ic_mean,4) }}</td><td>{{ num(row.overall?.icir,2) }}</td><td>{{ num(row.overall?.rank_ic_mean,4) }}</td><td>{{ num(row.overall?.rank_icir,2) }}</td><td>{{ pct(row.overall?.rank_ic_positive_rate) }}</td><td>{{ num(row.latest_rolling?.['12m']?.rank_ic_mean,4) }} / {{ num(row.latest_rolling?.['12m']?.rank_icir,2) }}</td></tr>
        </table>
      </div>
      <div class="card" v-if="result.factor_performance_correlation?.status==='OK'">
        <div class="panel-title-row"><div><h3>因子表现相关性</h3><span class="sub">费后 sleeve 日/月收益、滚动12个月收益与 RankIC 路径分开检查；高相关提示重复来源风险，不直接等同于同一经济机制。</span></div><span class="tag amber">factor_performance_correlation_v1</span></div>
        <table><tr><th>因子对</th><th>日收益ρ</th><th>月收益ρ</th><th>滚动12mρ</th><th>RankIC路径ρ</th><th>判定</th></tr><tr v-for="row in result.factor_performance_correlation.pairs" :key="row.left+row.right"><td><b>{{ row.left }} · {{ row.left_name }}</b><br/><b>{{ row.right }} · {{ row.right_name }}</b></td><td>{{ num(row.daily_return_correlation,3) }}</td><td>{{ num(row.monthly_return_correlation,3) }}</td><td>{{ num(row.rolling_12m_return_correlation,3) }}</td><td>{{ num(row.rank_ic_path_correlation,3) }}</td><td><span class="tag" :class="row.classification==='same_return_source_risk'?'red':(row.classification==='diversifying_negative_correlation'?'green':'amber')">{{ correlationClassLabel(row.classification) }}</span></td></tr></table>
      </div>
      <div class="card" v-if="result.monte_carlo?.status==='OK'">
        <div class="panel-title-row"><div><h3>移动区块蒙特卡洛</h3><span class="sub">{{ result.monte_carlo.simulations }}条路径 · {{ result.monte_carlo.block_size_sessions }}日区块 · seed {{ result.monte_carlo.seed }}；历史路径压力测试，不是未来收益预测。</span></div><span class="tag amber">{{ result.monte_carlo.protocol }}</span></div>
        <div class="metric-strip">
          <div class="metric-card"><span>CAGR 中位 / 5%</span><b>{{ pct(result.monte_carlo.cagr?.p50) }}</b><small>{{ pct(result.monte_carlo.cagr?.p05) }}</small></div>
          <div class="metric-card"><span>Sharpe 中位 / 5%</span><b>{{ num(result.monte_carlo.sharpe?.p50,2) }}</b><small>{{ num(result.monte_carlo.sharpe?.p05,2) }}</small></div>
          <div class="metric-card"><span>MDD 中位 / 95%</span><b>{{ pct(result.monte_carlo.max_drawdown?.p50) }}</b><small>{{ pct(result.monte_carlo.max_drawdown?.p95) }}</small></div>
          <div class="metric-card"><span>期末亏损概率</span><b>{{ pct(result.monte_carlo.risk_probabilities?.terminal_loss) }}</b><small>负Sharpe {{ pct(result.monte_carlo.risk_probabilities?.negative_sharpe) }}</small></div>
          <div class="metric-card"><span>MDD≥30% / ≥50%</span><b>{{ pct(result.monte_carlo.risk_probabilities?.max_drawdown_ge_30pct) }}</b><small>{{ pct(result.monte_carlo.risk_probabilities?.max_drawdown_ge_50pct) }}</small></div>
        </div>
        <table v-if="result.monte_carlo.sleeves?.length"><tr><th>Sleeve</th><th>正收益概率</th><th>期末收益5%</th><th>中位</th><th>95%</th></tr><tr v-for="row in result.monte_carlo.sleeves" :key="row.factor_id"><td>{{ row.factor_id }} · {{ row.name }}</td><td>{{ pct(row.probability_positive_terminal_return) }}</td><td>{{ pct(row.terminal_return?.p05) }}</td><td>{{ pct(row.terminal_return?.p50) }}</td><td>{{ pct(row.terminal_return?.p95) }}</td></tr></table>
      </div>
      <div class="card factor-attribution-card" v-if="result.factor_attribution?.length">
        <div class="panel-title-row"><div><h3>逐因子收益与成本贡献</h3><span class="sub">return contribution 是该袖套净盈亏 / 组合初始资本；各行之和严格等于组合区间总收益。</span></div><div><span class="tag green">{{ result.attribution_method }}</span> <a v-if="currentId" class="btn-link" :href="'/api/backtests/'+currentId+'/factor-attribution.csv'">下载归因 CSV</a></div></div>
        <div class="attribution-reconcile"><span>贡献合计 <b>{{ pct(attributionTotal) }}</b></span><span>组合收益 <b>{{ pct((result.stats.final_nav||1)-1) }}</b></span><span>对账误差 <b :class="Math.abs(attributionError)<=1e-8?'ok-text':'bad-text'">{{ num(attributionError,10) }}</b></span></div>
        <div class="ledger-scroll"><table><thead><tr><th>因子</th><th>方向</th><th>权重</th><th>收益贡献</th><th>净盈亏</th><th>独立收益</th><th>年化 / Sharpe</th><th>回撤</th><th>换手</th><th>成本</th><th>账本</th><th>表达式</th></tr></thead>
          <tbody><tr v-for="row in result.factor_attribution" :key="row.factor_id"><td><b>{{ row.factor_id }} · {{ row.name }}</b></td><td>{{ row.direction>0?'+1':'-1' }}</td><td>{{ pct(row.normalized_weight) }}</td><td :class="row.return_contribution>=0?'ok-text':'bad-text'"><b>{{ pct(row.return_contribution) }}</b></td><td :class="row.net_profit>=0?'ok-text':'bad-text'">{{ money(row.net_profit) }}</td><td>{{ pct(row.standalone_return) }}</td><td>{{ pct(row.ann_ret) }} / {{ num(row.sharpe,2) }}</td><td>{{ pct(row.max_dd) }}</td><td>{{ pct(row.avg_daily_turnover) }}</td><td>{{ money(row.total_execution_cost) }}</td><td :class="row.integrity_pass?'ok-text':'bad-text'">{{ row.integrity_pass?'PASS':'FAIL' }}</td><td class="mono-expr factor-expression">{{ row.expression }}</td></tr></tbody>
        </table></div>
        <p class="sub attribution-disclosure">{{ result.attribution_disclosure }}</p>
      </div>
      <div class="fee-disclosure" v-if="result.stats.portfolio_risk_trigger_events || result.stats.terminal_flat_sessions">
        <b>组合风控状态</b>
        <span>触发 {{ result.stats.portfolio_risk_trigger_events || 0 }} 次，重新武装 {{ result.stats.portfolio_risk_rearms || 0 }} 次，风险关闭 {{ result.stats.portfolio_risk_active_sessions || 0 }} 个交易日，单周期最大回撤 {{ pct(result.stats.max_portfolio_risk_cycle_drawdown) }}，区间末连续空仓 {{ result.stats.terminal_flat_sessions || 0 }} 日。</span>
        <span v-if="result.stats.portfolio_risk_active_at_end" class="bad-text">回测结束时仍处于清仓或冷却状态；这属于明确的风险状态，不是曲线数据截断。</span>
      </div>

      <div class="card ledger-card" v-if="result.round_trips?.length">
        <div class="panel-title-row"><div><h3>已平仓批次与退出归因</h3><span class="sub">包含部分减仓；成本、持有期、MAE/MFE及止损止盈原因。不是完整持仓生命周期计数。</span></div><div><span class="tag">{{ result.stats.closed_lots ?? result.stats.closed_trades }} 批次</span> <a v-if="currentId" class="btn-link" :href="'/api/backtests/'+currentId+'/round-trips.csv'">下载平仓归因CSV</a></div></div>
        <div class="ledger-scroll"><table><thead><tr><th>证券</th><th>入场/退出</th><th>方向</th><th>数量</th><th>成本/退出价</th><th>净盈亏</th><th>收益</th><th>持有</th><th>MFE/MAE</th><th>退出原因</th></tr></thead><tbody><tr v-for="row in result.round_trips" :key="row.symbol+row.entry_date+row.exit_date"><td><code>{{ row.symbol }}</code></td><td>{{ row.entry_date }} / {{ row.exit_date }}</td><td>{{ row.direction>0?'LONG':'SHORT' }}</td><td>{{ num(row.quantity,2) }}</td><td>{{ num(row.entry_price,4) }} / {{ num(row.exit_price,4) }}</td><td :class="row.net_pnl>=0?'ok-text':'bad-text'">{{ money(row.net_pnl) }}</td><td>{{ pct(row.return) }}</td><td>{{ row.holding_sessions }}</td><td>{{ pct(row.mfe_pct) }} / {{ pct(row.mae_pct) }}</td><td><span class="tag">{{ row.exit_reason }}</span></td></tr></tbody></table></div>
      </div>

      <div class="grid cols-2 backtest-analysis">
        <div class="card"><div class="panel-title-row"><div><h3>净值重放</h3><span class="sub">费后净值与相同成交的无成本代理</span></div><span class="tag">{{ result.curve?.dates?.length || 0 }} 日</span></div><div class="chart" ref="curveEl"></div></div>
        <div class="card integrity-card">
          <div class="panel-title-row"><div><h3>交割单回归检查</h3><span class="sub">API、CSV 与净值共用同一成交账本</span></div><span class="grade-pill" :class="result.integrity?.all_pass ? 'grade-f5' : 'grade-low'">{{ result.integrity?.all_pass ? '全部通过' : '存在异常' }}</span></div>
          <table><tr><th>检查项</th><th>结果</th></tr>
            <tr v-for="(value,key) in integrityRows" :key="key"><td>{{ integrityLabels[key] || key }}</td><td :class="integrityOk(key,value) ? 'ok-text' : 'bad-text'">{{ formatIntegrityValue(value) }}</td></tr>
          </table>
          <details v-if="result.integrity?.gross_leverage_violation_details?.length" class="backtest-policy">
            <summary>杠杆控制明细（{{ result.integrity.gross_leverage_violation_details.length }} 次）</summary>
            <table><tr><th>日期</th><th>控制前</th><th>控制后</th><th>硬上限</th><th>减仓单</th><th>结果</th></tr>
              <tr v-for="row in result.integrity.gross_leverage_violation_details" :key="row.trade_date"><td>{{ row.trade_date }}</td><td>{{ num(row.before,6) }}</td><td>{{ num(row.after,6) }}</td><td>{{ num(row.hard_limit,4) }}</td><td>{{ row.orders }}</td><td :class="row.resolved?'ok-text':'bad-text'">{{ row.resolved?'已修复':'未修复' }}</td></tr>
            </table>
          </details>
        </div>
      </div>

      <div class="card step-inspector" v-if="result.daily_steps?.length">
        <div class="panel-title-row"><div><h3>逐日步进状态</h3><span class="sub">拖动时间轴查看当日现金、敞口、成交与事件</span></div><span class="tag blue">{{ currentStep?.trade_date }}</span></div>
        <input class="step-range" type="range" min="0" :max="result.daily_steps.length-1" v-model.number="stepCursor" />
        <div class="step-state-grid" v-if="currentStep">
          <div><span>收盘 NLV</span><b>{{ money(currentStep.close_nlv) }}</b></div><div><span>现金</span><b>{{ money(currentStep.cash) }}</b></div><div><span>多头市值</span><b>{{ money(currentStep.long_market_value) }}</b></div><div><span>空头市值</span><b>{{ money(currentStep.short_market_value) }}</b></div><div><span>成交 / 事件</span><b>{{ currentStep.fills }} / {{ currentStep.events }}</b></div><div><span>净敞口</span><b>{{ pct(currentStep.net_exposure) }}</b></div><div><span>开盘杠杆（控制前/后）</span><b>{{ num(currentStep.open_gross_exposure_before_control,4) }} / {{ num(currentStep.open_gross_exposure,4) }}</b></div><div><span>自动减仓单</span><b>{{ currentStep.leverage_control_orders || 0 }}</b></div>
        </div>
      </div>

      <div class="card ledger-card">
        <div class="panel-title-row">
          <div><h3>详细交割与事件账本</h3><span class="sub">每笔费用、滑点、现金与成交后持仓均可独立复算</span></div>
          <div class="ledger-actions"><button class="btn" :class="{primary:ledgerTab==='trades'}" @click="switchLedger('trades')">交割单</button><button class="btn" :class="{primary:ledgerTab==='events'}" @click="switchLedger('events')">事件流</button><a v-if="currentId" class="btn-link" :href="'/api/backtests/'+currentId+'/statement.csv'">下载完整 CSV</a></div>
        </div>
        <div class="ledger-scroll" v-if="ledgerTab==='trades'">
          <table><thead><tr><th>成交日</th><th>信号日</th><th>证券</th><th>方向</th><th>成交数量</th><th>基准/成交价</th><th>佣金</th><th>印花税</th><th>过户费</th><th>滑点</th><th>总费用</th><th>成交后现金</th><th>成交后持仓</th></tr></thead>
          <tbody><tr v-for="row in ledgerRows" :key="row.fill_id"><td>{{ row.trade_date }}</td><td>{{ row.signal_date }}</td><td><code>{{ row.symbol }}</code><div class="sub">{{ row.factor_name ? row.factor_id+' · '+row.factor_name+' / ' : '' }}{{ row.name }}</div></td><td :class="row.side==='BUY'?'ok-text':'bad-text'">{{ row.side }}</td><td>{{ num(row.filled_quantity,2) }}</td><td>{{ num(row.reference_price,4) }} / {{ num(row.fill_price,4) }}</td><td>{{ money(row.commission) }}</td><td>{{ money(row.stamp_duty) }}</td><td>{{ money(row.transfer_fee) }}</td><td>{{ money(row.slippage_cost) }}</td><td><b>{{ money(row.total_fees) }}</b></td><td>{{ money(row.cash_after) }}</td><td>{{ num(row.position_after,2) }}</td></tr></tbody></table>
        </div>
        <div class="ledger-scroll" v-else>
          <table><thead><tr><th>#</th><th>日期</th><th>阶段</th><th>事件</th><th>证券</th><th>订单</th><th>说明</th></tr></thead><tbody><tr v-for="row in ledgerRows" :key="row.seq"><td>{{ row.seq }}</td><td>{{ row.trade_date }}</td><td>{{ row.phase }}</td><td>{{ row.event_type }}</td><td><code>{{ row.symbol }}</code></td><td>{{ row.order_id }}</td><td>{{ row.message }}</td></tr></tbody></table>
        </div>
        <div class="ledger-pager"><span>{{ ledgerPage.offset+1 }}–{{ Math.min(ledgerPage.offset+ledgerRows.length, ledgerPage.total) }} / {{ ledgerPage.total }}</span><button class="btn" @click="pageLedger(-1)" :disabled="ledgerPage.offset===0">上一页</button><button class="btn" @click="pageLedger(1)" :disabled="ledgerPage.offset+ledgerPage.limit>=ledgerPage.total">下一页</button></div>
      </div>
    </template>

    <div class="card history-card">
      <div class="panel-title-row"><div><h3>历史回测档案</h3><span class="sub">旧向量回测保留但标记 legacy；新记录可重放交割单</span></div><button class="btn" @click="loadHistory">刷新</button></div>
      <table><tr><th>#</th><th>协议</th><th>组合</th><th>因子数</th><th>状态</th><th>因子 / 表达式</th><th>区间</th><th>Sharpe</th><th>年化</th><th>回撤</th><th>交割检查</th><th>时间</th><th>交易计划</th></tr>
        <tr v-for="b in history" :key="b.id" class="clickable" @click="openHistory(b)">
          <td>{{ b.id }}</td><td><span class="tag" :class="b.protocol?.startsWith('step_event_')?'green':'amber'">{{ b.protocol }}</span></td><td><span class="tag amber">{{ b.params?.mode==='long_short' ? '多空' : '纯多' }}</span></td><td>{{ b.params?.factors?.length || 1 }}</td><td>{{ b.status }}</td><td class="mono-expr factor-expression">{{ b.params?.factors?.length ? b.params.factors.map(f=>(f.name||'因子')+'×'+f.weight).join(' · ') : b.params.expression }}</td><td class="sub">{{ b.params.start }}~{{ b.params.end }}</td><td>{{ num(b.stats?.sharpe,2) }}</td><td>{{ pct(b.stats?.ann_ret) }}</td><td>{{ pct(b.stats?.max_dd) }}</td><td :class="b.integrity?.all_pass?'ok-text':'bad-text'">{{ b.integrity?.all_pass ? 'PASS' : '—' }}</td><td class="sub">{{ b.created_at?.slice(0,16) }}</td>
          <td><button v-if="b.status==='done' && b.integrity?.all_pass && b.protocol==='step_event_v2_weighted_sleeves_v1'" class="btn" @click.stop="openTradePlan(b)">创建计划</button></td>
        </tr>
      </table>
    </div>
  </section>`,
  setup() {
    const form = reactive({
      expression:"-rank(ts_delta(close, 20))", universe_n:500,
      start:"2015-01-01", end:"2024-12-31", direction:1, mode:null,
      initial_capital:1000000, rebalance_every:5, top_fraction:0.20,
      slippage_bps:2, max_volume_participation:0.10,
      borrow_cost_bps_annual:0, fee_profile:null,
      account_type:"auto", cash_buffer_fraction:0.02, max_gross_leverage:1,
      margin_interest_bps_annual:500, position_sizing:"equal_weight",
      max_positions:100, max_position_weight:0.10, min_trade_notional:0,
      rebalance_buffer_pct:0.02, long_gross_target:0.95, short_gross_target:0.95,
      risk_per_position_fraction:0.01, spread_bps:0, impact_model:"square_root",
      impact_coefficient_bps:10, unfilled_order_policy:"cancel",
      max_order_age_sessions:3, max_stale_sessions:20, liquidate_at_end:false,
      portfolio_stop_drawdown_pct:null, portfolio_daily_loss_pct:null,
      risk_cooldown_sessions:5,
      monte_carlo_enabled:true, monte_carlo_simulations:2000,
      monte_carlo_block_size_sessions:20, monte_carlo_seed:20260824,
      exit_policy:{
        fixed_stop_loss_pct:null, fixed_take_profit_pct:null, trailing_stop_pct:null,
        atr_period:14, atr_stop_multiple:null, atr_take_profit_multiple:null,
        atr_trailing_multiple:null, break_even_activation_pct:null,
        time_stop_sessions:null, intrabar_conflict_policy:"conservative",
      },
    });
    let factorUid = 2;
    const factorRows = ref([
      {uid:1, name:"因子1", expression:"-rank(ts_delta(close, 20))", weight:1, direction:1},
    ]);
    const taskMode = ref("long_only"), market = ref("us");
    const result = ref(null), history = ref([]), err = ref(""), running = ref(false);
    const currentId = ref(null), curveEl = ref(null), stepCursor = ref(0);
    const ledgerTab = ref("trades"), ledgerRows = ref([]);
    const ledgerPage = reactive({ offset:0, limit:200, total:0 });
    let loadedExperimentVersion = -1;
    const feeLabel = computed(() => market.value === "ashare" ? "万2免5" : "IBKR Pro Fixed");
    const currentStep = computed(() => result.value?.daily_steps?.[stepCursor.value] || null);
    const factorWeightTotal = computed(() => factorRows.value.reduce((sum,row) => sum + Math.max(0,Number(row.weight)||0), 0));
    const normalizedWeights = computed(() => factorRows.value.map(row => factorWeightTotal.value>0 ? Math.max(0,Number(row.weight)||0)/factorWeightTotal.value : 0));
    const factorInputValid = computed(() => factorRows.value.length>0 && factorRows.value.length<=12 && factorWeightTotal.value>0 && factorRows.value.every(row => String(row.expression||"").trim() && Number(row.weight)>0 && [1,-1].includes(Number(row.direction))));
    const attributionTotal = computed(() => (result.value?.factor_attribution || []).reduce((sum,row) => sum + Number(row.return_contribution||0), 0));
    const attributionError = computed(() => attributionTotal.value - (Number(result.value?.stats?.final_nav||1)-1));
    const stabilitySleeves = computed(() => result.value?.stability_analysis?.annual?.find(row=>row.sleeves?.length)?.sleeves || []);
    const signalFactorRows = computed(() => {
      const diagnostics = result.value?.signal_diagnostics;
      if (!diagnostics) return [];
      if (diagnostics.factors) return diagnostics.factors.map(row => ({factor_id:row.factor_id,name:row.name,horizon_sessions:row.diagnostics?.horizon_sessions,overall:row.diagnostics?.overall,latest_rolling:row.diagnostics?.latest_rolling})).filter(row=>row.overall?.status==='OK');
      return diagnostics.overall?.status==='OK' ? [{factor_id:'F01',name:factorRows.value[0]?.name||'单因子',horizon_sessions:diagnostics.horizon_sessions,overall:diagnostics.overall,latest_rolling:diagnostics.latest_rolling}] : [];
    });
    const sleeveContribution = (period,factorId) => Number(period?.sleeves?.find(row=>row.factor_id===factorId)?.return_contribution || 0);
    const correlationClassLabel = value => ({same_return_source_risk:"疑似同源",realised_performance_overlap:"表现重叠",regime_overlap:"状态重叠",diversifying_negative_correlation:"负相关分散",distinct_or_inconclusive:"独立或证据不足"})[value] || value;
    const leveragePlan = computed(() => {
      const longTarget = Math.max(0, Number(form.long_gross_target) || 0);
      const shortTarget = form.mode === "long_short" ? Math.max(0, Number(form.short_gross_target) || 0) : 0;
      const configured = longTarget + shortTarget;
      const hard = Math.max(0, Number(form.max_gross_leverage) || 0);
      const buffer = Math.max(0, Math.min(0.99, Number(form.cash_buffer_fraction) || 0));
      const effective = Math.min(configured, hard * (1 - buffer));
      return {configured, effective, headroom:Math.max(0, hard-effective), invalid:configured > hard + 1e-12};
    });
    const integrityRows = computed(() => Object.fromEntries(
      Object.entries(result.value?.integrity || {}).filter(([key]) => key !== "gross_leverage_violation_details")
    ));
    const integrityLabels = {
      statement_rows:"交割单成交行数",
      cash_reconciliation_max_error:"逐笔现金恒等式最大误差",
      fee_formula_max_error:"费率公式复算最大误差",
      fee_component_sum_max_error:"费用分项加总最大误差",
      gross_amount_max_error:"成交额复算最大误差",
      slippage_formula_max_error:"滑点复算最大误差",
      position_formula_max_error:"持仓变动复算最大误差",
      same_day_signal_fill_violations:"同日信号成交违规",
      scheduled_execution_date_violations:"计划成交日不一致",
      duplicate_fill_id_violations:"重复成交编号",
      nonpositive_fill_violations:"非正数量或价格成交",
      side_sign_violations:"买卖方向与持仓变动冲突",
      fee_profile_violations:"费率档案不一致",
      ashare_buy_lot_violations:"A股买入非整手",
      event_phase_order_violations:"事件阶段乱序",
      long_only_negative_position_violations:"纯多头负持仓违规",
      cash_account_negative_cash_violations:"现金账户负现金违规",
      gross_leverage_violations:"总杠杆未修复超限",
      gross_leverage_breach_events:"开盘杠杆触线次数（含已修复）",
      automatic_deleveraging_events:"自动去杠杆次数",
      leverage_limited_orders:"逐笔杠杆限额订单数",
      max_open_gross_leverage_observed:"控制前最大开盘杠杆",
      max_open_gross_leverage_after_control:"控制后最大开盘杠杆",
      position_state_quantity_violations:"持仓状态与账本数量不一致",
      stale_position_writeoffs:"长期缺失行情减记次数",
      intrabar_ambiguities:"日内止损止盈先后不确定次数",
      ledger_source_of_truth:"净值是否来自账本",
      all_pass:"总检查",
    };
    const num = (value, digits=4) => value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
    const pct = value => value == null || Number.isNaN(Number(value)) ? "—" : (Number(value)*100).toFixed(2)+"%";
    const money = value => value == null || Number.isNaN(Number(value)) ? "—" : new Intl.NumberFormat("zh-CN",{style:"currency",currency:market.value==="ashare"?"CNY":"USD",maximumFractionDigits:2}).format(Number(value));
    const integrityOk = (key, value) => {
      if (key === "statement_rows") return Number(value) >= 0;
      if (key.endsWith("_max_error")) return Number(value) <= 1e-5;
      if (["gross_leverage_breach_events","automatic_deleveraging_events","leverage_limited_orders","intrabar_ambiguities","stale_position_writeoffs"].includes(key)) return Number(value) >= 0;
      if (key === "max_open_gross_leverage_observed") return Number(value) >= 0;
      if (key === "max_open_gross_leverage_after_control") return Number(value) <= Number(result.value?.config?.max_gross_leverage || 0) + 1e-6;
      return value === 0 || value === true;
    };
    const formatIntegrityValue = value => typeof value === "number" ? num(value,8) : String(value);
    async function run() {
      running.value = true; err.value = "";
      try {
        const payload = JSON.parse(JSON.stringify(form));
        payload.factors = factorRows.value.map((row,index) => ({
          name:String(row.name||`因子${index+1}`).trim() || `因子${index+1}`,
          expression:String(row.expression||"").trim(),
          weight:Number(row.weight), direction:Number(row.direction),
        }));
        payload.expression = payload.factors[0]?.expression || "";
        payload.combination_method = "independent_capital_sleeves";
        for (const [key,value] of Object.entries(payload.exit_policy || {})) if (value === "") payload.exit_policy[key] = null;
        for (const key of ["portfolio_stop_drawdown_pct","portfolio_daily_loss_pct"]) if (payload[key] === "") payload[key] = null;
        payload.short_gross_target = payload.mode === "long_short" ? payload.short_gross_target : 0;
        result.value = await api("/backtest", { method: "POST", body: payload });
        currentId.value = result.value.id;
        ledgerTab.value = "trades";
        ledgerRows.value = result.value.trades || [];
        Object.assign(ledgerPage, result.value.trade_page || {offset:0,limit:200,total:ledgerRows.value.length});
        stepCursor.value = Math.max(0, (result.value.daily_steps?.length || 1)-1);
        await nextTick(); drawCurve();
        loadHistory();
      } catch (e) {
        err.value = e.message;
        await loadHistory();
      }
      finally { running.value = false; }
    }
    function addFactor() {
      if (factorRows.value.length >= 12) return;
      factorUid += 1;
      factorRows.value.push({uid:factorUid, name:`因子${factorRows.value.length+1}`, expression:"", weight:1, direction:1});
    }
    function removeFactor(index) {
      if (factorRows.value.length <= 1) return;
      factorRows.value.splice(index,1);
    }
    function applyRiskPreset() {
      Object.assign(form.exit_policy, {
        fixed_stop_loss_pct:0.08, fixed_take_profit_pct:0.20,
        trailing_stop_pct:0.10, atr_period:14, atr_stop_multiple:2.5,
        atr_take_profit_multiple:null, atr_trailing_multiple:3,
        break_even_activation_pct:0.10, time_stop_sessions:60,
        intrabar_conflict_policy:"conservative",
      });
      form.portfolio_stop_drawdown_pct = 0.20;
      form.position_sizing = "atr_risk";
      form.risk_per_position_fraction = 0.01;
    }
    function clearExitPolicy() {
      Object.assign(form.exit_policy, {
        fixed_stop_loss_pct:null, fixed_take_profit_pct:null,
        trailing_stop_pct:null, atr_stop_multiple:null,
        atr_take_profit_multiple:null, atr_trailing_multiple:null,
        break_even_activation_pct:null, time_stop_sessions:null,
      });
    }
    function drawCurve() {
      if (!curveEl.value || !result.value?.curve?.dates) return;
      const c = result.value.curve;
      mountChart(curveEl.value, {
        ...DARK,
        legend:{data:["费后净值","相同成交无成本代理"],textStyle:{color:"#8b949e"}},
        grid: { left: 55, right: 20, top: 42, bottom: 40 },
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: c.dates, axisLabel: { color: "#8b949e" } },
        yAxis: { type: "value", scale: true, splitLine: { lineStyle: { color: "#21262d" } } },
        dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 6 }],
        series: [
          { name:"费后净值", type:"line", data:c.equity, showSymbol:false, lineStyle:{color:"#3fb950",width:1.7}, areaStyle:{color:"rgba(63,185,80,0.07)"} },
          { name:"相同成交无成本代理", type:"line", data:c.cost_free_proxy, showSymbol:false, lineStyle:{color:"#58a6ff",width:1,type:"dashed"} },
        ],
      });
    }
    async function loadHistory() { history.value = (await api("/backtests", { cacheTtl: 1000 })).backtests; }
    async function openHistory(backtest) {
      const detail = await api(`/backtests/${backtest.id}`, { cacheTtl:1000 });
      form.direction = Number(backtest.params?.direction || 1);
      form.mode = backtest.params?.mode || taskMode.value;
      const savedFactors = backtest.params?.factors?.length ? backtest.params.factors : [{name:"因子1",expression:backtest.params?.expression||"",weight:1,direction:Number(backtest.params?.direction||1)}];
      factorRows.value = savedFactors.map((factor,index) => ({uid:++factorUid,name:factor.name||`因子${index+1}`,expression:factor.expression||"",weight:Number(factor.weight||1),direction:Number(factor.direction||1)}));
      currentId.value = detail.id;
      result.value = detail.result?.stats ? detail.result : {stats:detail.result || {},curve:null,integrity:{}};
      ledgerTab.value = "trades"; ledgerRows.value = detail.trades?.rows || [];
      Object.assign(ledgerPage, detail.trades || {offset:0,limit:200,total:0});
      stepCursor.value = Math.max(0,(result.value.daily_steps?.length || 1)-1);
      await nextTick(); drawCurve();
    }
    async function switchLedger(tab) {
      ledgerTab.value = tab; ledgerPage.offset = 0; await loadLedger();
    }
    async function loadLedger() {
      if (!currentId.value) return;
      const page = await api(`/backtests/${currentId.value}/${ledgerTab.value}?offset=${ledgerPage.offset}&limit=${ledgerPage.limit}`);
      ledgerRows.value = page.rows || []; Object.assign(ledgerPage, page);
    }
    async function pageLedger(direction) {
      ledgerPage.offset = Math.max(0, ledgerPage.offset + direction * ledgerPage.limit);
      await loadLedger();
    }
    async function loadContext() {
      result.value = null;
      const meta = await api("/meta", { cacheTtl: 1500 });
      market.value = meta.market || "us";
      taskMode.value = meta.portfolio_mode || "long_only";
      form.mode = taskMode.value;
      form.direction = Number(meta.direction || 1);
      form.initial_capital = meta.evaluation_config?.target_capital ?? (market.value==="ashare"?10000000:1000000);
      form.slippage_bps = market.value === "ashare" ? 5 : 2;
      form.borrow_cost_bps_annual = meta.evaluation_config?.borrow_cost_bps_annual ?? 0;
      form.account_type = form.mode === "long_short" ? "margin" : "cash";
      form.max_gross_leverage = form.mode === "long_short" ? 2 : 1;
      form.long_gross_target = 0.95;
      form.short_gross_target = form.mode === "long_short" ? 0.95 : 0;
      form.min_trade_notional = market.value === "ashare" ? 1000 : 100;
      currentId.value = null; ledgerRows.value = []; Object.assign(ledgerPage,{offset:0,limit:200,total:0});
      await loadHistory();
      loadedExperimentVersion = appState.experimentVersion;
    }
    function ensureContext() {
      if (loadedExperimentVersion !== appState.experimentVersion) loadContext();
    }
    watch(
      () => appState.experimentVersion,
      () => { if (appState.activeTab === "backtest") ensureContext(); },
    );
    function activateBacktest() {
      ensureContext();
      if (appState.backtestDraft) {
        form.expression = appState.backtestDraft;
        form.direction = 1;
        factorRows.value = [{uid:++factorUid,name:"因子1",expression:appState.backtestDraft,weight:1,direction:1}];
        appState.backtestDraft = "";
      }
    }
    onActivated(activateBacktest);
    return {
      form, factorRows, factorWeightTotal, normalizedWeights, factorInputValid,
      attributionTotal, attributionError, stabilitySleeves, signalFactorRows, sleeveContribution, correlationClassLabel, addFactor, removeFactor,
      taskMode, market, feeLabel, result, history, err, running, run,
      curveEl, currentId, currentStep, stepCursor, ledgerTab, ledgerRows,
      ledgerPage, integrityLabels, integrityRows, leveragePlan, num, pct, money, loadHistory, openHistory,
      openTradePlan(b) { appState.planBacktestId=b.id; appState.requestedTab='trade-plans'; },
      integrityOk, formatIntegrityValue, switchLedger, pageLedger, applyRiskPreset, clearExitPolicy,
    };
  },
};

/* ============ 设置 ============ */
const SettingsView = {
  template: `
  <div>
    <div class="card task-definition-card" style="margin-bottom:14px">
      <div class="panel-title-row"><div><h2>定义研究任务</h2><span class="sub">任务配置会随实验保存；历史任务只允许归档，不会删除数据。</span></div><span class="tag blue">单端口 · 多任务并行</span></div>
      <div class="form-row">
        <div style="flex:2"><label>任务名称</label><input v-model="taskForm.name" placeholder="如：A股多头质量因子 V2" /></div>
        <div style="flex:3"><label>研究问题 / 假设</label><input v-model="taskForm.description" placeholder="要验证的经济机制、变更点与成功标准" /></div>
      </div>
      <div class="form-row">
        <div><label>市场</label><select v-model="taskForm.market" :disabled="Boolean(serviceMarket)" @change="syncMarketDefaults"><option value="ashare">A股</option><option value="us">美股</option></select><small v-if="serviceMarket">独立市场服务 · 数据库与研究任务隔离</small></div>
        <div><label>持仓约束</label><select v-model="taskForm.portfolio_mode"><option value="long_only">纯多头</option><option value="long_short" :disabled="taskForm.market==='ashare'">多空</option></select></div>
        <div><label>双向同分优先方向</label><select v-model.number="taskForm.direction"><option :value="1">+1 高值偏多</option><option :value="-1">-1 低值偏多</option></select><small>每个候选仍会同时评价正反两向</small></div>
        <div><label>引擎版本</label><select v-model="taskForm.engine_mode"><option value="v2">V2 MinerTemplate（当前）</option></select></div>
        <div><label>候选总预算</label><input v-model.number="taskForm.candidate_evaluation_budget" type="number" min="1" /></div>
        <div><label>正式因子目标（0=不限）</label><input v-model.number="taskForm.target_factor_count" type="number" min="0" /></div>
        <div style="flex:3"><label>面板路径（可选）</label><input v-model="taskForm.panel_glob" placeholder="留空使用服务默认面板" /></div>
      </div>
      <div class="architecture-builder">
        <div class="panel-title-row"><div><h3>研究架构</h3><span class="sub">后端会再次校验层级组合；proposal_mode 由架构自动推导，不能出现界面与实际运行不一致。</span></div><span class="tag blue">{{ architectureSchema || 'architecture/v1' }}</span></div>
        <div class="form-row">
          <div style="flex:3"><label>架构模板</label><select v-model="taskForm.architecture_template" @change="applyArchitectureTemplate"><option v-for="a in architectureTemplates" :key="a.key" :value="a.key">{{ a.label }}{{ a.recommended ? '（推荐）' : '' }}</option><option value="custom">自定义层级</option></select></div>
          <div v-if="taskForm.layer2_enabled"><label>第二层记忆</label><select v-model="taskForm.memory_mode"><option value="adaptive">任务连续记忆</option><option value="cold">冷记忆对照</option></select></div>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.start_after_create" /> 创建后立即启动</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.return_source_governance_enabled" /> 训练收益来源去重</label>
        </div>
        <div class="architecture-flow">
          <div class="architecture-layer" :class="{enabled:taskForm.layer1_enabled}"><b>L1 搜索</b><span>{{ layer1Label }}</span></div><i>→</i>
          <div class="architecture-layer" :class="{enabled:taskForm.layer2_enabled}"><b>L2 Researcher</b><span>{{ taskForm.layer2_enabled ? (taskForm.memory_mode==='adaptive'?'LLM · 连续记忆':'LLM · 冷记忆') : '关闭' }}</span></div><i>→</i>
          <div class="architecture-layer" :class="{enabled:taskForm.layer3_enabled}"><b>L3 Governor</b><span>{{ taskForm.layer3_enabled ? 'LLM · 单变量治理' : '关闭' }}</span></div>
        </div>
        <p class="sub">{{ (selectedArchitecture && selectedArchitecture.description) || '手工组合层级；第三层必须依赖第二层。' }}</p>
        <div v-if="taskForm.layer2_enabled && !llm.inner_provider" class="warn-banner">当前没有内层 LLM provider。可以先创建任务，但启动 L2 时会熔断，不会偷偷退化为随机任务。</div>
        <div v-if="taskForm.layer3_enabled && !llm.outer_provider" class="warn-banner">当前没有外层 LLM provider。可以先创建任务，但启动 L3 Governor 时会熔断。</div>
        <div v-if="taskForm.architecture_template==='custom'" class="architecture-custom">
          <label class="inline-check"><input type="checkbox" v-model="taskForm.layer1_enabled" /> 启用第一层</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.layer2_enabled" @change="syncCustomArchitecture" /> 启用第二层 LLM</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.layer3_enabled" :disabled="!taskForm.layer2_enabled" /> 启用第三层 LLM</label>
          <div v-if="taskForm.layer1_enabled" class="algorithm-options"><span>第一层算法：</span><label v-for="algorithm in architectureAlgorithms" :key="algorithm" :title="algorithmDescription(algorithm)"><input type="checkbox" :value="algorithm" v-model="taskForm.search_algorithms" /> {{ algorithmLabel(algorithm) }}</label></div>
        </div>
        <div v-if="taskForm.layer1_enabled" class="architecture-custom" style="margin-top:10px">
          <label class="inline-check"><input type="checkbox" v-model="taskForm.qlib_integration_enabled" @change="syncQlibIntegration" /> 启用 Qlib 研究增强</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.qlib_alpha158_prior_enabled" :disabled="!taskForm.qlib_integration_enabled" @change="syncQlibIntegration" /> Alpha158 先验种子</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.qlib_gbdt_candidate_pool_enabled" :disabled="!taskForm.qlib_integration_enabled" /> Alpha158 加入 GBDT 蒸馏池</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.qlib_joint_model_enabled" :disabled="!taskForm.qlib_integration_enabled" @change="syncQlibIntegration" /> Alpha158 样本级联合模型</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.qlib_residual_distillation_enabled" :disabled="!taskForm.qlib_joint_model_enabled" /> Residual OOF → DSL</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.qlib_adaptive_budget_enabled" :disabled="!taskForm.qlib_integration_enabled" /> 自适应算法预算</label>
          <label class="inline-check"><input type="checkbox" v-model="taskForm.qlib_dynamic_trial_governance_enabled" :disabled="!taskForm.qlib_integration_enabled" /> 动态试验治理</label>
          <span class="sub">联合模型只消费 INNER_PUBLIC/META_TRAIN；蒸馏DSL从零进入 V4.2。HOLDOUT/Vault、冻结评级不会进入搜索反馈。</span>
        </div>
        <div v-if="taskForm.qlib_integration_enabled" class="form-row" style="margin-top:10px">
          <div><label>联合模型最大训练行</label><input v-model.number="taskForm.qlib_max_training_rows" type="number" min="5000" max="5000000" step="5000" /></div>
          <div><label>模型刷新间隔（唯一评价）</label><input v-model.number="taskForm.qlib_model_refresh_unique_evals" type="number" min="20" step="20" /></div>
          <div><label>META_TRAIN 最少交易日</label><input v-model.number="taskForm.qlib_min_meta_dates" type="number" min="20" max="500" step="10" /></div>
          <div><label>Qlib结构先验份额</label><input v-model.number="taskForm.qlib_structural_prior_share" type="number" min="0.02" max="0.30" step="0.01" /></div>
        </div>
      </div>
      <div class="evaluation-config-grid">
        <div><label>建仓比例</label><input v-model.number="taskForm.top_fraction" type="number" min="0.05" max="0.5" step="0.05" /></div>
        <div><label>基础成本 bps</label><input v-model.number="taskForm.base_cost_bps" type="number" min="0" /></div>
        <div><label>压力成本 bps</label><input v-model="taskForm.stress_cost_bps" placeholder="10,20,35,50" /></div>
        <div><label>年借券成本 bps</label><input v-model.number="taskForm.borrow_cost_bps_annual" type="number" :disabled="taskForm.portfolio_mode==='long_only'" /></div>
        <div><label>目标资金规模</label><input v-model.number="taskForm.target_capital" type="number" min="1" /></div>
        <div><label>OOS 最低 Sharpe</label><input v-model.number="taskForm.min_oos_sharpe" type="number" step="0.1" /></div>
        <div><label>最大回撤</label><input v-model.number="taskForm.max_drawdown" type="number" step="0.05" /></div>
        <div><label>最大日换手</label><input v-model.number="taskForm.max_daily_turnover" type="number" step="0.05" /></div>
        <div><label>收益下界置信度</label><input v-model.number="taskForm.return_lcb_confidence" type="number" min="0.51" max="0.99" step="0.01" /></div>
        <div><label>收益 HAC t 门槛</label><input v-model.number="taskForm.min_return_hac_t" type="number" min="0" step="0.1" /></div>
        <div><label>最低盈利 era 比例</label><input v-model.number="taskForm.min_profitable_era_rate" type="number" min="0" max="1" step="0.05" /></div>
        <div><label>最低成本缓冲倍数</label><input v-model.number="taskForm.min_cost_cushion_multiple" type="number" min="0" step="0.25" /></div>
        <div><label>预声明检验次数</label><input v-model.number="taskForm.multiple_testing_trials" type="number" min="1" step="100" /></div>
        <div><label>排序目标年化</label><input v-model.number="taskForm.target_rank_ann_return" type="number" min="0.01" step="0.01" /></div>
        <div style="align-self:end"><button class="btn primary" @click="createTask">创建可审计架构任务</button></div>
      </div>
      <div v-if="taskMsg" :style="{color: taskOk ? 'var(--green)' : 'var(--red)'}">{{ taskMsg }}</div>
    </div>
    <div class="warn-banner">统一服务端口由启动环境决定；当前页面、A股与美股任务共用同一个 HTTP 端口，数据面板按任务配置隔离。</div>
    <div class="grid cols-2">
      <div class="card">
        <h3>大模型接入 (OpenAI / Anthropic 格式)</h3>
        <div v-for="(p,i) in llm.providers" :key="i" class="provider-card">
          <div class="form-row">
            <div><label>名称</label><input v-model="p.name" placeholder="如 my-openai" /></div>
            <div><label>协议格式</label><select v-model="p.format"><option value="openai">OpenAI 兼容</option><option value="anthropic">Anthropic</option></select></div>
          </div>
          <label>Base URL</label>
          <input v-model="p.base_url" :placeholder="p.format==='anthropic' ? 'https://api.anthropic.com' : 'https://api.openai.com/v1'" />
          <div class="form-row">
            <div><label>API Key (留空则保留已保存的)</label><input v-model="p.api_key" type="password" placeholder="sk-..." /></div>
            <div><label>模型</label><input v-model="p.model" placeholder="gpt-4o / claude-sonnet-4-5" /></div>
          </div>
          <button class="btn danger" style="margin-top:8px" @click="llm.providers.splice(i,1)">删除</button>
        </div>
        <button class="btn" @click="llm.providers.push({name:'', format:'openai', base_url:'', api_key:'', model:''})">+ 添加提供商</button>
        <div class="form-row" style="margin-top:12px">
          <div><label>内层挖掘用</label><select v-model="llm.inner_provider"><option value="">未配置（LLM 任务启动时熔断）</option><option v-for="p in llm.providers" :key="p.name" :value="p.name">{{ p.name }}</option></select></div>
          <div><label>外层元优化用</label><select v-model="llm.outer_provider"><option value="">未配置（Governor 启动时熔断）</option><option v-for="p in llm.providers" :key="p.name" :value="p.name">{{ p.name }}</option></select></div>
        </div>
      </div>
      <div class="card">
        <h3>引擎参数</h3>
        <div class="form-row">
          <div><label>每外层步内层评估预算</label><input v-model.number="eng.inner_budget_per_outer_step" type="number" /></div>
          <div><label>外层单边 p 阈值</label><input v-model.number="eng.outer_accept_p_value" type="number" min="0.001" max="0.5" step="0.01" /></div>
        </div>
        <div class="form-row">
          <div><label>在位者重测周期 (步)</label><input v-model.number="eng.incumbent_remeasure_every" type="number" /></div>
          <div><label>配对种子数</label><input v-model.number="eng.n_seeds_per_candidate" type="number" min="2" /></div>
          <div><label>单次 LLM 批量</label><input v-model.number="eng.batch_candidates_per_call" type="number" min="1" /></div>
        </div>
        <div class="form-row">
          <div><label>最大外层步</label><input v-model.number="eng.max_outer_steps" type="number" min="1" /></div>
          <div><label>最大运行小时</label><input v-model.number="eng.max_runtime_hours" type="number" min="0.1" step="0.5" /></div>
          <div><label>最大 LLM 调用</label><input v-model.number="eng.max_llm_calls" type="number" min="1" /></div>
          <div><label>研究树深度</label><input v-model.number="eng.max_tree_depth" type="number" min="1" /></div>
        </div>
        <h3 style="margin-top:16px">任务集</h3>
        <table>
          <tr><th>名称</th><th>股票池</th><th>持有期</th><th>A股成本</th><th>美股成本</th></tr>
          <tr v-for="t in eng.tasks" :key="t.name"><td>{{ t.name }}</td><td>{{ t.universe_n }}</td><td>{{ t.horizon }}日</td><td>{{ t.cost_bps_by_market?.ashare ?? '—' }} bps</td><td>{{ t.cost_bps_by_market?.us ?? t.cost_bps }} bps</td></tr>
        </table>
        <div class="sub" style="margin-top:8px">V4.2 在训练安全层同时评价 +1/-1，双向计入检验次数后冻结方向；Rating V4.3 在完整审计中计算 2020 至最新交易日评级。HOLDOUT/Vault 与评级结果都不能参与选方向或循环提示词。</div>
        <div class="protocol-card"><b>Evaluation {{ evalProtocol.version || 'v4.2' }} · Rating {{ evalProtocol.rating_version || 'v4.3' }}</b><span>双向训练评价 · 2020 至最新冻结评级 · 独立硬门槛 · 费后收益下界 · 多重检验</span><small>{{ evalProtocol.policy_label }}</small></div>
      </div>
    </div>
    <div style="margin-top:14px; display:flex; gap:10px; align-items:center">
      <button class="btn primary" @click="save">保存设置</button>
      <span :style="{color: saved==='ok' ? 'var(--green)' : 'var(--red)'}">{{ msg }}</span>
    </div>
  </div>`,
  setup() {
    const llm = reactive({ providers: [], inner_provider: "", outer_provider: "" });
    const eng = reactive({ tasks: [] });
    const evalProtocol = reactive({});
    const architectureTemplates = ref([]), architectureAlgorithms = ref([]), architectureSchema = ref("");
    const algorithmLabels = {
      grammar_enumerative: "Grammar 枚举", map_elites: "MAP-Elites", mcts_puct: "MCTS/PUCT",
      residual_oof_beam: "Residual OOF Beam", gbdt_residual_distill: "GBDT 残差蒸馏",
      evolutionary: "Evolutionary", tpe_smac: "TPE/SMAC", novelty_search: "Novelty",
      cegis_repair: "CEGIS 反例修复", structured_random: "结构化随机（旧）",
      surrogate_kernel: "Kernel 代理（旧）", q_learning: "Q-learning（旧）",
      qlib_alpha158_prior: "Qlib Alpha158 先验",
      qlib_joint_residual_distill: "Qlib联合模型 Residual→DSL",
    };
    const algorithmDescriptions = {
      grammar_enumerative: "结构搜索组 30%：语法候选经新颖性与复杂度预筛。",
      map_elites: "结构搜索组 30%：按结构生态位维护质量-多样性档案。",
      mcts_puct: "结构搜索组 30%：在持久化研究树上做 PUCT 父节点选择与 rollout。",
      residual_oof_beam: "残差组 25%：优先搜索现有收益路径未解释的候选；晋级仍要求精确 OOF。",
      gbdt_residual_distill: "ML 组 20%：残差代理学习后将候选蒸馏回可审计 DSL。",
      evolutionary: "局部优化组 15%：围绕训练期在位者进行可归因变异。",
      tpe_smac: "局部优化组 15%：密度比代理与在位者局部搜索。",
      novelty_search: "高风险组 10%：最大化与已有表达式的结构距离。",
      cegis_repair: "高风险组 10%：利用失败候选作为反例，定向修复或重新生成。",
      qlib_alpha158_prior: "结构搜索组 30% 内的固定来源先验：从完整 Alpha158 中按机制、字段和未探索度选种子，仍由本地评价器裁决。",
      qlib_joint_residual_distill: "ML组20%：完整Alpha158样本级模型只在训练安全层做时间OOF，再把稳定结构蒸馏回DSL并从零评价。",
    };
    const algorithmLabel = name => algorithmLabels[name] || name;
    const algorithmDescription = name => algorithmDescriptions[name] || name;
    const msg = ref(""), saved = ref("");
    const taskForm = reactive({
      name: "", description: "", market: "ashare", portfolio_mode: "long_only",
      direction: 1, engine_mode: "v2", panel_glob: "", top_fraction: 0.20,
      base_cost_bps: 20, stress_cost_bps: "10,20,35,50",
      borrow_cost_bps_annual: 0, target_capital: 10000000,
      min_oos_sharpe: 0.5, max_drawdown: 0.35, max_daily_turnover: 0.35,
      return_lcb_confidence: 0.90, min_return_hac_t: 1.2816,
      min_profitable_era_rate: 0.60, min_cost_cushion_multiple: 1.50,
      multiple_testing_trials: 1000, target_rank_ann_return: 0.10,
      architecture_template: "random_researcher",
      layer1_enabled: true, layer2_enabled: true, layer3_enabled: false,
      search_algorithms: ["structured_random"], memory_mode: "adaptive",
      candidate_evaluation_budget: 120, target_factor_count: 0,
      start_after_create: false, return_source_governance_enabled: true,
      qlib_integration_enabled: false, qlib_alpha158_prior_enabled: false,
      qlib_gbdt_candidate_pool_enabled: false,
      qlib_joint_model_enabled: false, qlib_residual_distillation_enabled: false,
      qlib_adaptive_budget_enabled: false, qlib_dynamic_trial_governance_enabled: false,
      qlib_max_training_rows: 250000, qlib_model_refresh_unique_evals: 100,
      qlib_min_meta_dates: 60,
      qlib_structural_prior_share: 0.10,
    });
    const taskMsg = ref(""), taskOk = ref(false);
    const serviceMarket = ref("");
    async function load() {
      const [s, architectures, identity] = await Promise.all([
        api("/settings"), api("/research-architectures", { cacheTtl: 10000 }),
        api("/service/identity"),
      ]);
      serviceMarket.value = identity.service_market || "";
      if (serviceMarket.value && taskForm.market !== serviceMarket.value) {
        taskForm.market = serviceMarket.value;
        syncMarketDefaults();
      }
      Object.assign(llm, s.llm_providers);
      Object.assign(eng, s.engine_config);
      Object.assign(evalProtocol, s.evaluation_protocol || {});
      architectureTemplates.value = architectures.templates || [];
      architectureAlgorithms.value = architectures.customization?.search_algorithms || [];
      architectureSchema.value = architectures.schema || "";
      applyArchitectureTemplate();
    }
    async function save() {
      try {
        await api("/settings", { method: "POST", body: { llm_providers: JSON.parse(JSON.stringify(llm)), engine_config: JSON.parse(JSON.stringify(eng)) } });
        saved.value = "ok"; msg.value = "已保存 ✓";
        load();
      } catch (e) { saved.value = "err"; msg.value = "保存失败: " + e.message; }
      setTimeout(() => (msg.value = ""), 3000);
    }
    async function createTask() {
      taskMsg.value = "";
      try {
        const stress = String(taskForm.stress_cost_bps).split(",").map(Number).filter(Number.isFinite);
        const created = await api("/experiments", { method: "POST", body: {
          name: taskForm.name, description: taskForm.description,
          research_config: { market: taskForm.market, portfolio_mode: taskForm.portfolio_mode,
            direction: taskForm.direction, direction_policy: "both_train_select", engine_mode: taskForm.engine_mode,
            architecture_template: taskForm.architecture_template,
            layer1_enabled: taskForm.layer1_enabled, layer2_enabled: taskForm.layer2_enabled,
            layer3_enabled: taskForm.layer3_enabled,
            search_algorithms: [...taskForm.search_algorithms], memory_mode: taskForm.memory_mode,
            qlib_integration: {
              enabled: taskForm.qlib_integration_enabled,
              alpha158_prior_enabled: taskForm.qlib_alpha158_prior_enabled,
              gbdt_candidate_pool_enabled: taskForm.qlib_gbdt_candidate_pool_enabled,
              joint_model_enabled: taskForm.qlib_joint_model_enabled,
              residual_distillation_enabled: taskForm.qlib_residual_distillation_enabled,
              adaptive_budget_enabled: taskForm.qlib_adaptive_budget_enabled,
              dynamic_trial_governance_enabled: taskForm.qlib_dynamic_trial_governance_enabled,
              max_training_rows: taskForm.qlib_max_training_rows,
              model_refresh_unique_evals: taskForm.qlib_model_refresh_unique_evals,
              min_meta_dates: taskForm.qlib_min_meta_dates,
              structural_prior_share: taskForm.qlib_structural_prior_share,
              include_low_fidelity_vwap: false,
            },
            candidate_evaluation_budget: taskForm.candidate_evaluation_budget,
            target_factor_count: taskForm.target_factor_count,
            engine_config: JSON.parse(JSON.stringify(eng)),
            return_source_governance: taskForm.return_source_governance_enabled ? {
              protocol: "training_return_source_governance_v2", correlation_threshold: 0.85,
              required_sources: 5, meta_score_weight: 0.15, cross_experiment_admission: false,
            } : { protocol: "disabled" },
            panel_glob: taskForm.panel_glob || undefined, evaluation_protocol: evalProtocol.version || "v4.2",
            evaluation_config: {
              top_fraction: taskForm.top_fraction, tail_fraction: taskForm.top_fraction,
              base_cost_bps: taskForm.base_cost_bps, stress_cost_bps: stress,
              borrow_cost_bps_annual: taskForm.borrow_cost_bps_annual,
              target_capital: taskForm.target_capital, min_oos_sharpe: taskForm.min_oos_sharpe,
              max_drawdown: taskForm.max_drawdown, max_daily_turnover: taskForm.max_daily_turnover,
              return_lcb_confidence: taskForm.return_lcb_confidence,
              min_return_hac_t: taskForm.min_return_hac_t,
              min_profitable_era_rate: taskForm.min_profitable_era_rate,
              min_cost_cushion_multiple: taskForm.min_cost_cushion_multiple,
              multiple_testing_trials: taskForm.multiple_testing_trials,
              target_rank_ann_return: taskForm.target_rank_ann_return,
            }},
        }});
        if (taskForm.start_after_create) {
          await api("/engine/start", { method: "POST", body: { mode: "v2", experiment_id: created.id } });
        }
        taskOk.value = true; taskMsg.value = `研究任务 #${created.id} 已创建${taskForm.start_after_create ? '并启动' : '，可在“实验”页启动'}`;
        taskForm.name = ""; taskForm.description = "";
      } catch (e) { taskOk.value = false; taskMsg.value = "创建失败: " + e.message; }
    }
    function syncMarketDefaults() {
      if (taskForm.market === "ashare") {
        taskForm.portfolio_mode = "long_only"; taskForm.base_cost_bps = 20;
        taskForm.stress_cost_bps = "10,20,35,50"; taskForm.borrow_cost_bps_annual = 0;
        taskForm.target_capital = 10000000; taskForm.max_daily_turnover = 0.35;
        taskForm.target_rank_ann_return = 0.10;
      } else {
        taskForm.portfolio_mode = "long_short"; taskForm.base_cost_bps = 15;
        taskForm.stress_cost_bps = "5,15,25,40"; taskForm.borrow_cost_bps_annual = 300;
        taskForm.target_capital = 1000000; taskForm.max_daily_turnover = 0.50;
        taskForm.target_rank_ann_return = 0.12;
      }
    }
    const selectedArchitecture = computed(() => architectureTemplates.value.find(a => a.key === taskForm.architecture_template));
    const layer1Label = computed(() => {
      if (!taskForm.layer1_enabled) return "关闭";
      if (taskForm.search_algorithms.length === 1 && taskForm.search_algorithms[0] === "structured_random") return "结构化随机";
      return `${taskForm.search_algorithms.length} 算法组合`;
    });
    function applyArchitectureTemplate() {
      if (taskForm.architecture_template === "custom") return;
      const template = architectureTemplates.value.find(a => a.key === taskForm.architecture_template);
      if (!template) return;
      taskForm.layer1_enabled = Boolean(template.layer1_enabled);
      taskForm.layer2_enabled = Boolean(template.layer2_enabled);
      taskForm.layer3_enabled = Boolean(template.layer3_enabled);
      taskForm.search_algorithms = [...(template.search_algorithms || [])];
      taskForm.memory_mode = template.default_memory_mode || "adaptive";
      const qlib = template.qlib_integration || {};
      taskForm.qlib_integration_enabled = Boolean(qlib.enabled);
      taskForm.qlib_alpha158_prior_enabled = Boolean(qlib.enabled && qlib.alpha158_prior_enabled !== false);
      taskForm.qlib_gbdt_candidate_pool_enabled = Boolean(qlib.enabled && qlib.gbdt_candidate_pool_enabled !== false);
      taskForm.qlib_joint_model_enabled = Boolean(qlib.enabled && qlib.joint_model_enabled !== false);
      taskForm.qlib_residual_distillation_enabled = Boolean(qlib.enabled && qlib.residual_distillation_enabled !== false);
      taskForm.qlib_adaptive_budget_enabled = Boolean(qlib.enabled && qlib.adaptive_budget_enabled !== false);
      taskForm.qlib_dynamic_trial_governance_enabled = Boolean(qlib.enabled && qlib.dynamic_trial_governance_enabled !== false);
      taskForm.qlib_max_training_rows = Number(qlib.max_training_rows || 250000);
      taskForm.qlib_model_refresh_unique_evals = Number(qlib.model_refresh_unique_evals || 100);
      taskForm.qlib_min_meta_dates = Number(qlib.min_meta_dates || 60);
      taskForm.qlib_structural_prior_share = Number(qlib.structural_prior_share || 0.10);
      syncQlibIntegration();
    }
    function syncCustomArchitecture() {
      if (!taskForm.layer2_enabled) taskForm.layer3_enabled = false;
    }
    function syncQlibIntegration() {
      if (!taskForm.qlib_integration_enabled) {
        taskForm.qlib_alpha158_prior_enabled = false;
        taskForm.qlib_gbdt_candidate_pool_enabled = false;
        taskForm.qlib_joint_model_enabled = false;
        taskForm.qlib_residual_distillation_enabled = false;
        taskForm.qlib_adaptive_budget_enabled = false;
        taskForm.qlib_dynamic_trial_governance_enabled = false;
      }
      const name = "qlib_alpha158_prior";
      const selected = taskForm.search_algorithms.includes(name);
      if (taskForm.qlib_integration_enabled && taskForm.qlib_alpha158_prior_enabled && !selected) taskForm.search_algorithms.push(name);
      if ((!taskForm.qlib_integration_enabled || !taskForm.qlib_alpha158_prior_enabled) && selected) taskForm.search_algorithms = taskForm.search_algorithms.filter(value => value !== name);
      const jointName = "qlib_joint_residual_distill";
      const jointSelected = taskForm.search_algorithms.includes(jointName);
      if (taskForm.qlib_integration_enabled && taskForm.qlib_joint_model_enabled && !jointSelected) taskForm.search_algorithms.push(jointName);
      if ((!taskForm.qlib_integration_enabled || !taskForm.qlib_joint_model_enabled) && jointSelected) taskForm.search_algorithms = taskForm.search_algorithms.filter(value => value !== jointName);
      if (!taskForm.qlib_joint_model_enabled) taskForm.qlib_residual_distillation_enabled = false;
    }
    onMounted(load);
    return { llm, eng, evalProtocol, save, msg, saved, taskForm, taskMsg, taskOk, serviceMarket, createTask, syncMarketDefaults,
      architectureTemplates, architectureAlgorithms, architectureSchema, selectedArchitecture, layer1Label,
      algorithmLabel, algorithmDescription,
      applyArchitectureTemplate, syncCustomArchitecture, syncQlibIntegration };
  },
};

/* ============ 实验管理 ============ */
const ExperimentsView = {
  template: `
  <div>
    <div class="card" style="margin-bottom:14px">
      <h3>研究任务与并行运行</h3>
      <div class="warn-banner">为避免隐式默认和错误层级组合，“基础创建”已取消。请在“设置 → 定义研究任务”中选择无 LLM、随机→LLM、算法池→LLM、完整三层或自定义架构；创建后回到本页启动和管理。</div>
      <div v-if="err" style="color:var(--red); margin-top:8px">{{ err }}</div>
    </div>
    <div class="card">
      <h3>研究任务列表 ({{ exps.length }})</h3>
      <table>
        <tr><th>#</th><th>名称</th><th>市场 / 约束</th><th>协议</th><th>任务状态</th><th>运行态</th><th>因子</th><th>节点</th><th>外层步</th><th>创建时间</th><th style="min-width:250px">操作</th></tr>
        <tr v-for="e in exps" :key="e.id" :style="{background: e.active ? '#1c2733' : ''}">
          <td>{{ e.id }}</td>
          <td>
            <input v-if="editing===e.id" v-model="editForm.name" style="width:180px" />
            <template v-else><b>{{ e.name }}</b> <span v-if="e.active" class="tag green">活动</span></template>
          </td>
          <td><span class="tag blue">{{ e.research_config?.market==='ashare' ? 'A股' : '美股' }}</span> <span class="sub">{{ e.research_config?.portfolio_mode==='long_only' ? '纯多头' : '多空' }} · {{ e.research_config?.direction_policy==='both_train_select' ? '双向训练，选中后冻结' : (Number(e.research_config?.direction || 1)===1 ? '固定 +1' : '固定 -1') }}</span><div class="sub">{{ architectureLabel(e.research_config) }} · 预算 {{ e.research_config?.candidate_evaluation_budget || '旧任务未限定' }}</div><div v-if="e.research_config?.qlib_integration?.effective" class="sub"><span class="tag green">Qlib</span> Alpha158先验{{ e.research_config?.qlib_integration?.joint_model_enabled ? ' · 联合模型 · Residual→DSL' : '' }}{{ e.research_config?.qlib_integration?.adaptive_budget_enabled ? ' · 自适应预算' : '' }}</div></td>
          <td><span class="tag" :class="String(e.research_config?.evaluation_protocol || '').startsWith('v4.')?'green':'amber'">{{ e.research_config?.evaluation_protocol || 'legacy' }}</span><div v-if="e.research_config?.provenance_warning" class="provenance-dot" :title="e.research_config.provenance_warning">来源警告</div></td>
          <td><span class="tag" :class="{green: e.status==='open', amber: e.status==='archived'}">{{ e.status }}</span></td>
          <td><span class="tag" :class="{green: runtime[e.id]?.state==='running', amber: runtime[e.id]?.state==='starting', red: runtime[e.id]?.state==='stopped'}">{{ runtime[e.id]?.state || 'stopped' }}</span><div class="sub" v-if="runtime[e.id]?.global_progress!=null">全局预算 {{ (Number(runtime[e.id].global_progress)*100).toFixed(1) }}% · LLM {{ runtime[e.id]?.llm_calls || 0 }}/{{ runtime[e.id]?.max_llm_calls || '—' }}</div><div class="sub" v-if="runtime[e.id]?.runtime_identity?.code_short">代码 {{ runtime[e.id].runtime_identity.code_short }}</div></td>
          <td>{{ e.counts.factors }}</td><td>{{ e.counts.nodes }}</td><td>{{ e.counts.outer_steps }}</td>
          <td class="sub">{{ e.created_at?.slice(0,16) }}</td>
          <td>
            <template v-if="editing===e.id">
              <button class="btn primary" @click="saveEdit(e)">保存</button>
              <button class="btn" @click="editing=null">取消</button>
            </template>
            <template v-else>
              <button class="btn" v-if="!e.active && e.status==='open'" @click="activate(e)">查看任务</button>
              <button class="btn primary" v-if="e.status==='open' && runtime[e.id]?.state!=='running'" @click="start(e)">启动 V2</button>
              <button class="btn danger" v-else-if="runtime[e.id]?.state==='running'" @click="stop(e)">停止</button>
              <button class="btn" @click="startEdit(e)">编辑</button>
              <button class="btn" v-if="e.status==='open'" @click="setStatus(e,'archived')">归档</button>
              <button class="btn" v-else @click="setStatus(e,'open')">重新开放</button>
            </template>
          </td>
        </tr>
      </table>
      <div class="sub" style="margin-top:8px">任务可以并行运行；历史数据不可物理删除，只能归档。点击“查看任务”只切换当前观察对象，不会停止其他 worker。</div>
    </div>
  </div>`,
  setup() {
    const exps = ref([]), runtime = reactive({});
    const editForm = reactive({ name: "", description: "" });
    const editing = ref(null), err = ref("");
    let loadedExperimentVersion = -1;
    async function refresh() {
      const [experiments, obs] = await Promise.all([
        api("/experiments", { cacheTtl: 800 }),
        api("/observability", { cacheTtl: 800 }),
      ]);
      exps.value = experiments.experiments;
      Object.keys(runtime).forEach(key => delete runtime[key]);
      (obs.workers || []).forEach(w => { runtime[w.experiment_id] = w; });
      loadedExperimentVersion = appState.experimentVersion;
    }
    function ensureFresh() {
      if (loadedExperimentVersion !== appState.experimentVersion) refresh();
    }
    function architectureLabel(config={}) {
      const layers = [config.layer1_enabled ? "L1" : "", config.layer2_enabled ? "L2" : "", config.layer3_enabled ? "L3" : ""].filter(Boolean).join("→");
      const known = { random_only:"随机无LLM", algorithm_pool_only:"算法池无LLM", random_researcher:"随机→Researcher", algorithm_pool_researcher:"算法池→Researcher", full_three_layer:"完整三层", direct_researcher:"直接Researcher" };
      if (known[config.architecture_template]) return known[config.architecture_template];
      if (layers) return `${layers}${config.memory_mode==='cold' && config.layer2_enabled ? '·冷记忆' : ''}`;
      return config.proposal_mode==='random' ? "历史随机基线" : config.memory_mode==='cold' ? "历史LLM冷记忆" : "历史LLM连续记忆";
    }
    function startEdit(e) { editing.value = e.id; editForm.name = e.name; editForm.description = e.description; }
    async function saveEdit(e) {
      err.value = "";
      try { await api(`/experiments/${e.id}`, { method: "PATCH", body: { ...editForm } }); editing.value = null; refresh(); }
      catch (ex) { err.value = ex.message; }
    }
    async function setStatus(e, status) {
      err.value = "";
      try { await api(`/experiments/${e.id}`, { method: "PATCH", body: { status } }); refresh(); }
      catch (ex) { err.value = ex.message; }
    }
    async function activate(e) {
      err.value = "";
      try { await activateExperiment(e.id); await refresh(); }
      catch (ex) { err.value = ex.message; }
    }
    async function start(e) {
      err.value = "";
      try { await api("/engine/start", { method: "POST", body: { mode: e.research_config?.engine_mode || "v2", experiment_id: e.id } }); refresh(); }
      catch (ex) { err.value = ex.message; }
    }
    async function stop(e) {
      try { await api("/engine/stop", { method: "POST", body: { experiment_id: e.id } }); refresh(); }
      catch (ex) { err.value = ex.message; }
    }
    watch(
      () => appState.experimentVersion,
      () => { if (appState.activeTab === "exps") ensureFresh(); },
    );
    onActivated(ensureFresh);
    return { exps, runtime, editForm, editing, err, architectureLabel, startEdit, saveEdit, setStatus, activate, start, stop };
  },
};

/* ============ Engineering observability ============ */
const ObservabilityView = {
  template: `
  <section class="ops-page">
    <div class="ops-heading">
      <div>
        <div class="eyebrow">RUNTIME / DATA / DATABASE / WORKERS</div>
        <h1>工程诊断台</h1>
        <p>面向排障的脱敏快照；活动面板支持双缓冲热重载，构建失败时继续服务旧代际。</p>
      </div>
      <div class="ops-actions">
        <span v-if="snapshot" class="ops-health" :class="'health-' + snapshot.health">
          ● {{ healthLabel(snapshot.health) }}
        </span>
        <button class="btn" @click="togglePause">{{ paused ? '继续自动刷新' : '暂停自动刷新' }}</button>
        <button class="btn" @click="reloadActivePanel" :disabled="panelReloading || !canReloadPanel">{{ panelReloading ? '面板重载中…' : '热重载活动面板' }}</button>
        <button class="btn primary" @click="load(true)" :disabled="loading">{{ loading ? '刷新中…' : '深度刷新' }}</button>
        <button class="btn" @click="copySnapshot" :disabled="!snapshot">{{ copied || '复制脱敏快照' }}</button>
      </div>
    </div>

    <div v-if="error" class="selector-error">{{ error }}</div>
    <div v-if="panelReloadMessage" class="selector-disclaimer"><span>ⓘ</span>{{ panelReloadMessage }}</div>
    <div v-if="!snapshot && loading" class="card ops-loading"><div class="loading-ring"></div>正在采集工程快照</div>

    <template v-if="snapshot">
      <div class="ops-meta">
        <span>schema {{ snapshot.schema_version }}</span>
        <span>生成 {{ formatDate(snapshot.generated_at) }}</span>
        <span>自动刷新 {{ paused ? '已暂停' : '5 秒' }}</span>
        <span>采集 {{ n(snapshot.collector.total_ms, 1) }} ms · 慢层缓存 {{ n(snapshot.caches.observability_components.hit_rate * 100, 0) }}%</span>
        <a href="/api/health/live" target="_blank">liveness</a>
        <a href="/api/health/ready" target="_blank">readiness</a>
        <a href="/api/metrics" target="_blank">Prometheus</a>
      </div>

      <div class="ops-findings">
        <article v-for="finding in snapshot.findings" :key="finding.code"
          class="ops-finding" :class="'finding-' + finding.severity">
          <div><span>{{ finding.severity.toUpperCase() }}</span><b>{{ finding.title }}</b></div>
          <p>{{ finding.detail }}</p>
          <small>{{ finding.action }}</small>
        </article>
      </div>

      <div class="ops-slo">
        <div class="panel-title-row">
          <div><h2>运行 SLO</h2><p>{{ snapshot.slo.note }}</p></div>
          <span class="tag" :class="{green:snapshot.slo.status==='pass', red:snapshot.slo.status==='fail'}">{{ snapshot.slo.failed }} failed</span>
        </div>
        <div class="ops-slo-grid">
          <article v-for="objective in snapshot.slo.objectives" :key="objective.code" :class="'slo-' + objective.status">
            <span>{{ objective.label }}</span>
            <b>{{ sloValue(objective) }}</b>
            <small>{{ objective.target }} · {{ objective.status }}</small>
          </article>
        </div>
      </div>

      <div class="ops-metrics">
        <article class="metric-card">
          <span>服务 / 部署</span>
          <b>PID {{ snapshot.service.pid }}</b>
          <small>{{ formatDuration(snapshot.service.uptime_seconds) }} · {{ snapshot.service.deployment.commit_short }}</small>
          <small>{{ snapshot.service.deployment.branch }}<template v-if="snapshot.service.deployment.dirty_at_start"> · dirty-at-start</template> · {{ snapshot.service.network.bind }}</small>
        </article>
        <article class="metric-card" :class="{danger: snapshot.requests.window.server_errors}">
          <span>HTTP · {{ snapshot.requests.window.seconds }} 秒窗口</span>
          <b>{{ snapshot.requests.window.latency_ms.p95 }} ms</b>
          <small>P95 · {{ snapshot.requests.window.requests }} 请求 · {{ snapshot.requests.window.server_errors }} 个 5xx</small>
          <small>{{ snapshot.requests.in_flight }} in-flight · 生命周期 {{ snapshot.requests.lifetime.requests }}</small>
        </article>
        <article class="metric-card">
          <span>进程 / 事件循环</span>
          <b>{{ formatBytes(snapshot.process.rss_bytes) }}</b>
          <small>RSS · CPU {{ n(snapshot.process.cpu_percent, 1) }}% · {{ snapshot.process.native_threads ?? snapshot.process.thread_count }} threads</small>
          <small>loop P95 {{ n(snapshot.requests.event_loop.p95_lag_ms, 2) }} ms · {{ snapshot.process.asyncio_tasks.active }} tasks</small>
        </article>
        <article class="metric-card" :class="{danger: snapshot.database.status!=='ok'}">
          <span>PostgreSQL / 连接池</span>
          <b>{{ snapshot.database.status }} · {{ n(snapshot.database.latency_ms, 1) }} ms</b>
          <small>{{ snapshot.database.pool.checked_out }}/{{ snapshot.database.pool.capacity }} checked out · {{ pct(snapshot.database.pool.utilization) }}</small>
          <small>{{ snapshot.database.driver }} · {{ snapshot.database.pool.class }}</small>
        </article>
        <article class="metric-card" :class="{danger:snapshot.data.reload_errors}">
          <span>数据面板</span>
          <b>{{ snapshot.data.loaded }}/{{ snapshot.data.instances }} loaded</b>
          <small>{{ snapshot.data.stale || 0 }} stale · {{ snapshot.data.reloading || 0 }} reloading · {{ snapshot.data.reload_errors || 0 }} reload error</small>
          <small>{{ totalPanelFiles }} files · {{ formatBytes(snapshot.data.estimated_size_bytes) }} memory</small>
        </article>
        <article class="metric-card">
          <span>计算缓存</span>
          <b>{{ pct(snapshot.caches.screener.hit_rate) }}</b>
          <small>选股命中 · {{ snapshot.caches.screener.entries }}/{{ snapshot.caches.screener.capacity }} entries</small>
          <small>相似度 {{ pct(snapshot.caches.factor_similarity.hit_rate) }} · 前端 {{ clientTelemetry.cacheEntries }} entries</small>
        </article>
        <article class="metric-card" :class="{danger: !snapshot.service.network.loopback_only}">
          <span>网络安全边界</span>
          <b>{{ snapshot.service.network.scope }}</b>
          <small>{{ snapshot.service.network.bind }} · {{ snapshot.service.network.loopback_only ? '仅本机可访问' : '无认证远程暴露' }}</small>
          <small>API no-store · frame deny · nosniff</small>
        </article>
        <article class="metric-card" :class="{danger: snapshot.process.supervised_tasks.failed}">
          <span>后台任务 / 事故留痕</span>
          <b>{{ snapshot.process.supervised_tasks.running }} / {{ snapshot.process.supervised_tasks.tracked }}</b>
          <small>running / tracked · {{ snapshot.process.supervised_tasks.failed }} failed</small>
          <small>journal {{ formatBytes(snapshot.service.journal.current_bytes) }} · {{ snapshot.service.journal.error || 'writable' }}</small>
        </article>
        <article class="metric-card" :class="{danger: snapshot.llm_pipeline?.calls?.errors_1h}">
          <span>双层 LLM 反馈闭环</span>
          <b>{{ pct(snapshot.llm_pipeline?.feedback_coverage?.nodes_ratio) }}</b>
          <small>节点反馈覆盖 · report {{ pct(snapshot.llm_pipeline?.feedback_coverage?.reports_ratio) }} · reflection {{ pct(snapshot.llm_pipeline?.feedback_coverage?.reflections_ratio) }}</small>
          <small>{{ snapshot.llm_pipeline?.calls?.calls_1h ?? 0 }} calls / 1h · {{ snapshot.llm_pipeline?.calls?.errors_1h ?? 0 }} errors · P95 {{ n(snapshot.llm_pipeline?.calls?.p95_latency_ms_1h, 1) }} ms</small>
        </article>
      </div>

      <div class="grid cols-2 ops-grid">
        <div class="card ops-table-card">
          <div class="panel-title-row"><div><h2>研究 worker</h2><p>{{ snapshot.engine.running_count }} running / {{ snapshot.engine.worker_count }} registered</p></div></div>
          <div class="ops-table-scroll">
            <table>
              <tr><th>任务</th><th>状态 / 阶段</th><th>当前工作</th><th>进度</th><th>心跳</th></tr>
              <tr v-for="worker in snapshot.workers" :key="worker.experiment_id">
                <td>#{{ worker.experiment_id }}<div class="sub">{{ worker.mode || '—' }}</div></td>
                <td><span class="tag" :class="{green:worker.running, red:worker.task_exception, amber:worker.heartbeat_stale}">{{ worker.state }}</span><div class="sub">{{ worker.phase }}</div></td>
                <td>{{ worker.current_task || '—' }}<div class="sub">{{ worker.current_operation || '—' }}</div></td>
                <td>{{ worker.current_budget_index ?? '—' }}/{{ worker.current_budget_total ?? '—' }}<div class="sub">outer {{ worker.outer_step }} · eval {{ worker.inner_evals }}</div></td>
                <td :class="{'bad-text':worker.heartbeat_stale}">{{ age(worker.heartbeat_age_seconds) }}<div class="sub">{{ formatDate(worker.last_heartbeat_at) }}</div></td>
              </tr>
              <tr v-if="!snapshot.workers.length"><td colspan="5" class="ops-empty">本进程还没有注册 worker</td></tr>
            </table>
          </div>
        </div>

        <div class="card ops-table-card">
          <div class="panel-title-row"><div><h2>面板身份与数据契约</h2><p>文件身份、Parquet schema、DSL 字段与加载状态</p></div></div>
          <div class="ops-table-scroll">
            <table>
              <tr><th>市场 / ID</th><th>状态</th><th>文件</th><th>样本</th><th>加载</th></tr>
              <tr v-for="panel in snapshot.data.panels" :key="panel.id">
                <td><span class="tag blue">{{ panel.market }}</span><div class="sub">source {{ panel.source_identity || panel.identity || panel.id }}</div><div class="sub">loaded {{ panel.loaded_identity || '—' }}</div></td>
                <td><span class="tag" :class="{green:panel.state==='ready' && panel.schema_status==='ok', red:panel.state==='error' || panel.reload_error, amber:panel.state==='loading' || panel.state==='stale' || panel.state==='reloading'}">{{ panel.state }} / {{ panel.schema_status }}</span><div v-if="panel.source_error || panel.load_error || panel.schema_error || panel.reload_error" class="bad-text ops-wrap">{{ panel.source_error || panel.load_error || panel.schema_error || panel.reload_error }}</div><div v-if="panel.missing_dsl_fields?.length" class="bad-text ops-wrap">missing DSL: {{ panel.missing_dsl_fields.join(', ') }}</div></td>
                <td>{{ panel.file_count }} · {{ formatBytes(panel.total_bytes) }}<div class="sub">{{ formatDate(panel.latest_mtime) }}</div></td>
                <td>{{ panel.rows ?? '—' }} rows<div class="sub">{{ panel.securities ?? '—' }} securities · {{ panel.date_min || '—' }} → {{ panel.date_max || '—' }}</div></td>
                <td>generation {{ panel.generation ?? 0 }}<div class="sub">load {{ panel.load_duration_ms == null ? '—' : n(panel.load_duration_ms,1)+' ms' }} · reload {{ panel.reload_duration_ms == null ? '—' : n(panel.reload_duration_ms,1)+' ms' }}</div><div class="sub">{{ panel.reload_count || 0 }} reload(s) · {{ panel.available_column_count ?? '—' }} cols</div></td>
              </tr>
            </table>
          </div>
          <details class="ops-details"><summary>显示面板路径</summary><code v-for="panel in snapshot.data.panels" :key="'path'+panel.id">{{ panel.source }}</code></details>
        </div>
      </div>

      <div class="card ops-table-card">
        <div class="panel-title-row"><div><h2>API 路由延迟与错误</h2><p>按 P95 排序；路由参数已归一化，避免指标基数爆炸</p></div><span class="count-badge">{{ snapshot.requests.routes.length }} routes</span></div>
        <div class="ops-table-scroll route-table">
          <table>
            <tr><th>路由</th><th>请求</th><th>5xx</th><th>4xx</th><th>平均</th><th>P50</th><th>P95</th><th>P99</th><th>最大</th><th>最后请求 ID</th></tr>
            <tr v-for="route in snapshot.requests.routes.slice(0,30)" :key="route.route">
              <td><code>{{ route.route }}</code></td><td>{{ route.count }}</td>
              <td :class="{'bad-text':route.errors}">{{ route.errors }}</td><td>{{ route.client_errors }}</td>
              <td>{{ n(route.avg_ms,2) }}</td><td>{{ n(route.p50_ms,2) }}</td>
              <td :class="{'bad-text':route.p95_ms>=1000}">{{ n(route.p95_ms,2) }}</td>
              <td>{{ n(route.p99_ms,2) }}</td><td>{{ n(route.max_ms,2) }}</td><td><code>{{ route.last_request_id }}</code></td>
            </tr>
          </table>
        </div>
      </div>

      <div class="grid cols-3 ops-grid">
        <div class="card ops-list-card">
          <div class="panel-title-row"><div><h2>当前进程异常</h2><p>HTTP 5xx、未捕获异常和 error 日志</p></div></div>
          <div class="ops-event-list">
            <article v-for="incident in snapshot.requests.incidents.slice(0,30)" :key="incident.at + incident.request_id + incident.message">
              <span class="tag red">{{ incident.kind || incident.level }}</span>
              <time>{{ formatDate(incident.at) }}</time>
              <code v-if="incident.request_id">{{ incident.request_id }} · {{ incident.route }}</code>
              <p>{{ incident.error || incident.message || ('HTTP ' + incident.status) }}</p>
            </article>
            <div v-if="!snapshot.requests.incidents.length" class="ops-empty">当前进程没有记录到异常</div>
          </div>
        </div>
        <div class="card ops-list-card">
          <div class="panel-title-row"><div><h2>跨重启事故日志</h2><p>滚动 JSONL 中保留的服务启停与严重错误</p></div></div>
          <div class="ops-event-list">
            <article v-for="event in snapshot.service.journal.recent_events" :key="event.at + event.kind + (event.request_id || '')">
              <span class="tag" :class="{red:event.kind?.includes('failure') || event.kind?.includes('5xx'), blue:event.kind?.includes('service_')}">{{ event.kind }}</span>
              <time>{{ formatDate(event.at) }}</time>
              <code>{{ event.request_id || event.task || ('PID ' + (event.pid ?? '—')) }}</code>
              <p>{{ event.error || event.message || event.route || event.commit || '—' }}</p>
            </article>
            <div v-if="!snapshot.service.journal.recent_events.length" class="ops-empty">尚无跨重启事件</div>
          </div>
        </div>
        <div class="card ops-list-card">
          <div class="panel-title-row"><div><h2>持久化引擎事件</h2><p>数据库中的最近事件；可按 experiment_id 交叉排查</p></div></div>
          <div class="ops-event-list">
            <article v-for="event in snapshot.recent_events" :key="event.id">
              <span class="tag" :class="{red:event.level==='error', amber:event.level==='warning'}">{{ event.level }}</span>
              <time>{{ formatDate(event.created_at) }}</time>
              <code>#{{ event.experiment_id ?? '—' }} · event {{ event.id }} · {{ event.payload?.phase || 'legacy' }}</code>
              <p>{{ event.message }}</p>
            </article>
            <div v-if="!snapshot.recent_events.length" class="ops-empty">{{ snapshot.recent_events_error || '没有事件' }}</div>
          </div>
        </div>
      </div>

      <div class="grid cols-3 ops-grid">
        <div class="card ops-kv">
          <h3>活动研究配置</h3>
          <dl>
            <template v-for="[key,value] in entries({
              id:snapshot.active_task.experiment_id,
              name:snapshot.active_task.name,
              task_status:snapshot.active_task.status,
              worker_state:snapshot.active_task.worker_state,
              worker_phase:snapshot.active_task.worker_phase,
              market:snapshot.active_task.market,
              portfolio_mode:snapshot.active_task.portfolio_mode,
              direction:snapshot.active_task.direction,
              direction_policy:snapshot.active_task.direction_policy,
              protocol:snapshot.active_task.evaluation_protocol,
              config_fingerprint:snapshot.active_task.config_fingerprint,
              panel_id:snapshot.active_task.panel_id
            })" :key="key"><dt>{{ key }}</dt><dd>{{ value ?? '—' }}</dd></template>
          </dl>
        </div>
        <div class="card ops-kv">
          <h3>LLM 路由（不含密钥）</h3>
          <dl>
            <dt>inner</dt><dd>{{ snapshot.providers.inner_provider || 'fallback' }} · {{ snapshot.providers.inner_provider_configured ? 'configured' : 'not configured' }}</dd>
            <dt>outer</dt><dd>{{ snapshot.providers.outer_provider || 'fallback' }} · {{ snapshot.providers.outer_provider_configured ? 'configured' : 'not configured' }}</dd>
          </dl>
          <div class="ops-provider" v-for="provider in snapshot.providers.providers" :key="provider.name">
            <b>{{ provider.name }}</b><span>{{ provider.model || '—' }}</span><small>{{ provider.endpoint_host || '—' }} · key {{ provider.api_key_present ? 'present' : 'absent' }}</small>
          </div>
        </div>
        <div class="card ops-kv">
          <h3>持久化与数据量</h3>
          <dl>
            <template v-for="[key,value] in entries(snapshot.database.counts)" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template>
            <dt>artifact_runs</dt><dd>{{ snapshot.artifacts.runs }}</dd>
            <dt>artifact_files</dt><dd>{{ snapshot.artifacts.files }}</dd>
            <dt>artifact_bytes</dt><dd>{{ formatBytes(snapshot.artifacts.total_bytes) }}</dd>
            <dt>artifact_latest</dt><dd>{{ formatDate(snapshot.artifacts.latest_mtime) }}</dd>
          </dl>
        </div>
      </div>

      <div class="grid cols-2 ops-grid">
        <div class="card ops-table-card">
          <div class="panel-title-row">
            <div><h2>LLM 调用与反馈血缘</h2><p>仅显示脱敏元数据；prompt hash 与反馈 fingerprint 可逐次核对</p></div>
            <a href="/api/llm/audits" target="_blank">打开审计 API</a>
          </div>
          <div class="ops-table-scroll">
            <table>
              <tr><th>ID / 时间</th><th>角色 / 阶段</th><th>任务 / 版本</th><th>状态 / 延迟</th><th>反馈指纹</th></tr>
              <tr v-for="call in (snapshot.llm_pipeline?.recent_calls || [])" :key="call.id">
                <td>#{{ call.id }}<div class="sub">{{ formatDate(call.created_at) }}</div></td>
                <td>{{ call.role }}<div class="sub">{{ call.phase }}</div></td>
                <td>#{{ call.experiment_id ?? '—' }} · {{ call.task_name || 'outer' }}<div class="sub">miner {{ call.miner_version_id ?? '—' }} · step {{ call.outer_step_no ?? '—' }}</div></td>
                <td><span class="tag" :class="{green:call.status==='accepted',amber:call.status==='response_ok',red:call.status==='transport_error'||call.status==='rejected'}">{{ call.status }}</span><div class="sub">{{ n(call.latency_ms,1) }} ms · {{ call.model || '—' }}</div><div v-if="call.error" class="bad-text ops-wrap">{{ call.error }}</div></td>
                <td><code>{{ call.feedback_fingerprint || 'no-context' }}</code><div class="sub">{{ call.evaluation_protocol }} · prompt {{ String(call.prompt_hash || '').slice(0,12) }}</div></td>
              </tr>
              <tr v-if="!(snapshot.llm_pipeline?.recent_calls || []).length"><td colspan="5" class="ops-empty">尚无新版 LLM 调用审计；随机回退不会伪装成 LLM 调用。</td></tr>
            </table>
          </div>
        </div>
        <div class="card ops-table-card">
          <div class="panel-title-row"><div><h2>评价协议隔离</h2><p>{{ snapshot.protocol_lineage?.policy }}</p></div><span class="tag green">{{ snapshot.protocol_lineage?.current_protocol }}</span></div>
          <div class="ops-table-scroll">
            <table>
              <tr><th>持久化表</th><th>协议计数</th></tr>
              <tr v-for="[tableName,protocols] in entries(snapshot.protocol_lineage?.tables)" :key="tableName">
                <td><code>{{ tableName }}</code></td>
                <td><span v-for="[protocol,count] in entries(protocols)" :key="protocol" class="tag" :class="protocol===snapshot.protocol_lineage?.current_protocol?'green':'amber'">{{ protocol }} · {{ count }}</span></td>
              </tr>
            </table>
          </div>
          <details class="ops-details">
            <summary>反馈隔离契约</summary>
            <pre>{{ pretty(snapshot.llm_pipeline?.isolation || {}) }}</pre>
          </details>
        </div>
      </div>

      <details class="card ops-raw">
        <summary>原始脱敏快照 / 运行日志 / 线程与 asyncio task 明细</summary>
        <pre>{{ pretty(snapshot) }}</pre>
      </details>
    </template>
  </section>`,
  setup() {
    const snapshot = ref(null), error = ref(""), loading = ref(false);
    const paused = ref(false), copied = ref("");
    const panelReloading = ref(false), panelReloadMessage = ref("");
    let timer = null, active = false;
    const clientTelemetry = computed(() => ({
      cacheEntries: responseCache.size,
      inflightGets: inflightGets.size,
      generation: apiCacheGeneration,
    }));
    const canReloadPanel = computed(() => Boolean(
      snapshot.value && snapshot.value.active_task && snapshot.value.active_task.experiment_id
    ));
    const totalPanelFiles = computed(() =>
      (snapshot.value?.data?.panels || []).reduce((sum, panel) => sum + Number(panel.file_count || 0), 0)
    );
    async function load(force = false) {
      if (loading.value) return;
      loading.value = true; error.value = "";
      try {
        const forceArg = force === true ? "&force=true" : "";
        snapshot.value = await api(`/observability?events_limit=40&window_seconds=300${forceArg}`);
      } catch (e) {
        error.value = `诊断快照加载失败：${e.message}`;
      } finally {
        loading.value = false;
      }
    }
    async function reloadActivePanel() {
      const experimentId = Number(snapshot.value?.active_task?.experiment_id);
      if (!experimentId || panelReloading.value) return;
      panelReloading.value = true; panelReloadMessage.value = ""; error.value = "";
      try {
        const response = await api("/panels/reload", {
          method: "POST",
          body: { experiment_id: experimentId, force: false },
        });
        const result = response.result || {};
        panelReloadMessage.value = result.status === "reloaded"
          ? `活动面板已切换到 generation ${result.generation}，最新交易日 ${result.date_max}；旧请求未中断。`
          : result.status === "unchanged"
            ? `活动面板已经是最新 generation ${result.generation}。`
            : `面板重载状态：${result.status}；${result.error || '旧 generation 继续服务。'}`;
        await load(true);
      } catch (e) {
        error.value = `活动面板热重载失败：${e.message}`;
      } finally {
        panelReloading.value = false;
      }
    }
    function startPolling() {
      if (active) return;
      active = true;
      if (!timer) timer = setInterval(() => {
        if (active && !paused.value) load();
      }, 5000);
      load();
    }
    function stopPolling() {
      active = false;
      if (timer) { clearInterval(timer); timer = null; }
    }
    function togglePause() { paused.value = !paused.value; if (!paused.value) load(); }
    async function copySnapshot() {
      try {
        await navigator.clipboard.writeText(JSON.stringify(snapshot.value, null, 2));
        copied.value = "已复制";
      } catch (e) {
        copied.value = "复制失败";
      }
      setTimeout(() => { copied.value = ""; }, 1600);
    }
    function n(value, digits = 0) {
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : "—";
    }
    function pct(value) {
      if (value == null) return "—";
      const number = Number(value);
      return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
    }
    function formatBytes(value) {
      let number = Number(value);
      if (!Number.isFinite(number)) return "—";
      const units = ["B", "KiB", "MiB", "GiB", "TiB"];
      let index = 0;
      while (number >= 1024 && index < units.length - 1) { number /= 1024; index += 1; }
      return `${number.toFixed(index ? 1 : 0)} ${units[index]}`;
    }
    function formatDuration(seconds) {
      const value = Number(seconds);
      if (!Number.isFinite(value)) return "—";
      if (value < 60) return `${value.toFixed(0)}s`;
      if (value < 3600) return `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`;
      return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
    }
    function age(seconds) {
      return seconds == null ? "—" : `${formatDuration(seconds)} ago`;
    }
    function formatDate(value) {
      if (!value) return "—";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
    }
    function healthLabel(value) {
      return ({ healthy: "健康", degraded: "降级", unhealthy: "故障" })[value] || value;
    }
    function sloValue(objective) {
      if (objective.status === "no_data") return "NO DATA";
      if (objective.code === "http_5xx_rate" || objective.code === "disk_free") return pct(objective.value);
      if (objective.code === "http_p95" || objective.code === "database" || objective.code === "event_loop") return `${n(objective.value, 1)} ms`;
      return String(objective.value ?? "—");
    }
    function entries(value) { return Object.entries(value || {}); }
    function pretty(value) { return JSON.stringify(value, null, 2); }
    onMounted(startPolling);
    onActivated(startPolling);
    onDeactivated(stopPolling);
    onUnmounted(stopPolling);
    return {
      snapshot, error, loading, paused, copied, panelReloading,
      panelReloadMessage, canReloadPanel, clientTelemetry, totalPanelFiles,
      load, reloadActivePanel, togglePause, copySnapshot, n, pct, formatBytes, formatDuration,
      age, formatDate, healthLabel, sloValue, entries, pretty,
    };
  },
};

/* ============ 榜单版本中心 ============ */
const LeaderboardsView = {
  template: `
  <section class="leaderboards-page" data-testid="leaderboards-page">
    <div class="leaderboards-heading">
      <div>
        <div class="eyebrow">IMMUTABLE REPORT CATALOG / PROTOCOL-AWARE NAVIGATION</div>
        <h1>因子榜单版本中心</h1>
        <p>统一查看 A股、美股纯多与美股多空的历史榜和全区间榜；每个入口都保留原始协议、数据窗口与审计边界。</p>
      </div>
      <div class="leaderboards-heading-actions">
        <span class="tag amber">研究用途 · 非实盘批准</span>
        <button class="btn" @click="load(true)" :disabled="loading">{{ loading ? '扫描中…' : '重新扫描' }}</button>
      </div>
    </div>

    <div class="leaderboard-guide-grid">
      <article v-for="item in catalog.guidance || []" :key="item.title" class="leaderboard-guide-card">
        <span>{{ item.title }}</span>
        <p>{{ item.text }}</p>
      </article>
    </div>

    <div class="leaderboard-summary-strip">
      <div><b>{{ catalog.summary?.logical_versions ?? '—' }}</b><span>逻辑版本</span></div>
      <div><b>{{ catalog.summary?.complete_versions ?? '—' }}</b><span>完整完成</span></div>
      <div><b>{{ catalog.summary?.archive_copies ?? '—' }}</b><span>归档副本</span></div>
      <div><b>{{ filteredReports.length }}</b><span>当前筛选</span></div>
    </div>

    <div class="leaderboard-toolbar card">
      <div class="leaderboard-filter-group" aria-label="市场筛选">
        <span>市场</span>
        <button v-for="item in marketFilters" :key="item.id" :class="{active: marketFilter===item.id}" @click="marketFilter=item.id">{{ item.label }}</button>
      </div>
      <div class="leaderboard-filter-group" aria-label="模式筛选">
        <span>组合</span>
        <button v-for="item in modeFilters" :key="item.id" :class="{active: modeFilter===item.id}" @click="modeFilter=item.id">{{ item.label }}</button>
      </div>
      <div class="leaderboard-filter-group" aria-label="版本筛选">
        <span>口径</span>
        <button v-for="item in kindFilters" :key="item.id" :class="{active: kindFilter===item.id}" @click="kindFilter=item.id">{{ item.label }}</button>
      </div>
      <input v-model.trim="query" class="leaderboard-search" type="search" placeholder="搜索市场、协议、目录或版本…" aria-label="搜索榜单版本" />
    </div>

    <div v-if="error" class="selector-error">榜单目录加载失败：{{ error }}</div>
    <div v-if="loading && !catalog.reports?.length" class="leaderboard-loading card"><div class="loading-ring"></div><span>正在读取不可变报告与协议元数据…</span></div>
    <div v-else-if="!filteredReports.length" class="leaderboard-loading card"><span>没有符合当前筛选条件的榜单版本。</span></div>

    <div v-else class="leaderboard-workbench">
      <aside class="leaderboard-version-rail" aria-label="榜单版本列表">
        <button v-for="report in filteredReports" :key="report.id"
          class="leaderboard-version-card" :class="{active:selected?.id===report.id}"
          :data-report-id="report.id" @click="selectReport(report)">
          <div class="leaderboard-version-topline">
            <span class="leaderboard-market" :class="report.market">{{ report.market_label }}</span>
            <span>{{ report.mode_label }}</span>
            <i v-if="report.is_latest_for_scope">该模式最新</i>
          </div>
          <strong>{{ report.protocol_label }}</strong>
          <small>{{ windowLabel(report) }}</small>
          <div class="leaderboard-version-meta">
            <span>{{ report.result_count.toLocaleString() }} 方向/结果</span>
            <span>{{ formatDate(report.generated_at) }}</span>
          </div>
          <div class="leaderboard-version-flags">
            <em :class="report.status==='complete' ? 'ok' : 'warn'">{{ statusLabel(report) }}</em>
            <em>{{ report.screening_only ? '全样本诊断' : '冻结方向' }}</em>
            <em v-if="report.archive_copy_count">归档 ×{{ report.archive_copy_count }}</em>
          </div>
        </button>
      </aside>

      <main v-if="selected" class="leaderboard-viewer card" data-testid="leaderboard-viewer">
        <div class="leaderboard-viewer-head">
          <div>
            <div class="eyebrow">{{ selected.protocol }}</div>
            <h2>{{ selected.title }}</h2>
            <p>{{ selected.description }}</p>
          </div>
          <a class="btn-link primary" :href="viewerUrl" target="_blank" rel="noopener">独立打开 ↗</a>
        </div>

        <div class="leaderboard-boundary" :class="selected.screening_only ? 'diagnostic' : 'holdout'">
          <b>{{ selected.screening_only ? '全样本诊断边界' : '样本外 / Vault 边界' }}</b>
          <span v-if="selected.screening_only">全区间参与排名，不是独立样本外；请结合事件复核与旧版 Vault 阅读。</span>
          <span v-else>榜单期与 Vault 分层，Vault 不参与排序；底层仍为 NON-PIT 研究数据。</span>
        </div>

        <div class="leaderboard-facts">
          <div><span>排名窗口</span><b>{{ windowLabel(selected) }}</b></div>
          <div><span>数据截止</span><b>{{ selected.data_end || '—' }}</b></div>
          <div><span>方向策略</span><b>{{ directionLabel(selected.direction_policy) }}</b></div>
          <div><span>复核层</span><b>{{ selected.has_vault ? 'Vault + 事件账本' : selected.has_event_replay ? '事件账本' : '报告内审计' }}</b></div>
          <div><span>政策标签</span><b>{{ selected.policy_label }}</b></div>
        </div>

        <nav class="leaderboard-jumpbar" aria-label="报告快速跳转">
          <span>快速跳转</span>
          <button v-for="link in selected.quick_links" :key="link.id" :class="{active:anchor===link.id}" @click="anchor=link.id">{{ link.label }}</button>
          <i>点击报告中的因子行查看经济机制与公式</i>
        </nav>
        <iframe ref="leaderboardFrame" class="leaderboard-frame" :key="selected.id + ':' + anchor" :src="viewerUrl"
          @load="bindLeaderboardFrame"
          :title="selected.title" loading="lazy"></iframe>
      </main>
    </div>

    <div v-if="factorDetail.open" class="leaderboard-detail-backdrop" @click.self="closeFactorDetail">
      <aside class="leaderboard-detail-card" role="dialog" aria-modal="true" aria-label="榜单因子详情">
        <header class="leaderboard-detail-head">
          <div>
            <div class="eyebrow">FROZEN REPORT FACTOR / STRUCTURAL INTERPRETATION</div>
            <h2 v-if="factorDetail.data">#{{ factorDetail.data.metrics?.overall_rank ?? '—' }} · {{ factorDetail.data.identity.expression_hash }}</h2>
            <h2 v-else>正在读取榜单快照…</h2>
            <p>{{ selected?.title }}</p>
          </div>
          <button class="btn leaderboard-detail-close" type="button" aria-label="关闭因子详情" @click="closeFactorDetail">×</button>
        </header>

        <div v-if="factorDetail.loading" class="leaderboard-detail-loading"><div class="loading-ring"></div><span>读取不可变榜单行并生成结构解释…</span></div>
        <div v-else-if="factorDetail.error" class="selector-error">详情加载失败：{{ factorDetail.error }}</div>
        <template v-else-if="factorDetail.data">
          <section class="leaderboard-detail-section">
            <div class="leaderboard-detail-section-title"><h3>公式与冻结方向</h3><span>{{ factorDirectionLabel(factorDetail.data) }}</span></div>
            <div ref="leaderboardLatexEl" class="leaderboard-detail-latex" data-testid="leaderboard-factor-latex"></div>
            <code class="leaderboard-detail-dsl">{{ factorDetail.data.expression }}</code>
            <p class="leaderboard-direction-note">{{ factorDetail.data.economics.direction_interpretation }}</p>
          </section>

          <section class="leaderboard-detail-section">
            <div class="leaderboard-detail-section-title"><h3>可能的经济学含义</h3><span class="tag blue">{{ factorDetail.data.economics.mechanism_label }}</span></div>
            <div class="leaderboard-economics-hero">
              <span>可能收益来源</span>
              <strong>{{ factorDetail.data.economics.possible_return_source }}</strong>
              <p>{{ factorDetail.data.economics.rationale }}</p>
            </div>
            <div class="leaderboard-disclaimer">{{ factorDetail.data.economics.disclaimer }}</div>
          </section>

          <section class="leaderboard-detail-section">
            <div class="leaderboard-detail-section-title"><h3>该榜单快照中的证据</h3><span>{{ factorDetail.data.report.screening_only ? '全窗口诊断' : '冻结样本外口径' }}</span></div>
            <div class="leaderboard-detail-metrics">
              <div><span>15bps {{ factorDetail.data.metrics?.portfolio_metric_basis === 'active' ? '主动' : '净' }}年化</span><b>{{ factorPct(factorDetail.data.metrics?.ranking_ann_return_bps_15) }}</b></div>
              <div><span>15bps {{ factorDetail.data.metrics?.portfolio_metric_basis === 'active' ? '主动' : '净' }} Sharpe</span><b>{{ factorNum(factorDetail.data.metrics?.ranking_sharpe_bps_15) }}</b></div>
              <div><span>IC / ICIR</span><b>{{ factorNum(factorDetail.data.metrics?.oos_ic_mean, 4) }} / {{ factorNum(factorDetail.data.metrics?.oos_icir) }}</b></div>
              <div><span>RankIC / IR</span><b>{{ factorNum(factorDetail.data.metrics?.oos_rank_ic_mean, 4) }} / {{ factorNum(factorDetail.data.metrics?.oos_rank_icir) }}</b></div>
            </div>
            <div class="leaderboard-structure-tags">
              <span>字段 {{ factorDetail.data.fields.join(', ') || '—' }}</span>
              <span>算子 {{ factorDetail.data.operators.join(', ') || '—' }}</span>
              <span>历史 {{ factorDetail.data.required_history ?? '—' }} 日</span>
              <span>复杂度 {{ factorDetail.data.complexity ?? '—' }}</span>
            </div>
          </section>

          <section class="leaderboard-detail-section risk">
            <div class="leaderboard-detail-section-title"><h3>主要失效条件与实盘风险</h3><span>必须单独验证</span></div>
            <ul><li v-for="warning in factorDetail.data.economics.failure_modes" :key="warning">{{ warning }}</li></ul>
          </section>
        </template>
      </aside>
    </div>
  </section>`,
  setup() {
    const catalog = ref({ reports: [], guidance: [], summary: {} });
    const selected = ref(null);
    const loading = ref(false), error = ref("");
    const marketFilter = ref("all"), modeFilter = ref("all"), kindFilter = ref("all");
    const query = ref("");
    const anchor = ref("overview");
    const leaderboardFrame = ref(null), leaderboardLatexEl = ref(null);
    const factorDetail = reactive({ open:false, loading:false, error:"", data:null, reportId:"", hash:"" });
    const marketFilters = [
      { id: "all", label: "全部" }, { id: "ashare", label: "A股" }, { id: "us", label: "美股" },
    ];
    const modeFilters = [
      { id: "all", label: "全部" }, { id: "long_only", label: "纯多" }, { id: "long_short", label: "多空" },
    ];
    const kindFilters = [
      { id: "all", label: "全部" }, { id: "holdout_event", label: "样本外事件榜" }, { id: "full_window_vector", label: "全区间双向榜" },
    ];
    const filteredReports = computed(() => {
      const needle = query.value.toLowerCase();
      return (catalog.value.reports || []).filter((report) => {
        if (marketFilter.value !== "all" && report.market !== marketFilter.value) return false;
        if (modeFilter.value !== "all" && report.mode !== modeFilter.value) return false;
        if (kindFilter.value !== "all" && report.version_kind !== kindFilter.value) return false;
        if (!needle) return true;
        return [report.title, report.protocol, report.directory_name, report.generated_at]
          .some((value) => String(value || "").toLowerCase().includes(needle));
      });
    });
    const viewerUrl = computed(() => selected.value ? `${selected.value.file_url}#${anchor.value}` : "");
    async function load(force = false) {
      if (loading.value) return;
      loading.value = true;
      error.value = "";
      try {
        const suffix = force ? `?refresh=${Date.now()}` : "";
        catalog.value = await api("/leaderboards" + suffix, { cacheTtl: force ? 0 : 3000 });
        const saved = localStorage.getItem("factorfactory.leaderboard");
        selected.value = catalog.value.reports.find((report) => report.id === saved)
          || catalog.value.reports[0]
          || null;
      } catch (e) { error.value = e.message; }
      finally { loading.value = false; }
    }
    function selectReport(report) {
      closeFactorDetail();
      selected.value = report;
      anchor.value = "overview";
      localStorage.setItem("factorfactory.leaderboard", report.id);
    }
    function windowLabel(report) {
      const window = report?.ranking_window || {};
      return window.start && window.end ? `${window.start} → ${window.end}` : "窗口未标记";
    }
    function formatDate(value) {
      if (!value) return "—";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
    }
    function statusLabel(report) {
      return report.status === "complete" ? "完成" : report.status === "completed_with_gaps" ? "完成但有缺口" : "未完成";
    }
    function directionLabel(value) {
      return ({ disabled_both_directions_forced: "正反双向强制", train_frozen: "训练期冻结方向" })[value] || value || "—";
    }
    async function renderLeaderboardLatex() {
      await nextTick();
      const detail = factorDetail.data;
      const target = leaderboardLatexEl.value;
      if (!detail || !target) return;
      if (window.katex?.render) {
        window.katex.render(detail.latex || detail.expression, target, {
          throwOnError:false, strict:"warn", trust:false, displayMode:true,
        });
      } else {
        target.textContent = detail.latex || detail.expression;
        target.classList.add("latex-fallback");
      }
    }
    async function openLeaderboardFactor(hash) {
      const report = selected.value;
      if (!report || !hash) return;
      Object.assign(factorDetail, { open:true, loading:true, error:"", data:null, reportId:report.id, hash });
      try {
        const detail = await api(`/leaderboards/${encodeURIComponent(report.id)}/factors/${encodeURIComponent(hash)}`, { cacheTtl:30000 });
        if (factorDetail.reportId !== report.id || factorDetail.hash !== hash) return;
        factorDetail.data = detail;
        factorDetail.loading = false;
        await renderLeaderboardLatex();
      } catch (e) {
        if (factorDetail.reportId === report.id && factorDetail.hash === hash) factorDetail.error = e.message;
      } finally {
        if (factorDetail.reportId === report.id && factorDetail.hash === hash) factorDetail.loading = false;
      }
    }
    function bindLeaderboardFrame() {
      const frame = leaderboardFrame.value;
      try {
        const document = frame?.contentDocument;
        if (!document) return;
        if (frame.__factorClickDocument && frame.__factorClickHandler) {
          frame.__factorClickDocument.removeEventListener("click", frame.__factorClickHandler, true);
        }
        const handler = (event) => {
          const target = typeof event.target?.closest === "function" ? event.target.closest("[data-hash]") : null;
          const hash = target?.dataset?.hash;
          if (!hash) return;
          event.preventDefault();
          event.stopImmediatePropagation();
          openLeaderboardFactor(hash);
        };
        document.addEventListener("click", handler, true);
        frame.__factorClickDocument = document;
        frame.__factorClickHandler = handler;
      } catch (e) {
        console.warn("leaderboard factor bridge unavailable", e);
      }
    }
    function closeFactorDetail() {
      factorDetail.open = false;
      factorDetail.loading = false;
      factorDetail.error = "";
      factorDetail.data = null;
      factorDetail.reportId = "";
      factorDetail.hash = "";
    }
    function factorNum(value, digits = 2) {
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : "—";
    }
    function factorPct(value) {
      const number = Number(value);
      return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "—";
    }
    function factorDirectionLabel(detail) {
      const side = Number(detail?.direction || 1) > 0 ? "+1 · 高值端" : "−1 · 低值端";
      const mode = detail?.report?.portfolio_mode === "long_short" ? "多头 / 空头" : "纯多入选";
      return `${side} ${mode}`;
    }
    watch(filteredReports, (reports) => {
      if (reports.length && !reports.some((report) => report.id === selected.value?.id)) selectReport(reports[0]);
    });
    onMounted(load);
    onUnmounted(() => {
      const frame = leaderboardFrame.value;
      if (frame?.__factorClickDocument && frame?.__factorClickHandler) {
        frame.__factorClickDocument.removeEventListener("click", frame.__factorClickHandler, true);
      }
    });
    return {
      catalog, selected, loading, error, marketFilter, modeFilter, kindFilter, query, anchor,
      leaderboardFrame, leaderboardLatexEl, factorDetail,
      marketFilters, modeFilters, kindFilters, filteredReports, viewerUrl,
      load, selectReport, windowLabel, formatDate, statusLabel, directionLabel,
      bindLeaderboardFrame, closeFactorDetail, factorNum, factorPct, factorDirectionLabel,
    };
  },
};

/* ============ 任务研究记录（非正式因子库） ============ */
const ResearchRecordsView = {
  template: `
  <section>
    <div class="selector-heading">
      <div><div class="eyebrow">TRAINING-SAFE CANDIDATE LEDGER</div><h1>任务研究记录库</h1><p>保留每个已评价候选，按任务内学习分排名；不等同于正式因子、封存验证或交易批准。</p></div>
      <span class="tag amber">研究记录 ≠ 正式因子</span>
    </div>
    <div class="metric-strip" v-if="data.tasks?.length">
      <div class="metric-card" v-for="task in data.tasks" :key="task.task_name"><span>{{ task.task_name }}</span><b>{{ task.records }}</b><small>有效 {{ task.valid }} · 预筛拒绝 {{ task.pre_eval_rejected || 0 }} · 通过 {{ task.passed }} · 正式 {{ task.formal_factors }} · 最高 {{ num(task.best_learning_score) }}</small></div>
    </div>
    <div class="card" style="margin-bottom:14px">
      <div class="form-row">
        <div style="flex:2"><input v-model="q" @keyup.enter="refresh" placeholder="搜索表达式、假设或机制" /></div>
        <div><select v-model="taskName" @change="refresh"><option value="">全部研究任务</option><option v-for="task in data.tasks" :key="task.task_name" :value="task.task_name">{{ task.task_name }}</option></select></div>
        <div><select v-model="status" @change="refresh"><option value="">全部状态</option><option value="ok">有效</option><option value="error">评价失败</option><option value="rejected">回测前拒绝</option></select></div>
        <button class="btn primary" @click="refresh" :disabled="loading">{{ loading ? '加载中…' : '刷新' }}</button>
      </div>
      <div class="sub" style="margin-top:8px">{{ data.interpretation_boundary }}</div>
    </div>
    <div class="card">
      <div class="panel-title-row"><div><h3>候选排名（{{ data.total || 0 }}）</h3><span class="sub">排名仅在各 task 内可比；列表总序按学习分降序展示</span></div></div>
      <table><tr><th>Task 排名</th><th>学习分</th><th>硬门分</th><th>结果</th><th>机制</th><th>算法 / Epoch</th><th>表达式</th><th>主要失败</th><th>时间</th></tr>
        <tr v-for="row in data.records" :key="row.id" class="clickable" @click="open(row)">
          <td><b>#{{ row.task_rank }}</b><div class="sub">{{ row.task_name }}</div></td><td>{{ num(row.learning_score) }}</td><td>{{ num(row.hard_gate_score) }}</td><td><span class="tag" :class="row.discovery_passed?'green':row.status==='ok'?'amber':'red'">{{ row.discovery_passed?'训练通过':row.status==='ok'?'待改进':row.status==='rejected'?'预筛拒绝':'失败' }}</span><div v-if="row.formal_factor_admitted" class="tag blue">正式研究因子</div><div v-else-if="row.research_candidate_registered" class="tag amber">已登记研究候选</div></td><td>{{ row.mechanism_family }}</td><td>{{ row.search_audit?.algorithm || row.source }}<div class="sub">epoch {{ row.search_audit?.search_epoch || 0 }} · {{ row.search_audit?.health_state || 'legacy' }}<span v-if="row.search_audit?.novelty_retries"> · retry {{ row.search_audit.novelty_retries }}</span></div></td><td class="mono-expr" style="max-width:360px">{{ row.expression || '—' }}</td><td class="sub">{{ row.failure_reasons?.[0] || '—' }}</td><td class="sub">{{ row.created_at?.slice(5,16) }}</td>
        </tr>
      </table>
    </div>
    <div class="drawer" v-if="detail"><button class="btn close" @click="detail=null">✕ 关闭</button><h2>研究记录 #{{ detail.id }}</h2><div class="tag amber" style="margin:8px 0">{{ detail.interpretation_boundary }}</div><div class="mono-expr" style="margin:12px 0">{{ detail.expression || '无有效表达式' }}</div><p>{{ detail.hypothesis }}</p><div class="card"><h3>训练安全指标</h3><table><tr v-for="(value,key) in detail.metrics" :key="key"><td>{{ key }}</td><td>{{ num(value) }}</td></tr></table></div><div class="card" style="margin-top:12px"><h3>搜索审计</h3><table><tr v-for="(value,key) in detail.search_audit" :key="key"><td>{{ key }}</td><td>{{ Array.isArray(value) ? value.join(', ') : String(value ?? '—') }}</td></tr></table></div><div class="card" style="margin-top:12px"><h3>失败与改进目标</h3><p>{{ detail.failure_reasons?.join('；') || '无' }}</p><p class="sub">{{ detail.improvement_targets?.join('；') || '无' }}</p></div></div>
  </section>`,
  setup() {
    const data = reactive({ tasks:[], records:[], total:0, interpretation_boundary:"" });
    const q = ref(""), taskName = ref(""), status = ref(""), detail = ref(null), loading = ref(false);
    let timer = null;
    const num = (value) => value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(3);
    async function refresh() {
      loading.value = true;
      try {
        const params = new URLSearchParams({ limit:"500" });
        if (q.value.trim()) params.set("q", q.value.trim());
        if (taskName.value) params.set("task_name", taskName.value);
        if (status.value) params.set("status", status.value);
        Object.assign(data, await api(`/research-records?${params}`));
      } finally { loading.value = false; }
    }
    async function open(row) { detail.value = await api(`/research-records/${row.id}`); }
    function start() { refresh(); clearInterval(timer); timer = setInterval(refresh, 8000); }
    watch(() => appState.experimentVersion, () => { detail.value=null; if (appState.activeTab==="records") refresh(); });
    onActivated(start); onDeactivated(() => clearInterval(timer)); onUnmounted(() => clearInterval(timer));
    return { data, q, taskName, status, detail, loading, refresh, open, num };
  },
};

/* ============ Manual Factor Correlation + DSL Builder ============ */
const FactorToolsView = {
  template: `
  <section class="factor-tools-page">
    <div class="selector-heading factor-tools-heading">
      <div><div class="eyebrow">COMMON-PANEL CORRELATION / EXECUTABLE DSL COMPOSER</div><h1>因子相关性与组合表达式工具</h1><p>手工输入N个因子，在统一口径下识别重复收益来源，并生成方向、权重和标准化都已内嵌的可执行DSL。</p></div>
      <div class="factor-tools-heading-tags"><span class="tag blue">{{ market==='ashare'?'A股':'美股' }}</span><span class="tag amber">组件快照优先</span><span class="tag">DSL ≤ {{ capabilities?.limits?.expression_characters || 4000 }}</span></div>
    </div>

    <div v-if="error" class="selector-error">{{ error }}</div>
    <div v-if="message" class="warn-banner factor-tools-message">{{ message }}</div>

    <div class="factor-tools-layout">
      <div class="card factor-tools-components">
        <div class="panel-title-row"><div><h3>01 · 因子输入</h3><span class="sub">相关性最多{{ capabilities?.limits?.correlation_components || 12 }}个；表达式生成最多{{ capabilities?.limits?.expression_components || 20 }}个</span></div><div class="factor-tools-actions"><button class="btn" v-if="market==='ashare'" @click="loadAshareExample">载入本轮五因子</button><button class="btn primary" @click="addComponent">＋ 添加因子</button></div></div>
        <div class="factor-tools-bulk">
          <textarea v-model="bulkText" rows="3" placeholder="也可以每行粘贴一个DSL表达式，然后点击批量导入"></textarea>
          <button class="btn" @click="importBulk">批量导入</button>
        </div>
        <div class="factor-tools-component-head"><span>#</span><span>名称 / Key</span><span>DSL表达式</span><span>方向</span><span>权重</span><span></span></div>
        <div v-for="(row,index) in components" :key="row.uid" class="factor-tools-component-row">
          <span class="factor-tools-index">{{ index+1 }}</span>
          <div><input v-model="row.name" placeholder="因子名称"><input v-model="row.key" class="factor-key-input" placeholder="F01"></div>
          <textarea v-model="row.expression" rows="2" class="mono-input" placeholder="输入单个DSL表达式"></textarea>
          <select v-model.number="row.direction"><option :value="1">+1 高值</option><option :value="-1">-1 低值</option></select>
          <input v-model.number="row.weight" type="number" min="0.0001" step="0.05">
          <button class="btn danger" @click="removeComponent(index)" :disabled="components.length<=2">×</button>
        </div>
        <details class="factor-tools-whitelist"><summary>当前市场字段白名单（{{ capabilities?.dsl_fields?.length || 0 }}）</summary><code>{{ (capabilities?.dsl_fields || []).join(', ') }}</code></details>
      </div>

      <div class="factor-tools-workspaces">
        <div class="card factor-tools-correlation-config">
          <div class="panel-title-row"><div><h3>02 · 相关性检测</h3><span class="sub">主去重口径是统一成本后的单因子收益路径相关</span></div><button class="btn primary" @click="runCorrelation" :disabled="correlating">{{ correlating?'正在计算共同面板…':'检测相互相关性' }}</button></div>
          <div class="factor-tools-form-grid">
            <label><span>市场</span><select v-model="market" @change="changeMarket"><option value="ashare">A股</option><option value="us">美股</option></select></label>
            <label><span>组合模式</span><select v-model="correlationForm.portfolio_mode"><option value="long_only">纯多</option><option value="long_short" :disabled="market==='ashare'">多空</option></select></label>
            <label><span>开始日期</span><input type="date" v-model="correlationForm.start"></label>
            <label><span>结束日期</span><input type="date" v-model="correlationForm.end"></label>
            <label><span>股票池</span><select v-model.number="correlationForm.universe_n"><option :value="300">Top300</option><option :value="500">Top500</option><option :value="1000">Top1000</option><option :value="1500">Top1500</option></select></label>
            <label><span>周期</span><select v-model.number="correlationForm.horizon"><option :value="1">1日</option><option :value="5">5日</option><option :value="20">20日</option></select></label>
            <label><span>选股比例</span><input type="number" v-model.number="correlationForm.top_fraction" min="0.05" max="0.5" step="0.05"></label>
            <label><span>成本BPS</span><input type="number" v-model.number="correlationForm.cost_bps" min="0" max="500" step="5"></label>
            <label><span>高相关阈值</span><input type="number" v-model.number="correlationForm.threshold" min="0.5" max="0.99" step="0.05"></label>
          </div>
        </div>

        <div class="card factor-tools-expression-config">
          <div class="panel-title-row"><div><h3>03 · 多因子DSL生成</h3><span class="sub">默认按方向做截面Rank，再按权重组合</span></div><button class="btn primary" @click="buildExpression" :disabled="building">{{ building?'正在验证…':'生成可回测DSL' }}</button></div>
          <div class="factor-tools-expression-options">
            <label><span>组件标准化</span><select v-model="builderForm.normalization"><option value="rank">截面Rank（推荐）</option><option value="zscore">截面ZScore</option><option value="none">不标准化</option></select></label>
            <label class="factor-tools-check"><input type="checkbox" v-model="builderForm.omit_common_scale"><span>等权时省略共同系数以缩短表达式</span></label>
          </div>
          <template v-if="built">
            <div class="factor-tools-built-meta"><span class="tag green">验证通过</span><span>{{ built.component_count }}因子</span><span>{{ built.length }}/{{ built.max_length }}字符</span><span>历史{{ built.required_history }}日</span><span>最终方向 +1</span></div>
            <textarea class="factor-tools-output mono-input" rows="7" readonly :value="built.expression"></textarea>
            <div class="factor-tools-actions"><button class="btn" @click="copyExpression">{{ copied?'已复制':'复制DSL' }}</button><button class="btn primary" @click="sendToBacktest">送入回测模块</button></div>
            <div class="sub">{{ built.scale_note }}</div><div v-for="warning in built.warnings" :key="warning" class="bad-text">{{ warning }}</div>
          </template>
          <div v-else class="combination-empty">填写至少两个因子后生成。结构化组件仍会随结果返回，长DSL只是选股与手工回测的兼容输出。</div>
        </div>
      </div>
    </div>

    <template v-if="correlation">
      <div class="metric-strip factor-tools-metrics">
        <div class="metric-card accent"><span>共同收益期</span><b>{{ correlation.common_periods }}</b><small>{{ correlation.window.start }}~{{ correlation.window.end }}</small></div>
        <div class="metric-card"><span>高相关对</span><b>{{ correlation.high_correlation_pairs.length }}</b><small>阈值 |ρ|≥{{ correlation.threshold }}</small></div>
        <div class="metric-card"><span>面板行数</span><b>{{ Number(correlation.panel.rows||0).toLocaleString() }}</b><small>Top{{ correlation.universe_n }} · H{{ correlation.horizon }}</small></div>
        <div class="metric-card"><span>计算耗时</span><b>{{ num(correlation.elapsed_seconds,2) }}s</b><small>{{ correlation.cache_hit?'命中缓存':'完整计算' }}</small></div>
      </div>
      <div class="card factor-tools-matrix-card">
        <div class="panel-title-row"><div><h3>相关矩阵</h3><span class="sub">方向已经内嵌；负相关表示收益来源符号相反</span></div><div class="combination-mode-tabs factor-tools-matrix-tabs"><button v-for="option in matrixOptions" :key="option.key" :class="{active:matrixKey===option.key}" @click="matrixKey=option.key">{{ option.label }}</button></div></div>
        <div class="factor-tools-matrix-scroll"><table class="factor-tools-matrix"><thead><tr><th></th><th v-for="label in correlation.labels" :key="label">{{ label }}</th></tr></thead><tbody><tr v-for="(row,rowIndex) in activeMatrix" :key="correlation.labels[rowIndex]"><th>{{ correlation.labels[rowIndex] }}</th><td v-for="(value,colIndex) in row" :key="colIndex" :style="correlationCellStyle(value)">{{ num(value,3) }}</td></tr></tbody></table></div>
        <p class="sub">{{ correlation.interpretation[matrixKey] }}</p>
      </div>
      <div class="grid cols-2 factor-tools-results-grid">
        <div class="card"><div class="panel-title-row"><div><h3>高相关因子对</h3><span class="sub">优先查看收益路径同源，再查看信号与IC状态重叠</span></div></div><div v-if="!correlation.high_correlation_pairs.length" class="combination-empty">当前阈值下没有高相关因子对。</div><table v-else><tr><th>因子对</th><th>分类</th><th>信号ρ</th><th>IC路径ρ</th><th>收益ρ</th></tr><tr v-for="row in correlation.high_correlation_pairs" :key="row.left+row.right"><td><b>{{ row.left }}</b> / <b>{{ row.right }}</b></td><td><span class="tag" :class="row.classification==='same_return_source'?'red':'amber'">{{ pairClass(row.classification) }}</span></td><td>{{ num(row.signal_rank_correlation,3) }}</td><td>{{ num(row.rank_ic_path_correlation,3) }}</td><td><b>{{ num(row.portfolio_return_correlation,3) }}</b></td></tr></table></div>
        <div class="card"><div class="panel-title-row"><div><h3>单因子共同口径表现</h3><span class="sub">仅用于理解相关矩阵，不是因子晋升评级</span></div></div><table><tr><th>因子</th><th>主动/净年化</th><th>Sharpe</th><th>Rank IC</th><th>Rank ICIR</th><th>换手</th></tr><tr v-for="row in correlation.factor_stats" :key="row.key"><td><b>{{ row.key }}</b><div class="sub">{{ row.direction>0?'+1':'-1' }}</div></td><td>{{ pct(row.ann_return) }}</td><td>{{ num(row.sharpe,3) }}</td><td>{{ pct(row.rank_ic_mean) }}</td><td>{{ num(row.rank_icir,3) }}</td><td>{{ pct(row.avg_turnover) }}</td></tr></table></div>
      </div>
    </template>
  </section>`,
  setup() {
    const capabilities = ref(null), correlation = ref(null), built = ref(null);
    const market = ref("ashare"), bulkText = ref(""), error = ref(""), message = ref("");
    const correlating = ref(false), building = ref(false), copied = ref(false), matrixKey = ref("portfolio_return_correlation");
    let uid = 2;
    const blankComponent = index => ({uid:++uid,key:`F${String(index).padStart(2,"0")}`,name:`因子${index}`,expression:"",direction:1,weight:1});
    const components = ref([blankComponent(1), blankComponent(2)]);
    const correlationForm = reactive({portfolio_mode:"long_only",start:"2020-01-01",end:"2026-12-31",universe_n:500,horizon:5,top_fraction:0.20,cost_bps:15,threshold:0.80});
    const builderForm = reactive({normalization:"rank",omit_common_scale:true});
    const matrixOptions = [
      {key:"portfolio_return_correlation",label:"收益路径"},
      {key:"signal_rank_correlation",label:"截面信号"},
      {key:"rank_ic_path_correlation",label:"Rank IC路径"},
    ];
    const activeComponents = computed(() => components.value.filter(row => row.expression.trim()).map((row,index) => ({key:row.key.trim()||`F${String(index+1).padStart(2,"0")}`,name:row.name.trim()||row.key,expression:row.expression.trim(),direction:Number(row.direction),weight:Number(row.weight)||1})));
    const activeMatrix = computed(() => correlation.value?.matrices?.[matrixKey.value] || []);
    function addComponent() { components.value.push(blankComponent(components.value.length+1)); }
    function removeComponent(index) { if (components.value.length>2) components.value.splice(index,1); }
    function importBulk() {
      const rows = bulkText.value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean);
      if (!rows.length) return;
      components.value = rows.slice(0,capabilities.value?.limits?.expression_components||20).map((expression,index)=>({...blankComponent(index+1),expression}));
      while (components.value.length<2) components.value.push(blankComponent(components.value.length+1));
      bulkText.value=""; correlation.value=null; built.value=null;
    }
    function loadAshareExample() {
      market.value="ashare"; correlationForm.portfolio_mode="long_only";
      const rows=[
        ["U0001","大单参与度","zscore(winsor_mad(ts_mean((buy_lg_amount+sell_lg_amount)/amount,120),5))",1],
        ["U0003","量比持续性","rank(-(ts_mean(volume_ratio,120)/ts_max(volume_ratio,120)))",-1],
        ["U0012","低市销率TTM","rank(-ps_ttm)",1],
        ["U0015","日内价量反转","-rank(ts_sum((close-open)/close*vol,120))",1],
        ["U0021","中期反转","zscore(winsor_mad(ts_mean(close,20)/ts_mean(close,60),5))",-1],
      ];
      components.value=rows.map(([key,name,expression,direction])=>({uid:++uid,key,name,expression,direction,weight:1}));
      correlation.value=null; built.value=null; loadCapabilities();
    }
    async function loadCapabilities() {
      try {
        const q=new URLSearchParams({market:market.value});
        if(appState.experimentId) q.set("experiment_id",appState.experimentId);
        capabilities.value=await api(`/factor-tools/capabilities?${q}`,{cacheTtl:1000});
      } catch(e){error.value=e.message;}
    }
    function changeMarket(){correlationForm.portfolio_mode=market.value==="ashare"?"long_only":"long_short";correlation.value=null;built.value=null;loadCapabilities();}
    function ensureComponents(limit) { if(activeComponents.value.length<2) throw new Error("请至少输入2个有效因子表达式"); if(activeComponents.value.length>limit) throw new Error(`当前操作最多支持${limit}个因子`); }
    async function runCorrelation(){correlating.value=true;error.value="";message.value="";try{ensureComponents(capabilities.value?.limits?.correlation_components||12);correlation.value=await api("/factor-tools/correlation",{method:"POST",body:{experiment_id:appState.experimentId,market:market.value,components:activeComponents.value,...correlationForm}});}catch(e){error.value=e.message;}finally{correlating.value=false;}}
    async function buildExpression(){building.value=true;error.value="";message.value="";try{ensureComponents(capabilities.value?.limits?.expression_components||20);built.value=await api("/factor-tools/build-expression",{method:"POST",body:{market:market.value,components:activeComponents.value,...builderForm}});}catch(e){error.value=e.message;}finally{building.value=false;}}
    async function copyExpression(){if(!built.value)return;try{await navigator.clipboard.writeText(built.value.expression);copied.value=true;setTimeout(()=>copied.value=false,1400);}catch(e){error.value="复制失败，请手工选择表达式";}}
    function sendToBacktest(){if(!built.value)return;appState.backtestDraft=built.value.expression;appState.requestedTab="backtest";message.value="表达式已送入回测模块，方向固定为 +1";}
    function correlationCellStyle(value){const number=Number(value||0),alpha=Math.min(.72,.08+.55*Math.abs(number));return {background:number>=0?`rgba(46,160,100,${alpha})`:`rgba(56,118,200,${alpha})`,color:Math.abs(number)>.65?"#fff":"var(--text)",fontWeight:Math.abs(number)>=(correlation.value?.threshold||.8)?"800":"500"};}
    const num=(value,digits=3)=>value==null||Number.isNaN(Number(value))?"—":Number(value).toFixed(digits);
    const pct=value=>value==null||Number.isNaN(Number(value))?"—":`${(Number(value)*100).toFixed(2)}%`;
    const pairClass=value=>({same_return_source:"收益同源",signal_overlap:"信号重叠",ic_regime_overlap:"IC状态重叠"})[value]||value;
    watch(()=>appState.experimentVersion,()=>{correlation.value=null;built.value=null;loadCapabilities();});
    onActivated(loadCapabilities);
    return {appState,capabilities,components,activeComponents,market,bulkText,error,message,correlation,built,correlating,building,copied,correlationForm,builderForm,matrixKey,matrixOptions,activeMatrix,addComponent,removeComponent,importBulk,loadAshareExample,changeMarket,runCorrelation,buildExpression,copyExpression,sendToBacktest,correlationCellStyle,num,pct,pairClass};
  },
};

/* ============ Factor Combination Laboratory ============ */
const CombinationLabView = {
  template: `
  <section class="combination-page">
    <div class="combination-heading">
      <div>
        <div class="eyebrow">NESTED COMPONENT ARRAY / PURGED VALIDATION / EVENT REPLAY</div>
        <h1>因子组合优化实验台</h1>
        <p>组件数组是权威数据，不再拼接超长DSL；程序化与LLM模式共享同一冻结协议和确定性裁判。</p>
      </div>
      <div class="combination-mode-tabs">
        <button :class="{active:form.search_mode==='programmatic'}" @click="form.search_mode='programmatic'">程序化优化</button>
        <button :class="{active:form.search_mode==='llm'}" @click="form.search_mode='llm'">LLM协作优化</button>
      </div>
    </div>

    <div v-if="form.search_mode==='llm' && capabilities && !capabilities.llm?.configured" class="warn-banner">
      LLM协作模式需要先在“设置”中配置 inner_provider 或 outer_provider；不会静默回退为程序化结果。
    </div>
    <div v-if="error" class="selector-error">{{ error }}</div>

    <div class="combination-layout">
      <div class="combination-builder">
        <div class="card combination-section">
          <div class="panel-title-row">
            <div><h2>1. 冻结候选因子</h2><span class="sub">从任务因子库选择，或手工加入DSL；最多12个。</span></div>
            <span class="count-badge">{{ components.length }} / 12</span>
          </div>
          <div class="combination-add-row">
            <input v-model="factorQuery" placeholder="搜索因子库名称或表达式">
            <select v-model="selectedLibraryId">
              <option value="">从因子库选择…</option>
              <option v-for="factor in filteredLibrary" :key="factor.id" :value="factor.id">#{{ factor.id }} {{ factor.name || factor.expression.slice(0,42) }}</option>
            </select>
            <button class="btn" @click="addLibraryFactor" :disabled="!selectedLibraryId || components.length>=12">加入</button>
          </div>
          <div class="combination-add-row manual">
            <input v-model="manualExpression" class="mono-input" placeholder="输入单个DSL表达式（每个组件独立保存）">
            <select v-model.number="manualDirection"><option :value="1">正向 +1</option><option :value="-1">反向 -1</option></select>
            <button class="btn" @click="addManualFactor" :disabled="!manualExpression.trim() || addingManual || components.length>=12">{{ addingManual?'校验中':'校验并加入' }}</button>
          </div>
          <div v-if="!components.length" class="combination-empty">请至少加入2个因子。组合不会被压成一条500字符DSL。</div>
          <div v-for="(component,index) in components" :key="component.key" class="combination-component-row">
            <div class="combination-index">{{ index+1 }}</div>
            <div class="combination-component-copy">
              <b>{{ component.name }}</b>
              <code>{{ component.expression }}</code>
              <small>{{ component.source }}<template v-if="component.source_ref"> · {{ component.source_ref }}</template></small>
            </div>
            <select v-model.number="component.direction"><option :value="1">+1 正向</option><option :value="-1">-1 反向</option></select>
            <input v-model="component.mechanism" title="收益机制">
            <button class="text-btn bad-text" @click="removeComponent(index)">移除</button>
          </div>
        </div>

        <div class="card combination-section">
          <div class="panel-title-row"><div><h2>2. 组合空间与执行约束</h2><span class="sub">先去重与相关性过滤，再进行预算有界的粗网格搜索。</span></div><span class="tag blue">路径预算 {{ form.path_budget }}</span></div>
          <div class="combination-form-grid">
            <label><span>最小组合数</span><input v-model.number="form.min_factors" type="number" min="2" :max="Math.max(2,components.length)"></label>
            <label><span>最大组合数</span><input v-model.number="form.max_factors" type="number" min="2" :max="Math.max(2,components.length)"></label>
            <label><span>最少收益机制</span><input v-model.number="form.min_mechanisms" type="number" min="1" :max="Math.max(1,mechanismCount)"></label>
            <label><span>权重粗步长</span><select v-model.number="form.coarse_step"><option :value="0.05">5%</option><option :value="0.10">10%</option><option :value="0.20">20%</option><option :value="0.25">25%</option></select></label>
            <label><span>最小有效权重</span><input v-model.number="form.min_weight" type="number" min="0.01" max="0.5" step="0.01"></label>
            <label><span>单因子权重上限</span><input v-model.number="form.max_weight" type="number" min="0.1" max="1" step="0.05"></label>
            <label><span>机制权重上限</span><input v-model.number="form.max_mechanism_weight" type="number" min="0.2" max="1" step="0.05"></label>
            <label><span>收益路径相关上限</span><input v-model.number="form.max_pair_correlation" type="number" min="0" max="1" step="0.05"></label>
            <label><span>搜索路径预算</span><input v-model.number="form.path_budget" type="number" min="10" max="20000" step="100"></label>
            <label><span>进入验证的路径</span><input v-model.number="form.validation_budget" type="number" min="50" max="500" step="25"></label>
            <label><span>股票池</span><input v-model.number="form.universe_n" type="number" min="100" max="5000" step="100"></label>
            <label><span>调仓/预测周期</span><select v-model.number="form.horizon"><option :value="1">1日</option><option :value="5">5日</option><option :value="20">20日</option></select></label>
            <label><span>选股比例</span><input v-model.number="form.top_fraction" type="number" min="0.05" max="0.5" step="0.05"></label>
            <label><span>基础成本 BPS</span><input v-model.number="form.cost_bps" type="number" min="0" max="200" step="5"></label>
            <label><span>压力成本 BPS</span><input v-model.number="form.stress_cost_bps" type="number" min="0" max="500" step="5"></label>
            <label v-if="capabilities?.market==='us'"><span>组合版本</span><select v-model="form.portfolio_mode"><option value="long_only">美股纯多</option><option value="long_short">美股多空</option></select></label>
          </div>
        </div>

        <div class="card combination-section">
          <div class="panel-title-row"><div><h2>3. 不重叠时间协议</h2><span class="sub">训练决定搜索，验证选择候选，评级只在胜者通过硬门槛后读取。</span></div><span class="tag green">边界自动Purge</span></div>
          <div class="combination-window-grid">
            <strong>训练</strong><input v-model="form.train_start" type="date"><span>至</span><input v-model="form.train_end" type="date">
            <strong>验证</strong><input v-model="form.validation_start" type="date"><span>至</span><input v-model="form.validation_end" type="date">
            <strong>冻结评级</strong><input v-model="form.rating_start" type="date"><span>至</span><input v-model="form.rating_end" type="date">
          </div>
        </div>
      </div>

      <aside class="combination-sidebar">
        <div class="card combination-launch-card">
          <div class="panel-title-row"><div><h2>协议检查</h2><span class="sub">创建后候选、方向、窗口和门槛全部冻结。</span></div><span class="tag" :class="canStart?'green':'amber'">{{ canStart?'可启动':'待补全' }}</span></div>
          <div class="combination-checks">
            <div><span :class="components.length>=2?'ok':'no'">●</span> 组件数量 2–12</div>
            <div><span :class="form.min_factors<=form.max_factors && form.max_factors<=components.length?'ok':'no'">●</span> 组合数量可行</div>
            <div><span :class="mechanismCount>=form.min_mechanisms?'ok':'no'">●</span> 收益机制覆盖</div>
            <div><span :class="validWindows?'ok':'no'">●</span> 训练/验证/评级不重叠</div>
            <div><span :class="form.search_mode!=='llm'||capabilities?.llm?.configured?'ok':'no'">●</span> LLM供应商</div>
          </div>
          <button class="btn primary combination-start" @click="startExperiment" :disabled="!canStart || creating">{{ creating?'正在冻结协议…':'冻结协议并开始优化' }}</button>
          <small>未通过门槛会诚实输出 NO_COMBINATION，不强制产生冠军。</small>
        </div>

        <div v-if="current" class="card combination-live-card">
          <div class="panel-title-row"><div><h2>#{{ current.id }} {{ current.name }}</h2><span class="sub">{{ current.protocol }}</span></div><span class="tag" :class="statusClass(current.status)">{{ current.status }}</span></div>
          <div class="combination-progress"><i :style="{width:progressPercent+'%'}"></i></div>
          <div class="combination-progress-copy"><span>{{ current.progress?.message || current.progress?.stage || '等待状态' }}</span><b>{{ progressPercent.toFixed(0) }}%</b></div>
          <button v-if="['queued','running'].includes(current.status)" class="btn danger" @click="stopExperiment">安全停止</button>
          <div v-if="current.error" class="selector-error" style="margin-top:10px">{{ current.error }}</div>
        </div>

        <div class="card combination-history-card">
          <div class="panel-title-row"><div><h2>实验历史</h2><span class="sub">当前研究任务隔离</span></div><button class="text-btn" @click="loadHistory">刷新</button></div>
          <div v-if="!history.length" class="combination-empty">暂无组合实验</div>
          <div v-for="row in history" :key="row.id" class="combination-history-row" :class="{active:current?.id===row.id}" @click="openExperiment(row.id)">
            <div><b>#{{ row.id }} {{ row.name }}</b><small>{{ row.search_mode }} · {{ row.portfolio_mode }} · {{ row.component_count }}因子</small></div>
            <div><span class="tag" :class="statusClass(row.status)">{{ row.decision || row.status }}</span><small>{{ formatTime(row.created_at) }}</small></div>
          </div>
        </div>
      </aside>
    </div>

    <div v-if="current?.result?.protocol" class="combination-results">
      <div class="combination-result-heading">
        <div><div class="eyebrow">AUDITABLE RESULT</div><h2>组合实验结果</h2><p>结果哈希 {{ current.result.result_hash }} · {{ current.result.elapsed_seconds }} 秒</p></div>
        <span class="combination-decision" :class="decisionClass(current.result.decision)">{{ current.result.decision }}</span>
      </div>
      <div class="metric-strip combination-metrics">
        <div class="metric-card"><span>生成路径</span><b>{{ current.result.search?.generated_candidates || 0 }}</b><small>相关过滤后 {{ current.result.search?.correlation_filtered_candidates || 0 }}</small></div>
        <div class="metric-card"><span>进入验证</span><b>{{ current.result.search?.validated_candidates || 0 }}</b><small>Top train only</small></div>
        <div class="metric-card"><span>LLM有效提案</span><b>{{ current.result.llm_proposals_accepted || 0 }}</b><small>收到 {{ current.result.llm_proposals_received || 0 }}</small></div>
        <div class="metric-card"><span>面板</span><b>{{ current.result.panel?.sessions || current.result.panel?.summary?.rows || '—' }}</b><small>{{ current.result.panel?.symbols || '—' }} symbols</small></div>
      </div>

      <div v-if="!winner && bestCandidate" class="warn-banner combination-candidate-warning">
        没有候选通过全部验证门槛。下面仍展示验证排序第一的“最佳被拒候选”及完整权重，供诊断使用；它不是获批组合，冻结评级区间没有被读取。
      </div>

      <div v-if="bestCandidate" class="grid cols-2 combination-result-grid">
        <div class="card">
          <div class="panel-title-row"><div><h2>{{ winner?'胜者组件权重':'当前最佳候选权重（未通过）' }}</h2><span class="sub">组件数组为权威记录；零权重因子已隐藏</span></div><span class="tag" :class="winner?'blue':'red'">{{ bestCandidate.source }}</span></div>
          <table><tr><th>组件</th><th>机制</th><th>方向</th><th>权重</th></tr>
            <tr v-for="row in bestCandidateWeights" :key="row.key"><td><b>{{ row.name }}</b><small class="mono-expr">{{ row.expression }}</small></td><td>{{ row.mechanism }}</td><td>{{ row.direction>0?'+1':'-1' }}</td><td><b :class="winner?'green-text':'amber-text'">{{ pct(row.weight) }}</b></td></tr>
          </table>
        </div>
        <div class="card">
          <div class="panel-title-row"><div><h2>训练与验证证据</h2><span class="sub">硬门槛先于综合分</span></div><span class="tag" :class="bestCandidate.all_rules_pass?'green':'red'">{{ bestCandidate.all_rules_pass?'全部通过':'存在失败' }}</span></div>
          <div class="metric-strip combination-evidence">
            <div class="metric-card"><span>验证净/主动年化</span><b>{{ pct(bestCandidate.validation?.ann_return) }}</b><small>训练 {{ pct(bestCandidate.training?.ann_return) }}</small></div>
            <div class="metric-card"><span>验证Sharpe</span><b>{{ num(bestCandidate.validation?.sharpe) }}</b><small>最差年 {{ num(bestCandidate.validation?.worst_year_sharpe) }}</small></div>
            <div class="metric-card"><span>Rank IC</span><b>{{ pct(bestCandidate.validation?.rank_ic_mean) }}</b><small>ICIR {{ num(bestCandidate.validation?.rank_icir) }}</small></div>
            <div class="metric-card"><span>FDR q</span><b>{{ num(bestCandidate.validation?.rank_ic_fdr_q) }}</b><small>泛化差 {{ num(bestCandidate.generalization_gap) }}</small></div>
          </div>
          <div class="combination-rule-list"><div v-for="(passed,key) in bestCandidate.rules" :key="key"><span :class="passed?'ok':'no'">{{ passed?'✓':'✕' }}</span>{{ ruleLabel(key) }}</div></div>
        </div>
      </div>

      <div v-if="current.result.rating" class="grid cols-2 combination-result-grid">
        <div class="card">
          <div class="panel-title-row"><div><h2>冻结评级：胜者</h2><span class="sub">向量筛选 + StepEvent逐事件复测</span></div><span class="tag green">账本 {{ current.result.rating.event?.integrity?.all_pass?'PASS':'FAIL' }}</span></div>
          <div class="metric-strip combination-evidence">
            <div class="metric-card"><span>向量主动/净年化</span><b>{{ pct(ratingMetric('ann')) }}</b></div>
            <div class="metric-card"><span>向量Sharpe</span><b>{{ num(ratingMetric('sharpe')) }}</b></div>
            <div class="metric-card"><span>步进总年化</span><b>{{ pct(current.result.rating.event?.ann_return) }}</b></div>
            <div class="metric-card"><span>步进最大回撤</span><b>{{ pct(current.result.rating.event?.max_drawdown) }}</b></div>
          </div>
        </div>
        <div class="card">
          <div class="panel-title-row"><div><h2>等权全组件基准</h2><span class="sub">不参与搜索，只做诚实对照</span></div></div>
          <div class="metric-strip combination-evidence">
            <div class="metric-card"><span>步进总年化</span><b>{{ pct(current.result.equal_all_components_benchmark?.event?.ann_return) }}</b></div>
            <div class="metric-card"><span>步进Sharpe</span><b>{{ num(current.result.equal_all_components_benchmark?.event?.sharpe) }}</b></div>
            <div class="metric-card"><span>最大回撤</span><b>{{ pct(current.result.equal_all_components_benchmark?.event?.max_drawdown) }}</b></div>
            <div class="metric-card"><span>执行成本</span><b>{{ money(current.result.equal_all_components_benchmark?.event?.total_execution_cost) }}</b></div>
          </div>
        </div>
      </div>

      <div class="card combination-finalists-card">
        <div class="panel-title-row"><div><h2>验证候选榜</h2><span class="sub">按硬门槛、验证分、最差年度和Rank ICIR排序</span></div></div>
        <div class="combination-table-wrap"><table><tr><th>#</th><th>来源</th><th>组合与权重</th><th>因子数</th><th>机制数</th><th>验证年化</th><th>Sharpe</th><th>最差年</th><th>Rank IC</th><th>ICIR</th><th>FDR q</th><th>泛化差</th><th>状态</th></tr>
          <tr v-for="(row,index) in (current.result.finalists||[])" :key="index"><td>{{ index+1 }}</td><td>{{ row.source }}</td><td><span class="combination-weight-summary">{{ finalistWeightText(row) }}</span></td><td>{{ row.validation?.active_factors }}</td><td>{{ row.validation?.active_mechanisms }}</td><td>{{ pct(row.validation?.ann_return) }}</td><td>{{ num(row.validation?.sharpe) }}</td><td>{{ num(row.validation?.worst_year_sharpe) }}</td><td>{{ pct(row.validation?.rank_ic_mean) }}</td><td>{{ num(row.validation?.rank_icir) }}</td><td>{{ num(row.validation?.rank_ic_fdr_q) }}</td><td>{{ num(row.generalization_gap) }}</td><td><span class="tag" :class="row.all_rules_pass?'green':'red'">{{ row.all_rules_pass?'PASS':'REJECT' }}</span></td></tr>
        </table></div>
      </div>
    </div>
  </section>`,
  setup() {
    const capabilities = ref(null);
    const library = ref([]);
    const components = ref([]);
    const history = ref([]);
    const current = ref(null);
    const factorQuery = ref("");
    const selectedLibraryId = ref("");
    const manualExpression = ref("");
    const manualDirection = ref(1);
    const addingManual = ref(false);
    const creating = ref(false);
    const error = ref("");
    const form = reactive({
      search_mode: "programmatic", portfolio_mode: "long_short",
      min_factors: 2, max_factors: 5, min_mechanisms: 2,
      coarse_step: 0.10, min_weight: 0.05, max_weight: 0.65,
      max_mechanism_weight: 0.70, max_pair_correlation: 0.85,
      path_budget: 3000, validation_budget: 250, universe_n: 500,
      top_fraction: 0.20, horizon: 5, cost_bps: 15, stress_cost_bps: 50,
      train_start: "2010-01-01", train_end: "2018-12-31",
      validation_start: "2019-01-01", validation_end: "2022-12-31",
      rating_start: "2023-01-01", rating_end: "2026-12-31",
    });
    let timer = null;

    const filteredLibrary = computed(() => {
      const q = factorQuery.value.trim().toLowerCase();
      return library.value.filter(row => !q || `${row.name||''} ${row.expression||''}`.toLowerCase().includes(q)).slice(0,100);
    });
    const mechanismCount = computed(() => new Set(components.value.map(row => row.mechanism || "unknown")).size);
    const validWindows = computed(() => form.train_start <= form.train_end && form.train_end < form.validation_start && form.validation_start <= form.validation_end && form.validation_end < form.rating_start && form.rating_start <= form.rating_end);
    const canStart = computed(() => components.value.length >= 2 && components.value.length <= 12 && form.min_factors >= 2 && form.min_factors <= form.max_factors && form.max_factors <= components.value.length && mechanismCount.value >= form.min_mechanisms && validWindows.value && (form.search_mode !== "llm" || capabilities.value?.llm?.configured));
    const progressPercent = computed(() => {
      const p = current.value?.progress || {};
      if (current.value?.status === "done") return 100;
      return p.total ? Math.max(2, Math.min(99, 100 * Number(p.completed || 0) / Number(p.total))) : 2;
    });
    const winner = computed(() => current.value?.result?.winner || null);
    const bestCandidate = computed(() => winner.value || current.value?.result?.finalists?.[0] || null);
    const bestCandidateWeights = computed(() => {
      const weights = bestCandidate.value?.weights || [];
      const snapshot = current.value?.request_spec?.components || [];
      return snapshot.map((row,index) => ({...row, weight:Number(weights[index]||0)})).filter(row => row.weight > 1e-10);
    });

    function finalistWeightText(candidate) {
      const snapshot = current.value?.request_spec?.components || [];
      const weights = candidate?.weights || [];
      return snapshot
        .map((row,index) => ({ name:row.name || row.key, weight:Number(weights[index] || 0) }))
        .filter(row => row.weight > 1e-10)
        .map(row => `${row.name} ${pct(row.weight)}`)
        .join(" · ") || "—";
    }
    function ruleLabel(key) {
      return ({
        rank_ic_direction:"Rank IC方向一致",
        rank_ic_fdr:"Rank IC多重检验",
        worst_year_sharpe:"最差年度Sharpe",
        positive_year_rate:"盈利年度比例",
        stress_return:"压力成本收益",
        minimum_mechanisms:"最少收益机制",
      })[key] || key;
    }

    async function loadCapabilities() {
      const q = appState.experimentId ? `?experiment_id=${appState.experimentId}` : "";
      capabilities.value = await api(`/combination-experiments/capabilities${q}`, {cacheTtl:1000});
      form.portfolio_mode = capabilities.value.portfolio_mode;
    }
    async function loadLibrary() {
      const suffix = appState.experimentId ? `&experiment_id=${appState.experimentId}` : "";
      const data = await api(`/factors?sort=grade${suffix}`, {cacheTtl:1000});
      library.value = (data.factors || []).filter(row => row.expression);
    }
    async function loadHistory() {
      const q = appState.experimentId ? `?experiment_id=${appState.experimentId}&limit=50` : "?limit=50";
      const data = await api(`/combination-experiments${q}`, {cacheTtl:600});
      history.value = data.experiments || [];
      if (!current.value && history.value.length) await openExperiment(history.value[0].id);
    }
    function factorMechanism(row) {
      return row.research_meta?.mechanism_family || row.fingerprint?.mechanism_family || row.public?.mechanism_family || "unknown";
    }
    function addLibraryFactor() {
      const row = library.value.find(item => Number(item.id) === Number(selectedLibraryId.value));
      if (!row || components.value.some(item => item.expression === row.expression)) return;
      components.value.push({
        key:`F${row.id}`, name:row.name || `Factor ${row.id}`, expression:row.expression,
        direction:Number(row.research_meta?.direction || 1), mechanism:factorMechanism(row),
        source:"factor_library", source_ref:`factor_id=${row.id}`,
      });
      selectedLibraryId.value = "";
      form.max_factors = Math.min(5, components.value.length);
      form.min_mechanisms = Math.min(2, mechanismCount.value);
    }
    async function addManualFactor() {
      addingManual.value = true; error.value = "";
      try {
        const inspected = await api("/dsl/inspect", {method:"POST", body:{expression:manualExpression.value.trim(), experiment_id:appState.experimentId}});
        if (components.value.some(item => item.expression === manualExpression.value.trim())) throw new Error("该表达式已经在候选池中");
        const sequence = components.value.filter(row => row.source === "manual").length + 1;
        components.value.push({
          key:`M${String(sequence).padStart(2,"0")}`, name:`手工因子 ${sequence}`,
          expression:manualExpression.value.trim(), direction:Number(manualDirection.value),
          mechanism:inspected.mechanism_family || "unknown", source:"manual", source_ref:"",
        });
        manualExpression.value = "";
        form.max_factors = Math.min(5, components.value.length);
        form.min_mechanisms = Math.min(2, mechanismCount.value);
      } catch (e) { error.value = e.message; }
      finally { addingManual.value = false; }
    }
    function removeComponent(index) {
      components.value.splice(index,1);
      form.max_factors = Math.min(form.max_factors, components.value.length);
      form.min_factors = Math.min(form.min_factors, Math.max(2,components.value.length));
      form.min_mechanisms = Math.min(form.min_mechanisms, Math.max(1,mechanismCount.value));
    }
    async function startExperiment() {
      creating.value = true; error.value = "";
      try {
        const body = {
          ...JSON.parse(JSON.stringify(form)),
          name:`${form.search_mode==='llm'?'LLM协作':'程序化'}组合 ${new Date().toLocaleString("zh-CN",{hour12:false})}`,
          experiment_id:appState.experimentId,
          market:capabilities.value?.market,
          panel_glob:capabilities.value?.panel_glob,
          components:components.value,
          start:true,
        };
        current.value = await api("/combination-experiments", {method:"POST", body});
        await loadHistory();
      } catch (e) { error.value = e.message; }
      finally { creating.value = false; }
    }
    async function openExperiment(id) {
      try { current.value = await api(`/combination-experiments/${id}`, {cacheTtl:0}); }
      catch (e) { error.value = e.message; }
    }
    async function stopExperiment() {
      if (!current.value) return;
      await api(`/combination-experiments/${current.value.id}/stop`, {method:"POST"});
      await openExperiment(current.value.id);
    }
    async function refresh() {
      if (current.value?.id && ["queued","running"].includes(current.value.status)) await openExperiment(current.value.id);
      if (!current.value?.id || current.value.status === "done" || current.value.status === "error" || current.value.status === "stopped") await loadHistory();
    }
    function pct(value) { return value == null ? "—" : `${(Number(value)*100).toFixed(2)}%`; }
    function num(value) { return value == null ? "—" : Number(value).toFixed(3); }
    function money(value) { return value == null ? "—" : Number(value).toLocaleString("zh-CN",{maximumFractionDigits:0}); }
    function formatTime(value) { return value ? new Date(value).toLocaleString("zh-CN",{hour12:false}) : "—"; }
    function statusClass(value) { return value === "done" ? "green" : value === "error" || value === "stopped" ? "red" : "amber"; }
    function decisionClass(value) { return value === "PASS" ? "pass" : value === "NO_COMBINATION" ? "reject" : "research"; }
    function scenarioAt(replay) {
      const scenarios = replay?.vector || {};
      return scenarios[String(Number(current.value?.request_spec?.cost_bps || form.cost_bps))] || scenarios[String(parseInt(current.value?.request_spec?.cost_bps || form.cost_bps))] || Object.values(scenarios)[0] || {};
    }
    function ratingMetric(key) {
      const scenario = scenarioAt(current.value?.result?.rating);
      const longOnly = current.value?.request_spec?.portfolio_mode === "long_only";
      if (key === "ann") return longOnly ? scenario.active_ann_return : scenario.ann_return;
      if (key === "sharpe") return longOnly ? scenario.active_sharpe : scenario.sharpe;
      return null;
    }
    async function initialise() {
      error.value = "";
      try { await Promise.all([loadCapabilities(), loadLibrary()]); await loadHistory(); }
      catch (e) { error.value = e.message; }
    }
    watch(() => appState.experimentVersion, () => {
      current.value=null; history.value=[]; components.value=[]; initialise();
    });
    onActivated(() => { initialise(); clearInterval(timer); timer=setInterval(refresh,3000); });
    onDeactivated(() => clearInterval(timer));
    onUnmounted(() => clearInterval(timer));
    return {
      appState, capabilities, library, components, history, current, factorQuery,
      selectedLibraryId, manualExpression, manualDirection, addingManual, creating,
      error, form, filteredLibrary, mechanismCount, validWindows, canStart,
      progressPercent, winner, bestCandidate, bestCandidateWeights, finalistWeightText, ruleLabel,
      addLibraryFactor, addManualFactor,
      removeComponent, startExperiment, stopExperiment, openExperiment, loadHistory,
      pct, num, money, formatTime, statusClass, decisionClass, ratingMetric,
    };
  },
};

/* ============ Important Research Documents ============ */
const ResearchDocumentsView = {
  template: `
  <section class="documents-page">
    <div class="documents-heading">
      <div><div class="eyebrow">VERSIONED RESEARCH CONCLUSIONS / READ-ONLY ARCHIVE</div><h1>重要研究文档</h1><p>保存经过整理的关键研究结论。HTML报告与原始实验工件分离、按版本登记，不覆盖历史结论。</p></div>
      <div class="documents-heading-actions"><span class="tag blue">{{ documents.length }} 份文档</span><button class="btn" @click="loadDocuments" :disabled="loading">{{ loading?'刷新中…':'刷新目录' }}</button></div>
    </div>
    <div v-if="error" class="selector-error">{{ error }}</div>
    <div class="documents-layout">
      <aside class="documents-sidebar card">
        <div class="documents-search"><input v-model="query" placeholder="搜索标题、摘要或标签"></div>
        <div v-if="!filteredDocuments.length && !loading" class="combination-empty">当前目录没有匹配文档。</div>
        <button v-for="doc in filteredDocuments" :key="doc.slug" class="document-list-item" :class="{active:current?.slug===doc.slug}" @click="openDocument(doc)">
          <span class="document-list-meta"><i>{{ doc.category }}</i><time>{{ doc.updated_at }}</time></span>
          <b>{{ doc.title }}</b>
          <small>{{ doc.summary }}</small>
          <span class="document-tags"><em v-for="tag in doc.tags" :key="tag">{{ tag }}</em></span>
        </button>
      </aside>
      <main class="document-reader card">
        <template v-if="current">
          <div class="document-reader-head">
            <div><div class="eyebrow">{{ current.category }} · {{ current.status }}</div><h2>{{ current.title }}</h2><p>{{ current.subtitle }}</p></div>
            <div><a class="btn" :href="current.html_url" target="_blank" rel="noopener">新窗口阅读</a></div>
          </div>
          <div class="document-integrity"><span>更新 {{ current.updated_at }}</span><span>{{ fileSize(current.size_bytes) }}</span><span>SHA256 {{ String(current.sha256||'').slice(0,16) }}</span></div>
          <iframe class="document-frame" :key="current.sha256" :src="current.html_url" :title="current.title"></iframe>
        </template>
        <div v-else class="document-reader-empty"><h2>选择一份研究文档</h2><p>目录中的HTML报告将在这里以只读方式打开。</p></div>
      </main>
    </div>
  </section>`,
  setup() {
    const documents = ref([]);
    const current = ref(null);
    const query = ref("");
    const loading = ref(false);
    const error = ref("");
    const filteredDocuments = computed(() => {
      const needle = query.value.trim().toLowerCase();
      if (!needle) return documents.value;
      return documents.value.filter(doc => `${doc.title} ${doc.subtitle} ${doc.summary} ${(doc.tags||[]).join(" ")}`.toLowerCase().includes(needle));
    });
    async function loadDocuments() {
      loading.value = true; error.value = "";
      try {
        const data = await api("/research-documents", {cacheTtl:0});
        documents.value = (data.documents || []).filter(doc => doc.available);
        if (!current.value || !documents.value.some(doc => doc.slug === current.value.slug)) current.value = documents.value[0] || null;
        else current.value = documents.value.find(doc => doc.slug === current.value.slug) || null;
      } catch (e) { error.value = e.message; }
      finally { loading.value = false; }
    }
    function openDocument(doc) { current.value = doc; }
    function fileSize(value) {
      const bytes = Number(value || 0);
      if (!bytes) return "—";
      return bytes < 1024 * 1024 ? `${(bytes/1024).toFixed(1)} KB` : `${(bytes/1024/1024).toFixed(1)} MB`;
    }
    onActivated(loadDocuments);
    return {documents,current,query,loading,error,filteredDocuments,loadDocuments,openDocument,fileSize};
  },
};

/* ============ Qlib Native / Alpha158 ============ */
const QlibResearchView = {
  template: `
  <section>
    <div class="documents-heading">
      <div><div class="eyebrow">QLIB-NATIVE RESEARCH CONTRACT</div><h1>Qlib · Alpha158</h1><p>完整映射 158 个特征；Qlib 扩展研究能力，FactorFactory 的成本、双向冻结、HOLDOUT、Vault 与 2020—最新评级仍是最终裁判。</p></div>
      <div class="documents-heading-actions"><span class="tag blue">{{ catalog.feature_count || 0 }} FEATURES</span><button class="btn" @click="loadAll" :disabled="loading">{{ loading?'刷新中…':'刷新' }}</button></div>
    </div>
    <div v-if="error" class="selector-error">{{ error }}</div>
    <div class="grid cols-4" style="margin-bottom:14px">
      <div class="card"><h3>上游版本</h3><div class="big-num" style="font-size:18px">{{ String(cap.upstream?.commit||'—').slice(0,10) }}</div><div class="sub">Microsoft Qlib · MIT · 固定提交</div></div>
      <div class="card"><h3>当前进度</h3><div class="big-num" style="font-size:18px">{{ progress.state || 'not_started' }}</div><div class="sub">{{ progressLabel }}</div></div>
      <div class="card"><h3>A股报告</h3><div class="big-num">{{ ashareReports.length }}</div><div class="sub">纯多 · 20bps · Top500 · H5</div></div>
      <div class="card"><h3>美股报告</h3><div class="big-num">{{ usReports.length }}</div><div class="sub">纯多与多空 · 15bps · Top500 · H5</div></div>
    </div>
    <div class="card" style="margin-bottom:14px">
      <div class="panel-title-row"><div><h3>吸收边界</h3><span class="sub">不是替换回测器，也不会把 Qlib 官方榜单数字当成本地结果</span></div><span class="tag amber">NON_PIT_RESEARCH</span></div>
      <div class="grid cols-2"><div><b>已吸收</b><p class="sub">DataHandler / Dataset 分段、Processor、Alpha158 Loader、LightGBM 模型接口、Signal/SigAna/Portfolio Recorder、不可变实验清单。</p></div><div><b>本系统保留</b><p class="sub">t日收盘信号→t+1开盘成交、真实换手与市场成本、HAC/LCB、多重检验、双向训练冻结、HOLDOUT/Vault、冻结评级。</p></div></div>
      <p class="sub" style="margin-top:10px">VWAP：A股由成交额/成交量并按OHLC复权因子构造；美股 amount 多为 close×volume 代理，因此 VWAP0 标为低保真。</p>
    </div>
    <div class="card" style="margin-bottom:14px">
      <div class="panel-title-row"><div><h3>研究任务联合模型</h3><span class="sub">Alpha158可执行矩阵（默认剔除低保真VWAP0） · 训练期Processor · 严格过去数据OOF · Residual→DSL</span></div><span class="tag" :class="jointStatus.search_eligible?'green':jointStatus.state==='not_started'?'blue':'amber'">{{ jointStatus.search_eligible?'可进入DSL重测':(jointStatus.state||'未运行') }}</span></div>
      <div class="form-row" style="margin:10px 0">
        <div><label>Qlib任务</label><select v-model.number="selectedJointExpId" @change="pickJointExperiment"><option v-for="e in jointExperiments" :key="e.id" :value="e.id">#{{e.id}} {{e.name}} · {{e.research_config?.market}}</option></select></div>
        <div><label>研究子任务</label><select v-model="selectedJointTask" @change="loadJointStatus"><option v-for="t in jointTasks" :key="t.name" :value="t.name">{{t.name}} · Top{{t.universe_n}} · H{{t.horizon}}</option></select></div>
        <div style="align-self:end"><button class="btn primary" @click="runJoint" :disabled="jointRunning||!selectedJointTask">{{jointRunning?'训练中…':'立即刷新联合模型'}}</button></div>
      </div>
      <div v-if="jointStatus.schema" class="grid cols-4">
        <div><span class="sub">特征</span><div class="big-num">{{jointStatus.spec?.features ?? '—'}}</div></div>
        <div><span class="sub">INNER OOF RankIC</span><div class="big-num">{{num(jointStatus.inner_public_oof?.mean_rank_ic,4)}}</div></div>
        <div><span class="sub">META增量 RankIC</span><div class="big-num">{{num(jointStatus.meta_train?.incremental_mean_rank_ic,4)}}</div></div>
        <div><span class="sub">蒸馏候选</span><div class="big-num">{{jointStatus.distilled_candidates?.length ?? 0}}</div></div>
      </div>
      <div v-if="jointStatus.model" class="sub" style="margin-top:8px">{{jointStatus.model.backend}} · Processor {{jointStatus.processor?.fit_scope}} · HOLDOUT/Vault consumed={{jointStatus.holdout_vault_consumed}} · {{jointStatus.elapsed_seconds}}s</div>
      <div v-if="jointError" class="selector-error">{{jointError}}</div>
    </div>
    <div class="card" style="margin-bottom:14px">
      <div class="panel-title-row"><div><h3>双市场评价</h3><span class="sub">158 个先做训练安全双向筛查，再对训练排名头部做完整审计</span></div><div style="display:flex;gap:8px"><button class="btn" :class="{primary:market==='ashare'}" @click="market='ashare';pickReport()">A股</button><button class="btn" :class="{primary:market==='us'}" @click="market='us';pickReport()">美股</button></div></div>
      <div v-if="reports.length" style="display:flex;gap:8px;align-items:center;margin:10px 0"><label>组合模式</label><select v-model="reportIndex"><option v-for="(report,index) in reports" :key="report.portfolio_mode" :value="index">{{ report.portfolio_mode }}</option></select><a class="btn" :href="reportUrl" target="_blank" rel="noopener">HTML报告</a></div>
      <div v-if="current" class="grid cols-4" style="margin:12px 0"><div><span class="sub">完成</span><div class="big-num">{{ current.completed }}/158</div></div><div><span class="sub">计算失败</span><div class="big-num">{{ current.failed }}</div></div><div><span class="sub">训练通过</span><div class="big-num">{{ current.research_passed }}</div></div><div><span class="sub">完整通过</span><div class="big-num">{{ current.full_passed }}</div></div></div>
      <div v-if="!current" class="combination-empty">该市场尚无已完成报告；这里会显示实时进度。</div>
      <table v-else><thead><tr><th>#</th><th>特征</th><th>家族</th><th>方向</th><th>Gate</th><th>学习分</th><th>等级</th><th>评级RankIC</th><th>评级Sharpe</th><th>表达式</th></tr></thead><tbody><tr v-for="(row,index) in visibleRows" :key="row.name"><td>{{ index+1 }}</td><td><b>{{ row.name }}</b></td><td>{{ row.family }}</td><td>{{ row.direction || '—' }}</td><td>{{ num(row.gate_score,4) }}</td><td>{{ num(row.learning_score,3) }}</td><td><span class="tag" :class="row.full_audit?.grade==='F5'?'green':row.full_audit?'amber':'blue'">{{ row.full_audit?.grade || 'DISCOVERY' }}</span></td><td>{{ num(row.full_audit?.rating_rank_ic,4) }}</td><td>{{ num(row.full_audit?.rating_sharpe,2) }}</td><td><code>{{ row.expression }}</code></td></tr></tbody></table>
    </div>
    <div class="card"><h3>158 特征家族</h3><table><tr><th>机制家族</th><th>数量</th></tr><tr v-for="(count,family) in catalog.families" :key="family"><td>{{ family }}</td><td>{{ count }}</td></tr></table></div>
  </section>`,
  setup() {
    const cap = ref({}), catalog = ref({}), progress = ref({});
    const ashareReports = ref([]), usReports = ref([]), experiments = ref([]);
    const market = ref("ashare"), reportIndex = ref(0), loading = ref(false), error = ref("");
    const selectedJointExpId = ref(null), selectedJointTask = ref("");
    const jointStatus = ref({state:"not_started"}), jointRunning = ref(false), jointError = ref("");
    const jointExperiments = computed(() => experiments.value.filter(e => e.research_config?.qlib_integration?.joint_model_enabled));
    const selectedJointExperiment = computed(() => jointExperiments.value.find(e => Number(e.id)===Number(selectedJointExpId.value)) || null);
    const jointTasks = computed(() => selectedJointExperiment.value?.research_config?.engine_config?.tasks || []);
    const reports = computed(() => market.value === "ashare" ? ashareReports.value : usReports.value);
    const current = computed(() => reports.value[Number(reportIndex.value)] || null);
    const visibleRows = computed(() => (current.value?.rows || []).slice(0, 100));
    const progressLabel = computed(() => progress.value?.state === "complete"
      ? `${(progress.value.modes || []).length} 个模式全部完成 · ${progress.value.run_id || "—"}`
      : `${progress.value?.market || "—"} ${progress.value?.portfolio_mode || ""} · ${progress.value?.completed || 0}/${progress.value?.total || 158} · ${progress.value?.current || "—"}`);
    const reportUrl = computed(() => current.value ? `/api/qlib/alpha158/report?market=${encodeURIComponent(market.value)}&portfolio_mode=${encodeURIComponent(current.value.portfolio_mode)}` : "#");
    function pickReport() { reportIndex.value = 0; }
    async function loadJointStatus() {
      if (!selectedJointExpId.value || !selectedJointTask.value) { jointStatus.value={state:"not_started"}; return; }
      jointError.value="";
      try { jointStatus.value = await api(`/qlib/joint/status?experiment_id=${selectedJointExpId.value}&task_name=${encodeURIComponent(selectedJointTask.value)}`, {cacheTtl:0}); }
      catch(e) { jointError.value=e.message; }
    }
    function pickJointExperiment() {
      selectedJointTask.value = jointTasks.value[0]?.name || "";
      loadJointStatus();
    }
    async function runJoint() {
      jointRunning.value=true; jointError.value="";
      try { jointStatus.value = await api("/qlib/joint/run", {method:"POST", body:{experiment_id:Number(selectedJointExpId.value), task_name:selectedJointTask.value}}); }
      catch(e) { jointError.value=e.message; }
      finally { jointRunning.value=false; }
    }
    function num(value, digits=3) {
      if (value === null || value === undefined || value === "") return "—";
      const n=Number(value); return Number.isFinite(n) ? n.toFixed(digits) : "—";
    }
    async function loadAll() {
      loading.value = true; error.value = "";
      try {
        const [c, cat, p, a, u, e] = await Promise.all([api("/qlib/capabilities"), api("/qlib/alpha158/catalog"), api("/qlib/alpha158/progress"), api("/qlib/alpha158/results?market=ashare"), api("/qlib/alpha158/results?market=us"), api("/experiments", {cacheTtl:0})]);
        cap.value=c; catalog.value=cat; progress.value=p; ashareReports.value=a.reports||[]; usReports.value=u.reports||[]; experiments.value=e.experiments||[];
        if (!jointExperiments.value.some(row => Number(row.id)===Number(selectedJointExpId.value))) { selectedJointExpId.value=jointExperiments.value[0]?.id||null; selectedJointTask.value=jointTasks.value[0]?.name||""; }
        await loadJointStatus();
        if (Number(reportIndex.value) >= reports.value.length) reportIndex.value=0;
      } catch (e) { error.value=e.message; } finally { loading.value=false; }
    }
    onActivated(loadAll);
    return {cap,catalog,progress,progressLabel,ashareReports,usReports,market,reportIndex,reports,current,visibleRows,reportUrl,loading,error,pickReport,num,loadAll,
      jointExperiments,selectedJointExpId,selectedJointTask,jointTasks,jointStatus,jointRunning,jointError,pickJointExperiment,loadJointStatus,runJoint};
  },
};

/* ============ Unified compute progress ============ */
const ComputeTaskDock = {
  template: `
  <div class="compute-dock" :class="{open}">
    <button class="compute-dock-trigger" @click="open=!open" :title="activeCount ? activeCount+' 个计算任务正在运行' : '查看计算任务'">
      <span class="compute-pulse" :class="{active:activeCount>0, failed:activeCount===0&&failedCount>0}"></span>
      <span>计算</span>
      <b>{{ activeCount }}</b>
    </button>
    <div v-if="open" class="compute-dock-panel">
      <div class="compute-dock-heading">
        <div><strong>计算任务中心</strong><small>真实完成量 · 阶段心跳 · 最近结果</small></div>
        <div class="compute-dock-actions">
          <button @click="showRecent=!showRecent">{{ showRecent ? '仅运行中' : '含最近' }}</button>
          <button @click="refresh" :disabled="refreshing">↻</button>
          <button @click="open=false">×</button>
        </div>
      </div>
      <div class="compute-dock-summary">
        <span><b>{{ activeCount }}</b> 运行中</span>
        <span><b>{{ failedCount }}</b> 失败</span>
        <span>{{ observedAt }}</span>
      </div>
      <div v-if="error" class="compute-dock-error">{{ error }}</div>
      <div v-if="!visibleTasks.length" class="compute-dock-empty">当前没有计算任务</div>
      <div v-else class="compute-task-list">
        <article v-for="task in visibleTasks" :key="task.job_id" class="compute-task" :class="'state-'+task.state">
          <div class="compute-task-head">
            <div><span class="compute-kind">{{ kindLabel(task.kind) }}</span><strong>{{ task.title }}</strong></div>
            <span class="compute-state" :class="task.state">{{ stateLabel(task.state) }}</span>
          </div>
          <div class="compute-task-phase"><b>{{ phaseLabel(task.phase) }}</b><span>{{ task.message || '—' }}</span></div>
          <div class="compute-progress-track" :class="{indeterminate:task.indeterminate && isActive(task)}">
            <i v-if="!task.indeterminate" :style="{width:progressWidth(task)}"></i>
            <i v-else-if="isActive(task)"></i>
          </div>
          <div class="compute-task-meta">
            <span v-if="!task.indeterminate">{{ countLabel(task) }} · {{ progressText(task) }}</span>
            <span v-else>{{ isActive(task) ? '持续运行 / 总量未知' : '无可用完成量' }}</span>
            <span>耗时 {{ formatElapsed(task.elapsed_seconds) }}</span>
            <span v-if="isActive(task)">心跳 {{ formatElapsed(task.heartbeat_age_seconds) }}前</span>
            <button v-if="task.cancellable" @click="cancelTask(task)" :disabled="cancelling===task.job_id">{{ cancelling===task.job_id ? '停止中' : '安全停止' }}</button>
          </div>
          <div v-if="task.kind==='research'" class="compute-task-meta">
            <span>有效评价 {{ Number(task.metadata?.effective_evaluations_per_hour || 0).toFixed(1) }}/小时</span>
            <span>重复浪费 {{ (Number(task.metadata?.duplicate_waste_rate || 0)*100).toFixed(1) }}%</span>
            <span>隐藏重采样 {{ task.metadata?.hidden_novelty_resamples ?? 0 }}</span>
            <span>近期独特产出 {{ task.metadata?.recent_unique_yield_rate==null ? '—' : (Number(task.metadata.recent_unique_yield_rate)*100).toFixed(1)+'%' }}</span>
            <span>正式因子 {{ task.metadata?.formal_factor_count ?? task.metadata?.factor_count ?? '—' }}</span>
            <span>研究候选 {{ task.metadata?.research_candidate_count ?? '—' }}</span>
          </div>
          <div v-if="task.error" class="compute-task-error">{{ task.error }}</div>
        </article>
      </div>
      <div class="compute-dock-foot">没有可靠总量的计算只显示不定进度，不使用伪造百分比。</div>
    </div>
  </div>`,
  setup() {
    const open = ref(false), showRecent = ref(true), refreshing = ref(false);
    const tasks = ref([]), activeCount = ref(0), failedCount = ref(0);
    const observedAt = ref("—"), error = ref(""), cancelling = ref("");
    let timer = null;
    const activeStates = new Set(["queued", "starting", "running", "stopping"]);
    const visibleTasks = computed(() => showRecent.value ? tasks.value : tasks.value.filter(isActive));
    function isActive(task) { return activeStates.has(task?.state); }
    function kindLabel(kind) {
      return ({research:"研究",backtest:"回测",combination:"组合",qlib_joint:"联合模型",qlib_alpha158:"Alpha158",panel:"数据",residual_beam:"残差搜索",diagnostics:"诊断",factor_correlation:"相关性",screener:"选股",allocation:"配权"})[kind] || kind || "计算";
    }
    function stateLabel(state) {
      return ({queued:"排队",starting:"启动",running:"运行中",stopping:"停止中",done:"完成",failed:"失败",stopped:"已停止",cancelled:"已取消"})[state] || state || "未知";
    }
    function phaseLabel(phase) {
      return ({
        not_started:"未启动",queued:"排队",starting:"初始化",loading_panel:"加载面板",candidate_mining:"候选挖掘",
        panel_load:"加载面板",feature_graph:"编译特征",feature_materialize:"物化矩阵",feature_cache:"特征缓存",
        split_prepare:"训练切分",walk_forward_oof:"时序 OOF",final_model:"最终模型",dsl_distillation:"DSL 蒸馏",
        materialize:"因子物化",event_simulation:"事件仿真",artifact_write:"审计产物",search:"组合搜索",cross_section:"截面计算",covariance:"协方差估计",effective_trials:"有效试验数",
        rating:"冻结评级",llm_proposal:"LLM 提案",loading:"加载",reloading:"热重载",complete:"完成",done:"完成",
        failed:"失败",stopped:"已停止",stopping:"停止中",
      })[phase] || String(phase || "计算中").replaceAll("_", " ");
    }
    function progressWidth(task) { return `${Math.max(0, Math.min(100, Number(task.progress || 0) * 100)).toFixed(1)}%`; }
    function progressText(task) { return `${(Number(task.progress || 0) * 100).toFixed(1)}%`; }
    function countLabel(task) {
      const done = Number(task.completed), total = Number(task.total);
      return Number.isFinite(done) && Number.isFinite(total) ? `${done.toLocaleString()}/${total.toLocaleString()}` : "—";
    }
    function formatElapsed(value) {
      if (value === null || value === undefined || value === "") return "—";
      const seconds = Number(value);
      if (!Number.isFinite(seconds)) return "—";
      if (seconds < 60) return `${Math.max(0, Math.round(seconds))}秒`;
      if (seconds < 3600) return `${Math.floor(seconds/60)}分${Math.round(seconds%60)}秒`;
      return `${Math.floor(seconds/3600)}时${Math.floor((seconds%3600)/60)}分`;
    }
    async function refresh() {
      if (refreshing.value) return;
      refreshing.value = true;
      try {
        const data = await api("/compute-tasks?include_recent=true&limit=60", {cacheTtl:0});
        tasks.value = data.tasks || [];
        activeCount.value = Number(data.active_count || 0);
        failedCount.value = Number(data.failed_count || 0);
        observedAt.value = data.observed_at ? new Date(data.observed_at).toLocaleTimeString("zh-CN", {hour12:false}) : "—";
        error.value = "";
      } catch (e) { error.value = `进度读取失败：${e.message}`; }
      finally { refreshing.value = false; }
    }
    async function cancelTask(task) {
      if (!task.cancellable || cancelling.value) return;
      cancelling.value = task.job_id;
      try {
        await api(`/compute-tasks/${encodeURIComponent(task.job_id)}/cancel`, {method:"POST"});
        await refresh();
      } catch (e) { error.value = `停止失败：${e.message}`; }
      finally { cancelling.value = ""; }
    }
    onMounted(() => {
      refresh();
      timer = setInterval(refresh, 1800);
    });
    onUnmounted(() => clearInterval(timer));
    return {open,showRecent,refreshing,tasks,visibleTasks,activeCount,failedCount,observedAt,error,cancelling,isActive,kindLabel,stateLabel,phaseLabel,progressWidth,progressText,countLabel,formatElapsed,refresh,cancelTask};
  },
};

/* ============ App ============ */
const App = {
  components: { Dashboard, ResearchTree, ResearchRecordsView, FactorLibrary, FactorLibraryWorkbench, BacktestView, SettingsView, ExperimentsView, ObservabilityView, LeaderboardsView, FactorToolsView, CombinationLabView, ResearchDocumentsView, QlibResearchView, ComputeTaskDock },
  template: `
  <div class="topbar">
    <div class="logo">⚒ FactorFactory</div>
    <div class="tabs" ref="tabsEl">
      <button v-for="t in tabs" :key="t.id" :class="{active: tab===t.id}" @click="selectTab(t.id)">{{ t.label }}</button>
    </div>
    <div class="spacer"></div>
    <span v-if="appState.switchMessage" class="switch-message">{{ appState.switchMessage }}</span>
    <ComputeTaskDock />
    <select v-model="selExp" @change="switchExp" :disabled="appState.switching" style="margin-right:10px; max-width:220px" title="切换活动研究任务">
      <option v-for="e in exps" :key="e.id" :value="e.id" :disabled="e.status==='archived' && !e.active">
        {{ e.name }}{{ e.status==='archived' ? ' (归档)' : '' }}
      </option>
    </select>
    <span class="state-badge" :class="engState==='running' ? 'state-running' : 'state-stopped'">● {{ engState }}</span>
  </div>
  <div class="main" :class="{'main-report-view': tab==='leaderboards'}">
    <div v-if="appState.switching" class="task-switch-overlay"><div class="loading-ring"></div><span>正在切换任务上下文</span></div>
    <KeepAlive :max="16"><component :is="activeComponent" :key="tab" /></KeepAlive>
  </div>`,
  setup() {
    const savedTab = localStorage.getItem("factorfactory.tab");
    const tab = ref(savedTab || "dash");
    appState.activeTab = tab.value;
    const tabs = [
      { id: "dash", label: "总览" }, { id: "tree", label: "研发树" },
      { id: "records", label: "研究记录" }, { id: "factors", label: "因子库" }, { id: "leaderboards", label: "榜单" }, { id: "qlib", label: "Qlib" }, { id: "tools", label: "因子工具" }, { id: "combinations", label: "组合优化" }, { id: "documents", label: "文档" }, { id: "screener", label: "选股器" }, { id: "backtest", label: "回测" },
      { id: "trade-plans", label: "交易计划" }, { id: "exps", label: "实验" }, { id: "diagnostics", label: "诊断" }, { id: "settings", label: "设置" },
    ];
    const engState = ref("…");
    const exps = ref([]), selExp = ref(null);
    const tabsEl = ref(null);
    let timer = null, polling = false;
    const activeComponent = computed(() => ({
      dash: Dashboard, tree: ResearchTree, records: ResearchRecordsView, factors: FactorLibraryWorkbench,
      leaderboards: LeaderboardsView, qlib: QlibResearchView, tools: FactorToolsView, combinations: CombinationLabView, documents: ResearchDocumentsView, screener: ScreenerView, backtest: BacktestView, exps: ExperimentsView,
      diagnostics: ObservabilityView, settings: SettingsView, "trade-plans": TradePlansView,
    })[tab.value] || Dashboard);
    async function poll() {
      if (polling) return;
      polling = true;
      try { engState.value = (await api("/engine/status", { cacheTtl: 900 })).state; }
      catch (e) {}
      finally { polling = false; }
    }
    async function loadExps() {
      try {
        const d = await api("/experiments", { cacheTtl: 1200 });
        exps.value = d.experiments;
        selExp.value = d.active_id;
        appState.experimentId = Number(d.active_id);
      } catch (e) {}
    }
    async function switchExp() {
      try {
        await activateExperiment(selExp.value);
        await Promise.all([loadExps(), poll()]);
      }
      catch (e) { alert("切换失败: " + e.message); loadExps(); }
    }
    function selectTab(id) {
      appState.activeTab = id;
      tab.value = id;
      localStorage.setItem("factorfactory.tab", id);
    }
    function revealActiveTab() {
      tabsEl.value?.querySelector("button.active")
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
    watch(tab, () => nextTick(revealActiveTab), { flush: "post" });
    watch(() => appState.requestedTab, id => {
      if (!id) return;
      selectTab(id);
      appState.requestedTab = null;
    });
    onMounted(() => {
      poll();
      loadExps();
      nextTick(revealActiveTab);
      timer = setInterval(poll, 5000);
    });
    onUnmounted(() => clearInterval(timer));
    return { tab, tabs, tabsEl, engState, exps, selExp, switchExp, selectTab, activeComponent, appState };
  },
};

const ScreenerView = {
  template: `
  <section class="selector-page">
    <div class="selector-heading">
      <div>
        <div class="eyebrow">SINGLE-PASS POLARS / CACHED CROSS-SECTION</div>
        <h1>高性能多因子选股器</h1>
        <p>任务 #{{ appState.experimentId || '—' }} · 每次运行保存不可变快照，可回看当时的因子、参数与完整候选清单。</p>
      </div>
      <span class="tag amber">研究用途 · 非交易批准</span>
    </div>

    <div class="card factor-sleeve-editor screener-manual-factors">
      <div class="panel-title-row"><div><h2>手工多因子输入</h2><span class="sub">与回测页一致：每个因子独立设置名称、DSL、方向和正权重；可与左侧因子库候选共同组合。</span></div><button class="btn" @click="addManualFactor" :disabled="manualRows.length>=12">＋ 添加因子</button></div>
      <div class="factor-sleeve-head"><span>#</span><span>名称</span><span>DSL 表达式</span><span>方向</span><span>原始权重</span><span>归一权重</span><span></span></div>
      <div class="factor-sleeve-row" v-for="(factor,index) in manualRows" :key="factor.uid">
        <b>{{ index+1 }}</b><input v-model="factor.name" :placeholder="'手工因子'+(index+1)" /><textarea v-model="factor.expression" rows="2" class="backtest-expression-input" placeholder="例: rank(close / ts_mean(close, 20) - 1)"></textarea><select v-model.number="factor.direction"><option :value="1">+1 高值优先</option><option :value="-1">-1 低值优先</option></select><input type="number" v-model.number="factor.weight" min="0.000001" step="0.1" /><span class="weight-preview">{{ formatPercent(manualNormalizedWeight(factor)) }}</span><button class="btn danger" @click="removeManualFactor(index)" :disabled="manualRows.length===1">×</button>
      </div>
      <div class="factor-sleeve-summary"><span>空表达式行自动忽略；因子库与手工输入合计最多 12 个，权重统一归一。</span><b>有效 {{ combinedFactorCount }} 个 · 权重 {{ totalWeight.toFixed(4) }}</b></div>
    </div>

      <div class="selector-toolbar card">
      <div class="selector-field selector-date">
        <label>截面日期</label>
        <input type="date" v-model="date" />
      </div>
      <div class="selector-field">
        <label>股票池</label>
        <select v-model.number="univN">
          <option :value="100">Top 100 · 高流动性</option>
          <option :value="300">Top 300 · 大盘池</option>
          <option :value="500">Top 500 · 标准池</option>
          <option :value="1000">Top 1000 · 扩展池</option>
        </select>
      </div>
      <div class="selector-field selector-small">
        <label>输出数量</label>
        <select v-model.number="topN">
          <option :value="20">20 只</option>
          <option :value="50">50 只</option>
          <option :value="100">100 只</option>
        </select>
      </div>
      <div class="selector-field selector-small">
        <label>排名榜单</label>
        <select v-model="outputDirection"><option value="top">头部排名</option><option value="bottom">尾部排名</option><option value="both">头尾双榜</option></select>
      </div>
      <div class="selector-actions">
        <button class="btn" @click="loadFactors" :disabled="loading">刷新因子</button>
        <button class="btn primary" data-testid="screener-run-button" @click="run" :disabled="loading || !canRun">
          {{ loading ? "计算中..." : "执行选股" }}
        </button>
      </div>
    </div>

    <div class="selector-layout">
      <aside class="selector-sidebar">
        <div class="card factor-panel">
          <div class="panel-title-row">
            <div>
              <h2>因子组合</h2>
              <span class="sub">启用因子参与综合排名</span>
            </div>
            <span class="count-badge">{{ enabledCount }}/{{ factors.length }}</span>
          </div>
          <div class="factor-actions">
            <button class="text-btn" @click="selectAll">全选</button>
            <button class="text-btn" @click="clearAll">清空</button>
          </div>
          <div class="factor-filter-box">
            <input v-model="factorSearch" placeholder="快速查找因子…" />
            <select v-model="factorGroup"><option value="">全部结构组</option><option v-for="g in groups" :key="g.id" :value="g.id">{{ g.id }} · {{ g.size }}</option></select>
          </div>
          <div v-if="!factors.length" class="factor-empty">正在加载因子库…</div>
          <div v-else class="factor-list">
            <label v-for="f in filteredFactors" :key="f.expression" class="factor-row" :class="{active:f.enabled}">
              <input type="checkbox" v-model="f.enabled" />
              <span class="factor-mark"></span>
              <span class="factor-copy">
                <span class="factor-name">{{ f.name }}</span>
                <code :title="f.expression">{{ f.expression }}</code>
                <span class="factor-meta">{{ f.group || '单组' }} · {{ f.grade || '未审计' }} · 学习分 {{ formatScore(f.score) }} · {{ f.direction>0?'+1':'-1' }}</span>
              </span>
              <select class="direction-input" v-model.number="f.direction" :disabled="!f.enabled" title="冻结因子方向"><option :value="1">+1</option><option :value="-1">-1</option></select>
              <input class="weight-input" v-model.number="f.weight" type="number" min="0.5" max="5" step="0.5" :disabled="!f.enabled" title="因子权重" />
            </label>
          </div>
          <div class="factor-footer">
            <span>{{ filteredFactors.length }} 条可见 · 组合权重</span><b>{{ totalWeight.toFixed(1) }}</b>
          </div>
        </div>
      </aside>

      <main class="selector-results">
        <div v-if="error" class="selector-error">{{ error }}</div>
        <div v-if="!result && !loading" class="card selector-empty">
          <div class="empty-glyph">◎</div>
          <h2>准备一组研究快照</h2>
          <p>从左侧选择因子，设置日期与股票池，然后执行选股。结果仅代表该截面的模型排序。</p>
          <div class="empty-steps"><span>01 选择因子</span><span>02 冻结参数</span><span>03 查看排名</span></div>
        </div>
          <div v-if="loading" class="card selector-empty">
          <div class="loading-ring"></div><h2>正在计算截面排名</h2><p>正在按股票池过滤数据并合并 {{ combinedFactorCount }} 个因子。</p>
        </div>
        <template v-if="result && !loading">
          <div class="result-summary">
            <div class="result-title"><div><div class="eyebrow">SCREENING SNAPSHOT</div><h2>{{ result.date }} · {{ rankingTitle }}</h2><small v-if="result.date_adjusted" class="sub">请求日 {{ result.requested_date }} 非交易日或超出面板，已回退到最近交易日</small><small v-else-if="result.recorded_at" class="sub">记录时间 {{ formatRecordTime(result.recorded_at) }}</small></div><span class="tag blue">{{ result.run_id ? '记录 #' + result.run_id : '已完成' }}</span></div>
          <div class="metric-strip selector-metrics">
              <div class="metric-card"><span>股票池</span><b>Top {{ result.universe_n || univN }}</b><small>按 60 日成交额</small></div>
              <div class="metric-card"><span>启用因子</span><b>{{ result.factor_count }}</b><small>加权截面排名</small></div>
              <div class="metric-card"><span>输出数量</span><b>{{ result.stocks.length }}</b><small>候选清单</small></div>
              <div class="metric-card accent"><span>{{ result.direction === 'both' ? '头 / 尾首位分' : (result.direction === 'bottom' ? '尾部首位分' : '头部首位分') }}</span><b>{{ result.direction === 'both' ? topScore + ' / ' + tailScore : (result.direction === 'bottom' ? tailScore : topScore) }}</b><small>{{ result.expression_mode ? '单条 DSL 排名' : '方向调整后综合分' }}</small></div>
              <div class="metric-card"><span>有效截面</span><b>{{ result.eligible_count }}</b><small>完整因子交集</small></div>
              <div class="metric-card" :class="{accent:result.performance?.cache_hit}"><span>计算性能</span><b>{{ result.performance?.elapsed_ms }} ms</b><small>{{ result.performance?.cache_hit ? '命中缓存' : '单计划实时计算' }}</small></div>
            </div>
          </div>
          <div class="card result-table-card">
            <div class="panel-title-row candidate-heading"><div><h2>候选清单</h2><span class="sub">勾选任意 N 只股票计算购买比例；点击行查看因子归因</span></div><div class="candidate-actions"><span class="count-badge">已选 {{ selectedStockCount }}</span><button class="text-btn" @click="quickSelectStocks(5)">前 5</button><button class="text-btn" @click="quickSelectStocks(10)">前 10</button><button class="text-btn" @click="selectAllStocks">全选</button><button class="text-btn" @click="clearStockSelection">清空</button><span class="tag">{{ result.date }} · 历史起点 {{ result.history_start }}</span></div></div>
            <table class="result-table"><thead><tr><th class="stock-check-column">配权</th><th>榜内名次</th><th>榜单</th><th>头部名次</th><th>尾部名次</th><th>证券</th><th>名称</th><th>原始收盘</th><th>成交额</th><th>综合分</th><th>极端程度</th></tr></thead>
              <tbody><tr v-for="s in result.stocks" :key="s.side + ':' + s.ts_code" class="clickable" :class="{'top-pick':s.side_rank<=10,'selected-stock':selectedStock?.ts_code===s.ts_code,'portfolio-selected':isStockSelected(s.ts_code)}" @click="selectedStock=s"><td class="stock-check-column"><input type="checkbox" :checked="isStockSelected(s.ts_code)" :aria-label="'选择 ' + s.ts_code + ' 参与配权'" @click.stop @change.stop="toggleStockSelection(s.ts_code, $event.target.checked)" /></td><td><span class="rank-number">{{ String(s.side_rank).padStart(2,"0") }}</span></td><td><span class="tag" :class="s.side==='top'?'green':'red'">{{ s.side==='top'?'头部':'尾部' }}</span></td><td>{{ s.head_rank }}</td><td>{{ s.tail_rank }}</td><td><code class="ticker">{{ s.ts_code }}</code></td><td>{{ s.name || "—" }}</td><td>{{ formatPrice(s.raw_close) }}</td><td>{{ compactAmount(s.amount) }}</td><td><b>{{ formatScore(s.score) }}</b></td><td><span class="rank-bar"><i :style="{ width: rankWidth(s) }"></i></span></td></tr></tbody>
            </table>
          </div>
          <div class="card allocation-card" data-testid="screener-allocation">
            <div class="panel-title-row">
              <div><div class="eyebrow">ROBUST LONG-ONLY SIZING</div><h2>组合配权</h2><span class="sub">排名表达偏好，历史波动与相关性决定资本权重；只使用截面日及以前的数据。</span></div>
              <span class="tag blue">绑定记录 #{{ result.run_id }}</span>
            </div>
            <div class="allocation-controls">
              <label><span>配权模型</span><select v-model="allocationMethod"><option value="robust_risk_budget">稳健风险预算（推荐）</option><option value="inverse_volatility">逆波动率</option><option value="equal_weight">等权基准</option></select></label>
              <label><span>风险窗口</span><select v-model.number="allocationLookback"><option :value="60">60 日</option><option :value="120">120 日</option><option :value="252">252 日</option></select></label>
              <label><span>单股上限</span><select v-model.number="allocationMaxWeight"><option :value="20">20%</option><option :value="25">25%</option><option :value="35">35%</option><option :value="50">50%</option><option :value="100">不限制</option></select></label>
              <label><span>排名倾斜</span><select v-model.number="allocationScoreTilt" :disabled="allocationMethod==='equal_weight'"><option :value="0">0 · 不倾斜</option><option :value="0.2">0.20 · 轻微</option><option :value="0.35">0.35 · 稳健</option><option :value="0.6">0.60 · 较强</option></select></label>
              <button class="btn primary" data-testid="screener-allocation-run" @click="calculateAllocation" :disabled="allocationLoading || !selectedStockCount || !result.run_id">{{ allocationLoading ? '计算中…' : '计算购买比例' }}</button>
            </div>
            <div v-if="allocationError" class="selector-error allocation-error">{{ allocationError }}</div>
            <div v-if="!selectedStockCount" class="allocation-empty">先在候选清单中勾选任意 N 只股票。</div>
            <div v-else-if="!allocation && !allocationLoading" class="allocation-empty">已选择 {{ selectedStockCount }} 只股票，点击“计算购买比例”生成可审计的配权结果。</div>
            <div v-if="allocationLoading" class="allocation-empty">正在估计截至 {{ result.date }} 的稳健协方差并求解风险预算…</div>
            <template v-if="allocation && !allocationLoading">
              <div class="metric-strip allocation-metrics">
                <div class="metric-card"><span>选中股票</span><b>{{ allocation.selection_count }}</b><small>任意勾选集合</small></div>
                <div class="metric-card accent"><span>组合预估年化波动</span><b>{{ formatPercent(allocation.diagnostics.portfolio_annualised_volatility) }}</b><small>等权 {{ formatPercent(allocation.diagnostics.equal_weight_annualised_volatility) }}</small></div>
                <div class="metric-card"><span>有效持仓数</span><b>{{ Number(allocation.diagnostics.effective_holdings).toFixed(2) }}</b><small>1 / Σw²</small></div>
                <div class="metric-card"><span>分散化比率</span><b>{{ Number(allocation.diagnostics.diversification_ratio).toFixed(2) }}</b><small>越高表示风险分散越充分</small></div>
                <div class="metric-card"><span>平均相关性</span><b>{{ Number(allocation.diagnostics.average_correlation).toFixed(2) }}</b><small>{{ allocation.diagnostics.observations }} 个共同样本</small></div>
              </div>
              <div class="allocation-meta"><span>{{ allocationMethodLabel(allocation.method) }}</span><span>样本 {{ allocation.diagnostics.sample_start }} → {{ allocation.diagnostics.sample_end }}</span><span>收缩强度 {{ Number(allocation.parameters.covariance_shrinkage).toFixed(2) }}</span><span>单股有效上限 {{ formatPercent(allocation.parameters.effective_max_weight) }}</span><span v-if="allocation.panel_changed" class="amber-text">面板版本已变化</span></div>
              <div class="allocation-table-wrap"><table class="allocation-table"><thead><tr><th>证券</th><th>榜单 / 名次</th><th>购买比例</th><th>风险贡献</th><th>目标风险预算</th><th>个股年化波动</th><th>配权强度</th></tr></thead><tbody><tr v-for="row in allocation.allocations" :key="row.ts_code"><td><code class="ticker">{{ row.ts_code }}</code><small>{{ row.name || '—' }}</small></td><td><span class="tag" :class="row.side==='top'?'green':'red'">{{ row.side==='top'?'头部':'尾部' }} #{{ row.side_rank }}</span></td><td><b class="allocation-weight">{{ formatPercent(row.purchase_weight, 2) }}</b></td><td>{{ formatPercent(row.risk_contribution, 2) }}</td><td>{{ formatPercent(row.target_risk_budget, 2) }}</td><td>{{ formatPercent(row.annualised_volatility, 1) }}</td><td><span class="weight-bar"><i :style="{width:formatPercent(row.purchase_weight)}"></i></span></td></tr></tbody></table></div>
              <div class="allocation-warnings"><b>使用边界</b><ul><li v-for="warning in allocation.warnings" :key="warning">{{ warning }}</li></ul></div>
            </template>
          </div>
          <div class="card contribution-card" v-if="selectedStock">
            <div class="panel-title-row"><div><h2>{{ selectedStock.ts_code }} · 排名归因</h2><span class="sub">{{ selectedStock.name }} · 综合分 {{ formatScore(selectedStock.score) }}</span></div><button class="btn" @click="selectedStock=null">关闭</button></div>
            <table><tr><th>因子 / 表达式</th><th>方向</th><th>权重</th><th>因子原值</th><th>截面分</th><th>贡献</th></tr><tr v-for="row in selectedStock.components" :key="row.expression"><td><b>{{ row.name || '因子' }}</b><div class="mono-expr">{{ row.expression }}</div></td><td>{{ row.direction===1?'正向':'反向' }}</td><td>{{ (row.weight*100).toFixed(1) }}%</td><td>{{ formatScore(row.value) }}</td><td>{{ formatScore(row.rank_score) }}</td><td><b>{{ formatScore(row.contribution) }}</b></td></tr></table>
          </div>
          <div class="selector-disclaimer"><span>ⓘ</span> 这是基于当前研究面板的横截面排序。尾部榜代表方向调整后综合分最低；A 股纯多头任务中仅用于回避/负向观察，不代表允许做空。数据为 non-PIT 当前成分股回看历史，结果不等同于可交易信号。</div>
        </template>
      </main>
    </div>

    <div class="card screener-history-card" data-testid="screener-history">
      <div class="panel-title-row">
        <div>
          <div class="eyebrow">TASK-SCOPED AUDIT TRAIL</div>
          <h2>本任务选股记录</h2>
          <span class="sub">仅显示任务 #{{ appState.experimentId || '—' }}；记录按生成时快照保存，不随当前因子库或设置变化。</span>
        </div>
        <div class="history-heading-actions">
          <span class="count-badge">{{ historyTotal }} 次</span>
          <button class="btn" data-testid="screener-history-refresh" @click="loadHistory" :disabled="historyLoading">刷新</button>
        </div>
      </div>
      <div v-if="historyError" class="selector-error history-error">{{ historyError }}</div>
      <div v-if="historyLoading && !historyRuns.length" class="history-empty">正在读取任务记录…</div>
      <div v-else-if="!historyRuns.length" class="history-empty">这个任务还没有选股记录。执行一次选股后，完整快照会保存在这里。</div>
      <div v-else class="history-table-wrap">
        <table class="history-table">
          <thead><tr><th>记录</th><th>截面</th><th>榜单 / 模式</th><th>冻结配置</th><th>候选预览</th><th>性能</th><th></th></tr></thead>
          <tbody>
            <tr v-for="runRow in historyRuns" :key="runRow.id" :class="{active:activeHistoryId===runRow.id}" @click="openHistory(runRow)">
              <td><b>#{{ runRow.id }}</b><small>{{ formatRecordTime(runRow.created_at) }}</small></td>
              <td><b>{{ runRow.target_date }}</b><small v-if="runRow.date_adjusted">请求 {{ runRow.requested_date }} · 已回退</small><small v-else>实际交易日</small></td>
              <td><span class="tag" :class="runRow.direction==='bottom'?'red':(runRow.direction==='both'?'amber':'green')">{{ directionLabel(runRow.direction) }}</span><small>{{ runRow.expression_mode ? '直接 DSL' : runRow.factor_count + ' 因子组合' }} · {{ runRow.portfolio_mode }}</small></td>
              <td><b>Top {{ runRow.universe_n }} → {{ runRow.result_count }}</b><small :title="factorPreviewTitle(runRow)">{{ factorPreviewLabel(runRow) }}</small></td>
              <td><span class="stock-preview" v-if="runRow.stock_preview?.length"><code v-for="stock in runRow.stock_preview" :key="stock.side + ':' + stock.ts_code">{{ stock.ts_code }}</code></span><small v-else>没有候选</small></td>
              <td><b>{{ formatLatency(runRow.elapsed_ms) }}</b><small>{{ runRow.cache_hit ? '缓存命中' : '实时计算' }} · 有效 {{ runRow.eligible_count }}</small></td>
              <td><button class="text-btn" @click.stop="openHistory(runRow)" :disabled="openingRunId===runRow.id">{{ openingRunId===runRow.id ? '加载中' : '回看' }}</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>`,
  setup() {
    const date = ref(new Date().toISOString().slice(0,10));
    const univN = ref(500); const topN = ref(50);
    const factors = ref([]);
    const groups = ref([]);
    const factorSearch = ref("");
    const factorGroup = ref("");
    let manualUid = 1;
    const manualRows = ref([{uid:1,name:"手工因子1",expression:"",weight:1,direction:1}]);
    const outputDirection = ref("top");
    const result = ref(null); const error = ref(""); const loading = ref(false);
    const selectedStock = ref(null);
    const selectedStockSymbols = ref([]);
    const allocation = ref(null);
    const allocationLoading = ref(false);
    const allocationError = ref("");
    const allocationMethod = ref("robust_risk_budget");
    const allocationLookback = ref(120);
    const allocationMaxWeight = ref(35);
    const allocationScoreTilt = ref(0.35);
    let allocationRequestVersion = 0;
    const historyRuns = ref([]);
    const historyTotal = ref(0);
    const historyLoading = ref(false);
    const historyError = ref("");
    const activeHistoryId = ref(null);
    const openingRunId = ref(null);
    let loadedExperimentVersion = -1;
    const enabledCount = computed(() => factors.value.filter(f=>f.enabled).length);
    const selectedStockCount = computed(() => selectedStockSymbols.value.length);
    const validManualRows = computed(() => manualRows.value.filter(row=>String(row.expression||"").trim()));
    const combinedFactorCount = computed(() => enabledCount.value + validManualRows.value.length);
    const totalWeight = computed(() => factors.value.filter(f=>f.enabled).reduce((sum, f) => sum + Math.max(0,Number(f.weight)||0), 0) + validManualRows.value.reduce((sum,row)=>sum+Math.max(0,Number(row.weight)||0),0));
    const canRun = computed(() => combinedFactorCount.value > 0 && combinedFactorCount.value <= 12 && totalWeight.value > 0);
    const topScore = computed(() => {
      const row = result.value?.stocks?.find(stock => stock.side === "top");
      return row ? formatScore(row.score) : "—";
    });
    const tailScore = computed(() => {
      const row = result.value?.stocks?.find(stock => stock.side === "bottom");
      return row ? formatScore(row.score) : "—";
    });
    const rankingTitle = computed(() => (
      result.value?.direction === "bottom"
        ? "尾部排名"
        : result.value?.direction === "both"
          ? "头尾双榜"
          : "头部排名"
    ));
    const filteredFactors = computed(() => {
      const query = factorSearch.value.trim().toLowerCase();
      return factors.value.filter(f => {
        if (factorGroup.value && f.group !== factorGroup.value) return false;
        if (!query) return true;
        return [f.name, f.expression, f.group, f.family, f.grade]
          .some(value => String(value || "").toLowerCase().includes(query));
      });
    });
    function formatScore(value) { return value == null ? "—" : Number(value).toFixed(2); }
    function formatPercent(value, digits=1) {
      const number = Number(value);
      return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
    }
    function formatPrice(value) {
      if (value == null || !Number.isFinite(Number(value))) return "—";
      return Number(value).toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
    }
    function compactAmount(value) {
      const amount = Number(value);
      if (!Number.isFinite(amount)) return "—";
      if (Math.abs(amount) >= 1e9) return `${(amount / 1e9).toFixed(2)}B`;
      if (Math.abs(amount) >= 1e6) return `${(amount / 1e6).toFixed(2)}M`;
      if (Math.abs(amount) >= 1e3) return `${(amount / 1e3).toFixed(2)}K`;
      return amount.toFixed(0);
    }
    function rankWidth(stock) {
      const eligible = Number(result.value?.eligible_count || 0);
      if (!eligible) return "0%";
      const sideRank = Number(stock.side === "bottom" ? stock.tail_rank : stock.head_rank);
      return `${Math.max(0, Math.min(100, (eligible - sideRank + 1) / eligible * 100))}%`;
    }
    function formatRecordTime(value) {
      if (!value) return "—";
      return String(value).replace("T", " ").slice(0, 19);
    }
    function formatLatency(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "—";
      return number >= 1000 ? `${(number / 1000).toFixed(2)} s` : `${number.toFixed(0)} ms`;
    }
    function directionLabel(value) {
      return value === "bottom" ? "尾部" : value === "both" ? "头尾双榜" : "头部";
    }
    function factorPreviewLabel(runRow) {
      const factorsPreview = runRow.factor_preview || [];
      if (!factorsPreview.length) return "因子快照不可用";
      const suffix = runRow.factor_count > factorsPreview.length ? ` 等 ${runRow.factor_count} 条` : "";
      return `${factorsPreview[0].direction === -1 ? "反向 " : ""}${factorsPreview[0].expression}${suffix}`;
    }
    function factorPreviewTitle(runRow) {
      return (runRow.factor_preview || [])
        .map(row => `${row.direction === -1 ? "-1" : "+1"} × ${row.weight} · ${row.expression}`)
        .join("\n");
    }
    function allocationMethodLabel(value) {
      if (value === "inverse_volatility") return "逆波动率";
      if (value === "equal_weight") return "等权基准";
      return "稳健风险预算";
    }
    function resetAllocation() {
      allocationRequestVersion += 1;
      allocation.value = null;
      allocationError.value = "";
      allocationLoading.value = false;
    }
    function isStockSelected(symbol) {
      return selectedStockSymbols.value.includes(symbol);
    }
    function toggleStockSelection(symbol, checked) {
      const next = new Set(selectedStockSymbols.value);
      if (checked) next.add(symbol); else next.delete(symbol);
      selectedStockSymbols.value = [...next];
    }
    function quickSelectStocks(count) {
      selectedStockSymbols.value = (result.value?.stocks || []).slice(0, count).map(row => row.ts_code);
    }
    function selectAllStocks() {
      selectedStockSymbols.value = (result.value?.stocks || []).map(row => row.ts_code);
    }
    function clearStockSelection() {
      selectedStockSymbols.value = [];
    }

    async function loadFactors() {
      const requestedVersion = appState.experimentVersion;
      try {
        error.value = "";
        const experimentId = Number(appState.experimentId);
        const taskQuery = experimentId ? `&experiment_id=${encodeURIComponent(experimentId)}` : "";
        const similarityQuery = experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : "";
        const [d, similarity] = await Promise.all([
          api(`/factors?sort=grade${taskQuery}`, { cacheTtl: 1000 }),
          api(`/factors/similarity-groups${similarityQuery}`, { cacheTtl: 1000 }),
        ]);
        if (requestedVersion !== appState.experimentVersion) return;
        const groupByFactor = new Map();
        for (const group of similarity.groups || []) {
          for (const id of group.factor_ids || []) groupByFactor.set(Number(id), group);
        }
        groups.value = (similarity.groups || []).filter(group => group.size > 1);
        const fs = (d.factors||[]).filter(f=>f.expression);
        const seen=new Set(); const uniq=[];
        for (const f of fs.sort((a,b)=>(b.public?.score||0)-(a.public?.score||0))) {
          if (!seen.has(f.expression)) { seen.add(f.expression); uniq.push(f); }
          if (uniq.length>=50) break;
        }
        const audited = uniq.filter(f => ["F3","F4","F5"].includes(f.eligibility?.grade));
        const defaultExpressions = new Set(audited.slice(0,10).map(f => f.expression));
        factors.value = uniq.map((f)=>({
          expression: f.expression,
          name: f.name || f.expression.slice(0,30),
          enabled: defaultExpressions.has(f.expression),
          weight: 1.0,
          grade: f.eligibility?.grade,
          lifecycle: f.lifecycle_stage,
          score: f.public?.score,
          direction: Number(f.research_meta?.direction || 1),
          group: groupByFactor.get(Number(f.id))?.id,
          family: groupByFactor.get(Number(f.id))?.family,
        }));
        loadedExperimentVersion = appState.experimentVersion;
      } catch(e) {
        if (requestedVersion === appState.experimentVersion) {
          error.value="加载因子失败: "+e.message;
        }
      }
    }
    function ensureFresh() {
      if (loadedExperimentVersion !== appState.experimentVersion) loadFactors();
    }

    function selectAll() { filteredFactors.value.forEach(f => { f.enabled = true; }); }
    function clearAll() { factors.value.forEach(f => { f.enabled = false; }); }

    function addManualFactor() { if (manualRows.value.length<12) manualRows.value.push({uid:++manualUid,name:`手工因子${manualRows.value.length+1}`,expression:"",weight:1,direction:1}); }
    function removeManualFactor(index) { if (manualRows.value.length>1) manualRows.value.splice(index,1); }
    function manualNormalizedWeight(row) { return totalWeight.value>0 ? Math.max(0,Number(row.weight)||0)/totalWeight.value : 0; }

    async function run() {
      const enabled = factors.value.filter(f=>f.enabled);
      const manual = validManualRows.value;
      if (!enabled.length && !manual.length) { error.value="请至少选择一个因子或输入 DSL 表达式"; return; }
      if (enabled.length + manual.length > 12) { error.value="因子库与手工输入合计最多 12 个因子"; return; }
      const requestedVersion = appState.experimentVersion;
      const experimentId = Number(appState.experimentId) || undefined;
      loading.value=true; error.value=""; result.value=null; selectedStock.value=null; selectedStockSymbols.value=[]; resetAllocation();
      try {
        const r = await api("/screener", { method:"POST", body:{
          experiment_id: experimentId,
          factors: [
            ...enabled.map(f=>({name:f.name,expression:f.expression,weight:f.weight,direction:f.direction})),
            ...manual.map((f,index)=>({name:String(f.name||`手工因子${index+1}`).trim(),expression:String(f.expression).trim(),weight:f.weight,direction:f.direction})),
          ],
          date: date.value, universe_n: univN.value, top_n: topN.value, direction:outputDirection.value
        }});
        if (requestedVersion !== appState.experimentVersion) return;
        result.value = r;
        activeHistoryId.value = r.run_id;
        await loadHistory();
      } catch(e) {
        if (requestedVersion === appState.experimentVersion) {
          error.value = "选股失败: "+e.message;
        }
      }
      finally {
        if (requestedVersion === appState.experimentVersion) loading.value=false;
      }
    }

    async function calculateAllocation() {
      if (!selectedStockSymbols.value.length) {
        allocationError.value = "请至少勾选一只股票";
        return;
      }
      if (!result.value?.run_id) {
        allocationError.value = "当前结果没有不可变记录 ID，请重新执行选股";
        return;
      }
      const requestedVersion = appState.experimentVersion;
      const requestVersion = ++allocationRequestVersion;
      const experimentId = Number(appState.experimentId) || undefined;
      allocationLoading.value = true;
      allocationError.value = "";
      allocation.value = null;
      try {
        const response = await api("/screener/allocate", { method: "POST", body: {
          experiment_id: experimentId,
          run_id: result.value.run_id,
          symbols: selectedStockSymbols.value,
          method: allocationMethod.value,
          lookback: allocationLookback.value,
          max_weight: Number(allocationMaxWeight.value) / 100,
          score_tilt: allocationMethod.value === "equal_weight" ? 0 : allocationScoreTilt.value,
        }});
        if (requestedVersion !== appState.experimentVersion || requestVersion !== allocationRequestVersion) return;
        allocation.value = response;
      } catch (e) {
        if (requestedVersion === appState.experimentVersion && requestVersion === allocationRequestVersion) {
          allocationError.value = `配权失败: ${e.message}`;
        }
      } finally {
        if (requestedVersion === appState.experimentVersion && requestVersion === allocationRequestVersion) allocationLoading.value = false;
      }
    }

    async function loadHistory() {
      const experimentId = Number(appState.experimentId);
      const requestedVersion = appState.experimentVersion;
      const query = experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}&limit=50` : "?limit=50";
      historyLoading.value = true;
      historyError.value = "";
      try {
        const data = await api(`/screener/runs${query}`);
        if (requestedVersion !== appState.experimentVersion) return;
        historyRuns.value = data.runs || [];
        historyTotal.value = Number(data.total || 0);
      } catch (e) {
        if (requestedVersion === appState.experimentVersion) {
          historyError.value = `读取选股记录失败: ${e.message}`;
        }
      } finally {
        if (requestedVersion === appState.experimentVersion) historyLoading.value = false;
      }
    }

    async function openHistory(runRow) {
      const experimentId = Number(appState.experimentId);
      const requestedVersion = appState.experimentVersion;
      openingRunId.value = runRow.id;
      historyError.value = "";
      try {
        const suffix = experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : "";
        const data = await api(`/screener/runs/${runRow.id}${suffix}`);
        if (requestedVersion !== appState.experimentVersion) return;
        result.value = data.run.result;
        activeHistoryId.value = runRow.id;
        selectedStock.value = null;
        selectedStockSymbols.value = [];
        resetAllocation();
        error.value = "";
        nextTick(() => document.querySelector(".result-summary")?.scrollIntoView({ behavior: "smooth", block: "start" }));
      } catch (e) {
        if (requestedVersion === appState.experimentVersion) {
          historyError.value = `读取记录 #${runRow.id} 失败: ${e.message}`;
        }
      } finally {
        if (requestedVersion === appState.experimentVersion) openingRunId.value = null;
      }
    }

    watch(selectedStockSymbols, resetAllocation, { deep: true });
    watch([allocationMethod, allocationLookback, allocationMaxWeight, allocationScoreTilt], resetAllocation);
    watch(() => appState.experimentVersion, () => {
      result.value=null;
      selectedStock.value=null;
      selectedStockSymbols.value=[];
      resetAllocation();
      loading.value=false;
      historyRuns.value=[];
      historyTotal.value=0;
      historyLoading.value=false;
      historyError.value="";
      activeHistoryId.value=null;
      openingRunId.value=null;
      manualRows.value=[{uid:++manualUid,name:"手工因子1",expression:"",weight:1,direction:1}];
      factorSearch.value="";
      factorGroup.value="";
      if (appState.activeTab === "screener") {
        ensureFresh();
        loadHistory();
      }
    });
    onActivated(() => {
      ensureFresh();
      loadHistory();
    });
    return {
      appState,
      date, univN, topN, manualRows, validManualRows, combinedFactorCount, outputDirection,
      factors, groups, factorSearch, factorGroup, filteredFactors,
      result, selectedStock, error, loading, enabledCount, canRun,
      selectedStockSymbols, selectedStockCount, allocation, allocationLoading, allocationError,
      allocationMethod, allocationLookback, allocationMaxWeight, allocationScoreTilt,
      historyRuns, historyTotal, historyLoading, historyError, activeHistoryId, openingRunId,
      totalWeight, topScore, tailScore, rankingTitle, formatScore, formatPercent, formatPrice, compactAmount,
      rankWidth, formatRecordTime, formatLatency, directionLabel, factorPreviewLabel, factorPreviewTitle,
      allocationMethodLabel, isStockSelected, toggleStockSelection, quickSelectStocks, selectAllStocks, clearStockSelection,
      selectAll, clearAll, addManualFactor, removeManualFactor, manualNormalizedWeight, run, calculateAllocation, loadFactors, loadHistory, openHistory,
    };
  },
};
App.components.ScreenerView = ScreenerView;
createApp(App).mount("#app");
