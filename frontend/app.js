/* USStockFactorFactory 前端 — Vue3 全局构建 + ECharts */
const {
  createApp, ref, reactive, computed, onMounted, onUnmounted,
  onActivated, onDeactivated, watch, nextTick,
} = Vue;

const appState = reactive({
  experimentId: null,
  experimentVersion: 0,
  activeTab: localStorage.getItem("factorfactory.tab") || "dash",
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
        <div class="sub">实验: {{ st.experiment?.name ?? '—' }} · 外层步 {{ st.outer_step }} · 内层评估 {{ st.inner_evals }}</div>
        <div style="margin-top:10px; display:flex; gap:8px">
          <button class="btn primary" @click="start" :disabled="st.state==='running'">启动 7×24</button>
          <button class="btn danger" @click="stop" :disabled="st.state!=='running'">停止</button>
        </div>
      </div>
      <div class="card"><h3>因子库</h3><div class="big-num">{{ st.counts?.factors ?? '—' }}</div><div class="sub">已入库因子</div></div>
      <div class="card"><h3>搜索树节点</h3><div class="big-num">{{ st.counts?.nodes ?? '—' }}</div><div class="sub">累计内层评估节点</div></div>
      <div class="card"><h3>外层接受率</h3>
        <div class="big-num">{{ acceptRate }}</div>
        <div class="sub">{{ st.counts?.accepted ?? 0 }} / {{ st.counts?.outer_steps ?? 0 }} 步被接受</div>
      </div>
    </div>
    <div class="card" style="margin-bottom:14px">
      <div class="panel-title-row"><div><h3>并行任务与数据身份</h3><span class="sub">同一端口内独立 worker；历史数据按任务 ID 隔离</span></div><span class="tag blue">{{ (st.workers || []).length }} workers</span></div>
      <table><tr><th>任务</th><th>市场</th><th>模式</th><th>方向</th><th>状态</th><th>外层步</th><th>内层评估</th></tr>
        <tr v-for="w in (st.workers || [])" :key="w.experiment_id"><td>{{ w.experiment_id }}</td><td>{{ w.task_config?.market || '—' }}</td><td>{{ w.task_config?.portfolio_mode || '—' }}</td><td>{{ Number(w.task_config?.direction || 1)===1 ? '高值偏多' : '低值偏多' }}</td><td>{{ w.state }}</td><td>{{ w.outer_step }}</td><td>{{ w.inner_evals }}</td></tr>
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
    async function start() { await api("/engine/start", { method: "POST" }); refresh(); }
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
        <tr><th>版本</th><th>状态</th><th>meta-score</th><th>提案</th></tr>
        <tr v-for="v in data.versions" :key="v.id" class="clickable" @click="selectVersion(v.id)"
            :style="{background: v.id===selected ? '#1c2733' : ''}">
          <td>v{{ v.version_no }}</td>
          <td><span class="tag" :class="{green: v.status==='incumbent', red: v.status==='rejected', amber: v.status==='superseded', blue: v.status==='candidate'}">{{ v.status }}</span></td>
          <td>{{ v.meta_score == null ? '—' : v.meta_score.toFixed(4) }}</td>
          <td style="color:var(--muted); max-width:500px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">{{ v.note }}</td>
        </tr>
      </table>
    </div>
    <div class="card">
      <h3>内层搜索树 {{ selected ? '(v' + versionNo(selected) + ')' : '(全部, 最近800节点)' }} — 点击节点看表达式</h3>
      <div class="chart tall" ref="treeEl"></div>
      <div v-if="picked" style="margin-top:10px; padding:10px; border:1px solid var(--border); border-radius:6px">
        <div class="mono-expr">{{ picked.expression }}</div>
        <div class="sub">op={{ picked.op }} · source={{ picked.source }} · task={{ picked.task }} · public_score={{ picked.public_score?.toFixed(4) }} · 状态: {{ picked.status }}</div>
      </div>
    </div>
  </div>`,
  setup() {
    const data = ref({ versions: [], nodes: [] });
    const selected = ref(null), picked = ref(null);
    const treeEl = ref(null);
    let timer = null, refreshing = false;
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
        tooltip: { formatter: (p) => p.data.raw ? `${p.data.name}<br/>score=${(p.data.raw.public_score ?? 0).toFixed(3)}<br/>${p.data.raw.expression?.slice(0, 60)}` : "" },
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
    return { data, selected, picked, treeEl, selectVersion, versionNo };
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
        <table><tr><th></th><th>IC均值</th><th>ICIR</th><th>期一致性</th><th>换手</th><th>综合分</th></tr>
          <tr v-for="(m,k) in {public: evalResult.public, gate: evalResult.gate}" :key="k">
            <td>{{ k==='public' ? 'PUBLIC(训练可见)' : 'GATE(门禁)' }}</td>
            <td>{{ f(m.ic_mean) }}</td><td>{{ f(m.icir) }}</td><td>{{ f(m.era_consistency) }}</td><td>{{ f(m.turnover) }}</td><td><b>{{ f(m.score) }}</b></td>
          </tr></table>
      </div>
      <div v-if="evalErr" style="color:var(--red); margin-top:8px">{{ evalErr }}</div>
    </div>
    <div class="card">
      <h3>因子库 ({{ factors.length }})</h3>
      <table>
        <tr><th>名称</th><th>表达式</th><th>状态</th><th>任务</th><th>PUB ICIR</th><th>GATE ICIR</th><th>GATE 分</th><th>时间</th></tr>
        <tr v-for="fa in factors" :key="fa.id" class="clickable" @click="open(fa)">
          <td>{{ fa.name }}</td>
          <td class="mono-expr" style="max-width:380px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">{{ fa.expression }}</td>
          <td><span class="tag" :class="{green: fa.status==='library-admitted', blue: fa.status==='public-leading', amber: fa.status==='paper', red: fa.status==='retired'}">{{ fa.status }}</span></td>
          <td>{{ fa.task }}</td>
          <td>{{ f(fa.public?.icir) }}</td><td>{{ f(fa.gate?.icir) }}</td><td>{{ f(fa.gate?.score) }}</td>
          <td class="sub">{{ fa.created_at?.slice(5,16) }}</td>
        </tr>
      </table>
    </div>

    <div class="drawer" v-if="detail">
      <button class="btn close" @click="detail=null">✕ 关闭</button>
      <h2 style="margin-bottom:6px">{{ detail.factor.name }}</h2>
      <div class="mono-expr" style="margin-bottom:8px">{{ detail.factor.expression }}</div>
      <div class="sub" style="margin-bottom:12px">{{ detail.factor.hypothesis }}</div>
      <div style="display:flex; gap:8px; margin-bottom:14px">
        <button class="btn" v-for="s in ['library-admitted','paper','retired']" :key="s" @click="setStatus(s)">标记 {{ s }}</button>
      </div>
      <div class="card" style="margin-bottom:12px">
        <h3>分层 · 分期 IC 步进 (含隔离层)</h3>
        <div class="chart" ref="eraEl"></div>
      </div>
      <div class="card">
        <h3>各层指标</h3>
        <table><tr><th>层</th><th>IC均值</th><th>ICIR</th><th>一致性</th><th>综合分</th></tr>
          <tr v-for="(m,layer) in detail.layers" :key="layer">
            <td>{{ layer }}</td><td>{{ f(m.ic_mean) }}</td><td>{{ f(m.icir) }}</td><td>{{ f(m.era_consistency) }}</td><td>{{ f(m.score) }}</td>
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
        yAxis: { type: "value", name: "era IC", splitLine: { lineStyle: { color: "#21262d" } } },
        series: [{
          type: "bar",
          data: eras.map((e) => ({ value: e.ic_mean, itemStyle: { color: colors[e.layer] || "#8b949e" } })),
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
      <div class="panel-title-row"><div><div class="eyebrow">FACTOR LIBRARY / STRUCTURAL INDEX</div><h1>因子研究资产库</h1><span class="sub">生命周期、实战审计与结构相似度索引统一管理；HOLDOUT / VAULT 仅在显式审计时读取。</span></div><div><span class="tag blue">{{ factors.length }} 条记录</span> <span class="tag green">{{ groupStats.groups || 0 }} 个结构组</span> <span class="tag amber">NON_PIT_RESEARCH</span></div></div>
      <div class="form-row" style="margin-top:14px">
        <input style="flex:3" v-model="query" @keyup.enter="refresh" placeholder="搜索名称、表达式、经济学假设…" />
        <select v-model="status"><option value="">全部生命周期</option><option value="discovery_only">F1 · discovery_only</option><option value="research_pass">F2 · research_pass</option><option value="oos_pass">F3 · oos_pass</option><option value="paper_candidate">F4 · paper_candidate</option><option value="live_candidate_non_pit">F5 · live_candidate_non_pit</option><option value="legacy_unreviewed">旧协议未审计</option><option value="invalid_provenance">来源无效</option><option value="configuration_changed_requires_reaudit">配置变更待复审</option></select>
        <select v-model="groupFilter"><option value="">全部相似组</option><option v-for="g in groups" :key="g.id" :value="g.id">{{ g.id }} · {{ familyLabel(g.family) }} · {{ g.size }}个</option></select>
        <select v-model="sort"><option value="score">按研究分</option><option value="grade">按实战等级</option><option value="icir">按 ICIR</option><option value="created">按最新</option></select>
        <button class="btn" @click="refresh">刷新</button><button class="btn primary" @click="compare" :disabled="selected.length<2">比较 {{ selected.length }} 个</button>
      </div>
      <div class="similarity-summary">
        <span>SimHash LSH + 加权 Jaccard</span>
        <b>{{ groupStats.duplicate_groups || 0 }}</b><small>个重复簇</small>
        <b>{{ pct(groupStats.redundancy_ratio) }}</b><small>结构冗余率</small>
        <button class="text-btn" @click="groupFilter=''">清除分组筛选</button>
      </div>
    </div>

    <div class="grid cols-2" v-if="comparison">
      <div class="card">
        <div class="panel-title-row"><h2>V3 训练层复评</h2><span class="tag amber">{{ comparison.portfolio_mode }}</span></div>
        <table><tr><th>表达式</th><th>PUB ICIR</th><th>GATE ICIR</th><th>GATE 费后 Sharpe</th><th>研究分</th></tr>
          <tr v-for="r in comparison.results" :key="r.expression"><td class="mono-expr">{{ r.expression }}</td><td>{{ f(r.public?.icir) }}</td><td>{{ f(r.gate?.icir) }}</td><td>{{ f(layerSharpe(r.gate)) }}</td><td><b>{{ f(r.discovery?.score) }}</b></td></tr></table>
      </div>
      <div class="card"><h2>横截面冗余检查</h2><div class="sub">{{ comparison.correlation?.date }} · {{ comparison.correlation?.n }} 只股票</div>
        <table><tr><th></th><th v-for="(_,i) in comparison.correlation.matrix" :key="i">F{{ i+1 }}</th></tr>
          <tr v-for="(row,i) in comparison.correlation.matrix" :key="i"><th>F{{ i+1 }}</th><td v-for="(v,j) in row" :key="j" :class="Math.abs(v)>=0.8 && i!==j ? 'corr-high' : ''">{{ v.toFixed(2) }}</td></tr></table>
        <div class="sub" style="margin-top:8px">相关系数绝对值 ≥ 0.80 标红，表示候选可能是同一风险暴露的重复表达。</div>
      </div>
    </div>

    <div class="card">
      <div class="panel-title-row"><h2>研究资产</h2><span class="sub">{{ visibleFactors.length }} 条 · 点击行读取详情与相似因子</span></div>
      <table><tr><th><input type="checkbox" @change="toggleAll" /></th><th>名称</th><th>结构组</th><th>表达式</th><th>协议</th><th>生命周期</th><th>等级</th><th>PUB ICIR</th><th>GATE Sharpe</th><th>研究分</th><th>来源</th></tr>
        <tr v-for="fa in visibleFactors" :key="fa.id" class="clickable" @click="open(fa)">
          <td @click.stop><input type="checkbox" :value="fa.id" v-model="selected" /></td><td><b>{{ fa.name }}</b></td>
          <td><button class="group-chip" @click.stop="groupFilter=groupFor(fa.id)">{{ groupFor(fa.id) || '—' }}</button></td>
          <td class="mono-expr factor-expression">{{ fa.expression }}</td>
          <td><span class="tag" :class="fa.evaluation_protocol==='v3.0' ? 'green' : 'amber'">{{ fa.evaluation_protocol }}</span></td>
          <td><span class="tag" :class="{green:fa.lifecycle_stage?.includes('live'), blue:fa.lifecycle_stage==='research_pass', amber:fa.lifecycle_stage?.includes('paper'), red:fa.lifecycle_stage==='legacy_unreviewed'}">{{ fa.lifecycle_stage }}</span></td>
          <td><b :class="gradeClass(fa.eligibility?.grade)">{{ fa.eligibility?.grade || '—' }}</b></td>
          <td>{{ f(fa.public?.icir) }}</td><td>{{ f(layerSharpe(fa.gate)) }}</td><td><b>{{ f(fa.public?.score) }}</b></td>
          <td><span class="tag" :class="fa.provenance_status?.includes('invalid') ? 'red' : ''">{{ fa.provenance_status }}</span></td>
        </tr></table>
      <div v-if="!visibleFactors.length" class="selector-empty"><h2>当前筛选没有结果</h2><p>降低筛选条件，或清除相似组筛选。</p></div>
    </div>

    <div class="drawer" v-if="detail">
      <button class="btn close" @click="detail=null">✕ 关闭</button>
      <div class="panel-title-row"><div><h2>{{ detail.factor.name }}</h2><div class="sub">{{ detail.factor.lifecycle_stage }} · {{ detail.factor.provenance_status }}</div></div><div><span class="tag" :class="detail.factor.evaluation_protocol==='v3.0'?'green':'amber'">{{ detail.factor.evaluation_protocol }}</span> <span class="tag amber">NON_PIT</span></div></div>
      <div class="expression-display">
        <label class="latex-toggle"><input type="checkbox" v-model="showLatex" /> 以 Web LaTeX 渲染 DSL</label>
        <div v-if="showLatex" ref="latexEl" class="latex-expression"></div>
        <div v-else class="mono-expr">{{ detail.factor.expression }}</div>
        <div class="sub">DSL: <code>{{ detail.factor.expression }}</code></div>
      </div>
      <p class="sub">{{ detail.factor.hypothesis }}</p>
      <div class="card similarity-detail" v-if="detail.similarity">
        <div class="panel-title-row"><div><h3>结构近邻</h3><span class="sub">{{ detail.similarity.group_id || '单因子组' }} · 不读取未来收益</span></div><span class="tag blue">{{ detail.similarity.nearest?.length || 0 }} 个近邻</span></div>
        <div class="similar-factor-list">
          <button v-for="row in detail.similarity.nearest" :key="row.id" @click="openSimilar(row)"><span>{{ row.name }}</span><code>{{ row.expression }}</code><b>{{ (row.similarity*100).toFixed(0) }}%</b></button>
        </div>
      </div>
      <div v-if="detail.factor.validation?.source_provenance_warning" class="warn-banner">{{ detail.factor.validation.source_provenance_warning }}</div>
      <div class="card audit-controls">
        <div class="panel-title-row"><div><h3>完整 V3 审计</h3><span class="sub">显式读取 HOLDOUT 与 VAULT，并把结果持久化；不会反馈给 Miner。</span></div><button class="btn primary" @click="runAudit" :disabled="auditing">{{ auditing ? '审计中…' : '运行完整审计' }}</button></div>
        <div class="form-row"><div><label>股票池</label><input type="number" v-model.number="auditForm.universe_n" /></div><div><label>持有期</label><select v-model.number="auditForm.horizon"><option :value="1">1日</option><option :value="5">5日</option><option :value="10">10日</option><option :value="20">20日</option></select></div><div><label>基础成本 bps</label><input type="number" v-model.number="auditForm.cost_bps" /></div><div><label>目标资金规模</label><input type="number" v-model.number="auditForm.target_capital" /></div></div>
        <div v-if="auditErr" style="color:var(--red)">{{ auditErr }}</div>
      </div>
      <div class="card" v-if="detail.factor.validation?.layers">
        <div class="panel-title-row"><h3>四层实战指标</h3><div><span class="grade-pill" :class="gradeClass(detail.factor.eligibility?.grade)">{{ detail.factor.eligibility?.grade }}</span> <span class="tag">{{ detail.factor.eligibility?.stage }}</span></div></div>
        <table><tr><th>层</th><th>ICIR</th><th>费后 Sharpe</th><th>年化</th><th>最大回撤</th><th>日均等效换手</th><th>单调性</th><th>压力最差</th></tr>
          <tr v-for="(m,k) in detail.factor.validation.layers" :key="k"><td>{{ k.toUpperCase() }}</td><td>{{ f(m?.icir) }}</td><td>{{ f(layerSharpe(m)) }}</td><td>{{ pct(layerReturn(m)) }}</td><td>{{ pct(m?.net?.max_drawdown) }}</td><td>{{ pct(m?.daily_turnover ?? m?.turnover) }}</td><td>{{ f(m?.monotonicity) }}</td><td>{{ f(worstStress(m)) }}</td></tr>
        </table>
        <div v-if="detail.factor.eligibility?.failure_reasons?.length" class="failure-list"><b>未通过原因</b><ul><li v-for="reason in detail.factor.eligibility.failure_reasons" :key="reason">{{ reason }}</li></ul></div>
      </div>
      <div v-else class="card selector-empty"><h3>尚未完成 V3 全层审计</h3><p>旧评分仅作为历史记录。运行审计后才会生成 F1–F5 等级。</p></div>
      <div class="card"><h3>训练反馈层</h3><table><tr><th>层</th><th>ICIR</th><th>一致性</th><th>费后 Sharpe</th><th>日均等效换手</th><th>研究分</th></tr><tr v-for="(m,k) in {PUBLIC:detail.factor.public,GATE:detail.factor.gate}" :key="k"><td>{{ k }}</td><td>{{ f(m?.icir) }}</td><td>{{ f(m?.era_consistency) }}</td><td>{{ f(layerSharpe(m)) }}</td><td>{{ pct(m?.daily_turnover ?? m?.turnover) }}</td><td>{{ f(m?.score) }}</td></tr></table></div>
      <label>标签（逗号分隔）</label><input v-model="review.tags" placeholder="momentum, quality, low-turnover" /><label>研究备注</label><textarea v-model="review.note" rows="5" placeholder="记录经济机制、已知暴露、失败原因和后续动作"></textarea>
      <div style="margin-top:10px"><button class="btn primary" @click="saveReview">保存研究备注</button><span class="sub" style="margin-left:8px">experiment={{ detail.factor.experiment_id }}</span></div>
    </div>
  </div>`,
  setup() {
    const factors = ref([]), selected = ref([]), detail = ref(null), comparison = ref(null);
    const query = ref(""), status = ref(""), sort = ref("score");
    const groups = ref([]), groupFilter = ref("");
    const groupStats = reactive({ groups: 0, duplicate_groups: 0, redundancy_ratio: 0 });
    const showLatex = ref(true), latexEl = ref(null);
    const review = reactive({ tags: "", note: "" });
    const auditForm = reactive({ universe_n: 500, horizon: 5, cost_bps: 20, target_capital: 10000000 });
    const auditing = ref(false), auditErr = ref("");
    let loadedExperimentVersion = -1;
    const f = v => v == null ? "—" : Number(v).toFixed(3);
    const pct = v => v == null ? "—" : (Number(v) * 100).toFixed(1) + "%";
    const layerSharpe = m => m?.active?.sharpe ?? m?.net?.sharpe ?? m?.long_only_sharpe;
    const layerReturn = m => m?.active?.ann_return ?? m?.net?.ann_return;
    const worstStress = m => m?.cost_stress?.length ? Math.min(...m.cost_stress.map(x=>Number(x.sharpe))) : null;
    const gradeClass = grade => grade === "F5" ? "grade-f5" : grade === "F4" ? "grade-f4" : grade === "F3" ? "grade-f3" : "grade-low";
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
      const [factorData, groupData] = await Promise.all([
        api("/factors?" + params.toString(), { cacheTtl: 1000 }),
        api("/factors/similarity-groups", { cacheTtl: 5000 }),
      ]);
      factors.value = factorData.factors;
      groups.value = groupData.groups || [];
      Object.assign(groupStats, groupData.stats || {});
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
      groups, groupFilter, groupStats, groupFor, familyLabel, showLatex, latexEl,
      review, auditForm, auditing, auditErr, f, pct, layerSharpe, layerReturn,
      worstStress, gradeClass, refresh, toggleAll, open, openSimilar, compare,
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
      <div><span class="tag blue">{{ market==='ashare' ? 'A股' : '美股' }}</span> <span class="tag green">{{ feeLabel }}</span> <span class="tag amber">{{ taskMode==='long_only' ? '纯多头' : '多空' }}</span></div>
    </div>

    <div class="card backtest-config">
      <div class="execution-timeline">
        <span><b>01</b> t 日收盘读取信号</span><i>→</i><span><b>02</b> 创建目标持仓订单</span><i>→</i><span><b>03</b> t+1 开盘成交/拒单</span><i>→</i><span><b>04</b> 收盘逐仓估值与对账</span>
      </div>
      <label>因子 DSL 表达式</label>
      <input v-model="form.expression" placeholder="例: -rank(ts_delta(close, 20))" />
      <div class="backtest-form-grid">
        <div><label>股票池</label><select v-model.number="form.universe_n"><option :value="100">Top100</option><option :value="300">Top300</option><option :value="500">Top500</option><option :value="1000">Top1000</option><option :value="1500">Top1500</option></select></div>
        <div><label>开始日期</label><input type="date" v-model="form.start" /></div>
        <div><label>结束日期</label><input type="date" v-model="form.end" /></div>
        <div><label>初始资金</label><input type="number" v-model.number="form.initial_capital" min="10000" /></div>
        <div><label>调仓步长</label><select v-model.number="form.rebalance_every"><option :value="1">每日</option><option :value="5">每5日</option><option :value="10">每10日</option><option :value="20">每20日</option></select></div>
        <div><label>单边选股比例</label><input type="number" v-model.number="form.top_fraction" min="0.01" max="0.5" step="0.05" /></div>
        <div><label>基础滑点 bps</label><input type="number" v-model.number="form.slippage_bps" min="0" step="0.5" /></div>
        <div><label>最大成交量参与率</label><input type="number" v-model.number="form.max_volume_participation" min="0.01" max="1" step="0.01" /></div>
        <div><label>方向</label><select v-model.number="form.direction"><option :value="1">高值优先</option><option :value="-1">低值优先</option></select></div>
        <div v-if="taskMode==='long_short'"><label>年化借券成本 bps</label><input type="number" v-model.number="form.borrow_cost_bps_annual" min="0" /></div>
      </div>
      <div class="fee-disclosure">
        <b>{{ feeLabel }}</b>
        <span v-if="market==='ashare'">券商佣金万2免5；卖出印花税按历史日期；过户费双向按历史日期。</span>
        <span v-else>每股 $0.005、每单最低 $1、最高成交额 1%；固定费率中的交易规费不重复扣除。</span>
      </div>
      <div class="run-row"><button class="btn primary" @click="run" :disabled="running || !form.expression.trim()">{{ running ? '正在生成事件账本…' : '运行事件回测' }}</button><span v-if="err" class="selector-error inline-error">{{ err }}</span></div>
    </div>

    <template v-if="result">
      <div class="metric-strip backtest-metrics">
        <div class="metric-card accent"><span>费后期末净值</span><b>{{ num(result.stats.final_nav, 4) }}</b><small>{{ money(result.stats.final_nlv) }}</small></div>
        <div class="metric-card"><span>年化 / Sharpe</span><b>{{ pct(result.stats.ann_ret) }}</b><small>Sharpe {{ num(result.stats.sharpe, 2) }}</small></div>
        <div class="metric-card"><span>最大回撤</span><b>{{ pct(result.stats.max_dd) }}</b><small>日均换手 {{ pct(result.stats.avg_daily_turnover) }}</small></div>
        <div class="metric-card"><span>订单 / 成交</span><b>{{ result.stats.orders }} / {{ result.stats.fills }}</b><small>成交率 {{ pct(result.stats.fill_rate) }}</small></div>
        <div class="metric-card"><span>佣金税费</span><b>{{ money(result.stats.commission_and_tax) }}</b><small>滑点 {{ money(result.stats.slippage_cost) }}</small></div>
        <div class="metric-card"><span>账本完整性</span><b :class="result.integrity?.all_pass ? 'ok-text' : 'bad-text'">{{ result.integrity?.all_pass ? 'PASS' : 'FAIL' }}</b><small>{{ result.stats.protocol }}</small></div>
      </div>

      <div class="grid cols-2 backtest-analysis">
        <div class="card"><div class="panel-title-row"><div><h3>净值重放</h3><span class="sub">费后净值与相同成交的无成本代理</span></div><span class="tag">{{ result.curve?.dates?.length || 0 }} 日</span></div><div class="chart" ref="curveEl"></div></div>
        <div class="card integrity-card">
          <div class="panel-title-row"><div><h3>交割单回归检查</h3><span class="sub">API、CSV 与净值共用同一成交账本</span></div><span class="grade-pill" :class="result.integrity?.all_pass ? 'grade-f5' : 'grade-low'">{{ result.integrity?.all_pass ? '全部通过' : '存在异常' }}</span></div>
          <table><tr><th>检查项</th><th>结果</th></tr>
            <tr v-for="(value,key) in result.integrity" :key="key"><td>{{ integrityLabels[key] || key }}</td><td :class="integrityOk(key,value) ? 'ok-text' : 'bad-text'">{{ typeof value==='number' ? num(value,8) : value }}</td></tr>
          </table>
        </div>
      </div>

      <div class="card step-inspector" v-if="result.daily_steps?.length">
        <div class="panel-title-row"><div><h3>逐日步进状态</h3><span class="sub">拖动时间轴查看当日现金、敞口、成交与事件</span></div><span class="tag blue">{{ currentStep?.trade_date }}</span></div>
        <input class="step-range" type="range" min="0" :max="result.daily_steps.length-1" v-model.number="stepCursor" />
        <div class="step-state-grid" v-if="currentStep">
          <div><span>收盘 NLV</span><b>{{ money(currentStep.close_nlv) }}</b></div><div><span>现金</span><b>{{ money(currentStep.cash) }}</b></div><div><span>多头市值</span><b>{{ money(currentStep.long_market_value) }}</b></div><div><span>空头市值</span><b>{{ money(currentStep.short_market_value) }}</b></div><div><span>成交 / 事件</span><b>{{ currentStep.fills }} / {{ currentStep.events }}</b></div><div><span>净敞口</span><b>{{ pct(currentStep.net_exposure) }}</b></div>
        </div>
      </div>

      <div class="card ledger-card">
        <div class="panel-title-row">
          <div><h3>详细交割与事件账本</h3><span class="sub">每笔费用、滑点、现金与成交后持仓均可独立复算</span></div>
          <div class="ledger-actions"><button class="btn" :class="{primary:ledgerTab==='trades'}" @click="switchLedger('trades')">交割单</button><button class="btn" :class="{primary:ledgerTab==='events'}" @click="switchLedger('events')">事件流</button><a v-if="currentId" class="btn-link" :href="'/api/backtests/'+currentId+'/statement.csv'">下载完整 CSV</a></div>
        </div>
        <div class="ledger-scroll" v-if="ledgerTab==='trades'">
          <table><thead><tr><th>成交日</th><th>信号日</th><th>证券</th><th>方向</th><th>成交数量</th><th>基准/成交价</th><th>佣金</th><th>印花税</th><th>过户费</th><th>滑点</th><th>总费用</th><th>成交后现金</th><th>成交后持仓</th></tr></thead>
          <tbody><tr v-for="row in ledgerRows" :key="row.fill_id"><td>{{ row.trade_date }}</td><td>{{ row.signal_date }}</td><td><code>{{ row.symbol }}</code><div class="sub">{{ row.name }}</div></td><td :class="row.side==='BUY'?'ok-text':'bad-text'">{{ row.side }}</td><td>{{ num(row.filled_quantity,2) }}</td><td>{{ num(row.reference_price,4) }} / {{ num(row.fill_price,4) }}</td><td>{{ money(row.commission) }}</td><td>{{ money(row.stamp_duty) }}</td><td>{{ money(row.transfer_fee) }}</td><td>{{ money(row.slippage_cost) }}</td><td><b>{{ money(row.total_fees) }}</b></td><td>{{ money(row.cash_after) }}</td><td>{{ num(row.position_after,2) }}</td></tr></tbody></table>
        </div>
        <div class="ledger-scroll" v-else>
          <table><thead><tr><th>#</th><th>日期</th><th>阶段</th><th>事件</th><th>证券</th><th>订单</th><th>说明</th></tr></thead><tbody><tr v-for="row in ledgerRows" :key="row.seq"><td>{{ row.seq }}</td><td>{{ row.trade_date }}</td><td>{{ row.phase }}</td><td>{{ row.event_type }}</td><td><code>{{ row.symbol }}</code></td><td>{{ row.order_id }}</td><td>{{ row.message }}</td></tr></tbody></table>
        </div>
        <div class="ledger-pager"><span>{{ ledgerPage.offset+1 }}–{{ Math.min(ledgerPage.offset+ledgerRows.length, ledgerPage.total) }} / {{ ledgerPage.total }}</span><button class="btn" @click="pageLedger(-1)" :disabled="ledgerPage.offset===0">上一页</button><button class="btn" @click="pageLedger(1)" :disabled="ledgerPage.offset+ledgerPage.limit>=ledgerPage.total">下一页</button></div>
      </div>
    </template>

    <div class="card history-card">
      <div class="panel-title-row"><div><h3>历史回测档案</h3><span class="sub">旧向量回测保留但标记 legacy；新记录可重放交割单</span></div><button class="btn" @click="loadHistory">刷新</button></div>
      <table><tr><th>#</th><th>协议</th><th>状态</th><th>表达式</th><th>区间</th><th>Sharpe</th><th>年化</th><th>回撤</th><th>交割检查</th><th>时间</th></tr>
        <tr v-for="b in history" :key="b.id" class="clickable" @click="openHistory(b)">
          <td>{{ b.id }}</td><td><span class="tag" :class="b.protocol==='step_event_v1'?'green':'amber'">{{ b.protocol }}</span></td><td>{{ b.status }}</td><td class="mono-expr factor-expression">{{ b.params.expression }}</td><td class="sub">{{ b.params.start }}~{{ b.params.end }}</td><td>{{ num(b.stats?.sharpe,2) }}</td><td>{{ pct(b.stats?.ann_ret) }}</td><td>{{ pct(b.stats?.max_dd) }}</td><td :class="b.integrity?.all_pass?'ok-text':'bad-text'">{{ b.integrity?.all_pass ? 'PASS' : '—' }}</td><td class="sub">{{ b.created_at?.slice(0,16) }}</td>
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
    });
    const taskMode = ref("long_only"), market = ref("us");
    const result = ref(null), history = ref([]), err = ref(""), running = ref(false);
    const currentId = ref(null), curveEl = ref(null), stepCursor = ref(0);
    const ledgerTab = ref("trades"), ledgerRows = ref([]);
    const ledgerPage = reactive({ offset:0, limit:200, total:0 });
    let loadedExperimentVersion = -1;
    const feeLabel = computed(() => market.value === "ashare" ? "万2免5" : "IBKR Pro Fixed");
    const currentStep = computed(() => result.value?.daily_steps?.[stepCursor.value] || null);
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
      ledger_source_of_truth:"净值是否来自账本",
      all_pass:"总检查",
    };
    const num = (value, digits=4) => value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
    const pct = value => value == null || Number.isNaN(Number(value)) ? "—" : (Number(value)*100).toFixed(2)+"%";
    const money = value => value == null || Number.isNaN(Number(value)) ? "—" : new Intl.NumberFormat("zh-CN",{style:"currency",currency:market.value==="ashare"?"CNY":"USD",maximumFractionDigits:2}).format(Number(value));
    const integrityOk = (key, value) => {
      if (key === "statement_rows") return Number(value) >= 0;
      if (key.endsWith("_max_error")) return Number(value) <= 1e-5;
      return value === 0 || value === true;
    };
    async function run() {
      running.value = true; err.value = "";
      try {
        result.value = await api("/backtest", { method: "POST", body: { ...form } });
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
      form.initial_capital = meta.evaluation_config?.target_capital ?? (market.value==="ashare"?10000000:1000000);
      form.slippage_bps = market.value === "ashare" ? 5 : 2;
      form.borrow_cost_bps_annual = meta.evaluation_config?.borrow_cost_bps_annual ?? 0;
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
    onActivated(ensureContext);
    return {
      form, taskMode, market, feeLabel, result, history, err, running, run,
      curveEl, currentId, currentStep, stepCursor, ledgerTab, ledgerRows,
      ledgerPage, integrityLabels, num, pct, money, loadHistory, openHistory,
      integrityOk, switchLedger, pageLedger,
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
        <div><label>市场</label><select v-model="taskForm.market" @change="syncMarketDefaults"><option value="ashare">A股</option><option value="us">美股</option></select></div>
        <div><label>持仓约束</label><select v-model="taskForm.portfolio_mode"><option value="long_only">纯多头</option><option value="long_short" :disabled="taskForm.market==='ashare'">多空</option></select></div>
        <div><label>信号方向</label><select v-model.number="taskForm.direction"><option :value="1">高分做多{{ taskForm.portfolio_mode==='long_short' ? ' / 低分做空' : '' }}</option><option :value="-1">低分做多{{ taskForm.portfolio_mode==='long_short' ? ' / 高分做空' : '（仍为纯多头）' }}</option></select></div>
        <div><label>引擎版本</label><select v-model="taskForm.engine_mode"><option value="v2">V2 MinerTemplate（当前）</option></select></div>
        <div style="flex:3"><label>面板路径（可选）</label><input v-model="taskForm.panel_glob" placeholder="留空使用服务默认面板" /></div>
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
        <div style="align-self:end"><button class="btn primary" @click="createTask">按 V3 创建任务</button></div>
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
          <div><label>内层挖掘用</label><select v-model="llm.inner_provider"><option value="">(未配置=随机基线)</option><option v-for="p in llm.providers" :key="p.name" :value="p.name">{{ p.name }}</option></select></div>
          <div><label>外层元优化用</label><select v-model="llm.outer_provider"><option value="">(未配置=随机扰动)</option><option v-for="p in llm.providers" :key="p.name" :value="p.name">{{ p.name }}</option></select></div>
        </div>
      </div>
      <div class="card">
        <h3>引擎参数</h3>
        <div class="form-row">
          <div><label>每外层步内层评估预算</label><input v-model.number="eng.inner_budget_per_outer_step" type="number" /></div>
          <div><label>外层接受阈值 ε</label><input v-model.number="eng.outer_accept_epsilon" type="number" step="0.01" /></div>
        </div>
        <div class="form-row">
          <div><label>在位者重测周期 (步)</label><input v-model.number="eng.incumbent_remeasure_every" type="number" /></div>
        </div>
        <h3 style="margin-top:16px">任务集</h3>
        <table>
          <tr><th>名称</th><th>股票池</th><th>持有期</th><th>成本bps</th></tr>
          <tr v-for="t in eng.tasks" :key="t.name"><td>{{ t.name }}</td><td>{{ t.universe_n }}</td><td>{{ t.horizon }}日</td><td>{{ t.cost_bps }}</td></tr>
        </table>
        <div class="sub" style="margin-top:8px">挖掘评分仅使用 INNER_PUBLIC + META_TRAIN；HOLDOUT 与 VAULT 只允许通过因子库里的显式 V3 审计读取，绝不进入循环提示词。</div>
        <div class="protocol-card"><b>Evaluation Protocol {{ evalProtocol.version || 'v3.0' }}</b><span>真实目标权重换手 · 费后收益 · 多头/空头拆分 · 成本压力 · OOS 生命周期</span><small>{{ evalProtocol.policy_label }}</small></div>
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
    const msg = ref(""), saved = ref("");
    const taskForm = reactive({
      name: "", description: "", market: "ashare", portfolio_mode: "long_only",
      direction: 1, engine_mode: "v2", panel_glob: "", top_fraction: 0.20,
      base_cost_bps: 20, stress_cost_bps: "10,20,35,50",
      borrow_cost_bps_annual: 0, target_capital: 10000000,
      min_oos_sharpe: 0.5, max_drawdown: 0.35, max_daily_turnover: 0.35,
    });
    const taskMsg = ref(""), taskOk = ref(false);
    async function load() {
      const s = await api("/settings");
      Object.assign(llm, s.llm_providers);
      Object.assign(eng, s.engine_config);
      Object.assign(evalProtocol, s.evaluation_protocol || {});
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
        await api("/experiments", { method: "POST", body: {
          name: taskForm.name, description: taskForm.description,
          research_config: { market: taskForm.market, portfolio_mode: taskForm.portfolio_mode,
            direction: taskForm.direction, engine_mode: taskForm.engine_mode,
            panel_glob: taskForm.panel_glob || undefined, evaluation_protocol: "v3.0",
            evaluation_config: {
              top_fraction: taskForm.top_fraction, tail_fraction: taskForm.top_fraction,
              base_cost_bps: taskForm.base_cost_bps, stress_cost_bps: stress,
              borrow_cost_bps_annual: taskForm.borrow_cost_bps_annual,
              target_capital: taskForm.target_capital, min_oos_sharpe: taskForm.min_oos_sharpe,
              max_drawdown: taskForm.max_drawdown, max_daily_turnover: taskForm.max_daily_turnover,
            }},
        }});
        taskOk.value = true; taskMsg.value = "研究任务已创建，可在“实验”页启动";
        taskForm.name = ""; taskForm.description = "";
      } catch (e) { taskOk.value = false; taskMsg.value = "创建失败: " + e.message; }
    }
    function syncMarketDefaults() {
      if (taskForm.market === "ashare") {
        taskForm.portfolio_mode = "long_only"; taskForm.base_cost_bps = 20;
        taskForm.stress_cost_bps = "10,20,35,50"; taskForm.borrow_cost_bps_annual = 0;
        taskForm.target_capital = 10000000; taskForm.max_daily_turnover = 0.35;
      } else {
        taskForm.portfolio_mode = "long_short"; taskForm.base_cost_bps = 15;
        taskForm.stress_cost_bps = "5,15,25,40"; taskForm.borrow_cost_bps_annual = 300;
        taskForm.target_capital = 1000000; taskForm.max_daily_turnover = 0.50;
      }
    }
    onMounted(load);
    return { llm, eng, evalProtocol, save, msg, saved, taskForm, taskMsg, taskOk, createTask, syncMarketDefaults };
  },
};

/* ============ 实验管理 ============ */
const ExperimentsView = {
  template: `
  <div>
    <div class="card" style="margin-bottom:14px">
      <h3>研究任务与并行运行</h3>
      <div class="form-row">
        <div style="flex:1"><label>名称</label><input v-model="form.name" placeholder="如: 实验2-修复评估器" /></div>
        <div style="flex:2"><label>描述</label><input v-model="form.description" placeholder="研究假设 / 变更点" /></div>
        <div style="align-self:flex-end"><button class="btn primary" @click="create">创建（基础）</button></div>
      </div>
      <div v-if="err" style="color:var(--red); margin-top:8px">{{ err }}</div>
    </div>
    <div class="card">
      <h3>研究任务列表 ({{ exps.length }})</h3>
      <table>
        <tr><th>#</th><th>名称</th><th>市场 / 约束</th><th>协议</th><th>任务状态</th><th>运行态</th><th>因子</th><th>节点</th><th>外层步</th><th style="min-width:250px">操作</th></tr>
        <tr v-for="e in exps" :key="e.id" :style="{background: e.active ? '#1c2733' : ''}">
          <td>{{ e.id }}</td>
          <td>
            <input v-if="editing===e.id" v-model="editForm.name" style="width:180px" />
            <template v-else><b>{{ e.name }}</b> <span v-if="e.active" class="tag green">活动</span></template>
          </td>
          <td><span class="tag blue">{{ e.research_config?.market==='ashare' ? 'A股' : '美股' }}</span> <span class="sub">{{ e.research_config?.portfolio_mode==='long_only' ? '纯多头' : '多空' }} · {{ Number(e.research_config?.direction || 1)===1 ? '高值偏多' : '低值偏多' }}</span></td>
          <td><span class="tag" :class="e.research_config?.evaluation_protocol==='v3.0'?'green':'amber'">{{ e.research_config?.evaluation_protocol || 'legacy' }}</span><div v-if="e.research_config?.provenance_warning" class="provenance-dot" :title="e.research_config.provenance_warning">来源警告</div></td>
          <td><span class="tag" :class="{green: e.status==='open', amber: e.status==='archived'}">{{ e.status }}</span></td>
          <td><span class="tag" :class="{green: runtime[e.id]?.state==='running', amber: runtime[e.id]?.state==='starting', red: runtime[e.id]?.state==='stopped'}">{{ runtime[e.id]?.state || 'stopped' }}</span></td>
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
    const form = reactive({ name: "", description: "" });
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
    async function create() {
      err.value = "";
      try { await api("/experiments", { method: "POST", body: { ...form } }); form.name = ""; form.description = ""; refresh(); }
      catch (e) { err.value = e.message; }
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
    return { exps, runtime, form, editForm, editing, err, create, startEdit, saveEdit, setStatus, activate, start, stop };
  },
};

/* ============ App ============ */
const App = {
  components: { Dashboard, ResearchTree, FactorLibrary, FactorLibraryWorkbench, BacktestView, SettingsView, ExperimentsView },
  template: `
  <div class="topbar">
    <div class="logo">⚒ FactorFactory</div>
    <div class="tabs">
      <button v-for="t in tabs" :key="t.id" :class="{active: tab===t.id}" @click="selectTab(t.id)">{{ t.label }}</button>
    </div>
    <div class="spacer"></div>
    <span v-if="appState.switchMessage" class="switch-message">{{ appState.switchMessage }}</span>
    <select v-model="selExp" @change="switchExp" :disabled="appState.switching" style="margin-right:10px; max-width:220px" title="切换活动研究任务">
      <option v-for="e in exps" :key="e.id" :value="e.id" :disabled="e.status==='archived' && !e.active">
        {{ e.name }}{{ e.status==='archived' ? ' (归档)' : '' }}
      </option>
    </select>
    <span class="state-badge" :class="engState==='running' ? 'state-running' : 'state-stopped'">● {{ engState }}</span>
  </div>
  <div class="main">
    <div v-if="appState.switching" class="task-switch-overlay"><div class="loading-ring"></div><span>正在切换任务上下文</span></div>
    <KeepAlive :max="7"><component :is="activeComponent" :key="tab" /></KeepAlive>
  </div>`,
  setup() {
    const savedTab = localStorage.getItem("factorfactory.tab");
    const tab = ref(savedTab || "dash");
    appState.activeTab = tab.value;
    const tabs = [
      { id: "dash", label: "总览" }, { id: "tree", label: "研发树" },
      { id: "factors", label: "因子库" }, { id: "screener", label: "选股器" }, { id: "backtest", label: "回测" },
      { id: "exps", label: "实验" }, { id: "settings", label: "设置" },
    ];
    const engState = ref("…");
    const exps = ref([]), selExp = ref(null);
    let timer = null, polling = false;
    const activeComponent = computed(() => ({
      dash: Dashboard, tree: ResearchTree, factors: FactorLibraryWorkbench,
      screener: ScreenerView, backtest: BacktestView, exps: ExperimentsView,
      settings: SettingsView,
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
    onMounted(() => { poll(); loadExps(); timer = setInterval(poll, 5000); });
    onUnmounted(() => clearInterval(timer));
    return { tab, tabs, engState, exps, selExp, switchExp, selectTab, activeComponent, appState };
  },
};

const ScreenerView = {
  template: `
  <section class="selector-page">
    <div class="selector-heading">
      <div>
        <div class="eyebrow">SINGLE-PASS POLARS / CACHED CROSS-SECTION</div>
        <h1>高性能多因子选股器</h1>
        <p>一次懒执行计划计算所有因子，自动裁剪所需历史窗口并缓存完全相同的截面快照。</p>
      </div>
      <span class="tag amber">研究用途 · 非交易批准</span>
    </div>

      <div class="selector-toolbar card">
      <div class="selector-field selector-dsl">
        <label>直接 DSL 选股（可选）</label>
        <input v-model="directExpr" @blur="inspectDsl" @keyup.enter="inspectDsl" placeholder="例如：-rank(ts_delta(close, 20))" />
        <small v-if="dslInfo" class="dsl-hint">合法 · 需 {{ dslInfo.required_history }} 个交易日 · 复杂度 {{ dslInfo.complexity }}</small>
      </div>
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
        <label>输出方向</label>
        <select v-model="outputDirection"><option value="top">高分端</option><option value="bottom">低分端</option><option value="both">两端</option></select>
      </div>
      <div class="selector-actions">
        <button class="btn" @click="loadFactors" :disabled="loading">刷新因子</button>
        <button class="btn primary" @click="run" :disabled="loading || !canRun">
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
                <span class="factor-meta">{{ f.group || '单组' }} · {{ f.grade || '未审计' }} · 研究分 {{ formatScore(f.score) }}</span>
              </span>
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
          <div class="loading-ring"></div><h2>正在计算截面排名</h2><p>正在按股票池过滤数据并合并 {{ directExpr.trim() ? 1 : enabledCount }} 个因子。</p>
        </div>
        <template v-if="result && !loading">
          <div class="result-summary">
            <div class="result-title"><div><div class="eyebrow">SCREENING SNAPSHOT</div><h2>{{ result.date }} · 综合排名</h2><small v-if="result.date_adjusted" class="sub">请求日 {{ result.requested_date }} 非交易日或超出面板，已回退到最近交易日</small></div><span class="tag blue">已完成</span></div>
          <div class="metric-strip selector-metrics">
              <div class="metric-card"><span>股票池</span><b>Top {{ result.universe_n || univN }}</b><small>按 60 日成交额</small></div>
              <div class="metric-card"><span>启用因子</span><b>{{ result.factor_count }}</b><small>加权截面排名</small></div>
              <div class="metric-card"><span>输出数量</span><b>{{ result.stocks.length }}</b><small>候选清单</small></div>
              <div class="metric-card accent"><span>{{ result.direction === 'bottom' ? '最低综合分' : '首位综合分' }}</span><b>{{ topScore }}</b><small>{{ result.expression_mode ? '单条 DSL 排名' : '相对排序分数' }}</small></div>
              <div class="metric-card"><span>有效截面</span><b>{{ result.eligible_count }}</b><small>完整因子交集</small></div>
              <div class="metric-card" :class="{accent:result.performance?.cache_hit}"><span>计算性能</span><b>{{ result.performance?.elapsed_ms }} ms</b><small>{{ result.performance?.cache_hit ? '命中缓存' : '单计划实时计算' }}</small></div>
            </div>
          </div>
          <div class="card result-table-card">
            <div class="panel-title-row"><div><h2>候选清单</h2><span class="sub">点击股票查看每个因子的原值、截面名次与分数贡献</span></div><span class="tag">{{ result.date }} · 历史起点 {{ result.history_start }}</span></div>
            <table class="result-table"><thead><tr><th>排名</th><th>端</th><th>证券</th><th>名称</th><th>原始收盘</th><th>成交额</th><th>综合分</th><th>相对位置</th></tr></thead>
              <tbody><tr v-for="s in result.stocks" :key="s.rank" class="clickable" :class="{'top-pick':s.rank<=10,'selected-stock':selectedStock?.ts_code===s.ts_code}" @click="selectedStock=s"><td><span class="rank-number">{{ String(s.rank).padStart(2,"0") }}</span></td><td><span class="tag" :class="s.side==='top'?'green':'red'">{{ s.side }}</span></td><td><code class="ticker">{{ s.ts_code }}</code></td><td>{{ s.name || "—" }}</td><td>{{ formatPrice(s.raw_close) }}</td><td>{{ compactAmount(s.amount) }}</td><td><b>{{ formatScore(s.score) }}</b></td><td><span class="rank-bar"><i :style="{ width: rankWidth(s) }"></i></span></td></tr></tbody>
            </table>
          </div>
          <div class="card contribution-card" v-if="selectedStock">
            <div class="panel-title-row"><div><h2>{{ selectedStock.ts_code }} · 排名归因</h2><span class="sub">{{ selectedStock.name }} · 综合分 {{ formatScore(selectedStock.score) }}</span></div><button class="btn" @click="selectedStock=null">关闭</button></div>
            <table><tr><th>因子表达式</th><th>方向</th><th>权重</th><th>因子原值</th><th>截面分</th><th>贡献</th></tr><tr v-for="row in selectedStock.components" :key="row.expression"><td class="mono-expr">{{ row.expression }}</td><td>{{ row.direction===1?'正向':'反向' }}</td><td>{{ (row.weight*100).toFixed(1) }}%</td><td>{{ formatScore(row.value) }}</td><td>{{ formatScore(row.rank_score) }}</td><td><b>{{ formatScore(row.contribution) }}</b></td></tr></table>
          </div>
          <div class="selector-disclaimer"><span>ⓘ</span> 这是基于当前研究面板的横截面排序。数据为 non-PIT 当前成分股回看历史，结果不等同于可交易信号。</div>
        </template>
      </main>
    </div>
  </section>`,
  setup() {
    const date = ref(new Date().toISOString().slice(0,10));
    const univN = ref(500); const topN = ref(50);
    const factors = ref([]);
    const groups = ref([]);
    const factorSearch = ref("");
    const factorGroup = ref("");
    const directExpr = ref("");
    const dslInfo = ref(null);
    const outputDirection = ref("top");
    const result = ref(null); const error = ref(""); const loading = ref(false);
    const selectedStock = ref(null);
    let loadedExperimentVersion = -1;
    const enabledCount = computed(() => factors.value.filter(f=>f.enabled).length);
    const canRun = computed(() => Boolean(directExpr.value.trim()) || enabledCount.value > 0);
    const totalWeight = computed(() => factors.value.filter(f=>f.enabled).reduce((sum, f) => sum + (Number(f.weight) || 0), 0));
    const topScore = computed(() => result.value?.stocks?.length ? formatScore(result.value.stocks[0].score) : "—");
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
      if (!result.value?.stocks?.length) return "0%";
      return `${((result.value.stocks.length - stock.rank + 1) / result.value.stocks.length) * 100}%`;
    }

    async function loadFactors() {
      try {
        error.value = "";
        const [d, similarity] = await Promise.all([
          api("/factors?sort=grade", { cacheTtl: 1000 }),
          api("/factors/similarity-groups", { cacheTtl: 1000 }),
        ]);
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
      } catch(e) { error.value="加载因子失败: "+e.message; }
    }
    function ensureFresh() {
      if (loadedExperimentVersion !== appState.experimentVersion) loadFactors();
    }

    function selectAll() { filteredFactors.value.forEach(f => { f.enabled = true; }); }
    function clearAll() { factors.value.forEach(f => { f.enabled = false; }); }

    async function inspectDsl() {
      const expression = directExpr.value.trim();
      dslInfo.value = null;
      if (!expression) return;
      try {
        dslInfo.value = await api("/dsl/inspect", {
          method: "POST",
          body: { expression },
        });
        error.value = "";
      } catch (e) {
        error.value = `DSL 检查失败: ${e.message}`;
      }
    }

    async function run() {
      const enabled = factors.value.filter(f=>f.enabled);
      if (!enabled.length && !directExpr.value.trim()) { error.value="请至少选择一个因子或输入 DSL 表达式"; return; }
      loading.value=true; error.value=""; result.value=null; selectedStock.value=null;
      try {
        const r = await api("/screener", { method:"POST", body:{
          expression: directExpr.value.trim() || undefined,
          factors: enabled.map(f=>({expression:f.expression,weight:f.weight,direction:f.direction})),
          date: date.value, universe_n: univN.value, top_n: topN.value, direction:outputDirection.value
        }});
        result.value = r;
      } catch(e) { error.value = "选股失败: "+e.message; }
      finally { loading.value=false; }
    }

    watch(directExpr, () => { dslInfo.value = null; });
    watch(() => appState.experimentVersion, () => {
      result.value=null;
      selectedStock.value=null;
      directExpr.value="";
      factorSearch.value="";
      factorGroup.value="";
      if (appState.activeTab === "screener") ensureFresh();
    });
    onActivated(ensureFresh);
    return {
      date, univN, topN, directExpr, dslInfo, outputDirection,
      factors, groups, factorSearch, factorGroup, filteredFactors,
      result, selectedStock, error, loading, enabledCount, canRun,
      totalWeight, topScore, formatScore, formatPrice, compactAmount,
      rankWidth, selectAll, clearAll, inspectDsl, run, loadFactors,
    };
  },
};
App.components.ScreenerView = ScreenerView;
createApp(App).mount("#app");
