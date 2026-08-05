/* USStockFactorFactory 前端 — Vue3 全局构建 + ECharts */
const {
  createApp, ref, reactive, computed, onMounted, onUnmounted,
  onActivated, onDeactivated, watch, nextTick,
} = Vue;

const appState = reactive({
  experimentId: null,
  experimentVersion: 0,
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
    watch(() => appState.experimentVersion, refresh);
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
      selected.value = null; picked.value = null; refresh();
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
      <div class="panel-title-row"><div><div class="eyebrow">EVALUATION PROTOCOL V3</div><h1>因子研究资产库</h1><span class="sub">搜索分与实战准入分离；HOLDOUT / VAULT 仅在显式审计时读取。</span></div><div><span class="tag blue">{{ factors.length }} 条记录</span> <span class="tag amber">NON_PIT_RESEARCH</span></div></div>
      <div class="form-row" style="margin-top:14px">
        <input style="flex:3" v-model="query" @keyup.enter="refresh" placeholder="搜索名称、表达式、经济学假设…" />
        <select v-model="status"><option value="">全部生命周期</option><option value="discovery_only">F1 · discovery_only</option><option value="research_pass">F2 · research_pass</option><option value="oos_pass">F3 · oos_pass</option><option value="paper_candidate">F4 · paper_candidate</option><option value="live_candidate_non_pit">F5 · live_candidate_non_pit</option><option value="legacy_unreviewed">旧协议未审计</option><option value="invalid_provenance">来源无效</option><option value="configuration_changed_requires_reaudit">配置变更待复审</option></select>
        <select v-model="sort"><option value="score">按研究分</option><option value="grade">按实战等级</option><option value="icir">按 ICIR</option><option value="created">按最新</option></select>
        <button class="btn" @click="refresh">刷新</button><button class="btn primary" @click="compare" :disabled="selected.length<2">比较 {{ selected.length }} 个</button>
      </div>
      <div class="sub" style="margin-top:10px">V3 使用真实目标权重换手、费后组合收益、成本压力、分层单调性和跨时期稳定性；旧协议数据不会自动升级。</div>
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
      <div class="panel-title-row"><h2>研究资产</h2><span class="sub">点击行立即读取已存结果；完整审计由用户显式触发</span></div>
      <table><tr><th><input type="checkbox" @change="toggleAll" /></th><th>名称</th><th>表达式</th><th>协议</th><th>生命周期</th><th>等级</th><th>PUB ICIR</th><th>GATE Sharpe</th><th>研究分</th><th>来源</th></tr>
        <tr v-for="fa in factors" :key="fa.id" class="clickable" @click="open(fa)">
          <td @click.stop><input type="checkbox" :value="fa.id" v-model="selected" /></td><td><b>{{ fa.name }}</b></td><td class="mono-expr factor-expression">{{ fa.expression }}</td>
          <td><span class="tag" :class="fa.evaluation_protocol==='v3.0' ? 'green' : 'amber'">{{ fa.evaluation_protocol }}</span></td>
          <td><span class="tag" :class="{green:fa.lifecycle_stage?.includes('live'), blue:fa.lifecycle_stage==='research_pass', amber:fa.lifecycle_stage?.includes('paper'), red:fa.lifecycle_stage==='legacy_unreviewed'}">{{ fa.lifecycle_stage }}</span></td>
          <td><b :class="gradeClass(fa.eligibility?.grade)">{{ fa.eligibility?.grade || '—' }}</b></td>
          <td>{{ f(fa.public?.icir) }}</td><td>{{ f(layerSharpe(fa.gate)) }}</td><td><b>{{ f(fa.public?.score) }}</b></td>
          <td><span class="tag" :class="fa.provenance_status?.includes('invalid') ? 'red' : ''">{{ fa.provenance_status }}</span></td>
        </tr></table>
      <div v-if="!factors.length" class="selector-empty"><h2>当前筛选没有结果</h2><p>降低筛选条件，或等待任务产生新的因子。</p></div>
    </div>

    <div class="drawer" v-if="detail">
      <button class="btn close" @click="detail=null">✕ 关闭</button>
      <div class="panel-title-row"><div><h2>{{ detail.factor.name }}</h2><div class="sub">{{ detail.factor.lifecycle_stage }} · {{ detail.factor.provenance_status }}</div></div><div><span class="tag" :class="detail.factor.evaluation_protocol==='v3.0'?'green':'amber'">{{ detail.factor.evaluation_protocol }}</span> <span class="tag amber">NON_PIT</span></div></div>
      <div class="mono-expr">{{ detail.factor.expression }}</div><p class="sub">{{ detail.factor.hypothesis }}</p>
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
    const review = reactive({ tags: "", note: "" });
    const auditForm = reactive({ universe_n: 500, horizon: 5, cost_bps: 20, target_capital: 10000000 });
    const auditing = ref(false), auditErr = ref("");
    const f = v => v == null ? "—" : Number(v).toFixed(3);
    const pct = v => v == null ? "—" : (Number(v) * 100).toFixed(1) + "%";
    const layerSharpe = m => m?.active?.sharpe ?? m?.net?.sharpe ?? m?.long_only_sharpe;
    const layerReturn = m => m?.active?.ann_return ?? m?.net?.ann_return;
    const worstStress = m => m?.cost_stress?.length ? Math.min(...m.cost_stress.map(x=>Number(x.sharpe))) : null;
    const gradeClass = grade => grade === "F5" ? "grade-f5" : grade === "F4" ? "grade-f4" : grade === "F3" ? "grade-f3" : "grade-low";
    async function refresh() {
      const params = new URLSearchParams({ limit: "500", sort: sort.value });
      if (query.value) params.set("q", query.value); if (status.value) params.set("lifecycle", status.value);
      factors.value = (await api("/factors?" + params.toString(), { cacheTtl: 1000 })).factors;
    }
    function toggleAll(e) { selected.value = e.target.checked ? factors.value.map(fa=>fa.id) : []; }
    async function open(fa) {
      detail.value = await api(`/factors/${fa.id}/detail`, { cacheTtl: 2000 });
      review.tags = (detail.value.factor.research_meta?.tags || []).join(", ");
      review.note = detail.value.factor.research_meta?.note || "";
      const defaults = detail.value.audit_defaults || {};
      auditForm.universe_n = detail.value.factor.research_meta?.last_audit_universe_n || defaults.universe_n || 500;
      auditForm.horizon = detail.value.factor.research_meta?.last_audit_horizon || defaults.horizon || 5;
      auditForm.cost_bps = defaults.cost_bps ?? 15;
      auditForm.target_capital = defaults.target_capital ?? 1000000;
    }
    async function compare() { comparison.value = await api("/factors/compare", { method:"POST", body:{ factor_ids:selected.value } }); }
    async function runAudit() {
      auditing.value = true; auditErr.value = "";
      try {
        detail.value = await api(`/factors/${detail.value.factor.id}/audit`, { method:"POST", body:{ ...auditForm } });
        invalidateApiCache(); await refresh();
      } catch (e) { auditErr.value = e.message; }
      finally { auditing.value = false; }
    }
    async function saveReview() {
      await api(`/factors/${detail.value.factor.id}/review`, { method:"PATCH", body:{ tags:review.tags.split(","), note:review.note } });
      detail.value.factor.research_meta = { ...detail.value.factor.research_meta, tags:review.tags.split(",").filter(Boolean), note:review.note };
      invalidateApiCache(); refresh();
    }
    watch(() => appState.experimentVersion, () => {
      selected.value = []; detail.value = null; comparison.value = null; refresh();
    });
    onMounted(refresh);
    return { factors, selected, detail, comparison, query, status, sort, review, auditForm, auditing, auditErr, f, pct, layerSharpe, layerReturn, worstStress, gradeClass, refresh, toggleAll, open, compare, runAudit, saveReview };
  },
};

/* ============ 回测 ============ */
const BacktestView = {
  template: `
  <div>
    <div class="card" style="margin-bottom:14px">
      <h3>手动回测</h3>
      <label>因子表达式</label>
      <input v-model="form.expression" placeholder="例: -rank(ts_delta(close, 20))" />
      <div class="form-row">
        <div><label>股票池</label><select v-model.number="form.universe_n"><option :value="500">Top500</option><option :value="1500">Top1500</option></select></div>
        <div><label>开始</label><input v-model="form.start" /></div>
        <div><label>结束</label><input v-model="form.end" /></div>
        <div><label>成本 bps</label><input v-model.number="form.cost_bps" type="number" /></div>
        <div><label>方向</label><select v-model.number="form.direction"><option :value="1">正向</option><option :value="-1">反向</option></select></div>
        <div><label>任务持仓约束</label><span class="tag blue">{{ taskMode==='long_only' ? '纯多头' : '多空' }}</span><div class="sub">继承当前研究任务</div></div>
      </div>
      <div style="margin-top:12px"><button class="btn primary" @click="run" :disabled="running">{{ running ? '回测中…' : '运行回测' }}</button>
        <span v-if="err" style="color:var(--red); margin-left:12px">{{ err }}</span></div>
    </div>
    <div class="grid cols-2" v-if="result">
      <div class="card">
        <h3>净值曲线 (费后)</h3>
        <div class="chart" ref="curveEl"></div>
      </div>
      <div class="card">
        <h3>绩效统计</h3>
        <table>
          <tr v-for="(v,k) in result.stats" :key="k"><td style="color:var(--muted)">{{ labels[k] || k }}</td><td><b>{{ typeof v==='number' ? v.toFixed(4) : v }}</b></td></tr>
        </table>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h3>历史回测</h3>
      <table>
        <tr><th>#</th><th>表达式</th><th>区间</th><th>Sharpe</th><th>年化</th><th>最大回撤</th><th>时间</th></tr>
        <tr v-for="b in history" :key="b.id">
          <td>{{ b.id }}</td>
          <td class="mono-expr" style="max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">{{ b.params.expression }}</td>
          <td class="sub">{{ b.params.start }}~{{ b.params.end }}</td>
          <td>{{ b.stats?.sharpe?.toFixed(2) }}</td><td>{{ (b.stats?.ann_ret*100)?.toFixed(1) }}%</td><td>{{ (b.stats?.max_dd*100)?.toFixed(1) }}%</td>
          <td class="sub">{{ b.created_at?.slice(5,16) }}</td>
        </tr>
      </table>
    </div>
  </div>`,
  setup() {
    const form = reactive({ expression: "-rank(ts_delta(close, 20))", universe_n: 500, start: "2015-01-01", end: "2024-12-31", cost_bps: 15, direction: 1, mode: null });
    const taskMode = ref("long_only");
    const result = ref(null), history = ref([]), err = ref(""), running = ref(false);
    const curveEl = ref(null);
    const labels = { days: "交易日数", ann_ret: "年化收益", ann_vol: "年化波动", sharpe: "Sharpe", max_dd: "最大回撤", avg_daily_turnover: "日均双边交易额/净值", avg_one_way_turnover: "日均单边换手", final_nav: "期末净值" };
    async function run() {
      running.value = true; err.value = "";
      try {
        result.value = await api("/backtest", { method: "POST", body: { ...form } });
        nextTick(drawCurve);
        loadHistory();
      } catch (e) { err.value = e.message; }
      running.value = false;
    }
    function drawCurve() {
      if (!curveEl.value || !result.value) return;
      const c = result.value.curve;
      mountChart(curveEl.value, {
        ...DARK,
        grid: { left: 55, right: 20, top: 20, bottom: 40 },
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: c.dates, axisLabel: { color: "#8b949e" } },
        yAxis: { type: "value", scale: true, splitLine: { lineStyle: { color: "#21262d" } } },
        dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 6 }],
        series: [{ type: "line", data: c.equity, showSymbol: false, lineStyle: { color: "#3fb950", width: 1.5 }, areaStyle: { color: "rgba(63,185,80,0.08)" } }],
      });
    }
    async function loadHistory() { history.value = (await api("/backtests", { cacheTtl: 1000 })).backtests; }
    async function loadContext() {
      result.value = null;
      const meta = await api("/meta", { cacheTtl: 1500 });
      taskMode.value = meta.portfolio_mode || "long_only";
      form.mode = taskMode.value;
      form.cost_bps = meta.evaluation_config?.base_cost_bps ?? form.cost_bps;
      await loadHistory();
    }
    watch(() => appState.experimentVersion, loadContext);
    onMounted(loadContext);
    return { form, taskMode, result, history, err, running, run, curveEl, labels };
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
    async function refresh() {
      const [experiments, obs] = await Promise.all([
        api("/experiments", { cacheTtl: 800 }),
        api("/observability", { cacheTtl: 800 }),
      ]);
      exps.value = experiments.experiments;
      Object.keys(runtime).forEach(key => delete runtime[key]);
      (obs.workers || []).forEach(w => { runtime[w.experiment_id] = w; });
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
    watch(() => appState.experimentVersion, refresh);
    onMounted(refresh);
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
        <div class="eyebrow">RESEARCH WORKBENCH / CROSS-SECTIONAL RANKING</div>
        <h1>多因子选股器</h1>
        <p>从已入库因子构建一次研究快照，按截面综合排名生成候选股票清单。</p>
      </div>
      <span class="tag amber">研究用途 · 非交易批准</span>
    </div>

      <div class="selector-toolbar card">
      <div class="selector-field selector-dsl">
        <label>直接 DSL 选股（可选）</label>
        <input v-model="directExpr" placeholder="例如：-rank(ts_delta(close, 20))" />
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
          <div v-if="!factors.length" class="factor-empty">正在加载因子库…</div>
          <div v-else class="factor-list">
            <label v-for="f in factors" :key="f.expression" class="factor-row" :class="{active:f.enabled}">
              <input type="checkbox" v-model="f.enabled" />
              <span class="factor-mark"></span>
              <span class="factor-copy">
                <span class="factor-name">{{ f.name }}</span>
                <code :title="f.expression">{{ f.expression }}</code>
                <span class="factor-meta">{{ f.grade || '未审计' }} · {{ f.lifecycle }} · 研究分 {{ formatScore(f.score) }}</span>
              </span>
              <input class="weight-input" v-model.number="f.weight" type="number" min="0.5" max="5" step="0.5" :disabled="!f.enabled" title="因子权重" />
            </label>
          </div>
          <div class="factor-footer">
            <span>组合权重</span><b>{{ totalWeight.toFixed(1) }}</b>
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
          <div class="metric-strip">
              <div class="metric-card"><span>股票池</span><b>Top {{ result.universe_n || univN }}</b><small>按 60 日成交额</small></div>
              <div class="metric-card"><span>启用因子</span><b>{{ result.factor_count }}</b><small>加权截面排名</small></div>
              <div class="metric-card"><span>输出数量</span><b>{{ result.stocks.length }}</b><small>候选清单</small></div>
              <div class="metric-card accent"><span>最高综合分</span><b>{{ topScore }}</b><small>{{ result.expression_mode ? '单条 DSL 排名' : '相对排序分数' }}</small></div>
            </div>
          </div>
          <div class="card result-table-card">
            <div class="panel-title-row"><div><h2>候选清单</h2><span class="sub">综合分越高代表因子排名组合越靠前</span></div><span class="tag">{{ result.date }}</span></div>
            <table class="result-table"><thead><tr><th>排名</th><th>证券</th><th>名称</th><th>综合分</th><th>相对位置</th></tr></thead>
              <tbody><tr v-for="s in result.stocks" :key="s.rank" :class="{'top-pick':s.rank<=10}"><td><span class="rank-number">{{ String(s.rank).padStart(2,"0") }}</span></td><td><code class="ticker">{{ s.ts_code }}</code></td><td>{{ s.name || "—" }}</td><td><b>{{ formatScore(s.score) }}</b></td><td><span class="rank-bar"><i :style="{ width: rankWidth(s) }"></i></span></td></tr></tbody>
            </table>
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
    const directExpr = ref("");
    const result = ref(null); const error = ref(""); const loading = ref(false);
    const enabledCount = computed(() => factors.value.filter(f=>f.enabled).length);
    const canRun = computed(() => Boolean(directExpr.value.trim()) || enabledCount.value > 0);
    const totalWeight = computed(() => factors.value.filter(f=>f.enabled).reduce((sum, f) => sum + (Number(f.weight) || 0), 0));
    const topScore = computed(() => result.value?.stocks?.length ? formatScore(result.value.stocks[0].score) : "—");
    function formatScore(value) { return value == null ? "—" : Number(value).toFixed(2); }
    function rankWidth(stock) {
      if (!result.value?.stocks?.length) return "0%";
      return `${((result.value.stocks.length - stock.rank + 1) / result.value.stocks.length) * 100}%`;
    }

    async function loadFactors() {
      try {
        const d = await api("/factors?sort=grade", { cacheTtl: 1000 });
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
        }));
      } catch(e) { error.value="加载因子失败: "+e.message; }
    }

    function selectAll() { factors.value.forEach(f => { f.enabled = true; }); }
    function clearAll() { factors.value.forEach(f => { f.enabled = false; }); }

    async function run() {
      const enabled = factors.value.filter(f=>f.enabled);
      if (!enabled.length && !directExpr.value.trim()) { error.value="请至少选择一个因子或输入 DSL 表达式"; return; }
      loading.value=true; error.value=""; result.value=null;
      try {
        const r = await api("/screener", { method:"POST", body:{
          expression: directExpr.value.trim() || undefined,
          factors: enabled.map(f=>({expression:f.expression,weight:f.weight,direction:f.direction})),
          date: date.value, universe_n: univN.value, top_n: topN.value, direction:"top"
        }});
        result.value = r;
      } catch(e) { error.value = "选股失败: "+e.message; }
      finally { loading.value=false; }
    }

    watch(() => appState.experimentVersion, () => { result.value=null; directExpr.value=""; loadFactors(); });
    onMounted(loadFactors);
    return { date, univN, topN, directExpr, factors, result, error, loading, enabledCount, canRun, totalWeight, topScore, formatScore, rankWidth, selectAll, clearAll, run, loadFactors };
  },
};
App.components.ScreenerView = ScreenerView;
createApp(App).mount("#app");

// ========== 选股器组件 ==========
