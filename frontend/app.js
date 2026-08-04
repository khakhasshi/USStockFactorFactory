/* USStockFactorFactory 前端 — Vue3 全局构建 + ECharts */
const { createApp, ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } = Vue;

async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
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
        <div class="sub">外层步 {{ st.outer_step }} · 内层评估 {{ st.inner_evals }}</div>
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
    let timer = null;
    const fmt = (v) => (v == null ? "—" : Number(v).toFixed(4));
    const acceptRate = computed(() => {
      const c = st.value.counts;
      return c && c.outer_steps ? ((100 * c.accepted) / c.outer_steps).toFixed(0) + "%" : "—";
    });
    async function refresh() {
      try {
        st.value = await api("/engine/status");
        const prog = await api("/engine/progress");
        drawProgress(prog.steps);
        nextTick(() => { if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight; });
      } catch (e) { /* server booting */ }
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
    onMounted(() => { refresh(); timer = setInterval(refresh, 3000); });
    onUnmounted(() => clearInterval(timer));
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
    let timer = null;
    const versionNo = (id) => data.value.versions.find((v) => v.id === id)?.version_no;
    async function refresh() {
      const q = selected.value ? "?miner_version_id=" + selected.value : "";
      data.value = await api("/tree" + q);
      draw();
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
    onMounted(() => { refresh(); timer = setInterval(refresh, 6000); });
    onUnmounted(() => clearInterval(timer));
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
        <div><label>模式</label><select v-model="form.mode"><option value="long_short">多空</option><option value="long_only">纯多头</option></select></div>
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
    const form = reactive({ expression: "-rank(ts_delta(close, 20))", universe_n: 500, start: "2015-01-01", end: "2024-12-31", cost_bps: 15, direction: 1, mode: "long_short" });
    const result = ref(null), history = ref([]), err = ref(""), running = ref(false);
    const curveEl = ref(null);
    const labels = { days: "交易日数", ann_ret: "年化收益", ann_vol: "年化波动", sharpe: "Sharpe", max_dd: "最大回撤", avg_daily_turnover: "日均换手", final_nav: "期末净值" };
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
    async function loadHistory() { history.value = (await api("/backtests")).backtests; }
    onMounted(loadHistory);
    return { form, result, history, err, running, run, curveEl, labels };
  },
};

/* ============ 设置 ============ */
const SettingsView = {
  template: `
  <div>
    <div class="warn-banner">提示: 默认端口 9999, 访问 http://localhost:9999 (可用环境变量 FF_PORT 覆盖)。</div>
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
        <div class="sub" style="margin-top:8px">评估仅使用 INNER_PUBLIC(2010~2019) + META_TRAIN(2020~2022); META_HOLDOUT 与 FACTOR_VAULT 层永不进入循环提示词。</div>
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
    const msg = ref(""), saved = ref("");
    async function load() {
      const s = await api("/settings");
      Object.assign(llm, s.llm_providers);
      Object.assign(eng, s.engine_config);
    }
    async function save() {
      try {
        await api("/settings", { method: "POST", body: { llm_providers: JSON.parse(JSON.stringify(llm)), engine_config: JSON.parse(JSON.stringify(eng)) } });
        saved.value = "ok"; msg.value = "已保存 ✓";
        load();
      } catch (e) { saved.value = "err"; msg.value = "保存失败: " + e.message; }
      setTimeout(() => (msg.value = ""), 3000);
    }
    onMounted(load);
    return { llm, eng, save, msg, saved };
  },
};

/* ============ App ============ */
const App = {
  components: { Dashboard, ResearchTree, FactorLibrary, BacktestView, SettingsView },
  template: `
  <div class="topbar">
    <div class="logo">⚒ FactorFactory</div>
    <div class="tabs">
      <button v-for="t in tabs" :key="t.id" :class="{active: tab===t.id}" @click="tab=t.id">{{ t.label }}</button>
    </div>
    <div class="spacer"></div>
    <span class="state-badge" :class="engState==='running' ? 'state-running' : 'state-stopped'">● {{ engState }}</span>
  </div>
  <div class="main">
    <Dashboard v-if="tab==='dash'" />
    <ResearchTree v-else-if="tab==='tree'" />
    <FactorLibrary v-else-if="tab==='factors'" />
    <BacktestView v-else-if="tab==='backtest'" />
    <SettingsView v-else />
  </div>`,
  setup() {
    const tab = ref("dash");
    const tabs = [
      { id: "dash", label: "总览" }, { id: "tree", label: "研发树" },
      { id: "factors", label: "因子库" }, { id: "backtest", label: "回测" }, { id: "settings", label: "设置" },
    ];
    const engState = ref("…");
    let timer = null;
    async function poll() {
      try { engState.value = (await api("/engine/status")).state; } catch (e) {}
    }
    onMounted(() => { poll(); timer = setInterval(poll, 5000); });
    onUnmounted(() => clearInterval(timer));
    return { tab, tabs, engState };
  },
};

createApp(App).mount("#app");
