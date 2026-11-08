/* Visual building blocks for the static SEO pages.
 *
 * The comparison pages were walls of text: a TL;DR box, paragraphs, a table.
 * They ranked on the facts and gave a reader nothing to do or look at. These
 * blocks are the parts that earn a visit: the numbers that answer the query at a
 * glance, a calculator that prices the reader's own usage across tools, the
 * real output of the product, and how the workflows differ.
 *
 * Every block renders its full content as HTML at build time. The calculator's
 * script only re-prices for a different number of minutes; without it the page
 * still shows the table for the default value, which is what a crawler reads.
 */

import { PRICE_MODELS } from './data.js'

// Local copy: render.js imports this module for its CSS, so importing esc back
// from there would make the two modules depend on each other.
const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

/* The paste-a-link form (behaviour in dashboard/tools/paste-link.js, Vite
 * entry `tool-paste-link`). The clip tool pages put it under the H1; the
 * comparison pages a reader reaches from an AI answer put it where the CTA box
 * was, so the first video can be started in one step instead of a click
 * through to the homepage. One page holds at most one: the ids are fixed. */
export const PASTE_LINK_ENTRY = 'tool-paste-link'
export const pasteLinkWidget = ({ placeholder = 'Paste a YouTube link, or any video link', label = 'Video link', hint = '' } = {}) => `
<section class="tool" id="tool" aria-label="${esc(label)}">
  <form id="pl-form" class="tool-row" autocomplete="off" novalidate>
    <label for="pl-url" class="sr">${esc(label)}</label>
    <input id="pl-url" name="url" type="url" inputmode="url" placeholder="${esc(placeholder)}">
    <button type="submit" class="btn-primary">Get clips</button>
  </form>
  ${hint ? `<p class="tool-hint">${esc(hint)}</p>` : ''}
  <p class="tool-hint">No link? <a id="pl-upload" href="/#app">Upload the file in the app</a> instead. Sources need at least 45 seconds of material.</p>
  <div id="pl-status" class="tool-status" role="status" aria-live="polite" hidden></div>
</section>`

export const COMPONENTS_CSS = `
.facts{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr));margin:0 0 2.25rem}
.fact{background:var(--paper2);border:1px solid var(--rule2);border-radius:12px;padding:1rem 1.1rem}
.fact .k{font-family:var(--mono);font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);display:block}
.fact .v{font-family:var(--display);font-size:1.7rem;line-height:1.15;color:var(--ink);display:block;margin:.35rem 0 .2rem}
.fact .s{font-size:.82rem;color:var(--muted);line-height:1.45;display:block}
.summary{font-size:1.08rem;line-height:1.7;color:var(--ink2);margin:0 0 2rem;max-width:46rem}
.summary p{margin:0 0 .8rem}
.summary p:first-child{font-size:1.18rem;color:var(--ink)}
.demo{margin:1.5rem 0 .6rem;border:1px solid var(--rule2);border-radius:14px;overflow:hidden;background:#0b0b0f}
.demo video{width:100%;height:auto;display:block;aspect-ratio:16/9}
.demo + .calc-note{margin-bottom:1.75rem}
.calc{background:var(--paper2);border:1px solid var(--rule2);border-radius:14px;padding:1.25rem 1.25rem 1rem;margin:1.5rem 0 1rem}
.calc-head{display:flex;flex-wrap:wrap;gap:.75rem 1.5rem;align-items:center;justify-content:space-between;margin-bottom:1rem}
.calc-head label{font-size:.9rem;color:var(--ink2);display:flex;flex-direction:column;gap:.4rem;flex:1 1 16rem}
.calc-head output{font-family:var(--display);font-size:1.6rem;color:var(--ink)}
.calc input[type=range]{width:100%;accent-color:var(--brass)}
.calc .billing{display:flex;gap:.35rem}
.calc .billing button{background:transparent;border:1px solid var(--rule2);color:var(--ink2);border-radius:999px;padding:.35rem .85rem;font:inherit;font-size:.85rem;cursor:pointer}
.calc .billing button[aria-pressed=true]{background:var(--brass);border-color:var(--brass);color:oklch(17% 0.03 50)}
.bars{list-style:none;padding:0;margin:0}
.bars li{display:grid;grid-template-columns:8.5rem 1fr;gap:.25rem 1rem;align-items:center;padding:.6rem 0;border-top:1px solid var(--rule);margin:0}
.bars .bt{font-weight:500;color:var(--ink);line-height:1.3}
.bars .bt small{display:block;font-weight:400;font-size:.78rem;color:var(--muted)}
@media(max-width:36rem){.bars li{grid-template-columns:1fr}.bars .bt small{display:inline;margin-left:.4rem}}
.bars .track{position:relative;height:1.9rem;background:var(--paper);border-radius:6px;overflow:hidden}
.bars .fill{position:absolute;inset:0 auto 0 0;background:oklch(64% 0.012 262 / .35);border-radius:6px;min-width:2px;transition:width .25s}
.bars li.us .fill{background:var(--brass)}
.bars .val{position:absolute;left:.6rem;top:50%;transform:translateY(-50%);font-size:.85rem;color:var(--ink);white-space:nowrap}
.bars li.us .val{color:oklch(17% 0.03 50);font-weight:600}
.bars li.us .val.ext{color:var(--brass)}
.bars .val.ext{color:var(--ink)}
.calc-note{font-size:.8rem;color:var(--muted);margin:.75rem 0 0}
.flow{display:grid;gap:1rem;grid-template-columns:1fr;margin:1.5rem 0}
@media(min-width:44rem){.flow{grid-template-columns:1fr 1fr}}
.lane{background:var(--paper2);border:1px solid var(--rule2);border-radius:14px;padding:1.1rem 1.2rem}
.lane.us{border-color:oklch(76% 0.17 50 / .55)}
.lane h3{margin:0 0 .8rem;font-family:var(--display);font-weight:400;font-size:1.3rem}
.lane ol{list-style:none;padding:0;margin:0;counter-reset:s}
.lane li{position:relative;padding:.55rem 0 .55rem 2.3rem;margin:0;border-top:1px dashed var(--rule2);font-size:.92rem}
.lane li:first-child{border-top:0}
.lane li::before{counter-increment:s;content:counter(s);position:absolute;left:0;top:.5rem;width:1.6rem;height:1.6rem;border-radius:50%;
  display:grid;place-items:center;font-family:var(--mono);font-size:.72rem;background:var(--paper3);color:var(--ink2)}
.lane li.you::after{content:"you";font-family:var(--mono);font-size:.62rem;letter-spacing:.08em;text-transform:uppercase;
  margin-left:.5rem;padding:.1rem .45rem;border-radius:999px;background:oklch(76% 0.17 50 / .18);color:var(--brass)}
.lane.us li::before{background:oklch(76% 0.17 50 / .2);color:var(--brass)}
.verdict{display:grid;gap:1rem;grid-template-columns:1fr;margin:1.5rem 0}
@media(min-width:44rem){.verdict{grid-template-columns:1fr 1fr}}
.verdict div{border:1px solid var(--rule2);border-radius:14px;padding:1.1rem 1.2rem;background:var(--paper2)}
.verdict .us{border-color:oklch(76% 0.17 50 / .55)}
.verdict strong{display:block;font-family:var(--display);font-weight:400;font-size:1.3rem;color:var(--ink);margin-bottom:.6rem}
.verdict ul{margin:0;padding-left:1.1rem}
.verdict li{font-size:.92rem}
`

/** The answer at a glance: 3-4 tiles of { k: label, v: value, s: detail }. */
/** Non-affiliation line for pages that name a game, publisher or platform. */
export const trademarkNote = (marks, owners) =>
  `<p class="tm-note">OpenShorts is an independent tool and is not affiliated with, endorsed or sponsored by ${owners}. ${marks} ${marks.includes(',') || marks.includes(' and ') ? 'are trademarks of their respective owners' : 'is a trademark of its owner'}, used here only to say which footage the tool works with.</p>`

export const factTiles = (facts) =>
  `<div class="facts">${facts
    .map((f) => `<div class="fact"><span class="k">${esc(f.k)}</span><span class="v">${esc(f.v)}</span><span class="s">${esc(f.s || '')}</span></div>`)
    .join('')}</div>`

/* Real product output: a 16:9 source excerpt next to the 9:16 clip OpenShorts
 * made from it. Every source here is Creative Commons Attribution or our own,
 * and the credit line under the block is what the licence asks for. */
export const DEMOS = {
  split: {
    video: '/demo/case-split.mp4',
    poster: '/screens/case-split.webp',
    caption: 'A two-person podcast: both speakers stacked in half-frames, captions on the seam where they cover nobody.',
    credit:
      '<a href="https://www.youtube.com/watch?v=-KbQj_vboOU" rel="noopener">Turn the Tables with Dan and Shoshana Jordan</a>, Heritage of Faith, CC BY',
  },
  talk: {
    video: '/demo/case-talk.mp4',
    poster: '/screens/case-talk.webp',
    caption: 'A 39-minute interview: Gemini picked the moment, the crop follows the speaker, and the captions come from the word-level transcript.',
    credit:
      '<a href="https://www.youtube.com/watch?v=aqFhllJJzoQ" rel="noopener">Startup Founder &amp; CEO Mark Bush on How to Overcome Adversity</a>, Foundation for Economic Education, CC BY',
  },
}

/* Real product output: the original and the clip OpenShorts made from it,
 * side by side and in sync, in one video. Every source is Creative Commons
 * Attribution and the credit line under the block is what the licence asks
 * for. */
export const demoBlock = (which = 'split', caption) => {
  const d = DEMOS[which] || DEMOS.split
  return `
<figure class="demo">
  <video src="${d.video}" poster="${d.poster}" autoplay muted loop playsinline preload="none" width="960" height="540"
    aria-label="Original video next to the vertical clip OpenShorts made from it"></video>
</figure>
<p class="calc-note">Real OpenShorts output, not a mock-up. ${esc(caption || d.caption)} Source: ${d.credit}.</p>`
}

/** Two workflows side by side. Steps marked { you: true } need a human. */
export const flowCompare = (them, us) => {
  const lane = (l, cls) =>
    `<div class="lane ${cls}"><h3>${esc(l.title)}</h3><ol>${l.steps
      .map((s) => (typeof s === 'string' ? `<li>${esc(s)}</li>` : `<li class="you">${esc(s.text)}</li>`))
      .join('')}</ol></div>`
  return `<div class="flow">${lane(them, '')}${lane(us, 'us')}</div>`
}

/** "Pick them if / pick us if", as two cards instead of one paragraph. */
export const verdictCards = (name, pickThem, pickUs) => `
<div class="verdict">
  <div><strong>Pick ${esc(name)} if</strong><ul>${pickThem.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>
  <div class="us"><strong>Pick OpenShorts if</strong><ul>${pickUs.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>
</div>`

/* --- Cost calculator ------------------------------------------------------
 * Prices the cheapest plan that covers N minutes of source video a month, per
 * tool, on monthly or yearly billing. Plans come from PRICE_MODELS in data.js,
 * which carries the date each one was checked. A plan without a yearly price
 * (Opus Clip Starter is monthly only) is still offered on yearly billing,
 * because that is what a buyer would actually pay. */
export function priceFor(tool, minutes, billing) {
  const m = PRICE_MODELS[tool]
  let best = null
  for (const p of m.plans) {
    const opt = p[billing] || p.monthly
    if (!opt || opt.minutes < minutes) continue
    if (!best || opt.price < best.price) best = { plan: p.name, price: opt.price, watermark: !!p.watermark }
  }
  return best // null = no published plan covers it
}

const money = (n) => (n === 0 ? '$0' : `$${n % 1 ? n.toFixed(2) : n}`)

const barsHtml = (tools, minutes, billing) => {
  const rows = tools.map((t) => ({ t, r: priceFor(t, minutes, billing) }))
  const max = Math.max(1, ...rows.map((x) => (x.r ? x.r.price : 0)))
  return rows
    .map(({ t, r }) => {
      const m = PRICE_MODELS[t]
      const w = r ? Math.max(2, (r.price / max) * 100) : 0
      const label = r
        ? `${money(r.price)}/mo · ${r.plan}${r.watermark ? ' · watermark' : ''}`
        : 'No published plan, custom pricing'
      return `<li class="${t.startsWith('openshorts') ? 'us' : ''}" data-tool="${esc(t)}"><span class="bt">${esc(m.name)}<small>${esc(m.unit)}</small></span><span class="track"><span class="fill" style="width:${w}%"></span><span class="val${w < 45 ? ' ext' : ''}"${w < 45 ? ` style="left:calc(${w}% + .6rem)"` : ''}>${esc(label)}</span></span></li>`
    })
    .join('')
}

let calcCount = 0
export function costCalculator(tools, { minutes = 300, note } = {}) {
  const id = `calc${++calcCount}`
  const list = ['openshorts-self', 'openshorts', ...tools.filter((t) => !t.startsWith('openshorts') && PRICE_MODELS[t])]
  const models = Object.fromEntries(list.map((t) => [t, PRICE_MODELS[t]]))
  const checked = list.map((t) => PRICE_MODELS[t].checked).filter(Boolean).sort()[0]
  return `
<div class="calc" id="${id}">
  <div class="calc-head">
    <label for="${id}-m">Minutes of source video you clip a month
      <output id="${id}-o" for="${id}-m">${minutes} min</output>
      <input id="${id}-m" type="range" min="30" max="1500" step="30" value="${minutes}">
    </label>
    <div class="billing" role="group" aria-label="Billing">
      <button type="button" data-b="monthly" aria-pressed="true">Monthly</button>
      <button type="button" data-b="yearly" aria-pressed="false">Yearly</button>
    </div>
  </div>
  <ul class="bars" id="${id}-bars">${barsHtml(list, minutes, 'monthly')}</ul>
  <p class="calc-note">Cheapest published plan that covers that many minutes, per month. Prices checked ${esc(checked)}.
  ${note ? esc(note) : 'Self-hosted needs Docker and your own Gemini key, whose free tier covers 1,500 requests a day.'}</p>
</div>
<script>
(function(){
  var M=${JSON.stringify(models)},root=document.getElementById(${JSON.stringify(id)}),r=root.querySelector('input'),o=root.querySelector('output'),b='monthly';
  function price(t,n){var best=null;M[t].plans.forEach(function(p){var x=p[b]||p.monthly;if(!x||x.minutes<n)return;if(!best||x.price<best.price)best={plan:p.name,price:x.price,wm:!!p.watermark}});return best}
  function money(n){return n===0?'$0':'$'+(n%1?n.toFixed(2):n)}
  function draw(){var n=+r.value;o.textContent=n+' min';var rows=Object.keys(M).map(function(t){return[t,price(t,n)]});
    var max=Math.max.apply(null,[1].concat(rows.map(function(x){return x[1]?x[1].price:0})));
    rows.forEach(function(x){var li=root.querySelector('[data-tool="'+x[0]+'"]'),p=x[1],w=p?Math.max(2,p.price/max*100):0;
      li.querySelector('.fill').style.width=w+'%';var v=li.querySelector('.val');v.className='val'+(w<45?' ext':'');v.style.left=w<45?'calc('+w+'% + .6rem)':'';
      v.textContent=p?money(p.price)+'/mo · '+p.plan+(p.wm?' · watermark':''):'No published plan, custom pricing'})}
  r.addEventListener('input',draw);
  root.querySelectorAll('.billing button').forEach(function(btn){btn.addEventListener('click',function(){b=btn.dataset.b;
    root.querySelectorAll('.billing button').forEach(function(x){x.setAttribute('aria-pressed',x===btn)});draw()})});
})();
</script>`
}
