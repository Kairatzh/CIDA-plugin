"""
demo.py — демонстрация спасательной миссии: одна и та же сцена, несколько роёв рядом, в одном HTML-файле.

    python -m cida_swarm.demo runs/stig/model_best.pt runs/cida/model_best.pt \
        --labels "общая карта" "CIDA" --scenes 4 --out rescue_demo.html

Файл открывается в любом браузере без интернета: план здания с «туманом» неисследованного, завалы с весом,
пожар, пострадавшие с весом, дроны с подписью состояния (ищет / ждёт помощи / несёт / разбирает завал),
линии связи (толщина — вес сообщения), общее убеждение роя «где люди», табло и лента событий.
Все рои на сцене видят один и тот же мир: тот же лабиринт, те же пожар и обрушения.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.distributions import Categorical

from .arena import ArenaEnv
from .policy import RescueActor, forget_done, init_memory, policy_step
from .rescue import RescueConfig, RescueEnv

# состояния дронов для подписи
SEARCH, WAIT, CARRY, PUSH, LOST = range(5)


def load_actor(path: str):
    """Мозг спасателей и конфиг мира. У модели арены к мозгу прикреплены охотники (actor.hunter)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("world") not in ("rescue", "arena"):
        raise ValueError(f"{path}: это не модель спасательной миссии")
    cfg = RescueConfig(**ckpt["env_config"])
    actor = RescueActor(cfg, **ckpt["actor_kwargs"])
    actor.load_state_dict(ckpt["actor"])
    actor.hunter = None
    if "hunter" in ckpt:
        h_cfg = RescueConfig(**{**cfg.to_dict(), "n_drones": cfg.n_hunters, "min_drones": 1})
        hunter = RescueActor(h_cfg, **ckpt["actor_kwargs"])
        hunter.load_state_dict(ckpt["hunter"])
        object.__setattr__(actor, "hunter", hunter.eval())   # не подмодуль: не попадает в state_dict спасателей
    return actor.eval(), cfg


@torch.no_grad()
def record(actor: RescueActor, cfg: RescueConfig, seed: int, difficulty: int,
           n_drones: Optional[int] = None, greedy: bool = False) -> Dict:
    cfg1 = RescueConfig(**{**cfg.to_dict(), "n_envs": 1, "difficulty_min": difficulty, "difficulty_max": difficulty,
                           "bank_size": 0})
    hunter = getattr(actor, "hunter", None)
    env = (ArenaEnv if hunter is not None else RescueEnv)(cfg1, "cpu", seed=seed)
    torch.manual_seed(seed)
    obs = env.reset()
    hmem = init_memory(hunter, 1, cfg1.n_hunters, "cpu") if hunter is not None else None
    if n_drones is not None:
        env.active[:] = torch.arange(cfg1.n_drones) < n_drones
        env.alive = env.active.clone()
        obs = env.observe()
    g = env.G
    episode = {
        "grid": g, "coarse": cfg1.coarse,
        "walls": env.walls[0].nonzero().flatten().tolist(),
        "exit": env.exit_mask.nonzero().flatten().tolist(),
        "n_victims": cfg1.n_victims, "frames": [],
    }
    seen_before = torch.zeros(env.cells, dtype=torch.bool)
    state = torch.full((cfg1.n_drones,), SEARCH)
    events: List[str] = []
    mem = init_memory(actor, 1, cfg1.n_drones, "cpu")
    for t in range(cfg1.max_steps + 1):
        logits, aux, mem = policy_step(actor, obs, mem)
        alive = env.alive[0]
        frame = {
            "d": [[int(x), int(y), int(a and act), int(s)] for (x, y), a, act, s in
                  zip(env.pos[0].tolist(), alive.tolist(), env.active[0].tolist(), state.tolist())],
            "v": [[int(x), int(y), int(w), int(p)] for (x, y), w, p in
                  zip(env.vpos[0].tolist(), env.vweight[0].tolist(), env.present[0].tolist())],
            "r": [[int(c), int(env.rubble[0, c])] for c in env.rubble[0].nonzero().flatten().tolist()],
            "h": env.hpos[0].tolist() if bool(env.hazard_on[0]) else None,
            "q": ([[int(x), int(y), int(s > 0)] for (x, y), a, s in
                   zip(env.qpos[0].tolist(), env.q_active[0].tolist(), env.q_stun[0].tolist()) if a]
                  if hunter is not None else None),
            "s": (env.team_seen[0] & ~seen_before).nonzero().flatten().tolist(),
            "score": int(env.ep_rescued[0]),
            "ev": events,
        }
        seen_before = env.team_seen[0].clone()
        if aux is not None:
            attn = aux["attn"][0]
            frame["e"] = [[i, j, round(float(attn[i, j]), 2)] for i in range(attn.shape[0]) for j in range(attn.shape[1])
                          if attn[i, j] > 0.12 and alive[i] and alive[j]]
            belief = (torch.sigmoid(aux["beliefs"][-1][0][alive]).mean(0) if bool(alive.any())
                      else torch.zeros(cfg1.coarse ** 2))
            frame["b"] = [round(float(x), 2) for x in belief]
        episode["frames"].append(frame)
        if t == cfg1.max_steps or not bool(env.present[0].any()) or not bool(env.alive[0].any()):
            break

        actions = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
        before_rescued, before_alive = int(env.ep_rescued[0]), env.alive[0].clone()
        if hunter is not None:
            hlogits, _, hmem = policy_step(hunter, env.observe_hunters(), hmem)
            h_act = hlogits.argmax(-1) if greedy else Categorical(logits=hlogits).sample()
            before_stuns = int(env.ep_stuns[0])
            obs, _, done, info = env.step(actions, h_act)
        else:
            obs, _, done, info = env.step(actions)
        if bool(done[0]):
            # мир перезапустился — фиксируем итог по info и заканчиваем
            frame_end = dict(episode["frames"][-1])
            frame_end["score"] = int(round(float(info["rescued_frac"][0]) * cfg1.n_victims))
            frame_end["ev"] = ["Миссия завершена" if frame_end["score"] == cfg1.n_victims else "Время вышло"]
            episode["frames"].append(frame_end)
            break
        last = env.last
        state = torch.where(last["carriers"][0], CARRY, torch.where(last["waiting"][0], WAIT,
                            torch.where(last["pushing"][0], PUSH, SEARCH)))
        state = torch.where(env.alive[0], state, torch.full_like(state, LOST))
        events = []
        rescued = int(env.ep_rescued[0]) - before_rescued
        if rescued:
            events.append(f"Человек вынесен в безопасную зону ({int(env.ep_rescued[0])}/{cfg1.n_victims})")
        if int(last["cleared"][0]):
            events.append("Завал разобран вместе")
        if bool(last["collapse"][0]):
            events.append("Обрушение! Дверь завалило")
        for k in (before_alive & ~env.alive[0]).nonzero().flatten().tolist():
            events.append(f"Охотник поймал дрона {k + 1}" if hunter is not None else f"Дрон {k + 1} потерян в огне")
        if hunter is not None and int(env.ep_stuns[0]) > before_stuns:
            events.append("Дроны вдвоём оглушили охотника!")
        waiting = last["waiting"][0].nonzero().flatten().tolist()
        if waiting and t % 10 == 0:
            events.append(f"Дрон {waiting[0] + 1} ждёт помощи у тяжёлого пострадавшего")
    episode["result"] = {"rescued": episode["frames"][-1]["score"], "steps": len(episode["frames"]) - 1,
                         "lost": sum(1 for d in episode["frames"][-1]["d"] if d[3] == LOST)}
    return episode


def build(models: List[str], labels: List[str], seeds: List[int], difficulties: List[int],
          n_drones: Optional[int], greedy: bool) -> Dict:
    actors = [load_actor(m) for m in models]
    scenes = []
    for seed, diff in zip(seeds, difficulties):
        panels = [{"label": lab, "episode": record(a, c, seed, diff, n_drones, greedy)}
                  for (a, c), lab in zip(actors, labels)]
        scenes.append({"title": f"Сцена {len(scenes) + 1} · сложность {diff}", "difficulty": diff, "panels": panels})
        summary = " | ".join(f"{p['label']}: {p['episode']['result']['rescued']}/{p['episode']['n_victims']} "
                             f"за {p['episode']['result']['steps']} шагов" for p in panels)
        print(f"[demo] {scenes[-1]['title']}: {summary}", flush=True)
    return {"scenes": scenes}


def write_html(data: Dict, out: str) -> None:
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    Path(out).write_text(html, encoding="utf-8")
    print(f"[demo] сохранено: {out} ({Path(out).stat().st_size / 1e6:.1f} МБ)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="CIDA-Rescue HTML demo")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--difficulty", type=int, nargs="+", default=[2, 3, 4, 4],
                    help="сложность каждой сцены (список повторяется по кругу)")
    ap.add_argument("--n-drones", type=int, default=6)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", default="rescue_demo.html")
    args = ap.parse_args()
    labels = args.labels or [Path(m).parent.name for m in args.models]
    seeds = [args.first_seed + k for k in range(args.scenes)]
    diffs = [args.difficulty[k % len(args.difficulty)] for k in range(args.scenes)]
    write_html(build(args.models, labels, seeds, diffs, args.n_drones, args.greedy), args.out)


TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CIDA-Rescue</title>
<style>
:root{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--text:#e5e7eb;--muted:#8b95a7;--accent:#a78bfa;
--wall:#46536b;--floor:#182031;--fog:#0b0f17;--exit:#10b98133;--exitb:#10b981;--victim:#f43f5e;--rubble:#a16207;
--fire:#f97316;--drone:#38bdf8;--wait:#facc15;--carry:#34d399;--push:#fb923c;--hunter:#d946ef}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:22px 28px 8px;display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;justify-content:space-between}
h1{margin:0;font-size:26px;letter-spacing:.3px}
h1 span{color:var(--accent)}
.sub{color:var(--muted);margin-top:4px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:8px 28px 14px}
button,select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:8px 14px;
font:inherit;cursor:pointer}
button:hover,select:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:#0b0f17;border-color:var(--accent);font-weight:600}
input[type=range]{accent-color:var(--accent)}
label.toggle{display:flex;gap:6px;align-items:center;color:var(--muted);cursor:pointer}
.stage{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;padding:0 28px 18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px;display:flex;flex-direction:column;gap:10px}
.card h2{margin:0;font-size:17px;display:flex;justify-content:space-between;align-items:center}
.badge{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 10px}
canvas{width:100%;aspect-ratio:1/1;border-radius:12px;background:var(--fog);display:block}
.score{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.score div{background:#0f1522;border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.score b{display:block;font-size:20px}
.score small{color:var(--muted)}
.log{height:92px;overflow:hidden;font-size:13px;color:var(--muted);display:flex;flex-direction:column-reverse;gap:2px}
.log .new{color:var(--text)}
.legend{display:flex;flex-wrap:wrap;gap:14px;padding:0 28px 26px;color:var(--muted);font-size:13px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.winner{color:var(--carry)}
@media (max-width:600px){header,.controls,.stage,.legend{padding-left:16px;padding-right:16px}}
</style>
</head>
<body>
<header>
  <div><h1>CIDA-<span>Rescue</span></h1>
  <div class="sub">Рой дронов выносит людей из здания: тяжёлых несут вместе, завалы разбирают вместе, огонь обходят.</div></div>
  <div class="sub" id="sceneTitle"></div>
</header>
<div class="controls">
  <button class="primary" id="play">▶ Пуск</button>
  <button id="restart">⟲ Сначала</button>
  <select id="scene"></select>
  <label class="toggle">скорость <input id="speed" type="range" min="1" max="30" value="8"></label>
  <input id="seek" type="range" min="0" max="0" value="0" style="flex:1;min-width:160px">
  <span class="badge" id="tick">шаг 0</span>
  <label class="toggle"><input type="checkbox" id="showBelief" checked> убеждение роя</label>
  <label class="toggle"><input type="checkbox" id="showLinks" checked> связь</label>
</div>
<div class="stage" id="stage"></div>
<div class="legend">
  <span><i style="background:var(--drone)"></i>ищет</span>
  <span><i style="background:var(--wait)"></i>держит, ждёт помощи</span>
  <span><i style="background:var(--carry)"></i>несёт</span>
  <span><i style="background:var(--push)"></i>разбирает завал</span>
  <span><i style="background:var(--victim)"></i>пострадавший (цифра — сколько дронов нужно)</span>
  <span><i style="background:var(--rubble);border-radius:3px"></i>завал (цифра — вес)</span>
  <span><i style="background:var(--fire)"></i>пожар</span>
  <span><i style="background:var(--hunter);border-radius:2px;transform:rotate(45deg)"></i>охотник (серый — оглушён)</span>
  <span><i style="background:var(--exitb);border-radius:3px"></i>безопасная зона</span>
  <span><i style="background:var(--accent)"></i>линия связи</span>
</div>
<script>
const DATA = __DATA__;
const COLORS = {0:'--drone',1:'--wait',2:'--carry',3:'--push'};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let scene = 0, tick = 0, playing = false, acc = 0, last = 0;
const stage = document.getElementById('stage');
const sel = document.getElementById('scene');
DATA.scenes.forEach((s, i) => { const o = document.createElement('option'); o.value = i; o.textContent = s.title; sel.appendChild(o); });

let panels = [];
function setup() {
  stage.innerHTML = '';
  const sc = DATA.scenes[scene];
  document.getElementById('sceneTitle').textContent = sc.title;
  const maxT = Math.max(...sc.panels.map(p => p.episode.frames.length)) - 1;
  const seek = document.getElementById('seek'); seek.max = maxT; seek.value = 0;
  const scoreOf = r => r.rescued * 1000 - r.steps;
  const scores = sc.panels.map(p => scoreOf(p.episode.result));
  const best = Math.max(...scores);
  const unique = scores.filter(x => x === best).length === 1;
  panels = sc.panels.map(p => {
    const card = document.createElement('div'); card.className = 'card';
    const ep = p.episode, r = ep.result;
    const win = sc.panels.length > 1 && unique && r.rescued > 0 && scoreOf(r) === best;
    card.innerHTML = `<h2>${p.label}<span class="badge ${win ? 'winner' : ''}">итог: ${r.rescued}/${ep.n_victims} за ${r.steps} шагов${win ? ' · лучший' : ''}</span></h2>
      <canvas></canvas>
      <div class="score"><div><small>вынесено</small><b class="sc-r">0/${ep.n_victims}</b></div>
      <div><small>шаг</small><b class="sc-t">0</b></div><div><small>дронов в строю</small><b class="sc-a">0</b></div></div>
      <div class="log"></div>`;
    stage.appendChild(card);
    // накопленная карта исследованного по кадрам
    const seen = []; const acc = new Set();
    ep.frames.forEach(f => { f.s.forEach(c => acc.add(c)); seen.push(new Set(acc)); });
    return {card, ep, seen, canvas: card.querySelector('canvas'), log: card.querySelector('.log'), shown: -1};
  });
  tick = 0; draw();
}

function draw() {
  const showB = document.getElementById('showBelief').checked, showL = document.getElementById('showLinks').checked;
  document.getElementById('tick').textContent = 'шаг ' + tick;
  document.getElementById('seek').value = tick;
  panels.forEach(P => {
    const ep = P.ep, g = ep.grid, f = ep.frames[Math.min(tick, ep.frames.length - 1)];
    const cv = P.canvas, dpr = window.devicePixelRatio || 1, W = cv.clientWidth;
    if (cv.width !== Math.round(W * dpr)) { cv.width = cv.height = Math.round(W * dpr); }
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const s = W / g, seen = P.seen[Math.min(tick, ep.frames.length - 1)];
    const walls = new Set(ep.walls), exit = new Set(ep.exit);
    c.fillStyle = css('--fog'); c.fillRect(0, 0, W, W);
    // y растёт вверх, как на карте
    const X = x => x * s, Y = y => (g - 1 - y) * s;
    for (let x = 0; x < g; x++) for (let y = 0; y < g; y++) {
      const i = x * g + y, vis = seen.has(i);
      if (walls.has(i)) c.fillStyle = vis ? css('--wall') : '#252e3e';
      else if (exit.has(i)) c.fillStyle = css('--exit');
      else c.fillStyle = vis ? css('--floor') : '#0f141e';
      c.fillRect(X(x) + .5, Y(y) + .5, s - 1, s - 1);
    }
    c.strokeStyle = css('--exitb'); c.lineWidth = 2; c.strokeRect(X(1), Y(3), 3 * s, 3 * s);
    if (showB && f.b) {
      const q = g / ep.coarse;
      f.b.forEach((v, k) => { if (v < .08) return; const bx = Math.floor(k / ep.coarse), by = k % ep.coarse;
        c.fillStyle = `rgba(244,63,94,${Math.min(.45, v * .6)})`; c.fillRect(bx * q * s, (g - (by + 1) * q) * s, q * s, q * s); });
    }
    f.r.forEach(([cell, w]) => { const x = Math.floor(cell / g), y = cell % g;
      c.fillStyle = css('--rubble'); c.beginPath(); c.roundRect(X(x) + s * .1, Y(y) + s * .1, s * .8, s * .8, s * .18); c.fill();
      c.fillStyle = '#fff'; c.font = `bold ${s * .5}px system-ui`; c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(w, X(x) + s / 2, Y(y) + s / 2 + 1); });
    if (f.h) { const [hx, hy] = f.h, t = performance.now() / 300;
      const grd = c.createRadialGradient(X(hx) + s / 2, Y(hy) + s / 2, 1, X(hx) + s / 2, Y(hy) + s / 2, s * (1.6 + .15 * Math.sin(t)));
      grd.addColorStop(0, 'rgba(249,115,22,.95)'); grd.addColorStop(1, 'rgba(249,115,22,0)');
      c.fillStyle = grd; c.fillRect(X(hx) - 2 * s, Y(hy) - 2 * s, 5 * s, 5 * s); }
    if (f.q) f.q.forEach(([qx, qy, st]) => { const cx = X(qx) + s / 2, cy = Y(qy) + s / 2, r = s * .42;
      c.fillStyle = st ? '#64748b' : css('--hunter'); c.beginPath();
      c.moveTo(cx, cy - r); c.lineTo(cx + r, cy); c.lineTo(cx, cy + r); c.lineTo(cx - r, cy); c.closePath(); c.fill();
      if (st) { c.fillStyle = '#fff'; c.font = `bold ${s * .38}px system-ui`; c.textAlign = 'center'; c.textBaseline = 'middle';
        c.fillText('z', cx, cy + 1); } });
    f.v.forEach(([x, y, w, p]) => { if (!p) return;
      c.fillStyle = css('--victim'); c.beginPath(); c.arc(X(x) + s / 2, Y(y) + s / 2, s * .34, 0, 7); c.fill();
      c.fillStyle = '#fff'; c.font = `bold ${s * .42}px system-ui`; c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(w, X(x) + s / 2, Y(y) + s / 2 + 1); });
    if (showL && f.e) f.e.forEach(([i, j, w]) => { const a = f.d[i], b = f.d[j];
      c.strokeStyle = `rgba(167,139,250,${.25 + .6 * w})`; c.lineWidth = 1 + 3 * w;
      c.beginPath(); c.moveTo(X(a[0]) + s / 2, Y(a[1]) + s / 2); c.lineTo(X(b[0]) + s / 2, Y(b[1]) + s / 2); c.stroke(); });
    // дроны: несколько на одной клетке раскладываем по кругу
    const byCell = {};
    f.d.forEach((d, k) => { if (!d[2]) return; const key = d[0] + ',' + d[1]; (byCell[key] = byCell[key] || []).push(k); });
    Object.values(byCell).forEach(list => list.forEach((k, n) => {
      const d = f.d[k], ang = n / list.length * 6.283, off = list.length > 1 ? s * .22 : 0;
      const cx = X(d[0]) + s / 2 + off * Math.cos(ang), cy = Y(d[1]) + s / 2 + off * Math.sin(ang);
      c.fillStyle = css(COLORS[d[3]] || '--drone'); c.beginPath(); c.arc(cx, cy, s * .22, 0, 7); c.fill();
      c.strokeStyle = '#0b0f17'; c.lineWidth = 1.5; c.stroke(); }));
    const alive = f.d.filter(d => d[2]).length;
    P.card.querySelector('.sc-r').textContent = `${f.score}/${ep.n_victims}`;
    P.card.querySelector('.sc-t').textContent = Math.min(tick, ep.frames.length - 1);
    P.card.querySelector('.sc-a').textContent = alive;
    if (P.shown !== tick) {
      if (tick < P.shown) P.log.innerHTML = '';
      for (let t = Math.max(0, P.shown + 1); t <= Math.min(tick, ep.frames.length - 1); t++)
        ep.frames[t].ev.forEach(e => { const d = document.createElement('div'); d.textContent = `· ${t}: ${e}`;
          P.log.prepend(d); while (P.log.children.length > 6) P.log.lastChild.remove(); });
      P.shown = tick;
    }
  });
}

function loop(ts) {
  if (playing) {
    acc += (ts - last) / 1000 * +document.getElementById('speed').value;
    const maxT = +document.getElementById('seek').max;
    while (acc >= 1) { acc -= 1; if (tick < maxT) tick++; else { playing = false; setPlay(); } }
  }
  last = ts; draw(); requestAnimationFrame(loop);
}
function setPlay() { document.getElementById('play').textContent = playing ? '❚❚ Пауза' : '▶ Пуск'; }
document.getElementById('play').onclick = () => { if (tick >= +document.getElementById('seek').max) tick = 0; playing = !playing; setPlay(); };
document.getElementById('restart').onclick = () => { tick = 0; panels.forEach(P => { P.shown = -1; P.log.innerHTML = ''; }); };
document.getElementById('seek').oninput = e => { tick = +e.target.value; };
sel.onchange = () => { scene = +sel.value; playing = false; setPlay(); setup(); };
window.addEventListener('resize', draw);
setup(); requestAnimationFrame(loop);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
