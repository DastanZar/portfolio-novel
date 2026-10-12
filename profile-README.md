# Abhinav Chauhan — LLM Efficiency ×10

<!-- HERO: copy this file to github.com/DastanZar/DastanZar/README.md to activate profile -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:C4F03C,100:11161d&height=180&section=header&text=Abhinav%20Chauhan&fontSize=42&fontColor=11161d&desc=LLM%20Inference%20•%20Quantization%20•%20Serving&descAlignY=75" alt="hero"/>
</p>

<p align="center">
  <a href="https://github.com/DastanZar"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=800&color=C4F03C&center=true&width=700&lines=10+novel+builds%2C+all+tested%2C+all+visual;quantization+→+scheduling+→+kernels+→+planner;79.9%25+MSE+cut+•+52.8%25+HBM+saved+•+Jain+0.950" alt="typing"/></a>
</p>

<p align="center">
  <a href="https://dastanzar.github.io/portfolio-novel/"><img src="https://img.shields.io/badge/Portfolio_Live-10_builds-C4F03C?style=for-the-badge&logo=githubpages&logoColor=black" alt="portfolio"/></a>
  <a href="https://github.com/DastanZar?tab=repositories"><img src="https://img.shields.io/badge/Repos-53-11161d?style=for-the-badge" alt="repos"/></a>
  <img src="https://komarev.com/ghpvc/?username=DastanZar&style=for-the-badge&color=C4F03C" alt="views"/>
</p>

> I build small, honest, measured LLM-efficiency tools. Each project below is a **fresh implementation with a novel delta**, tested on CPU, with receipts + charts. Prior art is credited in every README — no silent forks.

## ⚡ Live demo — GitHub Pages

**→ https://dastanzar.github.io/portfolio-novel/** (from `portfolio-novel/docs/`)

| Motion | What |
|---|---|
| ![](https://dastanzar.github.io/portfolio-novel/assets/bandit_race.gif) | `bandit-spec` tracking oracle K |
| ![](https://dastanzar.github.io/portfolio-novel/assets/block_evict.gif) | `block-evict` freeing whole blocks |
| ![](https://dastanzar.github.io/portfolio-novel/assets/gate_gain.gif) | `gate-keep` memory beats drop |

Videos: record 30s per build with ScreenToGif / OBS → `docs/assets/*.mp4` → embed as `<video>` on Pages + link here. YouTube: upload playlist “LLM Efficiency ×10” and add thumbnails below.

## 🧪 The 10 — results, working, implementation, applications

> Images link to full PNGs. Numbers from `receipts/receipt.json` (reproduce: `pytest + demos/demo.py`).

| # | Build | Result | Working | Implementation | Applications |
|---|---|---|---|---|---|
| 1 | [**kurt-switch**](https://github.com/DastanZar/kurt-switch) ![](https://dastanzar.github.io/portfolio-novel/assets/kurt-switch.png) | 79.9% MSE cut | kurtosis-switched 3-codebook + LS scale | `compute_row_stats`, `select_codebook` | Q4 on-device serving |
| 2 | [**shell-vq**](https://github.com/DastanZar/shell-vq) ![](https://dastanzar.github.io/portfolio-novel/assets/shell-vq.png) | 64.5% vs flat | radius shells + per-shell scales + 1-bit residual | `quantize_shelled`, `residual_refine` | sub-2-bit compression |
| 3 | [**dual-bias-fp8**](https://github.com/DastanZar/dual-bias-fp8) ![](https://dastanzar.github.io/portfolio-novel/assets/dual-bias-fp8.png) | 2× accuracy @ same BW | per-block E5M2/E4M3 + bias 11–19 | `choose_format`, `gemv_reference` | no-FP8 GPUs |
| 4 | [**bandit-spec**](https://github.com/DastanZar/bandit-spec) ![](https://dastanzar.github.io/portfolio-novel/assets/bandit-spec.png) | regret 13.7 | Thompson over K + p99 SLO veto | `thompson_pick`, `slo_guard` | SLO chat serving |
| 5 | [**fair-batch**](https://github.com/DastanZar/fair-batch) ![](https://dastanzar.github.io/portfolio-novel/assets/fair-batch.png) | Jain 0.950, p99 −13% | DRR + DVFS + KV admission | `simulate_fair_continuous` | multi-tenant |
| 6 | [**block-evict**](https://github.com/DastanZar/block-evict) ![](https://dastanzar.github.io/portfolio-novel/assets/block-evict.png) | frees whole blocks | whole-block scoring, COW intact | `BlockCache.block_evict` | vLLM long-ctx |
| 7 | [**gate-keep**](https://github.com/DastanZar/gate-keep) ![](https://dastanzar.github.io/portfolio-novel/assets/gate-keep.png) | +11.8% @ b8 | ridge-gate + 8-d latent memory | `fit_gate_ridge`, `compress_with_memory` | 32k chat |
| 8 | [**epi-norm**](https://github.com/DastanZar/epi-norm) ![](https://dastanzar.github.io/portfolio-novel/assets/epi-norm.png) | 52.8% HBM saved | Norm+residual+SwiGLU fused, recompute | `fused_epilogue_ref`, `hbm_accountant` | training kernels |
| 9 | [**cache-roof**](https://github.com/DastanZar/cache-roof) ![](https://dastanzar.github.io/portfolio-novel/assets/cache-roof.png) | 81.5% of peak | L2-split + dequant-overlap + CIs | `cache_adjusted_roof` | kernel pick |
| 10 | [**green-roof**](https://github.com/DastanZar/green-roof) ![](https://dastanzar.github.io/portfolio-novel/assets/green-roof.png) | $3.59/1M, 393g CO₂ | $/J/CO₂ + autosolver + fair_split | `replicas_for_slo` | capacity quotes |

<details><summary>📦 How each is structured (portfolio standard)</summary>

```
<build>/ {core package, tests/ (5–8 passing), demos/demo.py → receipts/receipt.json + assets/*.png, README with attribution, pyproject, LICENSE}
```
</details>

## 📊 Stats (inspo: awesome-github-profile-readme + r/github minimal threads)

<p align="center">
  <img height=160 src="https://github-readme-stats.vercel.app/api?username=DastanZar&show_icons=true&theme=dark&hide_border=true&title_color=C4F03C&icon_color=C4F03C" alt="stats"/>
  <img height=160 src="https://github-readme-streak-stats.herokuapp.com?user=DastanZar&theme=dark&hide_border=true&ring=C4F03C" alt="streak"/>
</p>
<p align="center">
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=DastanZar&layout=compact&theme=dark&hide_border=true" alt="langs"/>
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=DastanZar&theme=github-compact&hide_border=true&color=C4F03C" alt="activity"/>
</p>
<p align="center"><img src="https://github-profile-trophy.vercel.app/?username=DastanZar&theme=flat&no-frame=true&column=6" alt="trophies"/></p>

## 🛠 Stack

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="py"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="torch"/>
<img src="https://img.shields.io/badge/Triton-505050?style=flat" alt="triton"/>
<img src="https://img.shields.io/badge/vLLM-646464?style=flat" alt="vllm"/>
<img src="https://img.shields.io/badge/Matplotlib-11557c?style=flat" alt="mpl"/>
<img src="https://img.shields.io/badge/GitHub_Pages-222222?style=flat&logo=githubpages" alt="pages"/>
</p>

## 🚀 Deploy checklist

1. `git init portfolio-novel && push` → create 10 repos (or one monorepo) + enable Pages from `/docs`
2. Create repo `DastanZar/DastanZar` → paste this file as `README.md`
3. Replace `dastanzar.github.io` URLs after first Pages deploy; add MP4s + YouTube links
4. Archive the 10 old templated repos (Settings → Archive) so portfolio shows only novel work

<p align="center"><img src="https://capsule-render.vercel.app/api?type=waving&color=0:11161d,100:C4F03C&height=120&section=footer" alt="footer"/></p>
