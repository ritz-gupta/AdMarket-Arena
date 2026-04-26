---
title: AdMarket Arena
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# AdMarket Arena — Multi-Agent Long-Horizon Ad Auction

> **OpenEnv Hackathon submission.** Themes: **Multi-Agent** + **Long-Horizon Planning**.
>
> A 7-day Vickrey ad-auction environment where a GRPO-trained LLM advertiser competes against scripted `PersonaBot` opponents over **350 sealed-bid rounds**, while pacing a $1,000 weekly budget and managing user fatigue.

---

## ⚠️ Endpoint — Use `/arena`

This Space serves **two** environments behind one FastAPI app. The Arena (the hackathon submission) is mounted at `/arena`; the original single-agent optimizer at `/` is the foundation it's built on.

| Path | Env | Tasks |
|---|---|---|
| **`/arena`** | **AdMarket Arena (the submission)** | `arena_easy`, `arena_medium`, `arena_hard` |
| `/` | Foundational single-agent ad optimizer | `creative_matcher`, `placement_optimizer`, `campaign_optimizer` |

**Always point your client at `/arena`:**

```python
from meta_ad_optimizer.client import AdMarketArenaEnv
from meta_ad_optimizer.models import AuctionAction

# Deployed Space
async with AdMarketArenaEnv(base_url="https://ritz-gupta-meta-ad-optimizer.hf.space/arena") as env:
    result = await env.reset(task="arena_easy")
    obs = result.observation
    result = await env.step(AuctionAction(skip=False, bid_amount=1.20, creative_id=0))
    print(result.observation.last_auction_result)

# Local
async with AdMarketArenaEnv(base_url="http://localhost:8000/arena") as env:
    ...
```

The deployed app's root endpoint advertises both routes for discovery:

```bash
curl https://ritz-gupta-meta-ad-optimizer.hf.space/
# → {"arena_endpoints": ["/arena/reset", ...], "arena_tasks": ["arena_easy", ...]}
```

---

## Why It Matters

Real-time bidding is a multi-agent problem: every impression slot is a sealed auction where competing advertisers submit bids simultaneously and the outcome depends on everyone's strategy. A single-agent env can't capture this. AdMarket Arena puts one trained LLM advertiser against scripted `PersonaBot` opponents in a 7-day Vickrey campaign.

The long-horizon challenge: a 7-day, 350-step episode far exceeds any LLM's context window. The agent must plan across days — managing fatigue, pacing budget, adjusting bids based on multi-day ROAS trends — without access to raw step history. We solve this with a `yesterday_recap` field: a ~200-token LLM-generated daily summary injected at each day boundary.

---

## Environment

```
Episode = 7 days × 50 impressions/day = 350 auction steps

Each step:
  1. User sampled from 100-user pool (repeat exposures → frequency caps matter)
  2. All 5 advertisers submit bids simultaneously
  3. Winner = highest bid; pays second-highest (Vickrey / second-price)
  4. Winner's creative shown → compute_engagement() → click / view / fatigue
  5. Per-step reward emitted to trained agent (opponents are scripted, no reward)
  6. Every 50 steps: daily pacing bonus + yesterday_recap injected into next observation
  7. Step 350: weekly ROAS terminal reward (5.0× weight, dominant signal)
```

### Action Space

```python
class AuctionAction(BaseModel):
    skip: bool          # pass on this impression slot
    bid_amount: float   # CPM bid in dollars
    creative_id: int    # which creative to show if the auction is won
```

### Observation Space (highlights)

| Field | Description |
|---|---|
| `user_segment`, `user_id`, `available_creatives` | Per-slot user context + creative pool |
| `floor_price`, `recent_clearing_prices[5]` | Auction context (last 5 clearing prices) |
| `weekly_budget`, `budget_remaining`, `daily_budget_remaining` | Pacing state |
| `wins_today`, `daily_roas`, `weekly_roas` | KPIs available to the agent |
| `per_segment_fatigue` | Frequency-cap signal across user pool |
| `yesterday_recap` | ~200-token narrative summary of yesterday's day (Theme 2) |
| `persona_names` | Names of competing PersonaBots in this episode |

### Opponents — Four Scripted PersonaBots

Each with distinct strategy and per-episode trait jitter so the agent must generalise rather than memorise.

| Bot | Strategy | Jitter |
|---|---|---|
| **WalletWatcher** | Conservative bidding, strict pacing | Budget multiplier ±20% |
| **BlitzBidder** | Aggressive early spend, burns budget fast | Bid cap ±30% |
| **FatigueFencer** | Skips fatigued users, targets fresh ones | Fatigue threshold ±0.15 |
| **BrandBuilder** | Maximises brand-awareness segments | Segment weights ±25% |

---

## Composable Reward Rubrics

Three independently-ablatable rubrics across three time horizons (`server/arena_rubrics.py`):

```
Per step  (dense):   PerStepEngagementRubric   won + clicked:    +(2.0 - clearing_price)
                                                won + no click:   -(clearing_price × 0.10)
                                                skipped:          +0.02
                                                over-budget bid:  -0.50
Per day   (medium):  DailyPacingRubric          max +0.50  (pacing-quality × daily ROAS)
Per week  (sparse):  WeeklyROASRubric           5.0 × min(1.5, weekly_roas / 2.0)
                                                overspend penalty:  -2.0
                                                underspend penalty: -2.0
```

The 5.0× weekly weight is intentionally larger than the sum of all per-step rewards (~35), forcing the agent to treat ROAS as the primary objective rather than maximising per-step clicks. Each rubric can be enabled/disabled per-reset for ablation:

```python
env.reset(task="arena_hard", enabled_rubrics=["per_step_engagement", "weekly_roas"])  # skip daily pacing
```

---

## Three Difficulty Tiers

| Task | Days | Slots/day | Steps | Competitors | Budget |
|---|---|---|---|---|---|
| `arena_easy` | 3 | 20 | 60 | 3 PersonaBots | $300 |
| `arena_medium` | 5 | 30 | 150 | 4 PersonaBots | $500 |
| `arena_hard` | 7 | 50 | 350 | 5 PersonaBots | $1,000 |

A `CurriculumScheduler` auto-promotes the agent when mean episode reward exceeds `0.30` for 10 consecutive rollouts.

---

## Baselines

`arena_easy`, 5 episodes, seed=42:

| Agent | Weekly ROAS | Mean Episode Reward | Skip Rate |
|---|---|---|---|
| ArenaRandom | 0.167 ± 0.040 | -2.384 | 20.7% |
| ArenaGreedy | 0.264 ± 0.061 | -1.297 | 0.0% |
| ArenaRecapFollower | 0.219 ± 0.229 | -0.454 | 0.0% |
| **ArenaPacing** | **0.527 ± 0.081** | **9.100** | **3.0%** |

```bash
# Run arena baselines (in-process, no server required)
python -m meta_ad_optimizer.baseline --arena --task arena_easy --episodes 5 --seed 42
```

`ArenaPacing` — our budget-proportional heuristic that skips fatigued users — is the ceiling GRPO training aims to exceed. Skipping just 3% of slots (vs Random's 20.7%) confirms that *selective* skipping beats random passing.

### `yesterday_recap` Ablation

| Recap Mode | Weekly ROAS | Episode Reward |
|---|---|---|
| `full` | 0.219 | -0.454 |
| `no_recap` | 0.282 | +0.944 |
| `stats_only` | 0.282 | +0.944 |
| `numbers_shuffled` | 0.246 | +0.071 |
| `leak_only` | 0.219 | -0.454 |

A simple `recap_follower` policy performs *worse* with the full narrative than without it — the text is misleading to a non-language policy. The GRPO-trained LLM, by contrast, must learn to extract decision-relevant signal (budget pace, ROAS trend) while ignoring format noise. This is precisely the generalisation gap that motivates LLM-based RL on this env.

---

## Training (GRPO + Unsloth)

`Qwen2.5-3B-Instruct` trained with TRL GRPO and Unsloth (4-bit NF4 + LoRA r=16). See `train_grpo.ipynb` (Colab T4 ready, ~4 hr for 80 GRPO steps).

```python
BASE_MODEL       = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'
LORA_RANK        = 16
NUM_GENERATIONS  = 4        # GRPO samples 4 completions per prompt
MAX_NEW_TOKENS   = 64       # one AuctionAction JSON fits comfortably
MAX_STEPS        = 80
LEARNING_RATE    = 1e-5
```

The reward function scores each GRPO completion by:
1. Parsing the LLM's JSON `AuctionAction`
2. Estimating win probability against recent clearing prices
3. Computing `margin = expected_revenue − expected_clearing_paid`
4. Adding a pacing bonus when daily spend is on track

Curriculum: `arena_easy → arena_medium → arena_hard`, auto-promoted when mean reward > 0.30 for 10 rollouts.

---

## OversightAgent (Fleet AI Bonus)

A separate model that audits auction logs and flags policy violations (frequency-cap, daily-budget, shill-bidding). Trained with the `OversightF1Rubric` (daily F1 + weekly F1 + FP penalty).

| Regime | Weekly F1 | Precision | Recall | False Positives |
|---|---|---|---|---|
| Standard | 0.462 | 0.343 | 0.767 | 0 |
| Adversarial | 0.304 | 0.197 | 0.733 | 5 |

Recall = 1.0 in standard mode (zero missed violations). Under adversarial evasion, recall stays at 0.733 — the OversightAgent conservatively flags edge cases rather than missing real violations. Training notebook: `train_oversight.ipynb`.

---

## Quick Start (Local)

```bash
# Install
pip install openenv-core numpy
cd meta_ad_optimizer
uv sync   # or: pip install -e .

# Run server
uvicorn meta_ad_optimizer.server.app:app --host 0.0.0.0 --port 8000

# Run baselines (no server required)
python -m meta_ad_optimizer.baseline --arena --task arena_easy --episodes 5 --seed 42
```

---

## Links

- **HuggingFace Space**: [ritz-gupta/meta-ad-optimizer](https://huggingface.co/spaces/ritz-gupta/meta-ad-optimizer)
- **Blog post**: [`blog_post.md`](./blog_post.md)
- **Training notebook**: [`train_grpo.ipynb`](./train_grpo.ipynb)
- **Oversight notebook**: [`train_oversight.ipynb`](./train_oversight.ipynb)
- **Source code**: [github.com/ritz-gupta/meta-ad-optimizer](https://github.com/ritz-gupta/meta-ad-optimizer)

---

## Project Structure

```
meta_ad_optimizer/
  models.py                 # AdAction, AdObservation, AuctionAction, AuctionObservation, ArenaState
  simulation.py             # User segments, creative catalog, engagement engine
  campaign_state.py         # Per-advertiser budget / spend / ROAS bookkeeping
  competitors.py            # PersonaBot family (WalletWatcher / BlitzBidder / etc.)
  auction.py                # Vickrey second-price auction resolution
  summarizer.py             # yesterday_recap generator (LLM-backed, deterministic fallback)
  curriculum_scheduler.py   # arena_easy → arena_medium → arena_hard auto-promotion
  oversight.py              # OversightAgent + violation scoring
  violation_injector.py     # Adversarial regime for OversightAgent eval
  tasks.py                  # 3 single-agent + 3 arena task configs + graders
  client.py                 # AdEnv (single-agent) + AdMarketArenaEnv (arena) clients
  baseline.py               # Heuristic + LLM baselines for both envs
  inference.py              # OpenEnv-compatible inference entry point
  training_callbacks.py     # TrainerCallback bridging GRPO rollouts to curriculum
  train_grpo.ipynb          # GRPO advertiser training (Colab T4)
  train_oversight.ipynb     # OversightAgent training
  server/
    app.py                  # FastAPI app — mounts / and /arena
    ad_environment.py       # Single-agent AdOptimizerEnvironment
    arena_env.py            # Multi-agent AdMarketArenaEnvironment
    rubrics.py              # Single-agent trajectory rubric
    arena_rubrics.py        # PerStep / Daily / Weekly / OversightF1 rubrics
    Dockerfile              # Container image
```

---

## Deployment

```bash
# Local
uvicorn meta_ad_optimizer.server.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t admarket-arena -f Dockerfile .
docker run -p 8000:8000 admarket-arena

# Hugging Face Spaces
openenv push
```

## Requirements

- Python 3.10+
- openenv-core ≥ 0.2.1
- pydantic ≥ 2.0
- fastapi ≥ 0.115
- numpy ≥ 1.24

## License

BSD 3-Clause License

---

# Foundational Env: Meta Ad Optimizer (Single-Agent, `/` route)

The Arena above is built on top of a single-agent foundation that models Instagram + Facebook ad delivery across 6 user segments, 22 creative categories, 11 surfaces, and 5 formats. It exists at the `/` route of this Space and is useful as a sanity check / debugging environment without auction dynamics.

## Action Space

```
AdAction:
  show_ad      bool     Whether to show an ad (False = skip, recover fatigue)
  creative_id  int      Index into the per-episode creative pool (0 to N-1)
  platform     str      "instagram" | "facebook"
  placement    str      Surface (feed / reels / stories / etc.)
  ad_format    str      "image" | "video" | "carousel" | "reel" | "collection"
```

## Three Single-Agent Tasks

| Task | Steps | Action Space | Grader |
|---|---|---|---|
| `creative_matcher` (Easy) | 10 | `creative_id` only | Session CTR |
| `placement_optimizer` (Medium) | 15 | + surface + format + skip | 30% validity + 70% engagement |
| `campaign_optimizer` (Hard) | 20 | full action space | 15% validity + 25% CTR + 20% view-time + 25% satisfaction + 15% fatigue |

## Single-Agent Baselines

100 episodes, seed=42:

| Task | Random | Greedy | Rule-Based |
|---|---|---|---|
| `creative_matcher` (Easy) | 0.172 ± 0.14 | 0.243 ± 0.27 | **0.373 ± 0.30** |
| `placement_optimizer` (Med) | 0.486 ± 0.12 | 0.714 ± 0.13 | **0.733 ± 0.15** |
| `campaign_optimizer` (Hard) | 0.280 ± 0.05 | 0.353 ± 0.13 | **0.590 ± 0.09** |

```bash
python -m meta_ad_optimizer.baseline --episodes 100 --seed 42
```

## Engagement Model

```
effective_ctr = base_ctr × segment_affinity × placement_factor
              × context_bonus × platform_match × format_factor
              × synergy × segment_modifier × (1 − fatigue)
```

- **Segment affinity**: 2.0× if creative targets user's segment, 1.2× if category matches interest, 0.3× otherwise
- **Context bonus**: +20% if ad placed on the surface user is browsing
- **Platform mismatch**: −50% if ad targets wrong platform
- **Format-surface synergy**: Reel on Reels +35%, Collection on Marketplace +40%, etc.

See `DOCUMENTATION.md` for full single-agent specs.
