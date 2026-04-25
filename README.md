---
title: Meta Ad Optimizer
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# Meta Ad Optimizer — OpenEnv RL Environment

An OpenEnv reinforcement-learning environment that simulates ad delivery across **Instagram** and **Facebook**. An agent learns to optimise ad placement, creative selection, format choice, and impression frequency to maximise engagement while managing user fatigue.

## Why This Matters

Ad delivery is Meta's core business. Every day billions of impression opportunities arise across Feed, Reels, Stories, Explore, Marketplace, and Search. This environment lets an RL agent learn the same trade-offs a real ad system faces:

- **Which creative** resonates with which user segment?
- **Which surface and format** produce the highest CTR and view time?
- **When to skip** an impression to avoid fatigue?
- **How to balance** competing objectives (CTR vs. view duration vs. user satisfaction)?

## Quick Start

```bash
# Install
pip install openenv-core numpy

# Clone and install this environment
cd meta_ad_optimizer
uv sync  # or: pip install -e .

# Run the server
uvicorn meta_ad_optimizer.server.app:app --host 0.0.0.0 --port 8000

# Run baselines (no server required)
python -m meta_ad_optimizer.baseline --episodes 100 --seed 42
```

### Interact via Python

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Reset with a task
        await ws.send(json.dumps({"type": "reset", "data": {"task": "creative_matcher"}}))
        obs = json.loads(await ws.recv())["data"]["observation"]
        print(obs["user_segment"], obs["available_creatives"])

        # Take a step
        await ws.send(json.dumps({
            "type": "step",
            "data": {
                "show_ad": True,
                "creative_id": 0,
                "platform": "instagram",
                "placement": "feed",
                "ad_format": "image",
            }
        }))
        result = json.loads(await ws.recv())["data"]
        print(result["reward"], result["observation"]["last_action_metrics"])

asyncio.run(main())
```

## Action Space

```
AdAction:
  show_ad      bool     Whether to show an ad (False = skip, recover fatigue)
  creative_id  int      Index into the per-episode creative pool (0 to N-1)
  platform     str      "instagram" | "facebook"
  placement    str      Surface — see valid placements below
  ad_format    str      "image" | "video" | "carousel" | "reel" | "collection"
```

### Valid Placements

| Platform  | Surfaces                                              |
|-----------|-------------------------------------------------------|
| Instagram | feed, reels, stories, explore, search                 |
| Facebook  | feed, reels, stories, marketplace, search, right_column |

### Valid Formats per Surface

| Surface     | Allowed Formats                     |
|-------------|-------------------------------------|
| Feed        | image, video, carousel, reel        |
| Reels       | reel                                |
| Stories     | image, video, reel, collection      |
| Explore     | image, video, carousel, reel        |
| Search      | image, video                        |
| Marketplace | image, carousel, collection         |
| Right Column| image                               |

Invalid format-surface combinations receive a penalty reward.

## Observation Space

```
AdObservation:
  task                str         Active task name
  user_segment        str         User archetype (e.g. "gen_z_creator")
  user_interests      list[str]   Interest categories
  user_device         str         "mobile" | "desktop" | "tablet"
  current_platform    str         Platform the user is browsing
  current_surface     str         Surface the user is on right now
  available_creatives list[dict]  Pool of ad creatives with properties
  impression_count    int         Ads shown so far this session
  fatigue_level       float       0.0–1.0 fatigue meter
  step / total_steps  int         Current step and episode length
  last_action_metrics dict        Engagement from the previous step
  session_metrics     dict        Cumulative CTR, view time, satisfaction
```

### Creative Properties

Each creative in the pool has:
- `category` — product category (fashion, electronics, health, etc.)
- `tone` — humorous, informational, emotional, action_oriented
- `target_segment` — which user segment it was designed for
- `base_ctr` — baseline click-through probability
- `base_view_time` — baseline expected view duration (seconds)

Creatives are sampled from a master catalog of 80 diverse ads each episode.

## User Segments

| Segment             | Interests                         | Platform Bias           |
|---------------------|-----------------------------------|-------------------------|
| gen_z_creator       | fashion, music, memes, beauty     | 80% Instagram           |
| millennial_parent   | kids, home decor, deals, family   | 60% Facebook            |
| business_pro        | SaaS, finance, productivity, B2B  | 80% Facebook            |
| casual_scroller     | entertainment, food, travel       | 60% Instagram           |
| fitness_enthusiast  | health, sports, outdoors          | 70% Instagram           |
| bargain_hunter      | deals, coupons, electronics       | 70% Facebook            |

## Three Tasks (Easy → Medium → Hard)

### Task 1: Creative Matcher (Easy)

| Property       | Value               |
|----------------|---------------------|
| Platform       | Instagram (fixed)   |
| Surface        | Feed (fixed)        |
| Format         | Image (fixed)       |
| Creatives      | 4 per episode       |
| Steps          | 10                  |
| Fatigue        | Disabled            |
| Skip allowed   | No                  |

**Agent only picks `creative_id`.** Grader = session CTR.

### Task 2: Placement Optimizer (Medium)

| Property       | Value               |
|----------------|---------------------|
| Platform       | Instagram (fixed)   |
| Surfaces       | All 5 IG surfaces   |
| Formats        | All 5 formats       |
| Creatives      | 8 per episode       |
| Steps          | 15                  |
| Fatigue        | Light               |
| Skip allowed   | Yes                 |

**Agent picks creative + surface + format + skip.** Grader = 30% validity + 70% engagement.

### Task 3: Campaign Optimizer (Hard)

| Property       | Value                  |
|----------------|------------------------|
| Platforms      | Instagram + Facebook   |
| Surfaces       | All surfaces           |
| Formats        | All formats            |
| Creatives      | 12 per episode         |
| Steps          | 20                     |
| Fatigue        | Aggressive             |
| Skip allowed   | Yes                    |
| Transitions    | User drifts between surfaces |

**Full action space.** Grader = 15% validity + 25% CTR + 20% view time + 25% satisfaction + 15% fatigue management.

## Reward Function

Per-step reward (multi-objective):

```
reward = 0.35 * click     (1.0 if clicked, 0.0 otherwise)
       + 0.25 * view_time (normalised to 0–1)
       + 0.25 * satisfaction (relevance × non-intrusiveness)
       + 0.15 * (-fatigue)  (penalises high fatigue)
```

Skip action: +0.05 fatigue recovery − 0.02 missed revenue = +0.03.

Invalid format-surface combo: −0.2 penalty.

## Baseline Scores

100 episodes, seed=42 (`python -m meta_ad_optimizer.baseline --episodes 100 --seed 42`):

```
Task: creative_matcher (Easy)
  Random          0.1720 ± 0.1393
  Greedy          0.2430 ± 0.2701
  Rule-Based      0.3730 ± 0.3021

Task: placement_optimizer (Medium)
  Random          0.4859 ± 0.1187
  Greedy          0.7141 ± 0.1282
  Rule-Based      0.7333 ± 0.1526

Task: campaign_optimizer (Hard)
  Random          0.2800 ± 0.0451
  Greedy          0.3529 ± 0.1315
  Rule-Based      0.5899 ± 0.0853
```

### LLM Baseline (optional)

Run an OpenAI-compatible LLM as the agent:

```bash
pip install openai
OPENAI_API_KEY=sk-... python -m meta_ad_optimizer.baseline --episodes 5 --seed 42 --llm --model gpt-4o-mini
```

The LLM agent receives the full observation as a structured prompt and returns a JSON action. It demonstrates that the environment is compatible with LLM-based RL training pipelines.

## Simulation Details

### Engagement Model

```
effective_ctr = base_ctr × segment_affinity × placement_factor
              × context_bonus × platform_match × format_factor
              × synergy × segment_modifier × (1 − fatigue)
```

- **Segment affinity**: 2.0× if creative targets user's segment, 1.2× if category matches interest, 0.3× otherwise
- **Context bonus**: +20% if ad placed on the surface user is browsing
- **Platform mismatch**: −50% if ad targets wrong platform
- **Format-surface synergy**: Reel on Reels +35%, Collection on Marketplace +40%, etc.

### Fatigue Model

```
show_ad  → fatigue += increment × (1 + 0.1 × impressions_shown)
skip     → fatigue -= recovery
```

### Surface Transitions (Hard task only)

Each step, the user may drift to another surface based on a Markov transition matrix (e.g., Feed → Reels 20%, Feed → Stories 15%).

## Project Structure

```
meta_ad_optimizer/
  __init__.py          # Package exports
  models.py            # AdAction, AdObservation, AdState
  simulation.py        # User segments, creative catalog, engagement engine
  tasks.py             # 3 task configs + grader functions (0.0–1.0)
  client.py            # WebSocket client (AdEnv)
  baseline.py          # Baseline inference script
  openenv.yaml         # OpenEnv manifest
  pyproject.toml       # Dependencies
  server/
    ad_environment.py  # AdOptimizerEnvironment (reset/step/state)
    rubrics.py         # Trajectory-level scoring rubric
    app.py             # FastAPI app
    Dockerfile         # Container image
```

## Deployment

```bash
# Local development
uvicorn meta_ad_optimizer.server.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t meta-ad-optimizer -f server/Dockerfile .
docker run -p 8000:8000 meta-ad-optimizer

# Hugging Face Spaces
openenv push
```

## Requirements

- Python 3.10+
- openenv-core >= 0.2.1
- numpy >= 1.24.0
- pydantic >= 2.0.0
- fastapi >= 0.115.0
- uvicorn >= 0.24.0

## License

BSD 3-Clause License

---

# AdMarket Arena — Multi-Agent Long-Horizon Ad Auction

> **Round 2 upgrade** built on top of the single-agent env above.
> Targets **Theme 1 (Multi-Agent)** + **Theme 2 (Long-Horizon Planning)**.

## Problem

Real-time bidding is a multi-agent problem: every impression slot is a sealed auction where competing advertisers submit bids simultaneously and the outcome depends on everyone's strategy. A single-agent env can't capture this dynamic. AdMarket Arena puts one trained LLM advertiser against four scripted `PersonaBot` opponents in a 7-day Vickrey (second-price) auction campaign.

The long-horizon challenge: a 7-day, 350-step episode far exceeds any LLM's context window. The agent must learn to plan across days — managing fatigue, pacing budget, and adjusting bids based on multi-day ROAS trends — without access to raw step history.

## Environment

```
Episode = 7 days × 50 impressions/day = 350 auction steps

Each step:
  1. User sampled from 100-user pool (repeat exposures → frequency caps matter)
  2. All 5 advertisers submit bids simultaneously
  3. Winner = highest bid; pays second-highest (Vickrey / second-price)
  4. Winner's creative shown → compute_engagement() → click / view / fatigue
  5. Per-step reward emitted to trained agent (opponents are scripted, no reward)
  6. Every 50 steps: daily pacing bonus + day recap injected into next observation
  7. Step 350: weekly ROAS terminal reward (5.0× weight, dominant signal)
```

### What makes it novel

| Feature | Why it matters |
|---|---|
| **Vickrey auction** | Truth-telling equilibrium; agent must learn bid shading against different personas |
| **5 scripted PersonaBots** with per-episode trait jitter | Agent must generalise across opponent archetypes, not memorise exact bid values |
| **`yesterday_recap` in observation** | ~200-token LLM-generated day summary; agent plans across 7 days without full history |
| **Composable rubric system** | PerStep + Daily + Weekly rewards ablatable independently; clean separation of signal horizons |
| **OversightAgent** (Fleet AI bonus) | Separate model reads auction logs and flags freq-cap / budget / shill-bidding violations |

### Reward decomposition

```
Per step  (dense):   PerStepEngagementRubric   won+clicked: +(2.0 - price)
                                                won+no click: -(price × 0.10)
                                                skipped:      +0.02
                                                over budget:  -0.50
Per day   (medium):  DailyPacingRubric          max +0.50  (pacing quality × daily ROAS)
Per week  (sparse):  WeeklyROASRubric           5.0 × min(1.5, weekly_roas / 2.0)
                                                overspend penalty: -2.0
                                                underspend penalty: -2.0
```

The 5.0× weekly weight is intentionally larger than the sum of all per-step rewards (~35), forcing the agent to treat ROAS as the primary objective.

### Three difficulty tiers

| Task | Days | Slots/day | Steps | Competitors | Budget |
|---|---|---|---|---|---|
| `arena_easy` | 3 | 20 | 60 | 3 PersonaBots | $300 |
| `arena_medium` | 5 | 30 | 150 | 4 PersonaBots | $500 |
| `arena_hard` | 7 | 50 | 350 | 5 PersonaBots | $1000 |

## Arena Quick Start

```python
from meta_ad_optimizer.client import AdMarketArenaEnv
from meta_ad_optimizer.models import AuctionAction

async with AdMarketArenaEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="arena_easy")
    obs = result.observation
    print(obs.user_segment, obs.budget_remaining, obs.floor_price)

    # Bid 1.20 CPM on this slot
    result = await env.step(AuctionAction(skip=False, bid_amount=1.20, creative_id=0))
    print(result.observation.last_auction_result)
    print(result.observation.yesterday_recap)  # Theme 2 recap on day boundaries
```

```bash
# Run arena baselines (no server required — env runs in-process)
python -m meta_ad_optimizer.baseline --arena --task arena_easy --episodes 5 --seed 42
```

Expected output:
```
=== AdMarket Arena — arena_easy (5 episodes, seed=42) ===
  Agent              weekly_roas        reward
  --------------------------------------------
  ArenaRandom           0.XXX±0.XXX    XX.XX   ← placeholder
  ArenaGreedy           0.XXX±0.XXX    XX.XX   ← placeholder
  ArenaPacing           0.XXX±0.XXX    XX.XX   ← placeholder

  Ordering check (pacing >= greedy >= random): PASS
```

## Arena Baselines (placeholder — Plan 4 fills real numbers)

| Agent | arena_easy ROAS | arena_hard ROAS | Notes |
|---|---|---|---|
| ArenaRandom | — | — | Random bids, 20% skip rate |
| ArenaGreedy | — | — | Always bids 5.0; ignores fatigue |
| ArenaPacing | — | — | Budget-proportional bids; skips fatigued segments |
| LLM (trained) | — | — | Qwen2.5-3B fine-tuned with GRPO |

## Training (GRPO on Colab T4)

```python
# See train_grpo.ipynb — Plan 3 fills this
# Model: Qwen/Qwen2.5-3B-Instruct, 4-bit via Unsloth
# Curriculum: arena_easy (60 steps) → arena_medium → arena_hard (350 steps)
# Expected: ~2–4 hours on Colab free T4 for arena_easy convergence
```

## Links (placeholder — Plan 4 fills)

- HuggingFace Space: `[YOUR_HF_SPACE_URL]`
- Trained checkpoint: `[YOUR_HF_MODEL_URL]`
- Demo video: `[DEMO_URL]`
- Blog post: `[BLOG_URL]`
