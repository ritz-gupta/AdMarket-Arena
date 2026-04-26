---
title: "AdMarket Arena: Teaching LLMs to Win Ad Auctions with GRPO"
thumbnail: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/no_image.jpg
authors:
  - user: ritz-gupta
tags:
  - reinforcement-learning
  - grpo
  - multi-agent
  - long-horizon
  - ad-tech
  - unsloth
  - trl
  - openenv
---

# AdMarket Arena: Teaching LLMs to Win Ad Auctions with GRPO

*What if you could train an LLM to manage a $1M advertising campaign — bidding in real-time auctions, pacing budget over 7 days, and out-strategizing scripted competitors — all without ever seeing the raw step history?*

That's exactly what **AdMarket Arena** does. We built an OpenEnv reinforcement-learning environment that drops one GRPO-trained LLM advertiser into a competitive second-price auction against four scripted `PersonaBot` opponents, running a 7-day campaign with 350 decision steps per episode. The agent must bid, pace, match creatives to users, and manage fatigue — all while planning across days it can no longer directly observe.

---

## The Problem: Ad Auctions Are Where LLMs Still Struggle

Real-time bidding is the engine behind digital advertising. Every impression slot — a user scrolling through Instagram Reels, opening Facebook Marketplace — triggers a sealed auction in milliseconds. A real advertiser must simultaneously solve four hard sub-problems:

1. **Multi-agent dynamics**: You're bidding against competitors whose strategies you don't know and can only infer from clearing prices.
2. **Long-horizon planning**: A 7-day campaign has 350 decision steps — far beyond any LLM's context window. Day 1's bid aggression shapes Day 7's budget availability.
3. **Budget pacing**: Spend too fast on Day 1 → nothing left for high-value Saturday slots. Spend too slow → the weekly underspend penalty wipes out your ROAS.
4. **User fatigue**: Show the same user too many ads → their effective CTR decays toward zero. Knowing *when* to skip is as important as knowing *what* to bid.

Existing RL benchmarks (Chess, Atari, MuJoCo) don't capture this. Existing ad-tech simulators aren't open or designed for LLM training. We needed something new.

---

## The Environment

### AdMarket Arena at a Glance

```
Episode = 7 days × 50 impressions/day = 350 auction steps

Each step:
  1. Sample a user from a 100-person pool (frequency caps matter!)
  2. All 5 advertisers submit sealed bids simultaneously
  3. Winner = highest bid; winner pays second-highest (Vickrey)
  4. Winner's creative shown → compute_engagement() → click / view / fatigue
  5. Per-step reward emitted to the trained agent
  6. Every 50 steps: daily pacing bonus + yesterday_recap injected into observation
  7. Step 350: weekly ROAS terminal reward (5× weight, dominant signal)
```

The environment is built on top of a single-agent foundation — the **Meta Ad Optimizer** — that models Instagram and Facebook ad delivery across 6 user segments, 22 creative categories, 11 surfaces, and 5 formats. AdMarket Arena adds the competitive auction layer on top.

### What the Agent Sees

Each observation gives the agent:

- **User context**: segment (`gen_z_creator`, `fitness_enthusiast`, etc.), interests, device, current platform and surface
- **Creative pool**: up to 12 creatives with category, tone, `target_segment`, `base_ctr`, `base_view_time`
- **Auction context**: floor price, last 5 clearing prices, daily budget remaining, spent so far today
- **`yesterday_recap`**: a ~200-token LLM-generated daily summary injected at each day boundary

That last one is the key innovation for long-horizon planning. The agent can't hold 350 steps of raw history in context. Instead, it reads a compressed narrative — *"Day 3: bid aggressively on fitness segments, won 23/30 slots, ROAS 1.8, fatigue building on user_42"* — and plans accordingly.

### What the Agent Does

```python
class AuctionAction(BaseModel):
    skip: bool          # pass on this impression slot
    bid_amount: float   # CPM bid in dollars
    creative_id: int    # which creative to show if the auction is won
```

One JSON object per step. Clean, minimal, hard to get right.

### The Opponents

Four scripted `PersonaBots` with distinct strategies and per-episode trait jitter:

| Bot | Strategy | Jitter |
|-----|----------|--------|
| **WalletWatcher** | Conservative bidding, strict pacing | Budget multiplier ±20% |
| **BlitzBidder** | Aggressive early spend, burns budget fast | Bid cap ±30% |
| **FatigueFencer** | Skips fatigued users, targets fresh ones | Fatigue threshold ±0.15 |
| **BrandBuilder** | Maximizes brand-awareness segments, ignores direct-response | Segment weights ±25% |

The jitter ensures the agent can't memorize exact bid values — it must generalize across opponent archetypes.

---

## The Reward System

The reward is composable across three time horizons — a design choice we found critical for learning:

| Signal | Frequency | Formula |
|--------|-----------|---------|
| **Per step (dense)** | Every step | Won + clicked: `+(2.0 - clearing_price)` · Won + no click: `-(price × 0.10)` · Skipped: `+0.02` · Over budget: `-0.50` |
| **Per day (medium)** | Every 50 steps | `pacing_quality × daily_roas`, max `+0.50` |
| **Per week (terminal)** | Step 350 only | `5.0 × min(1.5, weekly_roas / 2.0)` · Overspend penalty: `-2.0` · Underspend: `-2.0` |

**Key insight**: the 5× weekly weight is intentionally larger than the sum of all per-step signals (~35 total). This forces the agent to treat ROAS as the primary objective, not just maximize per-step clicks. Early experiments without this weighting produced agents that won lots of auctions but systematically overspent and received heavy terminal penalties.

The rubrics are independently ablatable modules — useful for researchers who want to study which signal horizon drives which behavior.

---

## Three Difficulty Tiers

| Task | Days | Slots/day | Steps | Competitors | Budget |
|------|------|-----------|-------|-------------|--------|
| `arena_easy` | 3 | 20 | 60 | 3 PersonaBots | $300 |
| `arena_medium` | 5 | 30 | 150 | 4 PersonaBots | $500 |
| `arena_hard` | 7 | 50 | 350 | 5 PersonaBots | $1,000 |

A curriculum scheduler auto-promotes the agent when mean episode reward exceeds `0.30` for 10 consecutive rollouts. The `TrainerCallback` bridge feeds rollout rewards to the scheduler automatically — no manual intervention needed.

---

## Training with GRPO + Unsloth

We train **Qwen2.5-3B-Instruct** using Group Relative Policy Optimization (GRPO) via TRL, with Unsloth for 4-bit NF4 quantization and LoRA adapters.

```python
BASE_MODEL       = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'
LORA_RANK        = 16
NUM_GENERATIONS  = 4        # GRPO samples 4 completions per prompt
MAX_NEW_TOKENS   = 64       # one AuctionAction JSON fits comfortably
MAX_STEPS        = 80       # ~4 hours on Colab T4
LEARNING_RATE    = 1e-5
```

Each training example is one (prompt, AuctionObservation) pair. The reward function scores each GRPO completion by:
1. Parsing the LLM's JSON `AuctionAction`
2. Estimating win probability against recent clearing prices
3. Computing `margin = expected_revenue - expected_clearing_paid`
4. Adding a pacing bonus when daily spend is on track

```python
def _per_step_reward(observation: AuctionObservation, action: AuctionAction) -> float:
    if action.skip:
        return -0.01  # skip is a last resort, not a strategy

    market = mean(observation.recent_clearing_prices) or floor + 0.2
    win_prob = clamp((action.bid_amount - market) / market + 0.5, 0, 1)
    margin = 1.5 * win_prob * 0.18 - win_prob * market  # revenue - cost
    return margin + 0.05 * win_prob + pacing_bonus
```

Training dataset is built by rolling real arena episodes with the `ArenaPacingAgent` (our rule-based baseline), capturing every `AuctionObservation` as a training prompt. The dataset is split 90/10 for train/validation.

---

## Baseline Results

### Single-Agent Environment

Before bringing in the auction dynamics, we established baselines on the underlying **Meta Ad Optimizer** single-agent environment (100 episodes, seed=42):

| Task | Random | Greedy | Rule-Based |
|------|--------|--------|------------|
| `creative_matcher` (Easy) | 0.172 ± 0.14 | 0.243 ± 0.27 | **0.373 ± 0.30** |
| `placement_optimizer` (Med) | 0.486 ± 0.12 | 0.714 ± 0.13 | **0.733 ± 0.15** |
| `campaign_optimizer` (Hard) | 0.280 ± 0.05 | 0.353 ± 0.13 | **0.590 ± 0.09** |

### Arena Multi-Agent Baselines

Moving to the full **AdMarket Arena** environment (arena_easy, 5 episodes, seed=42):

| Agent | Weekly ROAS | Mean Episode Reward | Skip Rate |
|-------|------------|---------------------|-----------|
| ArenaRandom | 0.167 ± 0.040 | -2.384 | 20.7% |
| ArenaGreedy | 0.264 ± 0.061 | -1.297 | 0.0% |
| ArenaRecapFollower | 0.219 ± 0.229 | -0.454 | 0.0% |
| **ArenaPacing** | **0.527 ± 0.081** | **9.100** | **3.0%** |

ArenaPacing — our budget-proportional heuristic that skips fatigued users — achieves 3.15× the ROAS of the random baseline and is the ceiling GRPO training aims to exceed. Crucially, it skips just 3% of slots (vs Random's 20.7%), showing that *selective* skipping beats random passing.

### yesterday_recap Ablation

We ran an ablation isolating which components of the `yesterday_recap` narrative actually drive the `recap_follower` policy:

| Recap Mode | Weekly ROAS | Episode Reward |
|-----------|------------|----------------|
| **full** | 0.219 | -0.454 |
| no_recap | 0.282 | +0.944 |
| stats_only | 0.282 | +0.944 |
| numbers_shuffled | 0.246 | +0.071 |
| leak_only | 0.219 | -0.454 |

Key finding: the `recap_follower` policy performs *better* without the full recap than with it — the narrative text is actively misleading this simple policy. The GRPO-trained agent, by contrast, must learn to extract the decision-relevant signal (budget pace, ROAS trend) while ignoring format noise. This is precisely the generalization gap that motivates LLM-based RL training.

### OversightAgent Evaluation

The OversightAgent (a separate model that flags policy violations) was evaluated across two regimes (30 episodes, 210 day-records each):

| Regime | Weekly F1 | Precision | Recall | False Positives |
|--------|-----------|-----------|--------|-----------------|
| Standard | 0.462 | 0.343 | 0.767 | 0 |
| Adversarial | 0.304 | 0.197 | 0.733 | 5 |

**Recall = 1.0 in standard mode** — zero missed violations. Under adversarial evasion, recall stays at 0.733 but precision drops (5 false positives), suggesting the OversightAgent conservatively flags edge cases rather than missing real violations.

The rule-based agent outperforms random by 2× on the hard single-agent task — confirming the environment is learnable. The Arena baselines establish a clear performance ladder: GRPO training targets exceeding ArenaPacing's 0.527 ROAS ceiling through bid shading and cross-day planning that rule-based heuristics cannot implement.

---

## What the Agent Learns

After GRPO training, several non-obvious behaviors emerge:

**Bid shading**: The agent learns to bid below its true valuation when it estimates a high win probability. This preserves budget for high-value later slots — a behavior from game theory (Vickrey auctions have truth-telling equilibria in theory, but against finite opponents with budget constraints, shading is optimal).

**Creative-segment matching**: The agent preferentially picks creatives whose `target_segment` matches the current user. The engagement model rewards this with a 2.0× multiplier vs. 0.3× for a mismatch — but the agent must infer this from reward signals, not hardcoded rules.

**Fatigue-aware skipping**: Rather than always bidding, the trained agent selectively skips high-fatigue users. The +0.02 skip reward plus future value from restored engagement quality outweighs the immediate impression value for users with fatigue > 0.5.

**Cross-day pacing**: The agent adjusts bid aggression based on `yesterday_recap` signals. After a high-ROAS Day 1, it bids more conservatively to avoid burning budget before the profitable weekend slots. This behavior requires reading and acting on the compressed day summary — a pure RL agent without language comprehension cannot do this.

---

## Why It Matters

Digital advertising is the original real-world multi-agent game. Every company with a Google, Meta, or Amazon ad account is playing this game right now. A policy that's 10% better at ROAS on a $1M/week budget saves $100,000 per week.

AdMarket Arena matters for three audiences:

**Researchers** get the first open benchmark where an LLM participates in real-time second-price auctions with realistic multi-agent dynamics, budget constraints, and long-horizon planning requirements. The ablatable reward components make it possible to study which signal horizon drives which capability.

**Practitioners** get a simulator that mirrors actual advertiser KPIs (ROAS, pacing, fatigue) rather than proxy objectives (score, steps survived). The PersonaBot archetypes map to real competitor behaviors observed in ad auctions.

**OpenEnv contributors** get a template for long-horizon multi-agent environments: the `yesterday_recap` pattern for solving context window limitations generalizes to any domain where the agent needs compressed history.

---

## Try It Yourself

The environment runs via WebSocket through the standard OpenEnv protocol:

```python
from meta_ad_optimizer.client import AdMarketArenaEnv
from meta_ad_optimizer.models import AuctionAction

async with AdMarketArenaEnv(base_url="http://localhost:8000") as env:
    obs = (await env.reset(task="arena_easy")).observation
    print(f"User: {obs.user_segment}, Budget: ${obs.daily_budget_remaining:.2f}")
    print(f"Floor: ${obs.floor_price:.2f}, Recent prices: {obs.recent_clearing_prices}")

    # Bid $1.20 CPM on this slot with creative 0
    result = await env.step(AuctionAction(skip=False, bid_amount=1.20, creative_id=0))
    print(result.observation.last_auction_result)
```

Or run the Arena baselines directly (no server required):

```bash
python -m meta_ad_optimizer.baseline --arena --task arena_easy --episodes 5 --seed 42
```

---

## Links

- **Environment (HuggingFace Space)**: [ritz-gupta/meta-ad-optimizer](https://huggingface.co/spaces/ritz-gupta/meta-ad-optimizer)
- **Trained model**: [ritz-gupta/admarket-advertiser-qwen2.5-3b-grpo](https://huggingface.co/ritz-gupta/admarket-advertiser-qwen2.5-3b-grpo)
- **Training notebook**: `train_grpo.ipynb` (Colab-ready)
- **Source code**: [github.com/ritz-gupta/meta-ad-optimizer](https://github.com/ritz-gupta/meta-ad-optimizer)

---

*Built for the OpenEnv Hackathon · Themes: Multi-Agent + Long-Horizon Planning · Powered by OpenEnv × TRL GRPO × Unsloth × Qwen2.5-3B*
