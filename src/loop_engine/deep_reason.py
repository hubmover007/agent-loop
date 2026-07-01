"""Deep Reasoning Loop - RDT-inspired iterative reasoning.

Mimics Mythos/OpenMythos Recurrent-Depth Transformer behavior at the Agent level:
  - Same reasoning context, iterated N times
  - Each iteration: h_{t+1} = f(h_t, input, memory_context)
  - ACT (Adaptive Computation Time): simple problems exit early
  - Continuous latent reasoning via model's native thinking mode
  - Confidence tracking: stop when converged

Key difference from Chain-of-Thought:
  - CoT: discrete token sequences, linear reasoning
  - DeepReason: iterative refinement in latent space, branching exploration
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from . import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Max total time for deep reasoning before forced stop
DEEP_REASON_TOTAL_TIMEOUT = 25.0  # seconds


@dataclass
class ReasonState:
    """State of a single reasoning iteration (analogous to h_t in RDT)."""
    iteration: int
    input_context: str               # Original input (analogous to e in RDT)
    previous_thought: str = ""        # Previous iteration's output (analogous to A·h_t)
    current_thought: str = ""         # Current iteration's output
    confidence: float = 0.0           # Convergence score
    insights: list[str] = field(default_factory=list)  # Key insights discovered
    dead_ends: list[str] = field(default_factory=list)  # Paths that didn't work
    exploration_branches: int = 0     # Number of alternative paths explored


@dataclass
class DeepReasonConfig:
    """Configuration for the deep reasoning loop."""
    max_iterations: int = 8           # Maximum reasoning iterations
    min_iterations: int = 1           # Minimum iterations (skip early exit)
    confidence_threshold: float = 0.85  # ACT: stop when confidence > this
    thinking_budget_tokens: int = 2048  # Tokens for model's native thinking
    temperature_start: float = 0.7     # Initial temperature
    temperature_decay: float = 0.9     # Temperature decay per iteration
    exploration_factor: float = 0.3    # How much to explore alternatives vs exploit
    enable_latent_thinking: bool = True  # Enable model's native thinking/CoT


class DeepReasonLoop:
    """Iterative deep reasoning engine.

    Implements the "model-level depth" of the hybrid reasoning approach.
    Works by running multiple reasoning iterations where each iteration
    refines the previous thinking, analogous to RDT loop steps.

    The model's native `thinking` mode provides the "continuous latent space"
    reasoning (no intermediate output tokens), while our iteration loop
    provides the external refinement cycle.
    """

    def __init__(self, llm: LLMProvider, config: DeepReasonConfig | None = None,
                 llm_pool: Any | None = None):
        self.llm = llm
        self.config = config or DeepReasonConfig()
        self.llm_pool = llm_pool

    def _get_llm(self) -> LLMProvider:
        """Get the best LLM for deep reasoning, falling back to default."""
        if self.llm_pool:
            provider = self.llm_pool.get_provider(
                capabilities=["reasoning"], strategy="most_capable"
            )
            if provider:
                return provider
        return self.llm

    async def reason(self, query: str, context: str = "",
                     system_prompt: str | None = None,
                     total_timeout: float | None = None) -> ReasonState:
        """Execute the deep reasoning loop.

        Args:
            query: The user's question or task
            context: Retrieved memory context
            system_prompt: Optional system prompt
            total_timeout: Max total seconds for the reasoning loop (default 25)

        Returns:
            ReasonState with the final reasoning output and metadata
        """
        _timeout = total_timeout if total_timeout is not None else DEEP_REASON_TOTAL_TIMEOUT
        loop_start = time.time()

        state = ReasonState(
            iteration=0,
            input_context=query,
        )

        # Build base system prompt
        if system_prompt is None:
            system_prompt = self._default_system_prompt(context)

        # Initial analysis
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_iteration_prompt(query, context, 0)},
        ]

        for iteration in range(1, self.config.max_iterations + 1):
            # Total timeout guard
            if time.time() - loop_start > _timeout:
                logger.warning(
                    "DeepReason: total timeout %.1fs after %d iterations",
                    _timeout, iteration - 1,
                )
                break

            state.iteration = iteration

            # Temperature annealing: cooler as we converge
            temperature = self.config.temperature_start * (
                self.config.temperature_decay ** (iteration - 1)
            )

            try:
                # Direct chat with timeout — no retry layer to avoid double-timeout issues
                llm = self._get_llm()
                response = await asyncio.wait_for(
                    llm.chat(
                        messages,
                        thinking=self.config.enable_latent_thinking,
                        temperature=temperature,
                        max_tokens=800,  # Reduced from 2000 to stay within timeout
                    ),
                    timeout=20.0,
                )
            except Exception as e:
                logger.error("Reason iteration %d failed: %s", iteration, e)
                break

            state.previous_thought = state.current_thought
            state.current_thought = response.content

            # Extract insights and dead ends
            self._extract_signals(state, response)

            # Estimate confidence (ACT halting check)
            state.confidence = self._estimate_confidence(state, iteration)

            logger.info(
                "DeepReason iteration %d: confidence=%.3f (threshold=%.2f)",
                iteration, state.confidence, self.config.confidence_threshold
            )

            # ACT: stop if confident enough
            if (iteration >= self.config.min_iterations and
                    state.confidence >= self.config.confidence_threshold):
                logger.info("ACT halt: converged at iteration %d", iteration)
                break

            # Prepare refinement prompt for next iteration
            if iteration < self.config.max_iterations:
                refinement = self._build_refinement_prompt(state)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": refinement})

        return state

    # ============================================================
    # Prompt Engineering
    # ============================================================

    def _default_system_prompt(self, context: str) -> str:
        """Default system prompt for deep reasoning."""
        base = """You are a deep reasoning engine. Your task is to THOROUGHLY analyze
problems before answering. You think in multiple passes:

1. First pass: Understand the core question. What is REALLY being asked?
2. Second pass: Map the problem space. What do we know? What's missing?
3. Third pass: Explore alternatives. Are there other ways to solve this?
4. Fourth pass: Validate. Check your reasoning for gaps or errors.
5. Final pass: Synthesize. Produce the best answer given all analysis.

For each pass, surface your reasoning explicitly. If you find a flaw
in your previous thinking, correct it. If you discover a better approach,
adopt it."""

        if context:
            base += f"\n\nRelevant context from memory:\n{context}"

        return base

    def _format_iteration_prompt(self, query: str, context: str, iteration: int) -> str:
        """Format the prompt for a specific iteration."""
        if iteration == 0:
            return f"""Analyze the following request. Think deeply before answering.

Request: {query}

{f'Context: {context}' if context else ''}

Provide your analysis and answer."""
        else:
            return f"""Re-examine your analysis. Consider:

1. Are there any gaps in your reasoning?
2. Could there be alternative interpretations?
3. Is there a simpler or more elegant solution?
4. What assumptions are you making? Are they valid?

Original request: {query}"""

    def _build_refinement_prompt(self, state: ReasonState) -> str:
        """Build a refinement prompt based on current state."""
        prompt_parts = ["Re-examine and refine your analysis."]

        if state.dead_ends:
            prompt_parts.append(f"\nPaths that didn't work: {', '.join(state.dead_ends[-3:])}")
        if state.insights:
            prompt_parts.append(f"\nKey insights so far: {', '.join(state.insights[-3:])}")

        prompt_parts.append("""
Focus on:
- What might you be missing?
- Are your assumptions valid?
- Is there a fundamentally different approach?
- Can you be more precise?

Be concise. Don't repeat what you already said.""")

        return "\n".join(prompt_parts)

    # ============================================================
    # Signal Extraction
    # ============================================================

    def _extract_signals(self, state: ReasonState, response: LLMResponse) -> None:
        """Extract insights and dead ends from reasoning output."""
        text = response.content.lower()

        # Insight markers
        insight_markers = ["insight:", "key finding:", "importantly,", "crucially,",
                          "discovered", "realized", "notably,"]
        for line in response.content.split("\n"):
            line_lower = line.lower().strip()
            if any(marker in line_lower for marker in insight_markers):
                if len(line) < 200 and line not in state.insights:
                    state.insights.append(line.strip())

        # Dead end markers
        dead_end_markers = ["doesn't work", "won't work", "dead end", "not viable",
                           "incorrect approach", "this fails", "not possible"]
        for line in response.content.split("\n"):
            line_lower = line.lower().strip()
            if any(marker in line_lower for marker in dead_end_markers):
                if len(line) < 200 and line not in state.dead_ends:
                    state.dead_ends.append(line.strip())

        # Count exploration branches
        branch_indicators = ["alternative", "another approach", "option 1", "option 2",
                            "on the other hand", "conversely", "alternatively"]
        state.exploration_branches += sum(
            1 for marker in branch_indicators if marker in text
        )

    def _estimate_confidence(self, state: ReasonState, iteration: int) -> float:
        """Estimate reasoning convergence (ACT halting proxy).

        Uses multiple signals:
        - Output stability: similar to previous iteration
        - Structure quality: well-organized output
        - Certainty markers: confident language
        - Convergence rate: diminishing returns
        """
        confidence = 0.3  # Base

        current = state.current_thought
        previous = state.previous_thought

        # 1. Length suggests thoroughness
        if len(current) > 500:
            confidence += 0.1
        if len(current) > 1000:
            confidence += 0.05

        # 2. Structure quality
        structure_markers = ["#", "##", "1.", "2.", "step", "first", "second"]
        structure_score = sum(1 for m in structure_markers if m in current.lower())
        confidence += min(structure_score * 0.03, 0.15)

        # 3. Certainty language
        certainty = ["clearly", "definitely", "therefore", "conclusion", "the answer is"]
        certainty_score = sum(1 for m in certainty if m in current.lower())
        confidence += min(certainty_score * 0.03, 0.1)

        # 4. Hedging reduces confidence
        hedging = ["might", "maybe", "possibly", "could be", "i'm not sure",
                  "uncertain", "approximately"]
        hedge_count = sum(1 for h in hedging if h in current.lower())
        confidence -= hedge_count * 0.03

        # 5. Stability with previous iteration
        if previous and len(current) > 100:
            # Simple stability: length ratio near 1 suggests convergence
            length_ratio = min(len(current), len(previous)) / max(len(current), len(previous))
            if length_ratio > 0.8:
                confidence += 0.1

        # 6. Progress signal: insights found
        if state.insights:
            confidence += min(len(state.insights) * 0.03, 0.1)

        # 7. Iteration-based boost (later iterations generally more refined)
        confidence += min(iteration * 0.02, 0.1)

        return max(0.1, min(0.95, confidence))
