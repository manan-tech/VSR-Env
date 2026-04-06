# Requirements Document: VSR-Env (Volatility Surface Reasoning Environment)

## Introduction

VSR-Env is an OpenEnv-compliant reinforcement learning environment simulating options portfolio management on implied volatility surfaces. The environment targets the Meta PyTorch OpenEnv Hackathon × SST ($30,000 prize pool) with a Round 1 deadline of April 8, 2026.

The system simulates genuine quantitative trading workflows used at major options desks in the $600T+ notional derivatives market. An LLM agent acts as a junior options trader, analyzing volatility surfaces, identifying mispricings, constructing delta-neutral trades, and generating reasoning traces.

## Glossary

- **VSR_Environment**: The OpenEnv-compliant environment server implementing options portfolio management simulation
- **Option_Chain_Engine**: The computational engine performing Black-Scholes pricing, Greeks calculation, and implied volatility solving using NumPy/SciPy
- **Market_Simulator**: The component generating realistic price dynamics and regime shifts
- **Task_Manager**: The component managing the three graded tasks (iv_reading, delta_hedging, arb_capture)
- **Grader**: A deterministic scoring function returning values in [0.0, 1.0] for task performance evaluation
- **Inference_Script**: The baseline inference.py script using OpenAI client to interact with the environment
- **IV_Surface**: An 8×3 matrix of implied volatility values across strikes and maturities
- **Delta_Neutral**: A portfolio state where the delta Greek is within ±0.05
- **Regime_Shift**: A market condition change occurring mid-episode in the hard task
- **Pretty_Printer**: A component formatting internal state back to human-readable configuration format
- **Round_Trip_Property**: The property that parsing then printing then parsing produces an equivalent object

## Requirements

### Requirement 1: OpenEnv Specification Compliance

**User Story:** As a hackathon evaluator, I want the environment to fully comply with OpenEnv specifications, so that it passes validation and integrates with the evaluation infrastructure.

#### Acceptance Criteria

1. THE VSR_Environment SHALL implement all Pydantic BaseModel types for VSRAction, VSRObservation, VSRState, and VSRReward
2. THE VSR_Environment SHALL provide a reset method accepting task_name and seed parameters
3. THE VSR_Environment SHALL provide a step method accepting VSRAction and returning observation, reward, done, and info
4. THE VSR_Environment SHALL expose a state property returning the current VSRState
5. WHEN openenv validate is executed, THE VSR_Environment SHALL pass all validation checks
6. THE VSR_Environment SHALL include an openenv.yaml manifest with name, version, description, tasks, action_type, observation_type, reward_type, and state_type fields
7. THE VSR_Environment SHALL serve via FastAPI on port 8000 with /reset and /step endpoints

### Requirement 2: Three-Task Progression System

**User Story:** As a hackathon evaluator, I want three tasks with clear difficulty progression, so that I can assess agent capabilities across easy, medium, and hard challenges.

#### Acceptance Criteria

1. THE Task_Manager SHALL implement an iv_reading task with maximum 3 steps and easy difficulty
2. THE Task_Manager SHALL implement a delta_hedging task with maximum 5 steps and medium difficulty
3. THE Task_Manager SHALL implement an arb_capture task with maximum 8 steps and hard difficulty
4. WHEN a task is initialized, THE Task_Manager SHALL configure task-specific state including mispriced options, portfolio positions, or regime parameters
5. FOR ALL tasks, THE Task_Manager SHALL provide a task_description field in observations explaining the objective

### Requirement 3: IV Reading Task (Easy)

**User Story:** As an LLM agent, I want to identify mispriced options on a volatility surface, so that I can demonstrate basic options analysis capability.

#### Acceptance Criteria

1. WHEN iv_reading task is reset, THE VSR_Environment SHALL generate an 8×3 IV surface with exactly 2 deliberately mispriced cells
2. THE VSR_Environment SHALL mark mispriced cells as either overpriced or underpriced with magnitude between 0.03 and 0.08
3. WHEN the agent selects a mispriced strike with correct direction, THE VSR_Environment SHALL award 0.5 identification reward
4. WHEN the agent selects a mispriced strike with incorrect direction, THE VSR_Environment SHALL award 0.1 identification reward
5. WHEN the episode completes, THE Grader SHALL return correct_identifications divided by 2.0 clamped to [0.0, 1.0]
6. THE Grader SHALL provide partial credit for identifying 1 of 2 mispriced options

### Requirement 4: Delta Hedging Task (Medium)

**User Story:** As an LLM agent, I want to neutralize portfolio delta cost-efficiently, so that I can demonstrate risk management capability.

#### Acceptance Criteria

1. WHEN delta_hedging task is reset, THE VSR_Environment SHALL initialize a portfolio with non-zero delta between 0.2 and 0.8
2. WHEN the agent executes a trade, THE VSR_Environment SHALL update portfolio delta based on the trade's delta contribution
3. WHEN the agent achieves portfolio delta within ±0.05, THE VSR_Environment SHALL award a neutrality bonus of 0.1
4. THE VSR_Environment SHALL compute per-step reward as delta_improvement × 0.6 + cost_efficiency × 0.4 + neutrality_bonus
5. WHEN the episode completes, THE Grader SHALL return neutralization_quality × 0.7 + cost_efficiency × 0.3 clamped to [0.0, 1.0]
6. THE Grader SHALL compute neutralization_quality as max(0, 1.0 - final_delta_abs / initial_delta_abs)

### Requirement 5: Arbitrage Capture Task (Hard)

**User Story:** As an LLM agent, I want to execute a full arbitrage workflow with regime shifts, so that I can demonstrate advanced trading capability.

#### Acceptance Criteria

1. WHEN arb_capture task is reset, THE VSR_Environment SHALL initialize a market with at least one exploitable mispricing
2. WHEN step count reaches 4 or 5, THE Market_Simulator SHALL trigger a regime shift changing volatility parameters
3. THE VSR_Environment SHALL compute per-step reward as pnl_component × 0.4 + greek_component × 0.3 + reasoning_component × 0.3
4. WHEN the episode completes, THE Grader SHALL return pnl_score × 0.4 + neutrality_score × 0.3 + reasoning_score × 0.3 clamped to [0.0, 1.0]
5. THE Grader SHALL compute pnl_score using sigmoid normalization centered at 0 with scale 0.3
6. THE Grader SHALL compute neutrality_score as max(0, 1.0 - average_delta_abs / 0.5)
7. THE Grader SHALL compute reasoning_score by evaluating numeric consistency and keyword presence

### Requirement 6: Deterministic Grading with Reproducibility

**User Story:** As a hackathon evaluator, I want grader scores to be deterministic and reproducible, so that I can fairly compare submissions.

#### Acceptance Criteria

1. WHEN reset is called with a seed parameter, THE VSR_Environment SHALL initialize a NumPy RandomState with that seed
2. FOR ALL random number generation, THE VSR_Environment SHALL use the seeded RandomState instance
3. WHEN the same task is reset with the same seed, THE VSR_Environment SHALL produce identical IV surfaces, mispriced cells, and initial portfolio states
4. FOR ALL graders, THE Grader SHALL return scores in the range [0.0, 1.0]
5. WHEN a grader computes a score, THE Grader SHALL clamp the result using min(max(score, 0.0), 1.0)
6. THE Grader SHALL produce identical scores for identical episode histories

### Requirement 7: Black-Scholes Pricing and Greeks

**User Story:** As the environment, I want accurate option pricing and Greeks calculation, so that portfolio valuations are realistic.

#### Acceptance Criteria

1. THE Option_Chain_Engine SHALL compute Black-Scholes call and put prices using the formula with spot, strike, maturity, rate, and volatility parameters
2. THE Option_Chain_Engine SHALL compute delta as the first derivative of option price with respect to spot price
3. THE Option_Chain_Engine SHALL compute gamma as the second derivative of option price with respect to spot price
4. THE Option_Chain_Engine SHALL compute vega as the derivative of option price with respect to volatility
5. THE Option_Chain_Engine SHALL vectorize all calculations using NumPy arrays for 8 strikes and 3 maturities simultaneously
6. THE Option_Chain_Engine SHALL use scipy.stats.norm for cumulative distribution and probability density functions

### Requirement 8: Implied Volatility Solver

**User Story:** As the environment, I want to solve for implied volatility from market prices, so that I can generate realistic IV surfaces.

#### Acceptance Criteria

1. THE Option_Chain_Engine SHALL implement Newton-Raphson method for implied volatility solving with initial guess 0.2
2. WHEN vega is less than 1e-8, THE Option_Chain_Engine SHALL fall back to Brent's method using scipy.optimize.brentq
3. THE Option_Chain_Engine SHALL iterate Newton-Raphson for maximum 100 iterations with tolerance 1e-6
4. THE Option_Chain_Engine SHALL clamp volatility guesses to the range [0.01, 5.0] during iteration
5. WHEN Brent's method fails, THE Option_Chain_Engine SHALL return intrinsic volatility computed as max(0.05, min(abs(log(S/K)) / sqrt(T) × 0.5, 3.0))
6. THE Option_Chain_Engine SHALL accept market_price, spot, strike, maturity, rate, and option_type parameters

### Requirement 9: IV Surface Generation

**User Story:** As the environment, I want to generate realistic implied volatility surfaces, so that agents face authentic market conditions.

#### Acceptance Criteria

1. THE Option_Chain_Engine SHALL generate IV surfaces with base volatility, skew, and term structure parameters
2. THE Option_Chain_Engine SHALL apply skew as base_vol + skew_coefficient × log_moneyness / sqrt(maturity)
3. THE Option_Chain_Engine SHALL apply term structure as additional term_slope × sqrt(maturity)
4. THE Option_Chain_Engine SHALL add Gaussian noise with standard deviation 0.005 using the seeded RNG
5. THE Option_Chain_Engine SHALL clamp all IV values to minimum 0.05
6. WHEN mispriced_cells parameter is provided, THE Option_Chain_Engine SHALL adjust specified cells by the given magnitude and direction
7. THE Option_Chain_Engine SHALL return the IV surface as an 8×3 list of lists

### Requirement 10: Meaningful Per-Step Rewards

**User Story:** As an RL researcher, I want per-step rewards that provide partial progress signals, so that agents can learn from trajectory feedback.

#### Acceptance Criteria

1. THE VSR_Environment SHALL compute non-zero rewards at every step, not only at episode termination
2. FOR iv_reading task, THE VSR_Environment SHALL award 0.5 per correct identification and 0.1 per partially correct identification per step
3. FOR delta_hedging task, THE VSR_Environment SHALL award delta_improvement × 0.6 + cost_efficiency × 0.4 per step
4. FOR arb_capture task, THE VSR_Environment SHALL award pnl_change × 0.4 + greek_quality × 0.3 + reasoning_quality × 0.3 per step
5. THE VSR_Environment SHALL return VSRReward with total, pnl_component, greek_component, identification_component, and reasoning_component fields
6. FOR ALL reward components, THE VSR_Environment SHALL normalize values to contribute to a total in approximate range [0.0, 1.0]

### Requirement 11: Reasoning Quality Scoring

**User Story:** As the environment, I want to score reasoning quality in a way that resists gaming, so that agents must genuinely analyze observations.

#### Acceptance Criteria

1. THE VSR_Environment SHALL score reasoning using both keyword presence (maximum 0.4) and numeric consistency (maximum 0.6)
2. THE VSR_Environment SHALL check for domain keywords including delta, hedge, neutral, skew, smile, regime, overpriced, underpriced, moneyness, vega, and gamma
3. THE VSR_Environment SHALL award 0.25 when reasoning cites the actual spot price within ±0.5 tolerance
4. THE VSR_Environment SHALL award 0.15 when reasoning cites at least one actual IV value from the surface
5. THE VSR_Environment SHALL award additional 0.1 when reasoning cites at least two actual IV values
6. THE VSR_Environment SHALL award 0.1 when reasoning cites the current portfolio delta value
7. WHEN reasoning length is 20 characters or less, THE VSR_Environment SHALL multiply the reasoning score by 0.3
8. THE VSR_Environment SHALL clamp the final reasoning score to [0.0, 1.0]

### Requirement 12: Baseline Inference Script

**User Story:** As a hackathon evaluator, I want a baseline inference script that demonstrates environment usage, so that I can verify the environment works and establish baseline scores.

#### Acceptance Criteria

1. THE Inference_Script SHALL use OpenAI client initialized with API_BASE_URL and HF_TOKEN environment variables
2. THE Inference_Script SHALL run all three tasks (iv_reading, delta_hedging, arb_capture) sequentially
3. WHEN a task starts, THE Inference_Script SHALL print "[START] task={task_name} env={benchmark} model={model_name}" to stdout
4. WHEN a step completes, THE Inference_Script SHALL print "[STEP] step={N} action={action_str} reward={reward:.2f} done={bool} error={error}" to stdout
5. WHEN a task ends, THE Inference_Script SHALL print "[END] success={bool} steps={N} score={score:.2f} rewards={comma_separated_rewards}" to stdout
6. THE Inference_Script SHALL use fixed seeds (42, 123, 456) for the three tasks to ensure reproducible baseline scores
7. THE Inference_Script SHALL parse LLM responses as JSON with fields strike_idx, maturity_idx, direction, quantity, and reasoning
8. WHEN JSON parsing fails, THE Inference_Script SHALL default to a hold action with zero quantity

### Requirement 13: CPU-Only Computation

**User Story:** As a hackathon organizer, I want the environment to run on CPU-only infrastructure (vcpu=2, 8GB RAM), so that it fits within evaluation resource constraints.

#### Acceptance Criteria

1. THE VSR_Environment SHALL use only NumPy and SciPy for all mathematical computations
2. THE VSR_Environment SHALL NOT import torch, cuda, or any GPU-dependent libraries
3. THE Option_Chain_Engine SHALL vectorize operations using NumPy arrays to maximize CPU efficiency
4. WHEN the environment runs all three tasks with 10 total episodes, THE VSR_Environment SHALL complete in less than 20 minutes on vcpu=2 infrastructure
5. THE VSR_Environment SHALL use scipy.stats.norm.cdf and scipy.stats.norm.pdf for normal distribution functions
6. THE VSR_Environment SHALL use scipy.optimize.brentq for Brent's method fallback in IV solving

### Requirement 14: Docker Deployment

**User Story:** As a hackathon evaluator, I want the environment to deploy via Docker, so that I can run it in a standardized infrastructure.

#### Acceptance Criteria

1. THE VSR_Environment SHALL provide a Dockerfile using python:3.11-slim base image
2. THE Dockerfile SHALL install dependencies from requirements.txt including openenv-core, fastapi, uvicorn, pydantic, numpy, and scipy
3. THE Dockerfile SHALL expose port 8000
4. THE Dockerfile SHALL set CMD to run uvicorn with vsr_env.server.app:app on host 0.0.0.0 port 8000
5. THE Dockerfile SHALL include a HEALTHCHECK calling curl on http://localhost:8000/health every 30 seconds
6. WHEN docker build is executed, THE Dockerfile SHALL complete successfully without errors
7. WHEN docker run is executed, THE VSR_Environment SHALL start and respond to HTTP requests within 10 seconds

### Requirement 15: HuggingFace Spaces Deployment

**User Story:** As a hackathon participant, I want to deploy the environment to HuggingFace Spaces, so that evaluators can access it via URL.

#### Acceptance Criteria

1. THE VSR_Environment SHALL deploy to a HuggingFace Space with the openenv tag
2. WHEN a POST request is sent to {space_url}/reset, THE VSR_Environment SHALL return status 200 with a valid VSRObservation
3. WHEN a POST request is sent to {space_url}/step with a VSRAction, THE VSR_Environment SHALL return status 200 with observation, reward, done, and info
4. THE HuggingFace Space SHALL use the Dockerfile for deployment
5. THE HuggingFace Space SHALL set environment variables API_BASE_URL, MODEL_NAME, and HF_TOKEN
6. WHEN the pre_validation script is executed against the Space URL, THE VSR_Environment SHALL pass all three validation checks

### Requirement 16: Action Space Definition

**User Story:** As an LLM agent, I want a clear action space definition, so that I know what actions I can take.

#### Acceptance Criteria

1. THE VSRAction SHALL include a selected_strike field with integer values 0-7 representing strike index
2. THE VSRAction SHALL include a selected_maturity field with integer values 0-2 representing maturity index
3. THE VSRAction SHALL include a direction field with enum values buy, sell, or hold
4. THE VSRAction SHALL include a quantity field with float values 0.0-10.0 representing trade size in contracts
5. THE VSRAction SHALL include a reasoning field with string value containing the agent's analysis
6. THE VSRAction SHALL use Pydantic Field with description for each field
7. THE VSRAction SHALL inherit from pydantic.BaseModel

### Requirement 17: Observation Space Definition

**User Story:** As an LLM agent, I want comprehensive observations of market state, so that I can make informed trading decisions.

#### Acceptance Criteria

1. THE VSRObservation SHALL include an iv_surface field containing an 8×3 list of implied volatility values
2. THE VSRObservation SHALL include a spot_price field with the current underlying price
3. THE VSRObservation SHALL include a portfolio_greeks field with dict containing delta, gamma, vega, and theta values
4. THE VSRObservation SHALL include a portfolio_pnl field with cumulative profit and loss
5. THE VSRObservation SHALL include a portfolio_positions field with list of current open positions
6. THE VSRObservation SHALL include a market_sentiment field with float value in range [-1.0, 1.0]
7. THE VSRObservation SHALL include step_number and steps_remaining fields
8. THE VSRObservation SHALL include task_name and task_description fields
9. THE VSRObservation SHALL include a last_action_error field with optional string describing validation errors
10. THE VSRObservation SHALL use Pydantic Field with description for each field
11. THE VSRObservation SHALL inherit from pydantic.BaseModel

### Requirement 18: State Space Definition

**User Story:** As the environment, I want a complete internal state representation, so that I can track all information including hidden variables.

#### Acceptance Criteria

1. THE VSRState SHALL include episode_id, step_count, and task_name fields
2. THE VSRState SHALL include true_mispriced_strikes and true_mispriced_directions fields for grading purposes
3. THE VSRState SHALL include regime field indicating current market regime
4. THE VSRState SHALL include spot_price and variance fields for market simulation
5. THE VSRState SHALL include portfolio_delta, portfolio_gamma, portfolio_vega, and portfolio_pnl fields
6. THE VSRState SHALL include positions field with list of current portfolio positions
7. THE VSRState SHALL use Pydantic Field with default_factory for list and dict fields
8. THE VSRState SHALL inherit from pydantic.BaseModel

### Requirement 19: Reward Structure Definition

**User Story:** As an RL researcher, I want structured reward breakdown, so that I can analyze which components drive agent behavior.

#### Acceptance Criteria

1. THE VSRReward SHALL include a total field with the aggregate reward value
2. THE VSRReward SHALL include a pnl_component field with profit/loss contribution
3. THE VSRReward SHALL include a greek_component field with Greek neutrality contribution
4. THE VSRReward SHALL include an identification_component field with mispricing identification contribution
5. THE VSRReward SHALL include a reasoning_component field with reasoning quality contribution
6. THE VSRReward SHALL use Pydantic Field with description for each field
7. THE VSRReward SHALL inherit from pydantic.BaseModel

### Requirement 20: Action Validation

**User Story:** As the environment, I want to validate actions before execution, so that I can provide clear error messages for invalid actions.

#### Acceptance Criteria

1. WHEN selected_strike is outside range [0, 7], THE VSR_Environment SHALL return error "Invalid strike index"
2. WHEN selected_maturity is outside range [0, 2], THE VSR_Environment SHALL return error "Invalid maturity index"
3. WHEN quantity is negative, THE VSR_Environment SHALL return error "Quantity must be non-negative"
4. WHEN quantity exceeds 10.0, THE VSR_Environment SHALL return error "Quantity exceeds maximum of 10 contracts"
5. WHEN direction is not buy, sell, or hold, THE VSR_Environment SHALL return error "Invalid direction"
6. WHEN an action validation error occurs, THE VSR_Environment SHALL set last_action_error in the observation
7. WHEN an action validation error occurs, THE VSR_Environment SHALL NOT modify portfolio state

### Requirement 21: Market Simulation

**User Story:** As the environment, I want realistic market dynamics, so that agents face authentic trading conditions.

#### Acceptance Criteria

1. THE Market_Simulator SHALL update spot price using Geometric Brownian Motion with drift and volatility parameters
2. THE Market_Simulator SHALL use the seeded RNG for all random price movements
3. WHEN arb_capture task reaches step 4 or 5, THE Market_Simulator SHALL trigger a regime shift
4. WHEN a regime shift occurs, THE Market_Simulator SHALL modify volatility parameters by 20-40%
5. THE Market_Simulator SHALL update variance using mean-reverting dynamics
6. THE Market_Simulator SHALL clamp spot price to range [50.0, 150.0] to prevent unrealistic values
7. THE Market_Simulator SHALL clamp variance to range [0.01, 0.16] to prevent unrealistic volatility

### Requirement 22: Portfolio Management

**User Story:** As the environment, I want accurate portfolio tracking, so that Greeks and P&L reflect actual positions.

#### Acceptance Criteria

1. WHEN a buy action is executed, THE VSR_Environment SHALL add the position to the portfolio with positive quantity
2. WHEN a sell action is executed, THE VSR_Environment SHALL add the position to the portfolio with negative quantity
3. WHEN a hold action is executed, THE VSR_Environment SHALL NOT modify the portfolio
4. THE VSR_Environment SHALL recompute portfolio Greeks as the sum of individual position Greeks
5. THE VSR_Environment SHALL recompute portfolio P&L as the sum of individual position P&L values
6. THE VSR_Environment SHALL update position P&L based on current market prices minus entry prices
7. THE VSR_Environment SHALL store each position with strike, maturity, direction, quantity, entry_price, and entry_iv fields

### Requirement 23: Documentation and README

**User Story:** As a hackathon evaluator, I want comprehensive documentation, so that I understand the environment design and usage.

#### Acceptance Criteria

1. THE VSR_Environment SHALL provide a README.md with project overview, motivation, and real-world utility explanation
2. THE README SHALL document the action space with field descriptions and valid ranges
3. THE README SHALL document the observation space with field descriptions
4. THE README SHALL document all three tasks with objectives, max steps, and grading criteria
5. THE README SHALL include baseline scores for each task with the model used
6. THE README SHALL include installation instructions with pip install and docker build commands
7. THE README SHALL include usage examples showing reset and step calls
8. THE README SHALL include environment variable requirements (API_BASE_URL, MODEL_NAME, HF_TOKEN)

### Requirement 24: Testing and Validation

**User Story:** As a developer, I want comprehensive tests, so that I can verify correctness before submission.

#### Acceptance Criteria

1. THE VSR_Environment SHALL include unit tests for Black-Scholes pricing that match scipy reference implementations
2. THE VSR_Environment SHALL include unit tests for each grader verifying scores are in [0.0, 1.0]
3. THE VSR_Environment SHALL include integration tests for reset/step/state cycle
4. THE VSR_Environment SHALL include tests for inference.py stdout format matching the required pattern
5. THE VSR_Environment SHALL include tests for action validation error messages
6. THE VSR_Environment SHALL include tests for deterministic reproducibility with fixed seeds
7. WHEN pytest is executed, THE VSR_Environment SHALL pass all tests without failures

### Requirement 25: Performance Requirements

**User Story:** As a hackathon evaluator, I want the environment to meet performance constraints, so that it runs within evaluation time limits.

#### Acceptance Criteria

1. WHEN a single step is executed, THE VSR_Environment SHALL complete in less than 2 seconds on vcpu=2 infrastructure
2. WHEN all three tasks are run with baseline inference, THE VSR_Environment SHALL complete in less than 5 minutes total
3. THE Option_Chain_Engine SHALL compute Greeks for the full 8×3 option chain in less than 10 milliseconds
4. THE Option_Chain_Engine SHALL solve implied volatility for a single option in less than 5 milliseconds on average
5. THE VSR_Environment SHALL use vectorized NumPy operations for all array computations
6. THE VSR_Environment SHALL avoid Python loops over strikes and maturities where vectorization is possible

### Requirement 26: Error Handling and Robustness

**User Story:** As the environment, I want robust error handling, so that invalid inputs don't crash the system.

#### Acceptance Criteria

1. WHEN reset is called with an invalid task_name, THE VSR_Environment SHALL default to iv_reading task
2. WHEN step receives a malformed action, THE VSR_Environment SHALL return an error observation without crashing
3. WHEN IV solver fails to converge, THE Option_Chain_Engine SHALL return intrinsic volatility estimate
4. WHEN vega is near zero, THE Option_Chain_Engine SHALL use Brent's method instead of Newton-Raphson
5. WHEN Brent's method fails, THE Option_Chain_Engine SHALL return a fallback volatility value
6. WHEN portfolio delta calculation encounters division by zero, THE VSR_Environment SHALL handle it gracefully
7. THE VSR_Environment SHALL log errors to stderr without exposing them in observations

### Requirement 27: Scoring Rubric Alignment

**User Story:** As a hackathon participant, I want the implementation to align with scoring rubric criteria, so that I maximize my submission score.

#### Acceptance Criteria

1. THE VSR_Environment SHALL demonstrate real-world utility by simulating workflows used in the $600T+ derivatives market
2. THE VSR_Environment SHALL provide three tasks with clear difficulty progression from easy to hard
3. THE VSR_Environment SHALL implement deterministic graders returning continuous scores in [0.0, 1.0]
4. THE VSR_Environment SHALL provide meaningful per-step rewards that signal partial progress
5. THE VSR_Environment SHALL pass openenv validate without errors
6. THE VSR_Environment SHALL include comprehensive README documentation
7. THE VSR_Environment SHALL demonstrate creativity through novel domain (quantitative finance) and interesting mechanics (regime shifts, reasoning scoring)

### Requirement 28: Configuration and Extensibility

**User Story:** As a researcher, I want configurable environment parameters, so that I can adjust difficulty and market conditions.

#### Acceptance Criteria

1. THE VSR_Environment SHALL accept base_volatility parameter in reset with default 0.2
2. THE VSR_Environment SHALL accept skew_coefficient parameter in reset with default -0.02
3. THE VSR_Environment SHALL accept term_slope parameter in reset with default 0.01
4. THE VSR_Environment SHALL accept initial_portfolio_delta parameter for delta_hedging task with default range [0.2, 0.8]
5. THE VSR_Environment SHALL accept regime_shift_step parameter for arb_capture task with default range [4, 5]
6. THE VSR_Environment SHALL accept regime_shift_magnitude parameter with default range [0.2, 0.4]
7. THE VSR_Environment SHALL document all configurable parameters in the README

### Requirement 29: Logging and Observability

**User Story:** As a developer, I want structured logging, so that I can debug issues and monitor environment behavior.

#### Acceptance Criteria

1. WHEN reset is called, THE VSR_Environment SHALL log the task_name, seed, and episode_id
2. WHEN a step is executed, THE VSR_Environment SHALL log the step number, action summary, and reward components
3. WHEN a grader computes a score, THE VSR_Environment SHALL log the final score and component breakdown
4. WHEN an error occurs, THE VSR_Environment SHALL log the error message and stack trace to stderr
5. THE VSR_Environment SHALL use structured logging with JSON format for machine readability
6. THE VSR_Environment SHALL include timestamps in all log entries
7. THE VSR_Environment SHALL support log level configuration via LOG_LEVEL environment variable

### Requirement 30: Round-Trip Property for State Serialization

**User Story:** As the environment, I want to serialize and deserialize state correctly, so that episodes can be saved and resumed.

#### Acceptance Criteria

1. THE VSR_Environment SHALL provide a serialize_state method returning JSON-compatible dict
2. THE VSR_Environment SHALL provide a deserialize_state method accepting JSON-compatible dict
3. THE Pretty_Printer SHALL format VSRState objects into human-readable JSON
4. FOR ALL valid VSRState objects, THE VSR_Environment SHALL satisfy the round-trip property: deserialize(serialize(state)) produces an equivalent state
5. THE VSR_Environment SHALL include unit tests verifying the round-trip property for all state fields
6. WHEN state contains NumPy arrays, THE VSR_Environment SHALL convert them to lists for JSON serialization
