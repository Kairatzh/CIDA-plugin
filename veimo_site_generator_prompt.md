# System Prompt: Regenerating the VEIMO CIDA Showcase Portal

Copy and paste the prompt below into any advanced AI model (such as Claude 3.5 Sonnet or GPT-4o) to regenerate the entire premium website, styles, animations, and serving scripts.

---

```markdown
You are a Principal Frontend Engineer and UI/UX Designer. Your task is to build a premium, highly animated, single-page developer portal and interactive simulator for "VEIMO Research Lab", showcasing our product "CIDA" (Continuous Latent Deliberation Agent) - a System 2 cognitive co-processor for Causal Language Models.

The project must consist of four clean files structured under a single folder:
1. `index.html` (Main markup, navigation, content, and SVG layout)
2. `styles.css` (Tailwind-compatible styling, animations, keyframes, scrollbars)
3. `app.js` (State machine, SVG network drawing, GSAP timelines, Chart.js telemetry)
4. `run_server.py` (Zero-dependency Python HTTP server with Colab tunnel integration)

Follow these specifications precisely:

### 1. DESIGN LANGUAGE & AESTHETICS (Claude/OpenAI Style)
*   **Theme**: Cyberpunk-minimalist dark mode. Background: `#030408` (obsidian/deep space). Accents: Neon Cyan (`#00f2fe`), Neon Purple (`#7f00ff`), Neon Emerald (`#00f5d4`), and Neon Gold (`#ffd700`).
*   **Background Elements**: Use a premium abstract image at `assets/background.png` blended with a dark gradient overlay.
*   **Typography**: Use Google Fonts: 'Space Grotesk' or 'Outfit' for headers, 'Inter' for body, and 'JetBrains Mono' for code snippets.
*   **Components**: Glassmorphism cards with translucent dark backgrounds, very thin borders (`rgba(255,255,255,0.07)`), and subtle glowing box-shadows when active.
*   **Icons & Emojis**: ABSOLUTELY NO EMOJIS. Use clean vector icons from the Lucide library (`https://unpkg.com/lucide@latest`).

### 2. PAGE SECTIONS
*   **Navigation Bar**: VEIMO branding with a glowing logo box, links to Demo, API, Benchmarks, Team, and a "Launch Arena" CTA button.
*   **Hero Section**: Striking gradient typography, value proposition subtitle, and two prominent call-to-actions.
*   **Deliberation Arena (The Core Interactive Demo)**:
    *   **Control Panel**: Quick-select buttons for 3 logical traps (Temporal Loop Grid, Monty Hall Dilemma, Transitive Order OOD), a custom text prompt field, a deliberation speed slider (0.1x to 2x), and a "Start Deliberation" button.
    *   **Neural Graph Map**: An SVG-overlaid canvas showing 4 nodes in a ring layout: Agent 1 (Cyan), Agent 2 (Purple), Agent 3 (Emerald), and Coordinator (Gold). Animate pulsing dash-arrays along connection lines during communication phases.
    *   **Thought Bubbles**: A grid of 4 small panels printing the real-time statements/hypotheses of each agent as they debate.
    *   **Telemetry Dashboard**: Real-time meters for Consensus Entropy, Disagreement Rate, and a live Chart.js line graph mapping both variables.
    *   **Causal Rollback Alert**: A full-canvas blinking red warning overlay that triggers screen shake and glitch animations when a contradiction is detected, rolling back the round count.
    *   **Final Output**: Displays the final converged, mathematically verified answer.
*   **API & SDK Documentation**: Clean tabbed code viewer showing integration scripts for `Kairos-CIDA-1.3B` and `Kairos-CIDA-7B` using the `cida-plugin` Python library.
*   **Scientific Benchmarks**: A detailed, high-contrast scoreboard table comparing baseline models against CIDA-enhanced variants.
*   **Lab Philosophy**: Introduces VEIMO's research vision and values.

### 3. JAVASCRIPT STATE MACHINE & ANIMATION LOGIC (`app.js`)
*   Dynamically calculate the coordinates of nodes to draw SVG paths, ensuring responsiveness.
*   Incorporate the following state traces for the 3 scenarios:
    1.  **Temporal Loop Grid (Causal Paradox)**:
        *   Round 1: High disagreement (0.90), high entropy. Agents formulate conflicting causal cycles.
        *   Round 2: Moderate disagreement (0.60). Agents recognize bistability.
        *   Round 3: Contradiction warning! Spikes disagreement to 0.95. Screen shakes, red alert overlays, rounds roll back.
        *   Round 4: Resolves to ground state (False/0) with near-zero disagreement.
    2.  **Monty Hall Dilemma**: 3 rounds converging smoothly to P(Win|Switch) = 2/3.
    3.  **Transitive Comparison Trap**: Length-5 asymmetric order comparison resolving cleanly.
*   Implement step-by-step delays governed by the speed slider.

### 4. PYTHON SERVER LAUNCHER (`run_server.py`)
*   Use built-in `http.server` and `socketserver` to serve the static folder on port `8000`.
*   Automatically check for Google Colab imports and call `google.colab.output.serve_kernel_port(8000)` to expose a secure tunnel link for the user.
```
