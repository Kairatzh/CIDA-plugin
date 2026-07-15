// CIDA Web Visualizer Application Logic

let chart = null;
let currentScenario = 'temporal';
let speedFactor = 1.0;
let isRunning = false;
let simulationTimeout = null;

// Scenario data repository
const SCENARIOS = {
    temporal: {
        prompt: "If step A causes B, B causes C, and C causes A (temporal loop). What is the stable output state of the system?",
        steps: [
            {
                round: 1,
                disagreement: 0.90,
                entropy: 0.95,
                savings: 82,
                coord: "Initiating analysis of temporal loop parameters. Unstable causal cycles detected.",
                pros: "Analyzing temporal step A to B. Causal flow indicates a standard dependency.",
                skep: "Wait, step C points back to A! This creates a closed temporal loop.",
                synth: "Gathering initial state parameters. Causal resolution requires breaking the cycle.",
                activeNodes: ['node-agent1', 'node-agent2'],
                activeLines: ['line-node-agent1-node-agent2']
            },
            {
                round: 2,
                disagreement: 0.60,
                entropy: 0.70,
                savings: 84,
                coord: "Bistable state check in progress. Core loop remains active.",
                pros: "If step C points to A, the state at A must oscillate.",
                skep: "Confirmed. There is no stable attractor. Let's try to isolate the loop boundary.",
                synth: "The system is entering an unstable bistable state.",
                activeNodes: ['node-agent2', 'node-agent3'],
                activeLines: ['line-node-agent2-node-agent3']
            },
            {
                round: 3,
                disagreement: 0.95,
                entropy: 0.98,
                savings: 85,
                isContradiction: true,
                coord: "CRITICAL CRASH: Bistable feedback loop has locked agent states. Invoking causal rollback!",
                pros: "LOGICAL COLLAPSE: A cannot be both true and false simultaneously.",
                skep: "CONTRADICTION DETECTED: Reflexive relation violates temporal asymmetry constraints.",
                synth: "Bistable state is diverging. Rollback required.",
                activeNodes: ['node-coordinator', 'node-agent1', 'node-agent2', 'node-agent3'],
                activeLines: ['line-node-coordinator-node-agent1', 'line-node-coordinator-node-agent2', 'line-node-coordinator-node-agent3']
            },
            {
                round: 4,
                disagreement: 0.05,
                entropy: 0.08,
                savings: 87,
                coord: "System stabilized. Resolving paradox to ground state.",
                pros: "Setting temporal boundary condition to False. System is stable.",
                skep: "No further loops detected. Asymmetry verified.",
                synth: "Final plan: The temporal loop is resolved to ground state.",
                activeNodes: ['node-agent3', 'node-coordinator'],
                activeLines: ['line-node-coordinator-node-agent3']
            }
        ],
        finalAnswer: "Ground state resolved. The temporal paradox is FALSE. The system cannot sustain a stable True state under closed loop oscillation."
    },
    monty: {
        prompt: "You choose Door 1. The host reveals a goat behind Door 3. Is it to your advantage to switch to Door 2?",
        steps: [
            {
                round: 1,
                disagreement: 0.88,
                entropy: 0.92,
                savings: 81,
                coord: "Evaluating probability space. Initial choice is Door 1.",
                pros: "Initial choice is Door 1, giving 1/3 probability. Other doors have 2/3.",
                skep: "Wait, the host opened Door 3 to show a goat. Is it a random choice?",
                synth: "The host's choice is conditional. He cannot open the car door or your door.",
                activeNodes: ['node-agent1', 'node-agent2'],
                activeLines: ['line-node-agent1-node-agent2']
            },
            {
                round: 2,
                disagreement: 0.30,
                entropy: 0.40,
                savings: 84,
                coord: "Updating beliefs using conditional probabilities.",
                pros: "The entire 2/3 probability of the other doors is now focused on Door 2.",
                skep: "Ah! By switching, we win if our initial choice was wrong (2/3 chance).",
                synth: "Consensus: Switching yields 2/3 winning probability.",
                activeNodes: ['node-agent2', 'node-agent3'],
                activeLines: ['line-node-agent2-node-agent3']
            },
            {
                round: 3,
                disagreement: 0.02,
                entropy: 0.05,
                savings: 86,
                coord: "Deliberation converged. Switching is optimal.",
                pros: "Switching confirmed optimal.",
                skep: "Bayesian model verified.",
                synth: "Output final plan with 2/3 switching advantage.",
                activeNodes: ['node-agent3', 'node-coordinator'],
                activeLines: ['line-node-coordinator-node-agent3']
            }
        ],
        finalAnswer: "Switching doors yields a 2/3 winning probability, doubling your initial chances. Recommended Action: SWITCH."
    },
    transitive: {
        prompt: "If A is older than B, B is older than C, C is older than D, D is older than E, and E is older than F. Is F older than A?",
        steps: [
            {
                round: 1,
                disagreement: 0.85,
                entropy: 0.90,
                savings: 83,
                coord: "Analyzing length-5 inequality chain.",
                pros: "Given: A > B, B > C, C > D, D > E, E > F.",
                skep: "Is the relation asymmetric and transitive?",
                synth: "We must trace the connection from F back to A.",
                activeNodes: ['node-agent1', 'node-agent2'],
                activeLines: ['line-node-agent1-node-agent2']
            },
            {
                round: 2,
                disagreement: 0.45,
                entropy: 0.50,
                savings: 85,
                coord: "Determining inequality direction.",
                pros: "Since A > B > C > D > E > F, A is oldest and F is youngest.",
                skep: "F < A, so F is younger than A. The prompt asks if F is older.",
                synth: "Consensus: F is younger than A, meaning the statement is False.",
                activeNodes: ['node-agent2', 'node-agent3'],
                activeLines: ['line-node-agent2-node-agent3']
            },
            {
                round: 3,
                disagreement: 0.08,
                entropy: 0.10,
                savings: 88,
                coord: "Convergence achieved. Outputting negative resolution.",
                pros: "Confirmed F < A.",
                skep: "No contradictions found. Consensus stable.",
                synth: "Output plan: F is older than A is False.",
                activeNodes: ['node-agent3', 'node-coordinator'],
                activeLines: ['line-node-coordinator-node-agent3']
            }
        ],
        finalAnswer: "Since A > B > C > D > E > F, by transitivity F must be younger than A. Therefore, the statement 'F is older than A' is FALSE."
    }
};

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('disagreementChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Disagreement Rate',
                    data: [],
                    borderColor: '#00f2fe',
                    backgroundColor: 'rgba(0, 242, 254, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Consensus Entropy',
                    data: [],
                    borderColor: '#7f00ff',
                    backgroundColor: 'rgba(127, 0, 255, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'JetBrains Mono', size: 10 }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } },
                    min: 0,
                    max: 1.0
                }
            }
        }
    });
}

// Draw connection lines on SVG
function drawConnections() {
    const svg = document.getElementById("networkSvg");
    const svgRect = svg.getBoundingClientRect();
    
    // Clear previous lines
    svg.innerHTML = "";
    
    const nodes = [
        { id: "node-coordinator" },
        { id: "node-agent1" },
        { id: "node-agent2" },
        { id: "node-agent3" }
    ];
    
    // Calculate center coordinates
    const coords = {};
    nodes.forEach(n => {
        const el = document.getElementById(n.id);
        const rect = el.getBoundingClientRect();
        coords[n.id] = {
            x: rect.left - svgRect.left + rect.width / 2,
            y: rect.top - svgRect.top + rect.height / 2
        };
    });
    
    // Draw paths
    const pairs = [
        ["node-coordinator", "node-agent1"],
        ["node-coordinator", "node-agent2"],
        ["node-coordinator", "node-agent3"],
        ["node-agent1", "node-agent2"],
        ["node-agent2", "node-agent3"],
        ["node-agent3", "node-agent1"]
    ];
    
    pairs.forEach(pair => {
        const start = coords[pair[0]];
        const end = coords[pair[1]];
        if (!start || !end) return;
        
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", start.x);
        line.setAttribute("y1", start.y);
        line.setAttribute("x2", end.x);
        line.setAttribute("y2", end.y);
        line.setAttribute("stroke", "rgba(255, 255, 255, 0.08)");
        line.setAttribute("stroke-width", "1.5");
        line.setAttribute("id", `line-${pair[0]}-${pair[1]}`);
        svg.appendChild(line);
    });
}

// Select a scenario
function setScenario(scen) {
    if (isRunning) return;
    currentScenario = scen;
    
    // Update active button state
    ['temporal', 'monty', 'transitive'].forEach(s => {
        const btn = document.getElementById(`btn-scenario-${s}`);
        if (s === scen) {
            btn.classList.add('border-neonCyan', 'bg-white/10');
            btn.classList.remove('border-white/5', 'bg-white/5');
        } else {
            btn.classList.remove('border-neonCyan', 'bg-white/10');
            btn.classList.add('border-white/5', 'bg-white/5');
        }
    });

    // Load prompt
    document.getElementById('customPrompt').value = SCENARIOS[scen].prompt;
}

// Start/Stop Deliberation loop simulation
async function startDeliberation() {
    if (isRunning) {
        // Stop
        clearTimeout(simulationTimeout);
        resetArena();
        return;
    }

    isRunning = true;
    const btn = document.getElementById('btnStart');
    btn.innerHTML = `<i data-lucide="square" class="w-4 h-4"></i> Stop Simulation`;
    btn.classList.replace('from-neonCyan', 'from-red-500');
    btn.classList.replace('to-neonPurple', 'to-red-600');
    lucide.createIcons();

    const scenario = SCENARIOS[currentScenario];
    const steps = scenario.steps;
    
    // Clear chart
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.data.datasets[1].data = [];
    chart.update();

    document.getElementById('statusIndicator').textContent = "DELIBERATING";
    document.getElementById('statusIndicator').className = "text-[10px] font-mono tracking-wider uppercase bg-neonCyan/10 border border-neonCyan/20 px-2 py-0.5 rounded text-neonCyan";

    // Play step by step
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        
        // Speed delay
        const delay = (1500 / speedFactor);

        // Highlight nodes and paths
        resetHighlights();
        
        step.activeNodes.forEach(nodeId => {
            const el = document.getElementById(nodeId).firstElementChild;
            if (nodeId.includes('coordinator')) {
                el.classList.add('glow-gold', 'border-neonGold');
                el.style.setProperty('--glow-color', 'var(--neon-gold)');
            } else if (nodeId.includes('agent1')) {
                el.classList.add('glow-cyan', 'border-neonCyan');
                el.style.setProperty('--glow-color', 'var(--neon-cyan)');
            } else if (nodeId.includes('agent2')) {
                el.classList.add('glow-purple', 'border-neonPurple');
                el.style.setProperty('--glow-color', 'var(--neon-purple)');
            } else if (nodeId.includes('agent3')) {
                el.classList.add('glow-emerald', 'border-neonEmerald');
                el.style.setProperty('--glow-color', 'var(--neon-emerald)');
            }
            el.classList.add('node-pulse');
        });

        step.activeLines.forEach(lineId => {
            const line = document.getElementById(lineId);
            if (line) {
                line.setAttribute("stroke", "#00f2fe");
                line.setAttribute("stroke-width", "3");
            }
        });

        // Set thought bubbles
        document.getElementById('bubble-coord').textContent = step.coord;
        document.getElementById('bubble-pros').textContent = step.pros;
        document.getElementById('bubble-skep').textContent = step.skep;
        document.getElementById('bubble-synth').textContent = step.synth;

        // Set metrics
        document.getElementById('metricRound').textContent = step.round;
        document.getElementById('metricDisagreement').textContent = step.disagreement.toFixed(2);
        document.getElementById('metricEntropy').textContent = step.entropy.toFixed(2);
        document.getElementById('metricSavings').textContent = step.savings + "%";

        // Update Chart
        chart.data.labels.push(`Round ${step.round}`);
        chart.data.datasets[0].data.push(step.disagreement);
        chart.data.datasets[1].data.push(step.entropy);
        chart.update();

        // Check for contradiction / rollback event
        if (step.isContradiction) {
            const overlay = document.getElementById('contradictionOverlay');
            overlay.style.opacity = '1';
            overlay.classList.remove('pointer-events-none');
            
            // Screen shake
            document.body.classList.add('shake-anim');
            
            await new Promise(r => setTimeout(r, 1800));
            
            overlay.style.opacity = '0';
            overlay.classList.add('pointer-events-none');
            document.body.classList.remove('shake-anim');
        } else {
            await new Promise(r => setTimeout(r, delay));
        }
    }

    // Finished
    document.getElementById('finalOutput').textContent = scenario.finalAnswer;
    document.getElementById('statusIndicator').textContent = "CONVERGED";
    document.getElementById('statusIndicator').className = "text-[10px] font-mono tracking-wider uppercase bg-neonEmerald/10 border border-neonEmerald/20 px-2 py-0.5 rounded text-neonEmerald";
    
    resetArenaBtn();
}

function resetHighlights() {
    ['node-coordinator', 'node-agent1', 'node-agent2', 'node-agent3'].forEach(id => {
        const el = document.getElementById(id).firstElementChild;
        el.className = el.className.replace(/glow-\w+/g, '');
        el.classList.remove('border-neonCyan', 'border-neonPurple', 'border-neonEmerald', 'border-neonGold', 'node-pulse');
    });

    const lines = document.getElementById("networkSvg").querySelectorAll("line");
    lines.forEach(l => {
        l.setAttribute("stroke", "rgba(255, 255, 255, 0.08)");
        l.setAttribute("stroke-width", "1.5");
    });
}

function resetArena() {
    resetHighlights();
    document.getElementById('bubble-coord').textContent = "Awaiting start...";
    document.getElementById('bubble-pros').textContent = "Awaiting start...";
    document.getElementById('bubble-skep').textContent = "Awaiting start...";
    document.getElementById('bubble-synth').textContent = "Awaiting start...";
    document.getElementById('metricRound').textContent = "0";
    document.getElementById('metricDisagreement').textContent = "0.00";
    document.getElementById('metricEntropy').textContent = "0.00";
    document.getElementById('metricSavings').textContent = "0%";
    document.getElementById('finalOutput').textContent = "Awaiting deliberation...";
    document.getElementById('statusIndicator').textContent = "IDLE";
    document.getElementById('statusIndicator').className = "text-[10px] font-mono tracking-wider uppercase bg-white/5 px-2 py-0.5 rounded text-slate-500";
    resetArenaBtn();
}

function resetArenaBtn() {
    isRunning = false;
    const btn = document.getElementById('btnStart');
    btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i> Start Deliberation`;
    btn.classList.replace('from-red-500', 'from-neonCyan');
    btn.classList.replace('to-red-600', 'to-neonPurple');
    lucide.createIcons();
}

// SDK Documentation Tabs switcher
function switchTab(tabId) {
    ['pySDK', 'openAI', 'gateway'].forEach(t => {
        const btn = document.getElementById(`btn-tab-${t}`);
        const content = document.getElementById(`tab-${t}-content`);
        if (t === tabId) {
            btn.classList.add('text-neonCyan', 'border-b-2', 'border-neonCyan');
            btn.classList.remove('text-slate-400');
            content.classList.remove('hidden');
        } else {
            btn.classList.remove('text-neonCyan', 'border-b-2', 'border-neonCyan');
            btn.classList.add('text-slate-400');
            content.classList.add('hidden');
        }
    });
}

// Speed slider listener
document.getElementById('speedSlider').addEventListener('input', (e) => {
    speedFactor = parseFloat(e.target.value);
    document.getElementById('speedVal').textContent = speedFactor.toFixed(1) + "x";
});

// Setup on Window Load
window.onload = () => {
    lucide.createIcons();
    initChart();
    setScenario('temporal');
    drawConnections();
    
    // Recalculate SVG connection coordinates on window resize
    window.addEventListener('resize', drawConnections);
};
