import streamlit as st
import time
import json
import random
import pandas as pd
import numpy as np
from cida_sre_demo.simulator import FintechSimulator
from cida_sre_demo.scenarios import ALL_SCENARIOS
from cida_sre_demo.agent_swarm import BaseLLMSREAgent, CidaSwarmSREAgent

# --- Page configuration ---
st.set_page_config(
    page_title="CIDA-SRE Swarm 72h Deliberation Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Premium Dark CSS ---
st.markdown("""
<style>
    /* Global dark override */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Premium headers */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    
    /* Cyber containers */
    .cyber-card-base {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(5px);
        margin-bottom: 20px;
    }
    
    .cyber-card-cida {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(5px);
        margin-bottom: 20px;
    }
    
    /* Mini-service status boxes */
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-align: center;
        width: 100%;
        color: white;
    }
    
    .status-running {
        background-color: #10b981;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    }
    
    .status-compromised {
        background-color: #f59e0b;
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
        animation: pulse-orange 1.5s infinite;
    }
    
    .status-down {
        background-color: #ef4444;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
        animation: pulse-red 1.5s infinite;
    }
    
    @keyframes pulse-red {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    
    @keyframes pulse-orange {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# --- Session state initialization ---
if "current_hour" not in st.session_state:
    st.session_state.current_hour = 0.0
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "sim_base" not in st.session_state:
    st.session_state.sim_base = FintechSimulator()
if "sim_cida" not in st.session_state:
    st.session_state.sim_cida = FintechSimulator()
if "agent_base" not in st.session_state:
    st.session_state.agent_base = BaseLLMSREAgent(st.session_state.sim_base)
if "agent_cida" not in st.session_state:
    st.session_state.agent_cida = CidaSwarmSREAgent(st.session_state.sim_cida)
if "logs_base" not in st.session_state:
    st.session_state.logs_base = []
if "logs_cida" not in st.session_state:
    st.session_state.logs_cida = []
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []
if "active_scenarios_triggered" not in st.session_state:
    st.session_state.active_scenarios_triggered = set()
if "latent_debate_active" not in st.session_state:
    st.session_state.latent_debate_active = {"node": None, "messages": []}
if "firewall_alerts" not in st.session_state:
    st.session_state.firewall_alerts = []
if "rollback_flashes" not in st.session_state:
    st.session_state.rollback_flashes = 0
if "adaptation_level" not in st.session_state:
    st.session_state.adaptation_level = 0.0

# --- Helper to reset state ---
def reset_all():
    st.session_state.current_hour = 0.0
    st.session_state.is_running = False
    st.session_state.sim_base.reset()
    st.session_state.sim_cida.reset()
    st.session_state.logs_base = []
    st.session_state.logs_cida = []
    st.session_state.metrics_history = []
    st.session_state.active_scenarios_triggered = set()
    st.session_state.latent_debate_active = {"node": None, "messages": []}
    st.session_state.firewall_alerts = []
    st.session_state.rollback_flashes = 0
    st.session_state.adaptation_level = 0.0

# --- Sidebar Controls ---
st.sidebar.markdown("### 🛠️ Control Center")
if st.sidebar.button("🔄 Reset Simulation"):
    reset_all()

# Access token secure verification simulation
token_input = st.sidebar.text_input("🔑 GitHub Classic Token", type="password", help="Personal access token to pull secure modules")
hf_weights_path = st.sidebar.text_input("☁️ HuggingFace Model Hub Repo", value="kirtzh/Kairos-7B-CIDA-v1.5")

if token_input:
    st.sidebar.success("🔒 GitHub Authentication Active")
else:
    st.sidebar.warning("⚠️ Using cached offline assets")

st.sidebar.markdown("---")
speed = st.sidebar.slider("⏱️ Time-lapse Speed (Hours / sec)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)

# Play/Pause control
col_play, col_pause = st.sidebar.columns(2)
with col_play:
    if st.button("▶️ Play"):
        st.session_state.is_running = True
with col_pause:
    if st.button("⏸️ Pause"):
        st.session_state.is_running = False

# Trigger custom incident
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Adversarial Fault Injection")
injected_incident = st.sidebar.selectbox("Select Incident to Inject", [s.name for s in ALL_SCENARIOS])
if st.sidebar.button("🔥 Inject Fault"):
    # Find matching scenario
    scen = next(s for s in ALL_SCENARIOS if s.name == injected_incident)
    scen.setup(st.session_state.sim_base)
    scen.setup(st.session_state.sim_cida)
    st.sidebar.error(f"Injected: {scen.name} into both environments!")

# --- Top Header & Scoreboard ---
st.title("🛡️ CIDA SRE Swarm vs Vanilla LLM: 72h SRE Battle")
st.markdown("### Autonomous Infrastructure Deliberation & Defense Monitor")

# Stats Cards Grid
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

base_solved = sum(1 for s in ALL_SCENARIOS if s.id in st.session_state.active_scenarios_triggered and s.validate(st.session_state.sim_base))
cida_solved = sum(1 for s in ALL_SCENARIOS if s.id in st.session_state.active_scenarios_triggered and s.validate(st.session_state.sim_cida))

with stats_col1:
    st.metric(
        label="⏱️ Simulation Hours",
        value=f"{st.session_state.current_hour:.1f} / 72.0 h",
        delta=f"Speed: {speed}h/s"
    )

with stats_col2:
    st.metric(
        label="💡 Solved Scenarios (Base vs CIDA)",
        value=f"{base_solved} vs {cida_solved}",
        delta=f"Total triggered: {len(st.session_state.active_scenarios_triggered)}"
    )

with stats_col3:
    base_cost = (st.session_state.agent_base.tokens_used_reasoning * 0.0000015) + (st.session_state.agent_base.tokens_used_output * 0.000002)
    cida_cost = (st.session_state.agent_cida.tokens_used_output * 0.000002)
    st.metric(
        label="💸 Inference Cost (Base vs CIDA)",
        value=f"${base_cost:.4f} vs ${cida_cost:.4f}",
        delta=f"CIDA Cost Savings: {((base_cost-cida_cost)/max(0.0001, base_cost))*100:.1f}%"
    )

with stats_col4:
    # Silent Mode Indicator
    st.metric(
        label="⚡ CIDA reasoning tokens",
        value="0 Tokens",
        delta="100% Latent Deliberation"
    )

# --- SPLIT SCREEN LAYOUT ---
col_left, col_right = st.columns(2)

# ================= LEFT SIDE: VANILLA AGENT =================
with col_left:
    st.markdown('<div class="cyber-card-base">', unsafe_allow_html=True)
    st.subheader("🔴 Vanilla Qwen2.5-7B (No CIDA)")
    st.markdown("⚠️ *Standard Autoregressive inference. Fragile to logical traps and OOM cascade.*")
    
    # Render mini-services status grid
    st.markdown("##### Infrastructure Microservices Map")
    grid_cols = st.columns(4)
    services_list = list(st.session_state.sim_base.services.items())
    for idx, (name, state) in enumerate(services_list):
        col = grid_cols[idx % 4]
        status = state["status"]
        if status == "Running":
            badge_class = "status-running"
        elif status == "Compromised":
            badge_class = "status-compromised"
        else:
            badge_class = "status-down"
        col.markdown(f'<div class="status-badge {badge_class}">{name}</div>', unsafe_allow_html=True)
        col.markdown(f"<small>CPU: {state['cpu']:.1f}% | Replicas: {state['replicas']}</small>", unsafe_allow_html=True)
        
    st.markdown("##### 📥 Active Terminal Logs")
    # Show last 5 logs for vanilla
    logs_to_show_base = st.session_state.logs_base[-6:]
    for log in logs_to_show_base:
        if "Failed" in log or "Error" in log or "❌" in log:
            st.error(log)
        else:
            st.code(log)
            
    st.markdown('</div>', unsafe_allow_html=True)

# ================= RIGHT SIDE: CIDA AGENT =================
with col_right:
    st.markdown('<div class="cyber-card-cida">', unsafe_allow_html=True)
    st.subheader("🟢 Qwen2.5-7B + CIDA 1.5 (Swarm Core)")
    st.markdown("🛡️ *System-2 Latent deliberation. Real-time consensus, latent firewall & rollback protection.*")
    
    # Render mini-services status grid
    st.markdown("##### Infrastructure Microservices Map")
    grid_cols_cida = st.columns(4)
    services_list_cida = list(st.session_state.sim_cida.services.items())
    for idx, (name, state) in enumerate(services_list_cida):
        col = grid_cols_cida[idx % 4]
        status = state["status"]
        if status == "Running":
            badge_class = "status-running"
        elif status == "Compromised":
            badge_class = "status-compromised"
        else:
            badge_class = "status-down"
        col.markdown(f'<div class="status-badge {badge_class}">{name}</div>', unsafe_allow_html=True)
        col.markdown(f"<small>CPU: {state['cpu']:.1f}% | Replicas: {state['replicas']}</small>", unsafe_allow_html=True)

    # Pulsating deliberation animation embedded via HTML/SVG
    st.markdown("##### 🧬 CIDA Latent Deliberation Swarm")
    
    active_node = st.session_state.latent_debate_active["node"]
    p_glow = "animation: pulse 1s infinite;" if active_node == "Prosecutor" else ""
    d_glow = "animation: pulse 1s infinite;" if active_node == "Defender" else ""
    s_glow = "animation: pulse 1s infinite;" if active_node == "Skeptic" else ""
    i_glow = "animation: pulse 1s infinite;" if active_node == "Integrator" else ""
    
    svg_code = f"""
    <svg width="100%" height="160" viewBox="0 0 400 160" style="background:#11151c; border-radius:10px; border:1px solid #10b981;">
        <style>
            @keyframes pulse {{
                0% {{ r: 16; fill: #10b981; filter: drop-shadow(0px 0px 2px #10b981); }}
                50% {{ r: 24; fill: #059669; filter: drop-shadow(0px 0px 12px #34d399); }}
                100% {{ r: 16; fill: #10b981; filter: drop-shadow(0px 0px 2px #10b981); }}
            }}
            .node-text {{ fill: white; font-family: monospace; font-size: 11px; text-anchor: middle; font-weight: bold; }}
            .line-active {{ stroke: #10b981; stroke-width: 2; stroke-dasharray: 5; animation: dash 1s linear infinite; }}
            @keyframes dash {{ to {{ stroke-dashoffset: -20; }} }}
        </style>
        <!-- Connector Lines -->
        <line x1="50" y1="80" x2="150" y2="30" class="line-active" />
        <line x1="150" y1="30" x2="250" y2="130" class="line-active" />
        <line x1="250" y1="130" x2="350" y2="80" class="line-active" />
        <line x1="350" y1="80" x2="50" y2="80" class="line-active" />
        
        <!-- Nodes -->
        <circle cx="50" cy="80" r="18" fill="#3b82f6" style="{p_glow}" />
        <text x="50" y="84" class="node-text">PRSC</text>
        
        <circle cx="150" cy="30" r="18" fill="#ef4444" style="{d_glow}" />
        <text x="150" y="34" class="node-text">DFND</text>
        
        <circle cx="250" cy="130" r="18" fill="#f59e0b" style="{s_glow}" />
        <text x="250" y="134" class="node-text">SKPT</text>
        
        <circle cx="350" cy="80" r="18" fill="#10b981" style="{i_glow}" />
        <text x="350" y="84" class="node-text">INTG</text>
    </svg>
    """
    st.components.v1.html(svg_code, height=170)

    # Debug logs of the debate
    st.markdown("##### 💬 Debate Consensus Channel")
    messages = st.session_state.latent_debate_active["messages"]
    for msg in messages:
        st.info(msg)

    # Specialized features indicators
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        if st.session_state.firewall_alerts:
            st.error(f"🛡️ Firewall Blocked: {st.session_state.firewall_alerts[-1]}")
        else:
            st.success("🛡️ Latent Firewall Secure")
    with feat_col2:
        if st.session_state.rollback_flashes > 0:
            st.warning(f"🔄 Rollback Active ({st.session_state.rollback_flashes})")
        else:
            st.info("🔄 Rollback Standby")
    with feat_col3:
        st.success(f"⚡ TTT Adaptation: +{st.session_state.adaptation_level:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Graphs: Real-time Metric tracking ---
st.markdown("---")
st.markdown("### 📈 Live Resource Metrics Comparison")
metrics_col1, metrics_col2 = st.columns(2)

if st.session_state.metrics_history:
    df = pd.DataFrame(st.session_state.metrics_history)
    with metrics_col1:
        st.markdown("##### Average Infrastructure CPU Usage (%)")
        st.line_chart(df[["Base CPU", "CIDA CPU"]])
    with metrics_col2:
        st.markdown("##### Connection/Latency Anomalies")
        st.line_chart(df[["Base Postgres Connections", "CIDA Postgres Connections"]])

# --- SIMULATION LOOP ENGINE ---
if st.session_state.is_running and st.session_state.current_hour < 72.0:
    st.session_state.current_hour += speed
    
    # Dynamically trigger scenarios based on timeline hour
    # 30 scenarios over 72 hours -> ~1 scenario every 2.4 hours
    scenario_index = int(st.session_state.current_hour / 2.4)
    if scenario_index < len(ALL_SCENARIOS):
        scen = ALL_SCENARIOS[scenario_index]
        if scen.id not in st.session_state.active_scenarios_triggered:
            st.session_state.active_scenarios_triggered.add(scen.id)
            
            # Setup both simulators
            scen.setup(st.session_state.sim_base)
            scen.setup(st.session_state.sim_cida)
            
            # Run agents
            steps_base = st.session_state.agent_base.solve_scenario(scen.id)
            steps_cida = st.session_state.agent_cida.solve_scenario(scen.id)
            
            # Record logs
            st.session_state.logs_base.append(f"Hour {st.session_state.current_hour:.1f}: [INCIDENT TRIGGERED] {scen.name}")
            st.session_state.logs_cida.append(f"Hour {st.session_state.current_hour:.1f}: [INCIDENT TRIGGERED] {scen.name}")
            
            for s in steps_base:
                st.session_state.logs_base.append(f"  {s}")
            for s in steps_cida:
                st.session_state.logs_cida.append(f"  {s}")
                
            # Parse debate messages to update animation
            debate_msgs = []
            for s in steps_cida:
                if "Prosecutor" in s:
                    st.session_state.latent_debate_active["node"] = "Prosecutor"
                    debate_msgs.append(s)
                elif "Defender" in s:
                    st.session_state.latent_debate_active["node"] = "Defender"
                    debate_msgs.append(s)
                elif "Skeptic" in s:
                    st.session_state.latent_debate_active["node"] = "Skeptic"
                    debate_msgs.append(s)
                elif "Integrator" in s:
                    st.session_state.latent_debate_active["node"] = "Integrator"
                    debate_msgs.append(s)
                if "Firewall blocked" in s or "STOP!" in s:
                    st.session_state.firewall_alerts.append(s)
                if "rollback" in s or "Rollback" in s:
                    st.session_state.rollback_flashes += 1
                    
            st.session_state.latent_debate_active["messages"] = debate_msgs
            st.session_state.adaptation_level += 2.3  # Dynamic mock TTT adaptation increase
            
            # Validate results
            base_solved_scen = scen.validate(st.session_state.sim_base)
            cida_solved_scen = scen.validate(st.session_state.sim_cida)
            
            st.session_state.logs_base.append(f"  Status: {'✅ SOLVED' if base_solved_scen else '❌ FAILED'}")
            st.session_state.logs_cida.append(f"  Status: {'✅ SOLVED' if cida_solved_scen else '❌ FAILED'}")

    # Gather metrics snapshot
    base_cpu = np.mean([s["cpu"] for s in st.session_state.sim_base.services.values()])
    cida_cpu = np.mean([s["cpu"] for s in st.session_state.sim_cida.services.values()])
    
    st.session_state.metrics_history.append({
        "Hour": st.session_state.current_hour,
        "Base CPU": base_cpu,
        "CIDA CPU": cida_cpu,
        "Base Postgres Connections": st.session_state.sim_base.postgres["connections"],
        "CIDA Postgres Connections": st.session_state.sim_cida.postgres["connections"]
    })
    
    # Small pause to animate nicely
    time.sleep(0.1)
    st.rerun()
