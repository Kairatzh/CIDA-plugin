import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for charts if it doesn't exist
os.makedirs("pitch_charts", exist_ok=True)

# Define color palette
colors = {
    'cida': '#00e676', # Bright green
    'competitor_light': '#ff8a80', # Light red
    'competitor_dark': '#d50000', # Dark red
    'text': '#ffffff',
    'bg': '#121212',
    'panel_bg': '#1e1e1e',
    'grid': '#333333'
}

# Set global style for pitch deck (Dark Mode, Premium)
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = colors['bg']
plt.rcParams['figure.facecolor'] = colors['bg']
plt.rcParams['text.color'] = colors['text']
plt.rcParams['axes.labelcolor'] = colors['text']
plt.rcParams['xtick.color'] = colors['text']
plt.rcParams['ytick.color'] = colors['text']
plt.rcParams['grid.color'] = colors['grid']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[colors['cida'], colors['competitor_light']])

def create_david_vs_goliath_chart():
    """Generates the parameter size vs performance chart."""
    models = [
        "BLOOM\n(176B)",
        "Falcon\n(40B)",
        "GLM\n(130B)",
        "Llama-2\n(13B)",
        "Llama-1\n(65B)",
        "Llama-2\n(70B)",
        "Command R\n(35B)",
        "Grok-1\n(314B)",
        "Kairos-CIDA\n(7B)"
    ]
    scores = [9.0, 21.0, 28.0, 34.3, 50.9, 57.5, 60.0, 62.3, 62.5]
    params = [176, 40, 130, 13, 65, 70, 35, 314, 7] # For circle size if doing scatter, but we'll do a bar chart sorted by score

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar colors: grey/red for competitors, bright green for CIDA
    bar_colors = [colors['competitor_dark'] if m != "Kairos-CIDA\n(7B)" else colors['cida'] for m in models]
    
    bars = ax.bar(models, scores, color=bar_colors, edgecolor='none', width=0.6, alpha=0.85)
    
    # Highlight CIDA
    bars[-1].set_alpha(1.0)
    bars[-1].set_edgecolor('#ffffff')
    bars[-1].set_linewidth(2)

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('GSM8K Benchmark Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('David vs Goliath: 7B Model Beating 300B+ Giants', fontsize=18, fontweight='bold', pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add a subtitle/annotation
    ax.text(0.02, 0.95, 'CIDA-enabled 7B model matches Grok-1 (314B) & beats Llama-2 (70B)', 
            transform=ax.transAxes, fontsize=11, color='#aaaaaa', style='italic')

    plt.tight_layout()
    plt.savefig('pitch_charts/david_vs_goliath.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

def create_death_loop_chart():
    """Generates the chart showing prevention of death loops."""
    labels = ['Base 7B (No CIDA)', 'Kairos-CIDA 7B']
    success_rate = [0, 100]
    token_waste = [4096, 128] # Max context vs early exit
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Success Rate (Recovery from API Error)
    bars1 = ax1.bar(labels, success_rate, color=[colors['competitor_dark'], colors['cida']], width=0.5, alpha=0.85)
    ax1.set_title('API Failure Recovery Rate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success %')
    ax1.set_ylim(0, 110)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{bar.get_height()}%", ha='center', fontweight='bold')

    # Token Waste
    bars2 = ax2.bar(labels, token_waste, color=[colors['competitor_dark'], colors['cida']], width=0.5, alpha=0.85)
    ax2.set_title('Tokens Wasted in "Death Loop"', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Tokens Generated')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{bar.get_height()}", ha='center', fontweight='bold')
        
    fig.suptitle('Latent Safety: Eliminating API Death Loops', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('pitch_charts/death_loops_prevention.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

def create_swarm_efficiency_chart():
    """Generates a chart showing multi-agent token efficiency."""
    labels = ['LangChain / AutoGen\n(Text-based Swarm)', 'CidaSwarm\n(Latent Workspace)']
    tokens_exchanged = [2500, 0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(labels, tokens_exchanged, color=[colors['competitor_dark'], colors['cida']], height=0.4, alpha=0.85)
    
    ax.set_title('Multi-Agent Communication Cost', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Text Tokens Exchanged Between Agents')
    
    # Add values
    for bar in bars:
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                f"{int(bar.get_width())} Tokens", 
                va='center', fontweight='bold', 
                color=colors['cida'] if bar.get_width() == 0 else colors['text'])

    plt.tight_layout()
    plt.savefig('pitch_charts/swarm_efficiency.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    print("Generating high-res charts for investor pitch...")
    create_david_vs_goliath_chart()
    create_death_loop_chart()
    create_swarm_efficiency_chart()
    print("Done! Charts saved to pitch_charts/ directory.")
