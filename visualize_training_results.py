"""
Visualization script for Parallel Programming Training Results
Parses log files and creates comprehensive visualizations
"""

import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class TrainingLogParser:
    """Parse training log files and extract metrics"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.epochs = []
        self.losses = []
        self.times = []
        self.total_time = 0.0
        self.final_loss = 0.0
        self.epoch_times = []
        self.memory_used_mb = 0
        self.memory_total_mb = 0
        self.early_stopped = False
        self.best_epoch = 0
        
    def parse(self) -> bool:
        """Parse the log file and extract all metrics"""
        if not os.path.exists(self.log_file):
            print(f"[WARNING] Log file not found: {self.log_file}")
            return False
            
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Parse epoch logs: [timestamp] Epoch   X | Loss: Y | Time: Zs
        epoch_pattern = r'Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)s'
        matches = re.findall(epoch_pattern, content)
        
        for match in matches:
            epoch = int(match[0])
            loss = float(match[1])
            time = float(match[2])
            self.epochs.append(epoch)
            self.losses.append(loss)
            self.times.append(time)
        
        # Parse training summary
        # Total Training Time: X seconds
        total_time_match = re.search(r'Total Training Time:\s+([\d.]+)\s+seconds', content)
        if total_time_match:
            self.total_time = float(total_time_match.group(1))
        
        # Final Reconstruction Loss: X
        final_loss_match = re.search(r'Final Reconstruction Loss:\s+([\d.]+)', content)
        if final_loss_match:
            self.final_loss = float(final_loss_match.group(1))
        
        # Parse epoch times from summary
        epoch_time_pattern = r'Epoch\s+(\d+):\s+([\d.]+)\s+seconds'
        epoch_time_matches = re.findall(epoch_time_pattern, content)
        for match in epoch_time_matches:
            self.epoch_times.append(float(match[1]))
        
        # Parse memory usage
        memory_match = re.search(r'Used:\s+(\d+)\s+MB.*?Total:\s+(\d+)\s+MB', content, re.DOTALL)
        if memory_match:
            self.memory_used_mb = int(memory_match.group(1))
            self.memory_total_mb = int(memory_match.group(2))
        
        # Check for early stopping
        if 'Early stopping triggered' in content or 'EARLY STOP' in content:
            self.early_stopped = True
            # Find best epoch
            best_match = re.search(r'Best loss:.*?at epoch\s+(\d+)', content)
            if best_match:
                self.best_epoch = int(best_match.group(1))
        
        return len(self.epochs) > 0
    
    def get_data(self) -> Dict:
        """Return parsed data as dictionary"""
        return {
            'epochs': self.epochs,
            'losses': self.losses,
            'times': self.times,
            'total_time': self.total_time,
            'final_loss': self.final_loss,
            'epoch_times': self.epoch_times,
            'memory_used_mb': self.memory_used_mb,
            'memory_total_mb': self.memory_total_mb,
            'memory_usage_percent': (self.memory_used_mb / self.memory_total_mb * 100) if self.memory_total_mb > 0 else 0,
            'early_stopped': self.early_stopped,
            'best_epoch': self.best_epoch,
            'num_epochs': len(self.epochs)
        }


def plot_loss_curves(parsers: Dict[str, TrainingLogParser], save_path: str = None):
    """Plot loss curves for all phases"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'P2': '#1f77b4', 'P3.1': '#ff7f0e', 'P3.2': '#2ca02c'}
    markers = {'P2': 'o', 'P3.1': 's', 'P3.2': '^'}
    
    # Plot 1: Loss vs Epoch
    for phase, parser in parsers.items():
        if parser.epochs:
            data = parser.get_data()
            ax1.plot(data['epochs'], data['losses'], 
                    label=f'{phase} (Final: {data["final_loss"]:.6f})',
                    color=colors.get(phase, 'gray'),
                    marker=markers.get(phase, 'o'),
                    linewidth=2,
                    markersize=8,
                    alpha=0.8)
            
            # Mark best epoch if early stopped
            if data['early_stopped'] and data['best_epoch'] > 0:
                best_loss = data['losses'][data['best_epoch'] - 1] if data['best_epoch'] <= len(data['losses']) else data['final_loss']
                ax1.scatter([data['best_epoch']], [best_loss],
                           color=colors.get(phase, 'gray'),
                           s=200, marker='*', zorder=5,
                           label=f'{phase} Best (Epoch {data["best_epoch"]})')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reconstruction Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Loss vs Time
    for phase, parser in parsers.items():
        if parser.epochs:
            data = parser.get_data()
            cumulative_time = np.cumsum([0] + data['times'])
            ax2.plot(cumulative_time[1:], data['losses'],
                    label=f'{phase}',
                    color=colors.get(phase, 'gray'),
                    marker=markers.get(phase, 'o'),
                    linewidth=2,
                    markersize=8,
                    alpha=0.8)
    
    ax2.set_xlabel('Cumulative Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reconstruction Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Loss vs Training Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved loss curves to {save_path}")
    plt.show()


def plot_performance_comparison(parsers: Dict[str, TrainingLogParser], save_path: str = None):
    """Plot performance comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    phases = list(parsers.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Extract metrics
    total_times = [parsers[p].total_time if parsers[p].total_time > 0 else sum(parsers[p].times) for p in phases]
    final_losses = [parsers[p].final_loss if parsers[p].final_loss > 0 else (parsers[p].losses[-1] if parsers[p].losses else 0) for p in phases]
    avg_epoch_times = [np.mean(parsers[p].times) if parsers[p].times else 0 for p in phases]
    memory_usage = [parsers[p].get_data()['memory_usage_percent'] for p in phases]
    num_epochs = [parsers[p].get_data()['num_epochs'] for p in phases]
    
    # Plot 1: Total Training Time
    ax = axes[0, 0]
    bars = ax.bar(phases, total_times, color=colors[:len(phases)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Total Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Total Training Time Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, total_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_times)*0.02,
               f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Final Loss
    ax = axes[0, 1]
    bars = ax.bar(phases, final_losses, color=colors[:len(phases)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Final Loss', fontsize=11, fontweight='bold')
    ax.set_title('Final Reconstruction Loss Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, final_losses)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_losses)*0.02,
               f'{val:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Average Time per Epoch
    ax = axes[1, 0]
    bars = ax.bar(phases, avg_epoch_times, color=colors[:len(phases)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Time per Epoch (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Average Time per Epoch', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, avg_epoch_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_epoch_times)*0.02,
               f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: GPU Memory Usage
    ax = axes[1, 1]
    bars = ax.bar(phases, memory_usage, color=colors[:len(phases)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Memory Usage (%)', fontsize=11, fontweight='bold')
    ax.set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, memory_usage)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_usage)*0.02,
                   f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved performance comparison to {save_path}")
    plt.show()


def create_summary_table(parsers: Dict[str, TrainingLogParser], inference_times: Dict[str, float] = None):
    """Create a comprehensive summary table"""
    data = []
    
    for phase, parser in parsers.items():
        pdata = parser.get_data()
        row = {
            'Phase': phase,
            'Epochs': pdata['num_epochs'],
            'Final Loss': f"{pdata['final_loss']:.6f}" if pdata['final_loss'] > 0 else f"{pdata['losses'][-1]:.6f}" if pdata['losses'] else 'N/A',
            'Total Time (s)': f"{pdata['total_time']:.2f}" if pdata['total_time'] > 0 else f"{sum(pdata['times']):.2f}",
            'Avg Time/Epoch (s)': f"{np.mean(pdata['times']):.2f}" if pdata['times'] else 'N/A',
            'Memory Usage (%)': f"{pdata['memory_usage_percent']:.2f}%" if pdata['memory_usage_percent'] > 0 else 'N/A',
            'Early Stopped': 'Yes' if pdata['early_stopped'] else 'No',
            'Best Epoch': str(pdata['best_epoch']) if pdata['best_epoch'] > 0 else 'N/A'
        }
        
        if inference_times and phase in inference_times:
            row['Inference (60k)'] = f"{inference_times[phase]:.2f}s"
            row['Speedup vs P2'] = f"{inference_times.get('P2', 1) / inference_times[phase]:.2f}x" if phase != 'P2' and 'P2' in inference_times else 'N/A'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_inference_comparison(inference_times: Dict[str, float], save_path: str = None):
    """Plot inference time comparison"""
    if not inference_times:
        print("[WARNING] No inference times provided")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phases = list(inference_times.keys())
    times = list(inference_times.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(phases, times, color=colors[:len(phases)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Time for 60,000 Images', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    if 'P2' in inference_times:
        p2_time = inference_times['P2']
        for i, (phase, time) in enumerate(zip(phases, times)):
            if phase != 'P2':
                speedup = p2_time / time
                ax.text(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height() + max(times)*0.02,
                       f'{speedup:.1f}x faster', ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax.text(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height()/2,
                   f'{time:.2f}s', ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved inference comparison to {save_path}")
    plt.show()


def main():
    """Main function to generate all visualizations"""
    # Define log file paths (adjust based on your directory structure)
    base_dir = Path("train")
    log_files = {
        'P2': base_dir / "P2" / "logs" / "phase2_training.log",
        'P3.1': base_dir / "P3_1" / "logs" / "phase3_v1_training.log",
        'P3.2': base_dir / "P3_2" / "logs" / "phase3_v2_training.log"
    }
    
    # Parse all log files
    parsers = {}
    for phase, log_path in log_files.items():
        parser = TrainingLogParser(str(log_path))
        if parser.parse():
            parsers[phase] = parser
            print(f"[INFO] Parsed {phase}: {parser.get_data()['num_epochs']} epochs")
        else:
            print(f"[WARNING] Failed to parse {phase} log file")
    
    if not parsers:
        print("[ERROR] No log files could be parsed. Please check file paths.")
        return
    
    # Inference times from notebook (in seconds for 60k images)
    inference_times = {
        'P2': 10.97,
        'P3.1': 1.57,
        'P3.2': 0.64
    }
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Loss curves
    print("\n[1/4] Plotting loss curves...")
    plot_loss_curves(parsers, save_path="training_loss_curves.png")
    
    # 2. Performance comparison
    print("\n[2/4] Plotting performance comparison...")
    plot_performance_comparison(parsers, save_path="performance_comparison.png")
    
    # 3. Inference comparison
    print("\n[3/4] Plotting inference comparison...")
    plot_inference_comparison(inference_times, save_path="inference_comparison.png")
    
    # 4. Summary table
    print("\n[4/4] Creating summary table...")
    df = create_summary_table(parsers, inference_times)
    print("\n" + "="*60)
    print("TRAINING SUMMARY TABLE")
    print("="*60)
    print(df.to_string(index=False))
    print("\n")
    
    # Save table to CSV
    df.to_csv("training_summary.csv", index=False)
    print("[INFO] Saved summary table to training_summary.csv")


if __name__ == "__main__":
    main()

