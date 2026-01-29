"""
Performance Benchmark: CPU vs FPGA
Compares inference performance between CPU and FPGA acceleration

Features:
- Automated benchmark with synthetic data
- Real image benchmark
- Statistical analysis
- Performance visualization
"""

import numpy as np
import time
import os
import sys
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='CPU vs FPGA Performance Test')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--output', type=str, default='logs/benchmark_results.json', 
                       help='Output file for results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--fpga_port', type=str, default=None, help='FPGA serial port')
    return parser.parse_args()


class BenchmarkResult:
    """Container for benchmark results"""
    def __init__(self, name: str):
        self.name = name
        self.latencies: List[float] = []
    
    def add(self, latency_ms: float):
        self.latencies.append(latency_ms)
    
    @property
    def mean(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0
    
    @property
    def std(self) -> float:
        return np.std(self.latencies) if self.latencies else 0
    
    @property
    def min(self) -> float:
        return np.min(self.latencies) if self.latencies else 0
    
    @property
    def max(self) -> float:
        return np.max(self.latencies) if self.latencies else 0
    
    @property
    def p50(self) -> float:
        return np.percentile(self.latencies, 50) if self.latencies else 0
    
    @property
    def p95(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0
    
    @property
    def p99(self) -> float:
        return np.percentile(self.latencies, 99) if self.latencies else 0
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'iterations': len(self.latencies),
            'mean_ms': self.mean,
            'std_ms': self.std,
            'min_ms': self.min,
            'max_ms': self.max,
            'p50_ms': self.p50,
            'p95_ms': self.p95,
            'p99_ms': self.p99,
            'throughput_fps': 1000 / self.mean if self.mean > 0 else 0
        }


def benchmark_cpu(model, images: List[np.ndarray], warmup: int = 10) -> BenchmarkResult:
    """Benchmark CPU inference"""
    print("[INFO] Benchmarking CPU inference...")
    result = BenchmarkResult("CPU (TensorFlow)")
    
    # Warmup
    for i in range(warmup):
        _ = model.predict(images[i % len(images)].reshape(1, 64, 64, 1), verbose=0)
    
    # Benchmark
    for img in images:
        start = time.perf_counter()
        _ = model.predict(img.reshape(1, 64, 64, 1), verbose=0)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        result.add(elapsed)
    
    return result


def benchmark_fpga(fpga, images: List[np.ndarray], warmup: int = 10) -> BenchmarkResult:
    """Benchmark FPGA inference"""
    print("[INFO] Benchmarking FPGA inference...")
    result = BenchmarkResult("FPGA")
    
    # Warmup
    for i in range(warmup):
        _ = fpga.process_image(images[i % len(images)])
    
    # Benchmark
    for img in images:
        start = time.perf_counter()
        _ = fpga.process_image(img)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        result.add(elapsed)
    
    return result


def benchmark_fpga_simulator(images: List[np.ndarray], latency_ms: float = 5.0) -> BenchmarkResult:
    """Benchmark simulated FPGA (for testing)"""
    from our_improvements.fpga_interface import FPGASimulator
    
    print(f"[INFO] Benchmarking FPGA Simulator ({latency_ms}ms latency)...")
    result = BenchmarkResult(f"FPGA Simulator ({latency_ms}ms)")
    
    sim = FPGASimulator(latency_ms=latency_ms)
    
    for img in images:
        start = time.perf_counter()
        _ = sim.process_image(img)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        result.add(elapsed)
    
    return result


def generate_test_images(count: int, size: int = 64) -> List[np.ndarray]:
    """Generate random test images"""
    print(f"[INFO] Generating {count} test images ({size}x{size})...")
    images = []
    for _ in range(count):
        img = np.random.rand(size, size).astype(np.float32)
        images.append(img)
    return images


def plot_results(results: List[BenchmarkResult], save_path: str = 'logs/benchmark_plot.png'):
    """Generate benchmark comparison plots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('CPU vs FPGA Performance Comparison', fontsize=14, fontweight='bold')
    
    # Plot 1: Latency Comparison (Box Plot)
    ax1 = axes[0]
    data = [r.latencies for r in results]
    labels = [r.name for r in results]
    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean Latency Bar Chart
    ax2 = axes[1]
    means = [r.mean for r in results]
    stds = [r.std for r in results]
    x = range(len(results))
    bars = ax2.bar(x, means, yerr=stds, capsize=5, color=colors[:len(results)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Mean Latency ± Std Dev')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Throughput (FPS)
    ax3 = axes[2]
    fps = [1000 / r.mean if r.mean > 0 else 0 for r in results]
    bars = ax3.bar(x, fps, color=colors[:len(results)])
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15)
    ax3.set_ylabel('Throughput (FPS)')
    ax3.set_title('Inference Throughput')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, f in zip(bars, fps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{f:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Plot saved to {save_path}")


def print_comparison(results: List[BenchmarkResult]):
    """Print comparison table"""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    # Header
    print(f"{'Method':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
    print("-" * 70)
    
    # Results
    for r in results:
        fps = 1000 / r.mean if r.mean > 0 else 0
        print(f"{r.name:<25} {r.mean:<12.2f} {r.std:<12.2f} {r.p95:<12.2f} {fps:<10.1f}")
    
    # Speedup comparison
    if len(results) >= 2:
        print("-" * 70)
        base = results[0].mean
        for r in results[1:]:
            if r.mean > 0:
                speedup = base / r.mean
                print(f"Speedup ({r.name} vs {results[0].name}): {speedup:.2f}x")
    
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs('logs', exist_ok=True)
    
    print("=" * 60)
    print("CPU vs FPGA Performance Benchmark")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print("=" * 60)
    
    # Generate test images
    images = generate_test_images(args.iterations, args.img_size)
    
    results: List[BenchmarkResult] = []
    
    # Benchmark CPU
    try:
        from tensorflow import keras
        model_path = "models/mask_detector_64x64.h5"
        
        if os.path.exists(model_path):
            print(f"\n[INFO] Loading model from {model_path}")
            model = keras.models.load_model(model_path)
            cpu_result = benchmark_cpu(model, images, args.warmup)
            results.append(cpu_result)
        else:
            print(f"[WARN] Model not found at {model_path}, skipping CPU benchmark")
            print("[INFO] Run 'python modified/train_simplified.py' to create model")
    except ImportError:
        print("[WARN] TensorFlow not available, skipping CPU benchmark")
    
    # Benchmark FPGA Simulator (always available for testing)
    for latency in [5.0, 2.0, 1.0]:
        sim_result = benchmark_fpga_simulator(images, latency_ms=latency)
        results.append(sim_result)
    
    # Benchmark Real FPGA (if available)
    if args.fpga_port:
        try:
            from our_improvements.fpga_interface import FPGAInterface
            
            fpga = FPGAInterface(port=args.fpga_port)
            if fpga.is_connected():
                fpga_result = benchmark_fpga(fpga, images, args.warmup)
                results.append(fpga_result)
                fpga.close()
            else:
                print("[WARN] FPGA not connected")
        except Exception as e:
            print(f"[WARN] FPGA benchmark failed: {e}")
    
    # Print results
    if results:
        print_comparison(results)
        
        # Save results
        output_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'iterations': args.iterations,
                'warmup': args.warmup,
                'img_size': args.img_size
            },
            'results': [r.to_dict() for r in results]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"[INFO] Results saved to {args.output}")
        
        # Generate plots
        if args.plot and len(results) > 1:
            plot_results(results)
    else:
        print("[ERROR] No benchmark results to report")


if __name__ == "__main__":
    main()
