import numpy as np
import matplotlib.pyplot as plt

def plot_time_domain(t, y, title="Time Domain Signal", xlabel="Time (t)", ylabel="f(t)"):
    """Plot time domain signal"""
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'b-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_frequency_domain(s, F, title="Frequency Domain", xlabel="Real(s)", ylabel="F(s)"):
    """Plot frequency domain response"""
    plt.figure(figsize=(10, 6))
    plt.plot(s, np.real(F), 'b-', label='Real', linewidth=2)
    plt.plot(s, np.imag(F), 'r-', label='Imaginary', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_magnitude_phase(s, F, title="Magnitude and Phase"):
    """Plot magnitude and phase of frequency response"""
    magnitude = np.abs(F)
    phase = np.angle(F)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.semilogy(s, magnitude, 'b-', linewidth=2)
    ax1.set_ylabel('Magnitude |F(s)|')
    ax1.set_title(f'{title} - Magnitude')
    ax1.grid(True)
    
    ax2.plot(s, phase, 'r-', linewidth=2)
    ax2.set_xlabel('s')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title(f'{title} - Phase')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_signals(t, y1, y2, label1="Signal 1", label2="Signal 2", title="Signal Comparison"):
    """Compare two signals"""
    plt.figure(figsize=(10, 6))
    plt.plot(t, y1, 'b-', label=label1, linewidth=2)
    plt.plot(t, y2, 'r--', label=label2, linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_error(y_true, y_pred):
    """Compute various error metrics"""
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Max Error': max_error
    }

def print_error_metrics(error_dict):
    """Print error metrics in a formatted way"""
    print("Error Metrics:")
    print("-" * 30)
    for metric, value in error_dict.items():
        print(f"{metric:12s}: {value:.6e}")

def validate_convergence(results, tolerance=1e-6):
    """Check if numerical results have converged"""
    if len(results) < 2:
        return False, "Insufficient data for convergence check"
    
    # Check if last few results are within tolerance
    recent_results = results[-5:]
    converged = np.all(np.abs(np.diff(recent_results)) < tolerance)
    
    if converged:
        return True, "Converged successfully"
    else:
        max_diff = np.max(np.abs(np.diff(recent_results)))
        return False, f"Not converged. Max difference: {max_diff:.2e}"

def create_test_signal(signal_type, params, t):
    """Create various test signals"""
    if signal_type == "step":
        return np.where(t >= 0, 1, 0)
    elif signal_type == "ramp":
        return np.where(t >= 0, t, 0)
    elif signal_type == "exponential":
        a = params.get('a', 1)
        return np.where(t >= 0, np.exp(-a * t), 0)
    elif signal_type == "sinusoid":
        omega = params.get('omega', 1)
        phase = params.get('phase', 0)
        return np.where(t >= 0, np.sin(omega * t + phase), 0)
    elif signal_type == "cosinusoid":
        omega = params.get('omega', 1)
        phase = params.get('phase', 0)
        return np.where(t >= 0, np.cos(omega * t + phase), 0)
    elif signal_type == "damped_sin":
        a = params.get('a', 1)
        omega = params.get('omega', 1)
        return np.where(t >= 0, np.exp(-a * t) * np.sin(omega * t), 0)
    elif signal_type == "damped_cos":
        a = params.get('a', 1)
        omega = params.get('omega', 1)
        return np.where(t >= 0, np.exp(-a * t) * np.cos(omega * t), 0)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

def benchmark_function(func, *args, **kwargs):
    """Simple benchmark for function execution time"""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time
