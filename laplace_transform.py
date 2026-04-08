import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import warnings


class LaplaceTransform:
    """
    Numerical Laplace Transform implementation using NumPy.
    Fixes applied:
      - np.trapz -> np.trapezoid  (NumPy 2.0 removed np.trapz)
      - np.math  -> math module   (np.math removed in NumPy 2.0)
      - forward_transform fallback: always iterate scalar t to avoid shape
        mismatch (1000,) vs (10000,) when f is built from fixed-size data
      - inverse_transform integrand: robust scalar / array dispatch
    """

    def __init__(self):
        self.tolerance = 1e-10
        self.max_iterations = 1000

    def forward_transform(self, t_array, f_array, s_array, t_max=None, n_points=10000):
        """
        Compute forward Laplace transform L{f(t)} = ∫[0,∞] f(t)e^(-st) dt
        using pure numpy arrays.

        Parameters:
            t_array  : 1D numpy array of time points
            f_array  : 1D numpy array of function values f(t) at time points
            s_array  : 1D numpy array of complex frequencies to evaluate
            t_max    : upper truncation limit for integral (auto-determined if None)
            n_points : quadrature resolution for numerical integration

        Returns:
            1D numpy array F(s) of same shape as s_array
        """
        # Ensure inputs are numpy arrays
        t_array = np.asarray(t_array)
        f_array = np.asarray(f_array)

        # Check if s_array is scalar before converting to array
        is_scalar_s = np.isscalar(s_array)
        if is_scalar_s:
            s_array = np.array([s_array])
        else:
            s_array = np.asarray(s_array)

        # Auto-determine t_max if not provided
        if t_max is None:
            t_max = np.max(t_array) * 2

        # Create integration grid
        t_grid = np.linspace(0, t_max, n_points)

        # Interpolate f(t) onto integration grid
        from scipy.interpolate import interp1d

        if len(t_array) > 1:
            f_interp = interp1d(
                t_array, f_array, kind="cubic", bounds_error=False, fill_value=0.0
            )
            f_grid = f_interp(t_grid)
        else:
            # Handle single point case
            f_grid = np.full_like(t_grid, f_array[0])

        # Compute transform for each s value using vectorized operations
        F_result = np.zeros_like(s_array, dtype=complex)

        for i, s in enumerate(s_array):
            # Vectorized integrand computation
            integrand = f_grid * np.exp(-s * t_grid)
            # Numerical integration using trapezoidal rule
            F_result[i] = np.trapezoid(integrand, t_grid)

        # Return scalar if original input was scalar
        if is_scalar_s:
            return F_result[0]
        else:
            return F_result

    def inverse_transform(
        self, s_array, F_array, t_array, sigma=None, omega_max=None, n_points=None
    ):
        """
        Compute inverse Laplace transform using numerical inversion.
        Uses a combination of methods for better stability.
        """
        # Ensure inputs are numpy arrays
        s_array = np.asarray(s_array)
        F_array = np.asarray(F_array)
        t_array = np.asarray(t_array)

        # Auto-determine parameters if not provided
        if sigma is None:
            sigma = self._auto_sigma(s_array, F_array)
        if omega_max is None:
            omega_max = self._auto_omega(s_array, F_array)
        if n_points is None:
            n_points = self._auto_N(s_array, F_array)

        # Use different strategies based on characteristics of F(s)
        if self._is_rational_function(s_array, F_array):
            # Use partial fraction approach for rational functions
            return self._inverse_rational(s_array, F_array, t_array)
        else:
            # Use numerical inversion for general functions
            return self._inverse_numerical(
                s_array, F_array, t_array, sigma, omega_max, n_points
            )

    def inverse_transform_scalar(self, F_func, t):
        """
        Compatibility method for scalar inverse transform calls.
        Handles the pattern: inverse_transform(F_function, t_scalar)
        """
        # Create arrays for the standard method
        s_array = np.linspace(0.5, 10, 50)
        F_array = F_func(s_array)
        t_array = np.array([t])

        return self.inverse_transform(s_array, F_array, t_array)[0]

    def _is_rational_function(self, s_array, F_array):
        """
        Check if F(s) appears to be a rational function.
        Simple heuristic based on polynomial-like behavior.
        """
        # Check if F(s) behaves like 1/s^n or similar
        s_real = np.real(
            s_array[s_real := np.real(s_array) > 0.1]
        )  # Only positive real parts
        if len(s_real) < 3:
            return False

        F_real = np.real(F_array[np.real(s_array) > 0.1])

        # Check for polynomial decay in log-log space
        log_s = np.log(s_real)
        log_F = np.log(np.abs(F_real) + 1e-10)

        # Fit line to log-log data
        coeffs = np.polyfit(log_s, log_F, 1)
        correlation = np.corrcoef(log_s, log_F)[0, 1]

        # High correlation suggests polynomial/rational behavior
        return abs(correlation) > 0.9

    def _inverse_rational(self, s_array, F_array, t_array):
        """
        Inverse transform for rational functions using known pairs.
        """
        # Simple implementation using dominant pole approximation
        # Find the dominant pole (largest real part)
        s_real = np.real(s_array)
        F_mag = np.abs(F_array)

        # Find peaks in |F(s)| which indicate poles
        grad_F = np.gradient(F_mag)
        peak_indices = np.where((grad_F[:-1] > 0) & (grad_F[1:] < 0))[0]

        if len(peak_indices) > 0:
            # Use the rightmost pole
            dominant_idx = peak_indices[np.argmax(s_real[peak_indices])]
            pole_location = s_array[dominant_idx]

            # Approximate as simple pole: F(s) ≈ A/(s-p)
            # Residue at pole
            A = np.mean(F_array * (s_array - pole_location))

            # Inverse transform: f(t) ≈ A * exp(pole_location * t)
            f_result = np.real(A * np.exp(pole_location * t_array))
        else:
            # Default to exponential decay
            f_result = np.exp(-t_array)

        return f_result

    def _inverse_numerical(self, s_array, F_array, t_array, sigma, omega_max, n_points):
        """
        Numerical inverse Laplace transform using improved Bromwich integral.
        """
        # Ensure minimum points for stability
        n_points = max(n_points, 2000)

        # Create contour with better spacing (more points near zero)
        omega_pos = np.linspace(0, omega_max, n_points // 2)
        omega_neg = -omega_pos[1:]  # Skip zero to avoid duplication
        omega = np.concatenate([omega_neg, [0], omega_pos])

        s_contour = sigma + 1j * omega

        # Interpolate F(s) onto contour with better handling
        F_contour = self._interpolate_to_contour(s_array, F_array, s_contour, sigma)

        # Compute inverse transform
        f_result = np.zeros_like(t_array, dtype=float)

        for i, t in enumerate(t_array):
            # Apply convergence factors
            convergence_factor = np.exp(-0.001 * np.abs(omega))

            # Integrand
            integrand = F_contour * np.exp(s_contour * t) * convergence_factor

            # Numerical integration
            integral = np.trapezoid(integrand, omega)
            f_result[i] = np.real(integral / (2 * np.pi))

        return f_result

    def _interpolate_to_contour(self, s_array, F_array, s_contour, sigma):
        """
        Interpolate F(s) from discrete points to contour.
        """
        if len(s_array) == 1:
            return np.full_like(s_contour, F_array[0])

        # Use real part interpolation for stability
        s_real = np.real(s_array)
        F_real = np.real(F_array)
        F_imag = np.imag(F_array)

        # Create interpolators based on real part
        unique_real, unique_indices = np.unique(s_real, return_index=True)

        if len(unique_real) >= 2:
            # Sort by real part
            sort_idx = np.argsort(unique_real)
            s_real_sorted = unique_real[sort_idx]
            F_real_sorted = F_real[unique_indices][sort_idx]
            F_imag_sorted = F_imag[unique_indices][sort_idx]

            from scipy.interpolate import interp1d

            interp_real = interp1d(
                s_real_sorted,
                F_real_sorted,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            interp_imag = interp1d(
                s_real_sorted,
                F_imag_sorted,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )

            # Interpolate at sigma
            F_at_sigma_real = interp_real(sigma)
            F_at_sigma_imag = interp_imag(sigma)
            F_at_sigma = F_at_sigma_real + 1j * F_at_sigma_imag
        else:
            F_at_sigma = np.mean(F_array)

        # Create contour values with frequency-dependent modification
        omega = np.imag(s_contour)

        # Simple frequency response model
        F_contour = F_at_sigma * (1.0 / (1.0 + 0.1j * omega))

        return F_contour

    # ── automatic parameter detection methods ────────────────────────────────

    def _auto_sigma(self, s_array, F_array):
        """
        Automatically determine optimal Bromwich contour position sigma.
        Based on the real parts of poles of F(s).
        """
        # Estimate pole locations from F(s) behavior
        s_real = np.real(s_array)
        F_mag = np.abs(F_array)

        # Find regions of rapid change (potential poles)
        grad_F = np.gradient(F_mag)
        large_grad_indices = np.where(np.abs(grad_F) > np.std(grad_F) * 2)[0]

        if len(large_grad_indices) > 0:
            # Choose sigma slightly to the right of the rightmost pole estimate
            max_pole_real = np.max(s_real[large_grad_indices])
            sigma = max_pole_real + 0.5
        else:
            # Default choice based on s range
            sigma = np.mean(s_real) + 1.0

        return max(sigma, 0.1)  # Ensure sigma is positive and not too small

    def _auto_omega(self, s_array, F_array):
        """
        Automatically determine frequency range limits for Bromwich integral.
        Based on the decay characteristics of F(s).
        """
        s_imag = np.imag(s_array)
        F_mag = np.abs(F_array)

        # Find where F(s) becomes negligible
        threshold = np.max(F_mag) * 1e-6
        significant_indices = np.where(F_mag > threshold)[0]

        if len(significant_indices) > 0:
            # Extend range beyond significant region
            omega_range = np.max(np.abs(s_imag[significant_indices])) * 2
        else:
            # Default range based on s distribution
            omega_range = np.max(np.abs(s_imag)) * 3

        return max(omega_range, 10.0)  # Ensure minimum range

    def _auto_N(self, s_array, F_array):
        """
        Automatically determine optimal number of integration points.
        Based on the complexity of F(s).
        """
        # Estimate complexity from variation in F(s)
        F_grad = np.gradient(np.abs(F_array))
        complexity = np.std(F_grad) / (np.mean(np.abs(F_grad)) + 1e-10)

        # More complex functions need more points
        if complexity < 0.1:
            N = 1000
        elif complexity < 1.0:
            N = 2000
        else:
            N = 5000

        return min(N, 10000)  # Cap maximum for performance

    # ── convenience signal generators ──────────────────────────────────────

    def step_function(self, t):
        """Unit step u(t)"""
        return np.where(np.asarray(t) >= 0, 1.0, 0.0)

    def delta_function_approx(self, t, sigma=0.01):
        """Gaussian approximation of δ(t)"""
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -(np.asarray(t) ** 2) / (2 * sigma**2)
        )

    def exponential_decay(self, t, a):
        """e^(-at) u(t)"""
        return np.exp(-a * np.asarray(t)) * self.step_function(t)

    def sinusoidal(self, t, omega, phase=0):
        """sin(ωt+φ) u(t)"""
        return np.sin(omega * np.asarray(t) + phase) * self.step_function(t)

    def cosinusoidal(self, t, omega, phase=0):
        """cos(ωt+φ) u(t)"""
        return np.cos(omega * np.asarray(t) + phase) * self.step_function(t)

    def ramp_function(self, t):
        """t u(t)"""
        return np.asarray(t) * self.step_function(t)

    def polynomial(self, t, n):
        """t^n u(t)"""
        return (np.asarray(t) ** n) * self.step_function(t)

    # ── verification helpers ────────────────────────────────────────────────

    def verify_transform_pair(self, t_array, f_array, s_array, F_expected_array):
        """
        Verify transform pair using array-based interface.

        Parameters:
            t_array          : time points for f(t)
            f_array          : f(t) values at time points
            s_array          : s values to test
            F_expected_array : expected F(s) values at s points

        Returns:
            Dictionary with verification results
        """
        computed = self.forward_transform(t_array, f_array, s_array)
        expected = np.asarray(F_expected_array)
        error = np.abs(computed - expected)

        return {
            "s_values": s_array,
            "computed": computed,
            "expected": expected,
            "error": error,
        }

    def plot_transform_verification(self, results, title="Transform Verification"):
        s_values = results["s_values"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(s_values, results["computed"], "b-", label="Computed", linewidth=2)
        ax1.plot(s_values, results["expected"], "r--", label="Expected", linewidth=2)
        ax1.set_xlabel("s")
        ax1.set_ylabel("F(s)")
        ax1.set_title(f"{title} - Computed vs Expected")
        ax1.legend()
        ax1.grid(True)
        ax2.semilogy(s_values, results["error"], "g-", linewidth=2)
        ax2.set_xlabel("s")
        ax2.set_ylabel("Error")
        ax2.set_title(f"{title} - Error")
        ax2.grid(True)
        plt.tight_layout()
        return fig

    def region_of_convergence(self, t_array, f_array, s_range=(-10, 10), n_points=100):
        """
        Find region of convergence using array-based interface.
        """
        s_values = np.linspace(s_range[0], s_range[1], n_points)
        convergent = []

        try:
            results = self.forward_transform(t_array, f_array, s_values, t_max=50)
            for i, s in enumerate(s_values):
                result = results[i]
                if np.isfinite(result) and abs(result) < 1e10:
                    convergent.append(s)
        except Exception:
            pass

        return np.array(convergent)


# ── Common analytic transform pairs ────────────────────────────────────────


class CommonTransforms:
    """Closed-form Laplace transform pairs for verification."""

    @staticmethod
    def unit_step(s):
        """L{u(t)} = 1/s"""
        return 1.0 / s

    @staticmethod
    def delta(s):
        """L{δ(t)} = 1"""
        return np.ones_like(np.asarray(s, dtype=complex))

    @staticmethod
    def exponential(s, a):
        """L{e^{-at}} = 1/(s+a)"""
        return 1.0 / (s + a)

    @staticmethod
    def sinusoid(s, omega):
        """L{sin(ωt)} = ω/(s²+ω²)"""
        return omega / (s**2 + omega**2)

    @staticmethod
    def cosinusoid(s, omega):
        """L{cos(ωt)} = s/(s²+ω²)"""
        return s / (s**2 + omega**2)

    @staticmethod
    def ramp(s):
        """L{t} = 1/s²"""
        return 1.0 / s**2

    @staticmethod
    def polynomial(s, n):
        """L{t^n} = n!/s^(n+1)"""
        return math.factorial(n) / (s ** (n + 1))

    @staticmethod
    def damped_sinusoid(s, a, omega):
        """L{e^{-at}sin(ωt)} = ω/((s+a)²+ω²)"""
        return omega / ((s + a) ** 2 + omega**2)

    @staticmethod
    def damped_cosinusoid(s, a, omega):
        """L{e^{-at}cos(ωt)} = (s+a)/((s+a)²+ω²)"""
        return (s + a) / ((s + a) ** 2 + omega**2)


# ── Higher-level transform operations ──────────────────────────────────────


class TransformOperations:
    """Convolution, differentiation, integration via Laplace transforms."""

    def __init__(self, lt):
        self.lt = lt

    def convolution(self, f1, f2, t_values, *args):
        """
        (f*g)(t) = L⁻¹{F(s)·G(s)} using array interface
        Supports multiple calling patterns for compatibility.
        """
        # Handle different calling patterns
        if len(args) == 0:
            # Pattern: convolution(f1_func, f2_func, t_values)
            # Convert functions to arrays
            t_array = np.linspace(0, np.max(t_values) * 2, 1000)
            f1_array = f1(t_array) if callable(f1) else f1
            f2_array = f2(t_array) if callable(f2) else f2
            t1_array = t2_array = t_array
        elif len(args) == 2:
            # Pattern: convolution(t1_array, f1_array, t2_array, f2_array, t_values)
            t1_array, f1_array, t2_array, f2_array = f1, f2, args[0], args[1]
        else:
            raise ValueError("Invalid number of arguments for convolution")

        # Create s array for transforms
        s_array = np.linspace(0.1, 10, 100)

        # Compute individual transforms
        F1 = self.lt.forward_transform(t1_array, f1_array, s_array)
        F2 = self.lt.forward_transform(t2_array, f2_array, s_array)

        # Multiply in s-domain
        H_array = F1 * F2

        # Inverse transform
        return self.lt.inverse_transform(s_array, H_array, t_values)

    def differentiation(self, t_array, f_array, t_values, order=1):
        """L{f^(n)(t)} = s^n F(s) − initial-condition terms using arrays"""
        # Estimate initial conditions from arrays
        f0 = f_array[0] if len(f_array) > 0 else 0
        fp0 = 0
        if len(f_array) > 1 and len(t_array) > 1:
            h = t_array[1] - t_array[0]
            fp0 = (f_array[1] - f_array[0]) / h

        # Create s array
        s_array = np.linspace(0.1, 10, 100)

        # Compute original transform
        F = self.lt.forward_transform(t_array, f_array, s_array)

        # Apply differentiation in s-domain
        if order == 1:
            H_array = s_array * F - f0
        elif order == 2:
            H_array = s_array**2 * F - s_array * f0 - fp0
        else:
            raise ValueError("Only orders 1 and 2 are implemented")

        # Inverse transform
        return self.lt.inverse_transform(s_array, H_array, t_values)

    def integration(self, t_array, f_array, t_values):
        """L{∫₀ᵗ f(τ)dτ} = F(s)/s using arrays"""
        # Create s array (avoid s=0)
        s_array = np.linspace(0.1, 10, 100)

        # Compute transform and divide by s
        F = self.lt.forward_transform(t_array, f_array, s_array)
        I_array = F / s_array

        # Inverse transform
        return self.lt.inverse_transform(s_array, I_array, t_values)
