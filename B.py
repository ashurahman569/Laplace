def u_error_A(self, u):
    e = -0.02 * self.t + 0.08 * np.sin(8 * self.t)
    return u + e

def ISE(self, h):
    h_ss = self.steady_state(h)
    return np.trapezoid((h_ss - h) ** 2, self.t)

def energy(self, h):
    return np.trapezoid(h ** 2, self.t)

for name in ["Step Input", "Exponential Input"]:
    print(f"\nProcessing: {name}...")

    u_clean = inputs[name]
    u_corrupted = system.u_error_A(u_clean)

    # --- Clean ---
    U_s_clean = np.array([system.laplace_transform(u_clean, s) for s in s_list])
    H_s_clean = system.H_s(s_list, U_s_clean)
    h_clean = system.inverse_laplace(s_list, H_s_clean)

    # --- Corrupted ---
    U_s_corr = np.array([system.laplace_transform(u_corrupted, s) for s in s_list])
    H_s_corr = system.H_s(s_list, U_s_corr)
    h_corrupted = system.inverse_laplace(s_list, H_s_corr)

    # --- Metrics ---
    ise_clean = system.ISE(h_clean)
    ise_corr = system.ISE(h_corrupted)

    energy_clean = system.energy(h_clean)
    energy_corr = system.energy(h_corrupted)

    print(f"  Clean ISE: {ise_clean}")
    print(f"  Corrupted ISE: {ise_corr}")
    print(f"  Clean Energy: {energy_clean}")
    print(f"  Corrupted Energy: {energy_corr}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LEFT: Clean
    axes[0].plot(system.t, u_clean, 'b--', label="Input u(t)")
    axes[0].plot(system.t, h_clean, 'g', label="Output h_clean(t)")
    axes[0].set_title("Clean Input")
    axes[0].legend()
    axes[0].grid(True)

    # RIGHT: Corrupted
    axes[1].plot(system.t, u_corrupted, 'r--', label="Corrupted u(t)")
    axes[1].plot(system.t, h_corrupted, 'orange', label="Output h_corrupted(t)")
    axes[1].set_title("With Pump Degradation + Noise")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(
        f"{name} | ISE(clean)={ise_clean:.3f}, ISE(corr)={ise_corr:.3f}"
    )
    plt.tight_layout()
    plt.show()
