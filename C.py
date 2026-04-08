def u_error_C(self, u, Delta=0.2, tau_d=1.5):
    # --- Stage 1: Quantization ---
    u_quantized = Delta * np.round(u / Delta)

    # --- Stage 2: Time Delay ---
    delay_steps = int(tau_d / self.dt)
    u_corrupted = np.zeros_like(u)

    if delay_steps < len(u):
        u_corrupted[delay_steps:] = u_quantized[:-delay_steps]

    return u_corrupted

def MAE(self, h_clean, h_corrupted):
    return np.trapezoid(np.abs(h_clean - h_corrupted), self.t) / self.t[-1]

def PAE(self, h_clean, h_corrupted):
    return np.max(np.abs(h_clean - h_corrupted))

for name in ["Ramp Input", "Pulse Input"]:
    print(f"\nProcessing: {name}...")

    u_clean = inputs[name]
    u_corrupted = system.u_error_C(u_clean)

    # --- Clean output (Laplace) ---
    U_s_clean = np.array([system.laplace_transform(u_clean, s) for s in s_list])
    H_s_clean = system.H_s(s_list, U_s_clean)
    h_clean = system.inverse_laplace(s_list, H_s_clean)

    # --- Corrupted output (Laplace) ---
    U_s_corr = np.array([system.laplace_transform(u_corrupted, s) for s in s_list])
    H_s_corr = system.H_s(s_list, U_s_corr)
    h_corrupted = system.inverse_laplace(s_list, H_s_corr)

    # --- Metrics ---
    mae = system.MAE(h_clean, h_corrupted)
    pae = system.PAE(h_clean, h_corrupted)

    print(f"  MAE: {mae}")
    print(f"  PAE: {pae}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LEFT: Clean
    axes[0].plot(system.t, u_clean, 'b--', label="Input u(t)")
    axes[0].plot(system.t, h_clean, 'g', label="Output h_clean(t)")
    axes[0].set_title("Clean Input (No Error)")
    axes[0].legend()
    axes[0].grid(True)

    # RIGHT: Corrupted
    axes[1].plot(system.t, u_corrupted, 'r--', label="Corrupted u(t)")
    axes[1].plot(system.t, h_corrupted, 'orange', label="Output h_corrupted(t)")
    axes[1].set_title("Corrupted Input (Quantization + Delay)")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"{name} | MAE={mae:.4f}, PAE={pae:.4f}")
    plt.tight_layout()
    plt.show()
