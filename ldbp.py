import torch
import torch.nn as nn
import numpy as np


class LDBP_LinearLayer(nn.Module):
    """Learnable frequency-domain dispersion compensation layer.

    Replaces the analytical H = exp((-alpha/2)*h - j*beta_omega*h) with
    a directly learned complex frequency-domain filter.
    """

    def __init__(self, Nsub, H_real_init, H_imag_init, trainable_beta2=True):
        super().__init__()
        self.H_real = nn.Parameter(
            torch.tensor(H_real_init, dtype=torch.float32),
            requires_grad=trainable_beta2)
        self.H_imag = nn.Parameter(
            torch.tensor(H_imag_init, dtype=torch.float32),
            requires_grad=trainable_beta2)

    def forward(self, x):
        """
        Args:
            x: (Nsub, 2) complex tensor [X-pol, Y-pol]
        Returns:
            (Nsub, 2) complex tensor
        """
        X = torch.fft.fft(x, dim=0)
        X = torch.fft.fftshift(X, dim=0)
        H = torch.complex(self.H_real, self.H_imag)
        X = X * H  # (Nsub, 2) * (Nsub, 1) broadcasts to both polarizations
        X = torch.fft.ifftshift(X, dim=0)
        return torch.fft.ifft(X, dim=0)


class LDBP_NonlinearLayer(nn.Module):
    """Manakov Kerr nonlinearity compensation layer.

    Applies y * exp(-j * (8/9)*gamma * |y|^2 * h) where h is negative
    for back-propagation.
    """

    def __init__(self, gamma_init, h_step, trainable_gamma=True):
        super().__init__()
        self.h_step = h_step  # negative for DBP
        self.gamma = nn.Parameter(
            torch.tensor([gamma_init], dtype=torch.float32),
            requires_grad=trainable_gamma)

    def forward(self, x):
        """
        Args:
            x: (Nsub, 2) complex tensor [X-pol, Y-pol]
        Returns:
            (Nsub, 2) complex tensor
        """
        P_total = torch.sum(torch.abs(x) ** 2, dim=1, keepdim=True)
        gamma_eff = (8.0 / 9.0) * self.gamma
        phi = gamma_eff * P_total * self.h_step
        return x * torch.exp(-1j * phi)


class LDBP(nn.Module):
    """Learned Digital Back Propagation for dual-polarization signals.

    Stack of alternating LinearLayer (dispersion compensation) and
    NonlinearLayer (Kerr compensation) with EDFA gain removal between spans.

    Structure per span:
        x = x / sqrt(G_lin)
        for _ in range(steps_per_span):
            x = LinearLayer(x)
            x = NonlinearLayer(x)
    """

    def __init__(self, Nsub, fs_sub, fch, L_span, alpha_dBpm, beta2, beta3, gamma,
                 Nspans, steps_per_span, G_lin, trainable_gamma=True,
                 trainable_beta2=True):
        super().__init__()

        # Build frequency grid for H initialization
        df_sub = fs_sub / Nsub
        f_sub = np.arange(-Nsub / 2, Nsub / 2).reshape(-1, 1) * df_sub
        omega_total_sub = 2 * np.pi * (f_sub + fch)

        alpha_np = np.log(10 ** (alpha_dBpm / 10))
        beta_omega_sub = (0.5 * beta2 * omega_total_sub ** 2 +
                          (1.0 / 6.0) * beta3 * omega_total_sub ** 3)

        h_step = L_span / steps_per_span
        h_dbp = -h_step  # negative step for back-propagation

        # Physical-initialized H for a single step
        H_one_step = np.exp((-alpha_np / 2) * h_dbp - 1j * (beta_omega_sub * h_dbp))
        H_real_np = H_one_step.real.astype(np.float32)
        H_imag_np = H_one_step.imag.astype(np.float32)

        self.Nspans = Nspans
        self.steps_per_span = steps_per_span
        self.sqrtG_val = float(np.sqrt(G_lin))

        # Build alternating Linear -> Nonlinear layers
        layers = []
        for _ in range(Nspans):
            for _ in range(steps_per_span):
                layers.append(LDBP_LinearLayer(Nsub, H_real_np.copy(), H_imag_np.copy(),
                                                trainable_beta2))
                layers.append(LDBP_NonlinearLayer(gamma, h_dbp, trainable_gamma))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Args:
            x: (Nsub, 2) complex tensor
        Returns:
            (Nsub, 2) complex tensor
        """
        layer_idx = 0
        for _ in range(self.Nspans):
            x = x / self.sqrtG_val  # Remove EDFA gain at span boundary
            for _ in range(self.steps_per_span):
                x = self.layers[layer_idx](x)       # Linear
                x = self.layers[layer_idx + 1](x)   # Nonlinear
                layer_idx += 2
        return x

    def get_linear_weight_norms(self):
        """Returns the norm of each linear layer's H for monitoring."""
        norms = []
        for layer in self.layers:
            if isinstance(layer, LDBP_LinearLayer):
                H = torch.complex(layer.H_real, layer.H_imag)
                norms.append(torch.mean(torch.abs(H)).item())
        return norms
