import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def evaluate_fitness_vectorized(x, y, z):
    """
    A NumPy-vectorized version of the objective function.
    """
    term1 = -np.exp(-(x - 0.55*z)**2) * np.cos(81*(x + z + 0.12*y))
    term2 = np.sin(44*(y + 0.03*(x - z)))
    term3 = -np.cos(69 * np.sin(x*z + 0.07*y))
    term4 = -np.sin(1.32*(x - z))
    term5 = -0.32*(z + 0.22*x*z)**2 * np.exp(np.sin(63*(z - x)))
    term6 = np.cos(77*x) + np.sin(71*z)
    term7 = -np.exp(-y**2) * np.cos(75*y)
    term8 = np.sin(43*y) + np.cos(73*y)
    term9 = 0.05*(x**2 + y**2 + z**2)

    return (
        term1 + term2 + term3 +
        term4 + term5 + term6 +
        term7 + term8 + term9
    )


def generate_separated_plots():
    # Define the slice
    z_fixed = 0.199002

    # Zoom in around a known maximum
    x = np.linspace(3.75, 4.25, 800)
    y = np.linspace(0.95, 1.45, 800)
    X, Y = np.meshgrid(x, y)

    fitness = evaluate_fitness_vectorized(X, Y, z_fixed)

    # --- PLOT 1: 3D Surface ---
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(
        X, Y, fitness,
        cmap=cm.coolwarm,  # type: ignore
        edgecolor='none',
        alpha=0.9
    )
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel(f"f(x, y, z={z_fixed:.3f})")
    fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.05)
    fig1.subplots_adjust(left=0, right=0.01, bottom=0, top=0.01)

    plt.tight_layout()
    plt.savefig("maximising_function/landscape_3d_surface.png", dpi=300)
    print("Saved: maximising_function/landscape_3d_surface.png")
    plt.close(fig1)

    # --- PLOT 2: 2D Contour ---
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    contour = ax2.contourf(
        X, Y, fitness,
        levels=25,
        cmap=cm.coolwarm  # type: ignore
    )
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    fig2.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.savefig("maximising_function/landscape_2d_contour.png", dpi=300)
    print("Saved: maximising_function/landscape_2d_contour.png")
    plt.close(fig2)


if __name__ == "__main__":
    generate_separated_plots()
