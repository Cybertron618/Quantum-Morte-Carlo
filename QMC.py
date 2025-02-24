import numpy as np
from scipy.stats.qmc import Sobol, Halton, LatinHypercube
from scipy.integrate import nquad
import sympy as sp
from numba import cuda
import threading
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Check if CUDA is available
try:
    cuda_available = cuda.is_available()
except Exception as e:
    logging.warning(f"CUDA check failed: {e}")
    cuda_available = False

# Define CUDA Kernel
@cuda.jit
def gpu_integrand(d_samples, d_result):
    idx = cuda.grid(1)
    if idx < d_samples.shape[1]:
        d_result[idx] = 1
        for i in range(d_samples.shape[0]):
            d_result[idx] *= d_samples[i, idx]

class SupremeQMCIntegrator:
    def __init__(self, master):
        self.master = master
        master.title("Supreme QMC Integrator")

        # Input frame
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(pady=10)

        # Function input
        tk.Label(self.input_frame, text="Function (x1, x2, ...):").grid(row=0, column=0, padx=5, sticky="w")
        self.func_entry = tk.Entry(self.input_frame, width=50)
        self.func_entry.grid(row=0, column=1, padx=5, pady=5)

        # Limits input
        tk.Label(self.input_frame, text="Limits (min1, max1, min2, max2, ...):").grid(row=1, column=0, padx=5, sticky="w")
        self.limits_entry = tk.Entry(self.input_frame, width=50)
        self.limits_entry.grid(row=1, column=1, padx=5, pady=5)

        # Samples input
        tk.Label(self.input_frame, text="Number of samples:").grid(row=2, column=0, padx=5, sticky="w")
        self.samples_entry = tk.Entry(self.input_frame, width=15)
        self.samples_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Dimension input
        tk.Label(self.input_frame, text="Number of dimensions:").grid(row=3, column=0, padx=5, sticky="w")
        self.dim_entry = tk.Entry(self.input_frame, width=15)
        self.dim_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Method selection
        tk.Label(self.input_frame, text="Integration method:").grid(row=4, column=0, padx=5, sticky="w")
        self.method_var = tk.StringVar(value="Sobol")
        self.method_menu = ttk.Combobox(self.input_frame, textvariable=self.method_var, values=["Sobol", "Halton", "Latin Hypercube", "Adaptive (SciPy)"])
        self.method_menu.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # GPU acceleration checkbox
        self.gpu_var = tk.BooleanVar(value=False)
        self.gpu_check = tk.Checkbutton(self.input_frame, text="Enable GPU Acceleration", variable=self.gpu_var, state=tk.NORMAL if cuda_available else tk.DISABLED)
        self.gpu_check.grid(row=5, column=0, columnspan=2, pady=5)

        # Buttons
        tk.Button(self.input_frame, text="Integrate", command=self.start_integration).grid(row=6, column=0, columnspan=2, pady=10)
        tk.Button(self.input_frame, text="Export Results", command=self.export_results).grid(row=7, column=0, columnspan=2, pady=5)

        # Output field
        tk.Label(self.input_frame, text="Result:").grid(row=8, column=0, padx=5, sticky="w")
        self.output_entry = tk.Entry(self.input_frame, width=50)
        self.output_entry.grid(row=8, column=1, padx=5, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(master, length=300, mode="determinate")
        self.progress.pack(pady=10)

        # Figure for visualization (3D only)
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
        self.ax.set_title("QMC Sample Points")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def start_integration(self):
        threading.Thread(target=self.integrate, daemon=True).start()

    def integrate(self):
        try:
            func_str = self.func_entry.get()
            limits_str = self.limits_entry.get()
            samples_str = self.samples_entry.get()
            dim_str = self.dim_entry.get()
            method = self.method_var.get()
            use_gpu = self.gpu_var.get() and cuda_available

            if not dim_str.isdigit():
                raise ValueError("Number of dimensions must be a positive integer.")
            dim = int(dim_str)

            func = self.parse_function(func_str, dim)
            limits = self.validate_limits(limits_str, dim)
            samples = self.validate_samples(samples_str)

            if method == "Adaptive (SciPy)":
                result, error = self.adaptive_integrate(func, limits)
            else:
                result, error = self.qmc_integrate(func, limits, samples, method, dim, use_gpu)

            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(tk.END, f"{result:.6f} ± {error:.6f}")

        except Exception as e:
            logging.error(f"Integration failed: {e}")
            messagebox.showerror("Error", str(e))

    def parse_function(self, func_str, dim):
        try:
            vars = sp.symbols(f"x1:{dim + 1}")
            return sp.lambdify(vars, sp.sympify(func_str), "numpy")
        except Exception as e:
            raise ValueError(f"Invalid function: {e}")

    def validate_limits(self, limits_str, dim):
        limits = [float(x) for x in limits_str.split(",")]
        if len(limits) != 2 * dim:
            raise ValueError(f"Expected {2 * dim} values for limits.")
        return limits

    def validate_samples(self, samples_str):
        samples = int(samples_str)
        if samples <= 0:
            raise ValueError("Number of samples must be positive.")
        return samples

    def qmc_integrate(self, func, limits, samples, method, dim, use_gpu):
        self.master.after(0, self.update_progress, 0)  # Update progress on the main thread

        sampler = {"Sobol": Sobol, "Halton": Halton, "Latin Hypercube": LatinHypercube}[method](d=dim, scramble=True)
        qmc_samples = sampler.random(n=samples)
        self.scaled_samples = [limits[2 * i] + (limits[2 * i + 1] - limits[2 * i]) * qmc_samples[:, i] for i in range(dim)]

        if use_gpu:
            try:
                d_samples = cuda.to_device(np.array(self.scaled_samples, dtype=np.float32))
                d_result = cuda.device_array(samples, dtype=np.float32)
                gpu_integrand[(samples + 255) // 256, 256](d_samples, d_result)
                self.result = d_result.copy_to_host()
            except Exception as e:
                logging.warning(f"GPU failed: {e}, using CPU instead.")
                self.result = np.array([func(*sample) for sample in zip(*self.scaled_samples)])
        else:
            self.result = np.array([func(*sample) for sample in zip(*self.scaled_samples)])

        self.master.after(0, self.update_progress, 100)  # Update progress on the main thread

        volume = np.prod([limits[2 * i + 1] - limits[2 * i] for i in range(dim)])
        return np.mean(self.result) * volume, np.std(self.result) * volume / np.sqrt(samples)

    def adaptive_integrate(self, func, limits):
        result, error = nquad(func, [(limits[2 * i], limits[2 * i + 1]) for i in range(len(limits) // 2)])
        return result, error

    def export_results(self):
        if not hasattr(self, "scaled_samples") or not hasattr(self, "result"):
            messagebox.showerror("Error", "No results to export. Run the integration first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Sample Points", "Function Value"])
                writer.writerows(zip(*self.scaled_samples, self.result))

    def update_progress(self, value):
        self.progress["value"] = value
        self.master.update_idletasks()

root = tk.Tk()
app = SupremeQMCIntegrator(root)
root.mainloop()