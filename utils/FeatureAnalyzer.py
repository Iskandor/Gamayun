import numpy as np
import os
import csv
import datetime
from statsmodels.tsa.stattools import adfuller

class FeatureAnalyzer:
    def __init__(self, buffer_size=300, save_path=None):
        self.buffer_size = buffer_size
        self.save_path = save_path
        
        self.mean_buffer = [] 
        self.std_buffer = []
        
        # Flags to ensure we only run ONCE
        self.is_finished = False 
        
        # Initialize CSV with header if path is provided
        if self.save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Since we use unique files per trial, we can safely write the header
            # mode='w' creates a new file (or overwrites if restarting trial 0)
            with open(self.save_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'timestamp',
                    'avg_internal_std',  # Should be ~1.0 with LayerNorm
                    'mean_stability',    # Should be low (no drift)
                    'global_mean',       # Should be ~0.0
                    'adf_stat',
                    'adf_p_value',
                    'is_stationary'
                ]
                writer.writerow(headers)

    def add(self, z_batch):
        # 1. STOP if already finished
        if self.is_finished:
            return
        
        if np.random.rand() > 0.01: 
            return

        # 2. Collect Data
        # Take the first element of the batch, detach from graph, move to CPU
        z_vec = z_batch[0].detach().cpu().numpy()
        
        mu = np.mean(z_vec)
        sigma = np.std(z_vec)
        
        self.mean_buffer.append(mu)
        self.std_buffer.append(sigma)

        # 3. Check if full
        if len(self.mean_buffer) >= self.buffer_size:
            self.analyze_and_save()
            self.is_finished = True
            
            # Free memory
            self.mean_buffer = []
            self.std_buffer = []
            print(f"[FeatureAnalyzer]: Analysis complete and saved. Shutting down analyzer.")

    def analyze_and_save(self):
        means = np.array(self.mean_buffer)
        stds = np.array(self.std_buffer)
        
        # --- STATISTICS ---
        avg_internal_std = np.mean(stds)
        mean_stability = np.std(means)
        global_mean = np.mean(means)
        
        # --- ADF TEST ---
        adf_stat = float('nan')
        adf_p_value = float('nan')
        is_stationary = 0
        try:
            # Add micro-noise to avoid crash on perfect zeros
            noise = np.random.normal(0, 1e-6, size=means.shape)
            adf_result = adfuller(means + noise)
            
            adf_stat = adf_result[0]
            adf_p_value = adf_result[1]
            
            if adf_p_value < 0.05:
                is_stationary = 1
        except Exception:
            pass # Keep NaNs if test fails
    
        # --- SAVE TO CSV ---
        if self.save_path:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            row = [
                timestamp,
                f"{avg_internal_std:.6f}",
                f"{mean_stability:.6f}",
                f"{global_mean:.6f}",
                f"{adf_stat:.6f}",
                f"{adf_p_value:.6f}",
                is_stationary
            ]
            
            with open(self.save_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)