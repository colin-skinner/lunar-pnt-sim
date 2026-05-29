import numpy as np

import sys
sys.path.append("../..")

from lunanav.constants import R_MOON, GM_MOON
def debug_trajectory(s_bar, u_bar, s_goal, iteration, N, dt, mass):
    """Print detailed trajectory diagnostics"""
    
    print("\n" + "="*80)
    print(f"TRAJECTORY DEBUG - Iteration {iteration}")
    print("="*80)
    
    # Initial and final states
    print(f"\nInitial state:")
    print(f"  pos: [{s_bar[0,0]:8.1f}, {s_bar[0,1]:8.1f}, {s_bar[0,2]:8.1f}] m")
    print(f"  vel: [{s_bar[0,3]:8.2f}, {s_bar[0,4]:8.2f}, {s_bar[0,5]:8.2f}] m/s")
    print(f"  alt: {s_bar[0,2] - R_MOON:8.1f} m")
    
    print(f"\nFinal state:")
    print(f"  pos: [{s_bar[-1,0]:8.1f}, {s_bar[-1,1]:8.1f}, {s_bar[-1,2]:8.1f}] m")
    print(f"  vel: [{s_bar[-1,3]:8.2f}, {s_bar[-1,4]:8.2f}, {s_bar[-1,5]:8.2f}] m/s")
    print(f"  alt: {s_bar[-1,2] - R_MOON:8.1f} m")
    print(f"  quat: [{s_bar[-1,6]:6.3f}, {s_bar[-1,7]:6.3f}, {s_bar[-1,8]:6.3f}, {s_bar[-1,9]:6.3f}]")
    print(f"  omega: [{s_bar[-1,10]:6.3f}, {s_bar[-1,11]:6.3f}, {s_bar[-1,12]:6.3f}] rad/s")
    
    print(f"\nGoal state:")
    print(f"  pos: [{s_goal[0]:8.1f}, {s_goal[1]:8.1f}, {s_goal[2]:8.1f}] m")
    print(f"  vel: [{s_goal[3]:8.2f}, {s_goal[4]:8.2f}, {s_goal[5]:8.2f}] m/s")
    print(f"  alt: {s_goal[2] - R_MOON:8.1f} m")
    
    # Errors
    pos_err = s_bar[-1, 0:3] - s_goal[0:3]
    vel_err = s_bar[-1, 3:6] - s_goal[3:6]
    alt_err = s_bar[-1, 2] - s_goal[2]
    
    print(f"\nErrors:")
    print(f"  position: [{pos_err[0]:8.1f}, {pos_err[1]:8.1f}, {pos_err[2]:8.1f}] m")
    print(f"  |pos_err|: {np.linalg.norm(pos_err):8.1f} m")
    print(f"  altitude: {alt_err:8.1f} m")
    print(f"  velocity: [{vel_err[0]:8.2f}, {vel_err[1]:8.2f}, {vel_err[2]:8.2f}] m/s")
    print(f"  |vel_err|: {np.linalg.norm(vel_err):8.2f} m/s")
    
    # Control statistics
    print(f"\nControl statistics (over {N} steps, {N*dt:.1f}s):")
    print(f"  Fx: min={np.min(u_bar[:,0]):8.1f}, max={np.max(u_bar[:,0]):8.1f}, mean={np.mean(u_bar[:,0]):8.1f} N")
    print(f"  Fy: min={np.min(u_bar[:,1]):8.1f}, max={np.max(u_bar[:,1]):8.1f}, mean={np.mean(u_bar[:,1]):8.1f} N")
    print(f"  Fz: min={np.min(u_bar[:,2]):8.1f}, max={np.max(u_bar[:,2]):8.1f}, mean={np.mean(u_bar[:,2]):8.1f} N")
    print(f"  |F|: min={np.min(np.linalg.norm(u_bar[:,0:3], axis=1)):8.1f}, " + 
          f"max={np.max(np.linalg.norm(u_bar[:,0:3], axis=1)):8.1f}, " +
          f"mean={np.mean(np.linalg.norm(u_bar[:,0:3], axis=1)):8.1f} N")
    print(f"  τx: min={np.min(u_bar[:,3]):8.2f}, max={np.max(u_bar[:,3]):8.2f}, mean={np.mean(u_bar[:,3]):8.2f} N·m")
    print(f"  τy: min={np.min(u_bar[:,4]):8.2f}, max={np.max(u_bar[:,4]):8.2f}, mean={np.mean(u_bar[:,4]):8.2f} N·m")
    print(f"  τz: min={np.min(u_bar[:,5]):8.2f}, max={np.max(u_bar[:,5]):8.2f}, mean={np.mean(u_bar[:,5]):8.2f} N·m")
    
    # Trajectory characteristics
    altitudes = s_bar[:, 2] - R_MOON
    velocities_z = s_bar[:, 5]
    
    print(f"\nTrajectory characteristics:")
    print(f"  Altitude: min={np.min(altitudes):8.1f}, max={np.max(altitudes):8.1f} m")
    print(f"  Vertical vel: min={np.min(velocities_z):8.2f}, max={np.max(velocities_z):8.2f} m/s")
    print(f"  Descent rate at end: {velocities_z[-1]:8.2f} m/s")
    
    # Check for anomalies
    print(f"\nAnomaly checks:")
    if np.any(np.isnan(s_bar)):
        print("  ⚠️  NaN detected in states!")
        nan_steps = np.where(np.any(np.isnan(s_bar), axis=1))[0]
        print(f"     First NaN at step {nan_steps[0]}")
    if np.any(np.isnan(u_bar)):
        print("  ⚠️  NaN detected in controls!")
        nan_steps = np.where(np.any(np.isnan(u_bar), axis=1))[0]
        print(f"     First NaN at step {nan_steps[0]}")
    if np.any(np.abs(s_bar) > 1e10):
        print("  ⚠️  Very large state values detected!")
    if np.any(np.abs(u_bar) > 1e6):
        print("  ⚠️  Very large control values detected!")
    if np.min(altitudes) < -100:
        print(f"  ⚠️  Trajectory goes underground! Min altitude: {np.min(altitudes):.1f} m")
    if np.max(np.abs(velocities_z)) > 2000:
        print(f"  ⚠️  Very high velocities detected! Max |vz|: {np.max(np.abs(velocities_z)):.1f} m/s")
    
    # Energy analysis
    kinetic = 0.5 * mass * np.sum(s_bar[:, 3:6]**2, axis=1)
    potential = -GM_MOON * mass / np.linalg.norm(s_bar[:, 0:3], axis=1)
    total_energy = kinetic + potential
    
    print(f"\nEnergy analysis:")
    print(f"  Initial total energy: {total_energy[0]:.2e} J")
    print(f"  Final total energy: {total_energy[-1]:.2e} J")
    print(f"  Energy change: {total_energy[-1] - total_energy[0]:.2e} J")
    
    print("="*80 + "\n")