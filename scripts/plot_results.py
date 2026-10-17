#!/usr/bin/env python3
"""Plot estimated vs ground-truth trajectories and compute error metrics.

Usage:
    python3 scripts/plot_results.py [results/trajectory_log.csv]

Outputs (in the same folder as the CSV):
    trajectory_plot.png / .pdf   estimated vs ground-truth trajectories
    error_plot.png / .pdf        position and heading error over time
    metrics.txt                  RMSE, mean, max and final errors
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

LANDMARKS = np.array([(-2.0, -2.0), (3.0, -2.5), (8.0, -2.0), (8.0, 2.0),
                      (8.0, 6.0), (3.0, 6.5), (-2.0, 6.0), (-2.0, 2.0)])

# Colour-blind-safe colours, plus different line styles so the
# trajectories are distinguishable without colour.
C_GT, C_DR, C_PF = '#222222', '#D55E00', '#0072B2'


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/trajectory_log.csv'
    out_dir = os.path.dirname(os.path.abspath(path))
    d = np.genfromtxt(path, delimiter=',', names=True)
    t = d['t']

    err_dr = np.hypot(d['dr_x'] - d['gt_x'], d['dr_y'] - d['gt_y'])
    err_pf = np.hypot(d['pf_x'] - d['gt_x'], d['pf_y'] - d['gt_y'])
    herr_dr = np.degrees(np.abs(wrap(d['dr_th'] - d['gt_th'])))
    herr_pf = np.degrees(np.abs(wrap(d['pf_th'] - d['gt_th'])))

    def stats(e):
        return (np.sqrt(np.mean(e ** 2)), np.mean(e), np.max(e), e[-1])

    lines = [
        f'Samples: {len(t)}  Duration: {t[-1]:.1f} s',
        f'Distance travelled (ground truth): '
        f'{np.sum(np.hypot(np.diff(d["gt_x"]), np.diff(d["gt_y"]))):.2f} m',
        '',
        f'{"Position error (m)":28s}{"RMSE":>8s}{"Mean":>8s}'
        f'{"Max":>8s}{"Final":>8s}',
    ]
    for name, e in (('Dead reckoning (odometry)', err_dr),
                    ('Particle filter', err_pf)):
        lines.append(f'{name:28s}' + ''.join(f'{v:8.3f}' for v in stats(e)))
    lines += ['', f'{"Heading error (deg)":28s}{"RMSE":>8s}{"Mean":>8s}'
              f'{"Max":>8s}{"Final":>8s}']
    for name, e in (('Dead reckoning (odometry)', herr_dr),
                    ('Particle filter', herr_pf)):
        lines.append(f'{name:28s}' + ''.join(f'{v:8.2f}' for v in stats(e)))
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(report + '\n')

    # ---- Figure 1: trajectories ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(d['gt_x'], d['gt_y'], color=C_GT, lw=2.0, ls='-',
            label='Ground truth (simulator)')
    ax.plot(d['dr_x'], d['dr_y'], color=C_DR, lw=1.8, ls='--',
            label='Dead reckoning (noisy odometry only)')
    ax.plot(d['pf_x'], d['pf_y'], color=C_PF, lw=1.8, ls='-.',
            label='Particle filter estimate')
    ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], marker='^', s=90,
               color='#CC3311', edgecolor='white', zorder=5,
               label='Landmarks')
    ax.plot(d['gt_x'][0], d['gt_y'][0], 'o', color=C_GT, ms=9, zorder=6,
            label='Start')
    ax.annotate(f'Final error: DR {err_dr[-1]:.2f} m, '
                f'PF {err_pf[-1]:.2f} m',
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=9,
                color='#444444')
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    ax.set_title('Estimated vs ground-truth trajectory')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, color='#DDDDDD', lw=0.6)
    ax.legend(loc='center', fontsize=9, framealpha=0.95)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'trajectory_plot.{ext}'), dpi=200)
    plt.close(fig)

    # ---- Figure 2: error over time --------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    a1.plot(t, err_dr, color=C_DR, lw=1.8, ls='--', label='Dead reckoning')
    a1.plot(t, err_pf, color=C_PF, lw=1.8, ls='-', label='Particle filter')
    a1.set_ylabel('Position error (m)')
    a1.set_title('Localization error vs ground truth over time')
    a1.grid(True, color='#DDDDDD', lw=0.6)
    a1.legend(fontsize=9)
    a2.plot(t, herr_dr, color=C_DR, lw=1.8, ls='--', label='Dead reckoning')
    a2.plot(t, herr_pf, color=C_PF, lw=1.8, ls='-', label='Particle filter')
    a2.set_xlabel('Time (s)')
    a2.set_ylabel('Heading error (deg)')
    a2.grid(True, color='#DDDDDD', lw=0.6)
    a2.legend(fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'error_plot.{ext}'), dpi=200)
    plt.close(fig)
    print(f'\nSaved plots and metrics to {out_dir}')


if __name__ == '__main__':
    main()
