# mass_spring_miniproject
# Damped Mass-Spring Mini-Project
ASTE 404 - Numerical Methods  
Author: Yaffet Bedru

## Overview
This mini-project simulates a damped mass-spring system using three numerical time-integration methods:

- Forward Euler  
- Semi-Implicit (Symplectic) Euler  
- Runge-Kutta 4 (RK4)

The system has a closed-form analytic solution, which allows direct verification of numerical accuracy, stability, and convergence.

## Repository Structure
src/mass_spring.py      - main simulation code  
results/                - generated plots  
report.pdf              - final project report  
README.md               - project documentation  

## Requirements
Python 3.8+  
NumPy  
Matplotlib  

This will generate the following figures inside the results/ folder:
- trajectory_compare.png  
- energy_vs_time.png  
- error_vs_dt.png  

## Numerical Methods Summary
- Forward Euler: first-order accuracy, unstable for larger time steps  
- Semi-Implicit Euler: first-order accuracy, more stable for oscillatory systems  
- RK4: fourth-order accuracy, most stable and accurate  

Convergence results:
- Euler: ~1st order  
- Semi-Implicit Euler: ~1st order  
- RK4: ~4th order  

Energy plots show the expected exponential decay in a damped oscillator.

## ASTE 404 Project Requirements
This project includes:
- A numerical method implementation  
- Analytic-solution-based verification  
- Convergence and stability analysis  
- Diagnostic plotting  
- A complete written report with progress log and LLM development log  



