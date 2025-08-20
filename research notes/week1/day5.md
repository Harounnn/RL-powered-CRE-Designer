## Focus
We explored **how Position Weight Matrices (PWMs) score motifs** and connected them to our **RL reward function**.

## Key Findings
- A PWM captures motif preferences by assigning probabilities for each base at each position.  
- A **perfect match** (e.g., TATA) achieves a normalized score of **1.0**, while mismatches reduce the score significantly (e.g., ~0.14 for TACA).  
- By normalizing the PWM score, we make it comparable to other terms in the reward.  
- PWM scoring provides a concrete, biology-grounded **observation signal** for the agent.  
- Motif presence is no longer just abstract biology; it directly influences the agent’s reward.