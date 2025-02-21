from dls_automator import DLS

# creating graphs and tau values
dls = DLS(dir='DLS_files/0.75micro_100dil')
dls.AutoCorrelation()

# calculating q values and errors
q_vals, error_q_vals = dls.q_and_error()
print(f"Q values: {q_vals}")
print(f"Error in Q values: {error_q_vals}")
