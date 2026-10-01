"""Clean-room replication of Context Language Models (CLMs).

A CLM manages its own context: the live context is mirrored to a file the
model can edit with ordinary bash, and every edit is synced back into the
message list sent on the next turn (c_{t+1} = f_theta(c_t), Eq. 2).
"""

__version__ = "0.1.0"
