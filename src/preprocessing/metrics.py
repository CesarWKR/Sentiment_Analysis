
"""
# This module defines shared metrics for text preprocessing across different modules.
# It includes counters for invalid and empty texts, which can be used to track the quality of the data being processed.
"""
class Metrics:
    def __init__(self):
        self.empty_text_count = 0
        self.invalid_text_count = 0

    def reset(self):
        self.empty_text_count = 0
        self.invalid_text_count = 0

# Universal metrics instance for shared use
metrics = Metrics()