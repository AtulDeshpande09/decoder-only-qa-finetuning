import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, exp_name, out_dir="experiments"):
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"{exp_name}_{timestamp}.txt")

        with open(self.path, "w") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 60 + "\n\n")

    def log(self, text):
        with open(self.path, "a") as f:
            f.write(text + "\n")

    def section(self, title):
        self.log("\n" + "=" * 60)
        self.log(title)
        self.log("=" * 60 + "\n")


