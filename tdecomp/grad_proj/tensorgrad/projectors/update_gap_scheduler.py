import math

UPDATE_MODES = {"fixed", "linear", "cosine", "step", "exponential", "exponential2"}


class UpdateGapScheduler:
    """Zero-based projection updates. Fixed gap k updates at 0, k, 2k."""
    def __init__(self, start, end=None, mode="fixed", batch_size=1, epochs=1,
                 training_samples=1, verbose=False, total_iters=None):
        end = start if end is None else end
        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if mode not in UPDATE_MODES:
            raise ValueError(f"Unknown update mode={mode!r}")
        for name, value in (("batch_size", batch_size), ("epochs", epochs), ("training_samples", training_samples)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if total_iters is None:
            total_iters = math.ceil((training_samples or 1) / (batch_size or 1)) * (epochs or 1)
        if isinstance(total_iters, bool) or not isinstance(total_iters, int) or total_iters <= 0:
            raise ValueError("total_iters must be a positive integer")
        self.update_gap, self.update_gap_end, self.mode = start, end, mode
        self.batch_size, self.epochs, self.training_samples = batch_size, epochs, training_samples
        self.iter_per_epoch = math.ceil((training_samples or 1) / (batch_size or 1))
        self.total_iters, self.verbose = total_iters, verbose
        self.next_update, self.last_iter = 0, -1

    def compute_gap(self, current_iter):
        progress = min(1.0, max(0.0, current_iter / self.total_iters))
        start, end = self.update_gap, self.update_gap_end
        if self.mode == "fixed":
            value = start
        elif self.mode == "linear":
            value = start + (end - start) * progress
        elif self.mode == "cosine":
            value = start + (end - start) * (1 - math.cos(math.pi * progress)) / 2
        elif self.mode == "step":
            value = end if progress >= 0.5 else start
        else:
            value = start * (end / start) ** (progress**2 if self.mode == "exponential2" else progress)
        return max(1, int(value))

    def should_update(self, current_iter):
        if isinstance(current_iter, bool) or not isinstance(current_iter, int) or current_iter < self.last_iter or current_iter < 0:
            raise ValueError("iteration must be a nonnegative, nondecreasing integer")
        self.last_iter = current_iter
        if current_iter < self.next_update:
            return False
        self.next_update = current_iter + self.compute_gap(current_iter)
        return True

    step = should_update

    def state_dict(self):
        return {"version": 1, "start": self.update_gap, "end": self.update_gap_end, "mode": self.mode,
                "batch_size": self.batch_size, "epochs": self.epochs, "training_samples": self.training_samples,
                "total_iters": self.total_iters, "next_update": self.next_update, "last_iter": self.last_iter}

    def load_state_dict(self, state):
        if state.get("version") != 1 or not {"start", "end", "mode", "total_iters", "next_update", "last_iter"} <= state.keys():
            raise ValueError("unsupported or incomplete update schedule state")
        restored = type(self)(state["start"], state["end"], state["mode"], state.get("batch_size"),
                              state.get("epochs"), state.get("training_samples"), total_iters=state["total_iters"])
        if not isinstance(state["next_update"], int) or not isinstance(state["last_iter"], int) or state["last_iter"] < -1 or state["next_update"] <= state["last_iter"]:
            raise ValueError("invalid update schedule position")
        restored.next_update, restored.last_iter = state["next_update"], state["last_iter"]
        self.__dict__.update(restored.__dict__)

    def simulate_update_schedule(self):
        simulation = type(self)(self.update_gap, self.update_gap_end, self.mode, total_iters=self.total_iters)
        return [(i, simulation.compute_gap(i)) for i in range(self.total_iters) if simulation.should_update(i)]

    def plot_update_schedule(self, save_path=None):
        import matplotlib.pyplot as plt
        schedule = self.simulate_update_schedule()
        figure, axis = plt.subplots()
        axis.plot([x[0] for x in schedule], [x[1] for x in schedule], ".-")
        axis.set(xlabel="Iteration", ylabel="Update interval", title=self.mode)
        if save_path:
            figure.savefig(save_path)
            plt.close(figure)
        return figure
