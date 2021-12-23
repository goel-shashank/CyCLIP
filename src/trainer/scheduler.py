import numpy as np    

def cosine_scheduler(optimizer, base_lr, warmup_steps, steps):
    def _scheduler(step):
        if(step < warmup_steps):
            lr = base_lr * (step + 1) / warmup_steps
        else:
            n = step - warmup_steps
            d = steps - warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return _scheduler