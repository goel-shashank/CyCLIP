import numpy as np    

def cosine_scheduler(optimizer, base_lr, warmup_steps, total_steps):
    def _scheduler(current_step):
        if(current_step < warmup_steps):
            lr = base_lr * (current_step + 1) / warmup_steps
        else:
            n = current_step - warmup_steps
            d = total_steps - warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
    return _scheduler