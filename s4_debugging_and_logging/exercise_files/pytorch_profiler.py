import torch
from torch.profiler.profiler import tensorboard_trace_handler
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("02476-ml-ops\s4_debugging_and_logging")) as prof:
    with record_function("model_inference"):
        model(inputs)

#print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
#print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
prof.export_chrome_trace("trace.json")