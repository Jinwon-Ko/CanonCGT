import time
import torch


def check_runtime(model):
    resolutions = {'FHD': (1080, 1920),
                   '2K': (1440, 2560),
                   '4K': (2160, 3840),
                   '8K': (4320, 7680)}

    model.cuda()
    model.eval()

    with torch.no_grad():
        torch.cuda.empty_cache()

        dummy = torch.rand(3, 3, 480, 640).cuda()
        for _ in range(20):
            _ = model(dummy, dummy)

        for resol, size in resolutions.items():
            n = 0
            spend_time = 0
            img = torch.rand(10, 3, size[0], size[1]).cuda()

            for i in range(10):
                print(f'Processing {resol} resolution... [{i:02d}/{10}]', end='\r')

                torch.cuda.synchronize()
                t0 = time.time()

                for _ in range(10):
                    _ = model(img, img)
                    n += len(img)

                torch.cuda.synchronize()
                spend_time += (time.time() - t0)

            runtime = spend_time / n
            print(f'[{resol} resolution] Runtime in millisecond: {runtime * 1000:.3f} ms')

    print('# of trainable parameters : %.3f K' % (sum(p.numel() for p in model.parameters() if p.requires_grad) * 0.001))


def check_complexities(model):
    resolutions = {'FHD': (1080, 1920),
                   '2K': (1440, 2560),
                   '4K': (2160, 3840),
                   '8K': (4320, 7680)}

    model.cuda()
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()

        for resol, size in resolutions.items():
            gpu_mem = measure_gpu_peak_memory(model, input_shape=(1, 3, size[0], size[1]), device="cuda")
            print(f"[GPU Memory] peak_allocated={gpu_mem['peak_allocated_GB']:.3f} GB, "
                  f"peak_reserved={gpu_mem['peak_reserved_GB']:.3f} GB")


def measure_gpu_peak_memory(model, input_shape=(1,3,256,256), device="cuda"):
    """
    return MB (allocated and reserved)
    """
    assert torch.cuda.is_available(), "CUDA not available"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)

    with torch.inference_mode():
        x = torch.zeros(*input_shape, device=device)
        _ = model(x, x)  # warmup
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device=device)
    with torch.inference_mode():
        x = torch.zeros(*input_shape, device=device)
        _ = model(x, x)
        torch.cuda.synchronize()

    allocated = torch.cuda.max_memory_allocated(device=device) / (1024**3)
    reserved  = torch.cuda.max_memory_reserved(device=device) / (1024**3)
    return {"peak_allocated_GB": allocated, "peak_reserved_GB": reserved}

