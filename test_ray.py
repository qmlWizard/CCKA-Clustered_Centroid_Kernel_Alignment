import os

# ✅ Disable GPU detection (since Intel Arc isn't supported)
os.environ["RAY_DISABLE_GPU_AUTODETECT"] = "1"
os.environ["RAY_DASHBOARD_DISABLE"] = "1"
os.environ["RAY_LOG_TO_STDERR"] = "1"
os.environ["RAY_NODE_IP_ADDRESS"] = "127.0.0.1"

import ray

ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

@ray.remote
def hello():
    return "Ray is working!"

print(ray.get(hello.remote()))
