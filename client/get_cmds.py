def get_cmd(seed: int, task_name: str, run=1, use_veo=False):
    match task_name:
        case "drawer_open":
            return [
                "python", "ray_pipeline_client.py",
                "--first-frame", "./demo_data/rgb.png",
                "--config-file", "./demo_data/intrinsics.json",
                "--wan-prompt", "一只人手抓住黑色抽屉把手，顺利地将其从抽屉中拉出。抽屉应沿直线打开，且不会前后移动。人手不会在视觉上遮挡抽屉把手。",
                "--veo-prompt", "A human hand grasps the black drawer handle and smoothly pulls it out of the drawer. The drawer should open in a straight line respecting its articulation with no back and forth motion. The human hand does not visually obscure the drawer handle.",
                "--sam-prompt", "black drawer handle.",
                "--task_name", "drawer_open",
                "--run", str(run),
                "--output-dir", f"./demo_data/{seed}",
                "--seed", str(seed),
                "--prompt-extension",
                *(["--use-veo"] if use_veo else []),
            ]