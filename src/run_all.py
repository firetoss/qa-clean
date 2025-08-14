from __future__ import annotations

import subprocess
import sys


def main(cfg_path: str = 'src/configs/config.yaml') -> None:
    modules = [
        'src.stages.stage1_filter',
        'src.stages.stage2_recall',
        'src.stages.stage3_rerank',
        'src.stages.stage4_cluster',
        'src.stages.stage5_answer_govern',
    ]
    for m in modules:
        print(f"[run_all] running: {m}")
        cmd = [sys.executable, '-m', m, '--config', cfg_path]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"Stage failed: {m}")
    print("[run_all] all stages completed")


if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'src/configs/config.yaml'
    main(cfg)
