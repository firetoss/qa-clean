from __future__ import annotations

import argparse
import subprocess
import sys


def main(cfg_path: str = 'src/configs/config.yaml', input_file: str = None) -> None:
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
        if input_file:
            cmd.extend(['--input', input_file])
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"Stage failed: {m}")
    print("[run_all] all stages completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行QA清洗流水线')
    parser.add_argument('--config', '-c', default='src/configs/config.yaml', 
                       help='配置文件路径 (默认: src/configs/config.yaml)')
    parser.add_argument('--input', '-i', required=True,
                       help='输入数据文件路径 (支持格式: .parquet, .xlsx, .xls, .csv)')
    
    args = parser.parse_args()
    main(args.config, args.input)
