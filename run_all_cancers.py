# -*- coding: utf-8 -*-
"""按论文设置依次训练 10 个 TCGA 癌种。

等价于依次执行：
    python main.py --data_folder <CANCER>

训练产物写入 result/（会覆盖已有结果，原始结果已备份在
result_backup_original/）。日志同时写入 outputs/logs/<CANCER>.log。

用法
----
    python run_all_cancers.py                 # 全部 10 个癌种
    python run_all_cancers.py KIRC BRCA       # 只跑指定癌种
"""
import io
import os
import sys
import subprocess
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from paths import PROJECT_ROOT, OUTPUT_DIR, ensure_dir

CANCERS = ['STAD', 'LUAD', 'UCEC', 'BLCA', 'BRCA',
           'SKCM', 'KIRC', 'GBM', 'PAAD', 'UVM']

# 与论文一致的训练轮数
PRE_EPOCH = 300
NUM_EPOCH = 1200


def run(cancer, logdir):
    logfile = os.path.join(logdir, cancer + '.log')
    cmd = [sys.executable, 'main.py',
           '--data_folder', cancer,
           '--pre_epoch', str(PRE_EPOCH),
           '--num_epoch', str(NUM_EPOCH)]
    t0 = time.time()
    print('[%s] 开始训练 ... (pre=%d, finetune=%d)' % (cancer, PRE_EPOCH, NUM_EPOCH))
    with open(logfile, 'w', encoding='utf-8') as lf:
        lf.write(' '.join(cmd) + '\n\n')
        lf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=lf,
                              stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = (proc.returncode == 0)
    print('[%s] %s  用时 %.1f 分钟  日志 -> %s'
          % (cancer, '完成' if ok else '失败(returncode=%d)' % proc.returncode,
             dt / 60.0, logfile))
    return ok


def main():
    logdir = ensure_dir(os.path.join(OUTPUT_DIR, 'logs'))
    targets = sys.argv[1:] or CANCERS

    print('=' * 74)
    print('全量训练 Subtype-MVCL：%d 个癌种' % len(targets))
    print('每组：预训练 %d epoch + 微调 %d epoch' % (PRE_EPOCH, NUM_EPOCH))
    print('=' * 74)

    t0 = time.time()
    results = {}
    for c in targets:
        try:
            results[c] = run(c, logdir)
        except Exception as e:
            print('[%s] 异常: %s' % (c, e))
            results[c] = False

    print()
    print('=' * 74)
    print('全部结束，总用时 %.1f 分钟' % ((time.time() - t0) / 60.0))
    for c in targets:
        print('  %-6s %s' % (c, 'OK' if results.get(c) else 'FAILED'))
    print('=' * 74)
    print('提示：训练完成后可运行 `python export_figures.py` 用新结果重新出图。')


if __name__ == '__main__':
    main()
