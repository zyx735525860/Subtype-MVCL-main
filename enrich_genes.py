# -*- coding: utf-8 -*-
"""对基因列表做 GO / Reactome / 通路富集（Enrichr API），并构建条目间网络。

对应何军论文图 3.11 的 (A) 与 (B) 两个面板。
"""
import io
import json
import sys
import time
import urllib.parse
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'https://maayanlab.cloud/Enrichr'
UA = {'User-Agent': 'Mozilla/5.0 (research; lit-review)'}

LIBRARIES = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'GO_Cellular_Component_2023',
    'Reactome_2022',
    'KEGG_2021_Human',
    'MSigDB_Hallmark_2020',
]


def add_list(genes, desc):
    data = urllib.parse.urlencode({
        'list': '\n'.join(genes), 'description': desc}).encode()
    req = urllib.request.Request(BASE + '/addList', data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())['userListId']


def enrich(uid, lib):
    url = '%s/enrich?userListId=%d&backgroundType=%s' % (BASE, uid, lib)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        return json.loads(r.read().decode()).get(lib, [])


def run(genes, out_prefix):
    genes = [g.split('|')[0] for g in genes]
    uid = add_list(genes, 'Subtype-MVCL attribution')
    print('已提交 %d 个基因, userListId=%d' % (len(genes), uid))

    hits = []
    for lib in LIBRARIES:
        try:
            for row in enrich(uid, lib):
                # 行格式: [rank, term, p, z, combined, genes, adj_p, old_p, old_adj]
                hits.append({
                    'library': lib, 'term': row[1], 'p_value': row[2],
                    'combined_score': row[4], 'overlap': row[5] if len(row) > 5 else '',
                    'n_overlap': len(row[5].split(',')) if len(row) > 5 else 0,
                    'adj_p': row[6] if len(row) > 6 else None,
                })
        except Exception as e:
            print('  %-32s 失败: %s' % (lib, e))
        time.sleep(0.4)

    hits.sort(key=lambda h: h['p_value'])
    json.dump(hits, open(out_prefix + '_enrichment.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n%-44s %-34s %10s %6s' % ('Term', 'Library', 'P', 'n'))
    print('-' * 100)
    for h in hits[:20]:
        print('%-44s %-34s %10.2e %6d'
              % (h['term'][:44], h['library'][:34], h['p_value'], h['n_overlap']))
    return hits


if __name__ == '__main__':
    import pandas as pd
    src = sys.argv[1] if len(sys.argv) > 1 else 'BRCA'
    df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2
                     else 'outputs/ig/%s_IG_top150.csv' % src)
    genes = list(df['gene'])
    print('输入基因 %d 个（%s 归因 top-%d）' % (len(genes), src, len(genes)))
    run(genes, 'outputs/ig/%s' % src)
