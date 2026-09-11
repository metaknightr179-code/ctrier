import ast, os

configs = ['nodiv','lamb0005','lamb001','lamb005','lamb01','consec0001']
variants = ['kuairec_highest_individual','kuairec_highest_average','kuairec_first_individual','kuairec_first_average']

header = f"{'Config':<12} {'Variant':<28} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'N@20':>7} {'ILD@20':>7} {'CS@20':>7} {'CC@20':>7} {'Epoch':>6}"
print(header)
print('-'*len(header))

for cfg in configs:
    for var in variants:
        path = f'save_pt_type_{cfg}_{var}/test_result.txt'
        if not os.path.exists(path):
            print(f"{cfg:<12} {var:<28} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>6}")
            continue
        with open(path) as f:
            line = f.readline().strip()
        try:
            d = ast.literal_eval(line)
            print(f"{cfg:<12} {var:<28} {d['recall@5_f']:>7.4f} {d['recall@10_f']:>7.4f} {d['recall@20_f']:>7.4f} {d['ndcg@20_f']:>7.4f} {d['ILD@20']:>7.4f} {d['CS@20']:>7.4f} {d['CC@20']:>7.4f} {d['epoch']:>6}")
        except Exception as e:
            print(f"{cfg:<12} {var:<28} ERROR: {e}")
