import json

data = json.load(open('research_results/extracted_data/live_static_vs_dynamic_results.json'))
tasks = data['tasks']

domains = {}
for t in tasks:
    d = t['category']
    if d not in domains:
        domains[d] = {'base_t': 0, 'adapt_t': 0, 'base_c': 0.0, 'adapt_c': 0.0, 'count': 0}
    domains[d]['base_t'] += t['static_tokens']
    domains[d]['adapt_t'] += t['dynamic_tokens']
    domains[d]['base_c'] += t['static_cost_usd']
    domains[d]['adapt_c'] += t['dynamic_cost_usd']
    domains[d]['count'] += 1

print(f"{'Domain':<15} {'Base Tok':<10} {'Adapt Tok':<10} {'Savings':<10} {'Base Cost':<10} {'Adapt Cost':<10}")
for d, v in domains.items():
    sav = (v['base_t'] - v['adapt_t']) / v['base_t'] * 100
    print(f"{d:<15} {v['base_t']:<10} {v['adapt_t']:<10} {sav:>6.2f}%    ${v['base_c']:<9.5f} ${v['adapt_c']:<9.5f}")

tot_base_t = sum(v['base_t'] for v in domains.values())
tot_adapt_t = sum(v['adapt_t'] for v in domains.values())
tot_base_c = sum(v['base_c'] for v in domains.values())
tot_adapt_c = sum(v['adapt_c'] for v in domains.values())
tot_sav = (tot_base_t - tot_adapt_t) / tot_base_t * 100
print('-'*70)
print(f"{'Total':<15} {tot_base_t:<10} {tot_adapt_t:<10} {tot_sav:>6.2f}%    ${tot_base_c:<9.5f} ${tot_adapt_c:<9.5f}")

