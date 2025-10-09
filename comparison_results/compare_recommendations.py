from pathlib import Path
import pandas as pd
import json

CONCERNS = ['redness', 'eyebag', 'acne', 'oiliness', 'wrinkle', 'age', 'moisture']

base = Path('.').resolve()
old_path = base / 'Experiments/Hybrid_Concern_Test/Experiments/Hybrid_Concern_Test/recommendations.csv'
new_path = base / 'experiments_update/Hybrid_Concern_Test/Experiments/Hybrid_Concern_Test/recommendations.csv'
old_df = pd.read_csv(old_path)
new_df = pd.read_csv(new_path)

concern_cols = [f'concern_{c}' for c in CONCERNS]

def attach_predicted_concern(df: pd.DataFrame) -> pd.Series:
    subset = df[concern_cols]
    idx = subset.idxmax(axis=1)
    return idx.str.replace('concern_', '', regex=False)

old_df = old_df.assign(predicted_concern=attach_predicted_concern(old_df))
new_df = new_df.assign(predicted_concern=attach_predicted_concern(new_df))

comparison_rows = []
users = sorted(set(old_df['user_id']) | set(new_df['user_id']))
for user in users:
    old_user = old_df[old_df['user_id'] == user]
    new_user = new_df[new_df['user_id'] == user]
    old_products = old_user['name'].tolist()
    new_products = new_user['name'].tolist()
    overlap = len(set(old_products) & set(new_products))
    union_len = len(set(old_products) | set(new_products)) or 1
    overlap_ratio = overlap / union_len
    duplicate_old = len(old_products) - len(set(old_products))
    duplicate_new = len(new_products) - len(set(new_products))
    avg_score_old = old_user['score'].mean() if not old_user.empty else float('nan')
    avg_score_new = new_user['score'].mean() if not new_user.empty else float('nan')
    concern_match_old = (old_user['predicted_concern'] == old_user['target_concern']).mean() if not old_user.empty else float('nan')
    concern_match_new = (new_user['predicted_concern'] == new_user['target_concern']).mean() if not new_user.empty else float('nan')
    comparison_rows.append({
        'user_id': user,
        'old_recommendations': len(old_user),
        'new_recommendations': len(new_user),
        'overlap_count': overlap,
        'overlap_ratio': overlap_ratio,
        'avg_score_old': avg_score_old,
        'avg_score_new': avg_score_new,
        'avg_score_diff': avg_score_new - avg_score_old,
        'predicted_match_rate_old': concern_match_old,
        'predicted_match_rate_new': concern_match_new,
        'duplicate_count_old': duplicate_old,
        'duplicate_count_new': duplicate_new,
    })
comparison_df = pd.DataFrame(comparison_rows).sort_values('user_id')
output_dir = base / 'comparison_results'
output_dir.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(output_dir / 'recommendation_comparison.csv', index=False)
summary = {
    'total_users': len(users),
    'mean_overlap_ratio': float(comparison_df['overlap_ratio'].mean()),
    'mean_score_diff': float(comparison_df['avg_score_diff'].mean()),
    'predicted_match_rate_old': float(comparison_df['predicted_match_rate_old'].mean()),
    'predicted_match_rate_new': float(comparison_df['predicted_match_rate_new'].mean()),
    'duplicate_count_old_total': int(comparison_df['duplicate_count_old'].sum()),
    'duplicate_count_new_total': int(comparison_df['duplicate_count_new'].sum()),
}
(output_dir / 'comparison_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Comparison artifacts written to', output_dir)
