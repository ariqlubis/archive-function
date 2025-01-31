import pingouin as pg
import pandas as pd
from typing import List, Dict, Any

def non_parametric(data: pd.DataFrame, group_cols: List[str], val_col: str) -> pd.DataFrame:
    """
    2 Groups: Mann-Whitney U
    > 2 Groups: Kruskal-Wallis
    """
    group = data.groupby(group_cols)[val_col].agg(list).reset_index()
    num_groups = len(group)
    
    if num_groups < 2:
        raise ValueError("Minimum of 2 groups required for comparison")
        
    result = {
        'Group Columns': ', '.join(group_cols),
        'Number of Groups': num_groups
    }
    
    if num_groups == 2:
        test_result = pg.mwu(group[val_col].iloc[0],  group[val_col].iloc[1])
        result.update({
            'Test': "Mann-Whitney U",
            'Statistic': test_result['U-val'].iloc[0],
            'p-value': test_result['p-val'].iloc[0]
        })
    else:
        values = []
        labels = []
        for idx, row in enumerate(group[val_col]):
            values.extend(row)
            group_label = '_'.join(str(x) for x in group[group_cols].iloc[idx])
            labels.extend([group_label] * len(row))
            
        test_result = pg.kruskal(data=pd.DataFrame({
            'values': values,
            'groups': labels
        }), dv='values', between='groups')
        
        result.update({
            'Test': "Kruskal-Wallis H",
            'Statistic': test_result['H'].iloc[0],
            'p-value': test_result['p-unc'].iloc[0]
        })
    
    result['Significant'] = result['p-value'] < 0.05
    return pd.DataFrame([result])
