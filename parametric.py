import pingouin as pg
import pandas as pd
from typing import List, Dict, Any

def parametric(data: pd.DataFrame, group_cols: List[str], val_col: str) -> pd.DataFrame:
    """
    2 Groups: T-test
    > 2 Groups: ANOVA
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
        test_result = pg.ttest(group[val_col].iloc[0], group[val_col].iloc[1], paired=False)
        result.update({
            'Test': "Independent t-test",
            'Statistic': test_result['T'].iloc[0],
            'p-value': test_result['p-val'].iloc[0]
        })
    else:
        values = []
        labels = []
        for idx, row in enumerate(group[val_col]):
            values.extend(row)
            group_label = '_'.join(str(x) for x in group[group_cols].iloc[idx])
            labels.extend([group_label] * len(row))
        
        test_result = pg.anova(data=pd.DataFrame({
            'values': values,
            'groups': labels
        }), dv='values', between='groups')
        
        result.update({
            'Test': "ANOVA (One-way)",
            'Statistic': test_result['F'].iloc[0],
            'p-value': test_result['p-unc'].iloc[0]
        })
    
    result['Significant'] = result['p-value'] < 0.05
    return pd.DataFrame([result])
