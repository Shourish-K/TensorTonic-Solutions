import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """

    data_df = pd.DataFrame(data)

    output = {
        'data': data_df.to_dict('list'),
        'shape': list(data_df.shape),
        'columns': data_df.columns.tolist()
    }
    
    return output