

def corporate_car(df):
    """
    Identifies private accounts that have used cars registered to corporate accounts.
    
    Args:
        df (pandas.DataFrame): DataFrame containing parking transaction data
        
    Returns:
        pandas.DataFrame: DataFrame containing violations where private accounts used corporate cars,
                         with columns: ['parking_id', 'parkinguser_id', 'car_id', 'account_type']
    """
    # Get all car_ids used in corporate transactions
    corporate_cars = df[df['account_type'] == 'corporate']['car_id'].unique()
    
    # Find private account transactions using corporate cars
    violations = df[
        (df['account_type'] == 'private') & 
        (df['car_id'].isin(corporate_cars))
    ]
    
    # Return relevant columns
    return violations[['parking_id', 'parkinguser_id', 'car_id', 'account_type']]


def shared_car(df):
    """
    Identifies cars that are used by multiple different accounts.
    
    Args:
        df (pandas.DataFrame): DataFrame containing parking transaction data
        
    Returns:
        pandas.DataFrame: DataFrame containing cars used by multiple accounts with columns:
                         ['car_id', 'user_count', 'user_ids']
                         where:
                         - car_id: the car being shared
                         - user_count: number of unique users for this car
                         - user_ids: list of unique user_ids that used this car
    """
    # Group by car_id and aggregate unique user_ids
    shared_cars = (
        df.groupby('car_id')['parkinguser_id']
        .agg(['nunique', lambda x: list(x.unique())])
        .reset_index()
    )
    
    # Rename columns for clarity
    shared_cars.columns = ['car_id', 'user_count', 'user_ids']
    
    # Filter for cars used by more than one user
    shared_cars = shared_cars[shared_cars['user_count'] > 1]
    
    # Sort by number of users (descending)
    shared_cars = shared_cars.sort_values('user_count', ascending=False)
    
    return shared_cars


def parking_frequency(df):
    """
    Counts the number of parking transactions per user.
    
    Args:
        df (pandas.DataFrame): DataFrame containing parking transaction data
        
    Returns:
        dict: A dictionary mapping user_id to their number of parking transactions
              Example: {'user1': 5, 'user2': 3, ...}
    """
    # Count occurrences of each parkinguser_id and convert to dictionary
    return df['parkinguser_id'].value_counts().to_dict()