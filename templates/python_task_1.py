import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
     # Get unique values from id_1 and id_2 columns
  unique_ids = df['id_1'].unique().tolist() + df['id_2'].unique().tolist()

  # Create an empty DataFrame
  car_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)

  # Fill the car matrix with car values
  for index, row in df.iterrows():
    car_matrix.loc[row['id_1'], row['id_2']] = row['car']

  # Set diagonal values to 0
  car_matrix.values[np.diag_indices_from(car_matrix)] = 0

  return car_matrix
   


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here

    return dict()


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
     # Initialize an empty dictionary to store type counts
  type_counts = dict()

  # Iterate through each car value
  for car in df['car']:
    # Check if the car type already exists in the dictionary
    if car in type_counts:
      # Increment the count for the existing type
      type_counts[car] += 1
    else:
      # Add the new car type to the dictionary with a count of 1
      type_counts[car] = 1

  return type_counts
    


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    # Calculate average 'truck' values for each route
  avg_trucks_per_route = df.groupby('route')['truck'].mean()

  # Filter routes with average truck values exceeding 7
  filtered_routes = avg_trucks_per_route[avg_trucks_per_route > 7].index.tolist()

  return filtered_routes
  


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
  return matrix * np.where(matrix > 5, 1, 0)
  return np.where(np.diag(np.ones(matrix.shape[0])) == 1, 2 * matrix, 0.5 * matrix)
 


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Extract unique pairs from df
  unique_pairs = df[['id_1', 'id_2']].drop_duplicates()

  # Merge with dataset2 to obtain timestamps for each pair
  merged_df = unique_pairs.merge(dataset2, on=['id_1', 'id_2'])

  # Check if each pair has timestamps covering full 24-hour and 7-day periods
  def check_coverage(df):
    # Get timestamps for the pair
    timestamps = df['timestamp']

    # Extract hour and day information
    hours = timestamps.dt.hour
    days = timestamps.dt.dayofweek

    # Check if all hours and days are covered
    hour_coverage = hours.nunique() == 24
    day_coverage = days.nunique() == 7

    # Return True if both hour and day coverage are met
    return hour_coverage and day_coverage

  # Apply the check_coverage function to each pair
  is_complete = merged_df.groupby(['id_1', 'id_2']).apply(check_coverage)

  # Convert the result into a Series with the original index
  return pd.Series(is_complete, index=unique_pairs.index)

