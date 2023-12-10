import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
 from scipy.spatial.distance import pdist
  distance_matrix = pdist(df, metric='euclidean')
  distance_matrix = squareform(distance_matrix)
  return pd.DataFrame(distance_matrix, index=df.index, columns=df.index)
   


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
# Extract upper triangular portion of the matrix (excluding diagonal)
  mask = np.triu(np.ones_like(df), k=1).astype(bool)
  tri_matrix = df[mask]

  # Stack the matrix into a new DataFrame
  unrolled_df = pd.DataFrame(tri_matrix.stack())

  # Rename columns and add 'id_end' column
  unrolled_df.rename(columns={0: 'distance'}, inplace=True)
  unrolled_df['id_end'] = tri_matrix.index

  # Add 'id_start' column by copying the corresponding row index
  unrolled_df['id_start'] = unrolled_df['id_end']

  # Reset index and return
  return unrolled_df.reset_index(drop=True)
   


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
# Calculate average distance for reference ID
  reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

  # Calculate 10% threshold
  threshold_distance = reference_avg_distance * 0.1

  # Filter IDs within the threshold range
  filtered_df = df[(df['distance'] >= reference_avg_distance - threshold_distance) &
                   (df['distance'] <= reference_avg_distance + threshold_distance)]

  # Filter out the reference ID itself
  filtered_df = filtered_df[filtered_df['id_start'] != reference_id]

  # Return filtered DataFrame
  return filtered_df
  

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
# Define vehicle type mapping function (example)
  def get_vehicle_type(id):
    # Implement logic to map ID to vehicle type
    # This example simply returns a constant value for demonstration purposes
    return "truck"

  # Map vehicle types for each ID
  df['vehicle_type_start'] = df['id_start'].apply(get_vehicle_type)
  df['vehicle_type_end'] = df['id_end'].apply(get_vehicle_type)

  # Calculate toll amount
  df['toll_amount'] = df.apply(
      lambda row: toll_rates[row['vehicle_type_start']] + toll_rates[row['vehicle_type_end']], axis=1)

  return df
   


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
# Define function to check if timestamp falls within a time interval
  def is_in_interval(timestamp, interval):
    start_time, end_time = interval
    return (timestamp.hour >= start_time) and (timestamp.hour < end_time)

  # Add 'time_interval' column
  df['time_interval'] = df['timestamp'].apply(
      lambda timestamp: [interval for interval in time_intervals if is_in_interval(timestamp, interval)][0]
  )

  # Define function to calculate time-based toll amount
  def calculate_time_based_toll(row):
    vehicle_type = row['vehicle_type_start']
    base_rate = toll_rates[vehicle_type]
    interval_rate = toll_rates.get(f"{vehicle_type}_{row['time_interval']}", 0)
    return base_rate + interval_rate

  # Add 'time_based_toll_amount' column
  df['time_based_toll_amount'] = df.apply(calculate_time_based_toll, axis=1)

  return df
   
