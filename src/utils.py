import polars as pl


def calculate_percentile(df: pl.DataFrame, column: str, num: float) -> float:
    """
    Calculate the percentile of a number in a specified column of a Polars DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame.
        column (str): The column name in which to calculate the percentile.
        num (float): The number for which to calculate the percentile.

    Returns:
        float: The percentile of the number.
    """
    # Sort the column and find the position of the number
    sorted_df = df.sort(column)
    count_less_than_equal = sorted_df.filter(pl.col(column) <= num).height

    # Calculate the percentile
    percentile = (count_less_than_equal / sorted_df.height) * 100
    return percentile
