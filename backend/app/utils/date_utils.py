"""Utility functions for date and time operations."""

import time
import pytz
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any

# Define common timezones
COMMON_TIMEZONES = {
    'UTC': 'UTC',
    'EST': 'America/New_York',
    'CST': 'America/Chicago',
    'MST': 'America/Denver',
    'PST': 'America/Los_Angeles',
    'GMT': 'Europe/London',
    'CET': 'Europe/Paris',
    'IST': 'Asia/Kolkata',
    'JST': 'Asia/Tokyo',
    'AEST': 'Australia/Sydney',
}


def get_current_timestamp() -> int:
    """Get the current Unix timestamp.
    
    Returns:
        int: Current Unix timestamp (seconds since epoch)
    """
    return int(time.time())


def get_current_datetime(timezone: str = 'UTC') -> datetime:
    """Get the current datetime in the specified timezone.
    
    Args:
        timezone (str): Timezone name (default: 'UTC')
        
    Returns:
        datetime: Current datetime in the specified timezone
    """
    # Get the timezone object
    tz = pytz.timezone(COMMON_TIMEZONES.get(timezone, timezone))
    
    # Get the current datetime in UTC
    utc_now = datetime.now(pytz.UTC)
    
    # Convert to the specified timezone
    return utc_now.astimezone(tz)


def format_datetime(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Format a datetime object as a string.
    
    Args:
        dt (datetime): Datetime object to format
        format_str (str): Format string (default: '%Y-%m-%d %H:%M:%S')
        
    Returns:
        str: Formatted datetime string
    """
    return dt.strftime(format_str)


def parse_datetime(datetime_str: str, format_str: str = '%Y-%m-%d %H:%M:%S', timezone: str = 'UTC') -> datetime:
    """Parse a datetime string into a datetime object.
    
    Args:
        datetime_str (str): Datetime string to parse
        format_str (str): Format string (default: '%Y-%m-%d %H:%M:%S')
        timezone (str): Timezone name (default: 'UTC')
        
    Returns:
        datetime: Parsed datetime object
    """
    # Parse the datetime string
    dt = datetime.strptime(datetime_str, format_str)
    
    # Get the timezone object
    tz = pytz.timezone(COMMON_TIMEZONES.get(timezone, timezone))
    
    # Set the timezone
    return tz.localize(dt)


def timestamp_to_datetime(timestamp: int, timezone: str = 'UTC') -> datetime:
    """Convert a Unix timestamp to a datetime object.
    
    Args:
        timestamp (int): Unix timestamp (seconds since epoch)
        timezone (str): Timezone name (default: 'UTC')
        
    Returns:
        datetime: Datetime object
    """
    # Convert timestamp to datetime in UTC
    dt = datetime.fromtimestamp(timestamp, pytz.UTC)
    
    # Convert to the specified timezone
    if timezone != 'UTC':
        tz = pytz.timezone(COMMON_TIMEZONES.get(timezone, timezone))
        dt = dt.astimezone(tz)
    
    return dt


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert a datetime object to a Unix timestamp.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        int: Unix timestamp (seconds since epoch)
    """
    # Ensure the datetime is timezone-aware
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    
    # Convert to timestamp
    return int(dt.timestamp())


def get_date_range(start_date: Union[str, datetime], end_date: Union[str, datetime], 
                  format_str: str = '%Y-%m-%d') -> list:
    """Get a list of dates in a range.
    
    Args:
        start_date (Union[str, datetime]): Start date
        end_date (Union[str, datetime]): End date
        format_str (str): Format string for string dates (default: '%Y-%m-%d')
        
    Returns:
        list: List of dates in the range
    """
    # Convert string dates to datetime objects if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, format_str)
    
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, format_str)
    
    # Generate the date range
    date_range = []
    current_date = start_date
    
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    return date_range


def get_relative_date(days: int = 0, weeks: int = 0, months: int = 0, years: int = 0, 
                     from_date: Optional[datetime] = None, timezone: str = 'UTC') -> datetime:
    """Get a date relative to another date.
    
    Args:
        days (int): Number of days to add/subtract
        weeks (int): Number of weeks to add/subtract
        months (int): Number of months to add/subtract
        years (int): Number of years to add/subtract
        from_date (Optional[datetime]): Base date (default: current date)
        timezone (str): Timezone name (default: 'UTC')
        
    Returns:
        datetime: Relative date
    """
    # Get the base date
    if from_date is None:
        from_date = get_current_datetime(timezone)
    
    # Add/subtract days and weeks
    result = from_date + timedelta(days=days, weeks=weeks)
    
    # Add/subtract months and years
    if months != 0 or years != 0:
        # Calculate new year and month
        year = result.year + years + (result.month + months - 1) // 12
        month = (result.month + months - 1) % 12 + 1
        
        # Calculate new day (handle month length differences)
        day = min(result.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 
                              31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        
        # Create new datetime
        result = result.replace(year=year, month=month, day=day)
    
    return result


def format_time_ago(dt: datetime) -> str:
    """Format a datetime as a human-readable time ago string.
    
    Args:
        dt (datetime): Datetime to format
        
    Returns:
        str: Human-readable time ago string
    """
    # Ensure the datetime is timezone-aware
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    
    # Get the current datetime in the same timezone
    now = datetime.now(dt.tzinfo)
    
    # Calculate the time difference
    diff = now - dt
    
    # Format the time ago string
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif seconds < 2592000:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif seconds < 31536000:
        months = int(seconds / 2592000)
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = int(seconds / 31536000)
        return f"{years} year{'s' if years != 1 else ''} ago"


def get_date_components(dt: datetime) -> Dict[str, Any]:
    """Get the components of a date.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        Dict[str, Any]: Date components
    """
    return {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'minute': dt.minute,
        'second': dt.second,
        'microsecond': dt.microsecond,
        'weekday': dt.weekday(),
        'weekday_name': dt.strftime('%A'),
        'month_name': dt.strftime('%B'),
        'day_of_year': dt.timetuple().tm_yday,
        'week_of_year': dt.isocalendar()[1],
        'quarter': (dt.month - 1) // 3 + 1,
    }


def is_weekend(dt: datetime) -> bool:
    """Check if a date is a weekend.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        bool: True if the date is a weekend, False otherwise
    """
    return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday


def is_business_day(dt: datetime, holidays: list = None) -> bool:
    """Check if a date is a business day.
    
    Args:
        dt (datetime): Datetime object
        holidays (list): List of holiday dates
        
    Returns:
        bool: True if the date is a business day, False otherwise
    """
    # Check if the date is a weekend
    if is_weekend(dt):
        return False
    
    # Check if the date is a holiday
    if holidays is not None:
        date_only = dt.date()
        for holiday in holidays:
            if isinstance(holiday, datetime):
                holiday = holiday.date()
            if date_only == holiday:
                return False
    
    return True


def add_business_days(dt: datetime, days: int, holidays: list = None) -> datetime:
    """Add business days to a date.
    
    Args:
        dt (datetime): Datetime object
        days (int): Number of business days to add
        holidays (list): List of holiday dates
        
    Returns:
        datetime: Datetime with business days added
    """
    # Initialize result
    result = dt
    
    # Add business days
    while days > 0:
        result += timedelta(days=1)
        if is_business_day(result, holidays):
            days -= 1
    
    return result


def get_month_start_end(dt: datetime) -> tuple:
    """Get the start and end dates of a month.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        tuple: Tuple of (start_date, end_date)
    """
    # Get the start date (first day of the month)
    start_date = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Get the end date (last day of the month)
    if dt.month == 12:
        end_date = dt.replace(year=dt.year + 1, month=1, day=1, hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=1)
    else:
        end_date = dt.replace(month=dt.month + 1, day=1, hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=1)
    
    return (start_date, end_date)


def get_quarter_start_end(dt: datetime) -> tuple:
    """Get the start and end dates of a quarter.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        tuple: Tuple of (start_date, end_date)
    """
    # Calculate the quarter
    quarter = (dt.month - 1) // 3 + 1
    
    # Get the start date (first day of the quarter)
    start_month = (quarter - 1) * 3 + 1
    start_date = dt.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Get the end date (last day of the quarter)
    if quarter == 4:
        end_date = dt.replace(year=dt.year + 1, month=1, day=1, hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=1)
    else:
        end_date = dt.replace(month=start_month + 3, day=1, hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=1)
    
    return (start_date, end_date)


def get_year_start_end(dt: datetime) -> tuple:
    """Get the start and end dates of a year.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        tuple: Tuple of (start_date, end_date)
    """
    # Get the start date (first day of the year)
    start_date = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Get the end date (last day of the year)
    end_date = dt.replace(year=dt.year + 1, month=1, day=1, hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=1)
    
    return (start_date, end_date)