"""Utility functions for validation operations."""

import re
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.logging import get_logger
from ..core.errors import ValidationError

# Initialize logger
logger = get_logger("validation_utils")

# Regular expressions for validation
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_REGEX = re.compile(r'^\+?[0-9]{10,15}$')
URL_REGEX = re.compile(
    r'^(?:http|https)://'
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
    r'localhost|'
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
    r'(?::\d+)?'
    r'(?:/?|[/?]\S+)$', re.IGNORECASE
)
IPV4_REGEX = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
IPV6_REGEX = re.compile(r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$')
USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_-]{3,16}$')
PASSWORD_REGEX = re.compile(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$')
HEX_COLOR_REGEX = re.compile(r'^#(?:[0-9a-fA-F]{3}){1,2}$')
ISBN_REGEX = re.compile(r'^(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]$')
ZIP_CODE_REGEX = re.compile(r'^\d{5}(?:-\d{4})?$')
CREDIT_CARD_REGEX = re.compile(r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$')


def is_valid_email(email: str) -> bool:
    """Check if an email address is valid.
    
    Args:
        email (str): Email address to check
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    return bool(EMAIL_REGEX.match(email))


def is_valid_phone(phone: str) -> bool:
    """Check if a phone number is valid.
    
    Args:
        phone (str): Phone number to check
        
    Returns:
        bool: True if phone number is valid, False otherwise
    """
    return bool(PHONE_REGEX.match(phone))


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    return bool(URL_REGEX.match(url))


def is_valid_ipv4(ip: str) -> bool:
    """Check if an IPv4 address is valid.
    
    Args:
        ip (str): IPv4 address to check
        
    Returns:
        bool: True if IPv4 address is valid, False otherwise
    """
    return bool(IPV4_REGEX.match(ip))


def is_valid_ipv6(ip: str) -> bool:
    """Check if an IPv6 address is valid.
    
    Args:
        ip (str): IPv6 address to check
        
    Returns:
        bool: True if IPv6 address is valid, False otherwise
    """
    return bool(IPV6_REGEX.match(ip))


def is_valid_ip(ip: str) -> bool:
    """Check if an IP address (IPv4 or IPv6) is valid.
    
    Args:
        ip (str): IP address to check
        
    Returns:
        bool: True if IP address is valid, False otherwise
    """
    return is_valid_ipv4(ip) or is_valid_ipv6(ip)


def is_valid_username(username: str) -> bool:
    """Check if a username is valid.
    
    Args:
        username (str): Username to check
        
    Returns:
        bool: True if username is valid, False otherwise
    """
    return bool(USERNAME_REGEX.match(username))


def is_valid_password(password: str) -> bool:
    """Check if a password is valid.
    
    Args:
        password (str): Password to check
        
    Returns:
        bool: True if password is valid, False otherwise
    """
    return bool(PASSWORD_REGEX.match(password))


def is_valid_hex_color(color: str) -> bool:
    """Check if a hexadecimal color code is valid.
    
    Args:
        color (str): Hexadecimal color code to check
        
    Returns:
        bool: True if color code is valid, False otherwise
    """
    return bool(HEX_COLOR_REGEX.match(color))


def is_valid_isbn(isbn: str) -> bool:
    """Check if an ISBN is valid.
    
    Args:
        isbn (str): ISBN to check
        
    Returns:
        bool: True if ISBN is valid, False otherwise
    """
    return bool(ISBN_REGEX.match(isbn))


def is_valid_zip_code(zip_code: str) -> bool:
    """Check if a ZIP code is valid.
    
    Args:
        zip_code (str): ZIP code to check
        
    Returns:
        bool: True if ZIP code is valid, False otherwise
    """
    return bool(ZIP_CODE_REGEX.match(zip_code))


def is_valid_credit_card(card_number: str) -> bool:
    """Check if a credit card number is valid.
    
    Args:
        card_number (str): Credit card number to check
        
    Returns:
        bool: True if credit card number is valid, False otherwise
    """
    # Remove spaces and dashes
    card_number = card_number.replace(' ', '').replace('-', '')
    
    # Check if the number matches the regex pattern
    if not CREDIT_CARD_REGEX.match(card_number):
        return False
    
    # Luhn algorithm for credit card validation
    digits = [int(d) for d in card_number]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(divmod(d * 2, 10))
    return checksum % 10 == 0


def is_valid_date(date_str: str, format_str: str = '%Y-%m-%d') -> bool:
    """Check if a date string is valid.
    
    Args:
        date_str (str): Date string to check
        format_str (str): Date format string
        
    Returns:
        bool: True if date string is valid, False otherwise
    """
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False


def is_valid_json(json_str: str) -> bool:
    """Check if a JSON string is valid.
    
    Args:
        json_str (str): JSON string to check
        
    Returns:
        bool: True if JSON string is valid, False otherwise
    """
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """Validate that required fields are present in data.
    
    Args:
        data (Dict[str, Any]): Data to validate
        required_fields (List[str]): List of required field names
        
    Returns:
        List[str]: List of missing field names
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    return missing_fields


def validate_field_type(value: Any, expected_type: Union[type, List[type]], field_name: str = '') -> Optional[str]:
    """Validate that a field has the expected type.
    
    Args:
        value (Any): Value to validate
        expected_type (Union[type, List[type]]): Expected type or list of types
        field_name (str): Field name for error message
        
    Returns:
        Optional[str]: Error message if validation fails, None otherwise
    """
    if value is None:
        return None
    
    if isinstance(expected_type, list):
        if not any(isinstance(value, t) for t in expected_type):
            type_names = [t.__name__ for t in expected_type]
            return f"{field_name} must be one of the following types: {', '.join(type_names)}"
    elif not isinstance(value, expected_type):
        return f"{field_name} must be of type {expected_type.__name__}"
    
    return None


def validate_string_length(value: str, min_length: int = 0, max_length: Optional[int] = None, field_name: str = '') -> Optional[str]:
    """Validate that a string has the expected length.
    
    Args:
        value (str): String to validate
        min_length (int): Minimum length
        max_length (Optional[int]): Maximum length
        field_name (str): Field name for error message
        
    Returns:
        Optional[str]: Error message if validation fails, None otherwise
    """
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    
    if len(value) < min_length:
        return f"{field_name} must be at least {min_length} characters long"
    
    if max_length is not None and len(value) > max_length:
        return f"{field_name} must be at most {max_length} characters long"
    
    return None


def validate_number_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None, 
                         max_value: Optional[Union[int, float]] = None, field_name: str = '') -> Optional[str]:
    """Validate that a number is within the expected range.
    
    Args:
        value (Union[int, float]): Number to validate
        min_value (Optional[Union[int, float]]): Minimum value
        max_value (Optional[Union[int, float]]): Maximum value
        field_name (str): Field name for error message
        
    Returns:
        Optional[str]: Error message if validation fails, None otherwise
    """
    if not isinstance(value, (int, float)):
        return f"{field_name} must be a number"
    
    if min_value is not None and value < min_value:
        return f"{field_name} must be at least {min_value}"
    
    if max_value is not None and value > max_value:
        return f"{field_name} must be at most {max_value}"
    
    return None


def validate_list_length(value: List[Any], min_length: int = 0, max_length: Optional[int] = None, 
                        field_name: str = '') -> Optional[str]:
    """Validate that a list has the expected length.
    
    Args:
        value (List[Any]): List to validate
        min_length (int): Minimum length
        max_length (Optional[int]): Maximum length
        field_name (str): Field name for error message
        
    Returns:
        Optional[str]: Error message if validation fails, None otherwise
    """
    if not isinstance(value, list):
        return f"{field_name} must be a list"
    
    if len(value) < min_length:
        return f"{field_name} must have at least {min_length} items"
    
    if max_length is not None and len(value) > max_length:
        return f"{field_name} must have at most {max_length} items"
    
    return None


def validate_enum(value: Any, allowed_values: List[Any], field_name: str = '') -> Optional[str]:
    """Validate that a value is one of the allowed values.
    
    Args:
        value (Any): Value to validate
        allowed_values (List[Any]): List of allowed values
        field_name (str): Field name for error message
        
    Returns:
        Optional[str]: Error message if validation fails, None otherwise
    """
    if value not in allowed_values:
        return f"{field_name} must be one of the following values: {', '.join(str(v) for v in allowed_values)}"
    
    return None


def validate_regex(value: str, pattern: str, field_name: str = '') -> Optional[str]:
    """Validate that a string matches a regular expression pattern.
    
    Args:
        value (str): String to validate
        pattern (str): Regular expression pattern
        field_name (str): Field name for error message
        
    Returns:
        Optional[str]: Error message if validation fails, None otherwise
    """
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    
    if not re.match(pattern, value):
        return f"{field_name} does not match the required pattern"
    
    return None


def validate_data(data: Dict[str, Any], validation_rules: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Validate data against a set of validation rules.
    
    Args:
        data (Dict[str, Any]): Data to validate
        validation_rules (Dict[str, Dict[str, Any]]): Validation rules
        
    Returns:
        Dict[str, str]: Dictionary of field names and error messages
    """
    errors = {}
    
    for field_name, rules in validation_rules.items():
        # Skip validation if field is not required and not present
        if field_name not in data and not rules.get('required', False):
            continue
        
        # Check if field is required
        if rules.get('required', False) and (field_name not in data or data[field_name] is None):
            errors[field_name] = f"{field_name} is required"
            continue
        
        # Skip further validation if field is not present
        if field_name not in data:
            continue
        
        value = data[field_name]
        
        # Check type
        if 'type' in rules:
            error = validate_field_type(value, rules['type'], field_name)
            if error:
                errors[field_name] = error
                continue
        
        # Skip further validation if value is None
        if value is None:
            continue
        
        # Check string length
        if isinstance(value, str) and ('min_length' in rules or 'max_length' in rules):
            error = validate_string_length(
                value, 
                rules.get('min_length', 0), 
                rules.get('max_length'), 
                field_name
            )
            if error:
                errors[field_name] = error
                continue
        
        # Check number range
        if isinstance(value, (int, float)) and ('min_value' in rules or 'max_value' in rules):
            error = validate_number_range(
                value, 
                rules.get('min_value'), 
                rules.get('max_value'), 
                field_name
            )
            if error:
                errors[field_name] = error
                continue
        
        # Check list length
        if isinstance(value, list) and ('min_length' in rules or 'max_length' in rules):
            error = validate_list_length(
                value, 
                rules.get('min_length', 0), 
                rules.get('max_length'), 
                field_name
            )
            if error:
                errors[field_name] = error
                continue
        
        # Check enum
        if 'enum' in rules:
            error = validate_enum(value, rules['enum'], field_name)
            if error:
                errors[field_name] = error
                continue
        
        # Check regex
        if isinstance(value, str) and 'pattern' in rules:
            error = validate_regex(value, rules['pattern'], field_name)
            if error:
                errors[field_name] = error
                continue
        
        # Check custom validation function
        if 'validate' in rules and callable(rules['validate']):
            try:
                result = rules['validate'](value)
                if result is not True:
                    errors[field_name] = result if isinstance(result, str) else f"{field_name} is invalid"
            except Exception as e:
                errors[field_name] = str(e)
    
    return errors


def validate_and_raise(data: Dict[str, Any], validation_rules: Dict[str, Dict[str, Any]]) -> None:
    """Validate data against a set of validation rules and raise an exception if validation fails.
    
    Args:
        data (Dict[str, Any]): Data to validate
        validation_rules (Dict[str, Dict[str, Any]]): Validation rules
        
    Raises:
        ValidationError: If validation fails
    """
    errors = validate_data(data, validation_rules)
    
    if errors:
        raise ValidationError(errors)