import requests

print('Server status check:')
try:
    r = requests.get('http://localhost:8000/docs')
    print(f'Status code: {r.status_code}')
    if r.status_code == 200:
        print('Server is running correctly')
    else:
        print('Server returned non-200 status code')
except Exception as e:
    print(f'Error: {e}')