#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
    database='INCREMENTALITY',
    schema='INCREMENTALITY_RESEARCH'
)

print("Testing different date filters...\n")

# Test 1: Count total clicks in the table (no date filter)
query1 = """
SELECT COUNT(*) as total_clicks
FROM CLICKS
LIMIT 1
"""

cursor = conn.cursor()
cursor.execute(query1)
result = cursor.fetchone()
print(f"Total clicks in table: {result[0]:,}")

# Test 2: Get date range
query2 = """
SELECT
    MIN(OCCURRED_AT) as min_date,
    MAX(OCCURRED_AT) as max_date
FROM CLICKS
"""

cursor.execute(query2)
result = cursor.fetchone()
print(f"Date range: {result[0]} to {result[1]}")

# Test 3: Try a specific date with vendor aggregation
query3 = """
SELECT
    COUNT(DISTINCT VENDOR_ID) as vendors,
    COUNT(*) AS clicks
FROM CLICKS
WHERE
    DATE(OCCURRED_AT) = '2025-07-01'
"""

print("\nTesting DATE() function filter for 2025-07-01...")
cursor.execute(query3)
result = cursor.fetchone()
print(f"  Vendors: {result[0]:,}, Clicks: {result[1]:,}")

# Test 4: Try timestamp range
query4 = """
SELECT
    COUNT(DISTINCT VENDOR_ID) as vendors,
    COUNT(*) AS clicks
FROM CLICKS
WHERE
    OCCURRED_AT >= '2025-07-01'::TIMESTAMP_NTZ
    AND OCCURRED_AT < '2025-07-02'::TIMESTAMP_NTZ
"""

print("\nTesting TIMESTAMP_NTZ range filter for 2025-07-01...")
cursor.execute(query4)
result = cursor.fetchone()
print(f"  Vendors: {result[0]:,}, Clicks: {result[1]:,}")

# Test 5: Get top 5 vendors for the day
query5 = """
SELECT
    VENDOR_ID,
    COUNT(*) AS clicks
FROM CLICKS
WHERE
    DATE(OCCURRED_AT) = '2025-07-01'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 5
"""

print("\nTop 5 vendors for 2025-07-01:")
cursor.execute(query5)
results = cursor.fetchall()
for vendor_id, clicks in results:
    print(f"  {vendor_id}: {clicks:,} clicks")

conn.close()
print("\nConnection closed.")