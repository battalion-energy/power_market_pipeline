#!/bin/bash

KEY="5412f3d8eac44314bc423c31accb9180"

echo "Testing different PJM API endpoint patterns..."
echo "=============================================="

# Pattern 1: Direct dataminer2 API path
echo -e "\n1. Testing: https://dataminer2.pjm.com/api/v1/da_hrl_lmps"
curl -s -H "Ocp-Apim-Subscription-Key: $KEY" \
  "https://dataminer2.pjm.com/api/v1/da_hrl_lmps?rowCount=1&startRow=1" \
  -w "\nHTTP Status: %{http_code}\n" | head -20

# Pattern 2: dataminer2 with feed path
echo -e "\n2. Testing: https://dataminer2.pjm.com/feed/da_hrl_lmps"
curl -s -H "Ocp-Apim-Subscription-Key: $KEY" \
  "https://dataminer2.pjm.com/feed/da_hrl_lmps?rowCount=1&startRow=1" \
  -w "\nHTTP Status: %{http_code}\n" | head -20

# Pattern 3: apiportal
echo -e "\n3. Testing: https://apiportal.pjm.com/api/v1/da_hrl_lmps"
curl -s -H "Ocp-Apim-Subscription-Key: $KEY" \
  "https://apiportal.pjm.com/api/v1/da_hrl_lmps?rowCount=1&startRow=1" \
  -w "\nHTTP Status: %{http_code}\n" | head -20

# Pattern 4: pjm-dataminer2
echo -e "\n4. Testing: https://pjm-dataminer2.pjm.com/api/v1/da_hrl_lmps"
curl -s -H "Ocp-Apim-Subscription-Key: $KEY" \
  "https://pjm-dataminer2.pjm.com/api/v1/da_hrl_lmps?rowCount=1&startRow=1" \
  -w "\nHTTP Status: %{http_code}\n" | head -20

# Pattern 5: Check what dataminer2 root returns
echo -e "\n5. Testing root: https://dataminer2.pjm.com/api/v1/"
curl -s -H "Ocp-Apim-Subscription-Key: $KEY" \
  "https://dataminer2.pjm.com/api/v1/" \
  -w "\nHTTP Status: %{http_code}\n" | head -20

