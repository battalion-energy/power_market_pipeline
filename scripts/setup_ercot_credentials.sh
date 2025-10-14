#!/bin/bash
# Setup ERCOT Web Service API Credentials

echo "=========================================================================="
echo "ERCOT Web Service API Credentials Setup"
echo "=========================================================================="
echo ""
echo "You need to obtain API credentials from ERCOT:"
echo ""
echo "1. Go to: https://www.ercot.com/"
echo "2. Click 'Sign In' (or create account if you don't have one)"
echo "3. Navigate to: Market Info → Data → API Access"
echo "4. Subscribe to the 'Public API' (it's free)"
echo "5. Get your subscription key from the API portal"
echo ""
echo "=========================================================================="
echo ""

# Check current .env file
if [ -f .env ]; then
    echo "Current .env file found. Checking credentials..."
    echo ""

    USERNAME=$(grep "^ERCOT_USERNAME=" .env | cut -d'=' -f2)
    PASSWORD_SET=$(grep "^ERCOT_PASSWORD=" .env | wc -l)
    KEY_SET=$(grep "^ERCOT_SUBSCRIPTION_KEY=" .env | wc -l)

    echo "Current settings:"
    echo "  ERCOT_USERNAME: $USERNAME"

    if [ $PASSWORD_SET -eq 1 ]; then
        echo "  ERCOT_PASSWORD: [SET]"
    else
        echo "  ERCOT_PASSWORD: [NOT SET]"
    fi

    if [ $KEY_SET -eq 1 ]; then
        echo "  ERCOT_SUBSCRIPTION_KEY: [SET]"
    else
        echo "  ERCOT_SUBSCRIPTION_KEY: [NOT SET]"
    fi

    echo ""
    echo "=========================================================================="
    echo ""

    if [ "$USERNAME" = "your_username" ]; then
        echo "⚠️  Credentials are still set to placeholder values!"
        echo ""
        echo "You need to manually edit the .env file and update:"
        echo "  - ERCOT_USERNAME (your ERCOT account email/username)"
        echo "  - ERCOT_PASSWORD (your ERCOT account password)"
        echo "  - ERCOT_SUBSCRIPTION_KEY (from API portal)"
        echo ""
        echo "Edit with: nano .env"
        echo ""
    else
        echo "Testing API connection..."
        echo ""
        uv run python -c "
import asyncio
from ercot_ws_downloader import ERCOTWebServiceClient

async def test():
    try:
        client = ERCOTWebServiceClient()
        result = await client.test_connection()
        if result:
            print('✅ API connection successful!')
            print('You are ready to download data.')
            return True
        else:
            print('❌ API connection failed!')
            print('Please check your credentials.')
            return False
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

success = asyncio.run(test())
exit(0 if success else 1)
"
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ All set! You can now run:"
            echo "   uv run python ercot_ws_download_all.py --datasets DAM_Prices"
        else
            echo ""
            echo "❌ Connection failed. Please verify:"
            echo "   1. Username is correct (usually your email)"
            echo "   2. Password is correct"
            echo "   3. Subscription key is from the Public API portal"
            echo ""
            echo "Edit credentials: nano .env"
        fi
    fi
else
    echo "❌ No .env file found!"
    echo ""
    echo "Creating template .env file..."
    cat > .env << 'EOF'
# ERCOT Web Service API Credentials
ERCOT_USERNAME=your_email@example.com
ERCOT_PASSWORD=your_password
ERCOT_SUBSCRIPTION_KEY=your_subscription_key_here

# ERCOT Data Directory
ERCOT_DATA_DIR=/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/
EOF
    echo "✅ Template .env file created"
    echo ""
    echo "Please edit .env and add your credentials:"
    echo "   nano .env"
fi

echo ""
echo "=========================================================================="
