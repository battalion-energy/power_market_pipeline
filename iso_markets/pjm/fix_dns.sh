#!/bin/bash
# Fix DNS resolution for api.pjm.com
# Run with: sudo ./fix_dns.sh

echo "Adding api.pjm.com to /etc/hosts..."
echo "156.154.121.141 api.pjm.com" >> /etc/hosts
echo "Done! api.pjm.com should now resolve."
