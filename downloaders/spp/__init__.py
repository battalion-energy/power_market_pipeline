"""SPP (Southwest Power Pool) downloader module.

IMPORTANT: SPP API Access Requirements
======================================

The Southwest Power Pool Marketplace API requires digital certificate authentication.
This is NOT a simple username/password - you need client certificates issued by SPP.

Certificate Setup Process:
1. Contact: SPP Customer Relations at (501) 614-3200
2. Request API access and digital certificate
3. Provide certificate information to your SPP account representative
4. Wait for certificate issuance and API subscription approval
5. Process typically takes 2-4 weeks

Alternative Data Access:
========================

While waiting for certificate approval, consider using:

1. gridstatus.io Python library:
   ```python
   import gridstatus
   spp = gridstatus.SPP()
   lmp_data = spp.get_lmp(date='2024-01-01', market='DAM')
   ```

2. SPP Marketplace website: https://marketplace.spp.org/
   - Manual CSV downloads available for limited data

API Documentation:
==================
Latest: Marketplace Markets Web Service 41.0 (Jan 23, 2025)
Available at: https://www.spp.org/spp-documents-filings/?id=21070

Market Types:
- Day-Ahead Market (DAM)
- Real-Time Balancing Market (RTBM)

Data Available:
- LMP by Location (hubs, interfaces, zones)
- LMP by Bus (individual network nodes)
- Operating Reserves (Regulation, Spinning, Supplemental)
- Co-optimized energy and ancillary services

Environment Variables (when certificates are ready):
====================================================
SPP_CERT_PATH       # Path to client certificate (.pem or .p12)
SPP_CERT_PASSWORD   # Certificate password (if encrypted)
SPP_API_KEY         # API subscription key (if separate from cert)
"""

from downloaders.spp.downloader_v2 import SPPDownloaderV2

__all__ = ["SPPDownloaderV2"]
