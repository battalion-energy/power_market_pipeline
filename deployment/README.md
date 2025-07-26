# Power Market Pipeline Deployment Guide

## AWS EC2 Deployment

### Prerequisites

- AWS EC2 instance (recommended: t3.large or larger)
- Ubuntu 22.04 LTS
- PostgreSQL 15+ with TimescaleDB extension
- Python 3.11+
- Chrome/Chromium (for Selenium scrapers)

### Initial Setup

1. **Update system and install dependencies**:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip postgresql postgresql-contrib
sudo apt install -y chromium-browser chromium-chromedriver
```

2. **Install TimescaleDB**:
```bash
sudo apt install -y postgresql-15-timescaledb
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql
```

3. **Clone repository**:
```bash
git clone https://github.com/battalion-energy/power_market_pipeline.git
cd power_market_pipeline
```

4. **Set up Python environment**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip sync requirements.txt
```

5. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your credentials
nano .env
```

6. **Set up database**:
```bash
# Create database user and database
sudo -u postgres psql << EOF
CREATE USER power_market WITH PASSWORD 'your_secure_password';
CREATE DATABASE power_market OWNER power_market;
\c power_market
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
EOF

# Run migrations
python database/scripts/setup_database.py
```

### Running the Collector

#### Option 1: Manual (for testing)
```bash
source .venv/bin/activate
python realtime_collector.py
```

#### Option 2: Systemd Service (recommended for production)

1. **Copy service file**:
```bash
sudo cp deployment/power-market-collector.service /etc/systemd/system/
```

2. **Edit service file with correct paths**:
```bash
sudo nano /etc/systemd/system/power-market-collector.service
# Update paths and database URL as needed
```

3. **Enable and start service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable power-market-collector
sudo systemctl start power-market-collector
```

4. **Check service status**:
```bash
sudo systemctl status power-market-collector
sudo journalctl -u power-market-collector -f
```

### Monitoring

#### Application Logs
```bash
# View real-time logs
sudo journalctl -u power-market-collector -f

# View logs for specific time period
sudo journalctl -u power-market-collector --since "1 hour ago"
```

#### Database Monitoring
```sql
-- Check recent data
SELECT iso, market, COUNT(*) as records, 
       MAX(interval_start) as latest_data,
       NOW() - MAX(interval_start) as data_lag
FROM lmp 
WHERE interval_start > NOW() - INTERVAL '1 hour'
GROUP BY iso, market;

-- Check data growth
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### OpenTelemetry Integration (Optional)

For production monitoring with Dash0 or other OpenTelemetry collectors:

1. **Install OpenTelemetry dependencies**:
```bash
uv pip install opentelemetry-distro opentelemetry-exporter-otlp
```

2. **Configure OTEL environment**:
```bash
export OTEL_SERVICE_NAME="power-market-pipeline"
export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-otel-collector:4318"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer your-token"
```

3. **Run with OpenTelemetry**:
```bash
opentelemetry-instrument python realtime_collector.py
```

### Maintenance

#### Database Maintenance

1. **Set up compression policy** (saves disk space):
```sql
-- Compress data older than 7 days
SELECT add_compression_policy('lmp', interval '7 days');
SELECT add_compression_policy('ancillary_services', interval '7 days');
```

2. **Set up data retention** (optional):
```sql
-- Keep only 2 years of data
SELECT add_retention_policy('lmp', interval '2 years');
```

3. **Regular vacuum**:
```bash
# Add to crontab
0 2 * * * psql -U power_market -d power_market -c "VACUUM ANALYZE;"
```

#### Backup Strategy

1. **Database backups**:
```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backup/power_market"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U power_market -d power_market | gzip > "$BACKUP_DIR/power_market_$DATE.sql.gz"

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
```

2. **Add to crontab**:
```bash
0 3 * * * /home/ubuntu/backup_database.sh
```

### Troubleshooting

#### Common Issues

1. **Service won't start**:
   - Check logs: `sudo journalctl -u power-market-collector -n 100`
   - Verify paths in service file
   - Check database connectivity

2. **No data being collected**:
   - Verify API credentials in .env
   - Check network connectivity
   - Look for rate limiting errors in logs

3. **High memory usage**:
   - Adjust batch sizes in downloaders
   - Increase swap space if needed
   - Consider upgrading instance type

4. **Disk space issues**:
   - Enable TimescaleDB compression
   - Set up data retention policies
   - Monitor with: `df -h` and database size queries

### Security Best Practices

1. **Credentials**:
   - Never commit .env file
   - Use AWS Secrets Manager or Parameter Store for production
   - Rotate API keys regularly

2. **Network**:
   - Use security groups to restrict database access
   - Enable SSL for PostgreSQL connections
   - Keep system updated with security patches

3. **Monitoring**:
   - Set up CloudWatch alarms for disk space
   - Monitor for failed API calls
   - Alert on data collection gaps