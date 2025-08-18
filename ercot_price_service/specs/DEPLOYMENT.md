# ERCOT Price Service Deployment Guide

## Quick Start

### Local Development
```bash
# Build and run
cd /home/enrico/projects/power_market_pipeline/ercot_price_service
cargo build --release
./target/release/ercot-price-server \
  --data-dir /home/enrico/data/ERCOT_data/rollup_files
```

### Docker
```bash
# Build image
docker build -t ercot-price-service:latest .

# Run container
docker run -d \
  --name ercot-price-service \
  -p 8080:8080 \
  -p 50051:50051 \
  -v /home/enrico/data/ERCOT_data/rollup_files:/data:ro \
  ercot-price-service:latest
```

### Docker Compose
```bash
docker-compose up -d
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Path to parquet data files | `/data` |
| `JSON_ADDR` | JSON API bind address | `0.0.0.0:8080` |
| `FLIGHT_ADDR` | Arrow Flight bind address | `0.0.0.0:50051` |
| `RUST_LOG` | Log level | `info` |

### Command Line Arguments

```bash
ercot-price-server [OPTIONS]

Options:
  --data-dir <PATH>      Data directory path
  --json-addr <ADDR>     JSON API address [default: 0.0.0.0:8080]
  --flight-addr <ADDR>   Flight API address [default: 0.0.0.0:50051]
  --help                 Print help
  --version              Print version
```

## Production Deployment

### System Requirements

#### Minimum
- CPU: 2 cores
- RAM: 2 GB
- Disk: 10 GB SSD
- OS: Linux (x86_64)

#### Recommended
- CPU: 4+ cores
- RAM: 4-8 GB
- Disk: 20 GB NVMe SSD
- OS: Ubuntu 22.04 LTS

### Systemd Service

Create `/etc/systemd/system/ercot-price-service.service`:

```ini
[Unit]
Description=ERCOT Price Service
After=network.target

[Service]
Type=simple
User=ercot
Group=ercot
WorkingDirectory=/opt/ercot-price-service
Environment="RUST_LOG=info"
Environment="DATA_DIR=/data/ercot"
ExecStart=/opt/ercot-price-service/bin/ercot-price-server
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/ercot-price-service

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ercot-price-service
sudo systemctl start ercot-price-service
sudo systemctl status ercot-price-service
```

### Kubernetes Deployment

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ercot-price-config
data:
  RUST_LOG: "info"
  DATA_DIR: "/data"
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ercot-price-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ercot-price-service
  template:
    metadata:
      labels:
        app: ercot-price-service
    spec:
      containers:
      - name: ercot-price-service
        image: ercot-price-service:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 50051
          name: grpc
        envFrom:
        - configMapRef:
            name: ercot-price-config
        volumeMounts:
        - name: data
          mountPath: /data
          readOnly: true
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: ercot-data-pvc
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ercot-price-service
spec:
  selector:
    app: ercot-price-service
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  - port: 50051
    targetPort: 50051
    name: grpc
  type: LoadBalancer
```

#### Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ercot-price-ingress
  annotations:
    nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
spec:
  rules:
  - host: api.ercot-prices.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: ercot-price-service
            port:
              number: 8080
  - host: grpc.ercot-prices.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ercot-price-service
            port:
              number: 50051
```

## AWS Deployment

### ECS Task Definition
```json
{
  "family": "ercot-price-service",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "ercot-price-service",
      "image": "your-ecr-repo/ercot-price-service:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        },
        {
          "containerPort": 50051,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "RUST_LOG",
          "value": "info"
        },
        {
          "name": "DATA_DIR",
          "value": "/data"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "ercot-data",
          "containerPath": "/data",
          "readOnly": true
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/api/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ],
  "volumes": [
    {
      "name": "ercot-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-xxxxxxxx",
        "rootDirectory": "/ercot-data"
      }
    }
  ]
}
```

### Lambda Function (Serverless)
Not recommended due to:
- Cold start latency
- 15-minute timeout limit
- Persistent cache requirements

Consider AWS App Runner or ECS instead.

## Monitoring Setup

### Prometheus Metrics (Future)
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ercot-price-service'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

### Grafana Dashboard
Import dashboard JSON from `monitoring/grafana-dashboard.json`

### CloudWatch (AWS)
```json
{
  "MetricName": "RequestLatency",
  "Namespace": "ERCOT/PriceService",
  "Dimensions": [
    {
      "Name": "Endpoint",
      "Value": "prices"
    }
  ]
}
```

## Load Balancing

### Nginx Configuration
```nginx
upstream ercot_json {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

upstream ercot_grpc {
    server 127.0.0.1:50051;
    server 127.0.0.1:50052;
    server 127.0.0.1:50053;
}

server {
    listen 80;
    server_name api.ercot-prices.com;

    location /api {
        proxy_pass http://ercot_json;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 443 ssl http2;
    server_name grpc.ercot-prices.com;

    location / {
        grpc_pass grpc://ercot_grpc;
    }
}
```

### HAProxy Configuration
```
frontend http_front
    bind *:80
    default_backend http_back

frontend grpc_front
    bind *:50051 proto h2
    default_backend grpc_back

backend http_back
    balance roundrobin
    server node1 127.0.0.1:8080 check
    server node2 127.0.0.1:8081 check

backend grpc_back
    balance roundrobin
    server node1 127.0.0.1:50051 check
    server node2 127.0.0.1:50052 check
```

## Data Management

### Data Updates
```bash
#!/bin/bash
# update_data.sh

# Download new data
python /opt/scripts/download_ercot_data.py

# Process and flatten
python /opt/scripts/flatten_ercot_prices.py

# Restart service to clear cache
systemctl restart ercot-price-service
```

### Backup Strategy
```bash
# Backup parquet files to S3
aws s3 sync /data/ercot s3://backup-bucket/ercot \
  --exclude "*.log" \
  --storage-class GLACIER
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
journalctl -u ercot-price-service -n 100

# Verify data directory
ls -la /data/ercot/rollup_files/

# Check permissions
chown -R ercot:ercot /opt/ercot-price-service
```

#### High Memory Usage
```bash
# Monitor memory
htop -p $(pgrep ercot-price)

# Clear cache by restarting
systemctl restart ercot-price-service
```

#### Slow Response Times
```bash
# Check disk I/O
iotop -p $(pgrep ercot-price)

# Verify data is on SSD
df -h /data
lsblk -d -o name,rota
```

### Debug Mode
```bash
RUST_LOG=debug ./ercot-price-server
```

## Security Hardening

### Network Security
```bash
# Firewall rules
ufw allow 8080/tcp
ufw allow 50051/tcp

# Restrict to specific IPs
iptables -A INPUT -p tcp --dport 8080 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### File Permissions
```bash
# Read-only data mount
mount -o ro /dev/sdb1 /data

# Restricted service user
useradd -r -s /bin/false ercot
```

### TLS Configuration
```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes

# Run with TLS
./ercot-price-server \
  --tls-cert cert.pem \
  --tls-key key.pem
```

## Performance Tuning

### System Tuning
```bash
# /etc/sysctl.conf
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 8192
net.core.netdev_max_backlog = 5000
fs.file-max = 100000
```

### Service Tuning
```toml
# config.toml
[cache]
max_size = "4GB"
ttl = 3600

[server]
worker_threads = 8
max_connections = 10000
```

## Rollback Procedure

```bash
# Tag current version
docker tag ercot-price-service:latest ercot-price-service:backup

# Deploy new version
docker pull ercot-price-service:v2.0
docker stop ercot-price-service
docker run -d --name ercot-price-service ercot-price-service:v2.0

# If issues, rollback
docker stop ercot-price-service
docker run -d --name ercot-price-service ercot-price-service:backup
```