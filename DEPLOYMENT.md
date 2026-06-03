# 🚀 AWS EC2 Production Deployment Guide

This guide details the infrastructure configuration, container deployment strategy, and validation steps to host the Customer Segmentation Service on an AWS EC2 cloud instance using Docker.

## System Architecture Endpoints
- **Core Web Application (Streamlit)**: `http://<EC2_PUBLIC_IP>:5000`
- **Experiment Tracking Registry (MLflow UI)**: `http://<EC2_PUBLIC_IP>:8000`

---

## Step 1: Launch and Configure EC2 Instance

1. **Provision Compute**: Launch an Amazon EC2 instance using Ubuntu 22.04 LTS (a `t2.micro` or `t2.medium` instance is recommended).
2. **Configure Firewall / Security Groups**: Expose the minimum necessary ingress ports to authorize external traffic pipelines securely:
   - **SSH (Port 22)**: For secure remote shell administration.
   - **Streamlit App (Port 5000)**: To handle client web traffic and interactive dashboard.
   - **MLflow UI (Port 8000)**: To monitor analytical run histories and model parameter logs.

### Establish Secure SSH Connection

#### On Windows (PowerShell):
```powershell
# Restrict file permissions to the current user (Windows equivalent of chmod 400)
icacls "segment-key.pem" /inheritance:r
icacls "segment-key.pem" /grant:r "$($env:USERNAME):R"

# Connect to the remote instance
ssh -i "segment-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

#### On Linux/Mac:
```bash
# Set strict read-only permissions for the private key
chmod 400 segment-key.pem

# Connect to the remote instance
ssh -i "segment-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

---

## Step 2: Install Container Runtime Environment

Once authenticated within the remote Ubuntu shell, initialize and configure the Docker engine:
```bash
# Update local package indexes
sudo apt-get update

# Install the standard Docker runtime
sudo apt-get install -y docker.io

# Enable the Docker daemon to automatically initialize on system boot
sudo systemctl start docker
sudo systemctl enable docker

# Add the default ubuntu user to the docker group to execute commands without sudo
sudo usermod -aG docker ubuntu

# CRITICAL: Terminate session and reconnect via SSH for group updates to take effect
exit
```

---

## Step 3: Deploy the Customer Segmentation Service

Reconnect to your EC2 instance and run the following deployment steps to build the image layer and run the container with persistence guards:

```bash
# 1. Clone the production source code from the repository
git clone https://github.com/RedSamurai07/Customer-Segmentation-with-RFM-Analysis.git
cd Customer-Segmentation-with-RFM-Analysis

# 2. Build the Docker application image layer
docker build -t rfm-segmentation-app .

# 3. Instantiate the production container engine
# Maps runtime ports, ensures data persistence, and establishes crash auto-restart logic
docker run -d \
  -p 5000:5000 \
  -p 8000:8000 \
  -v mlflow_runs:/app/mlruns \
  --name rfm-service \
  --restart unless-stopped \
  rfm-segmentation-app
```

### 💡 Production Enhancements Added:
- `--restart unless-stopped`: Ensures the customer segmentation service automatically reboots if the application crashes or the underlying EC2 server undergoes a hardware reboot.
- `-v mlflow_runs:/app/mlruns`: Mounts a persistent named Docker volume so your tracked MLflow metadata and evaluation metrics survive container updates and deletions.

---

## Step 4: Infrastructure & Service Verification

1. **Health-Check Endpoint / UI Validation**
   Test the baseline responsiveness of the Streamlit application from your local machine web browser:
   `http://<EC2_PUBLIC_IP>:5000`

2. **Interactive Web Dashboard**
   Streamlit serves the interactive UI. You can upload transaction datasets, trigger RFM calculations, and visualize customer tiers in your web browser:
   `http://<EC2_PUBLIC_IP>:5000`

3. **Verify MLflow Experiment Logs**
   Access the MLflow tracking dashboard to view experiment runs and logged RFM metrics:
   `http://<EC2_PUBLIC_IP>:8000`

## CI/CD Pipeline Status
The operational integrity of the master codebase is continuously protected via automated integration testing gates:
