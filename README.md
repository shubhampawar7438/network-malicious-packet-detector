# 🛡️ Network Malicious Packet Detector

This project uses machine learning to detect malicious network packets using logistic regression. It takes in network logs (e.g., TCP/UDP/ICMP traffic), trains a model, and allows predictions on new traffic data.

---

## 🧠 What It Does

- Trains a Logistic Regression model on network logs.
- Supports manual prediction based on `Protocol` and `Length`.
- Detects malicious packets in a new dataset and saves results.
- Includes performance metrics like accuracy, confusion matrix, and classification report.

---

## 📊 Input Format

Your `.xlsx` files must include the following columns:

| Time     | Source         | Destination     | Protocol | Length | Malicious (training only) |
|----------|----------------|----------------|----------|--------|----------------------------|
| 0.000    | 192.168.0.1    | 8.8.8.8         | TCP      | 54     | 0                          |

- `Protocol` can be: TCP, UDP, ICMP
- `Malicious` is required **only for training**

---

## 🚀 Usage Instructions
- Install Requirements and Run the Script
  
  ```bash
  pip install -r requirements.txt  
  python main.py

### 1. Clone the Repository

  ```bash
  git clone https://github.com/shubhampawar7438/network-malicious-packet-detector.git
  cd network-malicious-packet-detector

