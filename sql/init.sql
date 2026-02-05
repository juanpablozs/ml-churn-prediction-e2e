-- Initialize database schema for churn prediction

CREATE DATABASE IF NOT EXISTS churn_db;
USE churn_db;

-- Customer churn table
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(10),
    dependents VARCHAR(10),
    tenure INT,
    phone_service VARCHAR(10),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    contract VARCHAR(20),
    paperless_billing VARCHAR(10),
    payment_method VARCHAR(50),
    monthly_charges DECIMAL(10, 2),
    total_charges DECIMAL(10, 2),
    churn VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_churn (churn),
    INDEX idx_tenure (tenure),
    INDEX idx_contract (contract)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for storing prediction logs (optional, for monitoring)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(50),
    prediction VARCHAR(10),
    probability DECIMAL(5, 4),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_predicted_at (predicted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
