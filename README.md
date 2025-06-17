# 📦 Purchase Order Optimization Project

## Overview

This project focuses on optimizing the **Purchase Order (PO) process** for a surgical supplies inventory system. It uses **Python for data cleaning and feature engineering**, **SQL for data extraction**, and **Tableau for visualization** to provide actionable insights into vendor performance, delivery delays, and inventory management.

---

## 🔍 Objectives

* **Analyze PO lifecycle data** to identify inefficiencies and delays.
* **Calculate key metrics** like Delay Days, Quantity Differences, Stockout Flags, Vendor Ratings, and PO Cycle Times.
* Use **machine learning (Random Forest Regressor)** to predict stockout risks based on past purchase and delivery data.
* Visualize PO and inventory trends through **Tableau dashboards** for better business decisions.

---

## 🛠️ Tech Stack

* **Python (Pandas, Seaborn, Matplotlib, Scikit-learn)** – Data preprocessing, analysis, and modeling.
* **SQL (SQLite / MySQL)** – Data extraction and joins.
* **Tableau** – Interactive dashboards for PO and inventory KPIs.

---

## 📂 Project Structure

```
├── PurchaseOrder Optimization.py    # Python notebook for data preprocessing, EDA & ML modeling
├── purchaseorder optimization.sql  # SQL query to join PO lifecycle with inventory data
├── Tableau_Dashboard.twbx          # (Not provided here, expected Tableau file for visual insights)
├── Final.csv                       # Cleaned and transformed dataset ready for ML and visualization
└── README.md                       # Project overview and documentation
```

---

## 🧩 Key Features

### 1. **Data Cleaning & Feature Engineering (Python)**

* Removed duplicate and irrelevant columns.
* Converted date columns to datetime format.
* Created new features:

  * `Delay_Days`: Difference between Delivery Date & Expected Delivery Date.
  * `Diff_Quantity`: Difference between Quantity Ordered and Received.
* Encoded categorical and datetime columns for ML modeling.
* Correlation heatmap for key variables.

### 2. **Predictive Modeling**

* **Random Forest Regressor** to predict the likelihood of stockouts (`Stockout_Flag`) based on inventory and PO features.

### 3. **SQL Integration**

* Merged **PO lifecycle** data with **Surgical Inventory** using:

```sql
SELECT * FROM po_lifecycle_vendor
LEFT JOIN surgical_inventory
ON surgical_inventory.Item_ID = po_lifecycle_vendor.Item_ID;
```

### 4. **Visualization (Tableau)**

* PO cycle trends.
* Vendor performance comparison.
* Delay reasons and distribution.
* Inventory levels vs. forecasted demand.

---

## 🚧 Future Improvements

* Deploy real-time PO status tracking.
* Integrate anomaly detection for supply chain disruptions.
* Automate stockout risk alerts.

---

## 📝 Requirements

```bash
pandas
matplotlib
seaborn
scikit-learn


