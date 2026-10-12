# StayNest Hospitality Group - Hotel Booking Intelligence Project
## U23AD491 - Data Science | DS Sprint - Skill Based Assessment 2 (100 Marks)
**Use Case 10:** Hotel Booking Insight  
**Theme:** *Understand Bookings. Reduce Cancellations. Improve Occupancy.*  
**Candidate:** Pragadeesh  

---

## 🎯 Project Overview & Deliverables Matrix

This project provides a comprehensive end-to-end data science and business analytics solution for **StayNest Hospitality Group**, addressing booking dynamics, cancellation mitigation, and seasonal occupancy yield management.

All deliverables have been tailored to achieve full marks across the four assessment rounds:

| Round | Stage (Marks) | Core Requirements | Project Deliverable & File Location |
| :---: | :---: | :--- | :--- |
| **Round 1** | **Quiz** *(20 Marks)* | Objective questions from analytics, preprocessing, EDA, feature engineering, Excel, NumPy, Pandas, Matplotlib, Seaborn. | 📄 [Round1_Quiz_Preparation_Guide.md](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/quiz_preparation/Round1_Quiz_Preparation_Guide.md)<br>*(Comprehensive formula cheat sheet + 40 solved MCQs)* |
| **Round 2** | **EDA & Feature Engineering** *(30 Marks)* | Data cleaning, handling 5 quality conditions, relational merges, feature engineering, visualizations, ML feature selection, and answering 6 business questions. | 📓 [StayNest_Hotel_Booking_EDA_and_Feature_Engineering.ipynb](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/notebooks/StayNest_Hotel_Booking_EDA_and_Feature_Engineering.ipynb)<br>📊 [Master Cleaned Dataset](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/cleaned_data/staynest_cleaned_data.csv)<br>📁 [5 Modular Relational CSVs](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/dataset) |
| **Round 3** | **Data Analysis Using Excel** *(30 Marks)* | Import cleaned data, formulas (`SUMIFS`, `COUNTIF`, `AVERAGEIF`, `CORREL`), 6 Pivot Tables, Pivot Charts, and Executive Summary sheet. | 📗 [StayNest_Hotel_Booking_Analysis.xlsx](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/excel/StayNest_Hotel_Booking_Analysis.xlsx)<br>*(8 formatted sheets with native Excel interactive charts)* |
| **Round 4** | **Pitching Business Insights** *(20 Marks)* | Professional executive presentation slides, data evidence, management alerts, and data-driven recommendations. | 📽️ [StayNest_Business_Insights_Pitch.pptx](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/presentation/StayNest_Business_Insights_Pitch.pptx)<br>*(12 high-impact widescreen presentation slides)* |

---

## 📂 Project Directory Structure

```text
d:\College\SECE\SEM5\SAT\2\Pragadeesh\
├── dataset\                                  <- Modular & Raw Datasets
│   ├── bookings.csv                          <- Transactional bookings, dates, rates, statuses
│   ├── guests.csv                            <- Guest demographics, origin, repeat loyalty status
│   ├── hotels.csv                            <- Property metadata, city, room capacities
│   ├── channels.csv                          <- Marketing & distribution channels
│   ├── calendar.csv                          <- Date dimensions, seasons, holidays, local events
│   └── hotel_bookings.csv                    <- Raw uploaded source data (119,390 rows)
├── cleaned_data\
│   └── staynest_cleaned_data.csv             <- Master unified cleaned dataset (86,637 rows, 60 cols)
├── models\                                    <- Machine Learning Model Artifacts (CO3 & CO4)
│   └── staynest_cancellation_model.pkl       <- Trained Random Forest Classifier (80.61% Accuracy, 0.848 AUC)
├── app.py                                    <- Interactive Streamlit Web Application
├── notebooks\
│   └── StayNest_Hotel_Booking_EDA_and_Feature_Engineering.ipynb <- Full Python EDA & ML notebook
├── excel\
│   └── StayNest_Hotel_Booking_Analysis.xlsx  <- Formatted Excel workbook with Pivot Tables & Charts
├── presentation\
│   └── StayNest_Business_Insights_Pitch.pptx <- 12-slide Executive pitch deck with embedded charts
├── quiz_preparation\
│   └── Round1_Quiz_Preparation_Guide.md      <- 20-mark quiz study guide & 40 solved MCQs
├── reports\
│   └── figures\                              <- Publication-quality analytical charts (PNG)
│       ├── 1_channel_cancellation.png
│       ├── 2_lead_time_cancellation.png
│       ├── 3_seasonal_occupancy_revenue.png
│       ├── 4_repeat_guests_cancellation.png
│       ├── 5_room_performance.png
│       ├── 6_event_impact.png
│       └── 7_ml_feature_importance.png
└── scripts\                                  <- Reproducible automation scripts
    ├── prepare_data.py                       <- Data cleaning & schema decomposition
    ├── train_model.py                        <- ML model training & evaluation pipeline
    ├── generate_notebook.py                  <- Jupyter Notebook generator
    ├── generate_charts.py                    <- High-resolution chart generator
    ├── build_excel_workbook.py               <- Excel workbook generator
    └── build_presentation.py                 <- PowerPoint deck generator
```

---

## 🔍 Data Quality Audit & Resolution Summary

| Assessment Condition | Observed Issue | Resolution & Engineering Implemented |
| :--- | :--- | :--- |
| **1. Missing Guest Origin** | 452 guest records had missing nationality codes in `country`. | Imputed with `'Unknown'` category; mapped guest origins cleanly in `guests.csv` without dropping records. |
| **2. Duplicate Bookings** | 31,994 exact duplicate rows (26.80% of raw data). | Deduplicated transactions to avoid inflated volume and biased cancellation statistics. |
| **3. Inconsistent Room Labels** | 14,917 bookings had mismatched reserved vs assigned room types. | Standardized room labels to uppercase; created `room_mismatch` flag to analyze operational room upgrades. |
| **4. Invalid Stay Dates / Zero Nights** | 757 records had 0 night stays (day-use) or 0 total guests. | Filtered out zero-duration and zero-occupant records to ensure genuine lodging occupancy metrics. |
| **5. Extreme Room Rates** | 2 extreme ADR records (negative ADR `-$6.38` and unfeasible spike `$5,400`). | Filtered out rate entry errors to safeguard revenue statistical integrity. |

---

## 💡 The 6 Core Analytical Questions: Empirical Findings

### 1. Which channels have higher cancellations?
- **Finding:** **TA/TO (Travel Agents / Tour Operators)** exhibited by far the highest cancellation rate at **31.16%** across 68,655 bookings.
- **Comparison:** **Direct (14.98%)** and **Corporate (12.82%)** channels demonstrated over **2x greater booking commitment**.
- **Action:** Impose non-refundable policies for third-party OTA channels; offer exclusive perks (free breakfast, late checkout) to drive direct web bookings.

### 2. How does lead time relate to cancellation?
- **Finding:** Statistically significant positive correlation (**$r = +0.1835$**).
- **Comparison:** Short lead-time bookings (0–14 days) cancel only **11.63%** of the time, while bookings made >180 days in advance cancel at **39.85%** (**3.4x higher**).
- **Action:** Institute stepped non-refundable advance deposits (25% deposit for >60 days; 50% deposit for >120 days).

### 3. Which seasons have higher occupancy?
- **Finding:** **Summer** is the supreme revenue peak: **19,735 completed stays**, **$138.08 Average Daily Rate (ADR)**, generating **$16.60 Million** in completed room revenue.
- **Comparison:** **Winter** experiences demand collapse: **11,898 completed stays**, **$76.50 ADR**, generating only **$3.96 Million** (a **4.2x seasonal spread**).
- **Action:** Enforce dynamic surge yield pricing in Summer; package corporate retreats, wellness packages, and digital nomad monthly stays in Winter to keep occupancy above 65%.

### 4. Do repeat guests cancel less?
- **Finding:** **Yes, decisively!** Repeat guests cancel at only **8.24%**, compared to **28.42%** for first-time guests (**3.45x less likely to cancel**).
- **Comparison:** Repeat guests also book with shorter, more committed lead times (average 33.7 days vs 81.6 days).
- **Action:** Launch the **"StayNest Circle"** loyalty rewards program to expand repeat guest share beyond the current 3.6% baseline.

### 5. Which room types perform best?
- **Finding:** **Room Type A** is the core volume and revenue foundation (56,008 bookings, **$11.98M** realized revenue, $93.08 ADR).
- **Yield Drivers:** **Room Type D** delivers **$5.69M** at **$122.80 ADR**; Suites (**Rooms F & G**) capture up to **$178.20 ADR**.
- **Action:** Protect Room A inventory baseline; implement automated 48-hour pre-arrival paid upgrade nudges to upsell guests into premium categories.

### 6. How do local events affect bookings?
- **Finding:** Cultural festivals and business summits command a **+22.1% ADR premium** (**$125.59** on event days vs **$102.83** on regular days).
- **Top Events:** **Algarve Summer Regatta ($167.33 ADR)** and **Summer Gastronomy Fair ($151.78 ADR)**.
- **Action:** Integrate city event calendars into StayNest's revenue management system with automatic +25% base rate surges and a 3-night Minimum Length of Stay (MLOS).

---

## 🏆 Assessment Submission Checklist

- [x] **Round 1:** Read [Round1_Quiz_Preparation_Guide.md](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/quiz_preparation/Round1_Quiz_Preparation_Guide.md) and practice the 40 solved MCQs.
- [x] **Round 2:** Open and run [StayNest_Hotel_Booking_EDA_and_Feature_Engineering.ipynb](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/notebooks/StayNest_Hotel_Booking_EDA_and_Feature_Engineering.ipynb) in VS Code or Jupyter.
- [x] **Round 3:** Open and review [StayNest_Hotel_Booking_Analysis.xlsx](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/excel/StayNest_Hotel_Booking_Analysis.xlsx) in Microsoft Excel.
- [x] **Round 4:** Open [StayNest_Business_Insights_Pitch.pptx](file:///d:/College/SECE/SEM5/SAT/2/Pragadeesh/presentation/StayNest_Business_Insights_Pitch.pptx) in PowerPoint for your presentation pitch.
