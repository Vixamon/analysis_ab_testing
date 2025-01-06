# A/B Testing Analysis

## **Project Overview**
This repository contains the analysis of two datasets related to A/B testing experiments:

1. **Cookie Cats A/B Test Dataset**
2. **Fast Food Marketing Campaign A/B Test Dataset**

Both analyses aim to evaluate and derive actionable insights from A/B testing experiments, using statistical and visual analysis techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Dashboard](#dashboard)
5. [Datasets](#datasets)
  - [Cookie Cats Dataset](#cookie-cats-dataset)
  - [Fast Food Marketing Campaign Dataset](#fast-food-marketing-campaign-dataset)
5. [Analysis Overview](#analysis-overview)
6. [Key Findings](#key-findings)
  - [Cookie Cats Findings](#cookie-cats-findings)
  - [Fast Food Findings](#fast-food-findings)
7. [Improvements and Future Work](#improvements-and-future-work)
8. [Contributors](#contributors)


## Introduction

A/B testing is a method of comparing multiple versions of a product or strategy to determine which one performs better. The purpose of these analyses is to extract actionable insights for decision-making by evaluating the results of A/B experiments.

## Setup

### Prerequisites
- Python 3.x
- Poetry for dependency management
- Jupyter Notebook (optional, for viewing the analysis)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Vixamon/analysis_ab_testing/
   cd analysis_ab_testing
   ```

2. Set up a virtual environment and install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. (Optional) If using Jupyter Notebook to view or modify the analysis:  
   ```bash
   poetry add notebook
   ```

## Project Structure
- `pyproject.toml`: Poetry configuration file listing dependencies.
- `poetry.lock`: Lock file with exact package versions.
- `WA_Marketing-Campaign.csv`: Dataset for the Fast Food A/B Testing.
- `cookie_cats.csv`: Dataset for the Cookie Cats A/B Testing.
- `fast_food_ab_analysis.ipynb`: Jupyter notebook with the analysis and insights derived from the Fast Food A/B Testing data.
- `cookie_cats_ab_analysis.ipynb`: Jupyter notebook with the analysis and insights derived from the Cookie Cats A/B Testing data.
- `utilities.py`: Contains helper functions like `find_missing_values`, `find_outliers` and other graphing functions for visualizations.

## Usage

### Running the Jupyter Notebook
To interact with the data analysis or run your own queries, use the Jupyter notebook:
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook (fast_food_ab_analysis.ipynb/cookie_cats_ab_analysis.ipynb)
   ```
2. Follow the cells to explore the analysis, or modify them to perform your own exploration.

## Dashboard

The project contains a [complementing Looker Dashboard](https://lookerstudio.google.com/s/uLMgV-dIlow), which contains graphs for target metrics such as total sales, average market size sales, average weekly sales and market contribution.

## Datasets

### Cookie Cats Dataset

- **Description:** A/B test for the mobile puzzle game Cookie Cats.
- **Columns:**
  - `user_id`: Unique user identifier.
  - `version`: Game version (control `gate_30` or experiment `gate_40`).
  - `sum_gamerounds`: Total game rounds played throughout first 14 days.
  - `retention_1`: Whether the user returned the next day.
  - `retention_7`: Whether the user returned after seven days.

### Fast Food Marketing Campaign Dataset

- **Description:** A/B test for a fast-food chain's marketing campaigns to promote a new menu item.
- **Columns:**
  - `MarketID`: Unique market identifier.
  - `MarketSize`: Market size (Small, Medium, Large).
  - `LocationID`: Unique location identifier.
  - `AgeOfStore`: Age of the store in years.
  - `Promotion`: Type of promotion applied.
  - `week`: One of four weeks when the promotions were run.
  - `SalesInThousands`: Sales in thousands of dollars.

## Analysis Overview

Each analysis includes:

- Exploratory Data Analysis (EDA) to understand the data.
- Visualization of key metrics.
- Statistical hypothesis testing to evaluate the performance of A/B tests.
- Recommendations based on the analysis.

## Key Findings

### Cookie Cats Findings

- **Retention Rates:**
  - Gate placement significantly affects 7-day retention rates.
  - `gate_30` showed better retention compared to `gate_40`.

- **Insights:**
  - Earlier gate placement (gate 30) encourages better user engagement.
  - Recommendations include implementing gate 30 for higher retention rates.

### Fast Food Findings

- **Promotion Performance:**
  - Promotion 1 consistently outperformed other strategies in increasing sales.
  - Promotion 3 has a slightly lower mean than 1, but not significantly.

- **Insights:**
  - Focus on deploying Promotion 1 for maximum impact.
  - Consider further testing of Promotion 3 to find significant difference from Promotion 1.

## Improvements and Future Work

1. **Data Analysis Enhancements:**
   - Perform deeper segmentation of users/markets to identify nuanced patterns.
   - Analyze data by further variables, such as time data in Fast Food dataset.

2. **Statistical Testing:**
   - Experiment with alternative statistical techniques.

## Stakeholders and Goals

### Cookie Cats

**Stakeholders:**
- Game development team.

**Goals:**
- Identify gate placement that maximizes user retention.
- Enhance user engagement.

### Fast Food Marketing Campaign

**Stakeholders:**
- Marketing team.
- Strategy team.

**Goals:**
- Determine the most effective promotional strategy for increasing sales.

## Actionable Insights

1. **Cookie Cats:**
   - Implement gate 30 placement to maximize retention rates.
   - Balance progression to improve player experience.

2. **Fast Food Marketing:**
   - Focus on deploying Promotion 1.
   - Consider further testing for Promotion 3.

## Contributors
- [Erikas Leonenka](https://github.com/Vixamon)
