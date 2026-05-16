# Customer Segmentation using K-Means Clustering

## Project Title
Customer Segmentation using K-Means Clustering

## Problem Statement
Businesses need to understand customer groups so they can target the right offers and increase revenue. This project segments mall customers based on income and spending behavior.

## Objective
- Segment customers into meaningful groups
- Identify customer types using K-Means clustering
- Interpret clusters for marketing and sales strategy
- Provide actionable business recommendations

## Dataset Information
- Dataset name: `Mall_Customers.csv`
- Description: Customer spending and demographic data for a mall
- Main columns:
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Workflow
1. Load and inspect the dataset
2. Clean and prepare the data
3. Use the elbow method to choose the best number of clusters
4. Train K-Means on customer income and spending
5. Visualize clusters and centroids
6. Interpret customer segments and business insights

## Elbow Method Explanation
The elbow method shows how the sum of squared distances decreases as cluster count increases. We choose the cluster count where the decrease starts to slow down, creating an “elbow” shape. This helps select the best number of clusters.

## K-Means Clustering Explanation
K-Means groups customers by similarity. It assigns each customer to the nearest centroid, then repositions centroids until stable. This results in clear customer clusters for analysis.

## Cluster Visualization
### Cluster plot
![Customer Segments](images/CustomerSegments.png)

### Elbow method plot
![Elbow Method](images/Elbowmethod.png)

## Business Insights
- Premium Customers (Blue Cluster): High income, high spending. Best candidates for premium offers and loyalty programs.
- Conservative Customers (Green Cluster): High income, low spending. They may respond well to value-based promotions.
- Standard Customers (Purple Cluster): Moderate income and spending. This group is stable and good for cross-selling.
- Budget Customers (Yellow Cluster): Low income, low spending. Focus on affordable products and discounts.
- Impulsive Customers (Cyan Cluster): Low income, high spending. Use impulse triggers and marketing for small, attractive deals.
- Red Points are centroids and represent the center of each cluster.

## Conclusion
This segmentation reveals distinct customer types and helps marketing teams target the right offers. High-value customers and impulse buyers can be engaged with tailored promotions, while low-spending customers benefit from budget-friendly campaigns.

## Future Improvements
- Include age and gender in clustering
- Add customer lifetime value analysis
- Build an interactive dashboard
- Test alternate clustering methods like DBSCAN
