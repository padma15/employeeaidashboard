# AI Adoption Tracking and Intervention System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime as dt

class AIAdoptionSystem:
    def __init__(self):
        self.employees_df = None
        self.adoption_scores = None
        self.clusters = None
        self.intervention_plans = {
            'High Resistance': {
                'actions': [
                    'One-on-one sessions with AI champions',
                    'Personalized skills assessment',
                    'Job role evolution pathway',
                    'Peer mentoring program'
                ],
                'resources': ['Training budget', 'Dedicated mentor time', 'Job security guarantee']
            },
            'Moderate Resistance': {
                'actions': [
                    'Department-level workshops',
                    'Hands-on AI tools training',
                    'Regular progress check-ins'
                ],
                'resources': ['Group training sessions', 'Practice projects']
            },
            'Curious Adopters': {
                'actions': [
                    'Advanced AI skills training',
                    'Innovation challenges',
                    'Champion program enrollment'
                ],
                'resources': ['Advanced course access', 'Innovation time allocation']
            },
            'AI Champions': {
                'actions': [
                    'Lead workshops for peers',
                    'Contribute to AI strategy',
                    'Develop use cases for department'
                ],
                'resources': ['Recognition program', 'Leadership exposure', 'Innovation budget']
            }
        }
    
    def load_sample_data(self, size=100):
        """Generate sample employee data for demonstration"""
        departments = ['Marketing', 'Finance', 'Operations', 'IT', 'HR', 'Customer Service']
        roles = ['Manager', 'Associate', 'Specialist', 'Analyst', 'Director']
        tenure_years = np.random.randint(1, 15, size=size)
        
        data = {
            'employee_id': range(1001, 1001+size),
            'department': np.random.choice(departments, size=size),
            'role': np.random.choice(roles, size=size),
            'tenure_years': tenure_years,
            'age': np.random.randint(23, 65, size=size),
            'previous_tech_score': np.random.randint(1, 10, size=size),
            'ai_knowledge': np.random.randint(1, 10, size=size),
            'training_attendance': np.random.randint(0, 5, size=size),
            'tool_usage_frequency': np.random.randint(0, 30, size=size),
            'sentiment_score': np.random.uniform(1, 10, size=size),
            'concern_level': np.random.randint(1, 10, size=size)
        }
        
        self.employees_df = pd.DataFrame(data)
        return self.employees_df
    
    def calculate_adoption_scores(self):
        """Calculate AI adoption scores based on multiple factors"""
        if self.employees_df is None:
            raise ValueError("Employee data not loaded. Call load_sample_data() first.")
        
        # Normalize factors for scoring
        self.employees_df['norm_ai_knowledge'] = self.employees_df['ai_knowledge'] / 10
        self.employees_df['norm_training'] = self.employees_df['training_attendance'] / 5
        self.employees_df['norm_usage'] = self.employees_df['tool_usage_frequency'] / 30
        self.employees_df['norm_sentiment'] = self.employees_df['sentiment_score'] / 10
        self.employees_df['norm_concern'] = (10 - self.employees_df['concern_level']) / 10  # Invert concern
        
        # Calculate weighted adoption score
        weights = {
            'norm_ai_knowledge': 0.15,
            'norm_training': 0.20,
            'norm_usage': 0.35,
            'norm_sentiment': 0.15,
            'norm_concern': 0.15
        }
        
        self.employees_df['adoption_score'] = sum(
            self.employees_df[col] * weight for col, weight in weights.items()
        ) * 100  # Scale to 0-100
        
        return self.employees_df[['employee_id', 'adoption_score']]
    
    def segment_employees(self, n_clusters=4):
        """Segment employees using K-means clustering"""
        if 'adoption_score' not in self.employees_df.columns:
            self.calculate_adoption_scores()
        
        # Select features for clustering
        features = [
            'adoption_score', 'ai_knowledge', 'training_attendance', 
            'tool_usage_frequency', 'sentiment_score', 'concern_level'
        ]
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.employees_df[features])
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.employees_df['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Calculate cluster centroids and assign labels
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroid_df = pd.DataFrame(centroids, columns=features)
        
        # Determine labels based on adoption_score
        sorted_clusters = centroid_df.sort_values('adoption_score').index
        labels = ['High Resistance', 'Moderate Resistance', 'Curious Adopters', 'AI Champions']
        
        # Map cluster numbers to labels
        cluster_mapping = {cluster_num: labels[i] for i, cluster_num in enumerate(sorted_clusters)}
        self.employees_df['segment'] = self.employees_df['cluster'].map(cluster_mapping)
        
        self.clusters = self.employees_df.groupby('segment').agg({
            'employee_id': 'count',
            'adoption_score': 'mean',
            'concern_level': 'mean',
            'ai_knowledge': 'mean',
            'tool_usage_frequency': 'mean'
        }).rename(columns={'employee_id': 'count'})
        
        return self.clusters
    
    def generate_intervention_plan(self, employee_id=None, segment=None):
        """Generate intervention plan for an employee or segment"""
        if employee_id is not None:
            employee = self.employees_df[self.employees_df['employee_id'] == employee_id]
            if employee.empty:
                return f"Employee {employee_id} not found"
            segment = employee['segment'].values[0]
        
        if segment is None:
            return "Must provide either employee_id or segment"
            
        if segment not in self.intervention_plans:
            return f"No intervention plan for segment: {segment}"
        
        return self.intervention_plans[segment]
    
    def track_adoption_over_time(self, weeks=10):
        """Simulate adoption tracking over time with interventions"""
        if self.employees_df is None:
            self.load_sample_data()
            self.segment_employees()
        
        # Create time-series data
        time_data = []
        today = dt.datetime.now()
        
        segments = self.employees_df['segment'].unique()
        
        # Initial values from current segments
        segment_scores = {
            segment: self.employees_df[self.employees_df['segment'] == segment]['adoption_score'].mean()
            for segment in segments
        }
        
        # Growth rates by segment (simulated)
        growth_rates = {
            'High Resistance': 0.05,
            'Moderate Resistance': 0.08,
            'Curious Adopters': 0.12,
            'AI Champions': 0.04
        }
        
        # Generate time series
        for week in range(weeks):
            week_date = today + dt.timedelta(weeks=week)
            
            # Apply growth rates with some randomness
            for segment in segments:
                base_growth = growth_rates.get(segment, 0.05)
                random_factor = np.random.uniform(0.8, 1.2)
                growth = base_growth * random_factor
                
                # Add intervention boost after week 4
                if week == 4:
                    intervention_boost = {
                        'High Resistance': 5,
                        'Moderate Resistance': 3,
                        'Curious Adopters': 2,
                        'AI Champions': 1
                    }.get(segment, 0)
                    segment_scores[segment] += intervention_boost
                
                # Apply growth with ceiling of 100
                segment_scores[segment] = min(100, segment_scores[segment] * (1 + growth))
                
                time_data.append({
                    'date': week_date,
                    'segment': segment,
                    'avg_adoption_score': segment_scores[segment]
                })
        
        return pd.DataFrame(time_data)
    
    def visualize_segments(self):
        """Visualize employee segments"""
        if self.clusters is None:
            self.segment_employees()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        self.clusters['count'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Employee Segments by Size', fontsize=14)
        ax.set_ylabel('Number of Employees')
        ax.set_xlabel('Segment')
        
        for i, v in enumerate(self.clusters['count']):
            ax.text(i, v + 1, str(int(v)), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Adoption scores by segment
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart for adoption scores
        self.clusters['adoption_score'].plot(kind='bar', ax=ax, color='green')
        ax.set_title('Average Adoption Score by Segment', fontsize=14)
        ax.set_ylabel('Adoption Score')
        ax.set_xlabel('Segment')
        
        for i, v in enumerate(self.clusters['adoption_score']):
            ax.text(i, v + 1, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_adoption_trend(self):
        """Visualize adoption trend over time"""
        trend_data = self.track_adoption_over_time(weeks=12)
        
        plt.figure(figsize=(12, 6))
        
        for segment in trend_data['segment'].unique():
            segment_data = trend_data[trend_data['segment'] == segment]
            plt.plot(segment_data['date'], segment_data['avg_adoption_score'], 
                    marker='o', label=segment)
        
        plt.axvline(x=trend_data['date'].unique()[4], color='red', linestyle='--', 
                   alpha=0.5, label='Intervention')
        
        plt.title('AI Adoption Trend by Segment', fontsize=14)
        plt.xlabel('Time')
        plt.ylabel('Average Adoption Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def generate_recommendations(self):
        """Generate strategic recommendations based on data analysis"""
        if self.clusters is None:
            self.segment_employees()
        
        recommendations = []
        
        # Analyze segments
        high_resistance_pct = (self.employees_df['segment'] == 'High Resistance').mean() * 100
        champion_pct = (self.employees_df['segment'] == 'AI Champions').mean() * 100
        
        if high_resistance_pct > 25:
            recommendations.append({
                'priority': 'High',
                'focus_area': 'Resistance Management',
                'recommendation': 'Implement job security guarantee program for 6 months',
                'expected_impact': 'Reduce resistance segment by 30% in 3 months'
            })
        
        if champion_pct < 15:
            recommendations.append({
                'priority': 'High',
                'focus_area': 'Champion Development',
                'recommendation': 'Create AI innovation rewards program',
                'expected_impact': 'Increase champion segment by 10% in 4 months'
            })
        
        # Department-specific analysis
        dept_adoption = self.employees_df.groupby('department')['adoption_score'].mean().sort_values()
        lowest_dept = dept_adoption.index[0]
        recommendations.append({
            'priority': 'Medium',
            'focus_area': f'{lowest_dept} Department Focus',
            'recommendation': f'Targeted AI use case workshop for {lowest_dept}',
            'expected_impact': f'Increase {lowest_dept} adoption score by 20 points in 2 months'
        })
        
        # Age-based analysis
        age_corr = self.employees_df['age'].corr(self.employees_df['adoption_score'])
        if age_corr < -0.3:  # Negative correlation indicates older employees adopt less
            recommendations.append({
                'priority': 'Medium',
                'focus_area': 'Age-Related Adoption Gap',
                'recommendation': 'Implement reverse mentoring program pairing younger and older employees',
                'expected_impact': 'Reduce age-adoption correlation by 50% in 6 months'
            })
        
        # General recommendation
        recommendations.append({
            'priority': 'Low',
            'focus_area': 'Communication Strategy',
            'recommendation': 'Weekly AI success stories newsletter',
            'expected_impact': 'Improve sentiment scores by 15% across all segments'
        })
        
        return pd.DataFrame(recommendations)

# Example usage
if __name__ == "__main__":
    system = AIAdoptionSystem()
    employees = system.load_sample_data(size=200)
    print("\nSample Employee Data:")
    print(employees.head())
    
    scores = system.calculate_adoption_scores()
    print("\nAdoption Scores:")
    print(scores.head())
    
    segments = system.segment_employees()
    print("\nEmployee Segments:")
    print(segments)
    
    # Example intervention plan
    plan = system.generate_intervention_plan(employee_id=1001)
    print("\nIntervention Plan for Employee 1001:")
    print(plan)
    
    # Visualize segments
    system.visualize_segments()
    
    # Visualize adoption trend
    system.visualize_adoption_trend()
    
    # Generate recommendations
    recommendations = system.generate_recommendations()
    print("\nStrategic Recommendations:")
    print(recommendations)
