import pandas as pd
import random

# Setting a random seed for reproducibility
random.seed(42)

# Generating sample data
data = {
    "age": [random.randint(22, 60) for _ in range(100)],
    "ai_knowledge": [random.choice(['Low', 'Medium', 'High']) for _ in range(100)],
    "concern_level": [random.choice(['Low', 'Medium', 'High']) for _ in range(100)],
    "department": [random.choice(['HR', 'Engineering', 'Sales', 'Marketing', 'Finance', 'Operations']) for _ in range(100)],
    "employee_id": [f"E{str(i).zfill(4)}" for i in range(1, 101)],
    "previous_tech_score": [random.randint(1, 10) for _ in range(100)],
    "role": [random.choice(['Manager', 'Senior', 'Junior', 'Lead', 'Intern']) for _ in range(100)],
    "sentiment_score": [random.uniform(1.0, 5.0) for _ in range(100)],
    "tenure_years": [random.randint(1, 20) for _ in range(100)],
    "tool_usage_frequency": [random.choice(['Rarely', 'Occasionally', 'Frequently', 'Always']) for _ in range(100)],
    "training_attendance": [random.choice([True, False]) for _ in range(100)],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Saving to CSV
csv_file_path = 'employee_ai_adoption_data.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file created at: {csv_file_path}")

