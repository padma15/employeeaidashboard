# app.py - Flask application for AI Adoption System
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Required for saving plots in web environment
import matplotlib.pyplot as plt
from ai_adoption_system import AIAdoptionSystem  # Import your system

# Initialize Flask app
app = Flask(__name__)

# Initialize our system
system = AIAdoptionSystem()

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    """Load data - either sample or from uploaded file"""
    if 'file' not in request.files:
        # Load sample data
        size = int(request.form.get('size', 100))
        employees = system.load_sample_data(size=size)
        return jsonify({
            'success': True,
            'message': f'Loaded sample data with {size} employees',
            'preview': employees.head().to_dict(orient='records')
        })
    else:
        # TODO: Handle file upload and parsing
        return jsonify({
            'success': False,
            'message': 'File upload not implemented yet'
        })

@app.route('/calculate_scores', methods=['POST'])
def calculate_scores():
    """Calculate adoption scores for employees"""
    try:
        scores = system.calculate_adoption_scores()
        return jsonify({
            'success': True,
            'message': 'Calculated adoption scores successfully',
            'preview': scores.head().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error calculating scores: {str(e)}'
        })

@app.route('/segment_employees', methods=['POST'])
def segment_employees():
    """Segment employees into groups"""
    try:
        n_clusters = int(request.form.get('n_clusters', 4))
        segments = system.segment_employees(n_clusters=n_clusters)
        
        # Convert to dictionary for JSON response
        segments_dict = segments.reset_index().to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'message': 'Segmented employees successfully',
            'segments': segments_dict
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error segmenting employees: {str(e)}'
        })

@app.route('/get_intervention_plan', methods=['POST'])
def get_intervention_plan():
    """Get intervention plan for employee or segment"""
    try:
        employee_id = request.form.get('employee_id')
        segment = request.form.get('segment')
        
        if employee_id:
            employee_id = int(employee_id)
            plan = system.generate_intervention_plan(employee_id=employee_id)
        elif segment:
            plan = system.generate_intervention_plan(segment=segment)
        else:
            return jsonify({
                'success': False,
                'message': 'Must provide either employee_id or segment'
            })
        
        return jsonify({
            'success': True,
            'message': f'Generated intervention plan',
            'plan': plan
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating plan: {str(e)}'
        })

@app.route('/visualize_segments', methods=['GET'])
def visualize_segments():
    """Generate and return segment visualization"""
    try:
        # Create plot in memory
        plt.figure(figsize=(10, 6))
        if system.clusters is None:
            system.segment_employees()
        
        system.clusters['count'].plot(kind='bar', color='skyblue')
        plt.title('Employee Segments by Size', fontsize=14)
        plt.ylabel('Number of Employees')
        plt.xlabel('Segment')
        
        # Save plot to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Return image as base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{image_base64}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating visualization: {str(e)}'
        })

@app.route('/visualize_trend', methods=['GET'])
def visualize_trend():
    """Generate and return adoption trend visualization"""
    try:
        # Create plot in memory
        weeks = int(request.args.get('weeks', 12))
        trend_data = system.track_adoption_over_time(weeks=weeks)
        
        plt.figure(figsize=(12, 6))
        
        for segment in trend_data['segment'].unique():
            segment_data = trend_data[trend_data['segment'] == segment]
            plt.plot(range(len(segment_data['date'].unique())), 
                    segment_data['avg_adoption_score'], 
                    marker='o', label=segment)
        
        plt.axvline(x=4, color='red', linestyle='--', 
                   alpha=0.5, label='Intervention')
        
        plt.title('AI Adoption Trend by Segment', fontsize=14)
        plt.xlabel('Weeks')
        plt.ylabel('Average Adoption Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Return image as base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{image_base64}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating visualization: {str(e)}'
        })

@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    """Get strategic recommendations"""
    try:
        recommendations = system.generate_recommendations()
        return jsonify({
            'success': True,
            'recommendations': recommendations.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating recommendations: {str(e)}'
        })

@app.route('/export_data', methods=['GET'])
def export_data():
    """Export employee data with segments and scores as CSV"""
    try:
        if system.employees_df is None:
            return jsonify({
                'success': False,
                'message': 'No data available to export'
            })
        
        # Create in-memory CSV
        output = io.StringIO()
        system.employees_df.to_csv(output, index=False)
        
        # Create response with CSV file
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=employee_ai_adoption.csv'}
        )
        
        return response
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error exporting data: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)
