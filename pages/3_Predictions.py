"""
Predictions Page for the CA Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import generate_student_data, generate_school_names
from utils.model_utils import make_prediction, load_model
from config import DATA_CONFIG, MODEL_CONFIG, DROPDOWN_OPTIONS, DEFAULT_VALUES
from utils.visualizations import (
    plot_bubble_chart,
    plot_heatmap,
    plot_stacked_bar,
    get_color_scale_for_risk
)

# Set page config
st.set_page_config(
    page_title="Predictions - CA Prediction System",
    page_icon="📊",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
<style>
    .risk-high {
        color: #ff6b6b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffd43b;
        font-weight: bold;
    }
    .risk-low {
        color: #51cf66;
        font-weight: bold;
    }
    .prediction-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .student-details {
        margin-top: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .intervention-suggestion {
        margin-top: 15px;
        padding: 10px;
        background-color: #e3f2fd;
        border-left: 4px solid #4e89ae;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def display_prepare_data_tab():
    """
    Display the prepare current data tab
    """
    st.subheader("Current Year Data")
    
    # Data source options
    data_source = st.radio(
        "Select Data Source",
        options=["Generated Current Year Data", "Generate New Data", "Upload Data"],
        index=0,
        key="prediction_data_source"
    )
    
    # Default prediction data
    prediction_data = None
    
    if data_source == "Generated Current Year Data":
        # Check if current year data exists in session state
        if "current_data" in st.session_state:
            prediction_data = st.session_state["current_data"]
            st.success(f"Using generated current year data with {len(prediction_data)} records")
        else:
            st.warning("No current year data found. Please generate data using the 'Generate New Data' option below.")
            return None
    elif data_source == "Generate New Data":
        # Generate new current year data
        st.subheader("Generate Current Year Data")
        
        with st.expander("Data Generation Options", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_students = st.number_input(
                    "Number of Students",
                    min_value=DATA_CONFIG["min_students"],
                    max_value=DATA_CONFIG["max_students"],
                    value=DEFAULT_VALUES["num_students"],
                    step=100,
                    key="current_gen_num_students"
                )
            
            with col2:
                # Use current year as default
                current_year = DATA_CONFIG["max_year"]
                st.number_input(
                    "Academic Year",
                    min_value=DATA_CONFIG["min_year"],
                    max_value=DATA_CONFIG["max_year"],
                    value=current_year,
                    step=1,
                    key="current_gen_year",
                    disabled=True,
                    help="Current year data is always for the current academic year"
                )
            
            with col3:
                num_schools = st.number_input(
                    "Number of Schools",
                    min_value=1,
                    max_value=50,
                    value=DEFAULT_VALUES["num_schools"],
                    step=1,
                    key="current_gen_num_schools"
                )
                
            # School name pattern
            school_base_name = st.text_input(
                "School Name Pattern",
                value="School-",
                help="The base name for schools (e.g., '10U', 'A-SCHOOL')",
                key="current_gen_school_base_name"
            )
            
            # School names
            school_names = generate_school_names(school_base_name, num_schools)
            st.write("Generated School Names:", ", ".join(school_names))
            
            # Generate button
            if st.button("Generate Current Year Data", key="generate_current_data_btn"):
                with st.spinner("Generating current year data..."):
                    try:
                        # Generate data with is_historical=False for current year
                        current_data = generate_student_data(
                            num_students=num_students,
                            academic_years=[current_year],
                            schools=school_names,
                            num_schools=num_schools,
                            school_base_name=school_base_name,
                            is_historical=False
                        )
                        
                        # Store in session state
                        st.session_state["current_data"] = current_data
                        
                        # Set as prediction data
                        prediction_data = current_data
                        
                        # Show success message with animation
                        st.success(f"Successfully generated {len(current_data)} student records for the current academic year!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error generating data: {str(e)}")
        
    else:
        # File uploader for prediction data
        uploaded_file = st.file_uploader(
            "Upload current year data (CSV or Excel)",
            type=["csv", "xlsx"],
            key="prediction_data_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Determine file type
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                # Read the file
                if file_extension == "csv":
                    prediction_data = pd.read_csv(uploaded_file)
                elif file_extension == "xlsx":
                    prediction_data = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded data with {len(prediction_data)} records")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None
        else:
            st.info("Please upload a data file")
            return None
    
    # Ensure we have prediction data
    if prediction_data is None:
        return None
    
    # Display data
    st.subheader("Data Preview")
    st.dataframe(prediction_data.head(5), use_container_width=True)
    
    # Data validation
    st.subheader("Data Validation")
    
    # Check if there's a trained model
    if "best_model" not in st.session_state:
        st.warning("No trained model found. Please train a model first.")
        return None
    
    # Get the required features from the best model
    best_model = st.session_state["best_model"]
    
    # Initialize model feature selections if they don't exist
    if "model_selected_categorical" not in st.session_state:
        st.session_state["model_selected_categorical"] = []
    if "model_selected_numerical" not in st.session_state:
        st.session_state["model_selected_numerical"] = []
        
    model_categorical = st.session_state["model_selected_categorical"]
    model_numerical = st.session_state["model_selected_numerical"]
    
    # Check if all required features are present
    missing_features = []
    for feature in model_categorical + model_numerical:
        if feature not in prediction_data.columns:
            missing_features.append(feature)
    
    if missing_features:
        st.error(f"The following features are missing in the prediction data: {', '.join(missing_features)}")
        
        # Suggest matching the data structure
        st.info("Please make sure your prediction data has the same structure as the training data.")
        return None
    
    # Store prediction data in session state
    st.session_state["prediction_data"] = prediction_data
    
    # Data validation success
    st.success("Data validation successful! The prediction data is compatible with the trained model.")
    
    # Show data statistics
    st.subheader("Data Statistics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(prediction_data))
    col2.metric("Schools", prediction_data["school"].nunique() if "school" in prediction_data.columns else "N/A")
    col3.metric("Grades", prediction_data["grade"].nunique() if "grade" in prediction_data.columns else "N/A")
    
    return prediction_data

def display_patterns_dashboard():
    """
    Display the patterns and correlations dashboard
    """
    st.subheader("Patterns & Correlations Dashboard")
    
    # Check if there's historical data
    if "historical_data" not in st.session_state:
        st.info("No historical data found. Patterns analysis requires historical data.")
        return
    
    historical_data = st.session_state["historical_data"]
    
    # Display historical patterns
    st.markdown("### Historical Attendance Patterns")
    
    # Create tabs for different pattern analyses
    pattern_tabs = st.tabs([
        "Attendance by Grade", 
        "Attendance by School", 
        "Gender Distribution", 
        "Special Needs Impact"
    ])
    
    # Attendance by Grade
    with pattern_tabs[0]:
        # Create a stacked bar chart of CA status by grade
        if "grade" in historical_data.columns and "ca_status" in historical_data.columns:
            grade_fig = plot_stacked_bar(historical_data, x="grade", color="ca_status")
            st.plotly_chart(grade_fig, use_container_width=True)
            
            st.markdown("""
            **Pattern Analysis:** This chart shows the distribution of Chronic Absenteeism (CA) across different grades.
            - Higher CA percentages in certain grades may indicate transition challenges
            - Grade levels with significantly higher CA rates may need targeted interventions
            """)
    
    # Attendance by School
    with pattern_tabs[1]:
        # Create a heatmap of attendance percentage by school and grade
        if "school" in historical_data.columns and "grade" in historical_data.columns and "attendance_percentage" in historical_data.columns:
            school_fig = plot_heatmap(historical_data, x="grade", y="school", values="attendance_percentage")
            st.plotly_chart(school_fig, use_container_width=True)
            
            st.markdown("""
            **Pattern Analysis:** This heatmap shows average attendance percentages across schools and grades.
            - Darker areas indicate lower attendance percentages
            - Schools with consistently low attendance may need system-wide interventions
            - Specific grade-school combinations with low attendance may need targeted support
            """)
    
    # Gender Distribution
    with pattern_tabs[2]:
        # Create a stacked bar chart of CA status by gender
        if "gender" in historical_data.columns and "ca_status" in historical_data.columns:
            gender_fig = plot_stacked_bar(historical_data, x="gender", color="ca_status")
            st.plotly_chart(gender_fig, use_container_width=True)
            
            st.markdown("""
            **Pattern Analysis:** This chart shows the distribution of Chronic Absenteeism (CA) across different genders.
            - Gender-specific attendance patterns may help tailor interventions
            - Significant differences may indicate social or cultural factors affecting attendance
            """)
    
    # Special Needs Impact
    with pattern_tabs[3]:
        # Create a stacked bar chart of CA status by special needs status
        if "special_need" in historical_data.columns and "ca_status" in historical_data.columns:
            special_needs_fig = plot_stacked_bar(historical_data, x="special_need", color="ca_status")
            st.plotly_chart(special_needs_fig, use_container_width=True)
            
            st.markdown("""
            **Pattern Analysis:** This chart shows the impact of special needs status on Chronic Absenteeism (CA).
            - Students with special needs may have different attendance patterns
            - This can help identify if additional support services are needed
            """)
    
    # Pattern identification and editing
    st.subheader("Pattern Configuration")
    
    # Initialize patterns if not already in session state
    if "identified_patterns" not in st.session_state:
        # Default patterns based on common findings
        st.session_state["identified_patterns"] = [
            {
                "name": "Grade Transition",
                "description": "Students tend to have higher absenteeism during grade transition years (e.g., entering middle or high school)",
                "enabled": True
            },
            {
                "name": "Special Needs Support",
                "description": "Students with special needs have higher CA rates when support services are insufficient",
                "enabled": True
            },
            {
                "name": "Transportation Issues",
                "description": "Students with long bus trips have higher absence rates, especially during winter months",
                "enabled": True
            },
            {
                "name": "Academic Performance Correlation",
                "description": "Lower academic performance is correlated with higher absenteeism rates",
                "enabled": True
            }
        ]
    
    # Display and edit patterns
    for i, pattern in enumerate(st.session_state["identified_patterns"]):
        col1, col2 = st.columns([1, 10])
        
        with col1:
            # Enable/disable pattern
            pattern_enabled = st.checkbox("", value=pattern["enabled"], key=f"pattern_enabled_{i}")
            st.session_state["identified_patterns"][i]["enabled"] = pattern_enabled
        
        with col2:
            # Display pattern name and description
            st.markdown(f"**{pattern['name']}**")
            st.markdown(pattern["description"])
    
    # Add new pattern
    with st.expander("Add New Pattern"):
        with st.form(key="add_pattern_form"):
            pattern_name = st.text_input("Pattern Name")
            pattern_description = st.text_area("Pattern Description")
            
            submit_button = st.form_submit_button("Add Pattern")
            
            if submit_button and pattern_name and pattern_description:
                st.session_state["identified_patterns"].append({
                    "name": pattern_name,
                    "description": pattern_description,
                    "enabled": True
                })
                st.success(f"Added new pattern: {pattern_name}")

def display_run_prediction_tab():
    """
    Display the run prediction tab
    """
    st.subheader("Run Prediction")
    
    # Check if prediction data is available
    if "prediction_data" not in st.session_state:
        st.warning("No prediction data available. Please prepare data first.")
        return
    
    # Check if a trained model is available
    if "best_model" not in st.session_state or "best_model_pipeline" not in st.session_state:
        st.warning("No trained model available. Please train a model first.")
        return
    
    # Get prediction data and model
    prediction_data = st.session_state["prediction_data"]
    best_model = st.session_state["best_model"]
    best_model_pipeline = st.session_state["best_model_pipeline"]
    
    # Model information
    st.markdown(f"### Model: {MODEL_CONFIG['models'][best_model]}")
    
    metrics = st.session_state["best_model_metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
    col2.metric("F1 Score", f"{metrics.get('f1', 0)*100:.2f}%")
    col3.metric("Model Type", MODEL_CONFIG['models'][best_model])
    
    # Prediction button
    if st.button("Predict CA Risk", key="run_prediction_button"):
        with st.spinner("Running predictions..."):
            try:
                # Get features
                model_categorical = st.session_state["model_selected_categorical"]
                model_numerical = st.session_state["model_selected_numerical"]
                
                # Select only the features needed for prediction
                features = model_categorical + model_numerical
                X_pred = prediction_data[features]
                
                # Make predictions
                predictions, probabilities = make_prediction(best_model_pipeline, X_pred)
                
                # Convert binary predictions to CA/No-CA if needed
                if np.issubdtype(predictions.dtype, np.number):
                    prediction_labels = np.where(predictions == 1, "CA", "No-CA")
                else:
                    prediction_labels = predictions
                
                # Add predictions to the data
                result_data = prediction_data.copy()
                result_data["predicted_ca_status"] = prediction_labels
                
                if probabilities is not None:
                    result_data["ca_risk_score"] = probabilities
                
                # Store prediction results in session state
                st.session_state["prediction_results"] = result_data
                
                # Success message
                st.success("Predictions completed successfully!")
                
                # Display summary
                ca_count = sum(prediction_labels == "CA")
                ca_percentage = (ca_count / len(prediction_labels)) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Predicted CA Students", ca_count)
                col2.metric("Predicted CA Percentage", f"{ca_percentage:.2f}%")
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

def display_single_student_dashboard(student_data):
    """
    Display a detailed dashboard for a single student
    
    Args:
        student_data (pd.Series): Data for a single student
    """
    st.markdown("""
    <div style="background-color: #f0f7fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    """, unsafe_allow_html=True)
    
    # Student header with ID and basic info
    st.markdown(f"""
    <h3 style="color: #1E3A8A; margin-bottom: 15px;">
        Student Analysis: {student_data.get('student_id', 'No ID')}
    </h3>
    """, unsafe_allow_html=True)
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <h4 style="margin: 0; font-size: 0.9rem; color: #6B7280;">Grade</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{student_data.get('grade', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <h4 style="margin: 0; font-size: 0.9rem; color: #6B7280;">School</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{student_data.get('school', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <h4 style="margin: 0; font-size: 0.9rem; color: #6B7280;">Academic Year</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{student_data.get('academic_year', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction result with gauge chart
    st.subheader("Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    # CA status
    ca_status = student_data.get('predicted_ca_status', 'Unknown')
    risk_score = student_data.get('ca_risk_score', 0)
    
    with col1:
        # Display CA status with appropriate styling
        if ca_status == "CA":
            st.markdown(f"""
            <div style="background-color: #FEE2E2; color: #DC2626; padding: 15px; 
                        border-radius: 5px; text-align: center; margin-bottom: 10px;">
                <h3 style="margin: 0;">High Risk - CA</h3>
                <p style="font-size: 0.9rem; margin: 5px 0;">
                    This student is predicted to have Chronic Absenteeism
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #DCFCE7; color: #16A34A; padding: 15px; 
                        border-radius: 5px; text-align: center; margin-bottom: 10px;">
                <h3 style="margin: 0;">Low Risk - No CA</h3>
                <p style="font-size: 0.9rem; margin: 5px 0;">
                    This student is not predicted to have Chronic Absenteeism
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Create a gauge chart for risk score
        if isinstance(risk_score, (int, float)):
            risk_level = risk_score
        else:
            # If risk_score is a probability series, get the value for CA class
            risk_level = risk_score[1] if len(risk_score) > 1 else 0.5
        
        # Determine color based on risk level
        if risk_level < 0.33:
            color = "green"
        elif risk_level < 0.67:
            color = "orange"
        else:
            color = "red"
            
        # Create gauge chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_level * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(0, 250, 0, 0.1)'},
                    {'range': [33, 67], 'color': 'rgba(255, 165, 0, 0.1)'},
                    {'range': [67, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        gauge_fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=50, b=10),
            font={'color': "#1E3A8A", 'family': "Arial"}
        )
        
        # Use a unique key combining student ID and a random suffix to avoid duplicates
        import random
        unique_suffix = random.randint(1000, 9999)
        st.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{student_data.get('student_id', 'unknown')}_{unique_suffix}")
    
    # Key Indicators section
    st.subheader("Key Indicators")
    
    # Create 3 columns for the indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Academic Performance gauge
        academic_performance = student_data.get('academic_performance', 0)
        academic_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=academic_performance,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Academic Performance", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 60], 'color': 'lightgray'},
                    {'range': [60, 80], 'color': 'lightblue'},
                    {'range': [80, 100], 'color': 'lightgreen'}
                ]
            }
        ))
        
        academic_fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(academic_fig, use_container_width=True, key=f"academic_{student_data.get('student_id', 'unknown')}_{random.randint(1000, 9999)}")
    
    with col2:
        # Attendance Percentage gauge
        attendance_percentage = student_data.get('attendance_percentage', 0)
        attendance_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attendance_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Attendance Percentage", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 80], 'color': 'lightgray'},
                    {'range': [80, 90], 'color': 'lightyellow'},
                    {'range': [90, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        attendance_fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(attendance_fig, use_container_width=True, key=f"attendance_{student_data.get('student_id', 'unknown')}_{random.randint(1000, 9999)}")
    
    with col3:
        # Absent Days gauge
        absent_days = student_data.get('absent_days', 0)
        max_absent = 40  # Assuming maximum absent days
        
        absent_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=absent_days,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Absent Days", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, max_absent], 'tickwidth': 1},
                'bar': {'color': "firebrick"},
                'steps': [
                    {'range': [0, 10], 'color': 'lightgreen'},
                    {'range': [10, 20], 'color': 'lightyellow'},
                    {'range': [20, max_absent], 'color': 'lightcoral'}
                ]
            }
        ))
        
        absent_fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(absent_fig, use_container_width=True, key=f"absent_{student_data.get('student_id', 'unknown')}_{random.randint(1000, 9999)}")
    
    # Intervention Recommendations
    st.subheader("Recommended Interventions")
    
    # Determine intervention categories based on student data
    interventions = []
    
    # Attendance intervention
    if attendance_percentage < 90:
        interventions.append({
            "category": "Attendance Support",
            "icon": "📅",
            "recommendations": [
                "Regular attendance check-ins with counselor",
                "Personalized attendance improvement plan",
                "Parent/guardian communication strategy"
            ]
        })
    
    # Academic intervention
    if academic_performance < 70:
        interventions.append({
            "category": "Academic Support",
            "icon": "📚",
            "recommendations": [
                "After-school tutoring program",
                "Study skills workshop",
                "Subject-specific intervention"
            ]
        })
    
    # Transportation intervention
    if student_data.get('bus_long_trip') == "Yes":
        interventions.append({
            "category": "Transportation Assistance",
            "icon": "🚌",
            "recommendations": [
                "Explore alternative transportation options",
                "Adjust bus route to reduce travel time",
                "Provide materials for productive use of travel time"
            ]
        })
    
    # Special needs intervention
    if student_data.get('special_need') == "Yes":
        interventions.append({
            "category": "Special Services Support",
            "icon": "🔍",
            "recommendations": [
                "Review and update IEP/504 plan",
                "Specialized attendance accommodations",
                "Additional support services evaluation"
            ]
        })
    
    # Housing/shelter intervention
    if student_data.get('shelter') in ["S", "ST"]:
        interventions.append({
            "category": "Housing Stability Support",
            "icon": "🏠",
            "recommendations": [
                "Connect with social services coordinator",
                "Provide stable learning environment resources",
                "Transportation assistance program"
            ]
        })
    
    # Behavioral intervention (if suspended)
    if student_data.get('suspended') == "Yes":
        interventions.append({
            "category": "Behavioral Support",
            "icon": "🤝",
            "recommendations": [
                "Behavioral intervention plan",
                "Counseling services",
                "Restorative practices implementation"
            ]
        })
    
    # Display interventions in a grid
    if interventions:
        # Create columns for interventions
        cols = st.columns(min(3, len(interventions)))
        
        for i, intervention in enumerate(interventions):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px; min-height: 200px;">
                    <h4 style="color: #1E3A8A; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">{intervention['icon']}</span>
                        {intervention['category']}
                    </h4>
                    <ul style="margin-top: 10px;">
                """, unsafe_allow_html=True)
                
                for rec in intervention['recommendations']:
                    st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
    else:
        st.info("No specific interventions recommended at this time.")
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_prediction_results_tab():
    """
    Display the prediction results tab
    """
    # Check if prediction results are available
    if "prediction_results" not in st.session_state:
        st.info("No prediction results available. Please run predictions first.")
        return
    
    # Get prediction results
    results = st.session_state["prediction_results"]
    
    # Summary statistics
    st.subheader("Prediction Summary")
    
    # Count CA and No-CA predictions
    ca_count = sum(results["predicted_ca_status"] == "CA")
    noca_count = sum(results["predicted_ca_status"] == "No-CA")
    total_count = len(results)
    
    ca_percentage = (ca_count / total_count) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", total_count)
    col2.metric("Predicted CA Students", ca_count)
    col3.metric("CA Percentage", f"{ca_percentage:.2f}%")
    
    # Add reset button for predictions
    if st.button("🔄 Reset Predictions", help="Clear all prediction results and start over"):
        if "prediction_results" in st.session_state:
            del st.session_state["prediction_results"]
        st.success("Predictions have been reset!")
        st.experimental_rerun()
    
    # Show results by school if available
    if "school" in results.columns:
        st.subheader("Results by School")
        
        # Group by school
        school_summary = results.groupby("school")["predicted_ca_status"].apply(
            lambda x: sum(x == "CA") / len(x) * 100
        ).reset_index()
        school_summary.columns = ["School", "CA Percentage"]
        
        # Sort by CA percentage (descending)
        school_summary = school_summary.sort_values("CA Percentage", ascending=False)
        
        # Create bar chart
        fig = px.bar(
            school_summary,
            x="School",
            y="CA Percentage",
            color="CA Percentage",
            color_continuous_scale="RdYlGn_r",
            title="Predicted CA Percentage by School"
        )
        
        fig.update_layout(xaxis_title="School", yaxis_title="CA Percentage (%)")
        
        st.plotly_chart(fig, use_container_width=True, key="school_summary_chart")
    
    # Show results by grade if available
    if "grade" in results.columns:
        st.subheader("Results by Grade")
        
        # Group by grade
        grade_summary = results.groupby("grade")["predicted_ca_status"].apply(
            lambda x: sum(x == "CA") / len(x) * 100
        ).reset_index()
        grade_summary.columns = ["Grade", "CA Percentage"]
        
        # Sort by grade
        grade_summary = grade_summary.sort_values("Grade")
        
        # Create bar chart
        fig = px.bar(
            grade_summary,
            x="Grade",
            y="CA Percentage",
            color="CA Percentage",
            color_continuous_scale="RdYlGn_r",
            title="Predicted CA Percentage by Grade"
        )
        
        fig.update_layout(xaxis_title="Grade", yaxis_title="CA Percentage (%)")
        
        st.plotly_chart(fig, use_container_width=True, key="grade_summary_chart")
    
    # Bubble chart of academic performance vs attendance
    if "academic_performance" in results.columns and "attendance_percentage" in results.columns:
        st.subheader("Academic Performance vs Attendance")
        
        bubble_fig = plot_bubble_chart(
            results,
            x="academic_performance",
            y="attendance_percentage",
            size="absent_days" if "absent_days" in results.columns else None,
            color="predicted_ca_status"
        )
        
        st.plotly_chart(bubble_fig, use_container_width=True, key="performance_attendance_bubble")
    
    # Display full results table
    st.subheader("Full Prediction Results")
    
    # Add risk category if risk score is available
    if "ca_risk_score" in results.columns:
        results["risk_category"] = results["ca_risk_score"].apply(
            lambda x: "High Risk" if x >= 0.75 else 
                     ("Medium Risk" if x >= 0.5 else 
                     ("Low-Medium Risk" if x >= 0.25 else "Low Risk"))
        )
    
    # Display the results table
    st.dataframe(results, use_container_width=True)
    
    # Download button for results
    csv = results.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ca_prediction_results.csv">Download Prediction Results as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Single student analysis
    st.subheader("Single Student Analysis")
    
    # Get student IDs if available
    if "student_id" in results.columns:
        student_ids = results["student_id"].tolist()
        
        # Select a student
        selected_student_id = st.selectbox(
            "Select Student ID",
            options=student_ids,
            key="select_student_analysis"
        )
        
        # Get student data
        student_data = results[results["student_id"] == selected_student_id].iloc[0]
        
        # Display the student dashboard
        display_single_student_dashboard(student_data)
        
        # Display student details
        with st.container():
            st.markdown("### Student Details")
            
            # Create columns for student information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Student ID:** {student_data['student_id']}")
                st.markdown(f"**School:** {student_data.get('school', 'N/A')}")
                st.markdown(f"**Grade:** {student_data.get('grade', 'N/A')}")
            
            with col2:
                st.markdown(f"**Gender:** {student_data.get('gender', 'N/A')}")
                st.markdown(f"**Meal Code:** {student_data.get('meal_code', 'N/A')}")
                st.markdown(f"**Special Needs:** {student_data.get('special_need', 'N/A')}")
            
            with col3:
                st.markdown(f"**Academic Performance:** {student_data.get('academic_performance', 'N/A'):.1f}%")
                st.markdown(f"**Attendance:** {student_data.get('attendance_percentage', 'N/A'):.1f}%")
                st.markdown(f"**Absent Days:** {student_data.get('absent_days', 'N/A')}")
            
            # Display prediction result
            st.markdown("### Prediction Result")
            
            # Risk level
            if "ca_risk_score" in student_data:
                risk_score = student_data["ca_risk_score"]
                risk_color_class = "risk-high" if risk_score >= 0.75 else ("risk-medium" if risk_score >= 0.5 else "risk-low")
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>CA Prediction: <span class="{risk_color_class}">{student_data['predicted_ca_status']}</span></h4>
                    <h4>Risk Score: <span class="{risk_color_class}">{risk_score:.2f}</span></h4>
                    <h4>Risk Category: <span class="{risk_color_class}">{student_data.get('risk_category', 'N/A')}</span></h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>CA Prediction: {student_data['predicted_ca_status']}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Contributing factors
            st.markdown("### Contributing Factors")
            
            # Get the enabled patterns
            if "identified_patterns" in st.session_state:
                enabled_patterns = [p for p in st.session_state["identified_patterns"] if p["enabled"]]
                
                if enabled_patterns:
                    st.markdown("The following patterns may be contributing to this student's absenteeism risk:")
                    
                    for pattern in enabled_patterns:
                        # Check if pattern applies to this student
                        applies = False
                        
                        # Example logic to determine if pattern applies
                        if pattern["name"] == "Grade Transition" and student_data.get("grade") in [6, 9]:
                            applies = True
                        elif pattern["name"] == "Special Needs Support" and student_data.get("special_need") == "Yes":
                            applies = True
                        elif pattern["name"] == "Transportation Issues" and student_data.get("bus_long_trip") == "Yes":
                            applies = True
                        elif pattern["name"] == "Academic Performance Correlation" and student_data.get("academic_performance", 100) < 70:
                            applies = True
                        
                        if applies:
                            st.markdown(f"- **{pattern['name']}**: {pattern['description']}")
            
            # Intervention recommendations
            st.markdown("### Intervention Recommendations")
            
            if student_data["predicted_ca_status"] == "CA" or student_data.get("ca_risk_score", 0) >= 0.5:
                st.markdown("""
                <div class="intervention-suggestion">
                    <h4>Recommended Interventions:</h4>
                    <ul>
                        <li><strong>Attendance Monitoring:</strong> Implement daily attendance tracking with immediate follow-up on absences</li>
                        <li><strong>Parent/Guardian Conference:</strong> Schedule meeting to discuss attendance concerns and create an improvement plan</li>
                        <li><strong>Academic Support:</strong> Provide additional tutoring or academic assistance to address performance gaps</li>
                        <li><strong>Social-Emotional Support:</strong> Connect student with counseling services or mentoring program</li>
                        <li><strong>Incentive Program:</strong> Establish rewards for improved attendance</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Add specific recommendations based on student factors
                if student_data.get("bus_long_trip") == "Yes":
                    st.markdown("""
                    <div class="intervention-suggestion">
                        <h4>Transportation Intervention:</h4>
                        <p>This student has a long bus trip which may be contributing to absences. Consider:</p>
                        <ul>
                            <li>Alternative transportation options</li>
                            <li>Modified schedule to accommodate transportation challenges</li>
                            <li>Remote learning options for days with transportation difficulties</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                if student_data.get("special_need") == "Yes":
                    st.markdown("""
                    <div class="intervention-suggestion">
                        <h4>Special Needs Support:</h4>
                        <p>Ensure appropriate accommodations are in place:</p>
                        <ul>
                            <li>Review and update IEP if applicable</li>
                            <li>Check for classroom accommodations that may improve attendance</li>
                            <li>Coordinate with special education team for additional support</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="intervention-suggestion">
                    <h4>Preventive Measures:</h4>
                    <p>This student is not currently at high risk for chronic absenteeism, but consider these preventive measures:</p>
                    <ul>
                        <li>Regular attendance monitoring</li>
                        <li>Positive reinforcement for good attendance</li>
                        <li>Include in school-wide attendance initiatives</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Student ID column not found in data. Single student analysis is not available.")

def main():
    # Custom CSS for animations and styling
    st.markdown("""
    <style>
        @keyframes slideInFromLeft {
            0% {
                transform: translateX(-30px);
                opacity: 0;
            }
            100% {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .prediction-banner {
            background: linear-gradient(135deg, #4338CA, #3B82F6);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            animation: slideInFromLeft 0.8s ease-out;
        }
        
        .prediction-banner h1 {
            font-size: 2.5rem;
            margin: 0;
            padding: 0;
        }
        
        .prediction-banner p {
            font-size: 1.1rem;
            margin-top: 10px;
            opacity: 0.9;
        }
        
        .prediction-icon {
            float: left;
            margin-right: 20px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
            }
        }
        
        /* Button effects */
        .stButton > button {
            background: linear-gradient(90deg, #3B82F6, #4338CA);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #4338CA, #3730A3);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Tab animations */
        .stTabs [data-baseweb="tab-panel"] {
            animation: fadeIn 0.6s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Animated banner
    st.markdown("""
    <div class="prediction-banner">
        <div class="prediction-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="white">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>
        </div>
        <h1>Chronic Absenteeism Predictions</h1>
        <p>Predict at-risk students and visualize patterns using trained machine learning models</p>
        <div style="clear: both;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load icon (but hide it since we're using our custom banner)
    st.image("assets/absenteeism_icon.svg", width=0)
    
    # Tabs for different sections
    tabs = st.tabs([
        "Prepare Current Data", 
        "Patterns & Correlations", 
        "Run Prediction", 
        "Prediction Results",
        "Student Analysis"
    ])
    
    # Prepare Current Data tab
    with tabs[0]:
        display_prepare_data_tab()
    
    # Patterns & Correlations tab
    with tabs[1]:
        display_patterns_dashboard()
    
    # Run Prediction tab
    with tabs[2]:
        display_run_prediction_tab()
    
    # Prediction Results tab
    with tabs[3]:
        display_prediction_results_tab()
    
    # Student Analysis tab
    with tabs[4]:
        st.subheader("Student Analysis Dashboard")
        
        # Check if prediction results are available
        if "prediction_results" not in st.session_state:
            st.warning("No prediction results available. Please run predictions first.")
        else:
            results_data = st.session_state["prediction_results"]
            
            # Filter controls
            st.markdown("### Filter Students")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filter by school
                schools = sorted(results_data["school"].unique()) if "school" in results_data.columns else []
                selected_school = st.selectbox(
                    "School",
                    options=["All Schools"] + schools,
                    key="student_analysis_school"
                )
            
            with col2:
                # Filter by grade
                grades = sorted(results_data["grade"].unique()) if "grade" in results_data.columns else []
                selected_grade = st.selectbox(
                    "Grade",
                    options=["All Grades"] + grades,
                    key="student_analysis_grade"
                )
            
            with col3:
                # Filter by prediction
                prediction_options = ["All Students", "Predicted CA", "Predicted No-CA"]
                selected_prediction = st.selectbox(
                    "Prediction",
                    options=prediction_options,
                    key="student_analysis_prediction"
                )
            
            # Apply filters
            filtered_data = results_data.copy()
            
            if selected_school != "All Schools":
                filtered_data = filtered_data[filtered_data["school"] == selected_school]
                
            if selected_grade != "All Grades":
                filtered_data = filtered_data[filtered_data["grade"] == selected_grade]
                
            if selected_prediction == "Predicted CA":
                filtered_data = filtered_data[filtered_data["predicted_ca_status"] == "CA"]
            elif selected_prediction == "Predicted No-CA":
                filtered_data = filtered_data[filtered_data["predicted_ca_status"] == "No-CA"]
            
            # Display student count
            st.metric("Filtered Students", len(filtered_data))
            
            # Student list
            st.markdown("### Student List")
            
            # Sort options
            sort_options = {
                "Student ID": "student_id",
                "Risk Score (Highest First)": "ca_risk_score" if "ca_risk_score" in filtered_data.columns else None,
                "Attendance": "attendance_percentage" if "attendance_percentage" in filtered_data.columns else None,
                "Academic Performance": "academic_performance" if "academic_performance" in filtered_data.columns else None
            }
            
            # Remove None values
            sort_options = {k: v for k, v in sort_options.items() if v is not None}
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Search by student ID
                search_id = st.text_input("Search by Student ID", key="student_analysis_search")
            
            with col2:
                # Sort by
                sort_by = st.selectbox(
                    "Sort By",
                    options=list(sort_options.keys()),
                    key="student_analysis_sort"
                )
            
            # Apply search filter
            if search_id:
                filtered_data = filtered_data[filtered_data["student_id"].str.contains(search_id, case=False)]
            
            # Apply sorting
            sort_column = sort_options[sort_by]
            ascending = True if sort_by != "Risk Score (Highest First)" else False
            filtered_data = filtered_data.sort_values(by=sort_column, ascending=ascending)
            
            # Display students in an interactive table
            st.dataframe(
                filtered_data[["student_id", "school", "grade", "predicted_ca_status"] + 
                             (["ca_risk_score"] if "ca_risk_score" in filtered_data.columns else []) +
                             (["attendance_percentage"] if "attendance_percentage" in filtered_data.columns else []) +
                             (["academic_performance"] if "academic_performance" in filtered_data.columns else [])],
                use_container_width=True
            )
            
            # Individual student analysis
            st.markdown("### Individual Student Analysis")
            
            # Select a student
            selected_student_id = st.selectbox(
                "Select a student to analyze",
                options=filtered_data["student_id"].tolist(),
                key="student_analysis_selected_student"
            )
            
            if selected_student_id:
                # Get student data
                student_data = filtered_data[filtered_data["student_id"] == selected_student_id].iloc[0]
                
                # Display student dashboard
                display_single_student_dashboard(student_data)
                
                # Intervention recommendations
                st.markdown("### Recommended Interventions")
                
                # Based on risk factors
                risk_factors = []
                
                # Attendance patterns
                if "attendance_percentage" in student_data and student_data["attendance_percentage"] < 90:
                    risk_factors.append("Low attendance rate")
                    
                # Academic performance
                if "academic_performance" in student_data and student_data["academic_performance"] < 70:
                    risk_factors.append("Low academic performance")
                    
                # Special needs
                if "special_need" in student_data and student_data["special_need"] == "Yes":
                    risk_factors.append("Special educational needs")
                    
                # Transportation issues
                if "bus_long_trip" in student_data and student_data["bus_long_trip"] == "Yes":
                    risk_factors.append("Long commute/transportation issues")
                    
                # Shelter
                if "shelter" in student_data and student_data["shelter"] == "Yes":
                    risk_factors.append("Housing instability")
                    
                # Generate intervention recommendations
                interventions = []
                
                if "Low attendance rate" in risk_factors:
                    interventions.append({
                        "title": "Attendance Monitoring",
                        "description": "Daily attendance check-ins and weekly progress reports",
                        "category": "Monitoring"
                    })
                    
                if "Low academic performance" in risk_factors:
                    interventions.append({
                        "title": "Academic Support",
                        "description": "Tutoring services and supplemental instruction",
                        "category": "Academic"
                    })
                    
                if "Special educational needs" in risk_factors:
                    interventions.append({
                        "title": "Specialized Support Services",
                        "description": "Custom educational plan with specialized resources",
                        "category": "Support"
                    })
                    
                if "Long commute/transportation issues" in risk_factors:
                    interventions.append({
                        "title": "Transportation Assistance",
                        "description": "Alternative transportation arrangements or schedule adjustments",
                        "category": "Logistics"
                    })
                    
                if "Housing instability" in risk_factors:
                    interventions.append({
                        "title": "Family Support Services",
                        "description": "Connect with social services and housing assistance programs",
                        "category": "Support"
                    })
                    
                # Add general interventions if needed
                if not interventions:
                    interventions.append({
                        "title": "Preventive Check-ins",
                        "description": "Regular monitoring to ensure continued attendance",
                        "category": "Monitoring"
                    })
                
                # Display interventions
                for intervention in interventions:
                    st.markdown(f"""
                    <div style="background-color: #f0f7ff; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #3366cc;">
                        <h4 style="color: #1E3A8A; margin-top: 0;">{intervention["title"]} <span style="font-size: 0.8em; background-color: #e6f0ff; padding: 3px 8px; border-radius: 10px;">{intervention["category"]}</span></h4>
                        <p style="margin-bottom: 0;">{intervention["description"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Notes section
                st.markdown("### Student Notes")
                notes = st.text_area(
                    "Add notes about this student",
                    key=f"student_notes_{selected_student_id}",
                    height=100
                )
                
                if st.button("Save Notes", key=f"save_notes_{selected_student_id}"):
                    # In a real app, we would save these notes to a database
                    # For now, we'll just show a success message
                    st.success("Notes saved successfully!")
    
    # Documentation
    with st.expander("Documentation: Making Predictions"):
        st.markdown("""
        ## Predictions Module Documentation

        This module allows you to apply trained machine learning models to predict chronic absenteeism (CA) risk for current year students.

        ### Workflow

        1. **Prepare Current Data**
            - Upload or use generated current year student data
            - The system validates that the data structure matches the requirements of the trained model
            - Data statistics are displayed to provide an overview

        2. **Patterns & Correlations**
            - View historical attendance patterns across different factors (grade, school, gender, etc.)
            - Configure which patterns to consider in predictions and interventions
            - Add new patterns based on your domain knowledge

        3. **Run Prediction**
            - Select a trained model to use for predictions
            - Run the prediction process on the current year data
            - View a summary of prediction results

        4. **Prediction Results**
            - Explore detailed prediction results with visualizations
            - Analyze CA risk by school, grade, and other factors
            - Examine individual student predictions and risk levels
            - Get recommended interventions based on student-specific factors
            - Download the complete prediction results for further analysis
            
        5. **Student Analysis**
            - Filter and search for specific students based on various criteria
            - View detailed analysis of individual student risk factors
            - Get personalized intervention recommendations for each student
            - Add and save notes about individual students for tracking progress

        ### Key Features

        - **Risk Scoring**: Students receive a risk score (0-1) indicating their likelihood of chronic absenteeism
        - **Pattern Analysis**: Historical patterns inform predictions and intervention recommendations
        - **Visualization**: Interactive charts help identify trends and correlations
        - **Single Student Analysis**: Detailed view of individual student risk factors and recommended interventions
        - **Exportable Results**: Save prediction results for use in other systems or reports

        ### Tips for Making Predictions

        - Ensure your current year data has the same structure as the training data
        - Review historical patterns to understand the factors affecting attendance
        - Pay attention to both CA prediction and risk score to prioritize interventions
        - Use the single student analysis to develop individualized support plans
        - Compare predictions across different demographic groups to identify systemic issues
        """)

if __name__ == "__main__":
    main()
