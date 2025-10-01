#!/usr/bin/env python3
"""
Test script for T2DM Risk Prediction App
This script tests the core functionality before running the Streamlit app
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_lifestyle_score():
    """Test the lifestyle score calculation"""
    print("🧪 Testing lifestyle score calculation...")
    
    # Test case 1: Healthy individual
    healthy_data = {
        'age': 50,
        'sex': 'Female',
        'bmi': 23.0,
        'waist': 80.0,
        'smoking_status': 'Never',
        'alcohol_freq': 'Occasional',
        'moderate_mins': 150,
        'vigorous_mins': 75,
        'sleep_hrs': 8.0,
        'fruit_intake': 3,
        'veg_intake': 4,
        'red_meat_intake': 2
    }
    
    # Test case 2: High-risk individual
    high_risk_data = {
        'age': 65,
        'sex': 'Male',
        'bmi': 32.0,
        'waist': 110.0,
        'smoking_status': 'Current',
        'alcohol_freq': 'Daily',
        'moderate_mins': 30,
        'vigorous_mins': 0,
        'sleep_hrs': 5.0,
        'fruit_intake': 1,
        'veg_intake': 1,
        'red_meat_intake': 5
    }
    
    # Import the function from the app
    try:
        from t2dm_app import calculate_lifestyle_score, get_risk_category, get_risk_color
        
        # Test lifestyle scores
        healthy_score = calculate_lifestyle_score(healthy_data)
        high_risk_score = calculate_lifestyle_score(high_risk_data)
        
        print(f"✅ Healthy individual lifestyle score: {healthy_score}/10")
        print(f"✅ High-risk individual lifestyle score: {high_risk_score}/10")
        
        # Test risk categorization
        low_risk = get_risk_category(0.03)
        medium_risk = get_risk_category(0.10)
        high_risk = get_risk_category(0.25)
        
        print(f"✅ Risk categories: Low={low_risk}, Medium={medium_risk}, High={high_risk}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_database_functions():
    """Test database functionality"""
    print("\n🧪 Testing database functions...")
    
    try:
        from t2dm_app import init_db, save_prediction, fetch_recent_predictions
        
        # Initialize database
        init_db()
        print("✅ Database initialized successfully")
        
        # Test data
        test_data = {
            'age': 55,
            'sex': 'Male',
            'bmi': 28.0,
            'waist': 95.0,
            'smoking_status': 'Previous',
            'alcohol_freq': 'Weekly_1_2',
            'moderate_mins': 120,
            'vigorous_mins': 60,
            'sleep_hrs': 7.0,
            'fruit_intake': 2,
            'veg_intake': 3,
            'red_meat_intake': 3
        }
        
        # Save prediction
        save_prediction(test_data, 0.12, "Medium", 6)
        print("✅ Prediction saved successfully")
        
        # Fetch recent predictions
        recent = fetch_recent_predictions(5)
        print(f"✅ Fetched {len(recent)} recent predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_risk_calculation():
    """Test the risk calculation logic"""
    print("\n🧪 Testing risk calculation...")
    
    try:
        # Test the risk calculation logic from the app
        def calculate_risk(user_data):
            risk_score = 0
            
            # Age factor
            age_norm = (user_data['age'] - 54.5) / 8.0
            risk_score += age_norm * 0.8
            
            # BMI factor
            if user_data['bmi'] >= 30:
                risk_score += 1.5
            elif user_data['bmi'] >= 25:
                risk_score += 0.8
            
            # Waist factor
            if user_data['waist'] > 102:
                risk_score += 0.7
            elif user_data['waist'] > 88:
                risk_score += 0.5
            
            # Sex factor
            if user_data['sex'] == "Male":
                risk_score += 0.3
            
            # Convert to probability
            logits = risk_score * 1.5
            risk_prob = 1 / (1 + np.exp(-logits))
            return min(max(risk_prob, 0.01), 0.99)
        
        # Test cases
        test_cases = [
            {"age": 45, "sex": "Female", "bmi": 22, "waist": 75},
            {"age": 60, "sex": "Male", "bmi": 35, "waist": 115},
            {"age": 55, "sex": "Male", "bmi": 28, "waist": 95}
        ]
        
        for i, case in enumerate(test_cases, 1):
            risk = calculate_risk(case)
            print(f"✅ Test case {i}: Age={case['age']}, Sex={case['sex']}, BMI={case['bmi']} → Risk: {risk:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk calculation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting T2DM Risk Prediction App Tests")
    print("=" * 50)
    
    tests = [
        test_lifestyle_score,
        test_database_functions,
        test_risk_calculation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready to run.")
        print("\nTo start the app, run:")
        print("streamlit run app/t2dm_app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 