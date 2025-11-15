#!/bin/bash

# Output directory for results
OUTPUT_DIR="employee_risk_reports"
mkdir -p "$OUTPUT_DIR"

# Output files
LOW_RISK="$OUTPUT_DIR/low_risk_employees.json"
MEDIUM_RISK="$OUTPUT_DIR/medium_risk_employees.json"
HIGH_RISK="$OUTPUT_DIR/high_risk_employees.json"
SUMMARY="$OUTPUT_DIR/risk_summary.txt"

# Initialize output files
echo "[" > "$LOW_RISK"
echo "[" > "$MEDIUM_RISK"
echo "[" > "$HIGH_RISK"

# Counters
TOTAL=0
LOW=0
MED=0
HIGH=0

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq to run this script."
    exit 1
fi

# Check if the data file exists
DATA_FILE="data/emp_attrition.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file $DATA_FILE not found."
    exit 1
fi

# Read CSV and process each employee
tail -n +2 "$DATA_FILE" | while IFS=, read -r line; do
    # Extract fields from CSV
    IFS=, read -r Age Attrition BusinessTravel DailyRate Department \
        DistanceFromHome Education EducationField EmployeeCount EmployeeNumber \
        EnvironmentSatisfaction Gender HourlyRate JobInvolvement JobLevel \
        JobRole JobSatisfaction MaritalStatus MonthlyIncome MonthlyRate \
        NumCompaniesWorked Over18 OverTime PercentSalaryHike PerformanceRating \
        RelationshipSatisfaction StandardHours StockOptionLevel TotalWorkingYears \
        TrainingTimesLastYear WorkLifeBalance YearsAtCompany YearsInCurrentRole \
        YearsSinceLastPromotion YearsWithCurrManager <<< "$line"

    # Skip if any required field is empty
    if [ -z "$Age" ] || [ -z "$JobRole" ] || [ -z "$MonthlyIncome" ] || [ -z "$YearsAtCompany" ] || [ -z "$OverTime" ]; then
        continue
    fi

    # Prepare JSON payload
    PAYLOAD=$(jq -n \
        --arg age "$Age" \
        --arg jobRole "$JobRole" \
        --arg monthlyIncome "$MonthlyIncome" \
        --arg yearsAtCompany "$YearsAtCompany" \
        --arg overTime "$OverTime" \
        '{age: $age|tonumber, jobRole: $jobRole, monthlyIncome: $monthlyIncome|tonumber, yearsAtCompany: $yearsAtCompany|tonumber, overTime: ($overTime == "Yes")}')

    # Make prediction request
    RESPONSE=$(curl -s -X POST http://localhost:5000/api/predict \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD")

    # Extract risk level
    RISK_LEVEL=$(echo "$RESPONSE" | jq -r '.riskLevel' 2>/dev/null || echo "")
    
    # Skip if no valid response
    if [ -z "$RISK_LEVEL" ]; then
        echo "Warning: Invalid response for employee $EmployeeNumber"
        continue
    fi

    # Add employee data to response
    ENHANCED_RESPONSE=$(echo "$RESPONSE" | jq --arg id "$EmployeeNumber" \
        --arg name "Employee $EmployeeNumber" \
        --arg department "$Department" \
        '. + {employeeId: $id, employeeName: $name, department: $department}')

    # Categorize by risk level
    case $RISK_LEVEL in
        "Low")
            if [ $LOW -gt 0 ]; then
                echo "," >> "$LOW_RISK"
            fi
            echo "$ENHANCED_RESPONSE" >> "$LOW_RISK"
            ((LOW++))
            ;;
        "Medium")
            if [ $MED -gt 0 ]; then
                echo "," >> "$MEDIUM_RISK"
            fi
            echo "$ENHANCED_RESPONSE" >> "$MEDIUM_RISK"
            ((MED++))
            ;;
        "High")
            if [ $HIGH -gt 0 ]; then
                echo "," >> "$HIGH_RISK"
            fi
            echo "$ENHANCED_RISK" >> "$HIGH_RISK"
            ((HIGH++))
            ;;
    esac

    ((TOTAL++))
    echo -n "."

done

# Close JSON arrays
echo "]" >> "$LOW_RISK"
echo "]" >> "$MEDIUM_RISK"
echo "]" >> "$HIGH_RISK"

# Generate summary
echo "Employee Attrition Risk Analysis" > "$SUMMARY"
echo "==============================" >> "$SUMMARY"
echo "Date: $(date)" >> "$SUMMARY"
echo "Total employees processed: $TOTAL" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Risk Level Distribution:" >> "$SUMMARY"
echo "- Low Risk: $LOW employees ($(awk "BEGIN {printf \"%.1f\", $LOW/$TOTAL*100}%"))" >> "$SUMMARY"
echo "- Medium Risk: $MED employees ($(awk "BEGIN {printf \"%.1f\", $MED/$TOTAL*100}%"))" >> "$SUMMARY"
echo "- High Risk: $HIGH employees ($(awk "BEGIN {printf \"%.1f\", $HIGH/$TOTAL*100}%"))" >> "$SUMMARY"

echo ""
echo "Analysis complete! Results saved to $OUTPUT_DIR/ directory."
cat "$SUMMARY"

    "age": 35,
    "jobRole": "manager",
    "monthlyIncome": 4500,
    "yearsAtCompany": 4,
    "overTime": true
  }'

# Sample prediction request - High risk
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "jobRole": "analyst",
    "monthlyIncome": 3500,
    "yearsAtCompany": 8,
    "overTime": true
  }'

# Test with missing fields (should return error)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "jobRole": "developer"
  }'
