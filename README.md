# Employee Attrition Predictor

A machine learning web application that predicts employee attrition risk and provides actionable insights for HR teams to improve employee retention.

![App Screenshot](static/images/screenshot.png)

## Features

- 🎯 **Predictive Analytics**: Forecast employee attrition risk using machine learning
- 📊 **Interactive Dashboard**: Visualize key HR metrics and trends
- 📈 **Risk Classification**: Categorize employees into low, medium, and high-risk groups
- 🔍 **Insightful Analysis**: Identify key factors contributing to attrition
- 🎨 **Modern UI**: Clean, responsive design with an intuitive interface

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python (Flask)
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Data Visualization**: Chart.js

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/employee-attrition-predictor.git
   cd employee-attrition-predictor
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Dataset

The model is trained on `HRDataset_v14.csv`, which includes comprehensive HR metrics such as:
- Employee demographics
- Job satisfaction levels
- Performance ratings
- Work-life balance metrics
- Compensation data

## Usage

1. Navigate to the web interface
2. Enter employee details or upload a CSV file
3. View predictions and insights
4. Download reports for further analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Dataset: [HR Analytics Dataset](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)
- Icons: [Bootstrap Icons](https://icons.getbootstrap.com/)
