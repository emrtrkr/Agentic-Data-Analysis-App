# 🚀 NoCodeDataAnalysis

**AI-Powered Professional Data Analysis Platform**

NoCodeDataAnalysis is an advanced AI platform that enables professional data analysis using natural language without requiring programming knowledge. Simply ask your questions, and AI will perform analyses, create visualizations, and generate professional reports for you.

## ✨ Features

### 🤖 **AI-Powered Analysis**
- **Natural Language Processing**: Understands Turkish and English queries with automatic code generation
- **Multi-Model Support**: Groq Cloud API and Ollama Local models
- **Smart Code Generation**: Automatic creation of Pandas, Matplotlib, Seaborn code

### 📊 **Visualization and Analysis**
- **Automatic Chart Generation**: Bar charts, line charts, histograms, heatmaps, pie charts
- **Professional Design**: Modern visualizations with dark theme
- **Expert Analysis**: AI-powered detailed insights for every chart
- **Interactive Interface**: User-friendly Streamlit-based interface

### 🔧 **Data Preprocessing**
- **AI-Powered Cleaning**: Automatic missing value imputation
- **Data Type Optimization**: Memory usage reduction
- **Quality Analysis**: Comprehensive data quality reports
- **Comparison Views**: Original vs processed data comparison

### 📄 **Executive Reports**
- **Automatic Report Generation**: Professional reports of all performed analyses
- **Multi-Format Support**: Word (.docx) and PDF export
- **Comprehensive Content**: Executive summary, detailed findings, recommendations
- **Tracking System**: Automatic logging of all analysis history

### 🎨 **Modern Interface**
- **Responsive Design**: Compatible with all screen sizes
- **Tab Structure**: Home, Data Analysis, Data Preprocessing
- **Dynamic Content**: Dataset-specific examples and insights
- **Dark Theme**: Eye-friendly professional appearance

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/emrtrkr/Agentic-Data-Analysis-App.git
cd Agentic_Data_Analysis
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Usage

### Start the Application
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Basic Usage Steps

1. **📁 Data Upload**
   - Upload your CSV file from the left panel
   - Automatic dataset analysis will be displayed

2. **🔧 Data Preprocessing (Optional)**
   - Switch to "Data Preprocessing" tab
   - Click "Start Data Preprocessing" button
   - AI automatically cleans your data

3. **💬 Data Analysis**
   - Ask questions in the "Data Analysis" tab
   - AI automatically performs analysis and creates charts

4. **📊 Report Generation**
   - Click the "Executive Report" button
   - Download reports in Word or PDF format

### Example Queries

#### Visualization Questions
- "Show sales trends by month"
- "Create revenue distribution by category"
- "Visualize customer count by age groups"
- "Generate correlation matrix"

#### Statistical Questions
- "What is the average sales amount?"
- "Which category generates the highest revenue?"
- "What is the average customer age?"
- "Which month had the highest sales?"

## 🏗️ Project Structure

```
NoCodeDataAnalysis/
├── app.py                      # Main Streamlit application
├── config_models.py            # AI model configurations
├── data_analysis_engine.py     # Data analysis engine
├── data_preprocessing.py       # Data preprocessing module
├── manager_report.py           # Executive report system
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

### Module Descriptions

#### `app.py`
- Main application file
- Streamlit interface
- Tab structure and UI components
- User interaction logic

#### `config_models.py`
- AI model configurations
- Groq and Ollama integration
- Translation services (Google Translate)
- Model selection and management system

#### `data_analysis_engine.py`
- Data analysis engine
- Code generation and execution
- Chart creation and analysis
- AI-powered interpretation system

#### `data_preprocessing.py`
- Data cleaning algorithms
- Missing value imputation strategies
- Data type optimization
- AI-powered data quality analysis

#### `manager_report.py`
- Analysis tracking system
- Report generation engine
- Word/PDF export functions
- Executive summary AI agent

## ⚙️ Configuration

### AI Model Settings

#### Groq Cloud API
```python
# In .env file
GROQ_API_KEY=your_api_key

# Supported models
- llama-3.3-70b-versatile
```

#### Ollama Local
```bash
# Ollama installation
curl -fsSL https://ollama.ai/install.sh | sh

# Model download
ollama pull llama3.1:8b
ollama pull llama2:7b
```

### Translation Service
Google Translate is automatically installed and used. Internet connection required.

## 🔧 Advanced Features

### Custom Data Type Support
- **Financial Data**: Currencies, percentage symbols
- **Date/Time**: Multiple date formats
- **Categorical**: Automatic category detection
- **Numerical**: Float/integer optimization

### AI Agent System
- **Query Understanding**: Question comprehension and categorization
- **Code Generation**: Automatic code creation
- **Execution**: Safe code execution
- **Reasoning**: Result interpretation
- **Report Generation**: Comprehensive report creation

### Security
- **Code Filtering**: Malicious code prevention
- **Sandbox Execution**: Isolated execution environment
- **Input Validation**: Comprehensive input validation
- **Error Handling**: Robust error management

## 📈 Supported Data Types

### Business Data
- **Sales Data**: Revenue, product, customer analytics
- **Marketing**: Campaign, ROI, channel analysis
- **Financial**: Income-expense, profitability, budget
- **Customer**: Demographics, behavior, segmentation

### Industry Data
- **Healthcare**: Patient, treatment, hospital data
- **Education**: Student, exam, performance data
- **Manufacturing**: Quality, machinery, capacity
- **Logistics**: Delivery, route, cost analysis

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 🐛 Troubleshooting

### Common Issues

#### Model Not Found
```bash
# Check Groq API key
echo $GROQ_API_KEY

# Check Ollama service
ollama list
```

#### Import Errors
```bash
# Reinstall packages
pip install -r requirements.txt --force-reinstall
```

#### Charts Not Displaying
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

### Log Files
- Streamlit: `~/.streamlit/logs/`
- Application: Console output

## 📞 Support

- **Issues**: Use GitHub Issues page
- **Documentation**: This README file
- **Community**: GitHub Discussions

## 🗺️ Roadmap

### v2.0 Plans
- [ ] Excel (.xlsx) file support
- [ ] Real-time data connections
- [ ] Multi-language support
- [ ] Dashboard creation
- [ ] API endpoints
- [ ] Cloud deployment

### v2.1 Plans
- [ ] Machine Learning models
- [ ] Prediction algorithms
- [ ] A/B testing analysis
- [ ] Time series analysis
- [ ] Anomaly detection

## 🎯 Performance

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **CPU**: 2+ core processor
- **Disk**: 1GB free space
- **Internet**: Required for API services

### Benchmarks
- **Medium dataset** (10K rows): ~5-10 seconds
- **Large dataset** (100K rows): ~30-60 seconds
- **Chart generation**: ~2-5 seconds
- **Report creation**: ~10-20 seconds

## 🌟 Key Benefits

### For Business Users
- **No Programming Required**: Natural language interface
- **Instant Insights**: Immediate analysis results
- **Professional Reports**: Executive-ready documentation
- **Time Savings**: Hours of analysis in minutes

### For Data Teams
- **Rapid Prototyping**: Quick data exploration
- **Consistent Output**: Standardized visualizations
- **Documentation**: Automatic analysis logging
- **Collaboration**: Easy sharing and reporting

### For Organizations
- **Cost Effective**: Reduces need for specialized analysts
- **Scalable**: Handles various data sizes and types
- **Secure**: Local processing options available
- **Extensible**: Open source and customizable

## 🏆 Use Cases

### Sales Analytics
- Revenue trend analysis
- Product performance tracking
- Customer segmentation
- Seasonal pattern identification

### Marketing Intelligence
- Campaign effectiveness measurement
- ROI calculation and optimization
- Channel performance comparison
- Customer acquisition analysis

### Operations Management
- Process efficiency monitoring
- Quality control analysis
- Resource utilization tracking
- Performance benchmarking

### Financial Reporting
- Budget vs actual analysis
- Profitability assessment
- Cost center evaluation
- Financial trend forecasting

---

**🚀 Experience the future of data analysis with NoCodeDataAnalysis!**

*AI-powered, user-friendly, professional data analysis platform.*