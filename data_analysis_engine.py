import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Tuple
from config_models import translate_to_english, translate_to_turkish, call_selected_llm

# === Advanced Query Understanding ===
def query_understanding_tool(query: str) -> bool:
    """Determine if query requests visualization with improved logic"""
    english_query = translate_to_english(query)
    
    # Basit keyword check ekleyelim
    visualization_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'görselleştir', 
        'çiz', 'grafik', 'show', 'göster', 'scatter', 'line',
        'bar', 'histogram', 'heatmap', 'pie'
    ]
    
    # Eğer visualization keyword varsa true döndür
    for keyword in visualization_keywords:
        if keyword.lower() in query.lower() or keyword.lower() in english_query.lower():
            return True
    
    # Original LLM logic
    system_prompt = """You determine if a query needs a visual chart/graph or just numerical results.

RESPOND 'true' ONLY if the query explicitly asks for:
- Charts, plots, graphs, visualizations
- Words like: "plot", "chart", "graph", "show distribution", "visualize", "draw"
- Visual comparisons that need charts
- Turkish: "grafik", "çiz", "görselleştir", "dağılım göster"

RESPOND 'false' for queries asking for:
- Descriptive statistics (describe, summary, mean, median, count)
- Numbers, calculations, tables
- Information that can be shown as text/numbers
- Turkish: "betimsel", "istatistik", "ortalama", "sayı", "hesapla"

Examples:
- "Show descriptive statistics" → false (numbers)
- "Plot a histogram" → true (visual)
- "What is the correlation?" → false (number)
- "Show correlation matrix as heatmap" → true (visual)"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {english_query}"}
    ]
    
    response = call_selected_llm(messages, max_tokens=10, temperature=0.1)
    return response.strip().lower() == "true"

# === Advanced Code Generation ===
def plot_code_generator_tool(cols: List[str], query: str, df_info: Dict) -> str:
    """Generate plotting code for visualizations"""
    english_query = translate_to_english(query)
    
    prompt_template = f"""
You are creating a professional visualization. The DataFrame 'df' is already loaded.

Available columns: {', '.join(cols)}
Dataset info: {str(df_info)}
Task: "{english_query}"

CRITICAL REQUIREMENTS:
1. Start with: result = plt.figure(figsize=(12,8))
2. Use futuristic dark styling:
   - plt.style.use('dark_background')
   - plt.gca().set_facecolor('#0a0a0a')
   - Use colors: '#00ffff', '#ff00ff', '#ffff00', '#00ff00'
   - plt.grid(True, alpha=0.3, color='#333333')
3. Import numpy as np if needed
4. NO plt.show() or plt.display()
5. Return ONLY ```python code block
6. ALWAYS calculate and store real data in plot_data dictionary
7. End with: plt.tight_layout()

SPECIFIC CHART EXAMPLES:

A) CORRELATION HEATMAP:
```python
import numpy as np
result = plt.figure(figsize=(12,8))
plt.style.use('dark_background')
plt.gca().set_facecolor('#0a0a0a')

# Calculate correlation matrix for numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Create heatmap
im = plt.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im)
cbar.ax.tick_params(colors='white')

# Set ticks and labels
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right', color='white')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, color='white')

# Add correlation values as text
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f'{{corr_matrix.iloc[i,j]:.2f}}', 
                ha='center', va='center', color='white', fontweight='bold')

plt.title('Correlation Matrix', color='white', fontsize=16)
plt.grid(False)

# Store correlation data and find top 3 positive correlations
corr_values = []
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_values.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j]))

# Sort by correlation value and get top 3 positive
top_corr = sorted([x for x in corr_values if x[2] > 0], key=lambda x: x[2], reverse=True)[:3]

plot_data = {{"chart_type": "heatmap", "correlation_matrix": corr_matrix.values.tolist(), 
             "columns": corr_matrix.columns.tolist(), "top_3_correlations": top_corr}}

plt.tight_layout()
```

B) STACKED BAR PLOT:
```python
import numpy as np
result = plt.figure(figsize=(12,8))
plt.style.use('dark_background')
plt.gca().set_facecolor('#0a0a0a')

# Create crosstab
crosstab = pd.crosstab(df['Channel'], df['TransactionType'])

# Create stacked bar plot
bottom = np.zeros(len(crosstab.index))
colors = ['#00ffff', '#ff00ff', '#ffff00', '#00ff00', '#ff6600']

for i, col in enumerate(crosstab.columns):
    plt.bar(crosstab.index, crosstab[col], bottom=bottom, 
           label=col, color=colors[i % len(colors)], alpha=0.8)
    bottom += crosstab[col]

plt.title('Transaction Count by Channel and Type', color='white', fontsize=16)
plt.xlabel('Channel', color='white')
plt.ylabel('Transaction Count', color='white')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3, color='#333333')

# Store crosstab data
plot_data = {{"chart_type": "stacked_bar", "crosstab": crosstab.to_dict(), 
             "channels": crosstab.index.tolist(), "transaction_types": crosstab.columns.tolist()}}

plt.tight_layout()
```

C) CATEGORICAL BAR CHART:
```python
import numpy as np
result = plt.figure(figsize=(12,8))
plt.style.use('dark_background')
plt.gca().set_facecolor('#0a0a0a')

# Calculate counts and percentages
counts = df['Channel'].value_counts()
percentages = (counts / len(df) * 100).round(1)

# Create bar chart
bars = plt.bar(counts.index, counts.values, color='#00ffff', alpha=0.8)

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, percentages.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
            f'{{pct}}%', ha='center', va='bottom', color='#ffff00', fontweight='bold')

plt.title('Channel Distribution', color='white', fontsize=16)
plt.xlabel('Channel', color='white')
plt.ylabel('Count', color='white')
plt.grid(True, alpha=0.3, color='#333333')

# Store actual data for analysis
plot_data = {{"chart_type": "bar", "categories": counts.index.tolist(), 
             "counts": counts.values.tolist(), "percentages": percentages.values.tolist()}}

plt.tight_layout()
```

Choose the appropriate example based on the task and modify column names accordingly.
Write clean visualization code with ACCURATE plot_data:
"""
    
    return prompt_template

def code_writing_tool(cols: List[str], query: str, df_info: Dict) -> str:
    """Generate data analysis code"""
    
    # Format columns safely
    column_list = "', '".join(cols)
    
    template = f"""
DataFrame 'df' is already loaded. Write pandas analysis code.

AVAILABLE COLUMNS: ['{column_list}']
TASK: {query}

RULES: 
- DataFrame 'df' is already loaded - DO NOT use pd.read_csv()
- Use ONLY the column names listed above
- For descriptive statistics: df.describe()
- For categorical data: check with df['column'].unique() first
- Assign result to 'result' variable

Write simple pandas code. Return ONLY ```python code block.
"""
    
    return template

# === Enhanced Code Generation Agent ===
def code_generation_agent(query: str, df: pd.DataFrame) -> Tuple[str, bool, str]:
    """Generate simple, focused code for basic EDA"""
    
    should_plot = query_understanding_tool(query)
    
    # Simple column info for context
    cols = df.columns.tolist()
    df_info = {'rows': len(df), 'columns': len(df.columns)}
    
    if should_plot:
        prompt = plot_code_generator_tool(cols, query, df_info)
    else:
        prompt = code_writing_tool(cols, query, df_info)

    messages = [
        {
            "role": "system",
            "content": """You write simple pandas/matplotlib code for data analysis.

IMPORTANT RULES:
- The DataFrame 'df' is already loaded in memory
- NEVER use pd.read_csv() or any file reading
- Use only basic pandas operations
- Keep code simple and clean
- Return ONLY a ```python code block"""
        },
        {"role": "user", "content": prompt}
    ]

    response = call_selected_llm(messages, max_tokens=500, temperature=0.2)
    code = extract_first_code_block(response)
    
    return code, should_plot, ""

# === Enhanced Execution Environment ===
def execution_agent(code: str, df: pd.DataFrame, should_plot: bool):
    """Execute generated code in an enhanced scientific computing environment"""
    import numpy as np
    import scipy.stats as stats
    
    # Clean the code - remove problematic lines
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove plt.show() and plt.display()
        if 'plt.show()' in line or 'plt.display()' in line:
            continue
        # Replace plt.figure() with result = plt.figure()
        if line.strip().startswith('plt.figure(') and 'result =' not in line:
            line = line.replace('plt.figure(', 'result = plt.figure(')
        cleaned_lines.append(line)
    
    # Add result assignment if missing
    cleaned_code = '\n'.join(cleaned_lines)
    if should_plot and 'result =' not in cleaned_code:
        cleaned_code = 'result = plt.figure(figsize=(12,8))\n' + cleaned_code
    
    # Create rich execution environment
    env = {
        "pd": pd, 
        "df": df, 
        "json": json, 
        "np": np,
        "stats": stats
    }
    
    if should_plot:
        # Configure professional matplotlib settings
        plt.style.use('dark_background')
        plt.rcParams.update({
            "figure.dpi": 120,
            "figure.facecolor": '#0a0a0a',
            "savefig.facecolor": '#0a0a0a',
            "axes.facecolor": '#0a0a0a',
            "figure.figsize": (12, 8),
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10
        })
        env.update({"plt": plt, "sns": sns})
    
    try:
        # Execute the cleaned code
        exec(cleaned_code, {}, env)
        result = env.get("result", None)
        plot_data = env.get("plot_data", None)
        return result, plot_data
    except Exception as exc:
        error_msg = f"Kod yürütme hatası: {str(exc)}"
        # Add debugging info
        if "format specifier" in str(exc):
            error_msg += "\n🔧 Debug: F-string formatting hatası tespit edildi."
        return error_msg, None

# === Professional Chart Analysis ===
def safe_json_serialize(obj):
    """Convert pandas objects to JSON-serializable format"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    if hasattr(obj, 'tolist'):  # pandas Series or numpy array
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):  # Timestamp objects
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, np.datetime64):  # numpy datetime
        return pd.to_datetime(obj).strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):  # numpy integers
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):  # numpy floats
        return float(obj)
    elif pd.isna(obj):  # NaN values
        return None
    else:
        return obj

def analyze_chart_data(plot_data: Dict, query: str) -> str:
    """Provide simple, practical analysis of visualization data without statistical jargon"""
    english_query = translate_to_english(query)
    
    # Convert plot_data to safe format
    safe_plot_data = safe_json_serialize(plot_data)
    
    # Try Turkish-first approach for better results
    from config_models import GOOGLE_TRANSLATE_AVAILABLE
    
    if GOOGLE_TRANSLATE_AVAILABLE:
        # Use English context but explicitly request Turkish response
        analysis_prompt = f"""
As a data analyst, provide a simple, practical analysis for: "{english_query}"

Visualization Data: {json.dumps(safe_plot_data, indent=2)}

Provide a clear analysis in EXACTLY 2-3 sentences covering:

1. MAIN FINDING: What does the chart show? State the key pattern with actual numbers from the data
2. COMPARISON: Which categories/values are highest/lowest? Give specific numbers
3. BUSINESS INSIGHT: What does this mean in practical terms? What should someone do with this information?

IMPORTANT RULES:
- Use ONLY the actual numbers from the visualization data
- NO statistical tests (no p-values, Cohen's d, significance tests)
- NO confidence intervals or sample size mentions
- Focus on what the data shows, not statistical validity
- Be practical and actionable
- Keep it simple and easy to understand

RESPOND IN TURKISH.
"""
        
        system_prompt = """Sen bir veri analisti olarak basit ve pratik açıklamalar yapıyorsun.

ÖNEMLI KURALLAR:
- Sadece grafikteki gerçek sayıları kullan
- İstatistiksel testler yapma (p-değeri, Cohen's d, anlamlılık testleri YOK)
- Güven aralığı veya örneklem büyüklüğünden bahsetme
- Sadece verinin ne gösterdiğine odaklan
- Pratik ve anlaşılır ol
- Maksimum 2-3 cümle

TÜRKÇE yanıt ver."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_prompt}
        ]
        
        try:
            response = call_selected_llm(messages, max_tokens=300, temperature=0.3)
            
            if response and not response.startswith("❌"):
                # Check if response is in Turkish, if not translate
                turkish_indicators = ['bu', 've', 'bir', 'için', 'ile', 'göre', 'olan', 'daha', 'çok', 'sonuç', 'analiz', 'veri', 'grafik']
                if any(word in response.lower() for word in turkish_indicators):
                    return response
                else:
                    # Try to translate English response to Turkish
                    turkish_response = translate_to_turkish(response)
                    if turkish_response and turkish_response != response and len(turkish_response) > 20:
                        return turkish_response
                    else:
                        return response  # Return original if translation fails
            else:
                return response
                
        except Exception:
            return "Grafik analizi tamamlandı."
    
    else:
        # Fallback: Direct Turkish prompt without translation
        analysis_prompt = f"""
Veri analisti olarak şu grafik için basit ve pratik bir analiz yap: "{query}"

Grafik Verisi: {json.dumps(safe_plot_data, indent=2)}

TAM OLARAK 2-3 cümle ile açıkla:

1. ANA BULGU: Grafik ne gösteriyor? Ana deseni gerçek sayılarla belirt
2. KARŞILAŞTIRMA: Hangi kategoriler/değerler en yüksek/düşük? Spesifik sayılar ver
3. PRATIK ANLAM: Bu pratik olarak ne anlama geliyor? Bu bilgiyle ne yapılmalı?

ÖNEMLI KURALLAR:
- SADECE grafikteki gerçek sayıları kullan
- İstatistiksel test yapma (p-değeri, Cohen's d, anlamlılık testleri YOK)
- Güven aralığı veya örneklem büyüklüğünden bahsetme
- Sadece verinin ne gösterdiğine odaklan
- Pratik ve anlaşılır ol
"""
        
        system_prompt = """Sen bir veri analisti olarak basit ve pratik açıklamalar yapıyorsun.

ÖNEMLI KURALLAR:
- Sadece grafikteki gerçek sayıları kullan
- İstatistiksel testler yapma (p-değeri, Cohen's d, anlamlılık testleri YOK)
- Güven aralığı veya örneklem büyüklüğünden bahsetme
- Sadece verinin ne gösterdiğine odaklan
- Pratik ve anlaşılır ol
- Maksimum 2-3 cümle

TÜRKÇE yanıt ver."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_prompt}
        ]
        
        try:
            response = call_selected_llm(messages, max_tokens=300, temperature=0.3)
            return response if response else "Grafik analizi tamamlandı."
        except Exception:
            return "Grafik analizi tamamlandı."

# === Enhanced Reasoning Agent ===
def reasoning_agent(query: str, result: Any, plot_data: Dict = None) -> str:
    """Generate simple, clear analysis explanations"""
    english_query = translate_to_english(query)
    
    is_error = isinstance(result, str) and "hatası" in result.lower()
    is_plot = hasattr(result, 'figure') or str(type(result)) == "<class 'matplotlib.figure.Figure'>"
    
    # Try Turkish-first approach for better results
    from config_models import GOOGLE_TRANSLATE_AVAILABLE
    
    if GOOGLE_TRANSLATE_AVAILABLE:
        # Use English prompt but explicitly request Turkish response
        if is_error:
            desc = result
            prompt = f'Query: "{english_query}". Error: {desc}. Briefly explain the issue in 1-2 sentences. RESPOND IN TURKISH.'
        elif is_plot:
            prompt = f'Query: "{english_query}". A visualization was created. Explain what this chart shows in 2-3 simple sentences. RESPOND IN TURKISH.'
        else:
            # Format result simply
            if isinstance(result, (pd.DataFrame, pd.Series)):
                desc = f"Analysis completed. Results: {str(result)[:300]}..."
            else:
                desc = str(result)[:300]
            
            prompt = f'Query: "{english_query}". Results: {desc}. Explain what these results mean in 2-3 simple sentences. RESPOND IN TURKISH.'
        
        system_prompt = """Sen bir veri analisti olarak açık ve basit açıklamalar yapıyorsun.

Yanıtların şu özelliklerde olmalı:
- Anlaşılması kolay
- Kısa ve öz (maksimum 2-3 cümle)
- Pratik içgörülere odaklanmış
- Karmaşık istatistiksel jargondan uzak

TÜRKÇE yanıt ver."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = call_selected_llm(messages, max_tokens=200, temperature=0.3)
            
            # Check if response is in Turkish, if not translate
            if response and not response.startswith("❌"):
                turkish_indicators = ['bu', 've', 'bir', 'için', 'ile', 'göre', 'olan', 'daha', 'çok', 'sonuç']
                if any(word in response.lower() for word in turkish_indicators):
                    return response
                else:
                    # Try to translate English response to Turkish
                    turkish_response = translate_to_turkish(response)
                    if turkish_response and turkish_response != response and len(turkish_response) > 10:
                        return turkish_response
                    else:
                        return response  # Return original if translation fails
            else:
                return response
                
        except Exception:
            return "Analiz tamamlandı."
    
    else:
        # Fallback: Direct Turkish prompt without translation
        if is_error:
            desc = result
            prompt = f'Sorgu: "{query}". Hata: {desc}. Sorunu 1-2 cümlede kısaca açıklayın.'
        elif is_plot:
            prompt = f'Sorgu: "{query}". Bir görselleştirme oluşturuldu. Bu grafiğin ne gösterdiğini 2-3 basit cümlede açıklayın.'
        else:
            # Format result simply
            if isinstance(result, (pd.DataFrame, pd.Series)):
                desc = f"Analiz tamamlandı. Sonuçlar: {str(result)[:300]}..."
            else:
                desc = str(result)[:300]
            
            prompt = f'Sorgu: "{query}". Sonuçlar: {desc}. Bu sonuçların ne anlama geldiğini 2-3 basit cümlede açıklayın.'
        
        system_prompt = """Sen bir veri analisti olarak açık ve basit açıklamalar yapıyorsun.

Yanıtların şu özelliklerde olmalı:
- Anlaşılması kolay
- Kısa ve öz (maksimum 2-3 cümle)
- Pratik içgörülere odaklanmış
- Karmaşık istatistiksel jargondan uzak

TÜRKÇE yanıt ver."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = call_selected_llm(messages, max_tokens=200, temperature=0.3)
            return response if response else "Analiz tamamlandı."
        except Exception:
            return "Analiz tamamlandı."

# === Dataset Analysis ===
def dataset_summary_tool(df: pd.DataFrame) -> str:
    """Generate comprehensive dataset analysis prompt in English"""
    
    # Advanced dataset analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing_data = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Statistical summary for numeric columns
    numeric_summary = {}
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().round(2).to_dict()
    
    prompt = f"""
Analyze this professional dataset comprehensively and provide response in English:

DATASET OVERVIEW:
- Dimensions: {len(df):,} rows × {len(df.columns)} columns
- Numeric variables: {len(numeric_cols)} ({', '.join(numeric_cols[:8])}{"..." if len(numeric_cols) > 8 else ""})
- Categorical variables: {len(categorical_cols)} ({', '.join(categorical_cols[:8])}{"..." if len(categorical_cols) > 8 else ""})
- Missing values: {missing_data.sum():,} total ({(missing_data.sum()/len(df)*100):.1f}% of dataset)
- Duplicate rows: {duplicate_rows:,}
- Memory usage: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB

STATISTICAL SUMMARY:
{json.dumps(numeric_summary, indent=2) if numeric_summary else "No numeric columns available"}

Provide a comprehensive EDA-focused assessment including:
1. Dataset description and potential domain/use case identification
2. Data quality assessment with specific recommendations for cleaning/preparation
3. 5-6 sophisticated analytical questions that could be explored through EDA

Focus on exploratory data analysis aspects only. Write for a technical audience in English. Be specific and actionable.
"""
    
    return prompt

def data_insight_agent(df: pd.DataFrame) -> str:
    """Generate professional insights with optimal English processing and Turkish output"""
    try:
        # Use English prompt for optimal model performance
        prompt = dataset_summary_tool(df)
        
        system_prompt = """You are a senior data scientist providing comprehensive dataset assessment for exploratory data analysis.
Your analysis should be:
- Technically rigorous and detailed
- Professionally written for data science teams
- Focus on EDA (Exploratory Data Analysis) opportunities
- Identify data patterns, relationships, and quality issues
- Suggest specific EDA approaches and visualization strategies

Write your response in clear, professional English using Markdown formatting.
Use proper headers, bullet points, and numbered lists for readability.
DO NOT use HTML tags."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Get high-quality English response
        english_response = call_selected_llm(messages, max_tokens=800, temperature=0.2)
        
        if english_response and not english_response.startswith("❌"):
            # Clean HTML tags and format text
            english_response = clean_html_tags(english_response)
            english_response = format_analysis_text(english_response)
            
            # Try Google Translate
            from config_models import GOOGLE_TRANSLATE_AVAILABLE
            
            if GOOGLE_TRANSLATE_AVAILABLE:
                try:
                    turkish_response = translate_to_turkish(english_response)
                    
                    # Check if translation was successful (basic check)
                    if turkish_response and turkish_response != english_response and len(turkish_response) > 50:
                        # Clean and format translated response
                        turkish_response = clean_html_tags(turkish_response)
                        turkish_response = format_analysis_text(turkish_response)
                        return turkish_response
                    else:
                        # Translation failed, use direct Turkish
                        return generate_direct_turkish_analysis(df)
                except Exception:
                    # Translation error, use direct Turkish
                    return generate_direct_turkish_analysis(df)
            else:
                # No Google Translate, use direct Turkish
                return generate_direct_turkish_analysis(df)
        else:
            # English response failed, use direct Turkish
            return generate_direct_turkish_analysis(df)
            
    except Exception as exc:
        # If everything fails, generate a basic analysis
        return generate_basic_analysis(df)

def generate_basic_analysis(df: pd.DataFrame) -> str:
    """Generate basic analysis when all else fails"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        missing_data = df.isnull().sum().sum()
        
        basic_analysis = f"""**Veri Seti Genel Bakış**

Bu veri seti {len(df):,} satır ve {len(df.columns)} sütundan oluşmaktadır. 

**Sütun Dağılımı:**
- Sayısal sütunlar ({len(numeric_cols)} adet): {', '.join(numeric_cols[:5])}{"..." if len(numeric_cols) > 5 else ""}
- Kategorik sütunlar ({len(categorical_cols)} adet): {', '.join(categorical_cols[:5])}{"..." if len(categorical_cols) > 5 else ""}

**Veri Kalitesi:**
- Toplam eksik değer: {missing_data:,}
- Veri seti büyüklüğü: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB

**Analiz Önerileri:**
1. Sayısal değişkenler için dağılım analizleri yapılabilir
2. Kategorik değişkenler için frekans analizleri incelenebilir
3. Değişkenler arası korelasyon analizi gerçekleştirilebilir
4. Eksik değerler için temizleme stratejileri belirlenebilir
5. Görselleştirmeler ile veri desenleri keşfedilebilir

Bu veri seti EDA (Keşifsel Veri Analizi) için uygun görünmektedir."""
        
        return basic_analysis.strip()
        
    except Exception:
        return "Veri seti başarıyla yüklendi. Analiz yapmak için sorularınızı sorabilirsiniz."

def generate_direct_turkish_analysis(df: pd.DataFrame) -> str:
    """Fallback: Generate analysis directly in Turkish"""
    try:
        prompt = dataset_summary_tool_turkish(df)
        
        system_prompt = """Sen uzman bir veri bilimcisin. EDA odaklı kapsamlı analiz yap.
TÜRKÇE yanıt ver. Markdown formatlaması kullan.
Başlıklar, madde işaretleri ve numaralı listeler ile okunabilir yaz.
HTML tagları kullanma."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = call_selected_llm(messages, max_tokens=800, temperature=0.2)
        
        # Check if response is valid and not empty
        if response and not response.startswith("❌") and len(response.strip()) > 50:
            # Clean HTML tags and format text
            response = clean_html_tags(response)
            response = format_analysis_text(response)
            return response
        else:
            # If LLM response fails, use basic analysis
            return generate_basic_analysis(df)
            
    except Exception:
        # If everything fails, use basic analysis
        return generate_basic_analysis(df)

def dataset_summary_tool_turkish(df: pd.DataFrame) -> str:
    """Generate comprehensive dataset analysis prompt in Turkish"""
    
    # Advanced dataset analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing_data = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Statistical summary for numeric columns
    numeric_summary = {}
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().round(2).to_dict()
    
    prompt = f"""
Bu profesyonel veri setini kapsamlı olarak analiz et ve Türkçe yanıt ver:

VERİ SETİ ÖZETİ:
- Boyutlar: {len(df):,} satır × {len(df.columns)} sütun
- Sayısal değişkenler: {len(numeric_cols)} adet ({', '.join(numeric_cols[:8])}{"..." if len(numeric_cols) > 8 else ""})
- Kategorik değişkenler: {len(categorical_cols)} adet ({', '.join(categorical_cols[:8])}{"..." if len(categorical_cols) > 8 else ""})
- Eksik değerler: {missing_data.sum():,} toplam (veri setinin %{(missing_data.sum()/len(df)*100):.1f}'i)
- Duplicate satırlar: {duplicate_rows:,}
- Hafıza kullanımı: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB

İSTATİSTİKSEL ÖZET:
{json.dumps(numeric_summary, indent=2) if numeric_summary else "Sayısal sütun mevcut değil"}

EDA odaklı profesyonel değerlendirme sağla:
1. Veri seti açıklaması ve potansiyel domain/kullanım alanı
2. Spesifik öneriler içeren veri kalitesi değerlendirmesi
3. Keşfedilebilecek 5-6 sofistike analitik soru

Sadece keşifsel veri analizi (EDA) açısından değerlendir. Teknik bir kitle için yaz. Spesifik ve uygulanabilir ol. TÜRKÇE yanıt ver.
"""
    
    return prompt

# === Helper Functions ===
def clean_html_tags(text: str) -> str:
    """Remove HTML tags from text while preserving Markdown formatting"""
    import re
    
    # Only remove specific problematic HTML tags, keep Markdown
    text = re.sub(r'</?div[^>]*>', '', text)  # Remove div tags
    text = re.sub(r'</?span[^>]*>', '', text)  # Remove span tags
    text = re.sub(r'</?p[^>]*>', '\n\n', text)  # Replace p tags with double newline
    text = re.sub(r'<br\s*/?>', '\n', text)  # Replace br tags with newline
    text = re.sub(r'</?[^>]+(>|$)', '', text)  # Remove any other HTML tags
    
    # Clean up extra whitespace but preserve line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    return text

def format_analysis_text(text: str) -> str:
    """Format analysis text for better readability"""
    if not text:
        return text
        
    # Ensure proper spacing after numbers and bullets
    import re
    
    # Add proper line breaks before numbered lists
    text = re.sub(r'(\d+\.\s)', r'\n\n\1', text)
    
    # Add line breaks before bullet points
    text = re.sub(r'([^\n])\s*(-\s)', r'\1\n\n\2', text)
    
    # Ensure double line break after headers (lines ending with :)
    text = re.sub(r'([^:\n]):([^\n])', r'\1:\n\n\2', text)
    
    # Clean up multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_first_code_block(text: str) -> str:
    """Extract first Python code block from markdown with enhanced error handling"""
    import re
    
    # Try different patterns
    patterns = [
        r'```python\s*(.*?)```',
        r'```\s*(.*?)```',
        r'`(.*?)`'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code and len(code) > 10:  # Ensure it's substantial code
                # Basic validation - check for dangerous patterns
                if 'exec(' in code or 'eval(' in code or '__import__' in code:
                    continue
                return code
    
    # If no code block found, return empty string
    return ""