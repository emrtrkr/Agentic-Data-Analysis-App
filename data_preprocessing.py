import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from config_models import call_selected_llm, translate_to_english, translate_to_turkish
import json
import re

class DataPreprocessingAgent:
    """AI-powered data preprocessing agent for automated data cleaning and transformation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.preprocessing_log = []
        
    def log_action(self, action: str, details: str):
        """Log preprocessing actions"""
        self.preprocessing_log.append({
            "action": action,
            "details": details,
            "timestamp": pd.Timestamp.now()
        })
    
    def detect_numeric_in_object_columns(self) -> Dict[str, Dict[str, Any]]:
        """Detect object columns that contain numeric data with symbols"""
        object_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        if not object_columns:
            return {}
        
        analysis_results = {}
        
        for col in object_columns:
            sample_values = self.df[col].dropna().head(20).tolist()
            if not sample_values:
                continue
            
            # Pattern detection
            patterns_found = {
                'currency': any(re.search(r'[\$€£¥₺₹]', str(val)) for val in sample_values),
                'percentage': any(re.search(r'%', str(val)) for val in sample_values),
                'thousands_separator': any(',' in str(val) and any(c.isdigit() for c in str(val)) for val in sample_values),
                'decimal_comma': any(re.search(r'\d+,\d+$', str(val)) for val in sample_values),
                'negative_parentheses': any(re.search(r'\([0-9,\.]+\)', str(val)) for val in sample_values),
                'whitespace': any(re.search(r'^\s+\d|\d\s+$', str(val)) for val in sample_values)
            }
            
            # Count convertible values
            potential_numeric_count = 0
            for val in sample_values[:10]:
                if self._can_be_converted_to_numeric(str(val)):
                    potential_numeric_count += 1
            
            conversion_ratio = potential_numeric_count / min(len(sample_values), 10)
            
            if conversion_ratio >= 0.7 and any(patterns_found.values()):
                analysis_results[col] = {
                    'sample_values': sample_values[:5],
                    'patterns': patterns_found,
                    'conversion_ratio': conversion_ratio,
                    'should_convert': True
                }
        
        return analysis_results
    
    def _can_be_converted_to_numeric(self, value: str) -> bool:
        """Check if a string value can be converted to numeric after cleaning"""
        if pd.isna(value) or value == '':
            return False
        
        cleaned = str(value).strip()
        
        # Remove symbols
        cleaned = re.sub(r'[\$€£¥₺₹%\|\-\#\*\?\@\&]', '', cleaned)
        cleaned = re.sub(r'\s*off$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*-\s*.*$', '', cleaned)
        
        # Skip pure text
        if re.match(r'^[A-Za-z\s]+$', cleaned):
            return False
        
        # Handle parentheses
        if re.match(r'^\([0-9,\.]+\)$', cleaned):
            cleaned = '-' + re.sub(r'[()]', '', cleaned)
        
        # Remove commas
        cleaned = cleaned.replace(',', '')
        
        # Try conversion
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def clean_numeric_columns(self) -> pd.DataFrame:
        """Clean object columns containing numeric data with symbols"""
        numeric_candidates = self.detect_numeric_in_object_columns()
        
        if not numeric_candidates:
            self.log_action("Numeric Cleaning", "No object columns with numeric data found")
            return self.df
        
        cleaning_log = []
        
        for col, analysis in numeric_candidates.items():
            if not analysis['should_convert']:
                continue
            
            try:
                original_sample = self.df[col].dropna().head(3).tolist()
                cleaned_series = self.df[col].astype(str).str.strip()
                
                # Clean step by step
                cleaned_series = cleaned_series.str.replace(r'\s*off$', '', regex=True, case=False)
                cleaned_series = cleaned_series.str.replace(r'\s*-\s*.*$', '', regex=True)
                
                # REMOVE PROBLEMATIC CHARACTERS (including pipe |)
                cleaned_series = cleaned_series.str.replace(r'[\|\-\#\*\?\@\&]', '', regex=True)
                
                # Remove text values and missing patterns
                text_mask = cleaned_series.str.match(r'^[A-Za-z\s]+$')
                cleaned_series.loc[text_mask] = pd.NA
                
                missing_patterns = ['', 'nan', 'null', 'none', 'n/a', 'na', 'missing', '?', '--', '..']
                for pattern in missing_patterns:
                    cleaned_series = cleaned_series.replace(pattern, pd.NA, regex=False)
                
                # Remove currency symbols
                cleaned_series = cleaned_series.str.replace(r'[\$€£¥₺₹]', '', regex=True)
                
                # Handle percentages
                is_percentage = analysis['patterns']['percentage']
                cleaned_series = cleaned_series.str.replace('%', '', regex=False)
                
                # Handle negative parentheses
                mask_parentheses = cleaned_series.str.match(r'^\([0-9,\.]+\)$')
                if mask_parentheses.any():
                    cleaned_series.loc[mask_parentheses] = '-' + cleaned_series.loc[mask_parentheses].str.replace(r'[()]', '', regex=True)
                
                # Remove commas
                cleaned_series = cleaned_series.str.replace(',', '', regex=False)
                
                # Convert to numeric
                numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                
                # Convert percentages to decimals (but not for ratings)
                if is_percentage and not col.lower() in ['rating', 'score', 'stars']:
                    numeric_series = numeric_series / 100
                
                # Calculate success rate
                original_non_null = self.df[col].dropna()
                if len(original_non_null) > 0:
                    success_rate = (numeric_series.notna().sum() / len(original_non_null)) * 100
                else:
                    success_rate = 0
                
                # Apply if successful
                if success_rate >= 50:
                    self.df[col] = numeric_series
                    
                    patterns = []
                    if analysis['patterns']['currency']: patterns.append("para sembolleri")
                    if analysis['patterns']['percentage']: patterns.append("yüzde işaretleri")  
                    if analysis['patterns']['thousands_separator']: patterns.append("binlik ayırıcılar")
                    
                    # Check if special characters were cleaned
                    if any(char in str(val) for val in original_sample for char in '|-#*?@&'):
                        patterns.append("özel karakterler")
                    
                    log_msg = f"'{col}': {', '.join(patterns)} temizlendi → sayısal ({success_rate:.1f}% başarı)"
                    if is_percentage and not col.lower() in ['rating', 'score', 'stars']:
                        log_msg += " [yüzde → ondalık]"
                    
                    cleaning_log.append(log_msg)
                    
                    cleaned_sample = numeric_series.dropna().head(3).tolist()
                    self.log_action(f"Numeric Conversion: {col}", f"Örnek: {original_sample} → {cleaned_sample}")
                else:
                    cleaning_log.append(f"'{col}': Dönüştürme başarısız ({success_rate:.1f}% başarı) - sütun korundu")
                    
            except Exception as e:
                cleaning_log.append(f"'{col}': Hata - {str(e)}")
        
        if cleaning_log:
            self.log_action("Numeric Data Cleaning", "; ".join(cleaning_log))
        
        return self.df
    
    def analyze_column_types(self) -> Dict[str, str]:
        """Enhanced AI agent with professional prompt for optimal data type analysis"""
        
        # Enhanced column analysis
        column_info = {}
        for col in self.df.columns:
            sample_values = self.df[col].dropna().head(50).tolist()  # Increased sampling
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            null_count = self.df[col].isnull().sum()
            current_dtype = str(self.df[col].dtype)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            column_info[col] = {
                "column_name": col,
                "current_dtype": current_dtype,
                "sample_values": sample_values[:15],
                "unique_count": unique_count,
                "total_count": total_count,
                "null_count": null_count,
                "null_percentage": round((null_count / total_count) * 100, 2),
                "unique_ratio": round(unique_ratio, 3)
            }
        
        # Professional AI prompt
        analysis_prompt = f"""
You are a senior data scientist specializing in data type optimization. Analyze the DataFrame columns and determine optimal data types.

COLUMN ANALYSIS DATA:
{json.dumps(column_info, indent=2, default=str)}

PROFESSIONAL DATA TYPE RULES:

🔢 NUMERIC TYPES:
- int64: Discrete integers (counts, years, ages, purely numeric IDs)
- float64: Continuous numeric data, ratings, prices, percentages, ANY numeric with decimals
  * CRITICAL: Ratings (4.2, 4.5) are ALWAYS float64, even if limited unique values
  * CRITICAL: Prices, amounts, percentages are ALWAYS float64

📝 TEXT TYPES:
- string: Text data, names, descriptions, URLs, mixed alphanumeric IDs
- category: Limited categories (<20% unique), status, departments, true categories, gender
  * WARNING: Do NOT use category for IDs, even if low cardinality

🆔 IDENTIFIERS:
- string: Default for all IDs (product_id, user_id, etc.) for flexibility

📅 TEMPORAL:
- datetime64: Dates, times, timestamps

✅ BOOLEAN:
- bool: Binary data (Yes/No, True/False, 1/0)

KEY DECISION RULES:
1. RATINGS/SCORES: Always float64 (mathematical operations required)
2. FINANCIAL DATA: Always float64 (arithmetic precision needed)
3. PERCENTAGES: Always float64 (decimal operations)
4. IDs: Prefer string (preserve format, avoid ordering)
5. Categories: Only for true categorical data with reasonable cardinality

Return ONLY a clean JSON object:
{{"column_name": "optimal_dtype", ...}}

Valid types: int64, float64, string, category, bool, datetime64
"""
        
        messages = [
            {"role": "system", "content": "You are a senior data scientist. Prioritize analytical utility over memory optimization."},
            {"role": "user", "content": analysis_prompt}
        ]
        
        try:
            response = call_selected_llm(messages, max_tokens=800, temperature=0.1)
            
            # Extract JSON
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    type_recommendations = json.loads(json_str)
                    if isinstance(type_recommendations, dict) and len(type_recommendations) > 0:
                        # Validate recommendations
                        valid_types = {'int64', 'float64', 'string', 'category', 'bool', 'datetime64'}
                        validated_recommendations = {}
                        
                        for col, dtype in type_recommendations.items():
                            if col in self.df.columns and dtype in valid_types:
                                validated_recommendations[col] = dtype
                        
                        if len(validated_recommendations) > 0:
                            return validated_recommendations
                except json.JSONDecodeError:
                    continue
            
            return self._enhanced_fallback_type_analysis()
                
        except Exception as e:
            self.log_action("AI Type Analysis Failed", f"Error: {str(e)}, using fallback")
            return self._enhanced_fallback_type_analysis()
    
    def _enhanced_fallback_type_analysis(self) -> Dict[str, str]:
        """Enhanced rule-based type analysis"""
        recommendations = {}
        
        for col in self.df.columns:
            col_lower = col.lower()
            unique_ratio = self.df[col].nunique() / len(self.df)
            
            # Enhanced rules
            if any(keyword in col_lower for keyword in ['id', 'code', 'key', 'reference']):
                recommendations[col] = 'string'
            elif any(keyword in col_lower for keyword in ['rating', 'score', 'grade', 'stars']):
                recommendations[col] = 'float64'  # Always float64 for ratings
            elif any(keyword in col_lower for keyword in ['price', 'cost', 'amount', 'revenue', 'salary']):
                recommendations[col] = 'float64'
            elif any(keyword in col_lower for keyword in ['percentage', 'percent', 'ratio', 'rate']):
                recommendations[col] = 'float64'
            elif any(keyword in col_lower for keyword in ['count', 'quantity', 'number', 'total']):
                recommendations[col] = 'int64'
            elif any(keyword in col_lower for keyword in ['date', 'time', 'timestamp']):
                recommendations[col] = 'datetime64'
            elif unique_ratio < 0.15:  # Very low cardinality
                recommendations[col] = 'category'
            else:
                recommendations[col] = 'string'
        
        return recommendations
    
    def apply_data_type_conversions(self) -> pd.DataFrame:
        """Apply AI-recommended data type conversions"""
        type_recommendations = self.analyze_column_types()
        conversion_log = []
        
        for col, recommended_type in type_recommendations.items():
            current_type = str(self.df[col].dtype)
            
            if current_type != recommended_type:
                try:
                    # Skip if already numeric
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        conversion_log.append(f"{col}: Already numeric ({current_type}), skipping")
                        continue
                    
                    if recommended_type == 'int64':
                        test_conversion = pd.to_numeric(self.df[col], errors='coerce')
                        success_rate = (test_conversion.notna().sum() / len(self.df[col].dropna())) * 100
                        
                        if success_rate >= 80:
                            self.df[col] = test_conversion.fillna(0).astype('int64')
                            conversion_log.append(f"{col}: {current_type} → {recommended_type}")
                        else:
                            self.df[col] = self.df[col].astype('string')
                            conversion_log.append(f"{col}: {current_type} → string (int64 failed)")
                            
                    elif recommended_type == 'float64':
                        test_conversion = pd.to_numeric(self.df[col], errors='coerce')
                        success_rate = (test_conversion.notna().sum() / len(self.df[col].dropna())) * 100
                        
                        if success_rate >= 80:
                            self.df[col] = test_conversion
                            conversion_log.append(f"{col}: {current_type} → {recommended_type}")
                        else:
                            self.df[col] = self.df[col].astype('string')
                            conversion_log.append(f"{col}: {current_type} → string (float64 failed)")
                            
                    elif recommended_type == 'category':
                        unique_ratio = self.df[col].nunique() / len(self.df)
                        if unique_ratio < 0.5:
                            self.df[col] = self.df[col].astype('category')
                            conversion_log.append(f"{col}: {current_type} → {recommended_type}")
                        else:
                            self.df[col] = self.df[col].astype('string')
                            conversion_log.append(f"{col}: {current_type} → string (too many categories)")
                        
                    elif recommended_type == 'datetime64':
                        test_conversion = pd.to_datetime(self.df[col], errors='coerce')
                        success_rate = (test_conversion.notna().sum() / len(self.df[col].dropna())) * 100
                        
                        if success_rate >= 80:
                            self.df[col] = test_conversion
                            conversion_log.append(f"{col}: {current_type} → {recommended_type}")
                        else:
                            self.df[col] = self.df[col].astype('string')
                            conversion_log.append(f"{col}: {current_type} → string (datetime failed)")
                            
                    elif recommended_type == 'string':
                        self.df[col] = self.df[col].astype('string')
                        conversion_log.append(f"{col}: {current_type} → {recommended_type}")
                    
                except Exception as e:
                    conversion_log.append(f"{col}: Conversion failed ({str(e)})")
        
        self.log_action("Data Type Conversions", "; ".join(conversion_log))
        return self.df
    
    def analyze_missing_data_strategy(self, column: str) -> str:
        """AI agent to determine optimal missing data filling strategy"""
        col_data = self.df[column]
        missing_count = col_data.isnull().sum()
        missing_percentage = (missing_count / len(col_data)) * 100
        
        if missing_percentage == 0:
            return "no_missing"
        
        # Simple strategy based on data type and column name
        col_lower = column.lower()
        
        if pd.api.types.is_numeric_dtype(col_data):
            if any(keyword in col_lower for keyword in ['count', 'quantity', 'number']):
                return 'zero'
            elif col_data.skew() > 1:
                return 'median'
            else:
                return 'mean'
        else:
            if any(keyword in col_lower for keyword in ['name', 'city', 'address', 'id']):
                return 'unknown'
            else:
                return 'mode'
    
    def handle_missing_data(self) -> pd.DataFrame:
        """Handle missing data with improved thresholds"""
        missing_info = []
        
        # Remove columns with >50% missing
        columns_to_drop = []
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            if missing_pct > 50:
                columns_to_drop.append(col)
                missing_info.append(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
        
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
        
        # Remove rows with >25% missing
        missing_threshold = 0.25 * len(self.df.columns)
        rows_before = len(self.df)
        self.df = self.df.dropna(thresh=len(self.df.columns) - missing_threshold)
        rows_dropped = rows_before - len(self.df)
        
        if rows_dropped > 0:
            missing_info.append(f"Dropped {rows_dropped} rows with >25% missing values")
        
        # Fill remaining missing values
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                strategy = self.analyze_missing_data_strategy(col)
                
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(fill_value)
                    missing_info.append(f"'{col}': filled with mean ({fill_value:.2f})")
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_value)
                    missing_info.append(f"'{col}': filled with median ({fill_value:.2f})")
                elif strategy == 'mode':
                    if not self.df[col].mode().empty:
                        fill_value = self.df[col].mode()[0]
                        self.df[col] = self.df[col].fillna(fill_value)
                        missing_info.append(f"'{col}': filled with mode ('{fill_value}')")
                    else:
                        self.df[col] = self.df[col].fillna('Unknown')
                        missing_info.append(f"'{col}': filled with 'Unknown'")
                elif strategy == 'zero':
                    self.df[col] = self.df[col].fillna(0)
                    missing_info.append(f"'{col}': filled with 0")
                elif strategy == 'unknown':
                    self.df[col] = self.df[col].fillna('Bilinmiyor')
                    missing_info.append(f"'{col}': filled with 'Bilinmiyor'")
        
        self.log_action("Missing Data Handling", "; ".join(missing_info))
        return self.df
    
    def run_full_preprocessing(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Run complete preprocessing pipeline"""
        self.log_action("Preprocessing Started", f"Original shape: {self.original_df.shape}")
        
        # Step 1: Clean numeric data
        self.clean_numeric_columns()
        self.log_action("Numeric Cleaning Complete", f"Shape: {self.df.shape}")
        
        # Step 2: Handle missing data
        self.handle_missing_data()
        self.log_action("Missing Data Complete", f"Shape: {self.df.shape}")
        
        # Step 3: Optimize data types
        self.apply_data_type_conversions()
        self.log_action("Data Types Complete", f"Final shape: {self.df.shape}")
        
        # Summary
        memory_before = self.original_df.memory_usage(deep=True).sum() / 1024**2
        memory_after = self.df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = memory_before - memory_after
        
        self.log_action("Preprocessing Complete", f"Memory: {memory_before:.2f}MB → {memory_after:.2f}MB")
        
        return self.df, self.preprocessing_log

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Main preprocessing function with enhanced reporting"""
    agent = DataPreprocessingAgent(df)
    processed_df, log = agent.run_full_preprocessing()
    
    # Enhanced summary report
    summary_parts = []
    summary_parts.append(f"**🔄 Veri Ön İşleme Tamamlandı**\n")
    summary_parts.append(f"• **Orijinal boyut:** {agent.original_df.shape[0]:,} satır × {agent.original_df.shape[1]} sütun")
    summary_parts.append(f"• **İşlenmiş boyut:** {processed_df.shape[0]:,} satır × {processed_df.shape[1]} sütun")
    
    memory_before = agent.original_df.memory_usage(deep=True).sum() / 1024**2
    memory_after = processed_df.memory_usage(deep=True).sum() / 1024**2
    summary_parts.append(f"• **Hafıza kullanımı:** {memory_before:.1f}MB → {memory_after:.1f}MB")
    
    # Data type changes
    dtype_changes = []
    for col in processed_df.columns:
        if col in agent.original_df.columns:
            old_type = str(agent.original_df[col].dtype)
            new_type = str(processed_df[col].dtype)
            if old_type != new_type:
                dtype_changes.append(f"{col}: {old_type}→{new_type}")
    
    if dtype_changes:
        summary_parts.append(f"\n**📊 Veri Tipi Değişiklikleri:**")
        for change in dtype_changes[:5]:
            summary_parts.append(f"• {change}")
        if len(dtype_changes) > 5:
            summary_parts.append(f"• ... ve {len(dtype_changes)-5} tane daha")
    
    # Numeric cleaning summary
    numeric_actions = [action for action in log if "Numeric" in action["action"]]
    if numeric_actions:
        summary_parts.append(f"\n**🧮 Sayısal Veri Temizleme:**")
        for action in numeric_actions:
            if action["details"] and "temizlendi" in action["details"]:
                details = action["details"].split(";")
                for detail in details[:3]:
                    summary_parts.append(f"• {detail.strip()}")
                if len(details) > 3:
                    summary_parts.append(f"• ... ve {len(details)-3} işlem daha")
    
    # Missing data summary
    missing_actions = [action for action in log if "Missing Data" in action["action"]]
    if missing_actions:
        summary_parts.append(f"\n**🔧 Eksik Veri İşlemleri:**")
        for action in missing_actions:
            if action["details"]:
                details = action["details"].split(";")
                for detail in details[:3]:
                    summary_parts.append(f"• {detail.strip()}")
                if len(details) > 3:
                    summary_parts.append(f"• ... ve {len(details)-3} işlem daha")
    
    # Data loss analysis for rating
    original_rating_count = agent.original_df['rating'].notna().sum() if 'rating' in agent.original_df.columns else 0
    processed_rating_count = processed_df['rating'].notna().sum() if 'rating' in processed_df.columns else 0
    
    if original_rating_count > 0 and processed_rating_count < original_rating_count:
        summary_parts.append(f"\n**⚠️ VERİ KAYBI TESPİT EDİLDİ:**")
        summary_parts.append(f"• **Rating sütunu:** {original_rating_count} → {processed_rating_count} "
                            f"({original_rating_count - processed_rating_count} kayıp)")
    elif original_rating_count > 0 and processed_rating_count == original_rating_count:
        summary_parts.append(f"\n**✅ VERİ KORUNDU:**")
        summary_parts.append(f"• **Rating sütunu:** {original_rating_count} → {processed_rating_count} (kayıp yok)")
    
    summary_parts.append(f"\n✅ **Veri seti analiz için hazır!**")
    summary_report = "\n".join(summary_parts)
    
    return processed_df, summary_report