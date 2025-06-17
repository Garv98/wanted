import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Install GPU libraries if in Colab
# if 'google.colab' in sys.modules:
#     print("Installing GPU-accelerated libraries...")
#     !pip install -q cudf-cu12 cuml-cu12 cudatoolkit==12 cupy-cuda12x
#     print("GPU libraries installed successfully!")
#
# # Import GPU-accelerated libraries
# try:
#     import cudf
#     import cupy as cp
#     from cuml.preprocessing import LabelEncoder
#     from cuml.ensemble import IsolationForest
#     GPU_ENABLED = True
#     print("GPU acceleration: ENABLED")
# except ImportError:
#     from sklearn.preprocessing import LabelEncoder
#     from sklearn.ensemble import IsolationForest
#     GPU_ENABLED = False
#     print("GPU acceleration: DISABLED")

# Fix: Always define GPU_ENABLED and fallback imports at the top
try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import LabelEncoder
    from cuml.ensemble import IsolationForest
    GPU_ENABLED = True
    print("GPU acceleration: ENABLED")
except ImportError:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import IsolationForest
    GPU_ENABLED = False
    print("GPU acceleration: DISABLED")

class CrimeGenomeAnalyzer:
    """
    CRIME GENOME ANALYZER - GPU Optimized Version
    "Dissect crime to the atom. Spin the helix, trace the pattern."
    """

    def __init__(self, csv_path=None):
        self.df = None
        self.gdf = None  # GPU DataFrame
        self.dna_matrix = None
        self.encoders = {}
        self.feature_ranges = {}  # <-- ensure this is always defined

        # DNA features with memory-efficient types
        self.dna_features = [
            'Hour', 'Victim_Age', 'City_Code', 'Weapon_Code',
            'Domain_Code', 'Police_Deployed', 'Case_Closed_Binary'
        ]
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

        # Always use the data/crime_dataset_india_raw.csv file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
        data_path = os.path.join(base_dir, "crime_dataset_india_raw.csv")
        if os.path.isfile(data_path):
            csv_path = data_path
        else:
            print(f"CrimeGenomeAnalyzer: CSV file not found at '{data_path}'. Please place 'crime_dataset_india_raw.csv' in the 'data' folder.")

        if csv_path and os.path.isfile(csv_path):
            self.load_data(csv_path)
        else:
            print(f"CrimeGenomeAnalyzer: CSV file not found at '{csv_path}'.")

    def load_data(self, csv_path):
        """Load and preprocess crime data with GPU acceleration"""
        print("Loading Crime Genome Data...")
        try:
            if GPU_ENABLED:
                # Load directly to GPU memory
                self.gdf = cudf.read_csv(csv_path)
                print(f"Loaded {len(self.gdf)} crime records to GPU")
                # Keep only a small subset in CPU memory for visualization
                self.df = self.gdf.head(1000).to_pandas()
            else:
                self.df = pd.read_csv(csv_path)
                print(f"Loaded {len(self.df)} crime records")

            self.preprocess_data()
            self.build_dna_matrix()
            self.compute_feature_ranges()  # <-- ensure feature_ranges is always set
            print("Crime DNA Matrix constructed successfully!")
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        return True

    def compute_feature_ranges(self):
        """Compute min/max for all numerical features in dna_features (except category codes)"""
        self.feature_ranges = {}
        for feature in self.dna_features:
            if feature in ['City_Code', 'Weapon_Code', 'Domain_Code']:
                continue
            # Defensive: skip if feature not in df
            if self.df is not None and feature in self.df.columns:
                col = self.df[feature]
                self.feature_ranges[feature] = (col.min(), col.max())
            else:
                self.feature_ranges[feature] = (0, 1)  # fallback

    def preprocess_data(self):
        """Clean and prepare data with GPU optimizations"""
        print("Preprocessing Crime DNA...")
        df = self.gdf if GPU_ENABLED else self.df

        # Handle missing values
        df['Date Case Closed'] = df['Date Case Closed'].fillna('')
        df['Case Closed'] = df['Case Closed'].fillna('No')

        # Time processing
        try:
            if GPU_ENABLED:
                df['Time of Occurrence'] = cudf.to_datetime(
                    df['Time of Occurrence'],
                    format='%d-%m-%Y %H:%M',
                    errors='coerce'
                )
                df['Hour'] = df['Time of Occurrence'].dt.hour.fillna(12)
            else:
                df['Time of Occurrence'] = pd.to_datetime(
                    df['Time of Occurrence'],
                    format='%d-%m-%Y %H:%M',
                    errors='coerce'
                )
                df['Hour'] = df['Time of Occurrence'].dt.hour.fillna(12)
        except Exception as e:
            print(f"Time parsing error: {e}, using random hours")
            if GPU_ENABLED:
                df['Hour'] = cp.random.randint(0, 24, len(df))
            else:
                df['Hour'] = np.random.randint(0, 24, len(df))

        # Numeric columns with memory-efficient types
        df['Victim_Age'] = df['Victim Age'].astype('float32').fillna(30)
        df['Police_Deployed'] = df['Police Deployed'].astype('int16').fillna(5)

        # Categorical columns
        categorical_cols = ['City', 'Weapon Used', 'Crime Domain', 'Victim Gender']
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')

        self.create_encodings()
        print(f"Preprocessed {len(df)} records successfully")

    def create_encodings(self):
        """GPU-accelerated label encoding"""
        df = self.gdf if GPU_ENABLED else self.df

        if GPU_ENABLED:
            # GPU encoding with cuML
            df['City_Code'] = LabelEncoder().fit_transform(df['City'].astype('str'))
            df['Weapon_Code'] = LabelEncoder().fit_transform(df['Weapon Used'].astype('str'))
            df['Domain_Code'] = LabelEncoder().fit_transform(df['Crime Domain'].astype('str'))
        else:
            # CPU encoding
            le = LabelEncoder()
            df['City_Code'] = le.fit_transform(df['City'].astype(str))
            df['Weapon_Code'] = le.fit_transform(df['Weapon Used'].astype(str))
            df['Domain_Code'] = le.fit_transform(df['Crime Domain'].astype(str))

        # Binary encoding with efficient type
        df['Case_Closed_Binary'] = (df['Case Closed'] == 'Yes').astype('int8')

    def build_dna_matrix(self):
        """Build DNA matrix with GPU acceleration"""
        print("Building DNA Helix Matrix...")
        df = self.gdf if GPU_ENABLED else self.df

        try:
            features = []
            for feature in self.dna_features:
                values = df[feature].astype('float32')

                # GPU normalization
                if GPU_ENABLED:
                    min_val = values.min()
                    max_val = values.max()
                    if max_val != min_val:
                        normalized = (values - min_val) / (max_val - min_val)
                    else:
                        normalized = cudf.Series(cp.zeros(len(df)), dtype='float32')
                    features.append(normalized)
                # CPU normalization
                else:
                    values_np = values.values
                    min_val = values_np.min()
                    max_val = values_np.max()
                    if max_val != min_val:
                        normalized = (values_np - min_val) / (max_val - min_val)
                    else:
                        normalized = np.zeros(len(df))
                    features.append(normalized)

            # Build matrix on GPU/CPU
            if GPU_ENABLED:
                self.dna_matrix = cudf.concat(features, axis=1).values
            else:
                self.dna_matrix = np.column_stack(features)

            print(f"DNA Matrix Shape: {self.dna_matrix.shape}")
        except Exception as e:
            print(f"Error building DNA matrix: {e}")
            if GPU_ENABLED:
                self.dna_matrix = cp.random.rand(len(df), len(self.dna_features)).astype('float32')
            else:
                self.dna_matrix = np.random.rand(len(df), len(self.dna_features)).astype('float32')

    def visualize_crime_dna_simple(self, crime_id=0):
        """Simple DNA visualization"""
        if self.dna_matrix is None:
            print("Please load data first!")
            return None

        # Get crime data from CPU copy
        if crime_id >= len(self.df):
            print(f"Crime ID {crime_id} not found in visualization cache. Using first record.")
            crime_id = 0

        crime_info = self.df.iloc[crime_id]

        # Get DNA sequence - if GPU, convert to numpy
        if GPU_ENABLED:
            dna_sequence = self.dna_matrix[crime_id].get() if isinstance(self.dna_matrix, cp.ndarray) else self.dna_matrix[crime_id]
        else:
            dna_sequence = self.dna_matrix[crime_id]

        # Create simple bar chart DNA visualization
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=self.dna_features,
            y=dna_sequence,
            marker=dict(
                color=self.colors[:len(self.dna_features)],
                opacity=0.8
            ),
            text=[f"{val:.2f}" for val in dna_sequence],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Crime DNA Profile - Case #{crime_info['Report Number']}<br>" +
                  f"<sub>{crime_info['Crime Description']} in {crime_info['City']}</sub>",
            xaxis_title="DNA Strands",
            yaxis_title="Genetic Expression",
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white', size=12),
            width=800,
            height=500,
            xaxis=dict(tickangle=45)
        )

        return fig

    def compare_crimes_simple(self, crime_id1, crime_id2):
        """Simple crime comparison"""
        if self.dna_matrix is None:
            print("Please load data first!")
            return None, 0

        # Get crime data from CPU copy
        max_id = len(self.df) - 1
        if crime_id1 > max_id or crime_id2 > max_id:
            print(f"Crime IDs exceed visualization cache. Using first record.")
            crime_id1, crime_id2 = 0, 1

        crime1 = self.df.iloc[crime_id1]
        crime2 = self.df.iloc[crime_id2]

        # Get DNA sequences
        if GPU_ENABLED:
            dna1 = self.dna_matrix[crime_id1].get() if isinstance(self.dna_matrix, cp.ndarray) else self.dna_matrix[crime_id1]
            dna2 = self.dna_matrix[crime_id2].get() if isinstance(self.dna_matrix, cp.ndarray) else self.dna_matrix[crime_id2]
        else:
            dna1 = self.dna_matrix[crime_id1]
            dna2 = self.dna_matrix[crime_id2]

        # Calculate similarity
        similarity = np.dot(dna1, dna2) / (np.linalg.norm(dna1) * np.linalg.norm(dna2))

        # Create comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name=f'Crime {crime_id1}',
            x=self.dna_features,
            y=dna1,
            marker=dict(color='#FF6B6B', opacity=0.7)
        ))

        fig.add_trace(go.Bar(
            name=f'Crime {crime_id2}',
            x=self.dna_features,
            y=dna2,
            marker=dict(color='#4ECDC4', opacity=0.7)
        ))

        fig.update_layout(
            title=f"DNA Comparison - Similarity: {similarity:.3f}<br>" +
                  f"<sub>Crime {crime_id1}: {crime1['Crime Description']} vs Crime {crime_id2}: {crime2['Crime Description']}</sub>",
            barmode='group',
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            width=900,
            height=500,
            xaxis=dict(tickangle=45)
        )

        return fig, similarity

    def detect_mutations(self):
        """Detect crime anomalies with GPU acceleration"""
        if self.dna_matrix is None:
            print("Please build DNA matrix first!")
            return []

        print("Detecting Crime Mutations...")

        try:
            if GPU_ENABLED:
                # GPU Isolation Forest
                detector = IsolationForest(
                    contamination=0.05,  # Lower for large datasets
                    random_state=42,
                    output_type='cudf'
                )
                outliers = detector.fit_predict(self.dna_matrix)
                anomaly_indices = cp.where(outliers == -1)[0].get()
            else:
                # CPU Isolation Forest
                detector = IsolationForest(contamination=0.05, random_state=42)
                outliers = detector.fit_predict(self.dna_matrix)
                anomaly_indices = np.where(outliers == -1)[0]

            print(f"Found {len(anomaly_indices)} crime mutations")
            return anomaly_indices
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return []

    def show_crime_summary(self, crime_id):
        """Show simple crime summary"""
        if crime_id >= len(self.df):
            print(f"Crime ID {crime_id} not found. Max ID: {len(self.df)-1}")
            return

        crime = self.df.iloc[crime_id]

        print(f"\nCRIME DNA REPORT - Case #{crime['Report Number']}")
        print("=" * 50)
        print(f"Crime: {crime['Crime Description']}")
        print(f"City: {crime['City']}")
        print(f"Victim: Age {crime['Victim_Age']}, {crime['Victim Gender']}")
        print(f"Weapon: {crime['Weapon Used']}")
        print(f"Domain: {crime['Crime Domain']}")
        print(f"Police: {crime['Police_Deployed']} officers")
        print(f"Status: {crime['Case Closed']}")

        # Show DNA sequence
        if self.dna_matrix is not None:
            if GPU_ENABLED:
                dna = self.dna_matrix[crime_id].get() if isinstance(self.dna_matrix, cp.ndarray) else self.dna_matrix[crime_id]
            else:
                dna = self.dna_matrix[crime_id]
            print(f"\nDNA SEQUENCE:")
            for feature, value in zip(self.dna_features, dna):
                print(f"   {feature}: {value:.3f}")

        print("=" * 50)
        
    def find_similar_crimes(self, user_crime, top_n=5):
        """Find most similar crimes to user-provided details"""
        if self.dna_matrix is None:
            print("Please load data first!")
            return [], []

        # Build user crime DNA vector
        user_vector = []
        for feature in self.dna_features:
            if feature == 'City_Code':
                user_vector.append(self._encode_category(user_crime['City'], 'City'))
            elif feature == 'Weapon_Code':
                user_vector.append(self._encode_category(user_crime['Weapon Used'], 'Weapon Used'))
            elif feature == 'Domain_Code':
                user_vector.append(self._encode_category(user_crime['Crime Domain'], 'Crime Domain'))
            else:
                # Normalize numerical features using stored ranges
                raw_value = user_crime[feature]
                min_val, max_val = self.feature_ranges[feature]
                if max_val == min_val:
                    user_vector.append(0.0)
                else:
                    user_vector.append((raw_value - min_val) / (max_val - min_val))
        
        user_vector = np.array(user_vector, dtype='float32')
        
        # Compute similarities
        if GPU_ENABLED:
            user_vector_gpu = cp.array(user_vector)
            matrix_gpu = cp.array(self.dna_matrix)
            dot_products = user_vector_gpu @ matrix_gpu.T
            user_norm = cp.linalg.norm(user_vector_gpu)
            matrix_norms = cp.linalg.norm(matrix_gpu, axis=1)
            similarities = dot_products / (user_norm * matrix_norms)
            similarities = cp.asnumpy(similarities)
        else:
            dot_products = np.dot(self.dna_matrix, user_vector)
            user_norm = np.linalg.norm(user_vector)
            matrix_norms = np.linalg.norm(self.dna_matrix, axis=1)
            similarities = dot_products / (user_norm * matrix_norms)
        
        # Handle NaN values
        similarities = np.nan_to_num(similarities, nan=0.0)
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return top_indices, similarities[top_indices]

    def interactive_explorer(self):
        """Simplified interactive explorer"""
        if self.df is None:
            print("No data loaded!")
            return

        print("\nCRIME GENOME EXPLORER")
        print("=" * 40)
        print("Commands:")
        print("  show <id>     - Show crime details")
        print("  dna <id>      - View DNA chart")
        print("  compare <id1> <id2> - Compare crimes")
        print("  mutations     - Find anomalies")
        print("  random        - Analyze random crime")
        print("  stats         - Show database stats")
        print("  help          - Show this help")
        print("  quit          - Exit")
        print("-" * 40)

        while True:
            try:
                command = input("\nCommand: ").strip().lower()

                if command in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                elif command == 'help':
                    print("\nAvailable Commands:")
                    print("  show 0        - Show details for crime ID 0")
                    print("  dna 0         - Show DNA chart for crime ID 0")
                    print("  compare 0 1   - Compare crimes 0 and 1")
                    print("  mutations     - Find unusual crimes")
                    print("  random        - Pick random crime to analyze")
                    print("  stats         - Database statistics")

                elif command == 'stats':
                    print(f"\nDATABASE STATISTICS:")
                    print(f"   Total Crimes: {len(self.gdf) if GPU_ENABLED else len(self.df):,}")
                    print(f"   Cities: {self.gdf['City'].nunique() if GPU_ENABLED else self.df['City'].nunique()}")
                    print(f"   Crime Types: {self.gdf['Crime Description'].nunique() if GPU_ENABLED else self.df['Crime Description'].nunique()}")
                    closed_count = (self.gdf['Case Closed'] == 'Yes').sum() if GPU_ENABLED else (self.df['Case Closed'] == 'Yes').sum()
                    open_count = (self.gdf['Case Closed'] == 'No').sum() if GPU_ENABLED else (self.df['Case Closed'] == 'No').sum()
                    print(f"   Closed Cases: {closed_count:,}")
                    print(f"   Open Cases: {open_count:,}")

                    if GPU_ENABLED:
                        print(f"   Most Common Crime: {self.gdf['Crime Description'].mode().iloc[0]}")
                        print(f"   Most Common City: {self.gdf['City'].mode().iloc[0]}")
                    else:
                        print(f"   Most Common Crime: {self.df['Crime Description'].mode().iloc[0]}")
                        print(f"   Most Common City: {self.df['City'].mode().iloc[0]}")

                elif command == 'random':
                    crime_id = np.random.randint(0, len(self.df))
                    print(f"\nRandom Crime Selected: ID {crime_id}")
                    self.show_crime_summary(crime_id)

                elif command.startswith('show '):
                    try:
                        crime_id = int(command.split()[1])
                        self.show_crime_summary(crime_id)
                    except (ValueError, IndexError):
                        print("Usage: show <crime_id>")

                elif command.startswith('dna '):
                    try:
                        crime_id = int(command.split()[1])
                        fig = self.visualize_crime_dna_simple(crime_id)
                        if fig:
                            fig.show()
                            self.show_crime_summary(crime_id)
                    except (ValueError, IndexError):
                        print("Usage: dna <crime_id>")

                elif command.startswith('compare '):
                    try:
                        parts = command.split()
                        id1, id2 = int(parts[1]), int(parts[2])
                        fig, similarity = self.compare_crimes_simple(id1, id2)
                        if fig:
                            fig.show()
                            print(f"\nSimilarity Score: {similarity:.3f}")
                            if similarity > 0.8:
                                print("Very similar DNA patterns!")
                            elif similarity > 0.6:
                                print("Moderately similar patterns")
                            else:
                                print("Different crime patterns")
                    except (ValueError, IndexError):
                        print("Usage: compare <id1> <id2>")

                elif command == 'mutations':
                    anomalies = self.detect_mutations()
                    if len(anomalies) > 0:
                        print(f"\nCRIME MUTATIONS FOUND:")
                        # Get first 5 from CPU cache
                        for i, idx in enumerate(anomalies[:5]):
                            if idx < len(self.df):
                                crime = self.df.iloc[idx]
                                print(f"   {i+1}. ID {idx}: {crime['Crime Description']} in {crime['City']}")

                        if len(anomalies) > 5:
                            print(f"   ... and {len(anomalies)-5} more")
                    else:
                        print("No mutations detected")

                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _encode_category(self, value, category_type):
        """Encode categorical values consistently with training data"""
        if category_type in self.encoders:
            try:
                # For sklearn LabelEncoder (CPU)
                return self.encoders[category_type].transform([value])[0]
            except Exception:
                # Handle unseen categories
                if hasattr(self.encoders[category_type], 'classes_') and 'Unknown' in self.encoders[category_type].classes_:
                    return self.encoders[category_type].transform(['Unknown'])[0]
                return 0
        return 0
