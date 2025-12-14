# -*- coding: utf-8 -*-
"""Original file is located at
    https://colab.research.google.com/drive/1Xw3cvM9wF8uVdm5zDObg4s7xuW5JOkCV
"""


randomstateee = 12
testsizeee = 0.25
ratiooo = 1.2


def denevekaydet(randomstateee, testsizeee, ratiooo):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    from datetime import datetime
   
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
                                VotingClassifier, StackingClassifier)
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_recall_fscore_support, f1_score)
    from sklearn.pipeline import Pipeline
    from scipy.sparse import hstack
    from scipy.optimize import minimize
    import re
    import warnings
    import time
    from collections import defaultdict

    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    warnings.filterwarnings('ignore')
    print("Paketler yüklendi")

    def veri_tipleri_analiz(df):
        """
        DataFrame'deki tüm kolonların veri tiplerini detaylı şekilde analiz eder.

        Parameters:
        -----------
        df : pandas.DataFrame
            Analiz edilecek DataFrame

        Returns:
        --------
        pandas.DataFrame
            Kolon adı, veri tipi, benzersiz değer sayısı ve örnek değerler içeren özet tablo
        """

        analiz = []

        for kolon in df.columns:
            veri_tipi = df[kolon].dtype
            benzersiz_sayi = df[kolon].nunique()
            null_sayi = df[kolon].isnull().sum()
            null_oran = (null_sayi / len(df)) * 100

            # İlk birkaç benzersiz değeri al (örnek olarak)
            ornekler = df[kolon].dropna().unique()[:3]
            ornek_str = ', '.join([str(x) for x in ornekler])

            analiz.append({
                'Kolon': kolon,
                'Veri_Tipi': str(veri_tipi),
                'Benzersiz_Sayi': benzersiz_sayi,
                'Null_Sayi': null_sayi,
                'Null_Oran_%': round(null_oran, 2),
                'Ornek_Degerler': ornek_str
            })

        sonuc = pd.DataFrame(analiz)

        # Özet bilgi
        print("=" * 80)
        print("VERİ TİPLERİ ANALİZİ")
        print("=" * 80)
        print(f"Toplam Satır Sayısı: {len(df):,}")
        print(f"Toplam Kolon Sayısı: {len(df.columns)}")
        print("\nVeri Tipi Dağılımı:")
        print(sonuc['Veri_Tipi'].value_counts())
        print("=" * 80)
        print()

        return sonuc

    # Kullanım:
    # tipler = veri_tipleri_analiz(df)
    # print(tipler.to_string())

    # @title
    def gelismis_korelasyon_haritasi(df):

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import pointbiserialr, chi2_contingency
        from sklearn.preprocessing import LabelEncoder

        kolonlar = df.columns.tolist()
        n = len(kolonlar)


        korelasyon_matrisi = np.zeros((n, n))


        for i, kolon1 in enumerate(kolonlar):
            for j, kolon2 in enumerate(kolonlar):
                if i == j:
                    korelasyon_matrisi[i, j] = 1.0
                elif i > j:
                    korelasyon_matrisi[i, j] = korelasyon_matrisi[j, i]
                else:
                    try:

                        if pd.api.types.is_numeric_dtype(df[kolon1]) and pd.api.types.is_numeric_dtype(df[kolon2]):
                            korelasyon_matrisi[i, j] = df[[kolon1, kolon2]].corr().iloc[0, 1]


                        elif not pd.api.types.is_numeric_dtype(df[kolon1]) and not pd.api.types.is_numeric_dtype(df[kolon2]):

                            confusion_matrix = pd.crosstab(df[kolon1], df[kolon2])
                            chi2 = chi2_contingency(confusion_matrix)[0]
                            n_obs = confusion_matrix.sum().sum()
                            phi2 = chi2 / n_obs
                            r, k = confusion_matrix.shape
                            korelasyon_matrisi[i, j] = np.sqrt(phi2 / min(k-1, r-1))


                        else:

                            if pd.api.types.is_numeric_dtype(df[kolon1]):
                                sayisal = df[kolon1].dropna()
                                kategorik = df[kolon2].dropna()
                            else:
                                sayisal = df[kolon2].dropna()
                                kategorik = df[kolon1].dropna()


                            ortak_index = sayisal.index.intersection(kategorik.index)

                            if len(ortak_index) > 0:
                                le = LabelEncoder()
                                kategorik_encoded = le.fit_transform(kategorik[ortak_index].astype(str))


                                kategoriler = pd.Series(kategorik_encoded)
                                degerler = sayisal[ortak_index]

                                genel_ortalama = degerler.mean()
                                kategori_ortalamalar = degerler.groupby(kategoriler).mean()
                                kategori_sayilari = kategoriler.value_counts()

                                toplam_varyans = ((degerler - genel_ortalama) ** 2).sum()
                                grup_ici_varyans = sum(
                                    kategori_sayilari[kat] * (kategori_ortalamalar[kat] - genel_ortalama) ** 2
                                    for kat in kategori_ortalamalar.index
                                )

                                eta = np.sqrt(grup_ici_varyans / toplam_varyans) if toplam_varyans > 0 else 0
                                korelasyon_matrisi[i, j] = eta
                            else:
                                korelasyon_matrisi[i, j] = 0

                    except Exception as e:
                        korelasyon_matrisi[i, j] = 0


        korelasyon_df = pd.DataFrame(korelasyon_matrisi, index=kolonlar, columns=kolonlar)


        plt.figure(figsize=(max(14, n), max(12, n * 0.8)))
        sns.heatmap(korelasyon_df, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1)
        plt.title('Gelişmiş Korelasyon Matrisi (Tüm Kolon Tipleri)', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return korelasyon_df

    # @title
    def drop_uncorrelated_columns(df, target_column, threshold=0.00):
        """
    # Usage
    df = drop_uncorrelated_columns(df, 'price')
    df = drop_uncorrelated_columns(df, 'sales', threshold=0.05)
    """


        if target_column not in df.columns:
            print(f"Error: Column '{target_column}' not found in DataFrame!")
            return df

        if not pd.api.types.is_numeric_dtype(df[target_column]):
            print(f"Error: Column '{target_column}' must be numeric!")
            return df

        numeric_df = df.select_dtypes(include=[np.number])

        correlations = numeric_df.corr()[target_column].abs()

        columns_to_drop = correlations[
            (correlations < threshold) & (correlations.index != target_column)
        ].index.tolist()

        print(f"\n{'='*80}")
        print(f"Correlation Analysis with '{target_column}'")
        print(f"{'='*80}")
        print(f"\nThreshold: {threshold}")
        print(f"\nColumns to drop ({len(columns_to_drop)}):")

        for col in columns_to_drop:
            corr_value = correlations[col]
            print(f"   • {col:<40} → Correlation: {corr_value:.4f}")

        if not columns_to_drop:
            print("   No columns to drop. All numeric columns have correlation >= threshold.")

        kept_columns = [col for col in numeric_df.columns if col not in columns_to_drop]
        print(f"\nColumns kept ({len(kept_columns)}):")
        for col in kept_columns:
            if col != target_column:
                corr_value = correlations[col]
                print(f"   • {col:<40} → Correlation: {corr_value:.4f}")

        print(f"{'='*80}\n")

        df_cleaned = df.drop(columns=columns_to_drop)

        print(f"Original shape: {df.shape}")
        print(f"New shape: {df_cleaned.shape}")
        print(f"Dropped {len(columns_to_drop)} columns\n")

        return df_cleaned


        """
        Kolonları kategorikleştirir ve isteğe bağlı encode eder.

        Parameters:
        df : pandas.DataFrame
        encode : bool
            True ise kategorileri sayısal değerlere çevirir

        Returns:
        pandas.DataFrame
        df = kategorize_et(df)  # Sadece kategorikleştir
        df = kategorize_et(df, encode=True)  # Kategorikleştir ve encode et
        """


        df_cat = df.copy()
        le = LabelEncoder()

        print(f"\nKategorikleştirme (Encode: {encode})...\n")

        for kolon in df.columns:

            if pd.api.types.is_numeric_dtype(df[kolon]):
                try:
                    df_cat[kolon] = pd.cut(df[kolon], bins=5, duplicates='drop')

                    if encode:
                        df_cat[kolon] = le.fit_transform(df_cat[kolon].astype(str))
                        print(f"✓ {kolon:<30} → Kategorik + Encoded")
                    else:
                        print(f"✓ {kolon:<30} → Kategorik")
                except:
                    print(f"✗ {kolon:<30} → Atlandı")

            else:
                if encode:
                    try:
                        df_cat[kolon] = le.fit_transform(df_cat[kolon].astype(str))
                        print(f"✓ {kolon:<30} → Encoded")
                    except:
                        print(f"✗ {kolon:<30} → Encode edilemedi")
                else:
                    df_cat[kolon] = df_cat[kolon].astype('category')
                    print(f"✓ {kolon:<30} → Category")

        print(f"\nTamamlandı!\n")
        return df_cat

    # @title
    def train_test_method(df, target_column, test_size=0.2, random_state=42):
        """
    # Kullanım
    X_train, X_test, y_train, y_test = train_test_method(df, 'price')
    X_train, X_test, y_train, y_test = train_test_method(df, 'target', test_size=0.3)
    X_train, X_test, y_train, y_test = train_test_method(df, 'sales', test_size=0.25, random_state=123)
        """
        from sklearn.model_selection import train_test_split


        if target_column not in df.columns:
            print(f"Hata: '{target_column}' kolonu bulunamadı!")
            return None, None, None, None

        print(f"\n{'='*80}")
        print(f"TRAIN-TEST AYIRMA İŞLEMİ")
        print(f"{'='*80}")
        print(f"\nHedef Kolon: {target_column}")
        print(f"Test Oranı: {test_size} (%{test_size*100})")
        print(f"Random State: {random_state}")


        X = df.drop(columns=[target_column])
        y = df[target_column]

        print(f"\n📊 Veri Bilgileri:")
        print(f"   • Toplam Satır: {len(df)}")
        print(f"   • Özellik Sayısı: {X.shape[1]}")
        print(f"   • Hedef Kolon Tipi: {y.dtype}")
        print(f"   • Hedef Kolondaki Benzersiz Değer: {y.nunique()}")


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"\n✅ Bölme Tamamlandı:")
        print(f"   • X_train shape: {X_train.shape}")
        print(f"   • X_test shape:  {X_test.shape}")
        print(f"   • y_train shape: {y_train.shape}")
        print(f"   • y_test shape:  {y_test.shape}")

        print(f"\n📈 Dağılım:")
        print(f"   • Train: {len(X_train)} satır (%{len(X_train)/len(df)*100:.1f})")
        print(f"   • Test:  {len(X_test)} satır (%{len(X_test)/len(df)*100:.1f})")

        print(f"{'='*80}\n")

        return X_train, X_test, y_train, y_test


    ######## csv açma ve kolon vs ayarlama önemli kısım
    df = pd.read_csv("C:\\Users\\emrea\\Desktop\\denemeveri\\train.csv")
    df = df.drop(columns=[df.columns[0],'Sensor8']) # 7 ve 8 aynı 8i düşürdüm geri getiririz duruma göre...
    df.head()
    df = df.drop(columns=['Date','Sensor1','Sensor6','Sensor5'])
    df['Device_ID'] = LabelEncoder().fit_transform(df['Device_ID']) # labelencoder kullandık.
    #gelismis_korelasyon_haritasi(df)
    #######




    tipler = veri_tipleri_analiz(df)
    print(tipler.to_string())


    def dengesiz_veriyi_dengele(df, target_col='Fail'):

        print("\n" + "=" * 80)
        print("--- Veri Dengeleme ---")
        print("=" * 80)

        # Fail=1 ve Fail=0 olanları ayır
        df_anomali = df[df[target_col] == 1].copy()
        df_normal = df[df[target_col] == 0].copy()

        anomali_sayisi = len(df_anomali)
        normal_sayisi_onceki = len(df_normal)

        print(f"\n✓ Orijinal veri dağılımı:")
        print(f"  - {target_col}=0 (Normal): {normal_sayisi_onceki:,}")
        print(f"  - {target_col}=1 (Anomali): {anomali_sayisi:,}")
        if anomali_sayisi > 0:
            print(f"  - Dengesizlik oranı: {normal_sayisi_onceki/anomali_sayisi:.1f}:1")

        # Fail=0 olanlarda duplicate temizliği (Fail kolonu hariç diğer kolonlara göre)
        feature_cols = [col for col in df_normal.columns if col != target_col]
        df_normal_cleaned = df_normal.drop_duplicates(subset=feature_cols, keep='first')

        normal_sayisi_sonraki = len(df_normal_cleaned)
        silinen_duplicate = normal_sayisi_onceki - normal_sayisi_sonraki

        print(f"\n✓ Duplicate temizliği ({target_col}=0 için):")
        print(f"  - Önceki Normal sayısı: {normal_sayisi_onceki:,}")
        print(f"  - Silinen duplicate: {silinen_duplicate:,}")
        print(f"  - Kalan Normal sayısı: {normal_sayisi_sonraki:,}")

        # Birleştir
        df_balanced = pd.concat([df_anomali, df_normal_cleaned], ignore_index=True)

        # Karıştır
        df_balanced = df_balanced.sample(frac=1, random_state=randomstateee).reset_index(drop=True)

        print(f"\n✓ Dengelenmiş veri dağılımı:")
        print(f"  - {target_col}=0 (Normal): {len(df_balanced[df_balanced[target_col]==0]):,}")
        print(f"  - {target_col}=1 (Anomali): {len(df_balanced[df_balanced[target_col]==1]):,}")
        if normal_sayisi_sonraki > 0:
            print(f"  - Yeni oran: {normal_sayisi_sonraki/anomali_sayisi:.2f}:1")
        print(f"  - Toplam satır: {len(df_balanced):,}")
        print("=" * 80)

        return df_balanced


    # Kullanım
    df = dengesiz_veriyi_dengele(df, target_col='Fail')

    def dengesiz_veriyi_dengele2(df, target_col='Fail', ratio=2.0):
        """
        Dengesiz veri setini dengeler.
        Fail=1 olanları tamamen korur, Fail=0 olanlardan ratio kadar seçer.

        Parameters:
        -----------
        df : pandas.DataFrame
            Veri seti
        target_col : str, default='Fail'
            Hedef değişken kolonu
        ratio : float, default=2.0
            Normal/Anomali oranı (2.0 = 2 kat normal veri)

        Returns:
        --------
        df_balanced : pandas.DataFrame
            Dengelenmiş veri seti
        """
        print("\n--- Veri Dengeleme ---")

        # Fail=1 ve Fail=0 olanları ayır
        df_anomali = df[df[target_col] == 1].copy()
        df_normal = df[df[target_col] == 0].copy()

        anomali_sayisi = len(df_anomali)
        normal_sayisi = len(df_normal)

        print(f"✓ Orijinal veri dağılımı:")
        print(f"  - Fail=0 (Normal): {normal_sayisi:,}")
        print(f"  - Fail=1 (Anomali): {anomali_sayisi:,}")
        print(f"  - Dengesizlik oranı: {normal_sayisi/anomali_sayisi:.1f}:1")

        # Normal verilerden ratio kadar seç
        hedef_normal_sayisi = int(anomali_sayisi * ratio)

        if hedef_normal_sayisi > normal_sayisi:
            print(f"⚠️  İstenen normal sayısı ({hedef_normal_sayisi:,}) mevcut sayıdan fazla!")
            print(f"   Tüm normal veriler kullanılacak.")
            hedef_normal_sayisi = normal_sayisi

        df_normal_sampled = df_normal.sample(n=hedef_normal_sayisi, random_state=22)

        # Birleştir
        df_balanced = pd.concat([df_anomali, df_normal_sampled], ignore_index=True)

        # Karıştır
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n✓ Dengelenmiş veri dağılımı:")
        print(f"  - Fail=0 (Normal): {len(df_balanced[df_balanced[target_col]==0]):,}")
        print(f"  - Fail=1 (Anomali): {len(df_balanced[df_balanced[target_col]==1]):,}")
        print(f"  - Oran: {ratio}:1")
        print(f"  - Toplam satır: {len(df_balanced):,}")
        print("=" * 80)

        return df_balanced


    # Kullanım:
    # df_balanced = dengesiz_veriyi_dengele(df, ratio=1.0)  # 1:1 (eşit)
    # df_balanced = dengesiz_veriyi_dengele(df, ratio=2.0)  # 2:1 (2 kat normal)
    df = dengesiz_veriyi_dengele2(df, ratio=ratiooo)  # 3:1 (3 kat normal)

    X_train, X_test, y_train, y_test = train_test_method(df, 'Fail', test_size=testsizeee)


 

    tipler = veri_tipleri_analiz(df)
    print(tipler.to_string())
    
    
    def model_performans_raporu(model, X_test, y_test, model_adi="Model", kaydet=False, dosya_adi=None, append=False, baslik_yaz=True):
     
        # Tahminler
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Rapor oluştur
        rapor_satirlari = []
        
        if baslik_yaz:
            rapor_satirlari.append("=" * 120)
            rapor_satirlari.append("")
            rapor_satirlari.append(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'TN':>7} {'FP':>7} {'FN':>7} {'TP':>7}")
            rapor_satirlari.append("-" * 120)

        # Model satırı
        rapor_satirlari.append(f"{model_adi:<25} {accuracy:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {roc_auc:>10.4f} {tn:>7} {fp:>7} {fn:>7} {tp:>7}")

        rapor_metni = "\n".join(rapor_satirlari)
        print(rapor_metni)

        # Kaydet
        if kaydet:
            if dosya_adi is None:
                dosya_adi = f"model_karsilastirma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(dosya_adi, 'a' if append else 'w', encoding='utf-8') as f:
                f.write(rapor_metni + "\n")

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1, 'roc_auc': roc_auc, 'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }


    def train_and_evaluate(X_train, X_test, y_train, y_test):
        """
        Random Forest modeli eğitir ve performansını değerlendirir.

        Parameters:
        -----------
        X_train : array-like
            Eğitim verileri (features)
        X_test : array-like
            Test verileri (features)
        y_train : array-like
            Eğitim verileri (hedef değişken)
        y_test : array-like
            Test verileri (hedef değişken)

        Returns:
        --------
        rf_model : RandomForestClassifier
            Eğitilmiş Random Forest modeli
        """
        print("\n--- Model Eğitimi (Random Forest) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        # Model eğitimi
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        # Tahmin
        y_pred = rf_model.predict(X_test)

        # Performans Metrikleri
        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

     

        return rf_model



    # 1. RANDOM FOREST
    def train_random_forest(X_train, X_test, y_train, y_test):
        """Random Forest modeli eğitir ve performansını değerlendirir"""
        from sklearn.ensemble import RandomForestClassifier

        print("\n--- Model Eğitimi (Random Forest) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    

        return model


    # 2. XGBOOST
    def train_xgboost(X_train, X_test, y_train, y_test):
        """XGBoost modeli eğitir ve performansını değerlendirir"""
        from xgboost import XGBClassifier

        print("\n--- Model Eğitimi (XGBoost) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    

        return model


    # 3. GRADIENT BOOSTING
    def train_gradient_boosting(X_train, X_test, y_train, y_test):
        """Gradient Boosting modeli eğitir ve performansını değerlendirir"""
        from sklearn.ensemble import GradientBoostingClassifier

        print("\n--- Model Eğitimi (Gradient Boosting) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        return model


    # 4. LOGISTIC REGRESSION
    def train_logistic_regression(X_train, X_test, y_train, y_test):
        """Logistic Regression modeli eğitir ve performansını değerlendirir"""
        from sklearn.linear_model import LogisticRegression

        print("\n--- Model Eğitimi (Logistic Regression) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        return model


    # 5. SVM
    def train_svm(X_train, X_test, y_train, y_test):
        """SVM modeli eğitir ve performansını değerlendirir"""
        from sklearn.svm import SVC

        print("\n--- Model Eğitimi (SVM) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            probability=True
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        

        return model


    # 6. KNN
    def train_knn(X_train, X_test, y_train, y_test):
        """KNN modeli eğitir ve performansını değerlendirir"""
        from sklearn.neighbors import KNeighborsClassifier

        print("\n--- Model Eğitimi (K-Nearest Neighbors) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    

        return model


    # 7. DECISION TREE
    def train_decision_tree(X_train, X_test, y_train, y_test):
        """Decision Tree modeli eğitir ve performansını değerlendirir"""
        from sklearn.tree import DecisionTreeClassifier

        print("\n--- Model Eğitimi (Decision Tree) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = DecisionTreeClassifier(
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    

        return model


    # 8. ADABOOST
    def train_adaboost(X_train, X_test, y_train, y_test):
        """AdaBoost modeli eğitir ve performansını değerlendirir"""
        from sklearn.ensemble import AdaBoostClassifier

        print("\n--- Model Eğitimi (AdaBoost) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    

        return model


    # 9. NAIVE BAYES
    def train_naive_bayes(X_train, X_test, y_train, y_test):
        """Naive Bayes modeli eğitir ve performansını değerlendirir"""
        from sklearn.naive_bayes import GaussianNB

        print("\n--- Model Eğitimi (Naive Bayes) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = GaussianNB()
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    

        return model


    # 10. LIGHTGBM
    def train_lightgbm(X_train, X_test, y_train, y_test):
        """LightGBM modeli eğitir ve performansını değerlendirir"""
        from lightgbm import LGBMClassifier

        print("\n--- Model Eğitimi (LightGBM) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    

        return model


    # 11. CATBOOST
    def train_catboost(X_train, X_test, y_train, y_test):
        """CatBoost modeli eğitir ve performansını değerlendirir"""
        from catboost import CatBoostClassifier

        print("\n--- Model Eğitimi (CatBoost) ---")
        print(f"✓ Train: {len(X_train):,} satır")
        print(f"✓ Test: {len(X_test):,} satır")
        print(f"  - Train dağılımı: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
        print(f"  - Test dağılımı: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}")

        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)

        print("✓ Model eğitimi tamamlandı")

        y_pred = model.predict(X_test)

        print("\n--- Model Performansı ---")
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        return model



    # modelleri üretmeeeee
    model1 = train_random_forest(X_train, X_test, y_train, y_test)
    model2 = train_xgboost(X_train, X_test, y_train, y_test)
    model3 = train_gradient_boosting(X_train, X_test, y_train, y_test)
    model4 = train_logistic_regression(X_train, X_test, y_train, y_test)
    model5 = train_svm(X_train, X_test, y_train, y_test)
    model6 = train_knn(X_train, X_test, y_train, y_test)
    model7 = train_decision_tree(X_train, X_test, y_train, y_test)
    model8 = train_adaboost(X_train, X_test, y_train, y_test)
    model9 = train_naive_bayes(X_train, X_test, y_train, y_test)
    model10 = train_lightgbm(X_train, X_test, y_train, y_test)
    model11 = train_catboost(X_train, X_test, y_train, y_test)


    dosya = f"{randomstateee}--{ratiooo}--{testsizeee}.txt"

    model_performans_raporu(model1, X_test, y_test, "Random Forest", kaydet=True, dosya_adi=dosya, append=False)  # İlk model - yeni dosya
    model_performans_raporu(model2, X_test, y_test, "XGBoost", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model3, X_test, y_test, "Gradient Boosting", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model4, X_test, y_test, "Logistic Regression", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model5, X_test, y_test, "SVM", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model6, X_test, y_test, "K-Nearest Neighbors", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model7, X_test, y_test, "Decision Tree", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model8, X_test, y_test, "AdaBoost", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model9, X_test, y_test, "Naive Bayes", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model10, X_test, y_test, "LightGBM", kaydet=True, dosya_adi=dosya, append=True)
    model_performans_raporu(model11, X_test, y_test, "CatBoost", kaydet=True, dosya_adi=dosya, append=True)

import numpy as np

for i in range(1, 101):                       
    for j in np.arange(1.0, 5.2, 0.1):       
        for k in np.arange(0.18, 0.31, 0.01):  
            denevekaydet(randomstateee=i, ratiooo=j, testsizeee=k)


