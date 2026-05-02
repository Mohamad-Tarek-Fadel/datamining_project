# الدليل الشامل لمشروع التخرج: Early Disease Prediction Using Healthcare Data Warehouse
**(Defense & Viva Comprehensive Guide)**

تم إعداد هذا الملف خصيصاً ليكون مرجعك الشامل في يوم المناقشة (Viva)، حيث يوثق كل مراحل تطور المشروع، وكيف تحول من مجرد مشروع Data Mining تقليدي إلى **End-to-End Enterprise Data Pipeline** يحاكي الأنظمة الطبية الحقيقية.

---

## 1. التطور الجوهري في المشروع (Project Evolution)
كان المشروع في بدايته يعتمد على ملفات CSV جاهزة ومهيكلة (Structured Data)، وهو ما يُعتبر أسلوباً أكاديمياً تقليدياً. لرفع مستوى المشروع وجعله يلامس الواقع العملي (Real-World Complexity)، قمنا بالانتقال إلى **البيانات غير المهيكلة (Unstructured Data)**.

### كيف أصبحت الداتا Unstructured بجد؟
بدلاً من قوالب نصية ثابتة ومملة (Static Templates)، قمنا ببرمجة محرك توليد (Stochastic Generator - `generate_unstructured.py`) يكتب التقارير وكأن طبيباً حقيقياً أو مريضاً يكتبها. شمل ذلك:
1. **Clinical Notes (تقارير الأطباء):** نصوص طبية تحتوي على اختصارات معقدة (مثل `Pt`, `hx`, `y/o`)، وأخطاء إملائية مقصودة (Typos)، وتنسيقات سردية (Narrative) وSOAP Notes.
2. **Social Media Posts (بوستات الأمهات/الآباء):** لبيانات الـ Toddler Autism، استخدمنا لغة غير رسمية تماماً (Informal Language) تحاكي بوستات الآباء على منصات الدعم النفسي (مثل: "My son is 24 months old, he never looks at me").
3. **IoT Sensor Logs (سجلات أجهزة الاستشعار):** لمرضى السكر، قمنا بدمج نصوص طبية مع رسائل JSON و Syslog ناتجة من أجهزة قياس السكر المستمرة (CGM - Continuous Glucose Monitors).

---

## 2. محرك استخراج البيانات (The Unstructured Parser Layer)
لتحويل هذه الفوضى النصية إلى بيانات يمكن للموديل فهمها، قمنا ببناء **Multi-pattern Regex Parser** في ملف (`parse_unstructured.py`).
- **ذكاء الـ Parser:** لا يعتمد على أماكن الكلمات، بل يفهم السياق. (مثلاً: يقدر يفرق بين `Family history: None` وبين `Patient has a strong family history`).
- **دمج البيانات (Data Integration):** قمنا بخطوة هندسية قوية جداً وهي دمج داتا الأطفال (Toddler Autism) مع داتا البالغين (Adult Autism).
  - تم تحويل أعمار الأطفال من **شهور إلى سنوات**.
  - تم تحويل الإجابات النصية (Yes/No و M/F) إلى أرقام (1/0).
  - تم توحيد الـ Q-Chat-10 مع الـ AQ-10.
- **النتيجة:** ارتفع عدد بيانات مرضى التوحد من 6,075 إلى **7,129 سجل**، وتم استخراج **12,759 سجل إجمالي** لجميع الأمراض بـ **Zero Missing Values**!

---

## 3. هندسة مستودع البيانات (Medallion Architecture & Data Warehouse)
تم تصميم المشروع وفقاً لمعمارية **Medallion Architecture** الاحترافية:
1. **Bronze Layer (الخام):** ملفات الـ Text غير المهيكلة (موجودة في فولدر `datasets/unstructured/`).
2. **Silver Layer (المنظفة):** ملفات الـ CSV المستخرجة والنظيفة (موجودة في فولدر `datasets/cleaned/`).
3. **Gold Layer (مستودع البيانات):** قاعدة بيانات SQLite (`health_warehouse.db`).

### تصميم الـ Star Schema:
تم بناء مستودع البيانات (`warehouse.py` و `schema.sql`) باستخدام **Star Schema**:
- **جدول الأبعاد (Dimension Table):** جدول `dim_patient` يحتوي على البيانات الديموغرافية الأساسية لكل المرضى (12,759 مريض).
- **جداول الحقائق (Fact Tables):** 3 جداول تفصيلية تعتمد على مفتاح `patient_id` كـ Foreign Key:
  - `fact_autism` (7,129 سجل).
  - `fact_diabetes` (520 سجل).
  - `fact_stroke` (5,110 سجل).
- تم إنشاء **Views** متقدمة داخل قاعدة البيانات (`vw_autism_full` وغيرها) لتسهيل استخراج تقارير الـ BI الـ Dashboard.

---

## 4. مرحلة تعلم الآلة (Machine Learning Pipeline)
تم تقسيم الـ ML إلى نوت بوكس احترافية لسهولة العرض في المناقشة:
- **`02_eda.ipynb` (الاستكشاف):** تحليل البيانات وإيجاد العلاقات (Correlations).
- **`03_feature_engineering.ipynb` (الهندسة):** 
  - معالجة الـ Imbalanced Data (استخدام SMOTE لبيانات الجلطات لأن نسبتها 1:19، واستخدام Class Weights للسكري والتوحد).
  - عمل Feature Scaling (StandardScaler) وتقسيم البيانات 80/20 بنظام Stratified.
- **`04_modeling.ipynb` (التدريب):**
  - تدريب أكثر من موديل (Random Forest, XGBoost, SVM, Logistic Regression).
  - تقييم الموديلات بدقة باستخدام تقارير الأداء ومصفوفة الخطأ (Confusion Matrix)، وحفظ أفضل الموديلات (`.pkl`) في فولدر `models/saved`.

---

## 5. الوضع النهائي للمشروع اليوم (Final Status)
المشروع الآن مقفول برمجياً بنسبة 100% واجتاز كل اختبارات فحص الجودة المكتوبة في سكربت `final_check.py`:
- **عدد الملفات البرمجية:** 5 سكربتات بايثون متكاملة لعملية الـ ETL.
- **قواعد البيانات:** ملف SQLite مبني بـ Foreign Keys صارمة ولا يوجد فيه أي بيانات يتيمة (0 Orphans).
- **الرسومات البيانية:** 39 رسم بياني جاهزين للـ Presentation (موجودين في `reports/figures`).
- **نسبة النجاح التقني:** اجتاز الكود 50 من أصل 50 اختبار تقني (50/50 Checks Passed).

---

## 6. أسئلة متوقعة في المناقشة (Viva Q&A) وكيفية الرد عليها

**سؤال 1: ليه معملتوش المشروع على داتا عادية (CSV) من الأول؟**
**الرد:** "عشان الواقع الطبي مش جداول CSV جاهزة، الأطباء بيكتبوا Clinical Notes حرة. هدفنا كان إثبات قدرتنا على بناء End-to-End Pipeline يقدر ياخد داتا عشوائية، مليانة Typos واختصارات، أو حتى سجلات أجهزة IoT، وينظفها ويستخرج منها Data مهيكلة قابلة للتحليل باستخدام Natural Language Processing (Regex) قبل ما تدخل الموديل."

**سؤال 2: إزاي اتعاملتوا مع داتا التوحد للأطفال (Toddlers) اللي كانت بتسأل أسئلة Q-Chat بدل AQ-10؟**
**الرد:** "برمجنا Standardization Layer في الـ Parser، حولنا فيها أعمار الأطفال من الشهور للسنوات (مقسومة على 12)، ووحدنا الإجابات النصية إلى (0 و 1)، ودمجنا الأسئلة مع الـ AQ-10 في جدول واحد (`fact_autism`). ده رفع عدد الداتا لـ 7,129 سجل وخلى الموديل قادر يتوقع التوحد لكل الفئات العمرية."

**سؤال 3: ليه استخدمتوا SMOTE لمرضى الجلطات ومستخدمتهوش للتوحد والسكري؟**
**الرد:** "في داتا الجلطات، نسبة المصابين للغير مصابين كانت 1 إلى 19 (Severe Imbalance)، فكان لازم نولد بيانات تخيلية (Synthetic Data) باستخدام SMOTE عشان الموديل ميبقاش Biased للمرضى الأصحاء. أما في التوحد والسكري، الـ Imbalance كان خفيف (مثلاً 1 إلى 2.4)، فاعتمدنا فقط على تعديل الأوزان (Class Weights) عشان نركز على الداتا الأصلية بدون تعديل."

**سؤال 4: ليه استخدمتوا Star Schema؟**
**الرد:** "عشان نطبق أصول الـ Data Warehousing. الـ Star Schema (عن طريق فصل الـ Patient Demographics في جدول Dimension ووضع القياسات الطبية في Fact tables) بتوفر سرعة هائلة في تنفيذ استعلامات الـ Analytics والـ Dashboards، وبتمنع تكرار البيانات (Data Redundancy)."
