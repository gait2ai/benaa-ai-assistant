# generate_bcfhd_embeddings.py
import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime

# النص الكامل لملف BCFHD (يجب نسخه هنا)
bcfhd_text = """
Bena Charity for Human Development (BCFHD) is a non-profit, non-governmental organization founded in 1996 and headquartered in Taiz, Yemen. BCFHD works across Yemen in multiple humanitarian and development sectors including WASH, Food Security, Education, Health, Shelter, Protection, and Empowerment. 



BCFHD implements projects in partnership with UN agencies, international organizations, and local partners. It is an active cluster partner for CCCM, WASH, Education, Protection, Health, Food Security, and Shelter. BCFHD also coordinates the CCCM, Cholera, and Education clusters in certain districts of Taiz.



The organization is led by President Abdulkareem Abdullah Ahmed Shamsan and Program Manager Riyadh Mohammed Shamsan. BCFHD operates with values of transparency, independence, neutrality, and commitment to providing quality services to vulnerable communities. 



BCFHD's interventions span emergency relief as well as early recovery and development programs. Its major projects in recent years include:



- Emergency WASH services for floods-affected IDPs in Taiz and Marib. Funded by IRC and reaching over 4,000 displacement-affected individuals. 



- Emergency Education assistance to over 11,000 vulnerable children through improving learning environments, providing school supplies and facilitating access to education in conflict-impacted districts of Taiz. Supported by UN OCHA's YHF.



- Food security and livelihoods aid to over 25,000 households across Taiz, Aden, Marib and Hadhramaut through emergency food supplies and agricultural livelihoods kits. Funded by FAO, KS Relief and YHF.



- Shelter support to 3,100 displaced households in Taiz through emergency kits, rental subsidies and sustainable shelter solutions. Funded by YHF and UNHCR. 



- Healthcare assistance to over 100,000 vulnerable individuals in Taiz to respond to disease outbreaks like Cholera, COVID-19 and Dengue. Funded by IRC and ECHO.



- Women empowerment initiatives to enable livelihoods through provision of vocational training in handicrafts. Funded by King Salman Humanitarian Center.



- Rehabilitation of conflict-damaged houses and WASH infrastructure in villages of Taiz. Supported by CARE International and UN Habitat.



With over 25 years of experience, a strong community network, and expertise across key humanitarian sectors, BCFHD is well-positioned to continue providing impactful aid to vulnerable populations in Yemen.



Association Memberships:

1. Member of the Arab Network for Non-Governmental Organizations (ANNGOs).

2. Member of the Local Network for Non-Governmental Organizations (NGOs).

3. Member of the Food Security Cluster

4. Member of the Water, Sanitation, and Hygiene Cluster

5. Member of the Shelter Cluster

6. Member of the Camp Coordination and Camp Management Cluster

7. Member of the Protection Cluster

8. Member of the Education Cluster

9. Member of the Inter-Agency Network for Education in Emergencies (INEE)

10. Member of the Dyslexia Organization

11. Member of the G8 NGOs

12. Member of the Islamic Union for Non-Governmental Organizations

13. Member of the Humanitarian Relief Coalition

14.  Member of the International Transparency Network

15. Member of the National Assembly (SAG) for the Wasit Region



Main sectors in which the association works:

•	Education

•	Protection (General, Child)

•	Livelihoods and Food Security

•	Al-Wash

•	Shelter

•	Camp Management

•	Health





BCFHD Cluster Membership

The association is a member of the

Education Cluster and serves as the coordinator for the Southern Taiz and West Coast Axis.

Al-Wash Cluster is a member of the SAG.

•	Protection Cluster

•	Shelter

•	Camp Management (acts as the coordinator for the three city directorates)

•	Food Security Cluster

•	Observer member of the Health Cluster


"""

# إعداد المسارات
os.makedirs("embeddings", exist_ok=True)

# 1. تقسيم النص إلى أجزاء ذكية
def smart_text_splitter(text, max_chunk_size=300):
    # تقسيم حسب العناوين الرئيسية
    sections = re.split(r'\n(?=\d+\. |·|Main sectors|Profile of Projects)', text)
    chunks = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # تقسيم الفقرات الطويلة
        paragraphs = re.split(r'\n\s*\n', section.strip())
        current_chunk = []
        word_count = 0
        
        for para in paragraphs:
            words = para.split()
            if word_count + len(words) > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                word_count = 0
            current_chunk.append(para)
            word_count += len(words)
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    
    return chunks

# 2. معالجة التضمينات
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
chunks = smart_text_splitter(bcfhd_text)
embeddings = model.encode(chunks, convert_to_numpy=True)

# 3. حفظ الفهرس
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
faiss.write_index(index, "embeddings/document_index.faiss")

# 4. إنشاء البيانات الوصفية
metadata = []
for idx, chunk in enumerate(chunks):
    # استخراج العناوين التلقائية
    header = re.search(r'^\d+\.\s+(.+)|^·\s+(.+)', chunk)
    metadata.append({
        "text": chunk,
        "source_file": "bcfhd_profile.md",
        "chunk_id": idx,
        "section": header.group(1) if header else "General Information",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

with open("embeddings/document_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# 5. ملف معلومات النموذج
model_info = {
    "model": "paraphrase-multilingual-MiniLM-L12-v2",
    "dimension": 384,
    "language_support": ["ar", "en"],
    "created_at": datetime.now().isoformat()
}

with open("embeddings/model_info.json", "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print("✅ تم إنشاء فهرس BCFHD بنجاح!")
