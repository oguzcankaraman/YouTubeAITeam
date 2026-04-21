import os
from pydantic_ai import Agent, BaseModel, Field
from typing import List

# --- 1. API Key Setup ---
# Lütfen kendi API anahtarınızı buraya girin veya ortam değişkeni olarak ayarlayın.
# Bu örnekte OpenAI kullanılmıştır.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Eğer ortam değişkeni ayarlanmamışsa, kodu çalıştırmadan önce ayarladığınızdan emin olun.
if "OPENAI_API_KEY" not in os.environ:
    print("HATA: Lütfen OPENAI_API_KEY ortam değişkenini ayarlayın.")
    # Çalışmayı durdurmak için basit bir çıkış yapıyoruz.
    # Gerçek bir ortamda, bu kısmı atlayıp sadece kodu çalıştırmalısınız.
    # exit()
    pass

# --- 2. Bilgi Tabanı (Knowledge Base) ---
# RAG için kullanılacak kaynak belgeler.
KNOWLEDGE_BASE = [
    "Pydantic, Python geliştiricileri için veri doğrulama ve modelleme aracıdır. Özellikle AI uygulamalarında veri yapısını tanımlamak için kullanılır.",
    "Pydantic AI, Pydantic'in yapay zeka yeteneklerini genişleten bir kütüphanedir. Agent, Tool ve Embedding gibi yeni özellikler ekler.",
    "RAG (Retrieval-Augmented Generation), bir LLM'e harici, güncel veya özel bir bilgi kaynağı (context) sağlayarak daha doğru ve halüsinasyon yapmayan cevaplar üretmesini sağlayan bir mimaridir.",
    "Bir RAG ajanı genellikle üç aşamadan oluşur: Retrieval (Bilgi Çekme), Augmentation (Zenginleştirme) ve Generation (Üretme)."
]


# --- 3. Retrieval (Bilgi Çekme) Fonksiyonu ---
def retrieve_context(query: str, knowledge_base: List[str], top_k: int = 2) -> str:
    """
    Basit bir anahtar kelime eşleşmesi ile ilgili bağlamı çeker.
    Gerçek bir uygulamada, bu kısım bir vektör veritabanı (Pinecone, Chroma vb.) kullanmalıdır.
    """
    print(f"\n[DEBUG] Sorgu için bağlam aranıyor: '{query}'")

    # Basit bir skorlama mekanizması (sadece örnek amaçlı)
    scores = {}
    for i, chunk in enumerate(knowledge_base):
        score = 0
        # Sorgu kelimelerinin chunk içinde kaç kez geçtiğini sayar
        for word in query.lower().split():
            if word in chunk.lower():
                score += 1
        scores[i] = score

    # En yüksek skora sahip olanları sırala
    sorted_indices = sorted(scores, key=scores.get, reverse=True)

    # En iyi K sonucu al
    top_indices = sorted_indices[:top_k]

    context_chunks = [knowledge_base[i] for i in top_indices]

    return "\n---\n".join(context_chunks)


# --- 4. Agent Tanımlama ve Çalıştırma ---

# Agent'ın görevini ve beklentisini tanımlayan Pydantic modeli
class RAGResponse(BaseModel):
    """
    RAG ajanı tarafından üretilecek nihai cevabı tanımlar.
    """
    answer: str = Field(
        description="Verilen bağlamı kullanarak, kullanıcının sorusunu net ve anlaşılır bir şekilde yanıtlayan nihai cevap.")
    source_context: str = Field(description="Cevabı oluşturmak için kullanılan kaynak bağlamın özeti.")


def run_rag_agent(user_query: str):
    """
    RAG sürecini yöneten ana fonksiyon.
    """
    print("=====================================================")
    print("🚀 RAG AJANI BAŞLATILIYOR...")
    print("=====================================================")

    # 1. Retrieval (Bilgi Çekme)
    context = retrieve_context(user_query, KNOWLEDGE_BASE)

    if not context:
        print("\n[UYARI] Yeterli bağlam bulunamadı. Lütfen farklı bir soru deneyin.")
        return

    # 2. Generation (Üretme)
    # Agent'a, sadece verilen bağlamı kullanması gerektiğini söyleyen bir sistem mesajı veriyoruz.
    system_prompt = (
        "Sen, yalnızca sana sağlanan bağlamı kullanarak cevap veren, son derece doğru ve güvenilir bir yapay zeka asistanısın. "
        "Kullanıcı sorusunu yanıtlamak için verilen 'BAĞLAM' kısmındaki bilgileri kullan. "
        "Eğer bağlamda cevap yoksa, bunu kibarca belirt ve tahmin yürütme."
    )

    full_prompt = f"""
    {system_prompt}

    --- BAĞLAM ---
    {context}
    --- SON ---

    Kullanıcı Sorusu: {user_query}
    """

    try:
        # Pydantic AI Agent'ı kullanarak LLM çağrısı yapıyoruz.
        # Bu, LLM'den yapılandırılmış bir çıktı (RAGResponse) beklediğimiz anlamına gelir.
        agent = Agent(
            model="openai",  # Kullanılan model sağlayıcısı
            model_kwargs={"model": "gpt-3.5-turbo-0125"},  # Kullanılacak model
            system_prompt=system_prompt,
            tools=[RAGResponse]  # Beklenen çıktı yapısı
        )

        # Agent'ı çağırırken, tüm talimatı tek bir prompt olarak veriyoruz.
        response: RAGResponse = agent.invoke(full_prompt)

        # 3. Sonuçları Gösterme
        print("\n=====================================================")
        print("✅ AJAN BAŞARILI BİR ŞEKİLDE CEVAP ÜRETTİ:")
        print("=====================================================")
        print(f"🤖 Cevap: {response.answer}")
        print("\n--- Kaynak Bilgiler ---")
        print(f"📚 Kullanılan Bağlam Özeti: {response.source_context}")
        print("=====================================================")

    except Exception as e:
        print(
            f"\n[HATA] Agent çağrılırken bir hata oluştu. API anahtarınızı ve model ayarlarınızı kontrol edin. Hata: {e}")


# --- 5. Örnek Kullanım ---

# Örnek 1: Başarılı RAG senaryosu (Pydantic AI ile ilgili bir soru)
user_query_1 = "Pydantic AI ne işe yarar ve hangi bileşenleri içerir?"
run_rag_agent(user_query_1)

# Örnek 2: Başarılı RAG senaryosu (RAG mimarisi ile ilgili bir soru)
user_query_2 = "RAG mimarisi nasıl çalışır?"
run_rag_agent(user_query_2)

# Örnek 3: Bağlamda cevap bulunmayan bir senaryo (Test amaçlı)
user_query_3 = "İstanbul'un en iyi restoranları hangileridir?"
run_rag_agent(user_query_3)