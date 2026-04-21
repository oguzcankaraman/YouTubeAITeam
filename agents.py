from langchain.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from crawl4ai import AsyncWebCrawler
from state import AgentState

from rich.console import Console

console = Console()

class ResearcherAgent:
    async def run(self, state: AgentState):
        console.print(f"\n[bold cyan]>[/bold cyan] [bold cyan]Araştırmacı:[/bold cyan] {state['target_url']} kazılıyor...")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=state["target_url"])
            markdown_content = result.markdown

            return {"scraped_data": markdown_content, "iteration": 0, "feedback": ""}

class CoderAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="gemma4:e4b",
            temperature=0.1
        )

    def run(self, state: AgentState):
        iteration = state.get("iteration", 0) + 1
        console.print(f"\n[bold magenta]>[/bold magenta] [bold magenta]Geliştirici (Deneme {iteration}):[/bold magenta] Kod inşası başlatılıyor...\n")

        feedback_context = f"\nÖnceki denemedeki hatalar: {state["feedback"]}\nBu hataları düzelterek kodu yeniden yaz." if state.get("feedback") else ""

        system_prompt = f""" Sen Senior bir Python geliştiricisisin.
        Dökümantasyon: {state["scraped_data"][:10000]}
        kullanıcı isteği: {state["user_request"]}\n
        {feedback_context}
        sadece çalışan bir python kodu yaz, başka bir şey yazma.   
        """


        response_content = ""
        for chunk in self.llm.stream([SystemMessage(content=system_prompt)]):
            print(f"\033[92m{chunk.content}\033[0m", end="", flush=True)
            response_content += chunk.content

        return {"final_code": response_content, "iteration": iteration}


class ReviewerAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="gemma4:e4b",
            temperature=0
        )

    def run(self, state: AgentState):
        console.print("\n[bold yellow]>[/bold yellow] [bold yellow]Kontrolcü:[/bold yellow] Yazılan kod denetleniyor...")

        prompt = f"""Sen bir Kod Reviewer (Denetmen) ajansın. 
                Kullanıcının isteği: {state['user_request']}
                Yazılan Kod: {state['final_code']}

                Kod eksiksiz, isteği karşılıyor ve sentaks hatası içermiyorsa sadece 'ONAYLANDI' yaz.
                Eğer kodda hata, eksik veya halüsinasyon varsa, nedenini kısaca açıklayarak 'REDDEDİLDİ: [Nedenler]' şeklinde yaz.
        """

        response = self.llm.invoke([HumanMessage(content=prompt)]).content

        if "ONAYLANDI" in response.upper():
            return {"is_approved": True, "feedback": ""}
        else:
            return {"is_approved": False, "feedback": response}