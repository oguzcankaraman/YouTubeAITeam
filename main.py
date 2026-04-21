import asyncio
from langgraph.graph import StateGraph, END
from state import AgentState
from agents import ResearcherAgent, CoderAgent, ReviewerAgent

from rich.console import Console

console = Console()


def route_checker(state: AgentState):
    if state.get("is_approved"):
        return "end"
    elif state.get("iteration", 0) >= 3:
        console.print("\n[bold red]Maksimum deneme sayısına ulaşıldı. Sistem durduruluyor.[/bold red]")
        return "end"
    return "coder"


async def main():
    console.clear()
    console.print("[bold yellow]🚀 Otonom Sistem Başlatılıyor...[/bold yellow]\n")
    researcher = ResearcherAgent()
    coder = CoderAgent()
    reviewer = ReviewerAgent()

    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", researcher.run)
    workflow.add_node("coder", coder.run)
    workflow.add_node("reviewer", reviewer.run)

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "reviewer")

    workflow.add_conditional_edges("reviewer", route_checker, {"end": END, "coder": "coder"})

    app = workflow.compile()

    final_state = await app.ainvoke({
        "target_url": "https://ai.pydantic.dev/agents/",
        "user_request": "Pydantic AI ile basit bir RAG ajanı kodla", }
    )

    console.print("\n[bold cyan]=== 📊 OTONOM SİSTEM FİNAL DURUM RAPORU (STATE) ===[/bold cyan]")

    console.print(f"[bold yellow]Hedef URL:[/bold yellow] {final_state['target_url']}")

    scraped_snippet = final_state['scraped_data'][:300].replace('\n', ' ')
    console.print(f"[bold yellow]Kazınan Veri (İlk 300 Karakter):[/bold yellow]\n[dim]{scraped_snippet}...[/dim]\n")

    console.print(f"[bold yellow]Toplam Deneme (Iteration):[/bold yellow] {final_state.get('iteration', 1)}")
    console.print(
        f"[bold yellow]Onay Durumu:[/bold yellow] {'[green]Geçti[/green]' if final_state.get('is_approved') else '[red]Kaldı[/red]'}")
    console.print(f"[bold yellow]Üretilen Kod Uzunluğu:[/bold yellow] {len(final_state['final_code'])} karakter")

    console.print("[bold cyan]=======================================================[/bold cyan]\n")

if __name__ == "__main__":
    asyncio.run(main())

